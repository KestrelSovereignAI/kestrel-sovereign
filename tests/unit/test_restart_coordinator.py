"""Tests for the durable restart coordinator (#1512).

Covers the store layer (table init, request lifecycle, status
transitions, race-protection on cancel) and the feature surface
(request/list/cancel @tools, the executor cron task's safety gate,
and the post-restart sweep that wakes the requesting agent).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.restart_coordinator import (
    RestartCoordinatorFeature,
)
from kestrel_sovereign.features.restart_coordinator.store import (
    ensure_restart_requests_table,
    get_request,
    insert_request,
    list_requests,
    record_update_log,
    update_status,
)
from kestrel_sovereign.features.restart_coordinator.update_profiles import (
    get_update_profile,
    is_valid_target_ref,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _CapturingDispatcher:
    def __init__(self):
        self.signals = []

    def enqueue_signal(self, signal):
        self.signals.append(signal)
        return None


class _StubRegistry:
    def __init__(self):
        self.registered = []
        self.by_name = {}

    def register(self, reg):
        if reg.name in self.by_name:
            raise RuntimeError(f"duplicate source {reg.name!r}")
        self.registered.append(reg)
        self.by_name[reg.name] = reg

    def get(self, name):
        return self.by_name.get(name)


async def _backend(tmp_path):
    """Wrap SQLiteBackend in AsyncDatabase to match the production
    surface ``resolve_feature_database`` returns. ``AsyncDatabase``
    exposes ``fetchall`` (no underscore) while the bare backend uses
    ``fetch_all`` — store code must work against the wrapper.
    """
    raw = SQLiteBackend(str(tmp_path / "restart.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    await ensure_restart_requests_table(db)
    return db


def _make_agent(backend, did="did:test:agent", dispatcher=None,
                registry=None):
    raw_storage = SimpleNamespace(db=backend)
    return SimpleNamespace(
        did=did,
        agent_id=did,
        _raw_storage=raw_storage,
        storage=None,
        dispatcher=dispatcher,
        signal_registry=registry,
        # An idle agent — empty active-request set and no background
        # tasks. Tests that need to model "busy" override these.
        _active_request_ids=set(),
        _background_tasks=set(),
        features={"RestartCoordinatorFeature": True},
    )


async def _make_feature(tmp_path, **kwargs):
    backend = await _backend(tmp_path)
    agent = _make_agent(backend, **kwargs)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    return feat, backend


# ---------------------------------------------------------------------------
# Store layer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_table_is_idempotent(tmp_path):
    backend = await _backend(tmp_path)
    # Second call must NOT error or wipe data.
    await ensure_restart_requests_table(backend)
    await insert_request(
        backend, requested_by_agent="a", reason="r",
    )
    await ensure_restart_requests_table(backend)
    rows = await list_requests(backend)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_insert_then_list_then_get(tmp_path):
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:t:a", reason="config landed",
        urgency="high", policy="idle_agents_only",
    )
    rows = await list_requests(backend)
    assert len(rows) == 1
    assert rows[0].id == req.id
    assert rows[0].status == "pending"
    assert rows[0].urgency == "high"
    # get_request returns the same row.
    fetched = await get_request(backend, req.id)
    assert fetched is not None
    assert fetched.reason == "config landed"


@pytest.mark.asyncio
async def test_update_status_gated_on_expected_current(tmp_path):
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="a", reason="r",
    )
    # Transition gated on the wrong expected status must NOT update.
    ok = await update_status(
        backend, req.id, status="executing",
        expected_current_status="executing",  # wrong — row is pending
    )
    assert ok is False
    fetched = await get_request(backend, req.id)
    assert fetched.status == "pending"
    # Correct expected status: transition lands.
    ok = await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )
    assert ok is True


@pytest.mark.asyncio
async def test_list_requests_filters_by_status_and_agent(tmp_path):
    backend = await _backend(tmp_path)
    await insert_request(backend, requested_by_agent="a", reason="r1")
    await insert_request(backend, requested_by_agent="a", reason="r2")
    await insert_request(backend, requested_by_agent="b", reason="r3")
    # Filter by agent.
    a_rows = await list_requests(backend, agent_id="a")
    assert len(a_rows) == 2
    # Filter by status (all pending).
    pending = await list_requests(backend, status="pending")
    assert len(pending) == 3


# ---------------------------------------------------------------------------
# Feature surface — @tool methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_restart_creates_pending_row(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    result = await feat.request_restart(
        reason="kestrel.toml change landed",
    )
    assert result.status is ToolResultStatus.OK
    req = result.data["request"]
    assert req["status"] == "pending"
    assert req["requested_by_agent"] == "did:test:agent"
    # Persisted to the table.
    rows = await list_requests(backend)
    assert any(r.id == req["id"] for r in rows)


@pytest.mark.asyncio
async def test_request_restart_rejects_unknown_urgency(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    result = await feat.request_restart(reason="r", urgency="bogus")
    assert result.error is not None
    assert "urgency" in result.error


@pytest.mark.asyncio
async def test_request_restart_rejects_unknown_policy(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    result = await feat.request_restart(reason="r", policy="anything")
    assert result.error is not None
    assert "policy" in result.error


@pytest.mark.asyncio
async def test_request_restart_rejects_empty_reason(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    result = await feat.request_restart(reason="   ")
    assert result.error is not None
    assert "reason" in result.error


@pytest.mark.asyncio
async def test_list_restart_requests_returns_all_then_filtered(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    await feat.request_restart(reason="r1")
    await feat.request_restart(reason="r2", urgency="high")
    # No filter.
    r = await feat.list_restart_requests()
    assert r.data["count"] == 2
    # Filter to pending — still both.
    r = await feat.list_restart_requests(status="pending")
    assert r.data["count"] == 2
    # Filter to completed — none yet.
    r = await feat.list_restart_requests(status="completed")
    assert r.data["count"] == 0


@pytest.mark.asyncio
async def test_cancel_pending_request(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="r")
    req_id = created.data["request"]["id"]
    cancel = await feat.cancel_restart_request(req_id, reason="never mind")
    assert cancel.status is ToolResultStatus.OK
    # Row reflects terminal canceled state.
    row = await get_request(backend, req_id)
    assert row.status == "canceled"
    assert "never mind" in row.status_reason


@pytest.mark.asyncio
async def test_cannot_cancel_executing_request(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="r")
    req_id = created.data["request"]["id"]
    # Move it to executing manually.
    await update_status(
        backend, req_id, status="executing",
        expected_current_status="pending",
    )
    cancel = await feat.cancel_restart_request(req_id)
    assert cancel.error is not None
    assert "executing" in cancel.error.lower()


@pytest.mark.asyncio
async def test_cancel_unknown_id_errors(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    cancel = await feat.cancel_restart_request("does-not-exist")
    assert cancel.error is not None
    assert "No restart request" in cancel.error


# ---------------------------------------------------------------------------
# Executor cron task — restart_coordinator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executor_no_pending_returns_no_op(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()
    assert result.data["executed"] == [] if isinstance(
        result.data["executed"], list
    ) else result.data["executed"] is False
    assert mock_spawn.call_count == 0


@pytest.mark.asyncio
async def test_executor_spawns_subprocess_for_idle_agent(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="ship")
    req_id = created.data["request"]["id"]

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 1
    assert result.data["executed"][0]["request_id"] == req_id
    # Row moved to executing.
    row = await get_request(backend, req_id)
    assert row.status == "executing"


@pytest.mark.asyncio
async def test_executor_defers_when_agent_reports_active_request(tmp_path):
    # Agent has at least one active request — executor must defer.
    backend = await _backend(tmp_path)
    agent = _make_agent(backend)
    agent._active_request_ids.add("req-1")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    await feat.request_restart(reason="r")

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 0
    assert len(result.data["deferred"]) == 1
    assert "busy" in result.data["deferred"][0]["reason"]


def _attach_lifecycle(agent):
    """Bind the real RequestLifecycleMixin surface onto a mock agent so
    the coordinator's stale-request sweep (#1558) can run against it.
    """
    from kestrel_sovereign.agent.request_lifecycle import (
        RequestLifecycleMixin,
    )

    agent._current_request_id = None
    agent._active_request_started_at = {}
    agent._cancelled_requests = set()
    for name in (
        "register_active_request",
        "prune_stale_active_requests",
        "active_request_ages",
        "_cleanup_cancelled_request",
    ):
        setattr(
            agent, name,
            getattr(RequestLifecycleMixin, name).__get__(agent),
        )
    return agent


@pytest.mark.asyncio
async def test_executor_sweeps_stale_active_request_and_executes(tmp_path):
    """A stale active request id (endpoint cleanup never ran) must NOT
    deadlock idle_agents_only — the coordinator sweeps it and executes
    (#1558).
    """
    backend = await _backend(tmp_path)
    agent = _make_agent(backend)
    _attach_lifecycle(agent)
    agent.register_active_request("stale-req")
    # Back-date past the staleness window so the sweep treats it as
    # abandoned rather than in-flight.
    agent._active_request_started_at["stale-req"] -= 1000

    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    created = await feat.request_restart(reason="r")
    req_id = created.data["request"]["id"]

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 1
    assert result.data["executed"][0]["request_id"] == req_id
    # The stale marker was swept out.
    assert "stale-req" not in agent._active_request_ids
    row = await get_request(backend, req_id)
    assert row.status == "executing"


@pytest.mark.asyncio
async def test_executor_still_defers_for_fresh_active_request(tmp_path):
    """A genuinely fresh active request still defers idle_agents_only,
    and the deferral reason exposes the request age (#1558).
    """
    backend = await _backend(tmp_path)
    agent = _make_agent(backend)
    _attach_lifecycle(agent)

    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    # File while idle, then an unrelated fresh request goes in flight —
    # one that is NOT the requester's own turn, so it still blocks (#1561).
    await feat.request_restart(reason="r")
    agent.register_active_request("fresh-req")

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 0
    assert len(result.data["deferred"]) == 1
    reason = result.data["deferred"][0]["reason"]
    assert "busy" in reason
    # Observability: oldest active-request age + stale window surfaced.
    assert "stale window" in reason
    # The fresh id was NOT swept.
    assert "fresh-req" in agent._active_request_ids


@pytest.mark.asyncio
async def test_executor_executes_when_only_requester_turn_active(tmp_path):
    """The chat/agent turn that filed the restart is itself an active
    request marker. When it is the ONLY thing in flight the restart must
    proceed — the requester's own marker must not deadlock the restart it
    asked for (#1561).
    """
    backend = await _backend(tmp_path)
    agent = _make_agent(backend)
    _attach_lifecycle(agent)
    # The in-flight chat turn that will file the restart.
    agent.register_active_request("chat-turn-1")

    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    created = await feat.request_restart(reason="config landed")
    req_id = created.data["request"]["id"]
    # The row records the requester's turn.
    row = await get_request(backend, req_id)
    assert row.requester_request_id == "chat-turn-1"

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 1
    assert result.data["executed"][0]["request_id"] == req_id
    # The requester marker was NOT swept (it's fresh, in flight) — it was
    # merely ignored for this restart's blocker count.
    assert "chat-turn-1" in agent._active_request_ids


@pytest.mark.asyncio
async def test_executor_defers_when_requester_plus_other_active(tmp_path):
    """Requester turn plus a second, unrelated active request → the
    second request still blocks an idle_agents_only restart (#1561).
    """
    backend = await _backend(tmp_path)
    agent = _make_agent(backend)
    _attach_lifecycle(agent)
    # A pre-existing unrelated request, then the chat turn that files the
    # restart (so _current_request_id points at the requester's turn).
    agent.register_active_request("other-req")
    agent.register_active_request("chat-turn-1")

    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    created = await feat.request_restart(reason="config landed")
    req_id = created.data["request"]["id"]

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 0
    assert len(result.data["deferred"]) == 1
    reason = result.data["deferred"][0]["reason"]
    assert "busy" in reason
    # Only the unrelated request counts as a blocker.
    assert "1 active request id(s)" in reason


@pytest.mark.asyncio
async def test_executor_defers_for_unrelated_active_request(tmp_path):
    """A restart filed with no requester turn must still defer when an
    unrelated request is active — the ignore only applies to the
    requester's own marker (#1561).
    """
    backend = await _backend(tmp_path)
    agent = _make_agent(backend)
    _attach_lifecycle(agent)

    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    # File while idle so requester_request_id is empty, then an unrelated
    # request goes in flight.
    created = await feat.request_restart(reason="config landed")
    req_id = created.data["request"]["id"]
    row = await get_request(backend, req_id)
    assert row.requester_request_id == ""
    agent.register_active_request("unrelated-req")

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 0
    assert len(result.data["deferred"]) == 1
    assert "busy" in result.data["deferred"][0]["reason"]


@pytest.mark.asyncio
async def test_executor_executes_on_busy_with_timeout_policy(tmp_path):
    backend = await _backend(tmp_path)
    agent = _make_agent(backend)
    agent._active_request_ids.add("req-busy")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    # File a request, then back-date requested_at past the 5-min
    # timeout so the policy allows execution despite a busy agent.
    req = await insert_request(
        backend,
        requested_by_agent=agent.did,
        reason="r",
        policy="allow_busy_after_timeout",
    )
    aged = (
        datetime.now(timezone.utc) - timedelta(seconds=600)
    ).isoformat()
    await backend.execute(
        "UPDATE restart_requests SET requested_at = ? WHERE id = ?",
        (aged, req.id),
    )

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 1
    assert result.data["executed"][0]["request_id"] == req.id


@pytest.mark.asyncio
async def test_executor_never_runs_manual_only_policy(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    await feat.request_restart(reason="r", policy="manual_only")
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()
    assert mock_spawn.call_count == 0
    assert len(result.data["deferred"]) == 1
    assert "manual_only" in result.data["deferred"][0]["reason"]


@pytest.mark.asyncio
async def test_executor_processes_higher_urgency_first(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    low = await feat.request_restart(reason="lo", urgency="low")
    high = await feat.request_restart(reason="hi", urgency="critical")
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()
    # Only the highest-urgency one is dispatched per poll.
    assert len(result.data["executed"]) == 1
    assert (
        result.data["executed"][0]["request_id"]
        == high.data["request"]["id"]
    )
    # Low-urgency one is still pending for the next poll.
    rows = await list_requests(backend, status="pending")
    assert len(rows) == 1
    assert rows[0].id == low.data["request"]["id"]


@pytest.mark.asyncio
async def test_executor_recovers_on_spawn_failure(tmp_path):
    """If the detached subprocess spawn raises, the row must move
    back from executing to pending so the next poll retries —
    not be left stuck in executing forever.
    """
    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="r")
    req_id = created.data["request"]["id"]

    with patch.object(
        RestartCoordinatorFeature,
        "_spawn_restart_subprocess",
        side_effect=OSError("kestrel binary missing"),
    ):
        await feat.restart_coordinator()

    row = await get_request(backend, req_id)
    assert row.status == "pending"
    assert "spawn failed" in row.status_reason


# ---------------------------------------------------------------------------
# Post-restart wakeup sweep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_restart_sweep_marks_completed_and_emits_signal(tmp_path):
    """A fresh feature constructed after restart must mark any
    ``executing`` row owned by this agent as ``completed`` and emit
    one ``restart.completed`` COGNITION signal so the requesting
    agent wakes.
    """
    backend = await _backend(tmp_path)
    # Pre-seed an executing row as if a prior poll spawned the
    # subprocess that took us down and brought us back up.
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )

    dispatcher = _CapturingDispatcher()
    registry = _StubRegistry()
    agent = _make_agent(backend, dispatcher=dispatcher, registry=registry)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    await feat.on_agent_ready()  # post-restart wake fires once the agent is ready

    row = await get_request(backend, req.id)
    assert row.status == "completed"
    assert len(dispatcher.signals) == 1
    sig = dispatcher.signals[0]
    assert sig.payload["request_id"] == req.id
    assert sig.payload["reason"] == "pre-restart"
    assert sig.source == "restart.completed"


@pytest.mark.asyncio
async def test_post_restart_sweep_only_touches_this_agents_rows(tmp_path):
    """Multi-agent: the sweep must NOT mark another agent's
    in-flight executing row as completed.
    """
    backend = await _backend(tmp_path)
    other = await insert_request(
        backend, requested_by_agent="did:test:other", reason="other",
    )
    await update_status(
        backend, other.id, status="executing",
        expected_current_status="pending",
    )

    agent = _make_agent(backend, did="did:test:agent")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    await feat.on_agent_ready()  # sweep runs but must not touch another agent's row

    row = await get_request(backend, other.id)
    assert row.status == "executing"


@pytest.mark.asyncio
async def test_update_status_race_only_winner_returns_true(tmp_path):
    """Two coordinators both call UPDATE ... WHERE status='pending';
    only the row whose pre-image actually matched updated, so only
    that caller must observe True. The pre-codex-P1 fix returned
    True for both because the int rowcount was misinterpreted as
    "no cursor" and a fallback SELECT was used.
    """
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="a", reason="r",
    )
    # First update wins.
    a_ok = await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )
    # Second update sees status already executing — must NOT update.
    b_ok = await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )
    assert a_ok is True
    assert b_ok is False


@pytest.mark.asyncio
async def test_executor_defers_when_no_introspection_surface(tmp_path):
    """If the agent doesn't expose ANY in-flight surface, the
    safety check must conservatively report busy. Pre-codex-P1-fix
    the absence of introspection was treated as "idle", which
    defeated the idle_agents_only policy on production agents.
    """
    backend = await _backend(tmp_path)
    raw_storage = SimpleNamespace(db=backend)
    bare_agent = SimpleNamespace(
        did="did:test:agent",
        agent_id="did:test:agent",
        _raw_storage=raw_storage,
        storage=None,
        dispatcher=None,
        signal_registry=None,
        features={"RestartCoordinatorFeature": True},
    )
    feat = RestartCoordinatorFeature(bare_agent)
    await feat.initialize()
    await feat.request_restart(reason="r")
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()
    assert mock_spawn.call_count == 0
    assert len(result.data["deferred"]) == 1
    assert "no idleness" in result.data["deferred"][0]["reason"]


@pytest.mark.asyncio
async def test_post_restart_sweep_retries_when_dispatcher_raises(tmp_path):
    """If the dispatcher is present but ``enqueue_signal`` raises,
    the row must STAY in executing so a future agent boot's sweep
    can retry. Pre-codex-P2-fix the row was already terminalized
    so the wake was permanently lost.
    """
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="r",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )

    class _BrokenDispatcher:
        def enqueue_signal(self, signal):
            raise RuntimeError("dispatcher down")

    agent = _make_agent(backend, dispatcher=_BrokenDispatcher())
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    await feat.on_agent_ready()  # sweep dispatches; broken dispatcher leaves it executing

    row = await get_request(backend, req.id)
    assert row.status == "executing", (
        f"row must stay executing for retry; got {row.status!r}"
    )


# ---------------------------------------------------------------------------
# Post-restart wake reaches the REAL durable/resumption path (#1796)
# ---------------------------------------------------------------------------


class _RealDispatchAgent:
    """Minimal agent wired to a REAL SignalDispatcher so a swept
    ``restart.completed`` signal travels the same pipeline every other
    COGNITION signal uses — registry validation → dispatch → the agent's
    ``process_input`` (the resuming turn) — instead of being captured by
    a stub (#1796).
    """

    def __init__(self, backend, did="did:test:agent"):
        self.did = did
        self.agent_id = did
        self._raw_storage = SimpleNamespace(db=backend)
        self.storage = None
        self.signal_registry = None  # set after the feature registers
        self.dispatcher = None       # set by the test wiring
        self._active_request_ids = set()
        self._background_tasks = set()
        self.features = {"RestartCoordinatorFeature": True}
        self.background_tasks: list[asyncio.Task] = []
        self.process_input_calls: list[str] = []
        # Session ids the wake turns were dispatched into (#1809) — parallel
        # to process_input_calls. None = system-initiated (no origin session).
        self.process_input_sessions: list = []
        # When set, process_input raises to model a wake that failed
        # inside the resuming turn (dispatcher records Status.FAILED).
        self.process_input_should_raise = False

    async def process_input(self, prompt: str, session_id=None):
        if self.process_input_should_raise:
            raise RuntimeError("simulated cognition failure")
        self.process_input_calls.append(prompt)
        self.process_input_sessions.append(session_id)
        return "resumed"

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task

    async def drain_background_tasks(self):
        """Await every supervised task (dispatch, ack supervisor, and the
        signal_log writes they spawn) until the queue is quiescent."""
        while True:
            pending = [t for t in self.background_tasks if not t.done()]
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)


async def _real_dispatch_feature(tmp_path, **kwargs):
    """Build a RestartCoordinatorFeature wired to a real SignalDispatcher.

    Returns ``(feature, backend, agent)``. The feature's ``initialize``
    registers the ``restart.completed`` source into the same registry the
    dispatcher routes against, so the post-restart sweep's wake is
    delivered for real.
    """
    from kestrel_sovereign.signals import (
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )

    backend = await _backend(tmp_path)
    agent = _RealDispatchAgent(backend, **kwargs)

    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )
    agent.signal_registry = registry
    agent.dispatcher = dispatcher

    feat = RestartCoordinatorFeature(agent)
    return feat, backend, agent


@pytest.mark.asyncio
async def test_post_restart_wake_reaches_process_input_and_then_completes(
    tmp_path,
):
    """The swept ``restart.completed`` wake must reach the agent's
    ``process_input`` (the real resumption path), and the row must be
    terminalized to ``completed`` only AFTER that delivery lands — not
    merely because ``enqueue_signal`` returned (#1796).
    """
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )

    await feat.initialize()
    await feat.on_agent_ready()  # fires the post-restart wake now the agent is ready
    # The wake runs as a supervised background task; the row is still
    # executing until that dispatch lands.
    row_mid = await get_request(backend, req.id)
    assert row_mid.status == "executing"

    await agent.drain_background_tasks()

    # The wake reached the agent's resuming turn...
    assert len(agent.process_input_calls) == 1
    assert req.id in agent.process_input_calls[0]
    # ...and ONLY THEN did the row terminalize.
    row = await get_request(backend, req.id)
    assert row.status == "completed"


@pytest.mark.asyncio
async def test_post_restart_wake_failure_leaves_row_retryable(tmp_path):
    """If the wake's resuming turn fails (dispatch returns FAILED), the
    row must STAY executing so a later sweep retries it — the pre-#1796
    code terminalized on enqueue success and lost the wake forever.
    """
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    agent.process_input_should_raise = True
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )

    await feat.initialize()
    await feat.on_agent_ready()
    await agent.drain_background_tasks()

    row = await get_request(backend, req.id)
    assert row.status == "executing", (
        f"a failed wake must stay retryable; got {row.status!r}"
    )


@pytest.mark.asyncio
async def test_restart_coordinator_cron_retries_undelivered_wake(tmp_path):
    """The ``restart_coordinator`` cron tick is the retry backstop: an
    ``executing`` row whose wake failed on the init sweep must be re-woken
    on a later tick (without waiting for a full reboot), and once the
    resuming turn succeeds the row terminalizes (#1796).
    """
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    agent.process_input_should_raise = True
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )

    # Ready sweep: wake fails, row stays executing.
    await feat.initialize()
    await feat.on_agent_ready()
    await agent.drain_background_tasks()
    assert (await get_request(backend, req.id)).status == "executing"
    assert agent.process_input_calls == []

    # The resuming turn now succeeds; a cron tick must re-wake and complete.
    # Re-anchor the dispatcher's coalescing window to model the >30s gap
    # between the init sweep and a production 1/min cron tick (a fast
    # in-test retry would otherwise coalesce against the failed wake).
    agent.dispatcher.notify_resume(60.0)
    agent.process_input_should_raise = False
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ):
        await feat.restart_coordinator()
    await agent.drain_background_tasks()

    assert len(agent.process_input_calls) == 1
    assert (await get_request(backend, req.id)).status == "completed"


@pytest.mark.asyncio
async def test_cron_does_not_complete_same_process_executing_row(tmp_path):
    """A row this SAME process just crossed to ``executing`` (the detached
    restart is still in flight, or failed to kill the parent) must NOT be
    falsely terminalized as ``completed`` by a later cron tick — the reap
    backstop only wakes rows left ``executing`` by a PRIOR process (#1796).

    Without the per-process boot stamp, the live-process cron reap would
    fire a ``restart.completed`` wake and complete a restart that never
    happened, masking a failed restart as success.
    """
    from kestrel_sovereign.features.restart_coordinator.feature import (
        _PROCESS_BOOT_ID,
    )

    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()
    created = await feat.request_restart(reason="ship")
    req_id = created.data["request"]["id"]

    # First tick: cross the row to executing and (mock-)spawn the restart.
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ):
        await feat.restart_coordinator()
    await agent.drain_background_tasks()

    row = await get_request(backend, req_id)
    assert row.status == "executing"
    # The row carries THIS process's boot stamp — the restart is in flight.
    assert row.executing_boot_id == _PROCESS_BOOT_ID
    assert agent.process_input_calls == []

    # A later cron tick in the SAME process must NOT wake/complete it (the
    # detached restart has not replaced this process).
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ):
        await feat.restart_coordinator()
    await agent.drain_background_tasks()

    assert agent.process_input_calls == []
    row = await get_request(backend, req_id)
    assert row.status == "executing", (
        "a same-process in-flight restart must stay visibly executing, "
        f"not be falsely completed; got {row.status!r}"
    )


# ---------------------------------------------------------------------------
# update_then_restart — audited update-and-restart (#1539)
# ---------------------------------------------------------------------------


def _git_checkout(tmp_path):
    """Make ``tmp_path`` look like a git checkout for repo validation."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    return str(tmp_path)


@pytest.mark.asyncio
async def test_request_update_then_restart_creates_row(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    repo = _git_checkout(tmp_path)
    result = await feat.request_restart(
        reason="ship merged fix",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="main",
        repo_path=repo,
    )
    assert result.status is ToolResultStatus.OK
    req = result.data["request"]
    assert req["operation"] == "update_then_restart"
    assert req["update_profile"] == "sovereign_local_uv_sync"
    assert req["update_target_ref"] == "main"
    assert req["update_repo_path"] == repo
    # Persisted with the new fields intact.
    row = await get_request(backend, req["id"])
    assert row.operation == "update_then_restart"
    assert row.update_target_ref == "main"


@pytest.mark.asyncio
async def test_request_update_then_restart_rejects_unknown_profile(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    repo = _git_checkout(tmp_path)
    result = await feat.request_restart(
        reason="r",
        operation="update_then_restart",
        update_profile="rm_rf_everything",
        target_ref="main",
        repo_path=repo,
    )
    assert result.error is not None
    assert "update_profile" in result.error


@pytest.mark.asyncio
async def test_request_update_then_restart_rejects_bad_target_ref(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    repo = _git_checkout(tmp_path)
    # Missing ref.
    missing = await feat.request_restart(
        reason="r",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="",
        repo_path=repo,
    )
    assert missing.error is not None
    assert "target_ref" in missing.error
    # Option-injection-style ref must be rejected.
    crafted = await feat.request_restart(
        reason="r",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="--upload-pack=evil",
        repo_path=repo,
    )
    assert crafted.error is not None
    assert "target_ref" in crafted.error


@pytest.mark.asyncio
async def test_request_update_then_restart_rejects_non_git_repo(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    result = await feat.request_restart(
        reason="r",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="main",
        repo_path=str(tmp_path / "not-a-checkout"),
    )
    assert result.error is not None
    assert "repo_path" in result.error


@pytest.mark.asyncio
async def test_request_restart_rejects_unknown_operation(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    result = await feat.request_restart(reason="r", operation="nuke")
    assert result.error is not None
    assert "operation" in result.error


@pytest.mark.asyncio
async def test_list_and_cancel_update_then_restart(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    repo = _git_checkout(tmp_path)
    created = await feat.request_restart(
        reason="r",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="v1.2.3",
        repo_path=repo,
    )
    req_id = created.data["request"]["id"]
    listed = await feat.list_restart_requests()
    assert listed.data["count"] == 1
    assert listed.data["requests"][0]["operation"] == "update_then_restart"
    cancel = await feat.cancel_restart_request(req_id, reason="stand down")
    assert cancel.status is ToolResultStatus.OK
    row = await get_request(backend, req_id)
    assert row.status == "canceled"


@pytest.mark.asyncio
async def test_coordinator_rejects_unknown_profile_terminal(tmp_path):
    """A row carrying an unknown profile (inserted outside the tool's
    validation) must be terminally rejected by the coordinator, not
    retried forever or executed.
    """
    feat, backend = await _make_feature(tmp_path)
    req = await insert_request(
        backend,
        requested_by_agent="did:test:agent",
        reason="r",
        operation="update_then_restart",
        update_profile="bogus_profile",
        update_target_ref="main",
        update_repo_path=_git_checkout(tmp_path),
    )
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        await feat.restart_coordinator()
    assert mock_spawn.call_count == 0
    row = await get_request(backend, req.id)
    assert row.status == "rejected"
    assert "unknown update profile" in row.status_reason


@pytest.mark.asyncio
async def test_coordinator_runs_update_then_spawns_restart(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    repo = _git_checkout(tmp_path)
    created = await feat.request_restart(
        reason="ship",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="main",
        repo_path=repo,
    )
    req_id = created.data["request"]["id"]

    fake_update = {
        "ok": True,
        "profile": "sovereign_local_uv_sync",
        "repo_path": repo,
        "target_ref": "main",
        "resolved_ref": "abc1234",
        "steps": [],
        "migration": {"ran": False, "reason": "additive"},
        "failed_step": None,
    }

    async def _fake_run_update(self, req, profile):
        return fake_update

    with patch.object(
        RestartCoordinatorFeature, "_run_update", _fake_run_update,
    ), patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 1
    assert result.data["executed"][0]["request_id"] == req_id
    row = await get_request(backend, req_id)
    assert row.status == "executing"
    # Update audit log persisted on the row.
    assert row.update_log_dict()["resolved_ref"] == "abc1234"


@pytest.mark.asyncio
async def test_coordinator_update_failure_leaves_retryable(tmp_path):
    """If the update fails before restart, the row must NOT restart and
    must be left retryable (back to pending) with a clear reason.
    """
    feat, backend = await _make_feature(tmp_path)
    repo = _git_checkout(tmp_path)
    created = await feat.request_restart(
        reason="ship",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="main",
        repo_path=repo,
    )
    req_id = created.data["request"]["id"]

    async def _fake_run_update(self, req, profile):
        return {
            "ok": False,
            "profile": "sovereign_local_uv_sync",
            "repo_path": repo,
            "target_ref": "main",
            "resolved_ref": "",
            "steps": [{"step": "install", "ok": False}],
            "migration": {"ran": False, "reason": "additive"},
            "failed_step": "install",
        }

    with patch.object(
        RestartCoordinatorFeature, "_run_update", _fake_run_update,
    ), patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 0
    row = await get_request(backend, req_id)
    assert row.status == "pending"
    assert "install" in row.status_reason
    assert row.update_log_dict()["failed_step"] == "install"
    assert len(result.data["deferred"]) == 1


@pytest.mark.asyncio
async def test_run_update_records_steps_and_resolved_ref(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    profile = get_update_profile("sovereign_local_uv_sync")
    req = SimpleNamespace(
        update_repo_path=str(tmp_path),
        update_target_ref="main",
        update_allow_migrations=False,
    )

    async def _fake_step(self, step):
        out = "deadbeef" if step.name == "resolve_ref" else ""
        return {
            "step": step.name,
            "argv": list(step.argv),
            "returncode": 0,
            "ok": True,
            "stdout_tail": out,
            "stderr_tail": "",
        }

    with patch.object(
        RestartCoordinatorFeature, "_run_update_step", _fake_step,
    ):
        update = await feat._run_update(req, profile)

    assert update["ok"] is True
    assert update["resolved_ref"] == "deadbeef"
    assert {s["step"] for s in update["steps"]} >= {
        "fetch", "checkout", "install", "resolve_ref",
    }
    assert update["migration"]["ran"] is False


@pytest.mark.asyncio
async def test_run_update_stops_at_first_failing_step(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    profile = get_update_profile("sovereign_local_uv_sync")
    req = SimpleNamespace(
        update_repo_path=str(tmp_path),
        update_target_ref="main",
        update_allow_migrations=False,
    )

    async def _fake_step(self, step):
        ok = step.name == "fetch"  # checkout fails
        return {
            "step": step.name,
            "argv": list(step.argv),
            "returncode": 0 if ok else 1,
            "ok": ok,
            "stdout_tail": "",
            "stderr_tail": "boom" if not ok else "",
        }

    with patch.object(
        RestartCoordinatorFeature, "_run_update_step", _fake_step,
    ):
        update = await feat._run_update(req, profile)

    assert update["ok"] is False
    assert update["failed_step"] == "checkout"
    # Stopped before install/resolve_ref ran.
    assert {s["step"] for s in update["steps"]} == {"fetch", "checkout"}


@pytest.mark.asyncio
async def test_post_restart_sweep_signal_includes_update_metadata(tmp_path):
    """The completion signal for an update_then_restart row must carry
    operation + target ref + the resolved commit so the agent can verify
    it booted into the requested ref.
    """
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend,
        requested_by_agent="did:test:agent",
        reason="merged fix",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        update_target_ref="main",
        update_repo_path=str(tmp_path),
    )
    await record_update_log(
        backend, req.id, json.dumps({"resolved_ref": "cafef00d"}),
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )

    dispatcher = _CapturingDispatcher()
    agent = _make_agent(backend, dispatcher=dispatcher)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    await feat.on_agent_ready()

    assert len(dispatcher.signals) == 1
    payload = dispatcher.signals[0].payload
    assert payload["operation"] == "update_then_restart"
    assert payload["target_ref"] == "main"
    assert payload["resolved_ref"] == "cafef00d"
    assert payload["update_profile"] == "sovereign_local_uv_sync"


def test_is_valid_target_ref_guards():
    assert is_valid_target_ref("main")
    assert is_valid_target_ref("v1.2.3")
    assert is_valid_target_ref("feature/foo-bar")
    assert is_valid_target_ref("a1b2c3d4")
    assert not is_valid_target_ref("")
    assert not is_valid_target_ref("--upload-pack=x")
    assert not is_valid_target_ref("a..b")
    assert not is_valid_target_ref("foo; rm -rf /")


# ---------------------------------------------------------------------------
# Real-git profile behaviour (#1539) — the update profile must actually
# advance a branch checkout, not silently no-op. This exercises the real
# fetch/checkout/resolve_ref steps (install is skipped — no `uv sync` in a
# unit test) against a local bare-remote + clone, the kestrel-talon
# test_git_worktree.py pattern.
# ---------------------------------------------------------------------------


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=str(cwd), check=True, capture_output=True,
    )


def _head(repo):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.mark.asyncio
async def test_profile_checkout_lands_on_fetched_branch_commit(tmp_path):
    """A named ``git checkout main`` after a fetch stays on the STALE
    local commit (fetch updates origin/main, not local main); the profile
    must instead land the working checkout on the freshly-fetched commit.
    This is the headline use case — update local checkout to a branch ref
    then restart — and is invisible to the fully-mocked unit tests.
    """
    if shutil.which("git") is None:
        pytest.skip("git not available")

    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))

    # Seed main with commit v1 and push.
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    (seed / "file.txt").write_text("v1")
    _git(seed, "add", "file.txt")
    _git(seed, "commit", "-m", "v1")
    _git(seed, "push", "origin", "main")

    # The local checkout the coordinator will update — currently at v1.
    work = tmp_path / "work"
    _git(tmp_path, "clone", str(remote), str(work))
    head_before = _head(work)

    # A new commit v2 lands on remote main AFTER the local checkout exists.
    (seed / "file.txt").write_text("v2")
    _git(seed, "add", "file.txt")
    _git(seed, "commit", "-m", "v2")
    _git(seed, "push", "origin", "main")
    remote_head = _head(seed)
    assert remote_head != head_before

    feat, _ = await _make_feature(tmp_path)
    profile = get_update_profile("sovereign_local_uv_sync")
    steps = profile.build_steps(
        repo_path=str(work), target_ref="main", allow_migrations=False,
    )
    for step in steps:
        if step.name == "install":
            continue  # do not run `uv sync` in a unit test
        outcome = await feat._run_update_step(step)
        assert outcome["ok"], f"step {step.name!r} failed: {outcome}"

    head_after = _head(work)
    assert head_after == remote_head, (
        "checkout must land on the fetched commit; a named branch "
        "checkout would have left it at the stale local commit"
    )
    assert head_after != head_before


@pytest.mark.asyncio
async def test_boot_resets_interrupted_updating_row(tmp_path):
    """A row left in ``updating`` by a host that went down mid-update must
    be reset to ``pending`` for retry on the next boot — NOT reported as a
    completed restart (no restart.completed signal), since the update
    never finished and we never restarted into the new code.
    """
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend,
        requested_by_agent="did:test:agent",
        reason="merged fix",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        update_target_ref="main",
        update_repo_path=str(tmp_path),
    )
    await update_status(
        backend, req.id, status="updating",
        expected_current_status="pending",
    )

    dispatcher = _CapturingDispatcher()
    agent = _make_agent(backend, dispatcher=dispatcher)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()

    row = await get_request(backend, req.id)
    assert row.status == "pending", (
        f"interrupted update must be retryable; got {row.status!r}"
    )
    assert "mid-update" in row.status_reason
    # No completion signal — nothing actually restarted.
    assert dispatcher.signals == []


@pytest.mark.asyncio
async def test_signal_source_registered_idempotent(tmp_path):
    """A second feature init must not double-register the signal
    source. The registry uses ``get`` to short-circuit.
    """
    backend = await _backend(tmp_path)
    registry = _StubRegistry()
    agent = _make_agent(backend, registry=registry)
    feat1 = RestartCoordinatorFeature(agent)
    await feat1.initialize()
    assert "restart.completed" in registry.by_name
    # Second init in the same process — no duplicate, no warning.
    feat2 = RestartCoordinatorFeature(agent)
    await feat2.initialize()
    # Still exactly one registration.
    assert len(registry.registered) == 1


# ---------------------------------------------------------------------------
# Chat-visible restart_status events (#1551)
# ---------------------------------------------------------------------------


def _attach_emit_capture(feat) -> list:
    """Give the feature's agent a capturing ``emit_event`` and return the
    list it records ``(event_type, payload)`` tuples into.
    """
    captured: list = []

    async def _emit(event_type, data):
        captured.append((event_type, data))

    feat.agent.emit_event = _emit
    return captured


def _restart_status_events(captured) -> list:
    return [d for (t, d) in captured if t == "restart_status"]


@pytest.mark.asyncio
async def test_request_restart_emits_pending_status_event(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    captured = _attach_emit_capture(feat)
    result = await feat.request_restart(reason="kestrel.toml change landed")
    req_id = result.data["request"]["id"]

    events = _restart_status_events(captured)
    assert len(events) == 1
    ev = events[0]
    assert ev["request_id"] == req_id
    assert ev["status"] == "pending"
    assert ev["operation"] == "restart_only"
    assert ev["requested_by_agent"] == "did:test:agent"
    assert ev["reason"] == "kestrel.toml change landed"
    assert ev["deferral_reason"] == ""


@pytest.mark.asyncio
async def test_request_restart_without_emit_event_is_safe(tmp_path):
    """Headless/test agents without ``emit_event`` must not break the
    request lifecycle — the status event is best-effort only.
    """
    feat, backend = await _make_feature(tmp_path)
    # Ensure no emit_event surface exists on the agent.
    assert not hasattr(feat.agent, "emit_event")
    result = await feat.request_restart(reason="no emitter present")
    assert result.status is ToolResultStatus.OK
    rows = await list_requests(backend)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_coordinator_executing_emits_status_event(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="ship")
    req_id = created.data["request"]["id"]
    captured = _attach_emit_capture(feat)

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ):
        await feat.restart_coordinator()

    states = [e["status"] for e in _restart_status_events(captured)
              if e["request_id"] == req_id]
    assert "executing" in states


@pytest.mark.asyncio
async def test_coordinator_defer_emits_status_with_reason(tmp_path):
    # Busy agent → idle_agents_only policy defers; the deferral and its
    # reason must surface as a status event (#1551).
    backend = await _backend(tmp_path)
    agent = _make_agent(backend)
    agent._active_request_ids = {"req-1"}
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    created = await feat.request_restart(reason="ship")
    req_id = created.data["request"]["id"]
    captured = _attach_emit_capture(feat)

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        await feat.restart_coordinator()

    assert mock_spawn.call_count == 0
    deferred_events = [
        e for e in _restart_status_events(captured)
        if e["request_id"] == req_id and e["deferral_reason"]
    ]
    assert len(deferred_events) == 1
    ev = deferred_events[0]
    assert ev["status"] == "pending"
    assert "active request" in ev["deferral_reason"]


@pytest.mark.asyncio
async def test_cancel_emits_canceled_status_event(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="r")
    req_id = created.data["request"]["id"]
    captured = _attach_emit_capture(feat)

    await feat.cancel_restart_request(req_id, reason="never mind")

    events = [e for e in _restart_status_events(captured)
              if e["request_id"] == req_id]
    assert len(events) == 1
    assert events[0]["status"] == "canceled"
    assert "never mind" in events[0]["status_reason"]


@pytest.mark.asyncio
async def test_post_restart_sweep_emits_completed_status_event(tmp_path):
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )
    dispatcher = _CapturingDispatcher()
    agent = _make_agent(backend, dispatcher=dispatcher)
    captured: list = []

    async def _emit(event_type, data):
        captured.append((event_type, data))

    agent.emit_event = _emit
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    await feat.on_agent_ready()

    events = [e for e in _restart_status_events(captured)
              if e["request_id"] == req.id]
    assert len(events) == 1
    assert events[0]["status"] == "completed"
    assert events[0]["completed_at"]


@pytest.mark.asyncio
async def test_idle_ignores_signal_log_infra_tasks(tmp_path):
    """#1626: signal_log:* bookkeeping tasks are minted continuously by
    heartbeats/scheduler ticks; counting them as 'busy' wedged
    idle_agents_only restarts forever. They must be excluded from the idle
    check, while real work (signal_dispatch:*) still defers a restart."""
    feat, _ = await _make_feature(tmp_path)

    async def _never():
        await asyncio.Event().wait()

    log_task = asyncio.create_task(_never(), name="signal_log:heartbeat:abc123")
    sweep_task = asyncio.create_task(_never(), name="a2a_question_expiry_sweep")
    work_task = asyncio.create_task(_never(), name="signal_dispatch:heartbeat:abc123")
    try:
        # Only infra bookkeeping + the permanent peers sweep in flight ->
        # the agent is idle (neither must wedge an idle restart).
        feat.agent._background_tasks = {log_task, sweep_task}
        idle = feat._agent_appears_idle()
        assert idle["idle"] is True, idle

        # A real signal_dispatch task still defers, and infra tasks don't
        # inflate the reported count.
        feat.agent._background_tasks = {log_task, sweep_task, work_task}
        busy = feat._agent_appears_idle()
        assert busy["idle"] is False
        assert busy["reason"] == "1 background task(s) in flight"
    finally:
        log_task.cancel()
        sweep_task.cancel()
        work_task.cancel()


# ---------------------------------------------------------------------------
# #1809: prompt wake (on_agent_ready) + same-session routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_origin_session_id_roundtrips_in_store(tmp_path):
    """insert_request persists origin_session_id and from_row reads it back."""
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:t:a", reason="r",
        origin_session_id="sess-abc",
    )
    assert req.origin_session_id == "sess-abc"
    fetched = await get_request(backend, req.id)
    assert fetched.origin_session_id == "sess-abc"


@pytest.mark.asyncio
async def test_request_restart_captures_origin_session(tmp_path):
    """request_restart records the chat session it was filed from (the
    session_id_var ContextVar set per HTTP turn)."""
    from kestrel_sovereign.logging_config import session_id_var

    feat, backend = await _make_feature(tmp_path)
    token = session_id_var.set("chat-session-42")
    try:
        result = await feat.request_restart(reason="ship it")
    finally:
        session_id_var.reset(token)

    req_id = result.data["request"]["id"]
    row = await get_request(backend, req_id)
    assert row.origin_session_id == "chat-session-42"


@pytest.mark.asyncio
async def test_request_restart_captures_active_session(tmp_path):
    """The authoritative per-turn _active_session_id (set by the turn body from
    the JSON-body session — the primary chat path) is captured and takes
    precedence over the logging ContextVar."""
    from kestrel_sovereign.logging_config import session_id_var

    feat, backend = await _make_feature(tmp_path)
    feat.agent._active_session_id = "body-session-9"
    token = session_id_var.set("header-session-1")
    try:
        result = await feat.request_restart(reason="ship it")
    finally:
        session_id_var.reset(token)

    row = await get_request(backend, result.data["request"]["id"])
    assert row.origin_session_id == "body-session-9"  # active wins over header


@pytest.mark.asyncio
async def test_request_restart_no_session_is_blank(tmp_path):
    """With no chat session in context (CLI/system), origin_session_id is blank."""
    feat, backend = await _make_feature(tmp_path)
    result = await feat.request_restart(reason="system filed")
    row = await get_request(backend, result.data["request"]["id"])
    assert row.origin_session_id == ""


def test_build_signal_carries_origin_session():
    """The restart.completed Signal routes to the request's origin session;
    empty origin → session_id None (system-initiated)."""
    from kestrel_sovereign.features.restart_coordinator.store import RestartRequest
    from kestrel_sovereign.signals.sources.restart import (
        build_signal_for_restart_completed,
    )

    base = dict(
        id="req-1", requested_by_agent="did:a", reason="r", requested_at="t",
        desired_window="", urgency="normal", policy="idle_agents_only",
        status="executing", status_reason="", completed_at=None,
    )
    with_sess = RestartRequest(origin_session_id="sess-xyz", **base)
    sig = build_signal_for_restart_completed(
        with_sess, target_agent="did:a", completed_at="now",
    )
    assert sig.session_id == "sess-xyz"

    without = RestartRequest(origin_session_id="", **base)
    sig2 = build_signal_for_restart_completed(
        without, target_agent="did:a", completed_at="now",
    )
    assert sig2.session_id is None


@pytest.mark.asyncio
async def test_wake_routes_into_origin_session(tmp_path):
    """End-to-end: a row filed from a session wakes the agent IN that session
    (process_input receives the origin session_id), not a fresh one (#1809)."""
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="r",
        origin_session_id="chat-7",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )
    await feat.initialize()
    await feat.on_agent_ready()
    await agent.drain_background_tasks()

    assert agent.process_input_calls and req.id in agent.process_input_calls[0]
    assert agent.process_input_sessions == ["chat-7"]


@pytest.mark.asyncio
async def test_wake_without_origin_session_is_system_initiated(tmp_path):
    """A row with no origin session wakes system-initiated (session_id None)."""
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="r",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )
    await feat.initialize()
    await feat.on_agent_ready()
    await agent.drain_background_tasks()

    assert agent.process_input_calls
    assert agent.process_input_sessions == [None]


@pytest.mark.asyncio
async def test_initialize_alone_does_not_wake_only_on_agent_ready(tmp_path):
    """The post-restart wake must NOT fire during initialize() (the context
    manager doesn't exist yet); it fires from on_agent_ready, which the agent
    calls once fully initialized. This is the #1809 promptness fix — the wake
    happens at end-of-init, not on a later cron tick."""
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )
    dispatcher = _CapturingDispatcher()
    registry = _StubRegistry()
    agent = _make_agent(backend, dispatcher=dispatcher, registry=registry)
    feat = RestartCoordinatorFeature(agent)

    await feat.initialize()
    # initialize() must NOT have dispatched a wake (too early — pre-context).
    assert dispatcher.signals == []
    assert (await get_request(backend, req.id)).status == "executing"

    await feat.on_agent_ready()
    # now the wake fires and the row terminalizes.
    assert len(dispatcher.signals) == 1
    assert (await get_request(backend, req.id)).status == "completed"


# ---------------------------------------------------------------------------
# #1809 follow-up: restart visible in chat (live wake + persisted bubble)
# ---------------------------------------------------------------------------


def test_restart_completed_signal_is_user_visible_with_summary():
    """The wake signal must be USER_VISIBLE with a result_summary so the
    dispatcher emits signal_completed and the frontend renders it live."""
    from kestrel_sdk.signals import Visibility
    from kestrel_sovereign.features.restart_coordinator.store import RestartRequest
    from kestrel_sovereign.signals.sources.restart import (
        build_restart_completed_registration,
        build_signal_for_restart_completed,
    )

    reg = build_restart_completed_registration()
    assert reg.result_summary is not None
    assert reg.result_summary("I'm back, booted d4e86bf.") == "I'm back, booted d4e86bf."
    assert reg.result_summary(None) == ""

    req = RestartRequest(
        id="r1", requested_by_agent="did:a", reason="x", requested_at="t",
        desired_window="", urgency="normal", policy="idle_agents_only",
        status="executing", status_reason="", completed_at=None,
        origin_session_id="1114",
    )
    sig = build_signal_for_restart_completed(req, target_agent="did:a", completed_at="now")
    assert sig.visibility == Visibility.USER_VISIBLE
    assert sig.session_id == "1114"
