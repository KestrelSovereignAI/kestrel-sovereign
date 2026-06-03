"""Tests for the durable restart coordinator (#1512).

Covers the store layer (table init, request lifecycle, status
transitions, race-protection on cancel) and the feature surface
(request/list/cancel @tools, the executor cron task's safety gate,
and the post-restart sweep that wakes the requesting agent).
"""

from __future__ import annotations

import asyncio
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
    update_status,
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

    row = await get_request(backend, req.id)
    assert row.status == "executing", (
        f"row must stay executing for retry; got {row.status!r}"
    )


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
