"""Tests for the durable restart coordinator (#1512).

Covers the store layer (table init, request lifecycle, status
transitions, race-protection on cancel) and the feature surface
(request/list/cancel @tools, the executor cron task's safety gate,
and the post-restart sweep that wakes the requesting agent).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from kestrel_sdk.hooks.base import HookEvent, HookInput, HookOutput
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.restart_coordinator import (
    RestartCoordinatorFeature,
)
from kestrel_sovereign.features.restart_coordinator.event_store import (
    list_events_for_request,
)
from kestrel_sovereign.features.restart_coordinator.feature import (
    _MAX_NAMED_BUSY_KINDS,
    MAX_IDLE_ONLY_DEFERRAL_SECONDS,
    _describe_background_tasks,
)
from kestrel_sovereign.features.restart_coordinator.store import (
    clear_deferral_started,
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

    def register_with_policy(self, reg, policy=None):
        """Model the real ``SourceRegistry.register_with_policy`` (#2522).

        The feature now registers under an explicit
        :class:`RegistrationPolicy`, so the stub must return the same
        structured :class:`RegistrationOutcome` envelope. Contract equivalence
        is decided by the *real* ``SourceRegistry.contract_signature``, so an
        identical re-registration on a second ``initialize()`` is a no-op
        ``ALREADY_EQUIVALENT`` rather than a duplicate.
        """
        from kestrel_sovereign.signals import (
            RegistrationOutcome,
            RegistrationState,
            SourceRegistry,
        )

        existing = self.by_name.get(reg.name)
        if existing is None:
            self.registered.append(reg)
            self.by_name[reg.name] = reg
            return RegistrationOutcome(reg.name, RegistrationState.REGISTERED)
        if SourceRegistry.contract_equivalent(existing, reg):
            return RegistrationOutcome(
                reg.name, RegistrationState.ALREADY_EQUIVALENT
            )
        return RegistrationOutcome(
            reg.name, RegistrationState.MISMATCH, "stub contract mismatch"
        )

    def get(self, name):
        return self.by_name.get(name)


_test_databases: list[tuple[AsyncDatabase, object]] = []
_real_dispatch_lifecycles: list[tuple[object, object]] = []


def _sqlite_worker_is_alive(connection: object) -> bool:
    """Support both aiosqlite worker-thread ownership models.

    aiosqlite 0.22 stores its worker on ``Connection._thread``; older releases
    subclassed ``Thread`` directly.  The lifecycle contract is identical.
    """
    worker = getattr(connection, "_thread", connection)
    return bool(worker.is_alive())


def _track_test_database(db: AsyncDatabase) -> AsyncDatabase:
    """Register a test-owned database for closure before its test loop ends."""
    connection = db._backend._connection
    assert connection is not None
    _test_databases.append((db, connection))
    return db


@pytest_asyncio.fixture(autouse=True)
async def _close_test_owned_resources():
    """Model production ownership: stop wakes before closing their database.

    ``SignalDispatcher`` owns the dispatch task through the agent and the
    restart feature owns its acknowledgement task.  Closing SQLite first lets
    either task submit work to an event loop that pytest has already torn down,
    which is the same lifecycle inversion that made these tests flaky under
    xdist.
    """
    _test_databases.clear()
    _real_dispatch_lifecycles.clear()
    try:
        yield
    finally:
        for feature, agent in reversed(_real_dispatch_lifecycles):
            await feature.shutdown()
            await agent.shutdown()

        for database, connection in reversed(_test_databases):
            await database.close()
            assert not _sqlite_worker_is_alive(connection), (
                "test-owned aiosqlite worker survived database shutdown"
            )

        _test_databases.clear()
        _real_dispatch_lifecycles.clear()


async def _backend(tmp_path):
    """Wrap SQLiteBackend in AsyncDatabase to match the production
    surface ``resolve_feature_database`` returns. ``AsyncDatabase``
    exposes ``fetchall`` (no underscore) while the bare backend uses
    ``fetch_all`` — store code must work against the wrapper.
    """
    raw = SQLiteBackend(str(tmp_path / "restart.db"))
    await raw.connect()
    db = _track_test_database(AsyncDatabase(raw))
    await ensure_restart_requests_table(db)
    return db


def _make_agent(backend, did="did:test:agent", name="Test Agent",
                dispatcher=None, registry=None):
    raw_storage = SimpleNamespace(db=backend)
    return SimpleNamespace(
        did=did,
        name=name,
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


class _TurnAgent(TurnLifecycleMixin):
    """Restart feature carrier with the production turn-session contract."""

    def __init__(self, backend):
        self.__dict__.update(vars(_make_agent(backend)))


class _AllowingHooksManager:
    """Minimal governed-dispatch hook surface for the inline-path regression."""

    def __init__(self):
        self.pre_calls: list[HookInput] = []

    async def execute_hooks(
        self, event: HookEvent, hook_input: HookInput,
    ) -> HookOutput:
        self.pre_calls.append(hook_input)
        return HookOutput.allow()

    async def execute_hooks_parallel(
        self, event: HookEvent, hook_input: HookInput,
    ) -> None:
        return None


class _InlineRestartAgent(TurnLifecycleMixin, OrchestratorEngineMixin):
    """Production lifecycle plus the real inline and named-tool dispatch path."""

    def __init__(self, backend):
        self.__dict__.update(vars(_make_agent(backend)))
        self.features = {}
        self.hooks_manager = _AllowingHooksManager()

    def _visible_features_by_tool_name(self):
        return {
            feature.tool_name: feature
            for feature in self.features.values()
            if getattr(feature, "enabled", True)
        }

    async def _get_denied_tools(self, _feature_name):
        return set()

    def _register_explored_feature_tools(self, _feature):
        return None

    async def _emit_tool_update_event(self, _session_id):
        return None


class _FrozenReaderHarness:
    """Model the app-server reader and its per-call handler task."""

    def __init__(self, agent):
        self._agent = agent
        self._queue: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self.handler_turn_ids: list[str | None] = []

    async def start(self) -> None:
        assert self._reader is None
        self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            executor, name, args, done = item
            asyncio.create_task(self._handle(executor, name, args, done))

    async def _handle(self, executor, name, args, done) -> None:
        self.handler_turn_ids.append(self._agent._get_current_turn_id())
        try:
            done.set_result(await executor(name, args))
        except Exception as exc:  # pragma: no cover - defensive harness path
            done.set_exception(exc)

    async def dispatch(self, executor, name, args):
        done = asyncio.get_running_loop().create_future()
        await self._queue.put((executor, name, args, done))
        return await done

    async def stop(self) -> None:
        if self._reader is not None:
            await self._queue.put(None)
            await self._reader
            self._reader = None


class _NestedRestartLLMService:
    """Drive a feature's own inline executor on a second frozen reader."""

    def __init__(self, reader):
        self._reader = reader
        self.effective_args = None
        self.result = None
        self.session_id = None

    async def generate(self, **kwargs):
        from kestrel_sdk.llm.adapter import LLMResponse

        self.session_id = kwargs.get("session_id")
        self.effective_args, self.result = await self._reader.dispatch(
            kwargs["tool_executor"],
            "request_restart",
            {"reason": "nested inline tool filed"},
        )
        return LLMResponse(content="Complete.", tool_calls=None)


class _EnqueueCognitionFeature(Feature):
    """Test tool that crosses the real inline-tool → signal-task boundary."""

    def __init__(self, agent, signal):
        super().__init__(agent)
        self.signal = signal
        self.dispatch_handle = None

    @property
    def tool_description(self) -> str:
        return "Enqueue a cognition signal"

    async def initialize(self) -> None:
        return None

    @tool(
        "enqueue_test_cognition",
        "Enqueue the configured cognition signal",
        category=ToolCategory.SYSTEM,
    )
    async def enqueue_test_cognition(self):
        self.dispatch_handle = await self.agent.dispatcher.enqueue_signal(
            self.signal
        )
        return {"queued": True}


async def _make_feature(tmp_path, **kwargs):
    backend = await _backend(tmp_path)
    agent = _make_agent(backend, **kwargs)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    return feat, backend


# ---------------------------------------------------------------------------
# Fleet idleness (#F235): a whole-host restart must defer while ANY co-hosted
# agent is busy, not just the requester.
# ---------------------------------------------------------------------------


def _idle_sibling(did: str, busy: bool = False):
    return SimpleNamespace(
        did=did,
        name=did,
        dispatcher=SimpleNamespace(),
        _active_request_ids={"r-active"} if busy else set(),
        _background_tasks=set(),
    )


@pytest.mark.asyncio
async def test_fleet_idle_defers_when_a_sibling_is_busy(tmp_path):
    """The requester is idle, but a co-hosted sibling is mid-turn. A whole-host
    restart would kill the sibling's turn, so idle_agents_only must defer."""
    feat, _ = await _make_feature(tmp_path)
    requester = feat.agent  # idle (empty active-request set)
    busy_sibling = _idle_sibling("did:test:sibling", busy=True)
    requester._cohosted_agents_provider = lambda: [requester, busy_sibling]

    state = feat._fleet_idle(ignore_request_id="")
    assert state["idle"] is False
    assert "did:test:sibling" in state["reason"]
    assert state["blocker"]["count"] is None
    assert state["blocker"]["oldest_age_seconds"] is None
    assert "1 active request" not in state["reason"]


@pytest.mark.asyncio
async def test_fleet_idle_true_when_all_agents_idle(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    requester = feat.agent
    idle_sibling = _idle_sibling("did:test:sibling", busy=False)
    requester._cohosted_agents_provider = lambda: [requester, idle_sibling]

    assert feat._fleet_idle(ignore_request_id="")["idle"] is True


@pytest.mark.asyncio
async def test_fleet_idle_falls_back_to_self_without_provider(tmp_path):
    """Single-agent host (no provider, no manager): behaviour-preserving — the
    requester-only check is used."""
    feat, _ = await _make_feature(tmp_path)
    assert not hasattr(feat.agent, "_cohosted_agents_provider")
    assert not hasattr(feat.agent, "_agent_manager")
    # Requester idle → idle; requester busy → not idle (self check).
    assert feat._fleet_idle(ignore_request_id="")["idle"] is True
    feat.agent._active_request_ids = {"r-active"}
    assert feat._fleet_idle(ignore_request_id="")["idle"] is False


@pytest.mark.asyncio
async def test_fleet_idle_resolves_via_manager_backref_when_no_provider(tmp_path):
    """SpawnFeature registers the parent in a lightweight AgentManager outside
    _load_one, so the parent lacks _cohosted_agents_provider but carries an
    _agent_manager backref. The fleet gate must still see a busy spawned child
    (codex round 2)."""
    feat, _ = await _make_feature(tmp_path)
    requester = feat.agent  # idle, no provider
    busy_child = _idle_sibling("did:test:child", busy=True)
    requester._agent_manager = SimpleNamespace(
        list_agents=lambda: {"parent": requester, "child": busy_child}
    )
    assert not hasattr(requester, "_cohosted_agents_provider")

    state = feat._fleet_idle(ignore_request_id="")
    assert state["idle"] is False
    assert "did:test:child" in state["reason"]


@pytest.mark.asyncio
async def test_fleet_idle_excludes_only_requesters_own_marker(tmp_path):
    """The requester's own turn (the one that filed the restart) is excluded on
    the REQUESTING agent — but the same id on a sibling still defers."""
    feat, _ = await _make_feature(tmp_path)
    requester = feat.agent
    requester._active_request_ids = {"req-turn"}
    sibling = _idle_sibling("did:test:sibling", busy=False)
    sibling._active_request_ids = {"req-turn"}  # same id, different agent
    requester._cohosted_agents_provider = lambda: [requester, sibling]

    state = feat._fleet_idle(ignore_request_id="req-turn")
    # Requester's own marker excluded, but the sibling's identical id still
    # counts (it is NOT the requester's turn).
    assert state["idle"] is False
    assert "did:test:sibling" in state["reason"]


@pytest.mark.asyncio
async def test_fleet_blocker_does_not_disclose_sibling_task_names(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    requester = feat.agent
    sibling = _idle_sibling("did:test:sibling", busy=False)
    blocked = asyncio.Event()

    async def private_counterparty_sync():
        await blocked.wait()

    task = asyncio.create_task(
        private_counterparty_sync(), name="private-counterparty-sync"
    )
    sibling._background_tasks = {task}
    requester._cohosted_agents_provider = lambda: [requester, sibling]
    try:
        state = feat._fleet_idle(ignore_request_id="")
        assert state["idle"] is False
        assert state["blocker"]["count"] is None
        assert state["blocker"]["summary"] is None
        assert state["blocker"]["oldest_age_seconds"] is None
        assert "private-counterparty-sync" not in state["reason"]
        assert "1 background task" not in state["reason"]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_fleet_blocker_does_not_disclose_sibling_dispatcher_load(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    requester = feat.agent
    sibling = _idle_sibling("did:test:sibling", busy=False)
    sibling.dispatcher = SimpleNamespace(in_flight_signals=7)
    requester._cohosted_agents_provider = lambda: [requester, sibling]

    state = feat._fleet_idle(ignore_request_id="")

    assert state["idle"] is False
    assert state["blocker"]["kind"] == "dispatcher"
    assert state["blocker"]["surface"] is None
    assert state["blocker"]["count"] is None
    assert "in_flight_signals" not in state["reason"]
    assert "7" not in state["reason"]


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
    assert req["first_blocked_at"] == ""
    assert req["escalation_acknowledged"] is True
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


# ---------------------------------------------------------------------------
# Self-documentation: advertised + normalized constrained enums (#1923/#1925)
# ---------------------------------------------------------------------------


def _tool_desc(method):
    """The @tool decorator stashes its description on the function."""
    return method._tool_schema["description"]


def test_request_restart_description_advertises_allowed_values():
    desc = _tool_desc(RestartCoordinatorFeature.request_restart)
    # urgency set + default + a synonym hint.
    for token in ("low", "normal", "high", "critical", "medium", "urgent"):
        assert token in desc
    # policy set + per-value meaning.
    for token in (
        "idle_agents_only",
        "allow_busy_after_timeout",
        "manual_only",
    ):
        assert token in desc
    # Returns shape so the agent can chain on data.request.id.
    assert "data.request.id" in desc
    assert "created" in desc


def test_list_restart_requests_description_includes_updating_and_returns():
    desc = _tool_desc(RestartCoordinatorFeature.list_restart_requests)
    for token in (
        "pending",
        "approved",
        "updating",
        "executing",
        "completed",
        "rejected",
        "canceled",
    ):
        assert token in desc
    assert "count" in desc and "requests" in desc


def test_cancel_restart_request_description_documents_returns():
    desc = _tool_desc(RestartCoordinatorFeature.cancel_restart_request)
    assert "canceled" in desc
    assert "request_id" in desc


@pytest.mark.asyncio
async def test_request_restart_normalizes_urgency_synonyms(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    # 'medium' is the universal taxonomy LLMs reach for → normal.
    result = await feat.request_restart(reason="r", urgency="medium")
    assert result.status is ToolResultStatus.OK
    assert result.data["request"]["urgency"] == "normal"
    # 'urgent' → high, 'emergency' → critical (case-insensitive).
    r2 = await feat.request_restart(reason="r", urgency="Urgent")
    assert r2.data["request"]["urgency"] == "high"
    r3 = await feat.request_restart(reason="r", urgency="EMERGENCY")
    assert r3.data["request"]["urgency"] == "critical"


@pytest.mark.asyncio
async def test_request_restart_normalizes_policy_synonyms(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    result = await feat.request_restart(reason="r", policy="manual")
    assert result.status is ToolResultStatus.OK
    assert result.data["request"]["policy"] == "manual_only"


@pytest.mark.asyncio
async def test_list_restart_requests_accepts_updating_status(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="r")
    req_id = created.data["request"]["id"]
    await update_status(
        backend, req_id, status="updating",
        expected_current_status="pending",
    )
    r = await feat.list_restart_requests(status="updating")
    assert r.status is ToolResultStatus.OK
    assert r.data["count"] == 1


@pytest.mark.asyncio
async def test_list_restart_requests_rejects_unknown_status(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    r = await feat.list_restart_requests(status="bogus")
    assert r.error is not None
    assert "status" in r.error
    # The error names the valid values so the agent can recover.
    for token in ("pending", "updating", "completed", "canceled"):
        assert token in r.error


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
@pytest.mark.parametrize("racing_status", ["canceled", "executing"])
async def test_deferral_status_race_cannot_dispatch_or_resurrect_request(
    tmp_path, racing_status,
):
    """A cancellation or competing claim after selection wins permanently."""

    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="race")
    request_id = created.data["request"]["id"]
    feat.agent._active_request_ids.add("unrelated-request")

    async def lose_race(db, selected_id, *, expected_current_status):
        assert selected_id == request_id
        assert expected_current_status == "pending"
        assert await update_status(
            db,
            selected_id,
            status=racing_status,
            status_reason="concurrent transition",
            expected_current_status="pending",
        )
        return await get_request(db, selected_id)

    with patch(
        "kestrel_sovereign.features.restart_coordinator.feature."
        "mark_deferral_started",
        side_effect=lose_race,
    ), patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    mock_spawn.assert_not_called()
    assert result.data["executed"] == []
    assert result.data["deferred"][0]["request_id"] == request_id
    assert "lost race" in result.data["deferred"][0]["reason"]
    row = await get_request(backend, request_id)
    assert row.status == racing_status
    assert row.first_blocked_at == ""


@pytest.mark.asyncio
async def test_escalation_event_is_not_emitted_before_lifecycle_cas(tmp_path):
    """A cancellation winning the dispatch CAS cannot leave a false event."""

    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="escalation CAS race")
    request_id = created.data["request"]["id"]
    feat.agent._active_request_ids.add("other-active-request")
    blocked_at = (
        datetime.now(timezone.utc)
        - timedelta(seconds=MAX_IDLE_ONLY_DEFERRAL_SECONDS + 1)
    ).isoformat()
    await backend.execute(
        "UPDATE restart_requests SET first_blocked_at = ? WHERE id = ?",
        (blocked_at, request_id),
    )
    captured = _attach_emit_capture(feat)

    async def lose_transition(*args, **kwargs):
        return False

    with patch(
        "kestrel_sovereign.features.restart_coordinator.feature.update_status",
        side_effect=lose_transition,
    ), patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as spawn:
        result = await feat.restart_coordinator()

    spawn.assert_not_called()
    assert result.data["executed"] == []
    assert not any(
        event["status"] == "escalated"
        for event in _restart_status_events(captured)
    )


@pytest.mark.asyncio
async def test_deferral_clear_cannot_reset_competing_execution_interval(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="clear race")
    request_id = created.data["request"]["id"]
    blocked_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    await backend.execute(
        "UPDATE restart_requests SET first_blocked_at = ? WHERE id = ?",
        (blocked_at, request_id),
    )
    assert await update_status(
        backend,
        request_id,
        status="executing",
        expected_current_status="pending",
    )

    cleared = await clear_deferral_started(
        backend,
        request_id,
        expected_current_status="pending",
    )

    assert cleared is False
    row = await get_request(backend, request_id)
    assert row.status == "executing"
    assert row.first_blocked_at == blocked_at


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
    events = await feat.list_restart_status_events()
    request_events = [
        event for event in events.data["events"]
        if event["request_id"] == req.id
    ]
    assert all(event["state"] != "escalated" for event in request_events)


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
    """If the dispatcher is present but ``enqueue_signal`` raises, the restart
    is still terminalized (it provably happened) but the wake stays UNDELIVERED
    (wake_delivered=0) so a future sweep retries it (#1819; pre-#1819 the row
    stayed executing).
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
    await feat.on_agent_ready()  # terminalizes, then the broken dispatcher fails the wake

    row = await get_request(backend, req.id)
    assert row.status == "completed"        # restart finished
    assert row.wake_delivered is False      # wake undelivered → retried later


@pytest.mark.asyncio
async def test_wake_not_delivered_when_terminalize_write_does_not_land(tmp_path):
    """A completion wake must NEVER be delivered against a row still
    ``executing`` in the DB (#1801).

    The #1819 sweep terminalizes the row to ``completed`` BEFORE dispatching
    the wake, but the ``expected_current_status="executing"`` guard can fail
    to land that write (a concurrent ``on_agent_ready`` / cron-tick sweep
    already terminalized it). When the durable write does not land and the
    row is not genuinely ``completed``, the sweep must NOT fire a
    ``restart.completed`` wake — otherwise chat sees a completion while
    ``list_restart_requests`` still reports ``executing`` with no
    ``completed`` status event (the exact inconsistency in #1801).
    """
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

    # Model the lost race: the durable executing → completed write does not
    # land and leaves the row untouched (still executing).
    async def _write_does_not_land(*args, **kwargs):
        return False

    with patch(
        "kestrel_sovereign.features.restart_coordinator.feature.update_status",
        _write_does_not_land,
    ):
        await feat.on_agent_ready()

    # No phantom completion wake while the durable row stays executing.
    assert dispatcher.signals == []
    row = await get_request(backend, req.id)
    assert row.status == "executing"
    assert row.completed_at is None


# ---------------------------------------------------------------------------
# Post-restart wake reaches the REAL durable/resumption path (#1796)
# ---------------------------------------------------------------------------


class _RealDispatchAgent(TurnLifecycleMixin, OrchestratorEngineMixin):
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
        # Optional one-shot continuation used by the #2928 regression: a
        # restart.completed cognition turn can file the next restart request.
        self.restart_request_reason = None
        self.restart_feature = None
        self.hooks_manager = _AllowingHooksManager()

    async def process_input(self, prompt: str, session_id=None):
        async with self._turn_lifecycle():
            self._active_session_id = session_id
            if self.process_input_should_raise:
                raise RuntimeError("simulated cognition failure")
            self.process_input_calls.append(prompt)
            self.process_input_sessions.append(session_id)
            reason = self.restart_request_reason
            self.restart_request_reason = None
            if reason is not None:
                assert self.restart_feature is not None
                await self.restart_feature.request_restart(reason=reason)
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

    async def shutdown(self):
        """Mirror the production ordering: tasks first, storage second."""
        tasks = set(self.background_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._raw_storage.db.close()


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
    locks = OrderedLockManager()
    agent._lock_manager = locks
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=locks,
        store=store,
    )
    agent.signal_registry = registry
    agent.dispatcher = dispatcher

    feat = RestartCoordinatorFeature(agent)
    agent.restart_feature = feat
    _real_dispatch_lifecycles.append((feat, agent))
    return feat, backend, agent


@pytest.mark.asyncio
async def test_post_restart_wake_reaches_process_input_and_then_completes(
    tmp_path,
):
    """The swept ``restart.completed`` wake must reach the agent's
    ``process_input`` (the real resumption path). The row is terminalized to
    ``completed`` up front (#1819), and ``wake_delivered`` flips only once that
    dispatch lands Status.OK — not merely because ``enqueue_signal`` returned.
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
    wake_tasks = await feat.on_agent_ready()
    # The restart is terminalized immediately (it provably happened); the wake
    # runs as a supervised background task, so wake_delivered is still 0.
    row_mid = await get_request(backend, req.id)
    assert row_mid.status == "completed"
    assert row_mid.wake_delivered is False

    assert len(wake_tasks) == 1
    await asyncio.gather(*wake_tasks)

    # The wake reached the agent's resuming turn...
    assert len(agent.process_input_calls) == 1
    assert req.id in agent.process_input_calls[0]
    # ...and ONLY THEN was the wake flagged delivered.
    row = await get_request(backend, req.id)
    assert row.status == "completed"
    assert row.wake_delivered is True


@pytest.mark.asyncio
async def test_post_restart_wake_failure_leaves_row_retryable(tmp_path):
    """If the wake's resuming turn fails (dispatch returns FAILED), the restart
    is still terminalized (it provably happened) but ``wake_delivered`` stays 0
    so a later sweep retries the wake (#1819; pre-#1819 the row stayed
    executing).
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
    wake_tasks = await feat.on_agent_ready()
    assert len(wake_tasks) == 1
    await asyncio.gather(*wake_tasks)

    row = await get_request(backend, req.id)
    assert row.status == "completed"        # restart finished
    assert row.wake_delivered is False      # wake failed → retried later


@pytest.mark.asyncio
async def test_real_dispatch_shutdown_closes_database_worker_before_loop_teardown(
    tmp_path,
):
    """Wake tasks finish before the agent releases its SQLite connection."""
    feat, backend, _agent = await _real_dispatch_feature(tmp_path)
    connection = backend._backend._connection
    assert connection is not None and _sqlite_worker_is_alive(connection)

    await feat.initialize()
    await feat.shutdown()
    await _agent.shutdown()

    assert backend._backend.is_connected is False
    assert not _sqlite_worker_is_alive(connection)


@pytest.mark.asyncio
async def test_restart_coordinator_cron_retries_undelivered_wake(tmp_path):
    """The ``restart_coordinator`` cron tick is the retry backstop: a
    ``completed`` row whose wake failed (wake_delivered=0) on the ready sweep
    must be re-woken on a later tick (without waiting for a full reboot), and
    once the resuming turn succeeds wake_delivered flips to 1 (#1796/#1819).
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

    # Ready sweep: restart terminalizes, but the wake fails → undelivered.
    await feat.initialize()
    await feat.on_agent_ready()
    await agent.drain_background_tasks()
    row_after_first = await get_request(backend, req.id)
    assert row_after_first.status == "completed"
    assert row_after_first.wake_delivered is False
    assert agent.process_input_calls == []

    # The resuming turn now succeeds; a cron tick must re-wake and deliver.
    # Re-anchor the dispatcher's coalescing window to model the >30s gap
    # between the ready sweep and a production 1/min cron tick (a fast
    # in-test retry would otherwise coalesce against the failed wake).
    agent.dispatcher.notify_resume(60.0)
    agent.process_input_should_raise = False
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ):
        await feat.restart_coordinator()
    await agent.drain_background_tasks()

    assert len(agent.process_input_calls) == 1
    row_final = await get_request(backend, req.id)
    assert row_final.status == "completed"
    assert row_final.wake_delivered is True


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
    assert ev["requested_by_agent_name"] == "Test Agent"
    assert ev["reason"] == "kestrel.toml change landed"
    assert ev["deferral_reason"] == ""


@pytest.mark.asyncio
async def test_status_event_resolves_cohosted_requester_name(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    captured = _attach_emit_capture(feat)
    feat.agent._cohosted_agents_provider = lambda: [
        feat.agent,
        SimpleNamespace(did="did:test:emma", name="Emma"),
    ]
    req = await insert_request(
        backend, requested_by_agent="did:test:emma", reason="from sibling",
    )
    await feat._emit_status_event(req, state="pending")

    events = _restart_status_events(captured)
    assert len(events) == 1
    assert events[0]["requested_by_agent"] == "did:test:emma"
    assert events[0]["requested_by_agent_name"] == "Emma"


@pytest.mark.asyncio
async def test_status_event_resolves_real_agent_private_name(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    captured = _attach_emit_capture(feat)
    delattr(feat.agent, "name")
    feat.agent._agent_name = "Production Agent"
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="self request",
    )
    await feat._emit_status_event(req, state="pending")

    events = _restart_status_events(captured)
    assert len(events) == 1
    assert events[0]["requested_by_agent"] == "did:test:agent"
    assert events[0]["requested_by_agent_name"] == "Production Agent"


@pytest.mark.asyncio
async def test_status_event_resolves_requester_name_from_agent_manager(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    captured = _attach_emit_capture(feat)
    feat.agent._agent_manager = SimpleNamespace(
        get_agent_name=lambda did: {
            "did:test:emma": "Emma",
        }.get(did)
    )
    req = await insert_request(
        backend, requested_by_agent="did:test:emma", reason="manager lookup",
    )
    await feat._emit_status_event(req, state="pending")

    events = _restart_status_events(captured)
    assert len(events) == 1
    assert events[0]["requested_by_agent"] == "did:test:emma"
    assert events[0]["requested_by_agent_name"] == "Emma"


@pytest.mark.asyncio
async def test_status_event_leaves_unknown_requester_name_empty(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    captured = _attach_emit_capture(feat)
    feat.agent._cohosted_agents_provider = lambda: [feat.agent]
    req = await insert_request(
        backend, requested_by_agent="did:test:remote", reason="remote",
    )
    await feat._emit_status_event(req, state="pending")

    events = _restart_status_events(captured)
    assert len(events) == 1
    assert events[0]["requested_by_agent"] == "did:test:remote"
    assert events[0]["requested_by_agent_name"] == ""


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

    log_task = asyncio.create_task(_never(), name="durable_signal_log:heartbeat:abc123")
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
        # The reason names the ONE genuine blocker (#2665) and no infra task,
        # so an operator can tell what is actually holding the restart.
        # Raw asyncio tasks carry no age stamp — only _track_background_task
        # stamps them — so the age reads as unknown rather than fabricated.
        assert busy["reason"] == (
            "1 background task(s) in flight: "
            "signal_dispatch:heartbeat:abc123 (age unknown)"
        )
        assert "durable_signal_log" not in busy["reason"]
        assert "a2a_question_expiry_sweep" not in busy["reason"]
    finally:
        log_task.cancel()
        sweep_task.cancel()
        work_task.cancel()


@pytest.mark.asyncio
async def test_idle_ignores_a2a_question_supervisor_tasks(tmp_path):
    """#2666: the sender-side ``a2a_question_supervisor:*`` tasks are passive,
    deadline-bounded waits for a peer's answer whose correlation lives durably
    in ``pending_a2a_questions`` (the startup replay re-arms them). Counting
    them as 'busy' pinned a phantom "N background tasks in flight" that
    survived restarts while ``list_my_tasks`` read empty, wedging
    idle_agents_only forever when a peer/model outage left questions
    unanswered. They must be excluded from the idle check — including the
    ``:replay:`` startup-replay variant — while real work still defers."""
    feat, _ = await _make_feature(tmp_path)

    async def _never():
        await asyncio.Event().wait()

    sup_task = asyncio.create_task(
        _never(), name="a2a_question_supervisor:Claw:task-abc",
    )
    replay_task = asyncio.create_task(
        _never(), name="a2a_question_supervisor:replay:Claw:task-def",
    )
    work_task = asyncio.create_task(
        _never(), name="signal_dispatch:heartbeat:abc123",
    )
    try:
        # Two unanswered questions in flight (the exact live repro: count
        # pinned at 2) -> the agent is still idle; neither supervisor may
        # wedge an idle restart.
        feat.agent._background_tasks = {sup_task, replay_task}
        idle = feat._agent_appears_idle()
        assert idle["idle"] is True, idle

        # A real signal_dispatch task still defers, and the supervisors don't
        # inflate the reported count past the one genuine blocker.
        feat.agent._background_tasks = {sup_task, replay_task, work_task}
        busy = feat._agent_appears_idle()
        assert busy["idle"] is False
        # Naming the blocker is what makes a phantom entry diagnosable: the
        # live report of "2 background tasks" against an empty task store was
        # unfalsifiable while the reason was a bare count (#2665).
        # Raw asyncio tasks carry no age stamp — only _track_background_task
        # stamps them — so the age reads as unknown rather than fabricated.
        assert busy["reason"] == (
            "1 background task(s) in flight: "
            "signal_dispatch:heartbeat:abc123 (age unknown)"
        )
        assert "a2a_question_supervisor" not in busy["reason"]
    finally:
        sup_task.cancel()
        replay_task.cancel()
        work_task.cancel()


@pytest.mark.asyncio
async def test_idle_ignores_isolated_runtime_lifecycle_daemons(tmp_path):
    feat, _ = await _make_feature(tmp_path)

    async def _never():
        await asyncio.Event().wait()

    supervisor = asyncio.create_task(_never(), name="isolated-feature:Search")
    idle_monitor = asyncio.create_task(
        _never(), name="isolated-feature-idle:Search"
    )
    telemetry = asyncio.create_task(
        _never(), name="isolated-runtime-telemetry:Search"
    )
    work_task = asyncio.create_task(_never(), name="isolated-call:Search")
    try:
        feat.agent._background_tasks = {supervisor, idle_monitor, telemetry}
        assert feat._agent_appears_idle()["idle"] is True

        feat.agent._background_tasks = {
            supervisor,
            idle_monitor,
            telemetry,
            work_task,
        }
        busy = feat._agent_appears_idle()
        assert busy["idle"] is False
        assert "isolated-call:Search" in busy["reason"]
        assert "isolated-feature" not in busy["reason"]
    finally:
        supervisor.cancel()
        idle_monitor.cancel()
        telemetry.cancel()
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
async def test_request_restart_captures_turn_session_through_lifecycle(tmp_path):
    """A normal chat turn captures its effective session at the producer."""
    backend = await _backend(tmp_path)
    agent = _TurnAgent(backend)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    async with agent._turn_lifecycle():
        agent._active_session_id = "chat-session-42"
        result = await feat.request_restart(reason="ship it")

    req_id = result.data["request"]["id"]
    row = await get_request(backend, req_id)
    assert row.origin_session_id == "chat-session-42"


@pytest.mark.asyncio
async def test_inline_enqueued_wake_owns_session_for_followup_restart(tmp_path):
    """An inline tool's child signal task owns the cognition turn it opens."""
    from kestrel_sovereign.signals.sources.restart import (
        build_signal_for_restart_completed,
    )

    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()
    wake_source = await insert_request(
        backend,
        requested_by_agent=agent.did,
        reason="wake source",
        origin_session_id="wake-session-B",
    )
    signal = build_signal_for_restart_completed(
        wake_source,
        target_agent=agent.did,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    enqueuer = _EnqueueCognitionFeature(agent, signal)
    agent.features = {feat.name: feat, enqueuer.name: enqueuer}
    agent.restart_request_reason = "restart filed from wake turn"
    reader = _FrozenReaderHarness(agent)

    # The app-server reader predates chat turn A. Its inline callback binds A,
    # and the tool creates SignalDispatcher's background task while that
    # binding is active. The task waits for A's conversation lock, then opens
    # wake turn B and must use B's routed session as its own authority.
    await reader.start()
    try:
        async with agent._turn_lifecycle():
            agent._active_session_id = "chat-session-A"
            executor = agent._make_inline_tool_executor("chat-session-A")
            await reader.dispatch(executor, "enqueue_test_cognition", {})
    finally:
        await reader.stop()

    await agent.drain_background_tasks()

    pending = await list_requests(backend, status="pending")
    followup = next(
        row for row in pending if row.reason == "restart filed from wake turn"
    )
    assert reader.handler_turn_ids == [None]
    assert agent.process_input_sessions == ["wake-session-B"]
    assert followup.origin_session_id == "wake-session-B"


@pytest.mark.asyncio
async def test_inline_restart_preserves_session_across_frozen_reader_task(tmp_path):
    """The live Codex path carries its owning turn into the pre-turn reader."""
    backend = await _backend(tmp_path)
    agent = _InlineRestartAgent(backend)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    agent.features = {feat.name: feat}
    reader = _FrozenReaderHarness(agent)

    # Production starts one long-lived reader before later turns. Its child
    # handlers therefore inherit this pre-turn context, not the active turn's.
    await reader.start()
    try:
        async with agent._turn_lifecycle():
            agent._active_session_id = "chat-inline-42"
            executor = agent._make_inline_tool_executor("transport-session-99")
            effective_args, result = await reader.dispatch(
                executor,
                "request_restart",
                {"reason": "inline tool filed"},
            )
    finally:
        await reader.stop()

    assert reader.handler_turn_ids == [None]
    assert effective_args == {"reason": "inline tool filed"}
    row = await get_request(backend, result["data"]["request"]["id"])
    assert row.origin_session_id == "chat-inline-42"
    assert agent.hooks_manager.pre_calls[0].tool_name == "request_restart"


@pytest.mark.asyncio
async def test_inline_restart_without_turn_stays_unbound_across_reader_task(
    tmp_path,
):
    """An origin-less inline invocation cannot borrow ambient session hints."""
    from kestrel_sovereign.logging_config import session_id_var

    backend = await _backend(tmp_path)
    agent = _InlineRestartAgent(backend)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    agent.features = {feat.name: feat}
    agent._active_session_id = "unrelated-agent-global"
    reader = _FrozenReaderHarness(agent)
    executor = agent._make_inline_tool_executor("unowned-transport-session")

    token = session_id_var.set("unrelated-logging-session")
    await reader.start()
    try:
        _effective_args, result = await reader.dispatch(
            executor,
            "request_restart",
            {"reason": "origin-less inline tool"},
        )
    finally:
        await reader.stop()
        session_id_var.reset(token)

    assert reader.handler_turn_ids == [None]
    row = await get_request(backend, result["data"]["request"]["id"])
    assert row.origin_session_id == ""


@pytest.mark.asyncio
async def test_off_turn_inline_restart_does_not_borrow_readers_live_turn(
    tmp_path,
):
    """An off-turn executor stays unbound on a reader born in a live chat."""
    backend = await _backend(tmp_path)
    agent = _InlineRestartAgent(backend)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    agent.features = {feat.name: feat}
    reader = _FrozenReaderHarness(agent)
    executor = agent._make_inline_tool_executor("unowned-transport-session")

    async with agent._turn_lifecycle() as turn_id:
        agent._active_session_id = "unrelated-live-chat"
        await reader.start()
        try:
            _effective_args, result = await reader.dispatch(
                executor,
                "request_restart",
                {"reason": "origin-less inline tool"},
            )
        finally:
            await reader.stop()

    assert reader.handler_turn_ids == [turn_id]
    row = await get_request(backend, result["data"]["request"]["id"])
    assert row.origin_session_id == ""


@pytest.mark.asyncio
async def test_nested_inline_restart_preserves_session_across_two_readers(
    tmp_path,
):
    """The feature-subagent executor carries the turn across its own reader."""
    backend = await _backend(tmp_path)
    agent = _InlineRestartAgent(backend)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    agent.features = {feat.name: feat}
    parent_reader = _FrozenReaderHarness(agent)
    feature_reader = _FrozenReaderHarness(agent)
    llm_service = _NestedRestartLLMService(feature_reader)
    agent.llm_service = llm_service

    # Both long-lived readers predate the turn. The parent executor restores
    # the binding once, then execute_as_subagent builds another executor that
    # must explicitly carry it through the feature reader's separate task.
    await parent_reader.start()
    await feature_reader.start()
    try:
        async with agent._turn_lifecycle():
            agent._active_session_id = "chat-nested-42"
            parent_executor = agent._make_inline_tool_executor(
                "parent-transport-session"
            )
            effective_args, result = await parent_reader.dispatch(
                parent_executor,
                feat.tool_name,
                {"task": "file a restart request"},
            )
    finally:
        await parent_reader.stop()
        await feature_reader.stop()

    assert parent_reader.handler_turn_ids == [None]
    assert feature_reader.handler_turn_ids == [None]
    assert effective_args == {"task": "file a restart request"}
    assert result["success"] is True
    assert llm_service.session_id == "chat-nested-42"
    assert llm_service.effective_args == {
        "reason": "nested inline tool filed"
    }
    row = await get_request(
        backend, llm_service.result["data"]["request"]["id"]
    )
    assert row.origin_session_id == "chat-nested-42"


@pytest.mark.asyncio
async def test_nested_inline_restart_built_off_turn_stays_unbound(tmp_path):
    """Two inline boundaries cannot turn ambient session hints into authority."""
    from kestrel_sovereign.logging_config import session_id_var

    backend = await _backend(tmp_path)
    agent = _InlineRestartAgent(backend)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    agent.features = {feat.name: feat}
    agent._active_session_id = "unrelated-agent-global"
    parent_reader = _FrozenReaderHarness(agent)
    feature_reader = _FrozenReaderHarness(agent)
    llm_service = _NestedRestartLLMService(feature_reader)
    agent.llm_service = llm_service
    parent_executor = agent._make_inline_tool_executor(
        "unowned-parent-transport-session"
    )

    token = session_id_var.set("unrelated-logging-session")
    await parent_reader.start()
    await feature_reader.start()
    try:
        _effective_args, result = await parent_reader.dispatch(
            parent_executor,
            feat.tool_name,
            {"task": "file a system restart request"},
        )
    finally:
        await parent_reader.stop()
        await feature_reader.stop()
        session_id_var.reset(token)

    assert parent_reader.handler_turn_ids == [None]
    assert feature_reader.handler_turn_ids == [None]
    assert result["success"] is True
    assert llm_service.session_id is None
    row = await get_request(
        backend, llm_service.result["data"]["request"]["id"]
    )
    assert row.origin_session_id == ""


@pytest.mark.asyncio
async def test_off_turn_nested_restart_does_not_borrow_readers_live_turn(
    tmp_path,
):
    """Explicit unbound authority propagates through the nested executor."""
    backend = await _backend(tmp_path)
    agent = _InlineRestartAgent(backend)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    agent.features = {feat.name: feat}
    parent_reader = _FrozenReaderHarness(agent)
    feature_reader = _FrozenReaderHarness(agent)
    llm_service = _NestedRestartLLMService(feature_reader)
    agent.llm_service = llm_service
    parent_executor = agent._make_inline_tool_executor(
        "unowned-parent-transport-session"
    )

    async with agent._turn_lifecycle() as turn_id:
        agent._active_session_id = "unrelated-live-chat"
        await parent_reader.start()
        await feature_reader.start()
        try:
            _effective_args, result = await parent_reader.dispatch(
                parent_executor,
                feat.tool_name,
                {"task": "file a system restart request"},
            )
        finally:
            await parent_reader.stop()
            await feature_reader.stop()

    assert parent_reader.handler_turn_ids == [turn_id]
    assert feature_reader.handler_turn_ids == [turn_id]
    assert result["success"] is True
    assert llm_service.session_id is None
    row = await get_request(
        backend, llm_service.result["data"]["request"]["id"]
    )
    assert row.origin_session_id == ""


@pytest.mark.asyncio
async def test_request_restart_ignores_agent_global_session_outside_a_turn(
    tmp_path,
):
    """An agent-global session is not routing authority outside its turn."""
    backend = await _backend(tmp_path)
    agent = _TurnAgent(backend)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    agent._active_session_id = "concurrent-chat-3"

    result = await feat.request_restart(reason="cron-filed")

    row = await get_request(backend, result.data["request"]["id"])
    assert row.origin_session_id == ""


@pytest.mark.asyncio
async def test_request_restart_does_not_infer_origin_from_logging_context(
    tmp_path,
):
    """A logging header outside a live turn is not routing authority."""
    from kestrel_sovereign.logging_config import session_id_var

    feat, backend = await _make_feature(tmp_path)
    token = session_id_var.set("header-session-1")
    try:
        result = await feat.request_restart(reason="ship it")
    finally:
        session_id_var.reset(token)

    row = await get_request(backend, result.data["request"]["id"])
    assert row.origin_session_id == ""


@pytest.mark.asyncio
async def test_request_restart_no_session_is_blank(tmp_path):
    """With no chat session in context (CLI/system), origin_session_id is blank."""
    feat, backend = await _make_feature(tmp_path)
    result = await feat.request_restart(reason="system filed")
    row = await get_request(backend, result.data["request"]["id"])
    assert row.origin_session_id == ""


@pytest.mark.asyncio
async def test_origin_session_survives_lifecycle_events_and_completion_wake(
    tmp_path,
):
    """The producer binding is immutable from pending through the wake."""
    backend = await _backend(tmp_path)
    agent = _TurnAgent(backend)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    async with agent._turn_lifecycle():
        agent._active_session_id = "chat-lifecycle-7"
        result = await feat.request_restart(reason="ship it")

    request_id = result.data["request"]["id"]
    row = await get_request(backend, request_id)
    assert await update_status(
        backend,
        request_id,
        status="executing",
        expected_current_status="pending",
    )
    row.status = "executing"
    await feat._emit_status_event(row, state="executing")
    assert await feat._terminalize_completed(
        row, datetime.now(timezone.utc).isoformat()
    )

    stored = await get_request(backend, request_id)
    assert stored.origin_session_id == "chat-lifecycle-7"
    events = await list_events_for_request(backend, request_id)
    assert [event.state for event in events] == [
        "pending",
        "executing",
        "completed",
    ]
    assert {
        event.to_public_dict()["payload"]["origin_session_id"]
        for event in events
    } == {"chat-lifecycle-7"}

    from kestrel_sovereign.signals.sources.restart import (
        build_signal_for_restart_completed,
    )

    wake = build_signal_for_restart_completed(
        stored,
        target_agent=agent.did,
        completed_at=stored.completed_at,
    )
    assert wake.session_id == "chat-lifecycle-7"


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
@pytest.mark.parametrize(
    ("origin_session_id", "expected_turn_session"),
    [("chat-chain-9", "chat-chain-9"), ("", None)],
)
async def test_restart_wake_followup_preserves_explicit_origin_binding(
    tmp_path, origin_session_id, expected_turn_session,
):
    """A wake chain preserves a bound origin and leaves no-origin unbound."""
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    req = await insert_request(
        backend,
        requested_by_agent="did:test:agent",
        reason="first restart",
        origin_session_id=origin_session_id,
    )
    await update_status(
        backend,
        req.id,
        status="executing",
        expected_current_status="pending",
    )
    agent.restart_request_reason = "follow-up restart"

    await feat.initialize()
    await feat.on_agent_ready()
    await agent.drain_background_tasks()

    pending = await list_requests(backend, status="pending")
    followup = next(row for row in pending if row.reason == "follow-up restart")
    assert agent.process_input_sessions == [expected_turn_session]
    assert followup.origin_session_id == origin_session_id


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


@pytest.mark.asyncio
async def test_migration_backfills_wake_delivered_for_old_completed_rows(tmp_path):
    """#1819 migration safety: a pre-existing 'completed' row (terminalized
    under the old delivery-gated scheme) must be backfilled wake_delivered=1 so
    the new sweep doesn't re-wake all restart history on first boot."""
    raw = SQLiteBackend(str(tmp_path / "old.db"))
    await raw.connect()
    db = _track_test_database(AsyncDatabase(raw))
    # Pre-#1819 schema: every column EXCEPT wake_delivered.
    await db.execute(
        """
        CREATE TABLE restart_requests (
            id TEXT PRIMARY KEY, requested_by_agent TEXT NOT NULL,
            reason TEXT NOT NULL, requested_at TEXT NOT NULL,
            desired_window TEXT DEFAULT '', urgency TEXT DEFAULT 'normal',
            policy TEXT DEFAULT 'idle_agents_only', status TEXT DEFAULT 'pending',
            status_reason TEXT DEFAULT '', completed_at TEXT,
            operation TEXT DEFAULT 'restart_only', update_repo_path TEXT DEFAULT '',
            update_target_ref TEXT DEFAULT '', update_profile TEXT DEFAULT '',
            update_allow_migrations INTEGER DEFAULT 0, update_log TEXT DEFAULT '',
            requester_request_id TEXT DEFAULT '', executing_boot_id TEXT DEFAULT '',
            origin_session_id TEXT DEFAULT ''
        )
        """
    )
    await db.execute(
        "INSERT INTO restart_requests "
        "(id, requested_by_agent, reason, requested_at, status, completed_at) "
        "VALUES ('old-done','did:a','r','t','completed','t')"
    )
    await db.execute(
        "INSERT INTO restart_requests "
        "(id, requested_by_agent, reason, requested_at, status) "
        "VALUES ('old-exec','did:a','r','t','executing')"
    )

    # Migrate (adds wake_delivered + one-time backfill).
    await ensure_restart_requests_table(db)

    done = await get_request(db, "old-done")
    assert done.wake_delivered is True  # old completed row → won't be re-woken
    execu = await get_request(db, "old-exec")
    assert execu.wake_delivered is False  # still-executing row is untouched

    # Idempotent: a second ensure (column already exists) must NOT reset a
    # legitimately-undelivered new-flow wake (completed + wake_delivered=0).
    await db.execute(
        "INSERT INTO restart_requests "
        "(id, requested_by_agent, reason, requested_at, status, wake_delivered) "
        "VALUES ('new-undelivered','did:a','r','t','completed',0)"
    )
    await ensure_restart_requests_table(db)
    nu = await get_request(db, "new-undelivered")
    assert nu.wake_delivered is False  # not clobbered by a re-run backfill


# ---------------------------------------------------------------------------
# Best-effort update steps: the reattach_branch step legitimately fails for
# tag/sha targets (refs/remotes/origin/<ref> doesn't resolve), so a failing
# allow_failure step must not abort the update — while mutating steps stay
# fatal-on-failure.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_update_continues_past_failing_allow_failure_step(tmp_path):
    import sys

    from kestrel_sovereign.features.restart_coordinator.update_profiles import (
        UpdateProfile,
        UpdateStep,
    )

    feat, _db = await _make_feature(tmp_path)

    def _build(repo_path, target_ref, allow_migrations):
        return [
            UpdateStep(
                "reattach_branch",
                [sys.executable, "-c", "import sys; sys.exit(1)"],
                allow_failure=True,
            ),
            UpdateStep(
                "resolve_ref",
                [sys.executable, "-c", "print('deadbeef')"],
                read_only=True,
            ),
        ]

    profile = UpdateProfile(
        name="probe", description="", supports_migrations=False, _build=_build,
    )
    req = SimpleNamespace(
        id="req-probe",
        update_repo_path=str(tmp_path),
        update_target_ref="v0.31.1",
        update_allow_migrations=False,
    )

    update = await feat._run_update(req, profile)
    assert update["ok"] is True
    assert update["failed_step"] is None
    # Later steps still ran after the best-effort failure.
    assert update["resolved_ref"] == "deadbeef"
    reattach = next(s for s in update["steps"] if s["step"] == "reattach_branch")
    assert reattach["ok"] is False


@pytest.mark.asyncio
async def test_run_update_still_fails_on_mutating_step(tmp_path):
    import sys

    from kestrel_sovereign.features.restart_coordinator.update_profiles import (
        UpdateProfile,
        UpdateStep,
    )

    feat, _db = await _make_feature(tmp_path)

    def _build(repo_path, target_ref, allow_migrations):
        return [
            UpdateStep(
                "install",
                [sys.executable, "-c", "import sys; sys.exit(1)"],
            ),
            UpdateStep(
                "resolve_ref",
                [sys.executable, "-c", "print('deadbeef')"],
                read_only=True,
            ),
        ]

    profile = UpdateProfile(
        name="probe", description="", supports_migrations=False, _build=_build,
    )
    req = SimpleNamespace(
        id="req-probe",
        update_repo_path=str(tmp_path),
        update_target_ref="main",
        update_allow_migrations=False,
    )

    update = await feat._run_update(req, profile)
    assert update["ok"] is False
    assert update["failed_step"] == "install"
    # Stopped at the failure: resolve_ref never ran.
    assert update["resolved_ref"] == ""


# ---------------------------------------------------------------------------
# Native reattach_branch routine: attach for branch targets, stay detached
# for tags, and never follow a same-named branch when the fetch landed on a
# tag (codex P2 on the reattach change — a name can be both).
# ---------------------------------------------------------------------------


def _real_origin_and_clone(tmp_path):
    """A real origin with branch `main`, plus a local clone."""
    origin = tmp_path / "origin"
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(origin)], check=True
    )
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(origin), "config", k, v], check=True)
    (origin / "f.txt").write_text("one\n")
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(origin), "commit", "-q", "-m", "one"], check=True
    )
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone)], check=True
    )
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(clone), "config", k, v], check=True)
    return origin, clone


def _git_out(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


async def _run_git_steps(feat, repo, target_ref):
    """Run the profile's git steps (fetch/checkout/reattach) for real,
    skipping install/feature_sync which need a real project."""
    profile = get_update_profile("sovereign_local_uv_sync")
    steps = profile.build_steps(
        repo_path=str(repo), target_ref=target_ref, allow_migrations=False
    )
    outcomes = {}
    for step in steps:
        if step.name in ("install", "feature_sync"):
            continue
        outcomes[step.name] = await feat._run_update_step(step)
    return outcomes


@pytest.mark.asyncio
async def test_reattach_attaches_branch_target_on_fetched_commit(tmp_path):
    origin, clone = _real_origin_and_clone(tmp_path / "repos")
    # Advance origin/main past the clone.
    (origin / "f.txt").write_text("two\n")
    subprocess.run(["git", "-C", str(origin), "commit", "-qam", "two"], check=True)
    origin_tip = _git_out(origin, "rev-parse", "HEAD")

    feat, _db = await _make_feature(tmp_path)
    outcomes = await _run_git_steps(feat, clone, "main")

    assert outcomes["reattach_branch"]["ok"] is True
    # Attached to main, exactly at the fetched origin tip.
    assert _git_out(clone, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert _git_out(clone, "rev-parse", "HEAD") == origin_tip


@pytest.mark.asyncio
async def test_reattach_stays_detached_for_tag_target(tmp_path):
    origin, clone = _real_origin_and_clone(tmp_path / "repos")
    subprocess.run(["git", "-C", str(origin), "tag", "v9.9.9"], check=True)
    tag_sha = _git_out(origin, "rev-parse", "v9.9.9^{commit}")

    feat, _db = await _make_feature(tmp_path)
    outcomes = await _run_git_steps(feat, clone, "v9.9.9")

    out = outcomes["reattach_branch"]
    assert out["ok"] is True
    assert "skip" in out["stdout_tail"]
    assert _git_out(clone, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    assert _git_out(clone, "rev-parse", "HEAD") == tag_sha


@pytest.mark.asyncio
async def test_reattach_ignores_branch_shadowing_a_tag(tmp_path):
    """Name exists as BOTH tag and branch: the fetch lands on the TAG
    commit, so the reattach must NOT move to the same-named branch."""
    origin, clone = _real_origin_and_clone(tmp_path / "repos")
    # Tag the first commit as 'v1', then grow a *branch* also named 'v1'.
    subprocess.run(["git", "-C", str(origin), "tag", "v1"], check=True)
    tag_sha = _git_out(origin, "rev-parse", "v1^{commit}")
    subprocess.run(
        ["git", "-C", str(origin), "checkout", "-q", "-b", "v1-branch-tmp"],
        check=True,
    )
    (origin / "f.txt").write_text("branch-two\n")
    subprocess.run(
        ["git", "-C", str(origin), "commit", "-qam", "branch two"], check=True
    )
    subprocess.run(
        ["git", "-C", str(origin), "branch", "-m", "v1-branch-tmp", "v1"],
        check=True,
    )

    feat, _db = await _make_feature(tmp_path)
    outcomes = await _run_git_steps(feat, clone, "v1")

    out = outcomes["reattach_branch"]
    assert out["ok"] is True
    assert "skip" in out["stdout_tail"]
    # Still detached on the TAG commit — never the shadowing branch tip.
    assert _git_out(clone, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    assert _git_out(clone, "rev-parse", "HEAD") == tag_sha


@pytest.mark.asyncio
async def test_reattach_ignores_stale_local_tag_named_like_branch(tmp_path):
    """A stale LOCAL tag named like the branch must not force tag intent:
    the fetch selected the branch, so the reattach must attach (codex
    round-2)."""
    origin, clone = _real_origin_and_clone(tmp_path / "repos")
    # Stale local-only tag in the CLONE shadowing the branch name.
    subprocess.run(["git", "-C", str(clone), "tag", "main"], check=True)
    (origin / "f.txt").write_text("two\n")
    subprocess.run(["git", "-C", str(origin), "commit", "-qam", "two"], check=True)
    origin_tip = _git_out(origin, "rev-parse", "HEAD")

    feat, _db = await _make_feature(tmp_path)
    outcomes = await _run_git_steps(feat, clone, "main")

    assert outcomes["reattach_branch"]["ok"] is True
    # Full symbolic-ref: --short would report 'heads/main' here because
    # the stale tag makes the bare name ambiguous.
    assert _git_out(clone, "symbolic-ref", "HEAD") == "refs/heads/main"
    assert _git_out(clone, "rev-parse", "refs/heads/main") == origin_tip


@pytest.mark.asyncio
async def test_reattach_sets_upstream_for_new_local_branch(tmp_path):
    """Attaching to a branch the clone never had locally must configure
    @{u}, or the next `kestrel update` bare pull fails (codex round-2)."""
    origin, clone = _real_origin_and_clone(tmp_path / "repos")
    subprocess.run(
        ["git", "-C", str(origin), "checkout", "-q", "-b", "deploy"],
        check=True,
    )
    (origin / "f.txt").write_text("deploy\n")
    subprocess.run(
        ["git", "-C", str(origin), "commit", "-qam", "deploy"], check=True
    )

    feat, _db = await _make_feature(tmp_path)
    outcomes = await _run_git_steps(feat, clone, "deploy")

    assert outcomes["reattach_branch"]["ok"] is True
    assert _git_out(clone, "symbolic-ref", "--short", "HEAD") == "deploy"
    assert (
        _git_out(clone, "rev-parse", "--abbrev-ref", "deploy@{upstream}")
        == "origin/deploy"
    )


@pytest.mark.asyncio
async def test_reattach_handles_fully_qualified_branch_ref(tmp_path):
    """refs/heads/<name> is a valid target ref; FETCH_HEAD records the
    short name, so the routine must normalize before matching (codex
    round-3)."""
    origin, clone = _real_origin_and_clone(tmp_path / "repos")
    (origin / "f.txt").write_text("two\n")
    subprocess.run(["git", "-C", str(origin), "commit", "-qam", "two"], check=True)
    origin_tip = _git_out(origin, "rev-parse", "HEAD")

    feat, _db = await _make_feature(tmp_path)
    outcomes = await _run_git_steps(feat, clone, "refs/heads/main")

    assert outcomes["reattach_branch"]["ok"] is True
    assert _git_out(clone, "symbolic-ref", "HEAD") == "refs/heads/main"
    assert _git_out(clone, "rev-parse", "HEAD") == origin_tip
    assert (
        _git_out(clone, "rev-parse", "--abbrev-ref", "main@{upstream}")
        == "origin/main"
    )


@pytest.mark.asyncio
async def test_reattach_ignores_incidental_not_for_merge_tag_line(tmp_path):
    """An origin tag sharing the branch's short name arrives as a
    not-for-merge FETCH_HEAD line when the branch was explicitly
    requested; intent must come from the for-merge line only (codex
    round-4)."""
    origin, clone = _real_origin_and_clone(tmp_path / "repos")
    # Origin-side tag named exactly like the branch, on the old commit.
    subprocess.run(["git", "-C", str(origin), "tag", "main"], check=True)
    (origin / "f.txt").write_text("two\n")
    subprocess.run(["git", "-C", str(origin), "commit", "-qam", "two"], check=True)
    origin_tip = _git_out(origin, "rev-parse", "refs/heads/main")

    feat, _db = await _make_feature(tmp_path)
    # Fully qualified: the fetch selects the BRANCH; --tags still writes
    # a not-for-merge line for the colliding tag.
    outcomes = await _run_git_steps(feat, clone, "refs/heads/main")

    out = outcomes["reattach_branch"]
    assert out["ok"] is True
    assert "skip" not in out["stdout_tail"]
    assert _git_out(clone, "symbolic-ref", "HEAD") == "refs/heads/main"
    assert _git_out(clone, "rev-parse", "refs/heads/main") == origin_tip


# ---------------------------------------------------------------------------
# The in-flight restart-completed ack supervisor is FEATURE-owned, so runtime
# disable / boot rollback cancel it — a delayed acknowledgement must never
# survive a disabled feature and flag wake_delivered against torn-down state
# (kestrel-sovereign#2522 P2).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disable_cancels_inflight_ack_supervisor(tmp_path):
    """``_spawn_ack_supervisor`` awaits ``handle.wait()`` and only then flags
    ``wake_delivered``. It must be a FEATURE-owned background task so
    ``Feature.shutdown()`` (what runtime disable / boot rollback call) cancels
    the in-flight wait — before the fix it lived only in the agent-global set,
    which is reaped ONLY at full agent shutdown, so a disabled feature's ack
    survived and could later mark a wake delivered against torn-down state."""
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()

    row = SimpleNamespace(id="req-inflight")
    feat._inflight_restart_acks.add(row.id)

    # A dispatch handle whose wake never lands on its own, so the ack supervisor
    # is a genuinely PENDING task at disable time.
    started = asyncio.Event()

    class _BlockingHandle:
        async def wait(self):
            started.set()
            await asyncio.Event().wait()  # blocks until cancelled

    marked: list = []

    async def _spy_mark_wake_delivered(r):
        marked.append(r)

    feat._mark_wake_delivered = _spy_mark_wake_delivered

    feat._spawn_ack_supervisor(row, _BlockingHandle())

    # The ack supervisor is FEATURE-owned AND still agent-tracked underneath.
    owned = list(feat._owned_background_tasks)
    assert len(owned) == 1
    ack_task = owned[0]
    assert ack_task in agent.background_tasks
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert not ack_task.done()

    # Canonical feature disable (runtime disable / boot rollback both call
    # Feature.shutdown()).
    await feat.shutdown()

    # The in-flight ack is cancelled — no delayed acknowledgement survives.
    assert ack_task.done()
    assert ack_task.cancelled()
    assert feat._owned_background_tasks == []
    assert marked == [], "a cancelled ack must NOT flag wake_delivered"
    # The in-flight marker is released on cancellation (the ack's finally).
    assert row.id not in feat._inflight_restart_acks


@pytest.mark.asyncio
async def test_delivered_ack_supervisor_self_removes_from_owned(tmp_path):
    """A completed ack must not linger in the feature's owned-task list — the
    owned tracker self-cleans on completion (mirroring the agent's global set),
    so repeated per-restart acks can't accumulate finished-task refs. Proves the
    ownership change stays leak-free on the happy path too (#2522 P2)."""
    from kestrel_sdk.signals import Status

    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()

    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing", expected_current_status="pending",
    )
    row = await get_request(backend, req.id)

    class _OkHandle:
        async def wait(self):
            return SimpleNamespace(status=Status.OK)

    feat._inflight_restart_acks.add(row.id)
    feat._spawn_ack_supervisor(row, _OkHandle())

    ack_task = list(feat._owned_background_tasks)[-1]
    await agent.drain_background_tasks()

    # The delivered ack landed...
    assert ack_task.done() and not ack_task.cancelled()
    updated = await get_request(backend, req.id)
    assert updated.wake_delivered is True
    # ...and self-removed from the owned list (no accumulation).
    assert ack_task not in feat._owned_background_tasks


# ---------------------------------------------------------------------------
# #2665 — deferral reasons must name the blocking handles, not just count them
# ---------------------------------------------------------------------------


class _FakeTask:
    """A background task with a name and a known age."""

    def __init__(self, name, age_seconds=0.0, now=1000000.0):
        self._name = name
        self._kestrel_started_at = now - age_seconds

    def get_name(self):
        return self._name


def test_busy_deferral_names_the_blocking_tasks():
    """A bare count is not reconcilable against the task store. The live report
    of "2 background tasks in flight" alongside `list_my_tasks` returning zero
    rows was undiagnosable precisely because the coordinator never said WHICH
    handles it meant (#2665).
    """
    described = _describe_background_tasks(
        [
            _FakeTask("signal_dispatch:talon:sig_1", age_seconds=5),
            _FakeTask("a2a_send:claw", age_seconds=1),
        ],
        now=1000.0,
    )
    assert "signal_dispatch:talon:sig_1" in described
    assert "a2a_send:claw" in described


def test_busy_deferral_reports_age_not_just_identity():
    """Age is what separates "busy" from "wedged", and #2665's symptom was a
    duration symptom. The adjacent active-request path already reports age;
    this one used to report none, so a task stuck for hours rendered
    identically to one that appeared a moment ago.
    """
    described = _describe_background_tasks(
        [_FakeTask("signal_dispatch:stuck", age_seconds=7200)],
        now=1000000.0,
    )
    assert "2h" in described


def test_oldest_task_is_reported_first():
    """Oldest first puts the likely culprit at the front of a truncated list."""
    described = _describe_background_tasks(
        [
            _FakeTask("young:a", age_seconds=1),
            _FakeTask("ancient:b", age_seconds=9999),
            _FakeTask("middle:c", age_seconds=100),
        ],
        now=100000.0,
    )
    assert described.index("ancient") < described.index("middle")
    assert described.index("middle") < described.index("young")


def test_a_flood_of_one_kind_cannot_hide_the_wedged_task():
    """The regression this ordering exists for. Sorted alphabetically and
    truncated to five, six `a2a_*` tasks pushed a wedged `signal_dispatch:*`
    out of the string entirely — the bound doing the OPPOSITE of its job at
    exactly the moment it engaged. Grouping by kind means no kind can be
    truncated away by the volume of another.
    """
    tasks = [
        _FakeTask(f"a2a_complete:{i:08d}", age_seconds=1) for i in range(6)
    ]
    tasks.append(_FakeTask("signal_dispatch:channels:sig_WEDGED", age_seconds=900))

    described = _describe_background_tasks(tasks, now=100000.0)

    assert "sig_WEDGED" in described, (
        f"the wedged task must not be truncated away; got: {described}"
    )
    # And the flood is summarised rather than spending the whole budget.
    assert "x6" in described


def test_duplicate_kinds_are_collapsed_with_a_count():
    """Five identically-named tasks used to render five times, spending the
    entire budget on one bit of information."""
    described = _describe_background_tasks(
        [_FakeTask("post_response_memory_enrichment", age_seconds=i)
         for i in range(5)],
        now=1000.0,
    )
    assert "x5" in described
    assert described.count("post_response_memory_enrichment") == 1


def test_busy_deferral_bounds_the_named_kinds():
    """Deferral reasons are persisted as a status-event row on every cron tick
    with no dedupe, so the string must stay bounded for a wedged restart."""
    described = _describe_background_tasks(
        [_FakeTask(f"kind{i}:x", age_seconds=i)
         for i in range(_MAX_NAMED_BUSY_KINDS + 3)],
        now=1000.0,
    )
    assert "+3 more kind(s)" in described


def test_a_single_pathological_name_cannot_dominate():
    described = _describe_background_tasks(
        [_FakeTask("x" * 500, age_seconds=1)], now=1000.0,
    )
    assert len(described) < 200


def test_busy_deferral_survives_a_task_without_a_readable_name():
    """An unnamed or introspection-hostile handle must not break the idle gate
    — reporting it as unnamed is still more useful than a bare count.
    """
    class _Hostile:
        def get_name(self):
            raise RuntimeError("no name for you")

    assert "<unnamed>" in _describe_background_tasks([_Hostile()], now=1000.0)


def test_unstamped_task_reports_unknown_age_rather_than_guessing():
    """A task predating the age stamp must not be reported with a fabricated
    age."""
    class _Unstamped:
        def get_name(self):
            return "legacy:task"

    described = _describe_background_tasks([_Unstamped()], now=1000.0)
    assert "age unknown" in described


@pytest.mark.asyncio
async def test_idle_gate_reason_carries_task_names_end_to_end(tmp_path):
    """The names must reach the reason string the coordinator actually emits,
    not just the helper.
    """
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()

    async def _never():
        await asyncio.sleep(3600)

    blocker = asyncio.create_task(_never(), name="signal_dispatch:blocking")
    agent._background_tasks = {blocker}
    try:
        state = feat._agent_appears_idle()
        assert state["idle"] is False
        assert "signal_dispatch:blocking" in state["reason"]
    finally:
        blocker.cancel()
        await asyncio.gather(blocker, return_exceptions=True)


# ---------------------------------------------------------------------------
# #2738 — a lost wake_delivered write must never pass for a successful one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_wake_delivered_reports_whether_the_write_landed(tmp_path):
    """It previously returned ``None`` and discarded the rowcount, so callers
    could not tell a persisted flag from a lost one."""
    from kestrel_sovereign.features.restart_coordinator.store import (
        mark_wake_delivered,
    )

    backend = await _backend(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="r",
    )

    assert await mark_wake_delivered(backend, req.id) is True
    assert await mark_wake_delivered(backend, "no-such-request") is False


@pytest.mark.asyncio
async def test_lost_wake_delivered_write_is_reported_not_swallowed(
    tmp_path, caplog,
):
    """The failure used to be swallowed at DEBUG while the in-memory row was
    marked delivered anyway — so the row claimed a delivery the database never
    recorded, and the sweep re-emitted forever with nothing saying why."""
    feat, _agent = await _make_feature(tmp_path)
    ghost = SimpleNamespace(id="does-not-exist", wake_delivered=False)

    with caplog.at_level(logging.WARNING):
        await feat._mark_wake_delivered(ghost)

    assert ghost.wake_delivered is False, (
        "a write that matched no row must not report success"
    )
    assert any(
        "matched no row" in r.getMessage() for r in caplog.records
    ), "a lost wake_delivered write must be reported at WARNING"


@pytest.mark.asyncio
async def test_a_lost_write_never_claims_delivery_across_repeated_sweeps(
    tmp_path, caplog,
):
    """When the write is LOST the sweep keeps re-emitting — that is the retry
    guarantee working. What must not happen is the pre-fix behaviour, where the
    row was marked delivered in memory anyway.

    The happy path alone would NOT catch this: the pre-fix code's write also
    landed when nothing went wrong, and #2738 only manifests when it does not.
    """
    from kestrel_sovereign.features.restart_coordinator import feature as fm

    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )
    await feat.initialize()

    async def _lost_write(db, request_id):
        return False

    original = fm.mark_wake_delivered
    fm.mark_wake_delivered = _lost_write
    try:
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                tasks = await feat._reap_post_restart_rows()
                if tasks:
                    await asyncio.gather(*tasks)
                await agent.drain_background_tasks()
    finally:
        fm.mark_wake_delivered = original

    row = await get_request(backend, req.id)
    assert row.wake_delivered is False, (
        "the row must never claim a delivery the database did not record"
    )
    assert any("matched no row" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_repeated_sweeps_emit_exactly_one_completion_wake(tmp_path):
    """With the write landing, repeated sweeps produce exactly ONE wake."""
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )
    await feat.initialize()

    for _ in range(5):
        tasks = await feat._reap_post_restart_rows()
        if tasks:
            await asyncio.gather(*tasks)
        await agent.drain_background_tasks()

    assert len(agent.process_input_calls) == 1
    assert (await get_request(backend, req.id)).wake_delivered is True


# ---------------------------------------------------------------------------
# #2667 — a dispatched restart that never happens must not vanish silently
# ---------------------------------------------------------------------------


def _dead_child(tmp_path, returncode=1, stderr="kestrel: no such command\n"):
    """A restart subprocess that exited instead of restarting the host.

    Carries a real stderr FILE, matching production: the child must outlive
    this process, so its stderr cannot be a pipe whose read end dies with us.
    """
    err_path = tmp_path / "restart.err"
    err_path.write_text(stderr)

    class _Dead:
        pid = 4242

        def __init__(self):
            self.returncode = returncode
            self._kestrel_stderr_path = str(err_path)

        def poll(self):
            return returncode

    return _Dead()


class _LiveChild:
    """A restart subprocess still running — the healthy case."""

    pid = 4243
    returncode = None

    def poll(self):
        return None


@pytest.mark.asyncio
async def test_dead_restart_child_returns_the_row_for_retry(tmp_path):
    """``Popen`` returning is not evidence the restart happened. A child that
    exits without restarting the host left the row ``executing`` forever with
    no error event and no notice to anyone (#2667).
    """
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()
    feat._restart_dispatch_grace = 0
    created = await feat.request_restart(reason="ship")
    req_id = created.data["request"]["id"]

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
        return_value=_dead_child(tmp_path),
    ):
        await feat.restart_coordinator()
    await agent.drain_background_tasks()

    row = await get_request(backend, req_id)
    assert row.status == "pending", "a failed dispatch must not stay executing"
    # The reason names the exit status AND the child's own complaint, which
    # used to go to DEVNULL.
    assert "exited 1" in row.status_reason
    assert "no such command" in row.status_reason


@pytest.mark.asyncio
async def test_live_restart_child_leaves_the_row_executing(tmp_path):
    """The watchdog must RUN and decline to act on a restart still in flight.

    Asserting only "the row is still executing" would be toothless: with no
    watchdog at all the row stays executing for exactly that reason, so the
    test passes with and without the fix. It must show the watchdog was
    consulted and returned no failure.
    """
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()
    feat._restart_dispatch_grace = 0
    created = await feat.request_restart(reason="ship")
    req_id = created.data["request"]["id"]

    consulted = []
    real_check = feat._restart_dispatch_failure

    def _spy(proc):
        verdict = real_check(proc)
        consulted.append(verdict)
        return verdict

    feat._restart_dispatch_failure = _spy
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
        return_value=_LiveChild(),
    ):
        await feat.restart_coordinator()
    await agent.drain_background_tasks()

    assert consulted, "the watchdog never ran"
    assert all(v is None for v in consulted), (
        f"a live child must not be judged a failure; got {consulted}"
    )
    row = await get_request(backend, req_id)
    assert row.status == "executing"
    # Still tracked as in flight, so the reconciler leaves it alone too.
    assert req_id in feat._executing_since


@pytest.mark.asyncio
async def test_unverifiable_child_is_not_claimed_as_a_failure(tmp_path):
    """A handle that cannot report an integer exit status is not evidence of
    anything. Concluding failure from it would bounce live restarts."""
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()
    assert feat._restart_dispatch_failure(MagicMock()) is None


@pytest.mark.asyncio
async def test_stranded_executing_row_is_recovered_by_the_sweep(tmp_path):
    """The durable backstop. Before this, the coordinator scanned only
    pending/approved and ``cancel_restart_request`` refused executing rows, so
    a row whose restart never happened had NO path back — and the next
    unrelated restart would terminalize it as 'completed', reporting success
    for a restart that never ran (#2667).
    """
    from kestrel_sovereign.features.restart_coordinator.feature import (
        _PROCESS_BOOT_ID,
    )

    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="stranded",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
        executing_boot_id=_PROCESS_BOOT_ID,
    )
    # Stamped by this process but with no dispatch in flight — nothing is
    # waiting on it and nothing else will ever move it.
    assert req.id not in feat._executing_since
    # This instance has been up past the grace window, so an untracked row is
    # genuinely orphaned rather than one a previous instance just started.
    from kestrel_sovereign.features.restart_coordinator.feature import (
        STALE_EXECUTING_SECONDS,
    )

    feat._instance_started_at -= STALE_EXECUTING_SECONDS + 1

    reset = await feat._reconcile_stranded_executing_rows()
    assert reset == [req.id]
    row = await get_request(backend, req.id)
    assert row.status == "pending"
    assert "did not happen" in row.status_reason


@pytest.mark.asyncio
async def test_reconciler_leaves_a_prior_boot_row_for_the_wake_sweep(tmp_path):
    """A row stamped by a PRIOR boot means the restart provably happened — it
    belongs to the post-restart wake sweep, not to stranded-row recovery."""
    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="prior boot",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
        executing_boot_id="a-different-boot",
    )

    assert await feat._reconcile_stranded_executing_rows() == []
    row = await get_request(backend, req.id)
    assert row.status == "executing"


@pytest.mark.asyncio
async def test_reconciler_leaves_an_in_flight_dispatch_alone(tmp_path):
    """A dispatch this process started moments ago is in flight, not stuck."""
    from kestrel_sovereign.features.restart_coordinator.feature import (
        _PROCESS_BOOT_ID,
    )

    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="in flight",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
        executing_boot_id=_PROCESS_BOOT_ID,
    )
    feat._executing_since[req.id] = time.monotonic()

    assert await feat._reconcile_stranded_executing_rows() == []
    assert (await get_request(backend, req.id)).status == "executing"


@pytest.mark.asyncio
async def test_restart_child_stderr_is_a_file_not_a_pipe(tmp_path):
    """The restart child must OUTLIVE this process, so its stderr cannot be a
    pipe: the read end dies with us, and the child would take EPIPE on its next
    write — breaking the very restart we are performing. A chatty child would
    also block forever on a full pipe buffer nobody is reading.

    Exercises the real spawn path, not a double.
    """
    feat, _agent = await _make_feature(tmp_path)
    with patch(
        "kestrel_sovereign.features.restart_coordinator.feature.shutil.which",
        return_value="/bin/echo",
    ):
        proc = feat._spawn_restart_subprocess()
    try:
        assert proc.stderr is None, (
            "stderr must not be a pipe — it dies with this process"
        )
        # The assertion above is equally true of the old DEVNULL, so it alone
        # proves nothing. THIS is the discriminator: a real file on disk that
        # survives us and that the child can write to without blocking.
        path = proc._kestrel_stderr_path
        assert os.path.exists(path), "stderr must go to a real file"
    finally:
        proc.wait(timeout=10)
        feat._read_restart_stderr_tail(proc)

    # The tail read cleans the file up rather than leaking one per restart.
    assert not os.path.exists(path)


@pytest.mark.asyncio
async def test_dead_child_stderr_tail_is_read_and_cleaned_up(tmp_path):
    """The child's last line is the only evidence of why a dispatch failed,
    and reading it must not leave a file behind on every failure."""
    feat, _agent = await _make_feature(tmp_path)
    err = tmp_path / "boom.err"
    err.write_text("first line\nkestrel: boom\n")

    class _Dead:
        pid = 99
        _kestrel_stderr_path = str(err)

        def poll(self):
            return 7

    proc = _Dead()
    reason = feat._restart_dispatch_failure(proc)
    assert "exited 7" in reason
    assert "kestrel: boom" in reason
    assert not err.exists(), "the stderr file must not be left behind"


@pytest.mark.asyncio
async def test_boot_sweeps_stderr_files_orphaned_by_successful_restarts(
    tmp_path, monkeypatch,
):
    """A SUCCESSFUL restart kills this process before it can clean up its
    child's stderr file, so those are orphaned by definition. Boot is the only
    place that can collect them; without this it is one file per restart,
    forever. A recent file must survive — it may belong to a live dispatch.
    """
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    old = tmp_path / "kestrel-restart-old.err"
    new = tmp_path / "kestrel-restart-new.err"
    unrelated = tmp_path / "something-else.err"
    for p in (old, new, unrelated):
        p.write_text("x")
    os.utime(old, (0, 0))
    os.utime(unrelated, (0, 0))

    removed = RestartCoordinatorFeature._sweep_orphaned_restart_stderr()

    assert removed == 1
    assert not old.exists()
    assert new.exists(), "a recent file may belong to a dispatch in flight"
    assert unrelated.exists(), "the sweep must not touch unrelated files"


@pytest.mark.asyncio
async def test_reconciler_cannot_reset_a_dispatch_mid_transition(tmp_path):
    """TOCTOU guard, probing the actual window.

    ``update_status`` awaits. If the in-flight record is written AFTER it
    returns, there is a window where the row is durably ``executing`` under
    this boot id with NO entry in ``_executing_since`` — and the reconciler
    treats exactly that as an orphan. A cron tick landing there resets a
    dispatch that is very much alive.

    This runs the reconciler INSIDE that window (from within update_status,
    immediately after the executing row commits) rather than after the fact,
    which is the only placement that can tell the two orderings apart.
    """
    from kestrel_sovereign.features.restart_coordinator import feature as fm

    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()
    feat._restart_dispatch_grace = 0
    created = await feat.request_restart(reason="ship")
    req_id = created.data["request"]["id"]

    real_update_status = fm.update_status
    fired = {"count": 0}

    async def _reconcile_inside_the_window(db, request_id, **kwargs):
        landed = await real_update_status(db, request_id, **kwargs)
        if kwargs.get("status") == "executing" and landed:
            # A concurrent cron tick, arriving at the worst possible moment.
            fired["count"] += 1
            await feat._reconcile_stranded_executing_rows()
        return landed

    fm.update_status = _reconcile_inside_the_window
    try:
        with patch.object(
            RestartCoordinatorFeature, "_spawn_restart_subprocess",
            return_value=_LiveChild(),
        ):
            await feat.restart_coordinator()
        await agent.drain_background_tasks()
    finally:
        fm.update_status = real_update_status

    assert fired["count"] == 1, "the window was never exercised"
    row = await get_request(backend, req_id)
    assert row.status == "executing", (
        "a concurrent reconciler reset a dispatch that was still in flight"
    )


@pytest.mark.asyncio
async def test_a_fresh_instance_does_not_orphan_a_previous_ones_dispatch(
    tmp_path,
):
    """The boot id is module-scoped but ``_executing_since`` is per-INSTANCE.
    A feature reload inside the same process therefore starts with an empty
    map, and the untracked-row branch does no age check of its own — so
    without a guard it would reset a dispatch the previous instance started
    moments ago, on the first tick.
    """
    from kestrel_sovereign.features.restart_coordinator.feature import (
        _PROCESS_BOOT_ID,
    )

    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="in flight",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
        executing_boot_id=_PROCESS_BOOT_ID,
    )

    # Simulate the reload: same process, brand-new feature instance.
    reloaded = RestartCoordinatorFeature(agent)
    await reloaded.initialize()
    assert req.id not in reloaded._executing_since

    assert await reloaded._reconcile_stranded_executing_rows() == [], (
        "a just-reloaded instance must not orphan a live dispatch"
    )
    assert (await get_request(backend, req.id)).status == "executing"


@pytest.mark.asyncio
async def test_repeated_dispatch_failures_stop_rather_than_flap(tmp_path):
    """Returning the row to `pending` means the next tick re-dispatches. For a
    permanently broken restart that turns "stuck forever" into "flaps forever"
    — a doomed subprocess every minute, each with its own status event. After
    a few identical failures the request is rejected terminally instead.
    """
    from kestrel_sovereign.features.restart_coordinator.feature import (
        MAX_RESTART_DISPATCH_ATTEMPTS,
    )

    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    await feat.initialize()
    feat._restart_dispatch_grace = 0
    created = await feat.request_restart(reason="ship")
    req_id = created.data["request"]["id"]

    for _ in range(MAX_RESTART_DISPATCH_ATTEMPTS):
        with patch.object(
            RestartCoordinatorFeature, "_spawn_restart_subprocess",
            return_value=_dead_child(tmp_path),
        ):
            await feat.restart_coordinator()
        await agent.drain_background_tasks()

    row = await get_request(backend, req_id)
    assert row.status == "rejected", (
        f"expected a terminal reject after "
        f"{MAX_RESTART_DISPATCH_ATTEMPTS} failures, got {row.status}"
    )
    assert "giving up" in row.status_reason
    assert row.completed_at is not None


# ---------------------------------------------------------------------------
# #2774 — schema migration is one atomic step, and a wake's DISPATCH is
# recorded separately from its DELIVERY.
#
# ``wake_delivered`` only flips once the woken cognition turn returns
# Status.OK, i.e. strictly after that turn ends. So the turn a wake wakes can
# never observe the flag as true, and a row mid-delivery is indistinguishable
# from one whose wake never fired at all. That made the field unusable as
# negative evidence — the report that opened #2774 was a row recording
# ``wake_delivered: false`` for a wake whose own consumer was reading it.
# ---------------------------------------------------------------------------


# The #1512 schema, before anything in ``_ADDED_COLUMNS`` existed.
_ORIGINAL_1512_COLUMNS = (
    "id TEXT PRIMARY KEY, requested_by_agent TEXT NOT NULL, "
    "reason TEXT NOT NULL, requested_at TEXT NOT NULL, "
    "desired_window TEXT DEFAULT '', urgency TEXT DEFAULT 'normal', "
    "policy TEXT DEFAULT 'idle_agents_only', status TEXT DEFAULT 'pending', "
    "status_reason TEXT DEFAULT '', completed_at TEXT"
)


async def _db_at_schema(tmp_path, filename, *, through_column=None):
    """A ``restart_requests`` table frozen at a historical schema point.

    ``through_column`` names the newest ``_ADDED_COLUMNS`` entry the table
    has; everything after it is absent, exactly as on a database last touched
    by that release. ``None`` gives the original #1512 table.

    Derived from ``_ADDED_COLUMNS`` rather than hand-copied so a test cannot
    quietly describe a schema that never shipped.
    """
    from kestrel_sovereign.features.restart_coordinator.store import (
        _ADDED_COLUMNS,
    )

    cols = [_ORIGINAL_1512_COLUMNS]
    if through_column is not None:
        names = [c for c, _ in _ADDED_COLUMNS]
        assert through_column in names, (
            f"{through_column} is not an _ADDED_COLUMNS entry"
        )
        for col, col_def in _ADDED_COLUMNS:
            cols.append(f"{col} {col_def}")
            if col == through_column:
                break

    raw = SQLiteBackend(str(tmp_path / filename))
    await raw.connect()
    db = _track_test_database(AsyncDatabase(raw))
    await db.execute(f"CREATE TABLE restart_requests ({', '.join(cols)})")
    return db


async def _insert_raw(db, request_id, **cols):
    """Insert a row through raw SQL, bypassing the store's column contract.

    Migration tests must be able to write rows a *pre-migration* build could
    have written, which ``insert_request`` cannot express.
    """
    fields = {
        "id": request_id,
        "requested_by_agent": "did:test:agent",
        "reason": "r",
        "requested_at": "t",
        **cols,
    }
    names = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    await db.execute(
        f"INSERT INTO restart_requests ({names}) VALUES ({marks})",
        tuple(fields.values()),
    )


@pytest.mark.asyncio
async def test_migrated_pending_request_requires_escalation_acknowledgement(
    tmp_path,
):
    db = await _db_at_schema(
        tmp_path, "legacy-escalation.db", through_column="wake_dispatch_count",
    )
    await _insert_raw(db, "legacy-pending", status="pending")

    await ensure_restart_requests_table(db)

    row = await get_request(db, "legacy-pending")
    assert row.first_blocked_at == ""
    assert row.escalation_acknowledged is False


@pytest.mark.asyncio
async def test_an_interrupted_backfill_rolls_back_its_column_too(
    tmp_path, monkeypatch,
):
    """A column and its legacy backfill land together or not at all.

    The previous implementation ran each ALTER as its own statement and
    inferred "already migrated" from the column merely existing. A process
    that died between the ALTER committing and the backfill running left a
    state no later boot could detect: the column was present, so every
    subsequent boot skipped the backfill forever. For ``wake_delivered`` that
    means every historical completed restart stays eligible for the sweep and
    gets re-woken, every tick, permanently.

    One transaction removes that state from the state space rather than adding
    a ledger to detect it.
    """
    from kestrel_sovereign.features.restart_coordinator import store as rc_store

    db = await _db_at_schema(
        tmp_path, "interrupted.db", through_column="origin_session_id",
    )
    await _insert_raw(db, "old-done", status="completed", completed_at="t")

    broken = dict(rc_store._COLUMN_BACKFILLS)
    broken["wake_delivered"] = (
        "UPDATE restart_requests SET no_such_column = 1", (),
    )
    monkeypatch.setattr(rc_store, "_COLUMN_BACKFILLS", broken)

    with pytest.raises(Exception):
        await ensure_restart_requests_table(db)

    assert not await db._column_exists("restart_requests", "wake_delivered"), (
        "the column must roll back with its own failed backfill — committing "
        "it strands the backfill permanently, because every later boot sees "
        "the column and skips"
    )

    # The next boot, with the backfill working, completes the whole migration.
    monkeypatch.undo()
    await ensure_restart_requests_table(db)
    assert await db._column_exists("restart_requests", "wake_delivered")
    assert (await get_request(db, "old-done")).wake_delivered is True, (
        "the retry must run the backfill it previously rolled back"
    )


@pytest.mark.asyncio
async def test_a_failed_alter_is_never_mistaken_for_an_existing_column(
    tmp_path, monkeypatch,
):
    """An ALTER that fails for a real reason must fail the migration.

    The previous code wrapped every ALTER in ``except Exception: continue``
    with the comment "column already exists — expected on every non-first
    run". A lock timeout, disk pressure, or a Postgres permission error is
    indistinguishable there, and because every SELECT projects the full column
    list, one silently-skipped ALTER makes each subsequent read raise for the
    rest of the boot while this function still reports the table ready.
    """
    db = await _db_at_schema(
        tmp_path, "failed-alter.db", through_column="origin_session_id",
    )

    async def _disk_full(self, table, column, col_def):
        raise OSError("disk full")

    monkeypatch.setattr(AsyncDatabase, "_migrate_add_column", _disk_full)

    with pytest.raises(Exception, match="disk full"):
        await ensure_restart_requests_table(db)


@pytest.mark.asyncio
async def test_a_silently_skipped_column_fails_instead_of_reporting_ready(
    tmp_path, monkeypatch,
):
    """The post-migration check is what makes "table ready" mean something.

    Modelled on an ALTER that neither raises nor adds the column — the shape a
    mis-scoped Postgres ``information_schema`` lookup produces, where the
    migration is skipped as already-applied and any verification asking the
    same wrong question agrees.
    """
    db = await _db_at_schema(
        tmp_path, "skipped.db", through_column="origin_session_id",
    )
    real = AsyncDatabase._migrate_add_column

    async def _skip_one(self, table, column, col_def):
        if column == "wake_dispatch_count":
            return
        await real(self, table, column, col_def)

    monkeypatch.setattr(AsyncDatabase, "_migrate_add_column", _skip_one)

    with pytest.raises(Exception, match="wake_dispatch_count"):
        await ensure_restart_requests_table(db)

    monkeypatch.undo()
    assert not await db._column_exists("restart_requests", "wake_delivered"), (
        "a migration that cannot complete must leave nothing behind, so the "
        "next boot starts from the same place rather than a partial schema"
    )


@pytest.mark.asyncio
async def test_the_dispatch_sentinel_marks_delivered_rows_not_completed_ones(
    tmp_path,
):
    """Legacy rows get dispatch evidence only where a wake actually landed.

    ``pre-migration`` means "delivered under the old flow; whether a wake was
    dispatched is unrecoverable" — it is deliberately not a claim that one
    was, because some of those rows provably had none (a host with no usable
    dispatcher marks a row delivered without sending anything). Its job is to
    keep '' meaning "no wake was ever dispatched for this row", which is the
    negative evidence #2774 needs.

    So the sentinel is keyed on ``wake_delivered``, never on ``status``: a
    completed-but-undelivered row is one the sweep is still retrying.
    """
    from kestrel_sovereign.features.restart_coordinator.store import (
        PRE_MIGRATION_BOOT_ID,
    )

    db = await _db_at_schema(
        tmp_path, "sentinel.db", through_column="wake_delivered",
    )
    await _insert_raw(
        db, "delivered", status="completed", completed_at="t", wake_delivered=1,
    )
    await _insert_raw(
        db, "undelivered", status="completed", completed_at="t",
        wake_delivered=0,
    )

    await ensure_restart_requests_table(db)

    assert (
        await get_request(db, "delivered")
    ).wake_dispatch_boot_id == PRE_MIGRATION_BOOT_ID
    assert (await get_request(db, "undelivered")).wake_dispatch_boot_id == "", (
        "a completed row whose wake never landed is still being retried; "
        "stamping it would forge evidence of a dispatch that never happened"
    )


@pytest.mark.asyncio
async def test_the_sentinel_backfill_never_reruns_on_a_migrated_database(
    tmp_path,
):
    """Backfills are gated on the schema, so they cannot touch live rows twice.

    A wake dispatched after the migration whose stamp was lost (the same
    SQLite lock contention that cost #2660 its 2,045 signal_log writes) leaves
    a delivered row with an empty stamp — indistinguishable, by value, from a
    legacy row. Any gate other than "am I the one adding this column" re-runs
    the sentinel over it and invents a dispatch record.
    """
    # The case with teeth: a LATER column is still missing, so the migration
    # runs for real rather than short-circuiting on the fast path. The
    # sentinel's own column is already present, so its backfill must not fire.
    db = await _db_at_schema(
        tmp_path, "partial.db", through_column="wake_dispatch_boot_id",
    )
    await _insert_raw(
        db, "stamp-lost", status="completed", completed_at="t",
        wake_delivered=1, wake_dispatch_boot_id="",
    )

    await ensure_restart_requests_table(db)

    assert await db._column_exists("restart_requests", "wake_dispatch_count"), (
        "the missing column must still be added — this is a real migration"
    )
    assert (await get_request(db, "stamp-lost")).wake_dispatch_boot_id == "", (
        "this row was delivered by the CURRENT code, which stamps on "
        "dispatch; an empty stamp here is a lost write, not a legacy row"
    )

    # And on a fully-migrated database the migration is not entered at all.
    # ``ensure_restart_requests_table`` runs on every agent init, and entering
    # takes SQLite's single writer slot (BEGIN IMMEDIATE) — the same
    # every-boot cost #2649 gated its ownership backfills behind.
    fresh = await _backend(tmp_path)
    await _insert_raw(
        fresh, "post-migration", status="completed", completed_at="t",
        wake_delivered=1,
    )
    with patch.object(
        AsyncDatabase, "migration_lock", side_effect=AssertionError(
            "a database with nothing missing must not take the writer slot"
        ),
    ):
        await ensure_restart_requests_table(fresh)
    assert (
        await get_request(fresh, "post-migration")
    ).wake_dispatch_boot_id == ""


@pytest.mark.asyncio
async def test_every_dispatch_is_counted_so_a_wake_storm_shows_in_the_row(
    tmp_path,
):
    """#2738 was ~18 re-emissions inside one boot. Recording only the first
    dispatch would have shown a single timestamp for the whole event."""
    from kestrel_sovereign.features.restart_coordinator.store import (
        mark_wake_dispatched,
    )

    db = await _backend(tmp_path)
    req = await insert_request(
        db, requested_by_agent="did:test:agent", reason="storm",
    )

    for i in range(3):
        assert await mark_wake_dispatched(
            db, req.id, dispatched_at=f"t{i}", boot_id="boot-1",
        ) is True

    row = await get_request(db, req.id)
    assert row.wake_dispatch_count == 3
    assert row.wake_dispatched_at == "t2", "the stamp is the LATEST dispatch"
    assert row.wake_dispatch_boot_id == "boot-1"

    assert await mark_wake_dispatched(
        db, "no-such-request", dispatched_at="t", boot_id="boot-1",
    ) is False, "a write that matched no row must report that it did not land"


@pytest.mark.asyncio
async def test_the_dispatch_is_stamped_before_the_signal_is_handed_over(
    tmp_path, monkeypatch,
):
    """Ordering, asserted directly rather than inferred from a race.

    ``enqueue_signal`` starts the dispatch immediately, so every await after
    it is a point at which the woken turn may already be running and reading
    the row. Stamping afterwards — even one await later — races the very
    reader the stamp exists for, and on a Postgres pool the two run on
    different connections concurrently. Observing the stamp from inside the
    turn (the test below) can pass on timing alone; this pins the mechanism.
    """
    import kestrel_sovereign.features.restart_coordinator.feature as fm

    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing", expected_current_status="pending",
    )
    await feat.initialize()

    order: list[str] = []
    real_enqueue = agent.dispatcher.enqueue_signal
    real_mark = fm.mark_wake_dispatched

    def _record_enqueue(signal):
        order.append("enqueue")
        return real_enqueue(signal)

    async def _record_mark(*args, **kwargs):
        order.append("stamp")
        return await real_mark(*args, **kwargs)

    agent.dispatcher.enqueue_signal = _record_enqueue
    monkeypatch.setattr(fm, "mark_wake_dispatched", _record_mark)

    await asyncio.gather(*await feat.on_agent_ready())

    assert order == ["stamp", "enqueue"], (
        f"the dispatch stamp must be durable before the dispatcher can start "
        f"the turn that reads it; got {order}"
    )


@pytest.mark.asyncio
async def test_the_woken_turn_can_see_that_its_own_wake_was_dispatched(
    tmp_path,
):
    """The whole point of #2774, asserted from inside the woken turn.

    ``wake_delivered`` is set from the acknowledgement of ``handle.wait()``,
    which resolves only after ``process_input`` returns — so during the turn
    it is necessarily still 0. Reading it there and concluding "no wake fired"
    is what made the reported row contradict its own consumer. The dispatch
    stamp is written before the turn starts and answers the question the flag
    structurally cannot.
    """
    from kestrel_sovereign.features.restart_coordinator.feature import (
        _PROCESS_BOOT_ID,
    )

    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing", expected_current_status="pending",
    )

    seen: dict = {}
    original = agent.process_input

    async def _observe_from_inside_the_turn(prompt, session_id=None):
        row = await get_request(backend, req.id)
        seen["dispatched_at"] = row.wake_dispatched_at
        seen["boot_id"] = row.wake_dispatch_boot_id
        seen["count"] = row.wake_dispatch_count
        seen["delivered"] = row.wake_delivered
        return await original(prompt, session_id=session_id)

    agent.process_input = _observe_from_inside_the_turn

    await feat.initialize()
    await asyncio.gather(*await feat.on_agent_ready())

    assert seen, "the wake never reached a turn"
    assert seen["dispatched_at"], (
        "the dispatch must be stamped BEFORE handle.wait() runs the turn; "
        "stamped afterwards it is invisible to the turn that would read it"
    )
    assert seen["boot_id"] == _PROCESS_BOOT_ID
    assert seen["count"] == 1
    assert seen["delivered"] is False, (
        "wake_delivered cannot be true inside the turn its own wake started — "
        "that asymmetry is what #2774 reported, not a bug to fix here"
    )

    # And delivery is still recorded once the turn returns.
    assert (await get_request(backend, req.id)).wake_delivered is True


@pytest.mark.asyncio
async def test_a_lost_dispatch_stamp_is_reported_and_never_blocks_the_wake(
    tmp_path, caplog,
):
    """Observability is best-effort, but its failure is not silent.

    These columns exist to BE the record of a dispatch, so dropping the write
    without a word is the same asymmetry that made ``wake_delivered``
    untrustworthy.
    """
    feat, _ = await _make_feature(tmp_path)
    ghost = SimpleNamespace(
        id="does-not-exist", wake_dispatched_at="",
        wake_dispatch_boot_id="", wake_dispatch_count=0,
    )

    with caplog.at_level(logging.WARNING):
        await feat._mark_wake_dispatched(ghost)

    assert ghost.wake_dispatched_at == "", (
        "the in-memory row must not claim a stamp the database rejected"
    )
    assert ghost.wake_dispatch_count == 0
    assert any(
        "wake dispatch stamp" in r.getMessage() for r in caplog.records
    ), "a lost dispatch stamp must be reported at WARNING"


@pytest.mark.asyncio
async def test_an_unusable_schema_disables_coordinator_storage_for_the_boot(
    tmp_path,
):
    """Fail closed. A half-migrated table makes every read raise, so carrying
    on with the handle leaves the feature reporting itself enabled while
    waking nobody for the entire boot — the failure mode the store's
    post-migration check exists to catch, reintroduced one level up."""
    backend = await _backend(tmp_path)
    agent = _make_agent(backend)
    feat = RestartCoordinatorFeature(agent)

    with patch(
        "kestrel_sovereign.features.restart_coordinator.feature."
        "ensure_restart_requests_table",
        side_effect=RuntimeError(
            "restart_requests is missing column(s) after migration: "
            "wake_dispatched_at"
        ),
    ), patch(
        "kestrel_sovereign.features.restart_coordinator.feature."
        "ensure_restart_status_events_table",
    ) as events_table:
        await feat.initialize()

    assert feat._db is None, (
        "an unusable schema must drop the storage handle, not just log"
    )
    events_table.assert_not_called()

    result = await feat.list_restart_requests()
    assert result.status is ToolResultStatus.ERROR
    assert "storage unavailable" in result.error


@pytest.mark.asyncio
async def test_a_lost_delivered_write_storms_but_never_claims_delivery(
    tmp_path, caplog, monkeypatch,
):
    """The #2738 mechanism end to end, with the write LOST.

    Re-emission is not the bug — it is the retry guarantee, and it is supposed
    to keep going while the flag reads 0. The bug was the pre-fix behaviour
    where the failed write was swallowed at DEBUG and the in-memory row was
    marked delivered anyway, so the row claimed a delivery the database never
    recorded. #2660 documents 2,045 signal_log writes lost to SQLite lock
    contention, so a lost write is not hypothetical.

    NOTE the logging is deliberately NOT one line per re-emission.
    ``_mark_wake_delivered`` is only reached when the ack supervisor observes
    ``Status.OK``; a re-dispatch that coalesces into an in-flight ack never
    gets there and leaves no log line. The storm is therefore visible in full
    only through ``wake_dispatch_count`` (#2774) — which is precisely what
    that column was added for. Both numbers are asserted exactly, so a change
    to either is caught rather than absorbed.
    """
    from kestrel_sovereign.features.restart_coordinator import feature as fm

    feat, backend, agent = await _real_dispatch_feature(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing", expected_current_status="pending",
    )
    await feat.initialize()

    async def _lost_write(db, request_id):
        """The write reports, truthfully, that it matched no row."""
        return False

    monkeypatch.setattr(fm, "mark_wake_delivered", _lost_write)

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            wake_tasks = await feat._reap_post_restart_rows()
            if wake_tasks:
                await asyncio.gather(*wake_tasks)
            await agent.drain_background_tasks()

    row = await get_request(backend, req.id)
    assert row.wake_delivered is False, (
        "the row must never claim a delivery the database did not record"
    )
    assert row.wake_dispatch_count == 3, (
        "the retry is intact and the row itself records the storm"
    )

    lost = [r for r in caplog.records if "matched no row" in r.getMessage()]
    assert len(lost) == 1, (
        f"exactly one dispatch reached the ack path; the other two coalesced "
        f"without reaching it (see docstring). Got {len(lost)} warnings"
    )
