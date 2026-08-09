"""Session binding for signal-fired cognition wakes (#2877).

A wake that carries no ``session_id`` does not land nowhere — it lands in a
session the conversation store *invents*. ``_derive_implicit_session_id``
reuses the previous message's session only while it is under 30 minutes old
and otherwise mints a fresh UUID, so an hour-long Talon job woke into a
two-message orphan session while the Sovereign's thread sat idle. That is why
the defect looked intermittent: a wake that happened to fire inside the
30-minute window inherited the right session by accident.

So every deferred-work registration point captures the session it was
registered from, the reconciler binds the wake to it, and a wake with no such
session is recorded as ``<status>_unbound`` rather than a bare ``ok``.
``unbound`` is deliberately weaker than "unsurfaced": the reconciler never
observes where an unbound wake landed, and
``test_unbound_wake_can_still_land_in_a_live_session`` below drives a REAL
conversation store to show it may well land in the user's thread. Both a bare
``ok`` and an "unsurfaced" verdict would state something nobody checked.

Covers:
  - ``resolve_origin_session_id`` precedence and its refusal to coerce
    non-string attributes into a session id
  - ``wait(target, mode="signal")`` recording the origin session in the ledger
  - the reconciler binding ledger / provider-supplied sessions onto the signal
    ENVELOPE (and stripping the key from the payload)
  - session-less wakes reported unbound, and the real implicit-session
    behaviour that makes a stronger claim unprovable
  - ``TalonWaitable`` surfacing the dispatching session off the durable job
    record
  - ``talon_claim`` routing a session-bound claim onto the durable CLI rail,
    which is the only Talon rail that can deliver the wake back
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sdk.signals import Status
from kestrel_sdk.tools import Outcome, WaitStatus

from kestrel_sovereign.session_origin import resolve_origin_session_id
from kestrel_sovereign.storage.async_wait_signal_store import WaitSignalStore
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import (
    ORIGIN_SESSION_KEY,
    UNBOUND_SUFFIX,
    WaitReconciler,
    register_wait_watch,
)

SESSION = "28248bca-2f8d-46c7-a0e6-43a93e828306"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _CapturingDispatcher:
    """Collects enqueued signals; the task resolves OK (harvested next tick)."""

    def __init__(self):
        self.signals = []

    async def enqueue_signal(self, signal):
        from kestrel_sdk.signals.models import SignalHandle, SignalResult

        self.signals.append(signal)

        async def _coro():
            return SignalResult(
                signal_id=signal.id, status=Status.OK, mode=signal.mode,
                duration_ms=1, error=None,
            )

        task = asyncio.create_task(_coro())
        await task
        return SignalHandle(signal_id=signal.id, task=task)


class _Provider:
    """Poll-only Waitable (no ``active_handles``) — reachable only through an
    explicit watch, which is exactly the ``wait(mode="signal")`` path."""

    signal = "wait.complete"

    def __init__(self, kind="task", monitorable=False):
        self.kind = kind
        self._states = {}
        if monitorable:
            # Bind the attribute so the reconciler's MonitorableWaitable
            # structural check sees an implicit auto-wake provider.
            self.active_handles = self._active_handles

    def set(self, handle, outcome=Outcome.DONE, *, summary="done", data=None):
        self._states[handle] = (outcome, summary, data or {})

    async def _active_handles(self):
        return list(self._states)

    async def poll(self, handle):
        entry = self._states.get(handle)
        if entry is None:
            return WaitStatus(Outcome.PENDING, f"pending {handle}", data={})
        outcome, summary, data = entry
        return WaitStatus(outcome, summary, data=dict(data))

    async def owns_handle(self, handle):
        return True


@pytest.fixture
def make_agent(tmp_path, sqlite_database_factory):
    async def create(provider=None, dispatcher=None, session_id=None):
        db = await sqlite_database_factory(tmp_path / "agent.db")
        registry = WaitRegistry()
        if provider is not None:
            registry.register(provider)
        agent = SimpleNamespace(
            did="did:test:agent",
            agent_id="did:test:agent",
            _raw_storage=SimpleNamespace(db=db),
            wait_registry=registry,
            dispatcher=dispatcher if dispatcher is not None else _CapturingDispatcher(),
            _active_session_id=session_id,
        )
        return agent

    return create


# ---------------------------------------------------------------------------
# resolve_origin_session_id
# ---------------------------------------------------------------------------


def test_active_session_wins_over_logging_contextvar():
    """The per-turn ``_active_session_id`` (set from the JSON body the primary
    chat path uses) beats the logging ContextVar (query param / header)."""
    from kestrel_sovereign.logging_config import session_id_var

    agent = SimpleNamespace(_active_session_id="body-session")
    token = session_id_var.set("header-session")
    try:
        assert resolve_origin_session_id(agent) == "body-session"
    finally:
        session_id_var.reset(token)


def test_falls_back_to_logging_contextvar():
    from kestrel_sovereign.logging_config import session_id_var

    agent = SimpleNamespace(_active_session_id=None)
    token = session_id_var.set("header-session")
    try:
        assert resolve_origin_session_id(agent) == "header-session"
    finally:
        session_id_var.reset(token)


def test_no_session_resolves_to_empty_string():
    """CLI/system-initiated callers have no observer thread — empty, never a
    substituted session of our own invention."""
    assert resolve_origin_session_id(SimpleNamespace()) == ""
    assert resolve_origin_session_id(None) == ""


def test_non_string_attribute_is_not_a_session():
    """A test double's auto-created attribute is not a session id; coercing it
    would bind the wake to a thread that does not exist."""
    from unittest.mock import MagicMock

    assert resolve_origin_session_id(MagicMock()) == ""
    assert resolve_origin_session_id(SimpleNamespace(_active_session_id=7)) == ""


# ---------------------------------------------------------------------------
# Registration captures the session (wait(target, mode="signal"))
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_wait_watch_records_origin_session(make_agent):
    agent = await make_agent(_Provider(kind="task"), session_id=SESSION)
    await register_wait_watch(agent, "task:t1")

    store = WaitSignalStore(agent._raw_storage.db, agent.did)
    row = await store.get("task", "t1")
    assert row.watching == 1
    assert row.origin_session_id == SESSION


@pytest.mark.asyncio
async def test_register_wait_watch_without_session_records_none(make_agent):
    agent = await make_agent(_Provider(kind="task"), session_id=None)
    await register_wait_watch(agent, "task:t1")

    store = WaitSignalStore(agent._raw_storage.db, agent.did)
    assert (await store.get("task", "t1")).origin_session_id is None


# ---------------------------------------------------------------------------
# The reconciler binds the wake to the originating session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watched_wake_resumes_the_registering_session(make_agent):
    """The core #2877 contract: a watch registered from a chat session wakes
    INTO that session instead of minting a fresh implicit one."""
    provider = _Provider(kind="task")
    provider.set("t1")
    agent = await make_agent(provider, session_id=SESSION)
    await register_wait_watch(agent, "task:t1")
    rec = agent._wait_reconciler

    t1 = await rec.reconcile()
    assert t1.data["signals_enqueued"] == 1
    sig = agent.dispatcher.signals[0]
    assert sig.session_id == SESSION

    t2 = await rec.reconcile()
    # Bound to a stated session → a clean "ok", no unbound marker.
    row = await rec._store.get("task", "t1")
    assert row.last_delivery_status == "ok"
    assert t2.data["transitions"][0]["delivery_status"] == "ok"
    assert t2.data["transitions"][0]["origin_session_id"] == SESSION
    assert t2.data["signals_unbound"] == 0
    assert "unbound" not in t2.confirmation


@pytest.mark.asyncio
async def test_provider_supplied_session_binds_the_implicit_wake(make_agent):
    """The auto-wake path registers no watch, so the session comes off the
    provider's WaitStatus.data (TalonWaitable reads it from the durable job
    record). It rides the ENVELOPE and is stripped from the payload."""
    provider = _Provider(kind="talon", monitorable=True)
    provider.set(
        "job-1",
        data={"status": "complete", ORIGIN_SESSION_KEY: SESSION},
    )
    agent = await make_agent(provider, session_id=None)
    rec = WaitReconciler(agent)

    await rec.reconcile()
    sig = agent.dispatcher.signals[0]
    assert sig.session_id == SESSION
    # Routing state belongs on the envelope, not in a rendered prompt.
    assert ORIGIN_SESSION_KEY not in sig.payload

    await rec.reconcile()
    row = await rec._store.get("talon", "job-1")
    assert row.last_delivery_status == "ok"
    assert row.origin_session_id == SESSION


@pytest.mark.asyncio
async def test_ledger_session_wins_over_provider_supplied(make_agent):
    """An explicit watch is the caller's stated binding, so it beats whatever
    the provider recorded at dispatch."""
    provider = _Provider(kind="talon", monitorable=True)
    provider.set(
        "job-1",
        data={"status": "complete", ORIGIN_SESSION_KEY: "dispatch-session"},
    )
    agent = await make_agent(provider, session_id="watch-session")
    await register_wait_watch(agent, "talon:job-1")
    rec = agent._wait_reconciler

    await rec.reconcile()
    assert agent.dispatcher.signals[0].session_id == "watch-session"


@pytest.mark.asyncio
async def test_sessionless_wake_is_reported_unbound(make_agent):
    """With no originating session the wake still fires, but the ledger records
    that it went out unbound instead of reporting a clean delivery."""
    provider = _Provider(kind="talon", monitorable=True)
    provider.set("job-1", data={"status": "complete"})
    agent = await make_agent(provider, session_id=None)
    rec = WaitReconciler(agent)

    await rec.reconcile()
    assert agent.dispatcher.signals[0].session_id is None

    t = await rec.reconcile()
    row = await rec._store.get("talon", "job-1")
    assert row.last_delivery_status == "ok" + UNBOUND_SUFFIX
    assert t.data["transitions"][0]["origin_session_id"] == ""
    # The tick's own report must not describe a possibly-stranded wake as a
    # clean delivery and nothing else — that misreport is what hid this for a
    # month.
    assert t.data["signals_emitted"] == 1
    assert t.data["signals_unbound"] == 1
    assert "unbound=1" in t.confirmation


@pytest.mark.asyncio
async def test_empty_provider_session_does_not_clear_the_watch_session(make_agent):
    """A provider that reports no session must not orphan a handle whose watch
    already recorded one — the store write is sticky."""
    provider = _Provider(kind="talon", monitorable=True)
    provider.set("job-1", data={"status": "complete", ORIGIN_SESSION_KEY: ""})
    agent = await make_agent(provider, session_id=SESSION)
    await register_wait_watch(agent, "talon:job-1")
    rec = agent._wait_reconciler

    await rec.reconcile()
    assert agent.dispatcher.signals[0].session_id == SESSION
    await rec.reconcile()
    row = await rec._store.get("talon", "job-1")
    assert row.origin_session_id == SESSION
    assert row.last_delivery_status == "ok"


# ---------------------------------------------------------------------------
# End-to-end: reconciler -> real dispatcher -> process_input
# ---------------------------------------------------------------------------


class _WakeRecordingAgent:
    """DispatcherAgent stand-in that records the session each wake ran in."""

    did = "did:test:agent"
    agent_id = "did:test:agent"

    def __init__(self, db, registry):
        self._raw_storage = SimpleNamespace(db=db)
        self.wait_registry = registry
        self.background_tasks = []
        self.wake_sessions = []

    async def process_input(self, prompt: str, session_id=None, **kwargs):
        self.wake_sessions.append(session_id)
        return "handled"

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.mark.asyncio
async def test_talon_wake_runs_in_the_dispatching_session(tmp_path, sqlite_database_factory):
    """The whole hop the issue is about: a Talon job dispatched from a chat
    session completes an hour later, and the cognition turn it wakes runs in
    THAT session — not a fresh one the conversation store invents."""
    from kestrel_sovereign.signals import (
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )
    from kestrel_sovereign.signals.sources.talon import (
        build_talon_job_complete_registration,
    )
    from kestrel_sovereign.storage.db import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    registry.register(build_talon_job_complete_registration())

    provider = _Provider(kind="talon", monitorable=True)
    provider.signal = "talon.job_complete"
    provider.set(
        "job-1",
        data={
            "job_id": "job-1",
            "status": "complete",
            ORIGIN_SESSION_KEY: SESSION,
        },
    )
    wait_registry = WaitRegistry()
    wait_registry.register(provider)
    agent = _WakeRecordingAgent(
        await sqlite_database_factory(tmp_path / "agent.db"), wait_registry,
    )
    agent.dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(),
        store=store,
    )
    try:
        await WaitReconciler(agent).reconcile()
        pending = [t for t in agent.background_tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        assert agent.wake_sessions == [SESSION]
    finally:
        await backend.close()


class _ConversationWritingAgent(_WakeRecordingAgent):
    """Same stand-in, but its turn actually PERSISTS through a real
    ``AsyncConversationStore`` — passing the dispatcher's ``session_id``
    straight through, which is what a real turn body does. With ``None`` the
    store runs its own ``_derive_implicit_session_id`` heuristic, so this
    exercises the production routing an unbound wake really gets."""

    def __init__(self, db, registry, conversation):
        super().__init__(db, registry)
        self.conversation = conversation

    async def process_input(self, prompt: str, session_id=None, **kwargs):
        self.wake_sessions.append(session_id)
        await self.conversation.add_conversation(
            "assistant", "woke and worked", session_id=session_id,
        )
        return "handled"


@pytest.mark.asyncio
async def test_unbound_wake_can_still_land_in_a_live_session(
    tmp_path, sqlite_database_factory,
):
    """``_unbound`` must NOT be read as "nobody saw it".

    This is the 21:59 counterexample from #2877, reproduced against the real
    ``AsyncConversationStore``: a wake with no bound session, fired while the
    user's thread is still inside the 30-minute reuse window, is persisted INTO
    that thread. So the honest ledger claim is that the wake went out unbound
    (which is checked), not that it was stranded (which would be false here).
    """
    import json

    from kestrel_sovereign.signals import (
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )
    from kestrel_sovereign.signals.sources.talon import (
        build_talon_job_complete_registration,
    )
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )
    from kestrel_sovereign.storage.db import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    registry = SourceRegistry()
    registry.register(build_talon_job_complete_registration())

    db = await sqlite_database_factory(tmp_path / "agent.db")
    conversation = AsyncConversationStore(db, agent_id="did:test:agent")
    # The Sovereign's live thread: a message that just landed, so the store's
    # gap heuristic will reuse its session for the next unbound write.
    await conversation.add_conversation("user", "still here", session_id=SESSION)

    # No ORIGIN_SESSION_KEY anywhere: not on the provider, not in the ledger.
    provider = _Provider(kind="talon", monitorable=True)
    provider.signal = "talon.job_complete"
    provider.set("job-1", data={"job_id": "job-1", "status": "complete"})
    wait_registry = WaitRegistry()
    wait_registry.register(provider)
    agent = _ConversationWritingAgent(db, wait_registry, conversation)
    agent.dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(),
        store=log_store,
    )
    try:
        rec = WaitReconciler(agent)
        await rec.reconcile()
        pending = [t for t in agent.background_tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        # The dispatcher genuinely called process_input with no session...
        assert agent.wake_sessions == [None]
        # ...and the real store still routed the turn into the live thread.
        rows = await db.fetchall(
            "SELECT metadata FROM conversation_history "
            "WHERE agent_id = ? ORDER BY id",
            ("did:test:agent",),
        )
        sessions = [json.loads(r[0])["session_id"] for r in rows]
        assert sessions == [SESSION, SESSION], (
            "an unbound wake inside the reuse window lands in the live "
            "session, so the reconciler cannot claim it was unsurfaced"
        )

        t = await rec.reconcile()
        row = await rec._store.get("talon", "job-1")
        # The ledger reports the binding gap (true, and knowable here) — not a
        # clean "ok" and not a stranded verdict, both of which this run
        # disproves in one direction or the other.
        assert row.last_delivery_status == "ok" + UNBOUND_SUFFIX
        assert t.data["signals_unbound"] == 1
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# Talon: the dispatching session travels on the durable job record
# ---------------------------------------------------------------------------


class _FakeTalonFeature:
    """The slice of TalonCoordinatorFeature that TalonWaitable.poll touches."""

    def __init__(self, jobs):
        self._jobs = jobs

    def _reload_persisted_jobs(self):
        return None

    def _reap_cli_job(self, info):
        return False

    def _persist_jobs(self):
        return True

    def _tail_job_log(self, path, lines=20):
        return ""

    def _discover_host_url(self):
        return None


@pytest.mark.asyncio
async def test_talon_waitable_surfaces_the_dispatching_session():
    from kestrel_sovereign.features.talon.wait_provider import TalonWaitable

    feature = _FakeTalonFeature({
        "job-1": {
            "method": "cli_background",
            "status": "complete",
            "returncode": 0,
            "origin_session_id": SESSION,
        },
    })
    status = await TalonWaitable(feature).poll("job-1")
    assert status.outcome is Outcome.DONE
    assert status.data[ORIGIN_SESSION_KEY] == SESSION


@pytest.mark.asyncio
async def test_talon_waitable_legacy_job_has_no_session():
    """A job record written before #2877 carries no session — honest empty,
    which the reconciler reports as unbound rather than inventing one."""
    from kestrel_sovereign.features.talon.wait_provider import TalonWaitable

    feature = _FakeTalonFeature({
        "job-1": {"method": "cli_background", "status": "failed", "returncode": 1},
    })
    status = await TalonWaitable(feature).poll("job-1")
    assert status.data[ORIGIN_SESSION_KEY] == ""


@pytest.mark.asyncio
async def test_cli_dispatch_records_the_session_durably(tmp_path, monkeypatch):
    """Dispatch is the only moment the originating session is knowable, and a
    Talon job routinely outlives the process — so it must survive the durable
    jobs.json round trip, not just live in memory."""
    from unittest.mock import patch

    from kestrel_sovereign.features.talon.coordinator import (
        TalonCoordinatorFeature,
    )

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_bin = tmp_path / "kestrel-talon"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)

    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        _active_session_id=SESSION,
    )
    feature = TalonCoordinatorFeature(agent)
    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        result = await feature._dispatch_via_cli_background(
            ["claim", "--repo", "x/y", "--issue", "1"],
            label="claim:x/y#1",
            extra_meta={"repo": "x/y", "issue": 1},
        )

    job_id = result["job_id"]
    try:
        assert feature._jobs[job_id]["origin_session_id"] == SESSION
        # Simulate the restart the wake usually happens after.
        feature._jobs.clear()
        feature._reload_persisted_jobs()
        assert feature._jobs[job_id]["origin_session_id"] == SESSION
    finally:
        proc = result.get("process") or feature._jobs.get(job_id, {}).get("process")
        if proc is not None:
            await proc.wait()


# ---------------------------------------------------------------------------
# talon_claim transport selection: only the CLI rail can deliver the wake back
# ---------------------------------------------------------------------------


def _claim_agent(session_id):
    """MagicMock agent, as the other talon_claim tests use, with an explicit
    ``_active_session_id``. Note a bare MagicMock's auto-attribute is NOT a
    str, which ``resolve_origin_session_id`` refuses — that is what keeps
    session-less fixtures on the unchanged A2A-preferred path."""
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.agent_name = "kestrel"
    agent._features = []
    agent._active_session_id = session_id
    return agent


def _ready_workspace(tmp_path):
    return {
        "repo": "org/repo", "path": str(tmp_path / "org__repo"),
        "exists": True, "is_git": True, "head": "main", "clean": True,
        "last_fetch_at": None, "safe": True,
    }


@pytest.mark.asyncio
async def test_session_bound_claim_takes_the_durable_cli_rail(tmp_path, monkeypatch):
    """A claim filed FROM a chat session must not go out over A2A (#2877).

    A2A job rows never reach ``_persist_jobs`` and are excluded from
    ``TalonWaitable.active_handles``, so that rail has no durable auto-wake to
    bind a session to — dispatching there would strand the completion outside
    the thread that asked for it. Only ``cli_background`` carries the binding.
    """
    from unittest.mock import AsyncMock, patch

    from kestrel_sovereign.features.talon.coordinator import (
        TalonCoordinatorFeature,
    )

    monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    feature = TalonCoordinatorFeature(_claim_agent(SESSION))

    with patch.object(feature, "_dispatch_via_a2a", new_callable=AsyncMock) as a2a, \
         patch.object(
             feature, "_dispatch_via_cli_background", new_callable=AsyncMock,
         ) as cli, \
         patch.object(
             TalonCoordinatorFeature, "_workspace_state",
             return_value=_ready_workspace(tmp_path),
         ):
        cli.return_value = {
            "dispatched": True, "method": "cli_background",
            "job_id": "abc", "pid": 1234,
        }
        result = await feature.talon_claim(repo="org/repo", issue=42)

    a2a.assert_not_awaited()
    cli.assert_awaited_once()
    assert result.data["method"] == "cli_background"


@pytest.mark.asyncio
async def test_sessionless_claim_still_prefers_a2a(tmp_path, monkeypatch):
    """The gate is narrow: a cron/system-filed claim has no thread to strand,
    so the A2A-preferred path is untouched."""
    from unittest.mock import AsyncMock, patch

    from kestrel_sovereign.features.talon.coordinator import (
        TalonCoordinatorFeature,
    )

    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    feature = TalonCoordinatorFeature(_claim_agent(None))

    with patch.object(feature, "_dispatch_via_a2a", new_callable=AsyncMock) as a2a:
        a2a.return_value = {
            "dispatched": True, "method": "a2a",
            "task_id": "abc", "repo": "org/repo", "issue": 42,
        }
        result = await feature.talon_claim(repo="org/repo", issue=42)

    a2a.assert_awaited_once_with("org/repo", 42)
    assert result.data["method"] == "a2a"


@pytest.mark.asyncio
async def test_a2a_job_record_carries_no_dead_session_field(tmp_path, monkeypatch):
    """The A2A row must not advertise a binding it cannot honour: nothing
    reads it (a2a handles are not in ``active_handles``) and nothing persists
    it (``_persist_jobs`` writes cli_background rows only)."""
    from unittest.mock import patch

    from kestrel_sovereign.features.talon.coordinator import (
        TalonCoordinatorFeature,
    )

    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    agent = _claim_agent(SESSION)
    # A real path, so the durable-registry assertions below hit this test's own
    # tmp dir rather than the shared /tmp stub fallback.
    agent.storage_path = str(tmp_path / "agent_data" / "kestrel_prime.db")
    feature = TalonCoordinatorFeature(agent)

    with patch.object(
        feature, "_discover_host_url", return_value="http://localhost:8888",
    ), patch(
        "kestrel_sovereign.features.talon.coordinator.urllib.request.urlopen",
    ) as urlopen:
        urlopen.return_value.read.return_value = b"{}"
        result = await feature._dispatch_via_a2a("org/repo", 42)

    row = feature._jobs[result["task_id"]]
    assert ORIGIN_SESSION_KEY not in row
    # And the durable registry never sees the row at all, which is exactly why
    # a session-bound claim is routed elsewhere.
    feature._persist_jobs()
    feature._jobs.clear()
    feature._reload_persisted_jobs()
    assert feature._jobs == {}
