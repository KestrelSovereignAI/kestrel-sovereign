"""#2877 end-to-end: a Talon job's completion wake lands in — and surfaces
in — the chat session that dispatched it.

The reported failure had two halves, and fixing only one leaves the bug:

  1. *Persisted in the wrong place.* The job record carried no originating
     session, so the reconciler had none to put on the wake envelope, so the
     dispatcher called ``process_input`` with no session and "no session"
     means "open a fresh one". Every wake became message 1 of a brand-new
     session, and a multi-attempt autonomous loop walked away from the user's
     thread one session per hop while ``delivery_status`` still read ``ok``.

  2. *Persisted in the right place but never surfaced.* The reconciler built
     the wake ``INTERNAL``, so the dispatcher log-only'd it and never emitted
     the ``signal_completed`` SSE event — the open chat stayed blank until a
     manual refresh even once the turn landed in the correct session.

So these tests deliberately avoid stubbing the seam under test. The
dispatcher, its source registry, the ``talon.job_complete`` registration and
its on-disk prompt template, the wait reconciler, the ``TalonWaitable``
provider and the coordinator's job registry are all REAL; only the agent is a
double, and it records exactly the two things the user experiences — which
session ``process_input`` was given, and what reached ``emit_event``.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sdk.signals import Visibility
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
from kestrel_sovereign.features.talon.coordinator import TalonCoordinatorFeature
from kestrel_sovereign.features.talon.wait_provider import TalonWaitable
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
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import WaitReconciler


class _RecordingAgent:
    """Minimal DispatcherAgent that records what the user would observe.

    ``process_input`` declares ``**kwargs`` so the dispatcher's signature
    inspection passes ``session_id`` through — the production
    ``KestrelAgent.process_input`` accepts it, and a double that did not
    would silently hide the very binding under test.
    """

    did = "did:test:2877"
    agent_name = "kestrel"

    def __init__(self, response: str = "Attempt 4 dispatched."):
        self.background_tasks: list[asyncio.Task] = []
        self.process_input_sessions: list[object] = []
        self.emitted: list[tuple[str, dict]] = []
        self._response = response

    async def process_input(self, prompt: str, **kwargs):
        self.process_input_sessions.append(kwargs.get("session_id"))
        return self._response

    async def emit_event(self, event_type: str, data: dict) -> None:
        self.emitted.append((event_type, data))

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


class _CoordinatorAgent(TurnLifecycleMixin):
    """The agent the coordinator dispatches from, with the REAL turn lifecycle.

    Not a MagicMock: the origin capture asks the lifecycle which chat session
    the *calling task* owns, and only the real ContextVar/lock machinery can
    answer that honestly (a stubbed getter would pass whatever the test wants).
    """

    agent_name = "kestrel"
    did = "did:test:2877"

    def __init__(self, storage_path):
        self._features = []
        self.storage_path = str(storage_path)
        self._active_session_id = None
        self._live_turn_id = None


@asynccontextmanager
async def _chat_turn(agent: _CoordinatorAgent, session_id: str):
    """A turn as production runs it: lifecycle assigns the turn id, the turn
    body assigns the session."""
    async with agent._turn_lifecycle():
        agent._active_session_id = session_id
        yield


async def _drain(agent: _RecordingAgent) -> None:
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.fixture
async def rig(tmp_path, sqlite_database_factory):
    """A real dispatcher + real reconciler over a real Talon job registry."""
    backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()

    registry = SourceRegistry()
    registry.register(build_talon_job_complete_registration())
    agent = _RecordingAgent()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )

    # The coordinator's job registry, backed by a real on-disk jobs.json.
    coordinator_agent = _CoordinatorAgent(tmp_path / "kestrel_prime.db")
    feature = TalonCoordinatorFeature(coordinator_agent)
    feature._tail_job_log = lambda path, lines=20: ""

    provider = TalonWaitable(feature)
    wait_registry = WaitRegistry()
    wait_registry.register(provider)

    db = await sqlite_database_factory(tmp_path / "agent.db")
    reconciler_agent = SimpleNamespace(
        did=agent.did,
        agent_id=agent.did,
        _raw_storage=SimpleNamespace(db=db),
        wait_registry=wait_registry,
        dispatcher=dispatcher,
    )
    reconciler = WaitReconciler(reconciler_agent)

    yield SimpleNamespace(
        agent=agent,
        feature=feature,
        coordinator_agent=coordinator_agent,
        reconciler=reconciler,
        backend=backend,
    )

    await _drain(agent)
    await backend.close()


def _finished_job(origin_session_id: str) -> dict:
    return {
        "method": "cli_background",
        "status": "complete",
        "returncode": 0,
        "label": "claim:org/repo#2877",
        "repo": "org/repo",
        "issue": 2877,
        "log_path": "",
        "started_at": "2026-08-10T00:00:00+00:00",
        "completed_at": "2026-08-10T01:00:00+00:00",
        "origin_session_id": origin_session_id,
    }


@pytest.mark.asyncio
async def test_wake_turn_runs_in_the_dispatching_session(rig):
    """The whole point: the cognition turn resumes the originating session
    instead of minting a new one."""
    rig.feature._jobs["job-1"] = _finished_job("chat-sess-1")

    await rig.reconciler.reconcile()
    await _drain(rig.agent)

    assert rig.agent.process_input_sessions == ["chat-sess-1"], (
        "the woken turn must run in the session that dispatched the job; a "
        "None here is the reported bug — process_input opens a fresh session"
    )


@pytest.mark.asyncio
async def test_bound_wake_emits_signal_completed_with_session_and_body(rig):
    """Landing in the right session is not enough — the open chat only paints
    the turn when the dispatcher emits ``signal_completed``, which requires a
    non-INTERNAL signal, and the frontend only renders it with a non-empty
    ``result_summary``. Both must be true on the real rail."""
    rig.feature._jobs["job-1"] = _finished_job("chat-sess-1")

    await rig.reconciler.reconcile()
    await _drain(rig.agent)

    emits = [e for e in rig.agent.emitted if e[0] == "signal_completed"]
    assert len(emits) == 1, (
        "an INTERNAL wake is log-only: no SSE event, and the open chat stays "
        "blank until a manual refresh"
    )
    payload = emits[0][1]
    assert payload["session_id"] == "chat-sess-1"
    assert payload["visibility"] == "user_visible"
    assert payload["mode"] == "cognition"
    assert payload["source"] == "talon.job_complete"
    assert payload["result_summary"] == "Attempt 4 dispatched.", (
        "the frontend paints result_summary; a metadata-only emit renders "
        "nothing (talon.job_complete had no result_summary callback)"
    )


@pytest.mark.asyncio
async def test_unattended_job_still_wakes_system_initiated(rig):
    """A job dispatched with no originating session (CLI/scheduler) must wake
    exactly as before #2877: a fresh session, and no SSE emit into whichever
    pane happens to be open."""
    rig.feature._jobs["job-2"] = _finished_job("")

    await rig.reconciler.reconcile()
    await _drain(rig.agent)

    assert rig.agent.process_input_sessions == [None]
    assert [e for e in rig.agent.emitted if e[0] == "signal_completed"] == []


@pytest.mark.asyncio
async def test_binding_survives_a_restart_of_the_coordinator(rig, tmp_path):
    """The origin is persisted with the job record, so a wake that fires after
    a restart still resumes the originating session rather than the reloaded
    job silently reverting to unattended."""
    rig.feature._jobs["job-3"] = _finished_job("chat-sess-3")
    assert rig.feature._persist_jobs() is True
    # Simulate the restart: drop the in-memory map, keep the durable registry.
    rig.feature._jobs = {}

    await rig.reconciler.reconcile()
    await _drain(rig.agent)

    assert rig.agent.process_input_sessions == ["chat-sess-3"]


@pytest.mark.asyncio
async def test_claim_from_a_chat_turn_wakes_that_chat_turn(
    rig, tmp_path, monkeypatch,
):
    """The user-visible loop, driven from the top: ``talon_claim`` as a chat
    turn calls it, through the real dispatch funnel, to the wake.

    Nothing between the tool call and the SSE emit is stubbed except the
    talon binary itself (a shell script that exits 0) and the workspace probe:
    the rail choice, the origin capture, the durable job registry, the
    reconciler, the source registration and the dispatcher are all real. The
    earlier tests seed ``_jobs`` by hand, which would keep passing even if
    ``talon_claim`` never recorded an origin — or sent the claim down the A2A
    rail, which cannot deliver a session-bound wake at all.
    """
    monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KESTREL_TALON_CWD", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_talon = tmp_path / "kestrel-talon-fake"
    fake_talon.write_text("#!/bin/sh\nexit 0\n")
    fake_talon.chmod(0o755)
    monkeypatch.setenv("KESTREL_TALON_BIN", str(fake_talon))

    ready_workspace = {
        "repo": "org/repo", "path": str(tmp_path / "org__repo"),
        "exists": True, "is_git": True, "head": "main", "clean": True,
        "last_fetch_at": None, "safe": True,
    }
    with patch.object(
        rig.feature, "_dispatch_via_a2a", new_callable=AsyncMock,
    ) as mock_a2a, patch.object(
        TalonCoordinatorFeature, "_workspace_state", return_value=ready_workspace,
    ):
        async with _chat_turn(rig.coordinator_agent, "chat-sess-live"):
            result = await rig.feature.talon_claim(repo="org/repo", issue=2877)

    assert result.status is ToolResultStatus.OK
    assert result.data["method"] == "cli_background", (
        "a session-bound claim must take the durable rail — the A2A rail "
        "creates the task on the recipient and produces no session-bound wake"
    )
    mock_a2a.assert_not_awaited()

    # Let the (immediately-exiting) job finish so the reconciler sees it
    # terminal, exactly as the cron tick would.
    job_id = result.data["job_id"]
    await rig.feature._jobs[job_id]["process"].wait()

    await rig.reconciler.reconcile()
    await _drain(rig.agent)

    assert rig.agent.process_input_sessions == ["chat-sess-live"]
    emits = [e for e in rig.agent.emitted if e[0] == "signal_completed"]
    assert len(emits) == 1
    assert emits[0][1]["session_id"] == "chat-sess-live"
    assert emits[0][1]["result_summary"] == "Attempt 4 dispatched."


@pytest.mark.asyncio
async def test_signal_envelope_is_schema_valid_and_bound(rig):
    """Belt and braces on the envelope itself: the reconciler's signal must
    survive the real source schema (``origin_session_id`` is defaulted, not
    required) and carry the binding on both the envelope and the payload."""
    rig.feature._jobs["job-4"] = _finished_job("chat-sess-4")

    captured: list = []
    original = rig.reconciler._agent.dispatcher.enqueue_signal

    async def _spy(signal):
        captured.append(signal)
        return await original(signal)

    rig.reconciler._agent.dispatcher.enqueue_signal = _spy
    await rig.reconciler.reconcile()
    await _drain(rig.agent)

    assert len(captured) == 1
    sig = captured[0]
    assert sig.source == "talon.job_complete"
    assert sig.session_id == "chat-sess-4"
    assert sig.visibility == Visibility.USER_VISIBLE
    assert sig.payload["origin_session_id"] == "chat-sess-4"
