"""#3663: a signal-mode wait wakes the session that registered it.

On 2026-10-15 an agent registered ``wait("ci:<pr>", mode="signal")`` from its
chain session. The wake came back with no session, opened a new one, and the
agent merged and deployed the PR there. Its chain session never saw that
turn, so on its next turn it had no record of it. The reconciler took a wake's
session only from ``provider.origin_session_id(handle)``, and only Talon's
provider has that method, so every ``ci:``, ``task:``, ``a2a:`` and
``workflow:`` wake was unbound: the gap #3429 (one session per chain) left for
non-Talon waits.

The watch row now records the registering turn's session, and the reconciler
binds the wake to it, asking the provider only when the row stores none.

The end-to-end tests run the real ``wait`` tool inside a real turn lifecycle,
the real reconciler and store over SQLite, the real dispatcher and
``wait.complete`` source, the real conversation store, and the real
``CIWaitable`` and ``TaskWaitable`` over fabricated GitHub and task-store
reads. Only the agent body is a double. Like the #3429 tests' double, it does
what ``KestrelAgent.process_input`` does with its ``session_id``: binds it as
the turn's session and writes every row under it. The provider-precedence
tests use the reconciler suite's capturing dispatcher.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sdk.signals import Visibility
from kestrel_sdk.tools import Outcome
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
from kestrel_sovereign.features.scheduler.ci_wait_provider import CIWaitable
from kestrel_sovereign.features.tasks.feature import TaskFeature
from kestrel_sovereign.features.tasks.wait_provider import TaskWaitable
from kestrel_sovereign.features.wait.feature import WaitFeature
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.wait import (
    SOURCE_NAME as WAIT_SOURCE,
    build_wait_complete_registration,
)
from kestrel_sovereign.signals.sources.wait_replay import (
    build_wait_replay_registration,
)
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.session_id_column import is_stampable_session_id
from kestrel_sovereign.storage.turn_provenance import turn_opener_metadata
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import (
    VISIBILITY_UNBOUND,
    WaitReconciler,
    register_wait_watch,
)
from tests.unit import test_wait_reconciler as doubles
from tests.unit import test_wait_watch_rearm as ci

AGENT_DID = "did:test:3663"
# The chain session the 2026-10-15 wait was registered from.
CHAIN = "chain-e25e6cd4"
OTHER = "operator-chat-2"
CI_HANDLE = "o/r#3659"
CI_REF = f"ci:{CI_HANDLE}"


class _WaitingAgent(TurnLifecycleMixin, OrchestratorEngineMixin, EventManagerMixin):
    """An agent whose turns are the session contract and nothing else."""

    did = AGENT_DID
    agent_name = "kestrel"

    def __init__(self, conversation: AsyncConversationStore, db: AsyncDatabase):
        self.agent_id = AGENT_DID
        self.conversation = conversation
        self._raw_storage = SimpleNamespace(db=db)
        self.wait_registry = WaitRegistry()
        self._active_session_id = None
        self._event_listeners: list = []
        self._pending_task_notifications: list = []
        self.background_tasks: list[asyncio.Task] = []
        # The session each wake turn ran in.
        self.wake_sessions: list = []

    async def process_input(
        self, prompt: str, session_id=None, signal_wake=None, **_kwargs
    ):
        async with self._turn_lifecycle():
            self._active_session_id = session_id
            self.wake_sessions.append(session_id)
            user_meta = {"sent_form": True, **turn_opener_metadata(signal_wake)}
            await self.conversation.add_conversation(
                "user", prompt, metadata=user_meta, session_id=session_id
            )
            await self.conversation.add_conversation(
                "assistant", "wake handled", session_id=session_id
            )
            return "wake handled"

    async def turn_in(self, session_id: str, work):
        """Run ``work`` inside a live turn bound to ``session_id``: where a
        model's tool call runs."""
        async with self._turn_lifecycle():
            self._active_session_id = session_id
            return await work()

    def _track_background_task(self, coro, *, name: str, admission=None):
        task = asyncio.create_task(coro, name=name)
        if admission is not None:
            # The dispatcher's runtime-owner heartbeat passes an admission
            # receipt, as the agent's own tracker records (#3570).
            admission.task = task
        self.background_tasks.append(task)
        return task


@pytest.fixture
async def boot(tmp_path, monkeypatch):
    """Start the agent over this test's databases. Starting again after
    ``stop()`` is a restart: new process state, the same durable rows."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    running: list = []

    async def start():
        db = await AsyncDatabase.sqlite(str(tmp_path / "agent.db"))
        backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
        await backend.connect()
        log = SignalLogStore(backend)
        await log.initialize()
        sources = SourceRegistry()
        sources.register(build_wait_complete_registration())
        sources.register(build_wait_replay_registration())

        conversation = AsyncConversationStore(db, agent_id=AGENT_DID)
        agent = _WaitingAgent(conversation, db)
        locks = OrderedLockManager()
        agent._lock_manager = locks
        dispatcher = SignalDispatcher(
            agent=agent, registry=sources, lock_manager=locks, store=log,
        )
        agent.dispatcher = dispatcher
        wakes: list = []
        enqueue = dispatcher.enqueue_signal

        async def capture(signal):
            wakes.append(signal)
            return await enqueue(signal)

        dispatcher.enqueue_signal = capture
        events: list = []

        async def listen(event_type, data):
            events.append((event_type, data))

        agent.add_event_listener(listen)
        agent._wait_reconciler = WaitReconciler(agent)
        wait = WaitFeature(agent=None)
        wait.agent = agent

        rig = SimpleNamespace(
            agent=agent,
            conversation=conversation,
            wait=wait,
            wakes=wakes,
            events=events,
            store=agent._wait_reconciler._store,
        )

        async def stop():
            running.remove(rig)
            pending = [t for t in agent.background_tasks if not t.done()]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await backend.close()
            await db.close()

        rig.stop = stop
        running.append(rig)
        return rig

    yield start

    for rig in list(running):
        await rig.stop()


def _serve_ci(rig, github) -> None:
    provider = CIWaitable(feature=None)
    provider._fetch = github.fetch
    rig.agent.wait_registry.register(provider)


async def _serve_tasks(rig, statuses: dict) -> None:
    """The real ``task`` provider, through TaskFeature, over a stubbed
    task-store read."""
    feature = TaskFeature(agent=None)
    feature.agent = rig.agent
    await feature.post_all_features_loaded(rig.agent)

    async def read(task_id):
        return {
            "ok": True, "task_id": task_id, "status": statuses[task_id],
            "task_type": "demo", "artifacts": [], "message": "done",
        }

    feature._get_task_status_data = read


async def _settle(rig) -> None:
    """Detect and enqueue, let the wake's turn run, then harvest it."""
    reconciler = rig.agent._wait_reconciler
    await reconciler.reconcile()
    pending = [t for t in rig.agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending)
    await reconciler.reconcile()


async def _register_in(rig, session_id: str, ref: str):
    result = await rig.agent.turn_in(
        session_id, lambda: rig.wait.wait(target=ref, mode="signal")
    )
    assert result.status is ToolResultStatus.OK, result.error
    return result


async def _row_sessions(conversation: AsyncConversationStore) -> set:
    rows = await conversation.get_conversation_history(limit=50)
    return {(row.get("metadata") or {}).get("session_id") for row in rows}


def _surfaced(rig) -> list:
    return [data for kind, data in rig.events if kind == "signal_completed"]


# ---------------------------------------------------------------------------
# Non-Talon kinds return to the registering session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_ci_wait_registered_from_a_session_wakes_that_session(boot):
    """The 2026-10-15 sequence, with the fix."""
    assert not hasattr(CIWaitable, "origin_session_id"), (
        "the CI provider records nothing about who asked, so the binding can "
        "only come from the watch"
    )
    rig = await boot()
    github = ci._GitHub()
    _serve_ci(rig, github)
    github.runs(ci._run(101, status="in_progress", conclusion=None))

    await _register_in(rig, CHAIN, CI_REF)
    assert (await rig.store.get("ci", CI_HANDLE)).watch_origin_session_id == CHAIN
    await _settle(rig)
    assert rig.wakes == [], "CI is still running"

    github.runs(ci._run(101, conclusion="success"))
    await _settle(rig)

    [wake] = rig.wakes
    assert wake.source == WAIT_SOURCE
    assert wake.payload["origin_session_id"] == CHAIN
    assert wake.session_id == CHAIN
    assert wake.visibility == Visibility.USER_VISIBLE
    # The turn lands in the chain session, and everything it wrote is there.
    assert rig.agent.wake_sessions == [CHAIN]
    assert await _row_sessions(rig.conversation) == {CHAIN}
    # ...and the open chain pane is handed something to paint (#2877 rule 1:
    # bind AND surface).
    [event] = _surfaced(rig)
    assert event["session_id"] == CHAIN
    assert event["result_summary"] == "wake handled"
    assert (await rig.store.get("ci", CI_HANDLE)).last_delivery_status == "ok_queued"


@pytest.mark.asyncio
async def test_a_task_wait_registered_from_a_session_wakes_that_session(boot):
    assert not hasattr(TaskWaitable, "origin_session_id")
    rig = await boot()
    statuses = {"t-42": "working"}
    await _serve_tasks(rig, statuses)

    await _register_in(rig, CHAIN, "task:t-42")
    await _settle(rig)
    assert rig.wakes == []

    statuses["t-42"] = "completed"
    await _settle(rig)

    [wake] = rig.wakes
    assert wake.payload["kind"] == "task"
    assert wake.payload["origin_session_id"] == CHAIN
    assert wake.session_id == CHAIN
    assert rig.agent.wake_sessions == [CHAIN]
    assert await _row_sessions(rig.conversation) == {CHAIN}


@pytest.mark.asyncio
async def test_a_watch_registered_from_another_session_wakes_there(boot):
    """The wake goes to the session that is waiting. Registering from a
    second session moves the watch there."""
    rig = await boot()
    github = ci._GitHub()
    _serve_ci(rig, github)
    github.runs(ci._run(101, status="in_progress", conclusion=None))

    await _register_in(rig, CHAIN, CI_REF)
    await _register_in(rig, OTHER, CI_REF)
    github.runs(ci._run(101, conclusion="failure"))
    await _settle(rig)

    assert [w.session_id for w in rig.wakes] == [OTHER]
    assert rig.agent.wake_sessions == [OTHER]


# ---------------------------------------------------------------------------
# Restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_watch_armed_before_a_restart_still_wakes_its_session(boot):
    """The session is on the durable row, not in process memory."""
    github = ci._GitHub()
    github.runs(ci._run(101, status="in_progress", conclusion=None))
    before = await boot()
    _serve_ci(before, github)
    await _register_in(before, CHAIN, CI_REF)
    await _settle(before)
    await before.stop()

    after = await boot()
    _serve_ci(after, github)
    github.runs(ci._run(101, conclusion="success"))
    await _settle(after)

    [wake] = after.wakes
    assert wake.session_id == CHAIN
    assert after.agent.wake_sessions == [CHAIN]


@pytest.mark.asyncio
async def test_rearming_after_a_restart_from_outside_a_session_keeps_it(boot):
    """A watch re-armed where no session is bound (a restarted host's
    unattended work, a workflow stage) keeps the session that asked."""
    github = ci._GitHub()
    github.runs(ci._run(101))  # failed
    before = await boot()
    _serve_ci(before, github)
    await _register_in(before, CHAIN, CI_REF)
    await _settle(before)
    assert [w.session_id for w in before.wakes] == [CHAIN]
    await before.stop()

    after = await boot()
    _serve_ci(after, github)
    github.runs(ci._run(102, status="queued", conclusion=None))  # the re-run
    result = await after.wait.wait(target=CI_REF, mode="signal")  # no turn
    assert result.status is ToolResultStatus.OK
    assert (await after.store.get("ci", CI_HANDLE)).watch_origin_session_id == CHAIN
    await _settle(after)
    github.runs(ci._run(102))  # the re-run fails too
    await _settle(after)

    [wake] = after.wakes
    assert wake.session_id == CHAIN
    assert after.agent.wake_sessions == [CHAIN]


# ---------------------------------------------------------------------------
# A watch with no stored session behaves as before
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_watch_registered_outside_any_session_stays_unbound(boot):
    rig = await boot()
    github = ci._GitHub()
    _serve_ci(rig, github)
    github.runs(ci._run(101, conclusion="success"))

    result = await rig.wait.wait(target=CI_REF, mode="signal")
    assert result.status is ToolResultStatus.OK
    assert (await rig.store.get("ci", CI_HANDLE)).watch_origin_session_id is None
    await _settle(rig)

    [wake] = rig.wakes
    assert wake.session_id is None
    assert wake.visibility == Visibility.INTERNAL
    assert wake.payload["origin_session_id"] == ""
    # #3429 still gives the turn a session of its own, never the chain's.
    [turn_session] = rig.agent.wake_sessions
    assert is_stampable_session_id(turn_session)
    assert turn_session != CHAIN
    assert _surfaced(rig) == []
    assert (
        await rig.store.get("ci", CI_HANDLE)
    ).last_delivery_status == f"ok_{VISIBILITY_UNBOUND}"


@pytest.mark.asyncio
async def test_a_task_detached_from_a_finished_turn_records_no_session(boot):
    """The session comes from the live turn, never from a task that merely
    inherited a finished turn's context (SIGNAL_SOURCES_GUIDE rule 3)."""
    rig = await boot()
    github = ci._GitHub()
    _serve_ci(rig, github)
    github.runs(ci._run(101, status="in_progress", conclusion=None))
    release = asyncio.Event()

    async def later():
        await release.wait()
        return await rig.wait.wait(target=CI_REF, mode="signal")

    async def spawn():
        return asyncio.create_task(later())

    detached = await rig.agent.turn_in(CHAIN, spawn)
    release.set()
    result = await detached

    assert result.status is ToolResultStatus.OK
    assert (await rig.store.get("ci", CI_HANDLE)).watch_origin_session_id is None


@pytest.mark.asyncio
async def test_unattended_work_never_takes_a_concurrent_turns_session(boot):
    """A registration from outside the live turn — a cron tick, a workflow
    stage — must not borrow the session of whichever chat turn happens to be
    in flight. The agent-global ``_active_session_id`` names that turn's
    session; the turn-bound accessor does not."""
    rig = await boot()
    github = ci._GitHub()
    _serve_ci(rig, github)
    github.runs(ci._run(101, status="in_progress", conclusion=None))
    in_turn = asyncio.Event()
    finish = asyncio.Event()

    async def hold():
        in_turn.set()
        await finish.wait()

    other_turn = asyncio.create_task(rig.agent.turn_in(OTHER, hold))
    await in_turn.wait()
    assert rig.agent._active_session_id == OTHER, "the other turn is live"
    try:
        result = await rig.wait.wait(target=CI_REF, mode="signal")
    finally:
        finish.set()
        await other_turn

    assert result.status is ToolResultStatus.OK
    assert (await rig.store.get("ci", CI_HANDLE)).watch_origin_session_id is None


# ---------------------------------------------------------------------------
# Precedence over the provider's own record (Talon)
# ---------------------------------------------------------------------------


class _TalonLike(doubles._OriginProvider):
    """Models TalonWaitable: auto-woken through ``active_handles`` and
    answering ``origin_session_id`` from the session that dispatched the job."""

    kind = "talon"


DISPATCHED_FROM = "talon-dispatch-session"


@pytest.fixture
def reconciler_agent(tmp_path, sqlite_database_factory):
    """An agent with one provider, a real reconciler and store, and the
    reconciler suite's capturing dispatcher."""

    async def create(provider):
        db = await sqlite_database_factory(tmp_path / "reconciler_agent.db")
        registry = WaitRegistry()
        registry.register(provider)
        dispatcher = doubles._CapturingDispatcher()
        agent = SimpleNamespace(
            did=AGENT_DID,
            agent_id=AGENT_DID,
            _raw_storage=SimpleNamespace(db=db),
            wait_registry=registry,
            dispatcher=dispatcher,
        )
        agent._wait_reconciler = WaitReconciler(agent)
        return agent, dispatcher

    return create


@pytest.fixture
def talon(reconciler_agent):
    async def create():
        provider = _TalonLike(
            origins={"job-1": DISPATCHED_FROM}, signal="talon.job_complete"
        )
        provider.set("job-1", Outcome.DONE)
        return await reconciler_agent(provider)

    return create


@pytest.mark.asyncio
async def test_an_unwatched_talon_job_still_wakes_its_dispatch_session(talon):
    agent, dispatcher = await talon()

    await agent._wait_reconciler.reconcile()

    [wake] = dispatcher.signals
    assert wake.source == "talon.job_complete"
    assert wake.session_id == DISPATCHED_FROM


@pytest.mark.asyncio
async def test_a_talon_watch_with_no_stored_session_keeps_the_provider_binding(
    talon,
):
    """A watch registered outside any session, or before #3663, stores no
    session; the provider's own record still binds the wake."""
    agent, dispatcher = await talon()
    await register_wait_watch(agent, "talon:job-1")
    assert (
        await agent._wait_reconciler._store.get("talon", "job-1")
    ).watch_origin_session_id is None

    await agent._wait_reconciler.reconcile()

    assert [w.session_id for w in dispatcher.signals] == [DISPATCHED_FROM]


@pytest.mark.asyncio
async def test_a_stored_session_is_preferred_over_the_providers_record(talon):
    """The watch's session wins; the provider is not asked."""
    agent, dispatcher = await talon()
    asked: list = []
    provider = agent.wait_registry.get("talon")
    answer = provider.origin_session_id

    async def spy(handle):
        asked.append(handle)
        return await answer(handle)

    provider.origin_session_id = spy
    await register_wait_watch(agent, "talon:job-1", origin_session_id=CHAIN)

    await agent._wait_reconciler.reconcile()

    assert [w.session_id for w in dispatcher.signals] == [CHAIN]
    assert dispatcher.signals[0].payload["origin_session_id"] == CHAIN
    assert asked == []


@pytest.mark.asyncio
async def test_a_stored_session_still_overrides_a_payload_key(
    reconciler_agent,
):
    """Routing authority stays local (#2877): the stored session, not the
    poll payload a provider may fill with a peer's data, binds the wake."""
    provider = doubles._PollOnlyProvider(kind="a2a")
    provider.set(
        "out-1", Outcome.DONE, data={"origin_session_id": "victim-session"}
    )
    agent, dispatcher = await reconciler_agent(provider)
    await register_wait_watch(agent, "a2a:out-1", origin_session_id=CHAIN)

    await agent._wait_reconciler.reconcile()

    [wake] = dispatcher.signals
    assert wake.session_id == CHAIN
    assert wake.payload["origin_session_id"] == CHAIN
