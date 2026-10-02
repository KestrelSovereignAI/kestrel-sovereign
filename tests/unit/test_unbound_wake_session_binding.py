"""#3429: an unbound wake turn's session is fixed when the turn starts.

A COGNITION wake whose signal carries no session used to reach
``process_input`` with ``session_id=None``. The turn then ran with no session
(``get_turn_bound_session_id()`` answered None) while the conversation store
filed each row through the 30-minute time-gap heuristic at write time. Two
values for one turn, and both wrong:

* the rows were glued into whatever conversation was newest — the operator's
  chat, when it was under 30 minutes old — or scattered across a new session
  per wake; and
* work dispatched from the turn stamped an empty origin, so its own wake was
  unbound again and the chain fed itself.

What these tests pin: the dispatcher mints the wake's session before the turn
starts, the turn writes everything under it, the turn's live binding reports
it (that is what a dispatch from inside the turn records as origin), and a
wake bound to that origin returns to the same session. One session per
autonomous chain.

Only the agent body is a double, and it does exactly what
``KestrelAgent.process_input`` does with its ``session_id``: binds it as the
turn's session under the turn lifecycle and threads it into every write. The
dispatcher, the turn lifecycle, the source registration and the conversation
store are production code.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sdk.signals import Signal, SignalMode, Status, Visibility

from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
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
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.session_id_column import is_stampable_session_id

AGENT_DID = "did:test:3429"
OPERATOR_SESSION = "operator-chat-1"


class _WakeAgent(TurnLifecycleMixin, OrchestratorEngineMixin, EventManagerMixin):
    """An agent whose turn body is the session contract and nothing else."""

    did = AGENT_DID
    agent_name = "kestrel"

    def __init__(self, conversation: AsyncConversationStore):
        self.agent_id = AGENT_DID
        self.conversation = conversation
        self._active_session_id = None
        self._event_listeners: list = []
        self._pending_task_notifications: list = []
        self.background_tasks: list[asyncio.Task] = []
        self.turn_sessions: list = []
        # What a dispatch made from inside the turn would stamp as its origin
        # (Talon's ``_origin_session_id``, request_restart, self_followup).
        self.dispatch_origins: list = []

    async def process_input(self, prompt: str, session_id=None, **_kwargs):
        async with self._turn_lifecycle():
            self._active_session_id = session_id
            self.turn_sessions.append(session_id)
            await self.conversation.add_conversation(
                "user", prompt, session_id=session_id
            )
            self.dispatch_origins.append(self.get_turn_bound_session_id())
            await self.conversation.add_conversation(
                "assistant", "wake handled", session_id=session_id
            )
            return "wake handled"

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.fixture
async def rig(tmp_path, sqlite_database_factory):
    backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
    await backend.connect()
    log = SignalLogStore(backend)
    await log.initialize()

    registry = SourceRegistry()
    registry.register(build_wait_complete_registration())

    db = await sqlite_database_factory(tmp_path / "agent.db")
    conversation = AsyncConversationStore(db, agent_id=AGENT_DID)
    agent = _WakeAgent(conversation)
    locks = OrderedLockManager()
    agent._lock_manager = locks
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=locks, store=log,
    )
    seen_events: list = []

    async def _listen(event_type, data):
        seen_events.append((event_type, data))

    agent.add_event_listener(_listen)

    yield SimpleNamespace(
        agent=agent,
        dispatcher=dispatcher,
        conversation=conversation,
        events=seen_events,
    )

    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


def _wake(handle: str, *, session_id=None) -> Signal:
    return Signal(
        source=WAIT_SOURCE,
        kind="complete",
        mode=SignalMode.COGNITION,
        payload={"kind": "example", "handle": handle, "outcome": "done"},
        target_agent=AGENT_DID,
        session_id=session_id,
        visibility=(
            Visibility.USER_VISIBLE if session_id else Visibility.INTERNAL
        ),
    )


async def _sessions_by_role(conversation: AsyncConversationStore) -> list:
    rows = await conversation.get_conversation_history(limit=50)
    return [
        (row["role"], (row.get("metadata") or {}).get("session_id"))
        for row in rows
    ]


@pytest.mark.asyncio
async def test_unbound_wake_runs_in_one_fresh_session_not_the_time_gap_one(rig):
    """The operator chatted minutes ago; an unbound wake must not join it.

    The store's time-gap heuristic WOULD have filed this wake into the
    operator's chat — asserted first, so the test cannot pass vacuously.
    """
    await rig.conversation.add_conversation(
        "user", "operator turn", session_id=OPERATOR_SESSION
    )
    assert await rig.conversation.resolve_session_id(None) == OPERATOR_SESSION

    result = await rig.dispatcher.dispatch_signal(_wake("job-1"))

    assert result.status == Status.OK
    [turn_session] = rig.agent.turn_sessions
    assert is_stampable_session_id(turn_session)
    assert turn_session != OPERATOR_SESSION
    # The session the turn runs in and the one its rows are saved under are
    # the same value.
    assert await _sessions_by_role(rig.conversation) == [
        ("user", OPERATOR_SESSION),
        ("user", turn_session),
        ("assistant", turn_session),
    ]
    # And it is the session a dispatch from inside the turn records as origin.
    assert rig.agent.dispatch_origins == [turn_session]


@pytest.mark.asyncio
async def test_unbound_wake_stays_internal(rig):
    """A minted session has nobody watching it: no live emit is attempted."""
    result = await rig.dispatcher.dispatch_signal(_wake("job-1"))

    assert result.status == Status.OK
    assert not any(kind == "signal_completed" for kind, _ in rig.events)


@pytest.mark.asyncio
async def test_each_unbound_wake_gets_its_own_session(rig):
    """Two unrelated unbound wakes, seconds apart, are two chains.

    Under the time-gap rule the second would have inherited the first's
    session merely for being under 30 minutes later.
    """
    await rig.dispatcher.dispatch_signal(_wake("job-1"))
    await rig.dispatcher.dispatch_signal(_wake("job-2"))

    first, second = rig.agent.turn_sessions
    assert first != second
    assert rig.agent.dispatch_origins == [first, second]


@pytest.mark.asyncio
async def test_bound_wake_resumes_its_origin_session(rig):
    """A bound wake is unchanged: it lands in the session that filed it."""
    result = await rig.dispatcher.dispatch_signal(
        _wake("job-1", session_id=OPERATOR_SESSION)
    )

    assert result.status == Status.OK
    assert rig.agent.turn_sessions == [OPERATOR_SESSION]
    assert rig.agent.dispatch_origins == [OPERATOR_SESSION]


@pytest.mark.asyncio
async def test_autonomous_chain_stays_in_one_session(rig):
    """wake -> dispatch -> wake: the chain returns to where it started.

    The first wake is unbound (its root had no chat turn). Work dispatched
    from it records the turn's session; that work's completion wake is
    therefore bound and resumes the same session. Before #3429 the recorded
    origin was empty, every hop was unbound again, and each one opened a new
    time-gap session the agent's main conversation could not see.
    """
    await rig.dispatcher.dispatch_signal(_wake("job-1"))
    [origin] = rig.agent.dispatch_origins
    assert origin

    await rig.dispatcher.dispatch_signal(_wake("job-2", session_id=origin))

    assert rig.agent.turn_sessions == [origin, origin]
    assert rig.agent.dispatch_origins == [origin, origin]
    assert {sid for _, sid in await _sessions_by_role(rig.conversation)} == {
        origin
    }
