"""#4001: a turn parked before its first LLM call, and Stop reaching it.

On 2026-11-09 and 11-10 Emma (twice) and Claw each had a turn begin and then
hold CONVERSATION, for up to 17 hours, without an LLM call, a tool call or a
log line. Later turns, and a scheduled tool holding the memory lock,
queued behind it. ``kestrel stop`` reported the agents ``unreachable``.

The reproduced cycle: a turn's effect-journal retirement holds the
privacy-transition lock after the turn releases CONVERSATION, and joins every
Task the turn created. A dispatched signal's COGNITION route is one of those
and waits for CONVERSATION; a turn that took CONVERSATION in between waits for
the privacy lock. A dispatched signal is now created outside the dispatching
turn's privacy lifetime; those regressions are in
``test_wake_detached_from_enclosing_turn.py`` (#4023). This file covers Stop
reaching a parked turn, and the settlement of work a wake that ended
undelivered carried.

Every agent here is real, on real storage; only the model is scripted.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from kestrel_sdk.signals import (
    ResourceLock,
)

from kestrel_sovereign.agent.invocation import InvocationCancelledError
from kestrel_sovereign.await_chain import format_await_chain
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from kestrel_sovereign.signals.context import get_current_signal
from kestrel_sovereign.signals.sources.a2a import SOURCE_NAME as COMPLETION_WAKE
from kestrel_sovereign.stop import StopCleanupRegistry
from kestrel_sovereign.stop.authority import CancellationAuthority
from kestrel_sovereign.stop.receipt import StopReceiptStore
from kestrel_sovereign.stop.runtime_target import build_runtime_stop_target
from kestrel_sovereign.stop.types import (
    StopDisposition,
    StopDoor,
    StopRequest,
    StopScope,
)
from tests.shared.booted_agent import booted_agent

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(60)]

REPLY = "Noted."
#: Each phase of a scenario must settle well inside this. A deadlock does not.
SETTLE_SECONDS = 10.0


def _script_replies(agent, monkeypatch, generate=None):
    async def generate_with_messages(**kwargs):
        if generate is not None:
            return await generate(kwargs)
        return LLMResponse(content=REPLY)

    async def stream_with_tool_detection(**kwargs):
        yield REPLY

    monkeypatch.setattr(agent.llm_service, "generate_with_messages", generate_with_messages)
    monkeypatch.setattr(
        agent.llm_service, "stream_with_tool_detection", stream_with_tool_detection
    )


def _woken_by(sources):
    """A scripted model that records the source of the wake whose turn called it.

    Ending a task gives its agent an ``a2a.task_complete`` wake, and that
    wake's turn calls the model. A test that a failed submission wake ran no
    cognition must tell those calls apart from the ones it rules out.
    """

    async def generate(kwargs):
        signal = get_current_signal()
        sources.append(None if signal is None else signal.source)
        return LLMResponse(content=REPLY)

    return generate


def _await_chains(agent) -> str:
    """Where every live turn and lock holder is parked, for a failure message."""
    chains = []
    for task in asyncio.all_tasks():
        if task.get_name().startswith(("invocation-turn", "turn-", "signal_dispatch")):
            chains.append(format_await_chain(task))
    return (
        f"holds: {agent._get_lock_manager().active_hold_diagnostics()}\n"
        f"privacy lock held: {agent._get_privacy_transition_lock().locked()}\n"
        + "\n".join(chains)
    )


async def _settle(agent, tasks, label):
    done, pending = await asyncio.wait(set(tasks), timeout=SETTLE_SECONDS)
    if pending:
        chains = _await_chains(agent)
        for task in pending:
            task.cancel()
        # A deadlocked turn can absorb its cancellation; report, do not wait.
        await asyncio.wait(pending, timeout=1.0)
        pytest.fail(f"{label}: {len(pending)} task(s) never finished\n{chains}")
    return done


async def _park_after_begin(agent, monkeypatch):
    """Park the next turn on a future nothing resolves, just after ``begin``.

    The genesis gate is the turn's first await inside the turn lifecycle,
    before the privacy span, the context build or any model call.
    """
    park = asyncio.get_running_loop().create_future()
    parked = asyncio.Event()
    armed = {"armed": True}
    real_gate = agent._genesis_audit_cognition_block

    async def gate(user_input):
        if agent._live_turn_id is not None and armed["armed"]:
            armed["armed"] = False
            parked.set()
            await park
        return await real_gate(user_input)

    monkeypatch.setattr(agent, "_genesis_audit_cognition_block", gate)
    return parked


def _authority(agent, receipts) -> CancellationAuthority:
    return CancellationAuthority(
        lambda: (build_runtime_stop_target(agent, agent_id=agent.agent_id),),
        cleanup_registry=StopCleanupRegistry(),
        receipt_store=receipts,
        door=StopDoor.AGENT,
    )


async def _receipt_store(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    database = await AsyncDatabase.sqlite(str(tmp_path / "stop-receipts.db"))
    store = StopReceiptStore(database)
    await store.ensure_schema()
    return database, store


SENDER = "did:kestrel:sender-4001"


async def _submit_task(agent, *, metadata=None):
    """An inbound A2A work order from ``SENDER``; its wake is enqueued."""
    from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message
    from kestrel_sdk.native_a2a.types import Role, SendMessageRequest

    manager = agent.task_manager
    return await manager.create_task(
        SendMessageRequest(
            tenant=manager.task_store.scope.tenant,
            message=new_text_message(
                "Please do this", context_id="context-4001", role=Role.ROLE_USER
            ),
            metadata=metadata,
        ),
        creator_agent_id=SENDER,
    )


def _looping_causation(agent):
    """Native causation metadata naming a wake this recipient already ran.

    Its submission wake is ``(recipient, a2a.task_submitted)`` again, so the
    dispatcher refuses it as a cycle before it persists any delivery.
    """
    from datetime import UTC, datetime

    from kestrel_sdk.signals import CausationFrame

    from kestrel_sovereign.a2a.extensions import (
        causation_metadata,
        serialize_chain_for_metadata,
    )

    frame = CausationFrame(
        agent_id=agent.did,
        source="a2a.task_submitted",
        signal_id="sig-earlier-4001",
        turn_id="turn-earlier-4001",
        depth=1,
        emitted_at=datetime.now(UTC),
    )
    return causation_metadata(serialize_chain_for_metadata([frame]))


async def _sender_state(agent, task):
    """The task's state as its sender reads it."""
    seen = await agent.task_manager.get_task_for_creator(
        task.id, SENDER, recipient_agent_id=agent.did
    )
    return seen.status.state


async def _submission_deliveries(agent):
    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        DURABLE_COGNITION_CONSUMER_ID,
    )

    return await agent.dispatcher.list_durable_deliveries(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID
    )


async def _until_completion_wake_settled(agent):
    """Wait for the ``a2a.task_complete`` wake an ended task gives its agent.

    That wake's turn runs after the test's own assertions. Leaving the agent
    while it runs cancels it, perhaps mid-COMMIT, and a COMMIT cancelled in
    flight is an unknown outcome the agent's teardown rightly reports.
    """
    from kestrel_sovereign.signals.sources.a2a import DURABLE_COGNITION_CONSUMER_ID

    async def settled():
        wakes = await agent.dispatcher.list_durable_deliveries(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID
        )
        return bool(wakes) and all(wake.is_terminal for wake in wakes)

    await _until(agent, settled, "the ended task's completion wake never settled")


async def _until(agent, predicate, label):
    """Poll an async predicate; fail with every await chain if it never holds."""
    deadline = asyncio.get_running_loop().time() + SETTLE_SECONDS
    while not await predicate():
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(f"{label}\n{_await_chains(agent)}")
        await asyncio.sleep(0.02)


async def _wait_until_queued(agent):
    while agent._get_lock_manager().active_hold_diagnostics()[0][
        "blocked_acquirers"
    ] == 0:
        await asyncio.sleep(0.01)


async def _stop_agent(agent, receipts):
    request = StopRequest(
        scope=StopScope.AGENT,
        actor_id="operator",
        target=agent.agent_id,
        target_agent_id=agent.agent_id,
    )
    outcomes = await asyncio.wait_for(
        _authority(agent, receipts).stop(request), SETTLE_SECONDS
    )
    return [outcome.disposition for outcome in outcomes]


def _gate_resident_drain(agent, monkeypatch):
    """Hold the resident drainer before its next scan until the event is set."""
    gate = asyncio.Event()
    scan = agent.dispatcher._drainable_durable_deliveries

    async def gated(*args, **kwargs):
        await gate.wait()
        return await scan(*args, **kwargs)

    monkeypatch.setattr(agent.dispatcher, "_drainable_durable_deliveries", gated)
    return gate


async def test_stopping_a_queued_a2a_wake_cancels_the_task_its_sender_waits_on(
    tmp_path, monkeypatch
):
    """The 11-10 Claw shape: Stop cancels a queued turn carrying an inbound A2A
    work order. The task must end where its sender can see it, not stay
    ``submitted`` after its only wake was consumed. The settlement pass ends
    it from the wake's committed terminal row."""
    from kestrel_sdk.native_a2a.types import TaskState

    database, receipts = await _receipt_store(tmp_path)
    try:
        async with booted_agent(
            tmp_path / "agent", name="Recipient", provenance="test:4001-a2a-stop", features=set()
        ) as agent:
            _script_replies(agent, monkeypatch)
            parked = await _park_after_begin(agent, monkeypatch)
            first = asyncio.create_task(agent.process_input("one", session_id="s"))
            await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
            task = await _submit_task(agent)
            await _wait_until_queued(agent)

            assert await _stop_agent(agent, receipts) == [StopDisposition.STOPPED]
            await _settle(agent, (first,), "the stopped turn")

            async def cancelled():
                return await _sender_state(agent, task) == TaskState.TASK_STATE_CANCELED

            await _until(agent, cancelled, "the stopped wake's task was never cancelled")
            assert [d.status for d in await _submission_deliveries(agent)] == [
                "terminal_ackable"
            ]
            await _until_completion_wake_settled(agent)
    finally:
        await database.close()


async def test_a_stopped_wake_whose_task_cannot_be_cancelled_is_settled_by_the_heartbeat(
    tmp_path, monkeypatch
):
    """Stop consumes a queued A2A wake, and the write that cancels its task
    fails. Stop still reports the turn stopped: its terminal row committed
    first, and the work it ended is settled from that row. Until the fault
    clears, the running owner heartbeat finds the task owed on every pass
    and retries; then the task ends cancelled, as the Stop asked, without a
    restart (#4001)."""
    from datetime import timedelta

    from kestrel_sdk.native_a2a.types import TaskState

    from kestrel_sovereign.signals.dispatcher import SignalDispatcher

    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__,
        "runtime_owner_stale_after",
        timedelta(seconds=0.6),
    )
    database, receipts = await _receipt_store(tmp_path)
    try:
        async with booted_agent(
            tmp_path / "agent", name="Recipient", provenance="test:4001-a2a-fail", features=set()
        ) as agent:
            _script_replies(agent, monkeypatch)
            parked = await _park_after_begin(agent, monkeypatch)
            first = asyncio.create_task(agent.process_input("one", session_id="s"))
            await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
            task = await _submit_task(agent)
            await _wait_until_queued(agent)
            manager = agent.task_manager
            real_cancel = manager.cancel_task
            faulted = {"on": True}
            attempts = []

            async def cancel_task(*args, **kwargs):
                attempts.append(args[0])
                if faulted["on"]:
                    raise OSError("database is locked")
                return await real_cancel(*args, **kwargs)

            monkeypatch.setattr(manager, "cancel_task", cancel_task)

            assert await _stop_agent(agent, receipts) == [StopDisposition.STOPPED]
            await _settle(agent, (first,), "the stopped turn")

            async def retried():
                return len(attempts) >= 3

            await _until(agent, retried, "the heartbeat never retried the settlement")
            assert set(attempts) == {task.id}
            assert await _sender_state(agent, task) == TaskState.TASK_STATE_SUBMITTED
            assert [d.status for d in await _submission_deliveries(agent)] == [
                "terminal_ackable"
            ]

            faulted["on"] = False

            async def cancelled():
                return await _sender_state(agent, task) == TaskState.TASK_STATE_CANCELED

            await _until(agent, cancelled, "the task was left submitted once the fault cleared")
            assert [d.status for d in await _submission_deliveries(agent)] == [
                "terminal_ackable"
            ]
            await _until_completion_wake_settled(agent)
    finally:
        await database.close()


async def test_a_stop_settled_after_its_dispatch_was_cancelled_still_ends_the_task(
    tmp_path, monkeypatch
):
    """Stop cancels the wake's turn, and its dispatch is itself cancelled before
    it can settle the delivery. The retained route's cleanup settles it
    instead, and must end the task with it (round-2 self-review, #4001)."""
    from contextlib import asynccontextmanager

    from kestrel_sdk.native_a2a.types import TaskState

    async with booted_agent(
        tmp_path / "agent", name="Recipient", provenance="test:4001-retained", features=set()
    ) as agent:
        _script_replies(agent, monkeypatch)
        parked = await _park_after_begin(agent, monkeypatch)
        renewal = agent.dispatcher._renew_durable_cognition_lease
        route_finished = asyncio.Event()

        @asynccontextmanager
        async def renewal_that_outlives_the_route(delivery):
            async with renewal(delivery) as lost:
                yield lost
            # Only the submission wake under test; the completion wake its
            # cancelled task causes must run, or teardown cancels it mid-turn.
            if delivery.event.source == "a2a.task_submitted":
                route_finished.set()
                await asyncio.Event().wait()

        monkeypatch.setattr(
            agent.dispatcher,
            "_renew_durable_cognition_lease",
            renewal_that_outlives_the_route,
        )
        task = await _submit_task(agent)
        await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
        handle = agent._a2a_submitted_signal_handles[task.id]
        assert agent.cancel_current_request(handle.signal_id)
        await asyncio.wait_for(route_finished.wait(), SETTLE_SECONDS)
        handle.cancel()
        await asyncio.gather(handle.task, return_exceptions=True)

        async def settled():
            deliveries = await _submission_deliveries(agent)
            return bool(deliveries) and all(d.is_terminal for d in deliveries)

        await _until(agent, settled, "the retained route never settled its delivery")

        async def cancelled():
            return await _sender_state(agent, task) == TaskState.TASK_STATE_CANCELED

        await _until(agent, cancelled, "the retained route's task was never cancelled")
        await _until_completion_wake_settled(agent)


@pytest.mark.parametrize("phase", ["queued", "in_flight"])
async def test_a_wake_interrupted_by_shutdown_ends_its_task_after_the_restart(
    tmp_path, monkeypatch, phase
):
    """The 11-11 restart route: the host shuts down while an inbound A2A wake
    is queued behind a parked turn, or is itself running. After the restart
    its retry is refused, because the tool operation its first attempt opened
    needs reconciliation (#3999). That refusal must end the task where its
    sender sees it, not leave it ``submitted`` with no wake left to run it."""
    from kestrel_sdk.native_a2a.types import TaskState

    home = tmp_path / "agent"
    queued_turn = None
    async with booted_agent(
        home, name="Recipient", provenance="test:4001-restart-route", features=set()
    ) as agent:
        _script_replies(agent, monkeypatch)
        parked = await _park_after_begin(agent, monkeypatch)
        if phase == "queued":
            queued_turn = asyncio.create_task(agent.process_input("one", session_id="s"))
            await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
            task = await _submit_task(agent)
            await _wait_until_queued(agent)
        else:
            task = await _submit_task(agent)
            await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
        did = agent.did
    if queued_turn is not None:
        # The caller owns its own turn; a host's request teardown cancels it.
        queued_turn.cancel()
        await asyncio.gather(queued_turn, return_exceptions=True)

    async with booted_agent(
        home,
        name="Recipient",
        provenance="test:4001-restart-route",
        features=set(),
        identity=did,
    ) as agent:
        _script_replies(agent, monkeypatch)

        async def ended():
            return await _sender_state(agent, task) != TaskState.TASK_STATE_SUBMITTED

        await _until(agent, ended, "the restarted agent left the task submitted")
        assert await _sender_state(agent, task) == TaskState.TASK_STATE_FAILED
        assert [d.status for d in await _submission_deliveries(agent)] == ["failed"]
        await _until_completion_wake_settled(agent)


_SETTLEMENT_FAULT = "inject_4001_task_settlement_fault"


async def test_a_failed_wake_whose_task_write_failed_is_settled_without_a_restart(
    tmp_path, monkeypatch
):
    """Final self-review of wo-4001-3: after a restart the wake's retry is
    refused and its delivery fails, but the write that ends its task fails
    too. A failed row is never reclaimed, so before the fix nothing settled
    the task until the next boot, and its sender saw ``submitted``. The
    owner heartbeat's settlement pass finds the task owed from durable state
    on every pass while the host runs, so clearing the fault is enough."""
    from datetime import timedelta

    from kestrel_sdk.native_a2a.types import TaskState

    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.signals.dispatcher import SignalDispatcher

    home = tmp_path / "agent"
    async with booted_agent(
        home, name="Recipient", provenance="test:4001-settle-retry", features=set()
    ) as agent:
        _script_replies(agent, monkeypatch)
        parked = await _park_after_begin(agent, monkeypatch)
        task = await _submit_task(agent)
        await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
        did = agent.did

    # A real write fault on the agent's own database: the task cannot be
    # marked failed while this trigger stands.
    fault = sqlite3.connect(home / "kestrel_prime.db")
    fault.execute(
        f"CREATE TRIGGER {_SETTLEMENT_FAULT} BEFORE UPDATE ON a2a_tasks "
        "WHEN NEW.status = 'failed' "
        "BEGIN SELECT RAISE(ABORT, 'injected task settlement fault'); END"
    )
    fault.commit()
    settlements = []
    settle = KestrelAgent.settle_undelivered_cognition_wake

    async def counted(self, **identity):
        settlements.append(identity["source_event_id"])
        return await settle(self, **identity)

    monkeypatch.setattr(KestrelAgent, "settle_undelivered_cognition_wake", counted)
    # The owner heartbeat is the recovery worker; run its cadence at test speed.
    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__,
        "runtime_owner_stale_after",
        timedelta(seconds=0.6),
    )
    try:
        async with booted_agent(
            home,
            name="Recipient",
            provenance="test:4001-settle-retry",
            features=set(),
            identity=did,
        ) as agent:
            _script_replies(agent, monkeypatch)

            async def retried_against_the_fault():
                deliveries = await _submission_deliveries(agent)
                return [d.status for d in deliveries] == ["failed"] and len(settlements) >= 3

            await _until(
                agent,
                retried_against_the_fault,
                "the failed wake's settlement was not retried against the fault",
            )
            assert len(set(settlements)) == 1
            assert await _sender_state(agent, task) == TaskState.TASK_STATE_SUBMITTED

            fault.execute(f"DROP TRIGGER {_SETTLEMENT_FAULT}")
            fault.commit()

            async def ended():
                return await _sender_state(agent, task) != TaskState.TASK_STATE_SUBMITTED

            await _until(agent, ended, "the task was left submitted once the fault cleared")
            assert await _sender_state(agent, task) == TaskState.TASK_STATE_FAILED
            assert [d.status for d in await _submission_deliveries(agent)] == ["failed"]
            assert await agent.dispatcher.owed_cognition_wake_settlements() == []
            assert agent.dispatcher.unresolved_cognition_control == ()
            await _until_completion_wake_settled(agent)
    finally:
        fault.close()


async def test_a_wake_already_failed_at_boot_is_settled_without_another_restart(
    tmp_path, monkeypatch
):
    """Self-review of wo-4001-4: a process can stop between committing a
    wake's failed row and settling its task. The next boot finds the row
    already failed, and the write that ends the task fails too. Before the
    fix boot reconciliation only logged that failure, and no process state
    recorded the task as owed, so the sender saw ``submitted`` until yet
    another restart. Boot's settlement pass is now the first of the owner
    heartbeat's passes, each reading what is owed from durable state.
    Settling it writes neither cognition nor the committed row again."""
    from datetime import timedelta

    from kestrel_sdk.native_a2a.types import TaskState

    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.signals.dispatcher import SignalDispatcher
    from kestrel_sovereign.signals.durable import DurableSignalStore

    home = tmp_path / "agent"
    provenance = "test:4001-boot-settle-retry"
    async with booted_agent(
        home, name="Recipient", provenance=provenance, features=set()
    ) as agent:
        _script_replies(agent, monkeypatch)
        parked = await _park_after_begin(agent, monkeypatch)
        task = await _submit_task(agent)
        await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
        did = agent.did

    # The restart refuses the wake's retry and commits its failed row; the
    # process then stops before it settles the task.
    async def stopped_before_settling(self, **identity):
        return False

    with monkeypatch.context() as stopped:
        stopped.setattr(
            KestrelAgent, "settle_undelivered_cognition_wake", stopped_before_settling
        )
        async with booted_agent(
            home, name="Recipient", provenance=provenance, features=set(), identity=did
        ) as agent:
            _script_replies(agent, monkeypatch)

            async def failed():
                return [d.status for d in await _submission_deliveries(agent)] == [
                    "failed"
                ]

            await _until(agent, failed, "the refused retry never failed its row")
            assert await _sender_state(agent, task) == TaskState.TASK_STATE_SUBMITTED

    fault = sqlite3.connect(home / "kestrel_prime.db")
    fault.execute(
        f"CREATE TRIGGER {_SETTLEMENT_FAULT} BEFORE UPDATE ON a2a_tasks "
        "WHEN NEW.status = 'failed' "
        "BEGIN SELECT RAISE(ABORT, 'injected task settlement fault'); END"
    )
    fault.commit()
    settlements = []
    settle = KestrelAgent.settle_undelivered_cognition_wake

    async def counted(self, **identity):
        settlements.append(identity["source_event_id"])
        return await settle(self, **identity)

    failed_row_writes = []
    write_row = DurableSignalStore.release_managed_delivery_after_task

    async def counted_row_write(self, **kwargs):
        failed_row_writes.append(kwargs["delivery_id"])
        return await write_row(self, **kwargs)

    monkeypatch.setattr(KestrelAgent, "settle_undelivered_cognition_wake", counted)
    monkeypatch.setattr(
        DurableSignalStore, "release_managed_delivery_after_task", counted_row_write
    )
    # The owner heartbeat is the recovery worker; run its cadence at test speed.
    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__,
        "runtime_owner_stale_after",
        timedelta(seconds=0.6),
    )
    woken_by = []

    try:
        async with booted_agent(
            home, name="Recipient", provenance=provenance, features=set(), identity=did
        ) as agent:
            _script_replies(agent, monkeypatch, _woken_by(woken_by))

            async def retried_against_the_fault():
                return len(settlements) >= 3

            await _until(
                agent,
                retried_against_the_fault,
                "boot left the failed wake's settlement to no retry",
            )
            assert len(set(settlements)) == 1
            assert await _sender_state(agent, task) == TaskState.TASK_STATE_SUBMITTED

            fault.execute(f"DROP TRIGGER {_SETTLEMENT_FAULT}")
            fault.commit()

            async def ended():
                return await _sender_state(agent, task) != TaskState.TASK_STATE_SUBMITTED

            await _until(agent, ended, "the task was left submitted once the fault cleared")
            assert await _sender_state(agent, task) == TaskState.TASK_STATE_FAILED
            assert [d.status for d in await _submission_deliveries(agent)] == ["failed"]
            assert await agent.dispatcher.owed_cognition_wake_settlements() == []
            assert failed_row_writes == []
            await _until_completion_wake_settled(agent)
            assert set(woken_by) <= {COMPLETION_WAKE}
    finally:
        fault.close()


async def test_a_failed_wake_whose_settlement_loses_a_lifecycle_race_is_settled_by_the_heartbeat(
    tmp_path, monkeypatch
):
    """Self-review of wo-4001-4: after a restart the wake's retry is refused
    and its row fails. Between the settlement's read of the task and its
    write, another recipient writer moves the task SUBMITTED -> WORKING, so
    the settlement's lifecycle compare-and-set loses and raises. Before the
    fix the dispatcher kept its in-process debt only for a storage error, so
    this one was dropped: the delivery stayed failed, the sender saw
    ``working``, and the owner heartbeat had nothing to retry. What is owed is
    now read from durable state on every pass, so the running heartbeat
    settles the task, writing neither cognition nor the committed row
    again."""
    from datetime import timedelta

    from kestrel_sdk.native_a2a.types import Task, TaskState

    from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore
    from kestrel_sovereign.signals.dispatcher import SignalDispatcher
    from kestrel_sovereign.signals.durable import DurableSignalStore

    home = tmp_path / "agent"
    provenance = "test:4001-settle-cas-race"
    async with booted_agent(
        home, name="Recipient", provenance=provenance, features=set()
    ) as agent:
        _script_replies(agent, monkeypatch)
        parked = await _park_after_begin(agent, monkeypatch)
        task = await _submit_task(agent)
        await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
        did = agent.did

    native_save = TaskStore.save_recipient_lifecycle
    raced = []

    async def concurrent_working_writer(self, value, **kwargs):
        if not raced and value.id == task.id and (
            value.status.state == TaskState.TASK_STATE_FAILED
        ):
            raced.append(value.id)
            baseline = kwargs["baseline"]
            working = Task.FromString(baseline.SerializeToString())
            working.status.state = TaskState.TASK_STATE_WORKING
            # The other writer wins the real store CAS for the same input.
            assert await native_save(
                self,
                working,
                baseline=baseline,
                recipient_agent_id=kwargs["recipient_agent_id"],
                expected_state=baseline.status.state,
            ) is not None
        return await native_save(self, value, **kwargs)

    failed_row_writes = []
    write_row = DurableSignalStore.release_managed_delivery_after_task

    async def counted_row_write(self, **kwargs):
        failed_row_writes.append(kwargs["delivery_id"])
        return await write_row(self, **kwargs)

    monkeypatch.setattr(TaskStore, "save_recipient_lifecycle", concurrent_working_writer)
    monkeypatch.setattr(
        DurableSignalStore, "release_managed_delivery_after_task", counted_row_write
    )
    # The owner heartbeat is the recovery worker; run its cadence at test speed.
    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__,
        "runtime_owner_stale_after",
        timedelta(seconds=0.6),
    )
    woken_by = []

    async with booted_agent(
        home, name="Recipient", provenance=provenance, features=set(), identity=did
    ) as agent:
        _script_replies(agent, monkeypatch, _woken_by(woken_by))

        async def lost_the_race():
            return bool(raced) and [d.status for d in await _submission_deliveries(agent)] == [
                "failed"
            ]

        await _until(agent, lost_the_race, "the refused retry never failed its row")

        async def ended():
            return await _sender_state(agent, task) == TaskState.TASK_STATE_FAILED

        await _until(
            agent,
            ended,
            "the owner heartbeat left the task live after its settlement lost a "
            "lifecycle race",
        )
        assert [d.status for d in await _submission_deliveries(agent)] == ["failed"]
        assert await agent.dispatcher.owed_cognition_wake_settlements() == []
        # The row was written once, by the refused retry; the heartbeat wrote
        # only the task.
        assert len(failed_row_writes) == 1
        await _until_completion_wake_settled(agent)
        assert set(woken_by) <= {COMPLETION_WAKE}


async def test_a_cycle_refused_submission_wake_fails_the_task_its_sender_waits_on(
    tmp_path, monkeypatch
):
    """Round-3 self-review: the dispatcher refuses a submission wake as a cycle
    before it persists a delivery, and its producer stops on the TERMINAL
    admission. The refusal is recorded as the wake's own failed row, so the
    one settlement pass ends the task where its sender sees it (#4001)."""
    from kestrel_sdk.native_a2a.types import TaskState

    async with booted_agent(
        tmp_path / "agent", name="Recipient", provenance="test:4001-cycle", features=set()
    ) as agent:
        _script_replies(agent, monkeypatch)
        task = await _submit_task(agent, metadata=_looping_causation(agent))

        async def ended():
            return await _sender_state(agent, task) != TaskState.TASK_STATE_SUBMITTED

        await _until(agent, ended, "the cycle-refused wake left its task submitted")
        seen = await agent.task_manager.get_task_for_creator(
            task.id, SENDER, recipient_agent_id=agent.did
        )
        assert seen.status.state == TaskState.TASK_STATE_FAILED
        assert "dropped_cycle" in seen.status.message.parts[0].text
        (refused,) = await _submission_deliveries(agent)
        assert refused.status == "failed" and refused.attempts == 0
        assert refused.last_error.startswith("refused: dropped_cycle:")
        assert refused.event.caller_identity is None
        assert refused.event.causation_chain == []
        assert await agent.dispatcher.owed_cognition_wake_settlements() == []
        await _until_completion_wake_settled(agent)


async def test_a_cycle_refused_wake_whose_settlement_fails_is_settled_by_the_heartbeat(
    tmp_path, monkeypatch
):
    """When the refused wake's settlement cannot commit, the refusal stays
    recorded and its producer is done with it: the owner heartbeat's
    settlement pass re-finds the same row on every pass, and the task ends
    once the fault clears, with no restart and no new wake (#4001)."""
    from datetime import timedelta

    from kestrel_sdk.native_a2a.types import TaskState

    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.signals.dispatcher import SignalDispatcher

    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__,
        "runtime_owner_stale_after",
        timedelta(seconds=0.6),
    )
    attempts = []
    fault = {"armed": True}
    settle = KestrelAgent.settle_undelivered_cognition_wake

    async def unavailable(self, **identity):
        attempts.append(identity["source_event_id"])
        if fault["armed"]:
            raise OSError("database is locked")
        return await settle(self, **identity)

    monkeypatch.setattr(KestrelAgent, "settle_undelivered_cognition_wake", unavailable)
    async with booted_agent(
        tmp_path / "agent",
        name="Recipient",
        provenance="test:4001-cycle-heartbeat",
        features=set(),
    ) as agent:
        _script_replies(agent, monkeypatch)
        task = await _submit_task(agent, metadata=_looping_causation(agent))

        async def retried():
            return len(attempts) >= 3

        await _until(agent, retried, "the refused wake's settlement was not retried")
        assert len(set(attempts)) == 1
        assert await _sender_state(agent, task) == TaskState.TASK_STATE_SUBMITTED
        assert [d.status for d in await _submission_deliveries(agent)] == ["failed"]

        fault["armed"] = False

        async def ended():
            return await _sender_state(agent, task) == TaskState.TASK_STATE_FAILED

        await _until(agent, ended, "the refused wake's task was left submitted")
        assert [d.status for d in await _submission_deliveries(agent)] == ["failed"]
        assert await agent.dispatcher.owed_cognition_wake_settlements() == []
        await _until_completion_wake_settled(agent)


async def test_a_turn_parked_after_begin_is_stopped_and_the_queued_turn_runs(
    tmp_path, monkeypatch
):
    database, receipts = await _receipt_store(tmp_path)
    try:
        async with booted_agent(
            tmp_path / "agent", name="Parked", provenance="test:4001-turn", features=set()
        ) as agent:
            _script_replies(agent, monkeypatch)
            parked = await _park_after_begin(agent, monkeypatch)
            first = asyncio.create_task(agent.process_input("one", session_id="s"))
            await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
            parked_turn = agent._live_turn_id
            second = asyncio.create_task(agent.process_input("two", session_id="s"))
            while agent._get_lock_manager().active_hold_diagnostics()[0][
                "blocked_acquirers"
            ] == 0:
                await asyncio.sleep(0.01)

            request = StopRequest(
                scope=StopScope.TURN,
                actor_id="operator",
                target=parked_turn,
                target_agent_id=agent.agent_id,
                target_is_turn_id=True,
            )
            outcomes = await asyncio.wait_for(
                _authority(agent, receipts).stop(request), SETTLE_SECONDS
            )

            assert [outcome.disposition for outcome in outcomes] == [
                StopDisposition.STOPPED
            ], outcomes
            await _settle(agent, (first, second), "the stopped and queued turns")
            assert isinstance(first.exception(), InvocationCancelledError)
            assert second.result() == REPLY
            assert not agent._get_lock_manager().is_held(ResourceLock.CONVERSATION)
            receipt = await receipts.load(request)
            assert receipt is not None
            assert [outcome.disposition for outcome in receipt.outcomes] == [
                StopDisposition.STOPPED
            ]
    finally:
        await database.close()


async def test_agent_stop_of_a_turn_parked_after_begin_is_receipted_stopped(
    tmp_path, monkeypatch
):
    """``kestrel stop <agent>``: every current unit stops, and says so."""
    database, receipts = await _receipt_store(tmp_path)
    try:
        async with booted_agent(
            tmp_path / "agent", name="Parked", provenance="test:4001-agent", features=set()
        ) as agent:
            _script_replies(agent, monkeypatch)
            parked = await _park_after_begin(agent, monkeypatch)
            first = asyncio.create_task(agent.process_input("one", session_id="s"))
            await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
            queued = asyncio.create_task(agent.process_input("two", session_id="s"))
            while agent._get_lock_manager().active_hold_diagnostics()[0][
                "blocked_acquirers"
            ] == 0:
                await asyncio.sleep(0.01)

            request = StopRequest(
                scope=StopScope.AGENT,
                actor_id="operator",
                target=agent.agent_id,
                target_agent_id=agent.agent_id,
            )
            outcomes = await asyncio.wait_for(
                _authority(agent, receipts).stop(request), SETTLE_SECONDS
            )

            assert [outcome.disposition for outcome in outcomes] == [
                StopDisposition.STOPPED
            ], outcomes
            await _settle(agent, (first, queued), "the stopped turns")
            assert isinstance(first.exception(), InvocationCancelledError)
            assert isinstance(queued.exception(), InvocationCancelledError)
            assert not agent._get_lock_manager().is_held(ResourceLock.CONVERSATION)
            assert await asyncio.wait_for(
                agent.process_input("three", session_id="s"), SETTLE_SECONDS
            ) == REPLY
    finally:
        await database.close()


async def test_a_failed_required_audit_leaves_the_next_turn_runnable(
    tmp_path, monkeypatch
):
    """The 11-09 trigger shape (#4002): a required dispatch audit write raises
    during a tool batch. The next turn still takes both locks and finishes."""
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError

    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", "1")
    async with booted_agent(
        tmp_path / "agent", name="Audited", provenance="test:4001-audit", features=set()
    ) as agent:

        async def generate(kwargs):
            messages = kwargs.get("messages") or [{}]
            if messages[-1].get("role") != "tool":
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="status-1", name="wait_status", arguments={"limit": 5}),
                        ToolCall(id="status-2", name="wait_status", arguments={"limit": 1}),
                    ],
                )
            return LLMResponse(content=REPLY)

        _script_replies(agent, monkeypatch, generate)
        store = agent.observability_store
        real_log = store.log_tool_dispatch
        failures = {"left": 1}

        async def log_tool_dispatch(entry, **kwargs):
            if failures["left"]:
                failures["left"] -= 1
                raise sqlite3.OperationalError(
                    "table a2a_tool_dispatches has no column named dispatch_id"
                )
            return await real_log(entry, **kwargs)

        monkeypatch.setattr(store, "log_tool_dispatch", log_tool_dispatch)
        with pytest.raises(ToolAuditUnconfirmedError):
            await asyncio.wait_for(
                agent.process_input("check waits", session_id="s"), SETTLE_SECONDS
            )
        assert failures["left"] == 0
        assert not agent._get_lock_manager().is_held(ResourceLock.CONVERSATION)
        assert not agent._get_privacy_transition_lock().locked()

        following = asyncio.create_task(agent.process_input("again", session_id="s"))
        await _settle(agent, (following,), "the turn after the failed audit")
        assert REPLY in following.result()


async def test_a_scheduled_tool_holding_the_memory_lock_before_its_body_is_stopped(
    native_tool_effect_agent, monkeypatch
):
    """The 03:00Z ``training_cycle`` shape: a cron dispatch holds the memory
    lock and its tool parks before the body runs. Stop cancels it, the memory
    lock is released, and the body never runs."""
    import json
    from datetime import UTC, datetime, timedelta

    from kestrel_sovereign.features.base import Feature, tool
    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
    from kestrel_sovereign.features.scheduler.runner import (
        SCHEDULER_PROTOCOL_VERSION,
        SchedulerRunner,
    )
    from tests.utils.scheduler_ticks import tick_and_settle

    owner = await native_tool_effect_agent(did="did:test:4001-scheduled")
    database = owner.tool_effect_store.database
    bodies = []

    class TrainingFeature(Feature):
        @property
        def tool_description(self):
            return "A scheduled tool that needs the memory lock"

        async def initialize(self):
            pass

        @tool(name="training_cycle_4001", description="Run one training cycle")
        async def training_cycle_4001(self) -> dict:
            bodies.append(True)
            return {"success": True}

    feature = TrainingFeature(owner)
    owner.features[feature.name] = feature
    scheduler = SchedulerFeature(owner)
    scheduler._db, scheduler._agent_id = database, owner.did

    park = asyncio.get_running_loop().create_future()
    parked = asyncio.Event()
    real_body = scheduler._run_tool_hook_gated_body

    async def body_after_park(*args, **kwargs):
        parked.set()
        await park
        return await real_body(*args, **kwargs)

    monkeypatch.setattr(scheduler, "_run_tool_hook_gated_body", body_after_park)
    locks = owner._get_lock_manager()

    async def execute(name, arguments):
        # The dispatcher's cron route takes the source's declared resources
        # before it runs the tool, as it does for ``cron.training_cycle``.
        async with locks.acquire({ResourceLock.MEMORY}, label=f"cron.{name}"):
            return await scheduler._lookup_and_run_tool(name, arguments)

    runner = SchedulerRunner(database, owner.did, execute, owner_id="scheduler-4001")
    try:
        await runner._ensure_tables()
        due = (datetime.now(UTC) - timedelta(seconds=2)).isoformat()
        await database.execute(
            """INSERT INTO scheduled_tasks
            (id,agent_id,task_name,cron_expression,args_json,enabled,next_run_at,
             created_at,schedule_kind,timezone_name,misfire_policy,idempotency_key,
             scheduler_protocol_version)
            VALUES (?,?,?,?,?,1,?,?,'cron','UTC','skip',?,?)""",
            ("schedule-4001", owner.did, "training_cycle_4001", "0 3 * * *",
             json.dumps({}), due, due, "schedule-4001-source", SCHEDULER_PROTOCOL_VERSION),
        )
        tick = asyncio.create_task(tick_and_settle(runner))
        await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
        assert locks.is_held(ResourceLock.MEMORY)

        agent_id = getattr(owner, "agent_id", None) or owner.did
        target = build_runtime_stop_target(owner, agent_id=agent_id)
        disposition = await asyncio.wait_for(
            target.cancel(
                StopRequest(
                    scope=StopScope.AGENT,
                    actor_id="operator",
                    target=agent_id,
                    target_agent_id=agent_id,
                )
            ),
            SETTLE_SECONDS,
        )

        assert disposition is StopDisposition.STOPPED
        await asyncio.wait_for(tick, SETTLE_SECONDS)
        assert not locks.is_held(ResourceLock.MEMORY)
        assert bodies == []
        (status,) = await database.fetchone(
            "SELECT status FROM task_execution_log WHERE task_id=?", ("schedule-4001",)
        )
        assert status != "success"
    finally:
        await runner.stop()


async def test_an_unknown_settlement_commit_fails_readiness_until_it_resolves(
    tmp_path, monkeypatch
):
    """Final self-review of wo-4001-6: the write that settles a failed wake's
    task ends with its commit outcome unknown. The settlement pass retains
    that as the delivery's native control, and owner retirement refuses
    while it stands, but before the fix the heartbeat dropped the pass's
    result and the host's readiness probe still answered ``ok``. Readiness
    now fails, and ``/health/detailed`` names the delivery, until a
    settlement returns."""
    from fastapi import FastAPI
    from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message
    from kestrel_sdk.native_a2a.types import Role, SendMessageRequest, TaskState
    from starlette.requests import Request

    from kestrel_sovereign import server
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        DURABLE_COGNITION_CONSUMER_ID,
    )
    from tests.unit.test_a2a_native_input_fencing import (
        failed_submission_wake,
        fault_task_writes,
        heartbeat,
    )

    async with booted_agent(
        tmp_path / "agent", name="Recipient", provenance="test:4001-readiness",
        features=set(),
    ) as agent:
        _script_replies(agent, monkeypatch)
        gate = _gate_resident_drain(agent, monkeypatch)
        app = FastAPI()
        app.state.agent = agent
        probe = Request({"type": "http", "app": app, "state": {}})

        async def detailed():
            async def healthy(_agent):
                return {"status": "healthy", "checks": []}

            monkeypatch.setattr(server, "_agent_health_result", healthy)
            return await server._agent_detailed_health(agent)

        assert server.health_check(probe) == {"status": "ok", "agent_initialized": True}
        task = await agent.task_manager.create_task(
            SendMessageRequest(
                tenant=agent.task_manager.task_store.scope.tenant,
                message=new_text_message("Please do this", role=Role.ROLE_USER),
            ),
            creator_agent_id=SENDER,
            notify_submitted=False,
        )
        failed = await failed_submission_wake(
            agent.task_manager, agent.dispatcher, DURABLE_COGNITION_CONSUMER_ID, task
        )
        faults = [ExecutionCommitOutcomeError("unknown")]
        fault_task_writes(agent.task_manager, faults)
        try:
            await heartbeat(agent.dispatcher)

            assert faults == []
            assert agent.dispatcher.unresolved_cognition_control == (failed.delivery_id,)
            assert server.health_check(probe).status_code == 503
            report = await detailed()
            assert report["status"] == "unhealthy"
            (check,) = [
                c for c in report["checks"] if c["name"] == "durable_cognition_control"
            ]
            assert check["status"] == "fail" and failed.delivery_id in check["message"]
            assert await _sender_state(agent, task) == TaskState.TASK_STATE_SUBMITTED

            # The retirement drain's fixed retry returns: the task is settled,
            # and readiness recovers with the control.
            await agent.dispatcher._settle_retained_source_control(failed)

            assert agent.dispatcher.unresolved_cognition_control == ()
            assert server.health_check(probe) == {
                "status": "ok", "agent_initialized": True,
            }
            assert not [
                c for c in (await detailed())["checks"]
                if c["name"] == "durable_cognition_control"
            ]
            assert await _sender_state(agent, task) == TaskState.TASK_STATE_FAILED
        finally:
            gate.set()
        await _until_completion_wake_settled(agent)
