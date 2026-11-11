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
turn's privacy lifetime.

Every agent here is real, on real storage; only the model is scripted.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from kestrel_sdk.signals import (
    RedactionPolicy,
    ResourceLock,
    Signal,
    SignalMode,
    SourceRegistration,
)

from kestrel_sovereign.agent.invocation import InvocationCancelledError
from kestrel_sovereign.await_chain import format_await_chain
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
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


def _cognition_registration(tmp_path: Path, name: str) -> SourceRegistration:
    template = tmp_path / f"{name}.md"
    template.write_text("A signal woke you. Reply briefly.\n", encoding="utf-8")
    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=template,
        log_redaction=RedactionPolicy(
            summarize=lambda _payload: "test wake",
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
    )


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


@pytest.mark.parametrize("delivery", ["enqueue_signal", "enqueue_durable_cognition"])
async def test_a_signal_a_turn_dispatches_leaves_the_turn_queued_behind_it_runnable(
    tmp_path, monkeypatch, delivery
):
    """The incident's cycle: a turn enqueues a COGNITION wake for its own agent
    while another turn waits for CONVERSATION.

    Before the fix the enqueuing turn's retirement held the privacy lock until
    the wake's turn ended, the queued turn took CONVERSATION and waited for the
    privacy lock, and the wake waited for CONVERSATION. All three hung.
    ``enqueue_durable_cognition`` is the path an A2A wake takes.
    """
    from kestrel_sovereign.signals import DurableConsumerRegistration

    async with booted_agent(
        tmp_path / "agent", name="Waker", provenance="test:4001-dispatch", features=set()
    ) as agent:
        agent.signal_registry.register(_cognition_registration(tmp_path, "test.wake"))
        if delivery == "enqueue_durable_cognition":
            await agent.dispatcher.register_durable_consumer(
                DurableConsumerRegistration(
                    consumer_id="test-wake-consumer",
                    source="test.wake",
                    agent_id=agent.did,
                    max_attempts=0,
                )
            )

        async def dispatch(signal):
            if delivery == "enqueue_signal":
                return await agent.dispatcher.enqueue_signal(signal)
            return await agent.dispatcher.enqueue_durable_cognition(
                signal, source_event_id="test-wake-1", consumer_id="test-wake-consumer"
            )
        queued = asyncio.Event()
        wakes = []
        handles = []

        async def generate(kwargs):
            messages = kwargs.get("messages") or [{}]
            if any("A signal woke you" in str(m.get("content")) for m in messages):
                wakes.append(True)
            elif not handles:
                # The second turn is already waiting for CONVERSATION when this
                # turn hands the wake to the dispatcher.
                await asyncio.wait_for(queued.wait(), SETTLE_SECONDS)
                handles.append(await dispatch(
                    Signal(
                        source="test.wake",
                        kind="tick",
                        mode=SignalMode.COGNITION,
                        payload={},
                        target_agent=agent.did,
                    )
                ))
            return LLMResponse(content=REPLY)

        _script_replies(agent, monkeypatch, generate)
        first = asyncio.create_task(agent.process_input("one", session_id="s"))
        while agent._live_turn_id is None:
            await asyncio.sleep(0.01)
        second = asyncio.create_task(agent.process_input("two", session_id="s"))
        while agent._get_lock_manager().active_hold_diagnostics()[0][
            "blocked_acquirers"
        ] == 0:
            await asyncio.sleep(0.01)
        queued.set()

        done = await _settle(agent, (first, second), "the dispatching and queued turns")
        assert [task.result() for task in (first, second)] == [REPLY, REPLY]
        assert all(task in done for task in (first, second))
        deadline = asyncio.get_running_loop().time() + SETTLE_SECONDS
        (handle,) = handles
        await _settle(agent, (handle.task,), "the dispatched wake")
        while not wakes:
            if asyncio.get_running_loop().time() > deadline:
                pytest.fail("the dispatched wake never reached its turn\n" + _await_chains(agent))
            await asyncio.sleep(0.05)
        assert not agent._get_privacy_transition_lock().locked()


async def test_an_inbound_a2a_wake_is_not_enrolled_in_the_submitting_turn(
    tmp_path, monkeypatch
):
    """TaskManager calls ``_on_task_submitted`` on the submitting task: for a
    co-hosted sender, that sender's own turn. The wake belongs to the
    recipient, so the submitting turn's retirement must not join it."""
    import kestrel_sovereign.agent.event_manager as event_manager

    async with booted_agent(
        tmp_path / "agent", name="Recipient", provenance="test:4001-a2a", features=set()
    ) as agent:
        enrolled = []

        async def deliver(**kwargs):
            lock = agent._get_privacy_transition_lock()
            enrolled.append(lock._scope_for(asyncio.current_task()))

        monkeypatch.setattr(agent, "_deliver_a2a_wake_until_durable", deliver)
        monkeypatch.setattr(
            event_manager, "require_native_task_snapshot", lambda candidate, _did: candidate
        )
        candidate = SimpleNamespace(
            task=SimpleNamespace(id="task-4001"), input_revision=1
        )
        in_span = []

        async def generate(kwargs):
            lock = agent._get_privacy_transition_lock()
            in_span.append(lock._scope_for(asyncio.current_task()) is not None)
            agent._on_task_submitted(candidate)
            return LLMResponse(content=REPLY)

        _script_replies(agent, monkeypatch, generate)
        assert await agent.process_input("send it", session_id="s") == REPLY
        for _ in range(100):
            if enrolled:
                break
            await asyncio.sleep(0.01)

    assert in_span == [True], "the callback did not run inside the turn's privacy lifetime"
    assert enrolled == [None]


async def test_a_restart_wake_a_turn_sweeps_leaves_the_turn_queued_behind_it_runnable(
    tmp_path, monkeypatch
):
    """The restart coordinator's ACK supervisor awaits the wake it enqueued.

    ``!restart coordinator`` sweeps post-restart rows from inside a turn. The
    dispatch itself is detached, but before the fix its ACK supervisor was
    still a join obligation of that turn: the turn's retirement held the
    privacy lock waiting for the supervisor, the supervisor waited for the
    wake, the wake waited for CONVERSATION and the queued turn holding it
    waited for the privacy lock.
    """
    from kestrel_sovereign.features.restart_coordinator.store import (
        get_request,
        insert_request,
        update_status,
    )

    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", "1")
    monkeypatch.setenv("KESTREL_API_KEY", "test-4001-restart-ack")
    async with booted_agent(
        tmp_path / "agent",
        name="Restarted",
        provenance="test:4001-restart",
        features={"RestartCoordinatorFeature"},
    ) as agent:
        feature = next(
            f for f in agent.features.values()
            if type(f).__name__ == "RestartCoordinatorFeature"
        )
        request = await insert_request(
            feature._db,
            requested_by_agent=agent.did,
            reason="test #4001 restart wake",
            allow_agent_request=True,
        )
        assert await update_status(
            feature._db, request.id, status="completed", expected_current_status="pending"
        )
        queued = asyncio.Event()
        swept = []

        async def generate(kwargs):
            messages = kwargs.get("messages") or [{}]
            if messages[-1].get("role") == "tool" or swept:
                return LLMResponse(content=REPLY)
            # Only the first turn sweeps; it does so once the second turn is
            # waiting for CONVERSATION.
            swept.append(True)
            await asyncio.wait_for(queued.wait(), SETTLE_SECONDS)
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="sweep-1", name="restart_coordinator", arguments={})],
            )

        _script_replies(agent, monkeypatch, generate)
        first = asyncio.create_task(agent.process_input("sweep", session_id="s"))
        while agent._live_turn_id is None:
            await asyncio.sleep(0.01)
        second = asyncio.create_task(agent.process_input("two", session_id="s"))
        while agent._get_lock_manager().active_hold_diagnostics()[0][
            "blocked_acquirers"
        ] == 0:
            await asyncio.sleep(0.01)
        queued.set()

        await _settle(agent, (first, second), "the sweeping and queued turns")
        assert [task.result() for task in (first, second)] == [REPLY, REPLY]
        deadline = asyncio.get_running_loop().time() + SETTLE_SECONDS
        while not (await get_request(feature._db, request.id)).wake_delivered:
            if asyncio.get_running_loop().time() > deadline:
                pytest.fail(
                    "the restart wake's ACK supervisor never recorded delivery\n"
                    + _await_chains(agent)
                )
            await asyncio.sleep(0.05)
        assert not agent._get_privacy_transition_lock().locked()


async def test_stopping_a_queued_a2a_wake_cancels_the_task_its_sender_waits_on(
    tmp_path, monkeypatch
):
    """The 11-10 Claw shape: Stop cancels a queued turn carrying an inbound A2A
    work order. The task must end where its sender can see it, not stay
    ``submitted`` after its only wake was consumed."""
    from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message
    from kestrel_sdk.native_a2a.types import Role, SendMessageRequest, TaskState

    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        DURABLE_COGNITION_CONSUMER_ID,
    )

    sender = "did:kestrel:sender-4001"
    database, receipts = await _receipt_store(tmp_path)
    try:
        async with booted_agent(
            tmp_path / "agent", name="Recipient", provenance="test:4001-a2a-stop", features=set()
        ) as agent:
            _script_replies(agent, monkeypatch)
            parked = await _park_after_begin(agent, monkeypatch)
            first = asyncio.create_task(agent.process_input("one", session_id="s"))
            await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
            manager = agent.task_manager
            task = await manager.create_task(
                SendMessageRequest(
                    tenant=manager.task_store.scope.tenant,
                    message=new_text_message(
                        "Please do this", context_id="context-4001", role=Role.ROLE_USER
                    ),
                ),
                creator_agent_id=sender,
            )
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
            await _settle(agent, (first,), "the stopped turn")
            seen = await manager.get_task_for_creator(
                task.id, sender, recipient_agent_id=agent.did
            )
            assert seen.status.state == TaskState.TASK_STATE_CANCELED
            deliveries = await agent.dispatcher.list_durable_deliveries(
                consumer_id=DURABLE_COGNITION_CONSUMER_ID
            )
            assert [d.status for d in deliveries] == ["terminal_ackable"]
    finally:
        await database.close()


async def test_a_stopped_wake_cancels_only_the_input_it_carried(manager):
    """A Stop settles the exact input its wake carried. A newer input the
    sender added meanwhile stays live for its own wake."""
    from kestrel_sdk.native_a2a.types import TaskState

    from tests.unit.test_a2a_native_input_fencing import followup, owner, snapshot, wake
    from tests.unit.test_a2a_native_task_manager import request

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    stale = wake(await snapshot(manager, task), manager)
    await manager.continue_task(followup(manager, task), creator_agent_id="did:creator")
    recipient = owner(manager)

    await recipient.settle_stopped_cognition_signal(stale)
    current = await snapshot(manager, task)
    assert current.task.status.state == TaskState.TASK_STATE_SUBMITTED
    assert current.input_revision == 1

    await recipient.settle_stopped_cognition_signal(wake(current, manager))
    cancelled = await snapshot(manager, task)
    assert cancelled.task.status.state == TaskState.TASK_STATE_CANCELED
    assert "Stop cancelled" in cancelled.task.status.message.parts[0].text


async def test_a_stop_that_cannot_cancel_the_task_leaves_its_wake_to_retry(
    tmp_path, monkeypatch
):
    """When the task cannot be cancelled, the stop is not durably final: the
    wake is not consumed, so the work still runs, and Stop does not claim the
    work stopped."""
    from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message
    from kestrel_sdk.native_a2a.types import Role, SendMessageRequest, TaskState

    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        DURABLE_COGNITION_CONSUMER_ID,
    )

    sender = "did:kestrel:sender-4001"
    database, receipts = await _receipt_store(tmp_path)
    try:
        async with booted_agent(
            tmp_path / "agent", name="Recipient", provenance="test:4001-a2a-fail", features=set()
        ) as agent:
            _script_replies(agent, monkeypatch)
            parked = await _park_after_begin(agent, monkeypatch)
            first = asyncio.create_task(agent.process_input("one", session_id="s"))
            await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
            manager = agent.task_manager
            task = await manager.create_task(
                SendMessageRequest(
                    tenant=manager.task_store.scope.tenant,
                    message=new_text_message(
                        "Please do this", context_id="context-4001", role=Role.ROLE_USER
                    ),
                ),
                creator_agent_id=sender,
            )
            while agent._get_lock_manager().active_hold_diagnostics()[0][
                "blocked_acquirers"
            ] == 0:
                await asyncio.sleep(0.01)

            async def cancel_task(*args, **kwargs):
                raise OSError("database is locked")

            monkeypatch.setattr(manager, "cancel_task", cancel_task)
            request = StopRequest(
                scope=StopScope.AGENT,
                actor_id="operator",
                target=agent.agent_id,
                target_agent_id=agent.agent_id,
            )
            outcomes = await asyncio.wait_for(
                _authority(agent, receipts).stop(request), SETTLE_SECONDS
            )

            assert [outcome.disposition for outcome in outcomes] != [
                StopDisposition.STOPPED
            ], outcomes
            await _settle(agent, (first,), "the stopped turn")
            seen = await manager.get_task_for_creator(
                task.id, sender, recipient_agent_id=agent.did
            )
            assert seen.status.state == TaskState.TASK_STATE_SUBMITTED
            deliveries = await agent.dispatcher.list_durable_deliveries(
                consumer_id=DURABLE_COGNITION_CONSUMER_ID
            )
            assert [d.status for d in deliveries] == ["retry"], deliveries
    finally:
        await database.close()


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
    from datetime import datetime, timedelta, timezone

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
        due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
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
