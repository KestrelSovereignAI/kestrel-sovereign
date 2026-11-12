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
reaching a parked turn or a parked scheduled tool, and the locks a failed
required audit leaves behind.

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
