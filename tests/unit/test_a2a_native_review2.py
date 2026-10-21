"""Full-review regressions using real native ledgers and feature dispatch."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from a2a.types import TaskState
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.a2a.projections import embedded_json_value
from kestrel_sovereign.a2a.question_supervisor import NativeQuestionSupervisor
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.skill_handler import TaskHandler
from kestrel_sovereign.a2a.stores.unified.outbound_store import DispatchState
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.restart_coordinator.feature import (
    RestartCoordinatorFeature,
)
from tests.unit.test_a2a_native_question_store import questions as questions
from tests.unit.test_a2a_native_question_store import reserve, response
from tests.unit.test_a2a_native_question_supervisor import dispatcher_for
from tests.utils.native_a2a import manager_for


@pytest.mark.parametrize("fault", ["correlation", "context", "missing-id"])
async def test_invalid_peer_answer_does_not_strand_an_unrelated_deadline(
    questions, fault
):
    p = questions
    bad_q, bad_d, request = await reserve(p, "invalid-answer")
    submitting = await p.outbound.begin_submission(bad_d, operation_id="original")
    active = response(bad_q, request, state=TaskState.TASK_STATE_WORKING)
    await p.outbound.acknowledge(submitting, active, operation_id="working")
    invalid = response(bad_q, request)
    if fault == "correlation":
        invalid.status.message.ClearField("metadata")
    elif fault == "context":
        invalid.context_id = "another-context"
    else:
        invalid.id = ""
    other, _, _ = await reserve(p, "unrelated")

    class Peer:
        async def get_task(self, dispatch):
            await p.store.acknowledge(dispatch, invalid, operation_id="invalid")
            raise AssertionError("Invalid answer must fail before committing a receipt")

    @asynccontextmanager
    async def open_dispatch(dispatch_id):
        yield Peer(), await p.outbound.get(dispatch_id)

    agent, dispatcher = await dispatcher_for(p)
    supervisor = NativeQuestionSupervisor(
        store=p.store,
        open_dispatch=open_dispatch,
        dispatcher=dispatcher,
        interval=1,
        is_current=lambda: True,
    )
    try:
        await supervisor._read_pass()
        assert (
            await p.outbound.get(bad_d.dispatch_id)
        ).state is DispatchState.ACKNOWLEDGED
        assert not (await p.store.get(bad_q.question_id)).receiver_settled
        p.clock.now += timedelta(seconds=11)
        await supervisor._deadline_pass()
        await supervisor.handoff_wakes()
        assert (await p.store.get(other.question_id)).status == "EXPIRED"
        event_id = (
            await p.backend.fetch_one(
                "SELECT event_id FROM a2a_native_question_wakes WHERE question_id = ?",
                (other.question_id,),
            )
        )[0]
        assert (await p.store.wake_for_event(event_id)).handed_off_at
    finally:
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


@pytest.mark.parametrize("sender_wins", [True, False])
async def test_expiry_and_original_send_race_uses_atomic_non_delivery_evidence(
    questions, sender_wins
):
    p = questions
    q, dispatch, _ = await reserve(p)
    p.clock.now += timedelta(seconds=11)
    await p.store.expire(q.question_id)
    if sender_wins:
        await p.outbound.begin_submission(dispatch, operation_id="original")
    assert await p.store.settle_unsent(q.question_id, dispatch) is (not sender_wins)
    assert bool((await p.store.get(q.question_id)).receiver_settled) is (
        not sender_wins
    )
    assert [wake.wave for wake in await p.store.pending_wakes()] == ["deadline"]


async def test_resident_question_service_does_not_block_idle_restart_but_cognition_does():
    async def resident():
        await asyncio.Event().wait()

    daemon = asyncio.create_task(resident(), name="native_a2a_question_recovery")
    work = asyncio.create_task(
        resident(), name="durable_cognition:a2a.question_answered:event"
    )
    agent = SimpleNamespace(_active_request_ids=set(), _background_tasks={daemon})
    coordinator = RestartCoordinatorFeature(agent)
    try:
        assert coordinator._agent_appears_idle()["idle"]
        agent._background_tasks.add(work)
        busy = coordinator._agent_appears_idle()
        assert not busy["idle"] and work.get_name() in busy["reason"]
        assert daemon.get_name() not in busy["reason"]
    finally:
        daemon.cancel()
        work.cancel()
        await asyncio.gather(daemon, work, return_exceptions=True)


class LocalFeature(Feature):
    @property
    def tool_description(self):
        return "Exercise local typed skill dispatch"

    async def initialize(self):
        pass

    @tool(name="local_int", description="Preserve local Python integers")
    async def local_int(self, limit: int, nested: dict) -> ToolResult:
        self.observed = (limit, nested)
        assert type(limit) is int and type(nested["count"]) is int
        return ToolResult.ok(confirmation="Local integers retained")

    @tool(name="local_datetime", description="Serialize an existing rich result")
    async def local_datetime(self) -> dict:
        self.effects += 1
        return {"at": datetime(2026, 10, 20, tzinfo=UTC)}

    @tool(name="local_nonfinite", description="Exercise an unencodable result")
    async def local_nonfinite(self) -> dict:
        self.effects += 1
        return {"invalid": float("nan")}

    @tool(name="local_failure", description="Retain a tool's error text")
    async def local_failure(self) -> ToolResult:
        return ToolResult.failed("The provider produced nothing")


def local_feature(manager):
    feature = LocalFeature(SimpleNamespace(did=manager.did, _execution_custody=None))
    feature.name = "local"
    feature.effects = 0
    assert isinstance(feature, TaskHandler)
    manager.register_provider(
        "local", feature.get_agent_skills(), feature, {"!local-fail": "local_failure"}
    )
    return feature


@pytest.mark.parametrize("sync", [False, True])
async def test_local_skill_arguments_never_round_trip_through_struct(
    native_a2a_backend_factory, sync
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, NativeRecipientScope("did:local"))
    try:
        await manager.initialize()
        feature = local_feature(manager)
        args = {"limit": 2**53 + 1, "nested": {"count": 2}}
        task = await manager.execute_skill("local", "local_int", args, sync=sync)
        await manager.drain_execution_tasks()
        stored = await manager.get_task_for_recipient(task.id, manager.did)
        assert stored.status.state == TaskState.TASK_STATE_COMPLETED
        assert feature.observed == (2**53 + 1, {"count": 2})
        assert args == {"limit": 2**53 + 1, "nested": {"count": 2}}
    finally:
        await manager.close()


async def test_rich_tool_result_completes_once_and_command_retains_failure_text(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, NativeRecipientScope("did:local"))
    try:
        await manager.initialize()
        feature = local_feature(manager)
        task = await manager.execute_skill("local", "local_datetime", {})
        assert task.status.state == TaskState.TASK_STATE_COMPLETED
        assert feature.effects == 1
        data = embedded_json_value(task.artifacts[0].parts[0])[1]
        assert data["result"]["at"] == "2026-10-20 00:00:00+00:00"
        command = await manager.execute_command("!local-fail")
        assert command == {"success": False, "error": "The provider produced nothing"}
        rows = await backend.fetch_all("SELECT status FROM a2a_tasks")
        assert sorted(row[0] for row in rows) == ["completed", "failed"]
    finally:
        await manager.close()


async def test_unresolved_reads_back_off_without_delaying_deadline_or_explicit_recovery(
    questions,
):
    p = questions
    q, dispatch, _ = await reserve(p)
    submitting = await p.outbound.begin_submission(
        dispatch, operation_id="lost-original"
    )
    unknown = await p.outbound.mark_unknown(submitting, operation_id="lost-ack")
    clock = SimpleNamespace(now=0)
    reads = []

    class UnresolvedPeer:
        async def reconcile(self, current):
            reads.append(clock.now)
            return None

    @asynccontextmanager
    async def open_dispatch(dispatch_id):
        yield UnresolvedPeer(), unknown

    supervisor = NativeQuestionSupervisor(
        store=p.store,
        open_dispatch=open_dispatch,
        dispatcher=None,
        interval=1,
        max_interval=8,
        clock=lambda: clock.now,
        is_current=lambda: True,
    )
    for second in range(19):
        clock.now = second
        await supervisor._read_pass()
    assert reads == [0, 2, 6, 14]
    p.clock.now += timedelta(seconds=11)
    await supervisor._deadline_pass()
    assert (await p.store.get(q.question_id)).status == "EXPIRED"
    assert [wake.wave for wake in await p.store.pending_wakes()] == ["deadline"]
    await supervisor._read_pass(force=True)
    assert reads == [0, 2, 6, 14, 18]
    assert (await p.outbound.get(dispatch.dispatch_id)).state is DispatchState.UNKNOWN


async def test_terminal_cognition_failure_is_retained_without_ack_or_endless_readmission(
    questions,
):
    import json

    from kestrel_sovereign.signals.sources.a2a_question_answered import (
        DURABLE_COGNITION_CONSUMER_ID,
        build_signal_for_question_answered,
    )

    p = questions
    q, dispatch, request = await reserve(p)
    submitting = await p.outbound.begin_submission(dispatch, operation_id="original")
    p.clock.now += timedelta(seconds=11)
    await p.store.acknowledge(
        submitting, response(q, request), operation_id="late-answer"
    )
    deadline, late = await p.store.pending_wakes()
    agent, dispatcher = await dispatcher_for(p)
    release = asyncio.Event()

    async def cognition(prompt):
        await release.wait()
        return "Handled"

    agent.process_input = cognition
    supervisor = NativeQuestionSupervisor(
        store=p.store,
        open_dispatch=None,
        dispatcher=dispatcher,
        interval=1,
        is_current=lambda: True,
    )
    try:
        signal = build_signal_for_question_answered(
            event_id=deadline.event_id,
            payload=json.loads(deadline.payload_json),
            target_agent=agent.did,
        )
        persisted = await dispatcher._durable_store.persist_signal(
            signal,
            agent_id=agent.did,
            source_event_id=deadline.event_id,
            retention_days=7,
        )
        claimed = await dispatcher.claim_durable_delivery_for_event(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID,
            event_id=persisted.event_id,
            executor_id=dispatcher._durable_delivery_owner,
        )
        assert claimed is not None
        failed = await dispatcher.nack_durable_delivery(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID,
            delivery_id=claimed.delivery_id,
            lease_token=claimed.lease_token,
            error="Terminal worker failure",
            terminal=True,
        )
        assert failed.status == "failed"
        await supervisor.handoff_wakes()
        # The failed deadline stays failed without an ACK. It cannot strand
        # an independently owed late answer behind a permanent admission retry.
        retained = await p.store.wake_for_event(deadline.event_id)
        assert retained.delivery_failed_at and retained.handed_off_at is None
        release.set()
        assert not await p.store.pending_wakes()
        assert (await p.store.wake_for_event(late.event_id)).handed_off_at
        await supervisor.handoff_wakes()
        assert (
            await p.backend.fetch_one("SELECT COUNT(*) FROM durable_signal_events")
        )[0] == 2
    finally:
        release.set()
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


async def test_nonfinite_result_fails_durably_after_one_effect_without_stranding_submission(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, NativeRecipientScope("did:local"))
    try:
        await manager.initialize()
        feature = local_feature(manager)
        result = await manager.execute_skill("local", "local_nonfinite", {})
        assert feature.effects == 1
        assert result.status.state == TaskState.TASK_STATE_FAILED
        assert not result.artifacts
        stored = await manager.get_task_for_recipient(result.id, manager.did)
        assert stored == result
    finally:
        await manager.close()
