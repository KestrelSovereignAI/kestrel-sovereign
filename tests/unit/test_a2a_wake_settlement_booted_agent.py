"""R9 (#4041, finding 7): a real agent never keeps a wake settlement's control.

Booted ``KestrelAgent`` on SQLite and PostgreSQL. A settlement pass is
cancelled while its task write reports a commit whose outcome is unknown; the
per-delivery entry is the only place that control lives. Another background
task that fails with an error derived from it is harvested without the agent
keeping a copy. Once a later pass reconciles the entry, nothing in the
agent-wide control inventory remains.
"""

import asyncio

import pytest
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message
from kestrel_sdk.native_a2a.types import Role, SendMessageRequest, TaskState

from kestrel_sovereign.a2a.stores.unified.wake_obligation_store import ENDED
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_finite_error,
)
from kestrel_sovereign.signals.sources.a2a_task_submitted import build_signal_for_submitted_task
from tests.shared.booted_agent import booted_agent
from tests.unit.test_turn_parked_before_first_llm_call import (  # noqa: F401
    agent_backend,
    native_pg,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(90)]


async def test_real_agent_harvester_does_not_retain_settlement_control(
    tmp_path, native_pg
):
    async with booted_agent(
        tmp_path / "agent", name="Kite", provenance="test:4041-r9",
        features=set(), database_url=native_pg,
    ) as agent:
        manager, dispatcher = agent.task_manager, agent.dispatcher
        manager._on_task_submitted = None
        task = await manager.create_task(
            SendMessageRequest(
                tenant=manager.task_store.scope.tenant,
                message=new_text_message("Settle me", role=Role.ROLE_USER),
            ),
            creator_agent_id="did:test:peer",
        )
        current = await manager.get_wake_obligation(task.id, 0)
        candidate = await manager.get_task_snapshot(task.id, recipient_agent_id=agent.did)
        persisted = await dispatcher._durable_store.persist_signal(
            build_signal_for_submitted_task(candidate, target_agent=agent.did),
            agent_id=agent.did, source_event_id=current.identity, retention_days=14,
        )
        [delivery_id] = persisted.delivery_ids
        row = await dispatcher._durable_store.get_delivery(
            agent_id=agent.did, consumer_id=current.consumer_id, delivery_id=delivery_id,
        )
        claimed = await dispatcher._durable_store.claim_delivery_for_event(
            agent_id=agent.did, consumer_id=current.consumer_id,
            event_id=row.event_id, executor_id=dispatcher._durable_delivery_owner,
        )
        await dispatcher._durable_store.nack_delivery(
            agent_id=agent.did, consumer_id=current.consumer_id,
            delivery_id=claimed.delivery_id, lease_token=claimed.lease_token,
            error="wake failed", terminal=True,
        )
        entered = asyncio.Event()
        real = agent.settle_wake_task

        async def commit_unknown_while_cancelled(obligation, reason):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                raise ExecutionCommitOutcomeError("unknown")

        agent.settle_wake_task = commit_unknown_while_cancelled
        try:
            # The owner tick's own pass: agent-tracked, harvested by the agent.
            running = dispatcher._start_wake_settlement_pass()
            assert running is not None
            await asyncio.wait_for(entered.wait(), 10)
            running.cancel()
            await asyncio.wait({running}, timeout=10)
            assert running.done()
            assert set(dispatcher._source_settlement_control) == {delivery_id}
            agent._raise_shutdown_control()
            # Another observer of the same lost commit fails in the agent's
            # own background work: its error's native source is the entry.
            # The harvester refuses it, so the agent keeps no copy it could
            # never clear.
            [(_, carrier)] = dispatcher._source_settlement_control.values()
            observed = execution_finite_error(
                carrier, asyncio.CancelledError(),
                retain_originals=True, isolate_control_originals=True,
            )
            assert observed is not carrier
            assert dispatcher.is_settlement_control(observed)

            async def observes_the_lost_commit():
                raise observed

            observer = agent._track_background_task(
                observes_the_lost_commit(), name="observes-settlement-commit"
            )
            await asyncio.wait({observer}, timeout=10)
            assert observer.done() and observer not in agent._background_tasks
            agent._raise_shutdown_control()
        finally:
            agent.settle_wake_task = real
        # Reconciliation is the next pass; it settles and clears the entry.
        await dispatcher.run_wake_settlement_pass()
        assert dispatcher._source_settlement_control == {}
        assert (await manager.get_wake_obligation(task.id, 0)).state == ENDED
        snapshot = await manager.get_task_snapshot(task.id, recipient_agent_id=agent.did)
        assert snapshot.task.status.state == TaskState.TASK_STATE_FAILED
        assert dispatcher._outcome_log_control_error is None
        agent._raise_shutdown_control()
