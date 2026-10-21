"""Actual bound rehearsal SQL must settle before context publication."""

import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.agent.context_manager import ContextManager
from kestrel_sovereign.agent.context_stages import (
    ContextAssembly,
    ContextBuildMode,
    ContextBuildPlan,
)
from kestrel_sovereign.execution_custody import ExecutionCustody
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_saved_items_original_control import (
    native_recall as native_recall,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door", ["record-accesses", "retrieve", "memory-manager", "context-plan"]
)
async def test_bound_actual_memory_write_precedes_context_publication(
    native_memory_owners, monkeypatch, door
):
    n = native_memory_owners
    row = await n.pg_storage.db.fetchval(
        "INSERT INTO conversation_history (agent_id,role,content,metadata) VALUES (?,?,?,?) RETURNING id",
        (n.did, "assistant", "trusted blue memory", '{"importance":0.9}'),
    )
    scope = ExecutionCustody(GenerationFence())
    n.backend._execution_custody = n.service._execution_custody = scope
    retriever = n.pg_memory.retriever
    manager = ContextManager(
        n.pg_storage,
        agent_id=n.did,
        memory_retriever=retriever,
        llm_service=n.service,
        model="gpt-4o",
    )
    actual_sql, actual_exit = (
        asyncpg.Connection.execute,
        asyncpg.pool.PoolAcquireContext.__aexit__,
    )
    wrote, acknowledgements = [], []
    entered, release = asyncio.Event(), asyncio.Event()

    async def observe_sql(connection, sql, *args, **kwargs):
        result = await actual_sql(connection, sql, *args, **kwargs)
        if "UPDATE conversation_history" in sql:
            wrote.append(sql)
        return result

    async def hold_original_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if wrote and not acknowledgements and typ is None:
            acknowledgements.append(checkout)
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(asyncpg.Connection, "execute", observe_sql)
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", hold_original_ack)
    if door == "record-accesses":
        operation = retriever.record_accesses([row], n.did)
    elif door == "retrieve":
        operation = retriever.retrieve(
            "blue memory", n.did, min_score=0, min_relevance=0
        )
    elif door == "memory-manager":
        operation = manager.memory_manager.retrieve_memories(
            "blue memory",
            1024,
            manager.counter,
            min_score=0,
            min_relevance=0,
        )
    else:
        plan = ContextBuildPlan(
            mode=ContextBuildMode.LIVE,
            model="gpt-4o",
            assembly=ContextAssembly(),
            sections={},
            budget_summary={},
            context_limit=4096,
            response_reserve=512,
            total_budget=3584,
            total_tokens=0,
            memory_access_ids=(row,),
        )
        operation = manager._commit_and_render_plan(plan)
    task = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not task.done(), f"{door} published before original native write ACK"
        release.set()
        outcome = await await_owned_task(task)
        assert outcome.error is outcome.cancellation is None
        assert not retriever._access_update_tasks
        assert (
            await n.observer.fetchval(
                f"SELECT (metadata::jsonb->>'access_count')::int FROM \"{n.schema}\".conversation_history WHERE id=$1",
                row,
            )
            == 1
        )
        assert (
            await n.backend.fetch_val("SELECT generation FROM authority WHERE id=1")
            == 1
        )
    finally:
        release.set()
        await await_owned_task(task)
        await retriever.drain_access_updates()


@pytest.mark.asyncio
async def test_actual_ordinary_memory_write_failure_remains_best_effort(
    native_recall, monkeypatch, caplog
):
    backend, storage, memory, agent, feature, store = native_recall
    row = await storage.db.fetchval(
        "INSERT INTO conversation_history (agent_id,role,content,metadata) VALUES (?,?,?,?) RETURNING id",
        (agent.did, "assistant", "ordinary unavailable ledger", "{}"),
    )
    # Genuine partial schema in this test's owned disposable namespace: the
    # driver's failed UPDATE and successful rollback are not control/ACK loss.
    await storage.db.execute("ALTER TABLE conversation_history DROP COLUMN metadata")
    await memory.retriever.record_accesses([row], agent.did)
    assert memory.retriever._access_update_control is None
    assert not memory.retriever._access_update_tasks
    assert any(
        "update_access failed" in record.getMessage() for record in caplog.records
    )
