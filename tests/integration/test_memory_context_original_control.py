"""Actual memory-context readers and accepted native rehearsal owners."""

import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.agent.context_manager import ContextManager
from kestrel_sovereign.agent.token_budget import TokenBudget
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    is_execution_control_error,
)
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_saved_items_original_control import (
    native_recall as native_recall,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["memory-manager", "context-producer"])
@pytest.mark.parametrize("wrapper", [None, ConnectionError, ValueError])
async def test_actual_memory_context_keeps_original_native_read_control(
    native_memory_owners, monkeypatch, door, wrapper
):
    n = native_memory_owners
    scope = ExecutionCustody(GenerationFence())
    n.backend._execution_custody = n.service._execution_custody = scope
    manager = ContextManager(
        n.pg_storage,
        agent_id=n.did,
        memory_retriever=n.pg_memory.retriever,
        llm_service=n.service,
        model="gpt-4o",
    )
    actual = n.pg_memory.retriever.retrieve
    if wrapper is not None:

        async def original_with_compatibility_carrier(*args, **kwargs):
            try:
                return await actual(*args, **kwargs)
            except BaseException as error:
                # Declared component compatibility carrier around the real
                # original native receipt, not a fabricated driver exception.
                if is_execution_control_error(error):
                    raise wrapper("declared retrieval compatibility carrier") from error
                raise

        monkeypatch.setattr(
            n.pg_memory.retriever, "retrieve", original_with_compatibility_carrier
        )
    original = OSError("actual memory-context read release ACK lost")
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    faulted = []

    async def lose_original_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if typ is None and not faulted:
            faulted.append(checkout)
            raise original
        return result

    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_original_ack)
    result = failure = None
    try:
        if door == "memory-manager":
            result = await manager.memory_manager.retrieve_memories(
                "retained local memory", 512, manager.counter, read_only=True
            )
        else:
            result = await manager._produce_memories(
                TokenBudget(model="gpt-4o"),
                "retained local memory",
                None,
                {"memory_min_score": 0, "memory_min_relevance": 0},
                include_memories=True,
                trivial_turn=False,
            )
    except BaseException as error:
        failure = error
    assert faulted, "actual original native retrieval was not reached"
    assert failure is not None and original in tuple(
        _explicit_control_chain(failure)
    ), (
        f"{door} converted native control into absent/warning memories: {result!r}, {failure!r}"
    )
    assert is_execution_control_error(failure)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door",
    [
        "completed-drain",
        "pending-drain",
        "shutdown",
        "retrieve-again",
        "schedule-again",
        "record-again",
        "emotion-again",
        "important-again",
        "access-again",
        "applied-again",
    ],
)
@pytest.mark.parametrize("carrier", ["error", "cancel"])
async def test_original_scheduled_memory_write_is_observed_after_native_ack_failure(
    native_recall, native_pg, monkeypatch, door, carrier
):
    backend, storage, memory, agent, feature, store = native_recall
    row = await storage.db.fetchval(
        "INSERT INTO conversation_history (agent_id,role,content,metadata) VALUES (?,?,?,?) RETURNING id",
        (agent.did, "assistant", "actual rehearsal", "{}"),
    )
    actual_sql = asyncpg.Connection.execute
    wrote, faulted = [], []
    original = (
        OSError("actual original rehearsal release ACK lost")
        if carrier == "error"
        else asyncio.CancelledError("actual original rehearsal release ACK lost")
    )

    async def observe_sql(connection, sql, *args, **kwargs):
        result = await actual_sql(connection, sql, *args, **kwargs)
        if "UPDATE conversation_history" in sql:
            wrote.append(sql)
        return result

    monkeypatch.setattr(asyncpg.Connection, "execute", observe_sql)
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    entered, release, done = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def lose_original_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if wrote and not faulted and typ is None:
            faulted.append(checkout)
            entered.set()
            await release.wait()
            raise original
        return result

    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_original_ack)
    retriever = memory.retriever
    task = retriever._schedule_access_update(row, agent.did)
    task.add_done_callback(lambda _: done.set())
    drain = followup = None
    iterated = []
    failure = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if door == "pending-drain":
            drain = asyncio.create_task(retriever.drain_access_updates())
            await asyncio.sleep(0)
            assert not drain.done()
        release.set()
        await asyncio.wait_for(done.wait(), 5)
        try:
            if drain is not None:
                await drain
            elif door == "shutdown":
                await retriever.shutdown()
            elif door == "retrieve-again":
                await retriever.retrieve("actual rehearsal", agent.did, read_only=True)
            elif door == "schedule-again":
                followup = retriever._schedule_access_update(row, agent.did)
            elif door == "record-again":

                def original_ids():
                    # Declared caller callback input: old control must be
                    # checked BEFORE consuming even its lazy ID iterable.
                    iterated.append(True)
                    yield row

                await retriever.record_accesses(original_ids(), agent.did)
            elif door == "emotion-again":
                await retriever.retrieve_by_emotion(agent.did, "joy")
            elif door == "important-again":
                await retriever.retrieve_important(agent.did)
            elif door == "access-again":
                await retriever.update_access(row, agent.did)
            elif door == "applied-again":
                await retriever.update_applied(row, agent.did)
            else:
                await retriever.drain_access_updates()
        except BaseException as error:
            failure = error
        assert task.done() and wrote and faulted
        assert (
            await native_pg[1].fetchval(
                f"SELECT (metadata::jsonb->>'access_count')::int FROM \"{native_pg[2]}\".conversation_history WHERE id=$1",
                row,
            )
            == 1
        )
        assert failure is not None and original in tuple(
            _explicit_control_chain(failure)
        ), f"{door} discarded original scheduled native write control: {failure!r}"
        assert is_execution_control_error(failure)
        assert not iterated, (
            "retired memory consumer re-entered the caller's ID callback"
        )
    finally:
        release.set()
        await await_owned_task(task)
        if drain is not None:
            await await_owned_task(drain)
        if followup is not None:
            await await_owned_task(followup)


@pytest.mark.asyncio
async def test_cancelled_original_drain_keeps_both_native_control_and_caller_without_hot_history(
    native_recall, native_pg, monkeypatch
):
    backend, storage, memory, agent, feature, store = native_recall
    row = await storage.db.fetchval(
        "INSERT INTO conversation_history (agent_id,role,content,metadata) VALUES (?,?,?,?) RETURNING id",
        (agent.did, "assistant", "original cancelled drain", "{}"),
    )
    actual_sql, actual_exit = (
        asyncpg.Connection.execute,
        asyncpg.pool.PoolAcquireContext.__aexit__,
    )
    wrote, faulted, forwarded, caller_receipts = [], [], [], []
    original = OSError("actual original cancelled-drain release ACK lost")
    entered, release, child_forwarded = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    async def observe_sql(connection, sql, *args, **kwargs):
        result = await actual_sql(connection, sql, *args, **kwargs)
        if "UPDATE conversation_history" in sql:
            wrote.append(sql)
        return result

    async def original_ack_owner(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if wrote and not faulted and typ is None:
            faulted.append(checkout)
            entered.set()
            # The actual physical release owner is shielded by the backend;
            # cancellation of its request must not interrupt this native tail.
            await release.wait()
            raise original
        return result

    from kestrel_sovereign.storage import memory_retriever as retriever_module

    actual_join = retriever_module.await_owned_task

    async def original_join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            caller_receipts.append(result.cancellation)
        return result

    monkeypatch.setattr(asyncpg.Connection, "execute", observe_sql)
    monkeypatch.setattr(
        asyncpg.pool.PoolAcquireContext, "__aexit__", original_ack_owner
    )
    monkeypatch.setattr(retriever_module, "await_owned_task", original_join)
    retriever = memory.retriever
    task = retriever._schedule_access_update(row, agent.did)
    actual_cancel = task.cancel

    def original_child_cancel(*args, **kwargs):
        result = actual_cancel(*args, **kwargs)
        forwarded.append(result)
        child_forwarded.set()
        return result

    monkeypatch.setattr(task, "cancel", original_child_cancel)
    drain = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        drain = asyncio.create_task(retriever.drain_access_updates())
        await asyncio.sleep(0)
        drain.cancel("first original drain caller withdrawal")
        await asyncio.wait_for(child_forwarded.wait(), 5)
        drain.cancel("second original drain caller withdrawal")
        release.set()
        outcome = await await_owned_task(drain)
        assert task.done() and forwarded == [True]
        assert caller_receipts
        terminal = outcome.error
        chain = tuple(_explicit_control_chain(terminal))
        assert original in chain and caller_receipts[0] in chain
        assert is_execution_control_error(terminal)
        retained = retriever._access_update_control
        assert caller_receipts[0] not in tuple(_explicit_control_chain(retained)), (
            "finite caller cancellation polluted the hot original receipt"
        )
        for _ in range(10):
            with pytest.raises(BaseException) as caught:
                await retriever.drain_access_updates()
            assert caught.value is retained
            assert caller_receipts[0] not in tuple(_explicit_control_chain(retained))
    finally:
        release.set()
        await await_owned_task(task)
        if drain is not None:
            await await_owned_task(drain)
