"""A bounded server Stop close cannot erase a still-live native settlement."""

import asyncio

import asyncpg
import pytest
from fastapi import FastAPI

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_cleanup_debts,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.stop import (
    DistributedInvocationRegistry,
    DistributedInvocationStore,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_cleanup_task_admission import _partial_native_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["distributed", "receipts"])
async def test_bounded_original_stop_close_keeps_pending_native_settlement(
    native_pg, tmp_path, monkeypatch, route
):
    from kestrel_sovereign import server
    from kestrel_sovereign.stop import invocation

    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    agent, agent_db = await _partial_native_agent(tmp_path, monkeypatch)
    registry = DistributedInvocationRegistry(store)
    registry.attach(agent)
    assert await registry.register(agent, "original-server-stop-retirement", 1)
    (generation_id,) = tuple(registry._active)
    backend._execution_custody = ExecutionCustody(GenerationFence())
    native_exit, committed, release = (
        asyncpg.transaction.Transaction.__aexit__,
        asyncio.Event(),
        asyncio.Event(),
    )
    original = OSError("actual distributed Stop settlement COMMIT ACK lost")
    commits = []

    async def held_ack(transaction, error_type, error, tb):
        result = await native_exit(transaction, error_type, error, tb)
        if (
            error_type is None
            and not transaction._nested
            and asyncio.current_task()
            .get_name()
            .startswith("distributed-stop-complete:")
            and not commits
        ):
            commits.append(True)
            committed.set()
            await release.wait()
            raise original
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_ack)
    monkeypatch.setattr(invocation, "_DEFAULT_CLOSE_DRAIN_SECONDS", 0.02)
    app = FastAPI()
    app.state.distributed_invocation_registry = registry
    app.state.stop_receipt_db = db
    owner = None
    try:
        registry.complete_soon(agent, "original-server-stop-retirement", 1)
        (owner,) = tuple(registry._cleanup_tasks)
        await asyncio.wait_for(committed.wait(), 5)
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".stop_active_invocations WHERE generation_id=$1',
                generation_id,
            )
            == 0
        )
        caught = None
        try:
            await (
                server._shutdown_distributed_invocations(app)
                if route == "distributed"
                else server._shutdown_stop_receipts(app)
            )
        except BaseException as error:
            caught = error
        assert not owner.done() and owner.cancelling() == 0
        assert app.state.distributed_invocation_registry is registry, (
            "bounded server close erased the actual registry while its native settlement remained live"
        )
        assert app.state.stop_receipt_db is db
        assert caught is not None and is_execution_control_error(caught)
        assert any(
            resource is registry and closer is owner
            for resource, closer in execution_cleanup_debts(caught)
        )
        release.set()
        result = await await_owned_task(owner)
        assert (
            result.error is not None
            and execution_commit_outcome(result.error) == "unknown"
        )
        assert execution_commit_outcome(registry._terminal_control) == "unknown"
        assert not agent._runtime_publication_ready.is_set()
    finally:
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        try:
            await registry.close()
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
        backend._execution_custody = None
        await agent.shutdown()
        await agent_db.close()
        await db.close()
