"""Startup retirement cannot lose actual owners when Task admission fails.

Actual PostgreSQL opens and commits; no agent genesis, provider call or READY
is manufactured. A discovery-only fixture input selects the storage branch;
it does not claim the empty actual manager owns a born agent.
"""

import asyncio
import inspect
import os

import pytest
from fastapi import FastAPI

from kestrel_sovereign.execution_custody import (
    ExecutionCustody, _explicit_control_chain, execution_cleanup_debts,
    execution_commit_outcome, is_execution_control_error,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.postgres_schema import with_search_path


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["pool", "preflight", "scheduler", "manager"])
@pytest.mark.parametrize("refuse", [False, True])
async def test_startup_refused_original_cleanup_keeps_native_owner_and_body(
    native_pg, tmp_path, monkeypatch, route, refuse,
):
    from kestrel_sovereign import server

    backend, observer, schema = native_pg
    original = None
    if route != "manager":
        backend._execution_custody = ExecutionCustody(GenerationFence())
        with monkeypatch.context() as ack:
            receipts = lose_native_ack(ack, boundary="commit", original=OSError("actual startup COMMIT ACK lost"))
            try:
                await backend.execute("INSERT INTO effects VALUES (182, 'committed original startup')")
            except BaseException as error:
                original = error
            assert receipts == [True] and execution_commit_outcome(original) == "unknown"
        backend._execution_custody = None
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 1
    original_graph = tuple(_explicit_control_chain(original))
    manager = AgentManager(base_data_dir=tmp_path / "actual-empty-manager")
    app = FastAPI()
    owners, native_owners, refused = [], [], []
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    allocation = RuntimeError("actual startup retirement Task admission refused")
    expected = {"pool": "PostgresBackend.close", "preflight": "AsyncStorage.close",
                "scheduler": "_shutdown_host_scheduler", "manager": "AgentManager.shutdown_all"}[route]

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if refuse and not refused and code is not None and code.co_qualname == expected:
            refused.append((coroutine, operation))
            raise allocation
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    async def selected_discovery_only(_config):
        # Only selects storage initialization. No agent is loaded or invoked.
        return {"did:test:unloaded-input-only": ("unloaded-input-only", None)}

    actual_connect, actual_initialize = PostgresBackend.connect, AsyncStorage.initialize

    async def connect_then_fail(owner):
        result = await actual_connect(owner)
        if route == "pool":
            owners.append(owner)
            native_owners.append(owner)
            raise original
        return result

    async def initialize_then_fail(owner):
        result = await actual_initialize(owner)
        owners.append(owner)
        native_owners.append(owner.db.backend)
        raise original

    caught = None
    try:
        with monkeypatch.context() as patch:
            patch.setenv("KESTREL_DB_BACKEND", "postgres")
            patch.setenv("KESTREL_DATABASE_URL", with_search_path(os.environ["TEST_POSTGRES_URL"], schema))
            patch.setenv("KESTREL_SHARED_AGENT_POSTGRES_MAX_POOL_SIZE", "2")
            patch.setenv("KESTREL_SHARED_AGENT_POSTGRES_ADVISORY_MAX_POOL_SIZE", "1")
            patch.setattr(manager, "local_agent_configs_by_did", selected_discovery_only)
            if route == "pool":
                patch.setattr(PostgresBackend, "connect", connect_then_fail)
            elif route in {"preflight", "scheduler"}:
                patch.setattr(AsyncStorage, "initialize", initialize_then_fail)
            else:
                owners.append(manager)
            loop.set_task_factory(factory)
            try:
                if route == "pool":
                    await server._start_shared_agent_postgres_backend(app)
                elif route == "preflight":
                    await server._prepare_shared_postgres_scheduler_protocol(app, manager, None)
                elif route == "scheduler":
                    await server._start_host_scheduler(app, manager, None)
                else:
                    await server._rollback_startup_agent_manager(manager)
            except BaseException as error:
                caught = error
            finally:
                loop.set_task_factory(previous)
        assert bool(refused) is refuse
        if not refuse:
            assert all(not owner.is_connected for owner in native_owners), "accepted startup cleanup did not physically retire its original pool"
            assert not execution_cleanup_debts(caught)
            if route == "manager":
                assert caught is None
            else:
                assert caught is not None and execution_commit_outcome(caught) == "unknown"
                assert original in tuple(_explicit_control_chain(caught)), "startup cleanup erased its original body receipt"
                assert tuple(_explicit_control_chain(original)) == original_graph, "startup cleanup mutated its original body receipt"
            if route == "pool":
                assert app.state.shared_agent_postgres_backend is None
            elif route == "scheduler":
                assert app.state.host_scheduler_storage is None
            return
        assert caught is not None, "startup cleanup admission silently reported successful retirement"
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
                   for pair in refused for coro in pair), "startup admission left an unstarted original cleanup coroutine open"
        assert is_execution_control_error(caught), "startup cleanup admission lost native retirement control"
        assert allocation in tuple(_explicit_control_chain(caught)), "startup cleanup discarded its original admission failure"
        assert any(resource is owners[0] and task is None
                   for resource, task in execution_cleanup_debts(caught)), "startup cleanup admission discarded its actual original owner"
        if original is not None:
            assert execution_commit_outcome(caught) == "unknown", "startup cleanup admission replaced original native COMMIT uncertainty"
            assert original in tuple(_explicit_control_chain(caught)), "startup cleanup erased its original body receipt"
            assert tuple(_explicit_control_chain(original)) == original_graph, "startup cleanup mutated its original body receipt"
        if route == "pool":
            assert app.state.shared_agent_postgres_backend is owners[0]
            assert owners[0].is_connected
        elif route in {"preflight", "scheduler"}:
            assert owners[0].db.backend.is_connected
            if route == "scheduler":
                assert app.state.host_scheduler_storage is owners[0]
        assert not manager.list_agents()
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        # Only these exact actual original owners, after custody assertions.
        for owner in owners:
            if owner is manager:
                await manager.shutdown_all()
            else:
                await owner.close()
        backend._execution_custody = None
