"""Startup retirement cannot lose actual owners when Task admission fails.

Actual PostgreSQL opens and commits; no agent genesis, provider call or READY
is manufactured. A discovery-only fixture input selects the storage branch;
it does not claim the empty actual manager owns a born agent.
"""

import asyncio
import inspect
import os
from types import SimpleNamespace

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


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["backend", "preflight", "scheduler"])
@pytest.mark.parametrize("pool_kind", ["primary", "advisory"])
async def test_accepted_startup_phase_retains_refused_native_pool_closer(
    native_pg, monkeypatch, route, pool_kind,
):
    from kestrel_sovereign import server

    _, _, schema = native_pg
    dsn = with_search_path(os.environ["TEST_POSTGRES_URL"], schema)
    app = FastAPI()
    owners, pools, refused, body_errors = [], [], [], []
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    allocation = RuntimeError("actual native pool closer admission refused")
    actual_initialize = AsyncStorage.initialize

    async def prepare(owner):
        native = owner.db.backend if isinstance(owner, AsyncStorage) else owner
        await native._ensure_advisory_pool()
        owners.append(owner)
        pools.extend([native._pool, native._advisory_pool])

    async def initialize_then_fail(owner):
        await actual_initialize(owner)
        await prepare(owner)
        original = ValueError("ordinary original preflight failure")
        body_errors.append(original)
        raise original

    async def selected_discovery_only(_config):
        return {"did:test:unloaded-input-only": ("unloaded-input-only", None)}

    manager = SimpleNamespace(local_agent_configs_by_did=selected_discovery_only,
                              cold_scheduler_identity_failures=[])

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        frame = getattr(operation, "cr_frame", None)
        owner = frame.f_locals.get("self") if frame is not None else None
        target = pools[0 if pool_kind == "primary" else 1] if pools else None
        code = getattr(operation, "cr_code", None)
        if not refused and owner is target and target is not None and code.co_qualname == "Pool.close":
            refused.append((coroutine, operation))
            raise allocation
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    caught = None
    try:
        with monkeypatch.context() as patch:
            patch.setenv("KESTREL_DB_BACKEND", "postgres")
            patch.setenv("KESTREL_DATABASE_URL", dsn)
            if route == "backend":
                owner = PostgresBackend(dsn=dsn, min_pool_size=0, max_pool_size=2, advisory_max_pool_size=1)
                await owner.connect()
                await prepare(owner)
            else:
                patch.setattr(AsyncStorage, "initialize", initialize_then_fail)
            loop.set_task_factory(factory)
            try:
                if route == "backend":
                    await owners[0].close()
                elif route == "preflight":
                    await server._prepare_shared_postgres_scheduler_protocol(app, manager, None)
                else:
                    await server._start_host_scheduler(app, manager, None)
            except BaseException as error:
                caught = error
            finally:
                loop.set_task_factory(previous)
        assert refused and caught is not None
        assert is_execution_control_error(caught), "accepted phase normalized native closer refusal as ordinary startup failure"
        target = pools[0 if pool_kind == "primary" else 1]
        independent = pools[1 if pool_kind == "primary" else 0]
        assert any(resource is target and task is None for resource, task in execution_cleanup_debts(caught)), "native pool closer lost its exact original resource and absent owner"
        native = owners[0].db.backend if isinstance(owners[0], AsyncStorage) else owners[0]
        assert getattr(native, "_pool" if pool_kind == "primary" else "_advisory_pool") is target, "native pool refusal cleared the retained original handle"
        assert not target._closed and independent._closed, "native pool refusal skipped independent physical retirement"
        assert allocation in tuple(_explicit_control_chain(caught)), "startup lost original native admission failure"
        assert all(error in tuple(_explicit_control_chain(caught)) for error in body_errors)
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
                   for pair in refused for coro in pair), "unstarted native pool closer leaked its coroutine"
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        for owner in owners:
            # Production correctly retains the refused original close record;
            # a healthy factory later is not permission to retry that owner.
            # Explicit fixture recovery closes these SAME actual pools only
            # after the retained handles/debt and independent ACK assertions.
            native = owner.db.backend if isinstance(owner, AsyncStorage) else owner
            for pool in (native._advisory_pool, native._pool):
                if pool is not None and not pool._closed:
                    await pool.close()


@pytest.mark.asyncio
async def test_outer_multi_agent_startup_preserves_original_body_and_rollback_refusal(
    native_pg, tmp_path, monkeypatch,
):
    from kestrel_sovereign import server

    backend, observer, schema = native_pg
    original = None
    backend._execution_custody = ExecutionCustody(GenerationFence())
    with monkeypatch.context() as ack:
        receipts = lose_native_ack(ack, boundary="commit", original=OSError("actual outer startup COMMIT ACK lost"))
        try:
            await backend.execute("INSERT INTO effects VALUES (183, 'original startup receipt')")
        except BaseException as error:
            original = error
    backend._execution_custody = None
    assert receipts == [True] and execution_commit_outcome(original) == "unknown"
    graph = tuple(_explicit_control_chain(original))
    config = tmp_path / "multi_agent.toml"
    config.write_text('[host]\nbind = "127.0.0.1"\nport = 8888\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "KESTREL_HOME": str(tmp_path), "PROJECT_DIR": str(tmp_path),
        "KESTREL_DB_PATH": str(tmp_path / "private-state"),
        "KESTREL_HOST_DB_PATH": str(tmp_path / "host.db"),
        "KESTREL_MULTI_AGENT_CONFIG": str(config), "KESTREL_MULTI_AGENT": "1",
        "KESTREL_API_KEY": "inert-native-host-auth-not-provider-credential",
        "KESTREL_DB_BACKEND": "sqlite", "KESTREL_HOLD_BACKEND": "sqlite",
        "KESTREL_PHOENIX_ENABLED": "0", "KESTREL_TRACING_ENABLED": "0",
    }.items():
        monkeypatch.setenv(key, value)
    managers, refused = [], []
    actual_load = AgentManager.load_from_config

    async def load_empty_then_fail(manager, *args, **kwargs):
        result = await actual_load(manager, *args, **kwargs)
        assert result == 0 and not manager.list_agents()
        managers.append(manager)
        raise original

    monkeypatch.setattr(AgentManager, "load_from_config", load_empty_then_fail)
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    allocation = RuntimeError("actual original manager rollback admission refused")

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if not refused and code is not None and code.co_qualname == "AgentManager.shutdown_all":
            refused.append((coroutine, operation))
            raise allocation
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    app, caught = FastAPI(), None
    try:
        loop.set_task_factory(factory)
        try:
            async with server._lifespan_startup(app):
                pytest.fail("original startup control reached serving publication")
        except BaseException as error:
            caught = error
        finally:
            loop.set_task_factory(previous)
        assert refused and len(managers) == 1 and caught is not None
        chain = tuple(_explicit_control_chain(caught))
        assert original in chain and execution_commit_outcome(caught) == "unknown"
        assert allocation in chain, "outer startup rollback discarded original Task admission failure"
        assert tuple(_explicit_control_chain(original)) == graph, "outer startup rollback mutated its original body receipt"
        assert any(resource is managers[0] and task is None for resource, task in execution_cleanup_debts(caught))
        assert app.state.agent_manager is None
        assert app.state.startup_cleanup_agent_manager is managers[0]
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=183') == 1
    finally:
        loop.set_task_factory(previous)
        await server._shutdown_server_resources(app)
        for manager in managers:
            await manager.shutdown_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("pool_kind", ["primary", "advisory"])
async def test_native_pool_close_joins_physical_retirement_after_repeated_withdrawal(
    native_pg, monkeypatch, pool_kind,
):
    import asyncpg
    from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task

    _, _, schema = native_pg
    owner = PostgresBackend(dsn=with_search_path(os.environ["TEST_POSTGRES_URL"], schema),
                            min_pool_size=0, max_pool_size=2, advisory_max_pool_size=1)
    await owner.connect()
    await owner._ensure_advisory_pool()
    primary, advisory = owner._pool, owner._advisory_pool
    target = primary if pool_kind == "primary" else advisory
    connection = await target.acquire()
    native_connection = connection._con
    entered = asyncio.Event()
    actual_close = asyncpg.Pool.close

    async def observe_actual_close(pool):
        if pool is target:
            entered.set()
        return await actual_close(pool)

    task = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(asyncpg.Pool, "close", observe_actual_close)
            task = create_owned_task(owner.close(), retain_originals=True)
            await asyncio.wait_for(entered.wait(), 5)
            for attempt in range(2):
                task.cancel(f"actual pool closer caller withdrawal {attempt}")
                await asyncio.sleep(0)
            await asyncio.wait({task}, timeout=0.05)
            assert not task.done() and not native_connection.is_closed(), "caller withdrawal abandoned accepted native pool close"
            releasing = connection
            connection = None
            await target.release(releasing)
            outcome = await await_owned_task(task)
        assert isinstance(outcome.error, asyncio.CancelledError), "physical pool close discarded original caller withdrawal"
        assert primary._closed and advisory._closed, "caller withdrawal skipped independent physical pool close"
        assert owner._pool is None and owner._advisory_pool is None
        assert not execution_cleanup_debts(outcome.error)
    finally:
        if connection is not None and not native_connection.is_closed():
            await target.release(connection)
        if task is not None:
            await await_owned_task(task)
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["pool", "preflight", "scheduler"])
@pytest.mark.parametrize("native_body,native_close,withdrawals", [
    (body, close, count) for body, close in [(True, False), (False, True), (True, True)]
    for count in (1, 2)
] + [(True, False, 0)])
async def test_outer_startup_phase_retains_actual_observer_withdrawal(
    native_pg, tmp_path, monkeypatch, route, native_body, native_close, withdrawals,
):
    from kestrel_sovereign import server
    from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task

    backend, observer, schema = native_pg

    async def unknown_receipt(identifier):
        original = None
        backend._execution_custody = ExecutionCustody(GenerationFence())
        try:
            with monkeypatch.context() as ack:
                receipts = lose_native_ack(ack, boundary="commit", original=OSError("actual phase COMMIT ACK lost"))
                try:
                    await backend.execute("INSERT INTO effects VALUES (?, 'original phase receipt')", (identifier,))
                except BaseException as error:
                    original = error
            assert receipts == [True] and execution_commit_outcome(original) == "unknown"
            return original
        finally:
            backend._execution_custody = None

    body = await unknown_receipt(185) if native_body else ValueError("ordinary original startup failure")
    closer = await unknown_receipt(186) if native_close else None
    originals = [error for error in (body, closer) if error is not None]
    graphs = [tuple(_explicit_control_chain(error)) for error in originals]
    entered, release = asyncio.Event(), asyncio.Event()
    owners = []
    actual_connect, actual_initialize, actual_close = PostgresBackend.connect, AsyncStorage.initialize, PostgresBackend.close

    async def connect_then_fail(owner):
        result = await actual_connect(owner)
        if route == "pool":
            owners.append(owner)
            raise body
        return result

    async def initialize_then_fail(owner):
        await actual_initialize(owner)
        owners.append(owner.db.backend)
        raise body

    async def actual_close_after_observation(owner):
        if owner in owners:
            entered.set()
            await release.wait()
        result = await actual_close(owner)
        if owner in owners and withdrawals == 0:
            asyncio.current_task().cancel("actual original native phase withdrawal")
            await asyncio.sleep(0)
        if owner in owners and closer is not None:
            raise closer
        return result

    manager = AgentManager(base_data_dir=tmp_path / "empty-manager")

    async def selected_discovery_only(_config):
        return {"did:test:unloaded-input-only": ("unloaded-input-only", None)}

    app, task = FastAPI(), None
    try:
        with monkeypatch.context() as patch:
            patch.setenv("KESTREL_DB_BACKEND", "postgres")
            patch.setenv("KESTREL_DATABASE_URL", with_search_path(os.environ["TEST_POSTGRES_URL"], schema))
            patch.setenv("KESTREL_SHARED_AGENT_POSTGRES_MAX_POOL_SIZE", "2")
            patch.setenv("KESTREL_SHARED_AGENT_POSTGRES_ADVISORY_MAX_POOL_SIZE", "1")
            patch.setattr(manager, "local_agent_configs_by_did", selected_discovery_only)
            patch.setattr(PostgresBackend, "close", actual_close_after_observation)
            if route == "pool":
                patch.setattr(PostgresBackend, "connect", connect_then_fail)
                operation = server._start_shared_agent_postgres_backend(app)
            else:
                patch.setattr(AsyncStorage, "initialize", initialize_then_fail)
                operation = (server._prepare_shared_postgres_scheduler_protocol(app, manager, None)
                             if route == "preflight" else server._start_host_scheduler(app, manager, None))
            task = create_owned_task(operation, retain_originals=True)
            await asyncio.wait_for(entered.wait(), 5)
            for attempt in range(withdrawals):
                task.cancel(f"actual outer phase observer withdrawal {attempt}")
                await asyncio.sleep(0)
            if withdrawals:
                await asyncio.wait({task}, timeout=0.05)
            assert not task.done() and all(owner.is_connected for owner in owners), "startup phase abandoned accepted physical cleanup on observer withdrawal"
            release.set()
            outcome = await await_owned_task(task)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert is_execution_control_error(outcome.error) and execution_commit_outcome(outcome.error) == "unknown"
        assert all(original in chain for original in originals)
        prefix = "actual original native phase withdrawal" if withdrawals == 0 else "actual outer phase observer withdrawal"
        assert any(isinstance(error, asyncio.CancelledError)
                   and error.args and str(error.args[0]).startswith(prefix)
                   for error in chain), "startup phase discarded its actual original observer cancellation"
        assert all(tuple(_explicit_control_chain(original)) == graph for original, graph in zip(originals, graphs)), "startup phase mutated an original source receipt"
        assert all(not owner.is_connected for owner in owners), "startup phase did not join its actual physical cleanup"
        expected = int(native_body) + int(native_close)
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id IN (185,186)') == expected
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        for owner in owners:
            await actual_close(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["manager", "teardown"])
@pytest.mark.parametrize("withdrawals", [0, 1, 2])
async def test_original_phase_observer_reaches_outer_lifespan_composer(
    native_pg, tmp_path, monkeypatch, route, withdrawals,
):
    from kestrel_sovereign import server
    from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
    from kestrel_sovereign.host_features import build_host_context

    backend, observer, schema = native_pg
    body = None
    backend._execution_custody = ExecutionCustody(GenerationFence())
    with monkeypatch.context() as ack:
        receipts = lose_native_ack(ack, boundary="commit", original=OSError("actual outer lifespan COMMIT ACK lost"))
        try:
            await backend.execute("INSERT INTO effects VALUES (187, 'original outer lifespan receipt')")
        except BaseException as error:
            body = error
    backend._execution_custody = None
    assert receipts == [True] and execution_commit_outcome(body) == "unknown"
    graph = tuple(_explicit_control_chain(body))
    config = tmp_path / "multi_agent.toml"
    config.write_text('[host]\nbind = "127.0.0.1"\nport = 8888\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "KESTREL_HOME": str(tmp_path), "PROJECT_DIR": str(tmp_path),
        "KESTREL_DB_PATH": str(tmp_path / "private-state"),
        "KESTREL_HOST_DB_PATH": str(tmp_path / "host.db"),
        "KESTREL_MULTI_AGENT_CONFIG": str(config), "KESTREL_MULTI_AGENT": "1",
        "KESTREL_API_KEY": "inert-native-host-auth-not-provider-credential",
        "KESTREL_DB_BACKEND": "sqlite", "KESTREL_HOLD_BACKEND": "sqlite",
        "KESTREL_PHOENIX_ENABLED": "0", "KESTREL_TRACING_ENABLED": "0",
    }.items():
        monkeypatch.setenv(key, value)
    entered, release = asyncio.Event(), asyncio.Event()
    managers, workers = [], []
    app, task = FastAPI(), None
    actual_load, actual_shutdown = AgentManager.load_from_config, AgentManager.shutdown_all
    actual_host_shutdown = server._shutdown_host_features

    async def load_empty_then_fail(manager, *args, **kwargs):
        result = await actual_load(manager, *args, **kwargs)
        assert result == 0 and not manager.list_agents()
        managers.append(manager)
        raise body

    async def retain_manager_observer(manager):
        entered.set()
        await release.wait()
        result = await actual_shutdown(manager)
        if withdrawals == 0:
            asyncio.current_task().cancel("actual original outer phase withdrawal")
            await asyncio.sleep(0)
        return result

    async def retain_context_observer(app):
        entered.set()
        await release.wait()
        result = await actual_host_shutdown(app)
        if withdrawals == 0:
            asyncio.current_task().cancel("actual original outer phase withdrawal")
            await asyncio.sleep(0)
        return result

    async def operation():
        if route == "manager":
            async with server._lifespan_startup(app):
                pytest.fail("original startup control reached serving publication")
        else:
            async with server._lifespan_teardown_owner(app):
                raise body

    try:
        with monkeypatch.context() as patch:
            if route == "manager":
                patch.setattr(AgentManager, "load_from_config", load_empty_then_fail)
                patch.setattr(AgentManager, "shutdown_all", retain_manager_observer)
            else:
                context = await build_host_context(db_path=str(tmp_path / "actual-owned-context.db"))
                app.state.host_context, app.state.host_features = context, []
                connection = context.db.backend._connection
                workers.append(getattr(connection, "_thread", connection))
                patch.setattr(server, "_shutdown_host_features", retain_context_observer)
            task = create_owned_task(operation(), retain_originals=True)
            await asyncio.wait_for(entered.wait(), 5)
            for attempt in range(withdrawals):
                task.cancel(f"actual outer lifespan observer withdrawal {attempt}")
                await asyncio.sleep(0)
            assert not task.done() and all(worker.is_alive() for worker in workers)
            release.set()
            outcome = await await_owned_task(task)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert body in chain and execution_commit_outcome(outcome.error) == "unknown"
        prefix = "actual original outer phase withdrawal" if withdrawals == 0 else "actual outer lifespan observer withdrawal"
        assert any(isinstance(error, asyncio.CancelledError) and error.args
                   and str(error.args[0]).startswith(prefix)
                   for error in chain), "outer lifespan discarded its actual original phase observer cancellation"
        assert tuple(_explicit_control_chain(body)) == graph, "outer lifespan mutated its original native body receipt"
        assert all(not worker.is_alive() for worker in workers)
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=187') == 1
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        await server._shutdown_server_resources(app)
        for manager in managers:
            await actual_shutdown(manager)
