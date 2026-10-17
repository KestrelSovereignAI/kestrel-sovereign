"""Original native inception publication survives local SQLite retirement.

This is the actual publication-custody component with a borrowed native PG
target and an owned local SQLite authority connection. No agent birth, audit,
provider call, keys or READY state is manufactured.
"""

import asyncio
import inspect
from contextlib import asynccontextmanager

import pytest

from kestrel_sovereign import inception_service
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, _explicit_control_chain, execution_cleanup_debts, execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refused", "close-ack", "withdrawal", "rollback-withdrawal", "rollback-handoff-refused"])
async def test_original_publication_control_survives_local_authority_retirement(
    native_pg, tmp_path, monkeypatch, mode,
):
    from kestrel_sovereign import _async_ownership as ownership_module

    backend, observer, schema = native_pg
    output = tmp_path / "component"
    output.mkdir()
    seed = await AsyncDatabase.sqlite(str(output / "kestrel_prime.db"))
    await seed.close()
    target = AsyncDatabase(backend)
    backend._execution_custody = ExecutionCustody(GenerationFence())
    actual_sqlite, actual_transaction = AsyncDatabase.sqlite, backend.transaction
    actual_join = ownership_module.await_owned_task
    opened, originals, graphs, close_tasks, withdrawals, refused = [], [], [], [], [], []
    rollback_withdrawals = []
    entered, release = asyncio.Event(), asyncio.Event()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    allocation = RuntimeError("actual local authority closer admission refused")
    lost_close_ack = OSError("actual local SQLite retirement ACK lost")
    owner = None

    @asynccontextmanager
    async def observed_transaction(*args, **kwargs):
        try:
            async with actual_transaction(*args, **kwargs):
                yield
        except BaseException as error:
            if execution_commit_outcome(error) == "unknown":
                originals.append(error)
                graphs.append(tuple(_explicit_control_chain(error)))
            raise

    async def observed_sqlite(*args, **kwargs):
        db = await actual_sqlite(*args, **kwargs)
        connection, close = db.backend._connection, db.close
        opened.append((db, connection, close))
        actual_rollback = connection.rollback

        async def observed_rollback():
            if mode.startswith("rollback-") and not entered.is_set():
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError as cancellation:
                    rollback_withdrawals.append(cancellation)
                    raise
            return await actual_rollback()

        monkeypatch.setattr(connection, "rollback", observed_rollback)

        async def observed_close():
            close_tasks.append(asyncio.current_task())
            entered.set()
            await release.wait()
            await close()
            if mode == "close-ack":
                raise lost_close_ack

        monkeypatch.setattr(db, "close", observed_close)
        return db

    async def observed_join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if task in close_tasks and outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome

    def factory(loop, coroutine, **kwargs):
        actual = _actual_operation(coroutine)
        code = getattr(actual, "cr_code", None)
        if mode == "rollback-handoff-refused" and code is not None and code.co_qualname == "SQLiteBackend._drain_cancelled_write":
            refused.append((coroutine, actual))
            raise allocation
        if mode == "refused" and code is not None and code.co_qualname.endswith("observed_close"):
            refused.append((coroutine, actual))
            raise allocation
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    async def run():
        async with inception_service._inception_publication_custody(
            target, output, "did:test:unborn-publication-component",
        ):
            await target.execute("INSERT INTO effects VALUES (190, 'actual publication before local retirement')")

    try:
        with monkeypatch.context() as ack:
            committed = lose_native_ack(ack, boundary="commit", original=OSError("actual publication COMMIT ACK lost"))
            monkeypatch.setattr(AsyncDatabase, "sqlite", observed_sqlite)
            monkeypatch.setattr(backend, "transaction", observed_transaction)
            monkeypatch.setattr(ownership_module, "await_owned_task", observed_join)
            loop.set_task_factory(factory)
            if mode not in {"withdrawal", "rollback-withdrawal", "rollback-handoff-refused"}:
                release.set()
            owner = create_owned_task(run(), retain_originals=True)
            if mode in {"withdrawal", "rollback-withdrawal", "rollback-handoff-refused"}:
                await asyncio.wait_for(entered.wait(), 5)
                owner.cancel("actual local authority retirement observer withdrawal")
                release.set()
            outcome = await await_owned_task(owner)
            loop.set_task_factory(previous)
        assert committed == [True]
        assert len(originals) == 1 and execution_commit_outcome(originals[0]) == "unknown"
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=190') == 1
        assert outcome.error is not None and is_execution_control_error(outcome.error), "local authority retirement masked original publication control"
        chain = tuple(_explicit_control_chain(outcome.error))
        assert originals[0] in chain and execution_commit_outcome(outcome.error) == "unknown"
        assert tuple(_explicit_control_chain(originals[0])) == graphs[0], "local retirement mutated original shared publication receipt"
        assert len(opened) == 1 and backend.is_connected, "local closer retired its borrowed PG target"
        db, connection, _close = opened[0]
        if mode == "refused":
            assert refused and allocation in chain
            assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
            assert connection is db.backend._connection and getattr(connection, "_thread", connection).is_alive()
        else:
            assert db.backend._connection is None and not getattr(connection, "_thread", connection).is_alive()
            if mode == "close-ack":
                assert lost_close_ack in chain
            elif mode in {"rollback-withdrawal", "rollback-handoff-refused"}:
                assert len(rollback_withdrawals) == 1 and rollback_withdrawals[0] in chain, "auxiliary rollback discarded its actual observer withdrawal"
                if mode == "rollback-handoff-refused":
                    assert allocation in chain and refused
                    assert any(resource is connection and task is None for resource, task in execution_cleanup_debts(outcome.error))
                    assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
            else:
                assert len(withdrawals) == 1 and withdrawals[0] in chain
    finally:
        loop.set_task_factory(previous)
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        for _db, connection, close in opened:
            await close()
            assert not getattr(connection, "_thread", connection).is_alive()
        backend._execution_custody = None


@pytest.mark.asyncio
async def test_failed_publication_readback_retains_actual_native_release_receipt(native_pg, tmp_path, monkeypatch):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    await db._init_schema()
    backend._execution_custody = ExecutionCustody(GenerationFence())
    await backend.execute("""CREATE FUNCTION refuse_inception_file() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'actual native inception publication refusal'; END $$""")
    await backend.execute("CREATE TRIGGER refuse_inception_file BEFORE INSERT ON files FOR EACH ROW EXECUTE FUNCTION refuse_inception_file()")
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    actual_custody = inception_service._inception_publication_custody
    actual_fetch = backend.fetch_one
    body_errors, readback_errors, graphs, committed = [], [], [], []

    @asynccontextmanager
    async def observed_custody(*args, **kwargs):
        try:
            async with actual_custody(*args, **kwargs) as value:
                yield value
        except BaseException as error:
            body_errors.append(error)
            raise

    async def observed_fetch(query, params=None):
        if body_errors and query == "SELECT node_id FROM graph_nodes WHERE node_id = ?":
            with monkeypatch.context() as ack:
                receipts = lose_native_ack(ack, boundary="release", original=OSError("actual readback session release ACK lost"))
                try:
                    return await actual_fetch(query, params)
                except BaseException as error:
                    readback_errors.append(error)
                    graphs.append(tuple(_explicit_control_chain(error)))
                    committed.extend(receipts)
                    raise
        return await actual_fetch(query, params)

    monkeypatch.setattr(inception_service, "_inception_publication_custody", observed_custody)
    monkeypatch.setattr(backend, "fetch_one", observed_fetch)
    try:
        with pytest.raises(BaseException) as caught:
            await inception_service.create_kestrel_identity_async(
                output_dir=str(tmp_path), database=db, is_test_instance=True,
                identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test",
                did_web_slug="native-readback-refusal",
            )
        assert len(body_errors) == len(readback_errors) == 1 and committed == [True]
        assert is_execution_control_error(readback_errors[0]), "readback fixture did not cross native execution custody"
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".graph_nodes WHERE node_type=\'agent\'') == 0
        chain = tuple(_explicit_control_chain(caught.value))
        assert readback_errors[0] in chain, "publication failure discarded actual native readback control"
        assert body_errors[0] in chain
        assert tuple(_explicit_control_chain(readback_errors[0])) == graphs[0]
        assert backend.is_connected, "failed inception retired its borrowed target"
    finally:
        backend._execution_custody = None
