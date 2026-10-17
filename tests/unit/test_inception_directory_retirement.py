"""A refused native SQLite closer cannot release replacement-birth custody."""

import asyncio
import inspect
from contextlib import ExitStack

import pytest

from kestrel_sovereign import inception_service
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.private_storage import exclusive_private_file_lock
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["publication", "initialization"])
async def test_original_database_debt_keeps_actual_birth_directory_locked(tmp_path, monkeypatch, phase):
    original_sqlite = AsyncDatabase.sqlite
    opened, refused = [], []
    allocation = RuntimeError("actual original birth database closer admission refused")
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    retained_owner = None
    retirement = None
    entered, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")

    async def initialize_then_fail(db):
        opened.append(db)
        await db.execute("CREATE TABLE original_initialization (id INTEGER)")
        await db.execute("SELECT * FROM original_missing_initialization_table")

    async def sqlite(path, **kwargs):
        if phase == "initialization":
            return await original_sqlite(path, schema_initializer=initialize_then_fail)
        db = await original_sqlite(path, **kwargs)
        opened.append(db)
        await db.execute_script("CREATE TRIGGER original_publication_refusal BEFORE INSERT ON files BEGIN SELECT RAISE(ABORT, 'actual native original publication refusal'); END;")
        return db

    def factory(loop, coroutine, **kwargs):
        actual = _actual_operation(coroutine)
        code, frame = getattr(actual, "cr_code", None), getattr(actual, "cr_frame", None)
        if code is not None and code.co_qualname == "AsyncDatabase.close" and frame.f_locals.get("self") in opened:
            refused.append((coroutine, actual))
            raise allocation
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    monkeypatch.setattr(AsyncDatabase, "sqlite", sqlite)
    try:
        loop.set_task_factory(factory)
        with pytest.raises(BaseException) as caught:
            await inception_service.create_kestrel_identity_async(
                output_dir=str(tmp_path), is_test_instance=True,
                identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test",
                did_web_slug="original-database-retirement",
            )
        loop.set_task_factory(previous)
        assert len(opened) == 1 and refused
        db = opened[0]
        connection = db.backend._connection
        worker = getattr(connection, "_thread", connection)
        assert connection is not None and worker.is_alive()
        assert allocation in tuple(_explicit_control_chain(caught.value))
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        # Assert the actual physical safety property before requiring a new
        # retirement attribute: an old implementation must fail on custody.
        with pytest.raises(Exception, match="inception|lock|held|busy"):
            with exclusive_private_file_lock(tmp_path / ".inception.lock", label="inception", blocking=False):
                pass
        retained_owner = caught.value.inception_retirement_owner
        assert retained_owner.database is db and retained_owner.directory_custody is not None
        actual_close = db.close

        async def held_close():
            entered.set()
            await release.wait()
            await actual_close()

        monkeypatch.setattr(db, "close", held_close)
        retirement = create_owned_task(retained_owner.retire(), retain_originals=True)
        await asyncio.wait_for(entered.wait(), 5)
        assert worker.is_alive() and not retirement.done()
        with pytest.raises(Exception, match="inception|lock|held|busy"):
            with exclusive_private_file_lock(tmp_path / ".inception.lock", label="inception", blocking=False):
                pass
        release.set()
        outcome = await await_owned_task(retirement)
        assert outcome.error is None and outcome.cancellation is None
        assert not worker.is_alive() and db.backend._connection is None
        assert retained_owner.directory_custody is None
        with exclusive_private_file_lock(tmp_path / ".inception.lock", label="inception", blocking=False):
            pass
    finally:
        loop.set_task_factory(previous)
        release.set()
        if retirement is not None:
            await await_owned_task(retirement)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        for db in opened:
            await db.close()
        if retained_owner is not None and retained_owner.directory_custody is not None:
            retained_owner.directory_custody.close()


@pytest.mark.asyncio
async def test_database_owner_retires_independent_auxiliary_after_first_refusal(tmp_path):
    databases, refused = [], []
    owner = None
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    allocation = RuntimeError("actual first retained database closer refused")

    def factory(loop, coroutine, **kwargs):
        actual = _actual_operation(coroutine)
        code, frame = getattr(actual, "cr_code", None), getattr(actual, "cr_frame", None)
        if code is not None and code.co_qualname == "AsyncDatabase.close" and frame.f_locals.get("self") is databases[0]:
            refused.append((coroutine, actual))
            raise allocation
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    try:
        for name in ("first.db", "auxiliary.db"):
            databases.append(await AsyncDatabase.sqlite(str(tmp_path / name)))
        first, second = databases
        second_connection = second.backend._connection
        custody = ExitStack()
        custody.enter_context(exclusive_private_file_lock(tmp_path / ".inception.lock", label="inception", blocking=False))
        owner = inception_service.InceptionProviderRetirementOwner(None, first, directory_custody=custody)
        owner.retain_database(second)
        loop.set_task_factory(factory)
        with pytest.raises(BaseException) as caught:
            await owner.retire()
        loop.set_task_factory(previous)
        assert refused and allocation in tuple(_explicit_control_chain(caught.value))
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        assert first.backend._connection is not None
        assert second.backend._connection is None and not getattr(second_connection, "_thread", second_connection).is_alive(), "one refused closer skipped independent owned native retirement"
        assert owner.database is first and owner.auxiliary_databases == ()
        assert caught.value.inception_retirement_owner is owner
        with pytest.raises(Exception, match="inception|lock|held|busy"):
            with exclusive_private_file_lock(tmp_path / ".inception.lock", label="inception", blocking=False):
                pass
        await owner.retire()
        assert first.backend._connection is None and owner.directory_custody is None
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        for database in databases:
            await database.close()
        if owner is not None and owner.directory_custody is not None:
            owner.directory_custody.close()
