"""Auxiliary SQLite debt retains birth custody without adopting borrowed PG."""

import asyncio
import inspect

import pytest

from kestrel_sovereign import inception_service
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.private_storage import exclusive_private_file_lock
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
async def test_auxiliary_authority_retirement_keeps_directory_and_borrowed_target(native_pg, tmp_path, monkeypatch):
    backend, observer, schema = native_pg
    target = AsyncDatabase(backend)
    await target._init_schema()
    seed = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
    await seed.close()
    actual_sqlite = AsyncDatabase.sqlite
    opened, refused = [], []
    allocation = RuntimeError("actual auxiliary authority closer admission refused")
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    retained_owner = None

    async def sqlite(*args, **kwargs):
        db = await actual_sqlite(*args, **kwargs)
        opened.append(db)
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
                output_dir=str(tmp_path), database=target, is_test_instance=True,
                identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test",
                did_web_slug="retained-auxiliary-authority",
            )
        loop.set_task_factory(previous)
        assert len(opened) == 1 and refused
        db = opened[0]
        connection = db.backend._connection
        worker = getattr(connection, "_thread", connection)
        assert connection is not None and worker.is_alive()
        assert allocation in tuple(_explicit_control_chain(caught.value))
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        with pytest.raises(Exception, match="inception|lock|held|busy"):
            with exclusive_private_file_lock(tmp_path / ".inception.lock", label="inception", blocking=False):
                pass
        retained_owner = caught.value.inception_retirement_owner
        assert retained_owner.database is db and retained_owner.database is not target
        assert retained_owner.directory_custody is not None and backend.is_connected
        await retained_owner.retire()
        assert not worker.is_alive() and db.backend._connection is None
        assert retained_owner.directory_custody is None and backend.is_connected
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".graph_nodes WHERE node_type=\'agent\'') == 0
        await target.execute("INSERT INTO effects VALUES (193, 'borrowed target still live')")
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=193') == 1
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        for db in opened:
            await db.close()
        if retained_owner is not None and retained_owner.directory_custody is not None:
            retained_owner.directory_custody.close()
