"""Actual SQLite COMMIT/withdrawal/refused-drain custody and write fencing."""
import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, execution_cleanup_debts,
    execution_commit_outcome,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["execute", "many", "script", "transaction"])
async def test_refused_actual_commit_drain_keeps_originals_and_fences_next_write(tmp_path, monkeypatch, route):
    database = await AsyncDatabase.sqlite(str(tmp_path / "owned-handoff.db"))
    await database.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY)")
    connection = database.backend._connection
    actual_commit = connection.commit
    withdrawals, refused = [], []
    refusal = RuntimeError("actual SQLite original drain Task admission refused")
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()

    async def commit():
        await actual_commit()
        if not withdrawals:
            asyncio.current_task().cancel("actual native SQLite COMMIT observer withdrawal")
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as cancellation:
                withdrawals.append(cancellation)
                raise

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname == "SQLiteBackend._drain_cancelled_write":
            refused.append((coroutine, operation))
            raise refusal
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    async def effect():
        if route == "transaction":
            async with database.transaction():
                await database.execute("INSERT INTO effects VALUES (1)")
        elif route == "many":
            await database.backend.execute_many("INSERT INTO effects VALUES (?)", [(1,)])
        elif route == "script":
            await database.backend.execute_script("INSERT INTO effects VALUES (1);")
        else:
            await database.execute("INSERT INTO effects VALUES (1)")

    try:
        monkeypatch.setattr(connection, "commit", commit)
        loop.set_task_factory(factory)
        outcome = await await_owned_task(create_owned_task(effect(), retain_originals=True))
        chain = tuple(_explicit_control_chain(outcome.error))
        assert len(withdrawals) == 1 and withdrawals[0] in chain and refusal in chain
        assert execution_commit_outcome(outcome.error) == "unknown"
        assert refused and all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        assert any(resource is connection and task is None for resource, task in execution_cleanup_debts(outcome.error))
        assert database.backend._connection is connection
        with pytest.raises(BaseException) as rejected:
            await database.execute("INSERT INTO effects VALUES (2)")
        assert refusal in tuple(_explicit_control_chain(rejected.value)), "unstarted drain debt did not fence a later write"
        async with connection.execute("SELECT id FROM effects") as cursor:
            assert [tuple(row) for row in await cursor.fetchall()] == [(1,)], "refused original drain admitted a second native write"
    finally:
        loop.set_task_factory(previous)
        monkeypatch.setattr(connection, "commit", actual_commit)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await await_owned_task(create_owned_task(database.close(), retain_originals=True))
        assert database.backend._connection is None and not getattr(connection, "_thread", connection).is_alive()
