"""Native rollback-drain cancellation is not cancellation of its close observer."""

import asyncio
import sqlite3

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_close", ["none", "after", "before"])
@pytest.mark.parametrize("historical_cancel", [False, True])
async def test_close_distinguishes_native_drain_from_original_caller(
    tmp_path,
    monkeypatch,
    cancel_close,
    historical_cancel,
):
    path = tmp_path / "native-drain-close.db"
    backend = SQLiteBackend(path)
    await backend.connect()
    await backend.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    connection = backend._connection
    await connection.execute("BEGIN")
    await connection.execute("INSERT INTO effects VALUES (1)")
    entered, release, closing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    rollback = connection.rollback
    interrupt = connection.interrupt

    async def native_rollback_then_cancel():
        await rollback()
        entered.set()
        await release.wait()
        raise asyncio.CancelledError("native rollback drain, not the close caller")

    async def observed_interrupt():
        await interrupt()
        if backend._closing:
            closing.set()

    monkeypatch.setattr(connection, "rollback", native_rollback_then_cancel)
    monkeypatch.setattr(connection, "interrupt", observed_interrupt)
    backend._handoff_cancelled_write(connection)
    drain = backend._cancelled_write_drain
    closer = None

    async def close_after_observed_cancellation():
        if historical_cancel:
            asyncio.current_task().cancel("previously observed cancellation")
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass  # Historical cancelling() is not a new close cancellation.
        if cancel_close == "before":
            release.set()
            asyncio.current_task().cancel("original close caller cancellation")
        await backend.close()

    try:
        await asyncio.wait_for(entered.wait(), 2)
        closer = create_owned_task(close_after_observed_cancellation())
        await asyncio.wait_for(closing.wait(), 2)
        # close has reached its actual retained drain, not just a done fake.
        await asyncio.sleep(0)
        if cancel_close == "after":
            closer.cancel("original close caller cancellation")
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(closer), 3)
        if cancel_close != "none":
            assert isinstance(outcome.error, asyncio.CancelledError)
            assert outcome.error.args == ("original close caller cancellation",)
        else:
            assert outcome.error is None, (
                "native drain injected cancellation into an uncancelled close"
            )
        assert drain.done()
        assert backend._connection is None
        assert not backend.connection_retirement_pending
        with sqlite3.connect(path) as observer:
            assert observer.execute("SELECT COUNT(*) FROM effects").fetchone()[0] == 0
    finally:
        release.set()
        await await_owned_task(drain)
        if closer is not None:
            await await_owned_task(closer)
        if backend._connection is not None:
            monkeypatch.undo()
            await backend.close()
