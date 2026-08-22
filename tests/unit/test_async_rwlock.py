"""Contracts for scheduler lifecycle reader/writer admission."""

import asyncio

import pytest

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock


@pytest.mark.asyncio
async def test_writer_closes_new_reader_admission_and_drains_existing_readers():
    """A queued lifecycle mutation cannot starve behind scheduled effects."""

    lock = AsyncReaderWriterLock()
    readers_entered = [asyncio.Event(), asyncio.Event()]
    release_readers = asyncio.Event()
    writer_waiting = asyncio.Event()
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()
    late_reader_entered = asyncio.Event()

    async def reader(entered: asyncio.Event) -> None:
        async with lock.read():
            entered.set()
            await release_readers.wait()

    async def writer() -> None:
        writer_waiting.set()
        async with lock:
            writer_entered.set()
            await release_writer.wait()

    async def late_reader() -> None:
        async with lock.read():
            late_reader_entered.set()

    active_readers = [
        asyncio.create_task(reader(entered)) for entered in readers_entered
    ]
    queued_writer = None
    late = None
    try:
        await asyncio.wait_for(
            asyncio.gather(*(entered.wait() for entered in readers_entered)), timeout=1
        )
        queued_writer = asyncio.create_task(writer())
        await asyncio.wait_for(writer_waiting.wait(), timeout=1)
        # Let the writer reach the lock's queue before attempting a new effect.
        await asyncio.sleep(0)
        late = asyncio.create_task(late_reader())
        await asyncio.sleep(0.02)
        assert not late_reader_entered.is_set()
        assert not writer_entered.is_set()

        release_readers.set()
        await asyncio.wait_for(writer_entered.wait(), timeout=1)
        assert not late_reader_entered.is_set()

        release_writer.set()
        await asyncio.wait_for(late_reader_entered.wait(), timeout=1)
    finally:
        release_readers.set()
        release_writer.set()
        await asyncio.gather(
            *active_readers,
            *(() if queued_writer is None else (queued_writer,)),
            return_exceptions=True,
        )
        if late is not None:
            await asyncio.gather(late, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_queued_writer_restores_reader_admission():
    """A cancelled rollout/removal waiter cannot leave the gate closed."""

    lock = AsyncReaderWriterLock()
    reader_entered = asyncio.Event()
    release_reader = asyncio.Event()
    writer_waiting = asyncio.Event()

    async def hold_reader() -> None:
        async with lock.read():
            reader_entered.set()
            await release_reader.wait()

    async def wait_for_writer() -> None:
        writer_waiting.set()
        async with lock:
            pytest.fail("cancelled writer must not acquire")

    reader = asyncio.create_task(hold_reader())
    writer = None
    try:
        await asyncio.wait_for(reader_entered.wait(), timeout=1)
        writer = asyncio.create_task(wait_for_writer())
        await asyncio.wait_for(writer_waiting.wait(), timeout=1)
        await asyncio.sleep(0)
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer

        release_reader.set()
        await asyncio.wait_for(reader, timeout=1)
        async with lock.read():
            assert lock.locked()
    finally:
        release_reader.set()
        if writer is not None and not writer.done():
            writer.cancel()
        await asyncio.gather(
            reader,
            *(() if writer is None else (writer,)),
            return_exceptions=True,
        )
