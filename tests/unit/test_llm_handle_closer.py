"""``HandleCloser``: a handle is forgotten only once its close finished (#3559)."""

import asyncio

import pytest

from kestrel_sovereign.llm.handle_closer import HandleCloser


class _Handle:
    """A handle whose close a test controls."""

    def __init__(self):
        self.closes = 0
        self.closed = False
        self.failures: list[BaseException] = []
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()

    async def close(self):
        self.closes += 1
        self.entered.set()
        if self.failures:
            raise self.failures.pop(0)
        if self.gate is not None:
            await self.gate.wait()
        self.closed = True


def _labels(closer: HandleCloser) -> list[str]:
    return [label for label, _handle, _close in closer.unconfirmed()]


@pytest.mark.asyncio
async def test_a_closed_handle_is_forgotten():
    closer = HandleCloser()
    handle = _Handle()

    await closer.close("handle", handle, handle.close, timeout=1.0)

    assert handle.closed
    assert closer.unconfirmed() == []


@pytest.mark.asyncio
async def test_a_synchronous_close_is_accepted():
    closer = HandleCloser()
    calls = []

    await closer.close("sync", object(), lambda: calls.append("closed"), timeout=1.0)

    assert calls == ["closed"]
    assert closer.unconfirmed() == []


@pytest.mark.asyncio
async def test_a_synchronous_close_that_raises_keeps_the_handle():
    closer = HandleCloser()
    handle = object()

    def close():
        raise OSError("sync close failed")

    with pytest.raises(OSError, match="sync close failed"):
        await closer.close("sync", handle, close, timeout=1.0)

    assert closer.unconfirmed() == [("sync", handle, close)]


@pytest.mark.asyncio
async def test_a_failed_close_keeps_the_handle_and_a_retry_starts_it_again():
    closer = HandleCloser()
    handle = _Handle()
    handle.failures.append(ConnectionError("close failed"))

    with pytest.raises(ConnectionError, match="close failed"):
        await closer.close("handle", handle, handle.close, timeout=1.0)

    assert _labels(closer) == ["handle"]
    assert not handle.closed

    await closer.close("handle", handle, handle.close, timeout=1.0)

    assert handle.closes == 2
    assert handle.closed
    assert closer.unconfirmed() == []


@pytest.mark.asyncio
async def test_a_timed_out_close_keeps_running_and_a_retry_waits_for_it():
    """Starting the close again could report a close still running finished."""
    closer = HandleCloser()
    handle = _Handle()
    handle.gate = asyncio.Event()

    with pytest.raises(TimeoutError, match="handle did not finish"):
        await closer.close("handle", handle, handle.close, timeout=0.01)

    assert _labels(closer) == ["handle"]
    assert not handle.closed

    with pytest.raises(TimeoutError):
        await closer.close("handle", handle, handle.close, timeout=0.01)

    handle.gate.set()
    await closer.close("handle", handle, handle.close, timeout=1.0)

    assert handle.closes == 1, "the close a timeout left running was not restarted"
    assert handle.closed
    assert closer.unconfirmed() == []


@pytest.mark.asyncio
async def test_a_close_that_fails_after_its_timeout_is_started_again():
    closer = HandleCloser()
    handle = _Handle()
    gate = asyncio.Event()
    failed = asyncio.Event()

    async def close_then_fail():
        handle.closes += 1
        await gate.wait()
        failed.set()
        raise ConnectionError("late close failure")

    with pytest.raises(TimeoutError):
        await closer.close("handle", handle, close_then_fail, timeout=0.01)
    gate.set()
    await asyncio.wait_for(failed.wait(), timeout=5.0)
    await asyncio.sleep(0)

    await closer.close("handle", handle, handle.close, timeout=1.0)

    assert handle.closes == 2
    assert handle.closed
    assert closer.unconfirmed() == []


@pytest.mark.asyncio
async def test_a_cancelled_caller_leaves_the_close_running_for_a_retry():
    closer = HandleCloser()
    handle = _Handle()
    handle.gate = asyncio.Event()

    waiter = asyncio.create_task(
        closer.close("handle", handle, handle.close, timeout=10.0)
    )
    await asyncio.wait_for(handle.entered.wait(), timeout=5.0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert _labels(closer) == ["handle"]

    handle.gate.set()
    await closer.close("handle", handle, handle.close, timeout=1.0)

    assert handle.closes == 1
    assert handle.closed
    assert closer.unconfirmed() == []


@pytest.mark.asyncio
async def test_a_close_cancelled_by_someone_else_is_a_failed_close():
    """The caller was not cancelled, so the sweep must not stop as if it were."""
    closer = HandleCloser()
    handle = _Handle()
    handle.gate = asyncio.Event()

    with pytest.raises(TimeoutError):
        await closer.close("handle", handle, handle.close, timeout=0.01)
    [entry] = closer._unconfirmed.values()
    entry.attempt.cancel()

    with pytest.raises(RuntimeError, match="handle was cancelled before it finished"):
        await closer.close("handle", handle, handle.close, timeout=1.0)

    assert _labels(closer) == ["handle"]

    handle.gate.set()
    await closer.close("handle", handle, handle.close, timeout=1.0)

    assert handle.closed
    assert closer.unconfirmed() == []


@pytest.mark.asyncio
async def test_concurrent_closes_of_one_handle_share_one_attempt():
    closer = HandleCloser()
    handle = _Handle()
    handle.gate = asyncio.Event()

    first = asyncio.create_task(closer.close("h", handle, handle.close, timeout=5.0))
    second = asyncio.create_task(closer.close("h", handle, handle.close, timeout=5.0))
    await asyncio.wait_for(handle.entered.wait(), timeout=5.0)
    handle.gate.set()
    await asyncio.gather(first, second)

    assert handle.closes == 1
    assert closer.unconfirmed() == []


@pytest.mark.asyncio
async def test_a_timeout_the_close_itself_raised_is_not_reported_still_running():
    closer = HandleCloser()
    handle = _Handle()
    handle.failures.append(TimeoutError("the killed process did not exit"))

    with pytest.raises(TimeoutError, match="the killed process did not exit"):
        await closer.close("handle", handle, handle.close, timeout=1.0)

    await closer.close("handle", handle, handle.close, timeout=1.0)

    assert handle.closes == 2, "a close that raised is started again"
    assert handle.closed


@pytest.mark.asyncio
async def test_a_close_that_finished_as_the_wait_timed_out_is_confirmed(monkeypatch):
    from kestrel_sovereign.llm import handle_closer

    closer = HandleCloser()
    handle = _Handle()

    async def wait_then_time_out(awaitable, timeout):
        await awaitable
        raise TimeoutError

    monkeypatch.setattr(handle_closer.asyncio, "wait_for", wait_then_time_out)
    await closer.close("handle", handle, handle.close, timeout=1.0)

    assert handle.closed
    assert closer.unconfirmed() == []


@pytest.mark.asyncio
async def test_a_close_that_failed_as_the_wait_timed_out_reports_its_own_error(
    monkeypatch,
):
    from kestrel_sovereign.llm import handle_closer

    closer = HandleCloser()
    handle = _Handle()
    handle.failures.append(ConnectionError("close failed"))

    async def wait_then_time_out(awaitable, timeout):
        try:
            await awaitable
        except ConnectionError:
            pass
        raise TimeoutError

    monkeypatch.setattr(handle_closer.asyncio, "wait_for", wait_then_time_out)
    with pytest.raises(ConnectionError, match="close failed"):
        await closer.close("handle", handle, handle.close, timeout=1.0)

    assert _labels(closer) == ["handle"]

