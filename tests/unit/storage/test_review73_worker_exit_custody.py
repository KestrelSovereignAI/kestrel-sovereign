"""Native SQLite worker-exit hooks must not erase retained control receipts."""

import asyncio

import aiosqlite
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
)
from tests.utils.original_failure import OriginalFailure
from kestrel_sovereign.storage.db.sqlite import (
    SQLiteBackend,
    _prune_retained_aiosqlite_closes,
    _retain_aiosqlite_close,
    _wait_for_aiosqlite_worker_shutdown,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_origin", ["close", "hook"])
async def test_native_worker_exit_hooks_keep_original_unreported_control(tmp_path, failure_origin):
    connection = await aiosqlite.connect(tmp_path / "native-exit.db")
    retained = {}
    calls = []
    original = ExecutionAuthorityError(f"original {failure_origin} control")

    async def close():
        await connection.close()
        await _wait_for_aiosqlite_worker_shutdown(connection)
        if failure_origin == "close":
            raise original

    def hook():
        calls.append("first")
        if failure_origin == "hook":
            raise original

    close_task = create_owned_task(close())
    _retain_aiosqlite_close(connection, close_task, retained)
    record = retained[connection]
    record.worker_exit_hooks.extend([hook, lambda: calls.append("second")])
    retired = asyncio.get_running_loop().create_future()
    record.retirement_task.add_done_callback(lambda _: retired.set_result(None))
    try:
        outcome = await asyncio.wait_for(await_owned_task(close_task), 3)
        await asyncio.wait_for(await_owned_task(record.retirement_task), 3)
        await asyncio.wait_for(retired, 3)
        assert outcome.error is (original if failure_origin == "close" else None)
        assert calls == ["first", "second"]
        assert retained[connection] is record
        assert record.control_error is original
        assert not record.control_reported
        _prune_retained_aiosqlite_closes(retained, preserve_control_outcomes=True)
        assert retained[connection] is record
        assert calls == ["first", "second"], "physical-exit hooks run exactly once"
        record.control_reported = True
        _prune_retained_aiosqlite_closes(retained, preserve_control_outcomes=True)
        assert not retained
    finally:
        await await_owned_task(close_task)
        await await_owned_task(record.retirement_task)
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_cancellation", [False, True])
async def test_actual_close_and_hook_failures_keep_strongest_original(
    tmp_path, monkeypatch, hook_cancellation,
):
    backend = SQLiteBackend(str(tmp_path / "combined-close.db"))
    await backend.connect()
    connection = backend._connection
    native_close = connection.close
    original = ExecutionCommitOutcomeError("unknown")
    original_source = OriginalFailure.capture(original)
    hook_error = (
        asyncio.CancelledError("worker exit hook cancelled")
        if hook_cancellation else ExecutionAuthorityError("worker exit hook denied")
    )
    calls = []

    async def close_with_original_receipt():
        await native_close()
        await _wait_for_aiosqlite_worker_shutdown(connection)
        raise original

    def failing_hook():
        calls.append("failing")
        raise hook_error

    backend._worker_exit_hooks[connection] = [
        failing_hook, lambda: calls.append("joined"),
    ]
    monkeypatch.setattr(connection, "close", close_with_original_receipt)
    owner = create_owned_task(backend.close())
    try:
        outcome = await asyncio.wait_for(await_owned_task(owner), 3)
        original_source.assert_retained(outcome.error)
        assert hook_error in tuple(_explicit_control_chain(outcome.error))
        assert calls == ["failing", "joined"]
        assert backend._connection is None
        assert not backend.connection_retirement_pending
    finally:
        await await_owned_task(owner)
        await native_close()
