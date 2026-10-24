"""Lost original PostgreSQL sessions cannot retain imaginary rollback owners."""

import asyncio

import asyncpg
import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.storage.db.interface import DatabaseError, QueryError
from kestrel_sovereign.storage.db.polling import is_transient_database_poll_error


@pytest.mark.parametrize("body_failure", [False, True])
@pytest.mark.parametrize("native_a2a_backend_factory", ["postgres"], indirect=True)
async def test_terminated_transaction_retires_original_checkout_without_commit(
    native_a2a_backend_factory, body_failure,
):
    backend = native_a2a_backend_factory()
    assert backend.backend_type == "postgres"
    control = native_a2a_backend_factory()
    await backend.connect()
    await control.connect()
    closed = asyncio.Event()
    try:
        await backend.execute("CREATE TABLE original_session_probe (value INTEGER)")
        with pytest.raises((DatabaseError, ExecutionCommitOutcomeError, BaseExceptionGroup)) as lost:
            async with backend.transaction():
                native = backend._current_txn_conn()._con
                native.add_termination_listener(lambda connection: closed.set())
                pid = await backend.fetch_val("SELECT pg_backend_pid()")
                await backend.execute("INSERT INTO original_session_probe VALUES (1)")
                assert await control.fetch_val("SELECT pg_terminate_backend(?)", (pid,))
                await asyncio.wait_for(closed.wait(), 5)
                if body_failure:
                    await backend.fetch_val("SELECT 1")
        assert native.is_closed()
        if body_failure:
            assert not is_execution_control_error(lost.value), (
                "a terminated original session retained unresolved rollback ownership"
            )
            assert is_transient_database_poll_error(lost.value)
            unproven = QueryError("another driver's unproven interface failure")
            unproven.__cause__ = asyncpg.InterfaceError("unproven original session")
            assert not is_transient_database_poll_error(
                ExceptionGroup("independent session failures", [lost.value, unproven])
            ), "one closed session vouched for another session's interface error"
        else:
            assert is_execution_control_error(lost.value), (
                "an accepted COMMIT owner's failed acknowledgement lost its control"
            )
            assert execution_commit_outcome(lost.value) == "unknown"
            assert not is_transient_database_poll_error(lost.value)
        assert await backend.fetch_val("SELECT COUNT(*) FROM original_session_probe") == 0
        assert await backend.fetch_val("SELECT 1") == 1
    finally:
        await control.close()
        # This test owns the disposable pool. Its deadline is a leak detector,
        # never an inference that a transaction or checkout physically retired.
        pool = backend._pool
        closing = asyncio.create_task(backend.close())
        done, _ = await asyncio.wait((closing,), timeout=5)
        if not done:
            pool.terminate()
        await closing
        assert done, "the original lost-session checkout leaked its pool lease"


@pytest.mark.parametrize("native_a2a_backend_factory", ["postgres"], indirect=True)
async def test_failed_native_exit_on_a_live_session_retains_original_control(
    native_a2a_backend_factory, monkeypatch,
):
    backend = native_a2a_backend_factory()
    assert backend.backend_type == "postgres"
    await backend.connect()
    original = asyncpg.transaction.Transaction.__aexit__
    originals = []

    async def duplicate_native_exit(transaction, exc_type, exc, tb):
        # The real driver rejects a repeated native exit while the same
        # session remains live. A finished exit task alone proves no closed
        # session, even when its first ROLLBACK happened to succeed internally.
        await original(transaction, exc_type, exc, tb)
        originals.append(transaction)
        return await original(transaction, exc_type, exc, tb)

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", duplicate_native_exit)
    try:
        with pytest.raises(DatabaseError) as failed:
            async with backend.transaction():
                native = backend._current_txn_conn()._con
                raise ValueError("original transaction body refusal")
        assert len(originals) == 1 and not native.is_closed()
        assert is_execution_control_error(failed.value), (
            "a live session was classified as physically closed"
        )
        assert not is_transient_database_poll_error(failed.value)
    finally:
        await backend.close()
