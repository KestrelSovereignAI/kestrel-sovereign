"""Original body control survives actual native PostgreSQL rollback cleanup."""

import asyncio

import asyncpg
import pytest

from kestrel_sdk.storage.database import QueryError
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, bind_execution_custody,
    execution_commit_outcome, is_execution_control_error,
)
from tests.integration.test_execution_authority_postgres import (
    GenerationFence, native_pg as _native_pg,
)

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("route", ["transaction", "operational", "nested"])
@pytest.mark.parametrize("control", ["denied", "unknown"])
@pytest.mark.parametrize("cleanup", ["error", "cancel"])
@pytest.mark.parametrize("boundary", ["rollback", "release"])
async def test_native_transaction_keeps_body_control_after_failed_rollback(native_pg, monkeypatch, route, control, cleanup, boundary):
    backend, observer, schema = native_pg
    original = asyncpg.transaction.Transaction.__aexit__
    original_release = asyncpg.pool.PoolAcquireContext.__aexit__
    marker = ExecutionCommitOutcomeError("unknown") if control == "unknown" else ExecutionAuthorityError("original child authority lost")
    rolled_back = []

    async def fail_after_real_rollback(transaction, exc_type, error, traceback):
        result = await original(transaction, exc_type, error, traceback)
        if boundary == "rollback" and isinstance(error, QueryError) and is_execution_control_error(error):
            rolled_back.append(True)
            if cleanup == "cancel":
                raise asyncio.CancelledError("second interruption after native rollback")
            raise OSError("native rollback acknowledgement failed")
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", fail_after_real_rollback)

    async def fail_after_real_release(checkout, exc_type, error, traceback):
        result = await original_release(checkout, exc_type, error, traceback)
        if boundary == "release" and error is not None and is_execution_control_error(error):
            rolled_back.append(True)
            if cleanup == "cancel":
                raise asyncio.CancelledError("second interruption after physical release")
            raise OSError("physical release acknowledgement failed")
        return result
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", fail_after_real_release)

    async def body():
        await backend.execute("INSERT INTO effects VALUES (1, 'rolled back')")
        raise QueryError("body's original child control") from marker

    with bind_execution_custody(GenerationFence()):
        with pytest.raises(BaseException) as caught:
            if route == "operational":
                async with backend.operational_session():
                    async with backend.transaction():
                        await body()
            elif route == "nested":
                async with backend.transaction():
                    async with backend.transaction():
                        await body()
            else:
                async with backend.transaction():
                    await body()
    assert rolled_back, "test did not cross the real native rollback boundary"
    assert is_execution_control_error(caught.value), "cleanup erased original body control"
    if control == "unknown":
        assert execution_commit_outcome(caught.value) == "unknown"
    assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 0
