"""Native driver retry classification, bounded passes and original control."""

import asyncio
import sqlite3
from types import SimpleNamespace

import asyncpg
import pytest
import pytest_asyncio

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    _explicit_control_chain,
)
from kestrel_sovereign.storage.db.interface import DatabaseError, QueryError
from kestrel_sovereign.storage.db.polling import (
    DatabasePollRetryPolicy,
    is_transient_database_poll_error,
)
from tests.utils.database_query_faults import prepare_transient_database_query_fault


@pytest_asyncio.fixture
async def native_fault(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        fault = await prepare_transient_database_query_fault(backend)
        yield SimpleNamespace(backend=backend, fail=fault)
    finally:
        await backend.close()


async def _original_error(fault):
    with pytest.raises(DatabaseError) as caught:
        await fault.fail()
    return caught.value


@pytest.mark.parametrize("transaction", [False, True])
async def test_real_driver_contention_is_recognized_after_known_rollback(
    native_fault, transaction
):
    p = native_fault
    if transaction:
        with pytest.raises(DatabaseError) as caught:
            async with p.backend.transaction():
                await p.fail()
        error = caught.value
    else:
        error = await _original_error(p)
    nodes = tuple(_explicit_control_chain(error))
    if p.backend.backend_type == "sqlite":
        assert any(
            isinstance(node, sqlite3.OperationalError)
            and node.sqlite_errorcode & 0xFF == sqlite3.SQLITE_BUSY
            for node in nodes
        )
    else:
        assert any(isinstance(node, asyncpg.SerializationError) for node in nodes)
    assert is_transient_database_poll_error(error)
    native = next(
        node for node in nodes if isinstance(node, (sqlite3.Error, asyncpg.PostgresError))
    )
    assert not is_transient_database_poll_error(native)
    assert tuple(_explicit_control_chain(error)) == nodes
    assert await p.backend.fetch_val("SELECT 1") == 1


@pytest.mark.parametrize(
    "carrier", ["bare", "message", "permanent", "foreign", "mixed", "cycle"]
)
async def test_generic_database_failures_and_mixed_graphs_are_not_retryable(
    native_fault, carrier
):
    error = await _original_error(native_fault)
    if carrier == "bare":
        error = QueryError("database is locked")
    elif carrier == "message":
        error = QueryError("transient database failure")
        error.__cause__ = sqlite3.OperationalError("database is locked")
    elif carrier == "permanent":
        with pytest.raises(DatabaseError) as caught:
            await native_fault.backend.fetch_all("SELECT * FROM missing_retry_table")
        error = caught.value
    elif carrier == "foreign":
        foreign = ValueError("business refusal")
        foreign.__cause__ = error
        error = QueryError("database carrier")
        error.__cause__ = foreign
    elif carrier == "mixed":
        error = ExceptionGroup("mixed errors", [error, ValueError("business refusal")])
    else:
        error.__cause__ = error
    assert not is_transient_database_poll_error(error)


@pytest.mark.parametrize("control", ["authority", "unknown", "committed"])
async def test_original_execution_control_prevents_poll_replay(native_fault, control):
    error = await _original_error(native_fault)
    original = (
        ExecutionAuthorityError("original admission withdrawn")
        if control == "authority"
        else ExecutionCommitOutcomeError(control)
    )
    original.__cause__ = error
    carrier = QueryError("retained database failure")
    carrier.__cause__ = original
    before = tuple(_explicit_control_chain(carrier))
    calls = []

    async def operation():
        calls.append(True)
        raise carrier

    with pytest.raises(QueryError) as caught:
        await DatabasePollRetryPolicy().run(
            operation, retry_seconds=0.001, label="owned-control-test"
        )
    assert caught.value is carrier and len(calls) == 1
    assert tuple(_explicit_control_chain(carrier)) == before
    assert any(node is original for node in before)


@pytest.mark.parametrize("attempts", [1, 2, 4])
async def test_exhausted_budget_raises_last_original_driver_failure(
    native_fault, attempts
):
    failures = []

    async def operation():
        try:
            await native_fault.fail()
        except DatabaseError as error:
            failures.append(error)
            raise

    with pytest.raises(DatabaseError) as caught:
        await DatabasePollRetryPolicy(max_attempts=attempts).run(
            operation, retry_seconds=0.001, label="owned-exhaustion-test"
        )
    assert len(failures) == attempts and caught.value is failures[-1]


async def test_success_resets_the_next_pass_budget_without_duplicate_writes(native_fault):
    failures = 0
    accepted = []
    policy = DatabasePollRetryPolicy(max_attempts=2)

    async def operation():
        nonlocal failures
        if failures == len(accepted):
            failures += 1
            await native_fault.fail()
        await native_fault.backend.execute(
            "INSERT INTO native_resident_retry_probe (value) VALUES (?)", (failures,)
        )
        accepted.append(failures)
        return failures

    for expected in (1, 2):
        assert await policy.run(
            operation, retry_seconds=0.001, label="owned-reset-test"
        ) == expected
    rows = await native_fault.backend.fetch_all(
        "SELECT value FROM native_resident_retry_probe ORDER BY value"
    )
    assert rows == [(1,), (2,)] and accepted == [1, 2]


async def test_cancellation_during_backoff_does_not_retry(native_fault):
    failed = asyncio.Event()
    calls = []

    async def operation():
        calls.append(True)
        try:
            await native_fault.fail()
        except DatabaseError:
            failed.set()
            raise

    owner = asyncio.create_task(
        DatabasePollRetryPolicy().run(
            operation, retry_seconds=60, label="owned-cancellation-test"
        )
    )
    try:
        await asyncio.wait_for(failed.wait(), 5)
        owner.cancel("original poll owner withdrawal")
        with pytest.raises(asyncio.CancelledError) as caught:
            await owner
        assert caught.value.args == ("original poll owner withdrawal",)
        assert len(calls) == 1
    finally:
        owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)


@pytest.mark.parametrize("attempts", [True, 0, -1, 1.5])
def test_invalid_database_retry_budget_is_refused(attempts):
    with pytest.raises(ValueError, match="positive integer"):
        DatabasePollRetryPolicy(max_attempts=attempts)


@pytest.mark.parametrize("delay", [True, 0, -1, float("nan"), float("inf")])
async def test_invalid_backoff_refuses_the_operation(delay):
    calls = []

    async def operation():
        calls.append(True)

    with pytest.raises(ValueError, match="finite and positive"):
        await DatabasePollRetryPolicy().run(
            operation, retry_seconds=delay, label="invalid-policy-test"
        )
    assert not calls


@pytest.mark.parametrize("label", [None, "", " "])
async def test_unidentified_polling_is_refused_before_the_operation(label):
    calls = []

    async def operation():
        calls.append(True)

    with pytest.raises(ValueError, match="concrete service label"):
        await DatabasePollRetryPolicy().run(
            operation, retry_seconds=0.001, label=label
        )
    assert not calls
