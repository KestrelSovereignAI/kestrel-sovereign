"""Original HTTP retirement survives known database outages without another POST."""

import asyncio
import sqlite3
from datetime import timedelta

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.signals.durable import ACKNOWLEDGED, LEASED
from kestrel_sovereign.storage.db.interface import DatabaseError
from kestrel_sovereign.storage.db.polling import is_transient_database_poll_error
from tests.unit.test_a2a_native_push_outbox import outbox as outbox  # noqa: PLC0414
from tests.unit.test_a2a_native_push_outbox import (
    push_store as push_store,  # noqa: PLC0414
)
from tests.unit.test_a2a_native_push_worker import register
from tests.utils.database_query_faults import prepare_transient_database_query_fault
from tests.utils.native_push import intercept_push_commit_ack
from tests.utils.outbound_tls import outbound_tls as outbound_tls  # noqa: PLC0414


async def retirement_fault(value):
    original = value.configurations.retire_delivery
    if value.backend.backend_type == "postgres":
        fault = await prepare_transient_database_query_fault(value.backend)

        async def fail(attempt):
            await fault()

        return fail

    async def fail(attempt):
        timeout = await value.backend.fetch_val("PRAGMA busy_timeout")
        await value.backend.execute("PRAGMA busy_timeout = 1")
        blocker = sqlite3.connect(value.backend.db_path)
        try:
            blocker.execute("BEGIN IMMEDIATE")
            await original(attempt)
        finally:
            blocker.rollback()
            blocker.close()
            await value.backend.execute(f"PRAGMA busy_timeout = {int(timeout)}")
        raise AssertionError("a real competing SQLite writer did not block retirement")

    return fail


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 4}], indirect=True)
@pytest.mark.parametrize("delete_during_retry", [False, True])
async def test_original_closed_http_owner_retries_only_its_retirement_write(
    outbox, outbound_tls, monkeypatch, delete_during_retry
):
    value, task = outbox
    fail = await retirement_fault(value)
    runner, configuration = await register(value, task, outbound_tls.url + "/callback")
    runner.transport.retry_delay = timedelta(milliseconds=10)
    runner.renewal_interval = 20
    original = value.configurations.retire_delivery
    attempts, owners, faults = [], [], []
    retry_entered, release_retry = asyncio.Event(), asyncio.Event()
    if not delete_during_retry:
        release_retry.set()

    async def retire_original(attempt):
        attempts.append(attempt)
        owners.append(asyncio.current_task())
        assert len(outbound_tls.posts) == 1, "retirement recovery repeated HTTP"
        if len(attempts) <= 4:
            try:
                await fail(attempt)
            except DatabaseError as error:
                faults.append(error)
                assert is_transient_database_poll_error(error)
                raise
        retry_entered.set()
        await release_retry.wait()
        return await original(attempt)

    monkeypatch.setattr(value.configurations, "retire_delivery", retire_original)
    sending = asyncio.create_task(runner.run_once())
    observing = deleting = None
    try:
        if delete_during_retry:
            observing = asyncio.create_task(retry_entered.wait())
            await asyncio.wait((sending, observing), timeout=5, return_when=asyncio.FIRST_COMPLETED)
            assert retry_entered.is_set(), (
                "a known retirement write failure discarded the original close owner"
            )
            deleting = asyncio.create_task(runner.delete(
                task.id, configuration.config.id, creator_id="did:creator"
            ))
            ended, _ = await asyncio.wait((deleting,), timeout=0.05)
            assert not ended and not sending.done()
            assert owners[-1].cancelling() == 0, "Delete cancelled the original retirement writer"
            assert await value.backend.fetch_all(
                "SELECT retired FROM a2a_push_delivery_attempts"
            ) == [(0,)]
    finally:
        release_retry.set()
        if observing is not None:
            observing.cancel()
            await asyncio.gather(observing, return_exceptions=True)
        outcome = await await_owned_task(sending)
        deletion = await await_owned_task(deleting) if deleting is not None else None
    assert outcome.error is None and outcome.result == 1, (
        "known metadata contention prevented retirement of a physically closed HTTP owner",
        outcome.error,
    )
    assert len(attempts) == 5 and len(faults) == 4
    assert all(attempt is attempts[0] for attempt in attempts)
    assert all(owner is owners[0] for owner in owners) and owners[0].done()
    assert len(outbound_tls.posts) == 1 and not runner._active
    if deletion is None:
        assert await value.backend.fetch_all(
            "SELECT status FROM durable_signal_deliveries"
        ) == [(ACKNOWLEDGED,)]
        await runner.delete(task.id, configuration.config.id, creator_id="did:creator")
    else:
        assert deletion.error is None
    assert await value.backend.fetch_one(
        "SELECT state, ciphertext FROM a2a_push_configs"
    ) == ("deleted", None)


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
async def test_permanent_retirement_sql_failure_stops_without_retry_or_ack(
    outbox, outbound_tls, monkeypatch
):
    value, task = outbox
    runner, _ = await register(value, task, outbound_tls.url + "/callback")
    runner.transport.retry_delay = timedelta(milliseconds=10)
    calls = []

    async def invalid_retirement(attempt):
        calls.append(attempt)
        assert len(calls) == 1, "a permanent retirement SQL defect was retried"
        await value.backend.fetch_all("SELECT missing_column FROM absent_retirement_table")

    monkeypatch.setattr(value.configurations, "retire_delivery", invalid_retirement)
    with pytest.raises(DatabaseError) as refused:
        await runner.run_once()
    assert not is_execution_control_error(refused.value)
    assert len(calls) == 1 and len(outbound_tls.posts) == 1
    assert await value.backend.fetch_all(
        "SELECT status FROM durable_signal_deliveries"
    ) == [(LEASED,)]
    assert await value.backend.fetch_all(
        "SELECT retired FROM a2a_push_delivery_attempts"
    ) == [(0,)]


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
@pytest.mark.parametrize("route", ["/callback", "/failure"])
async def test_uncertain_retirement_commit_never_retries_or_acknowledges(
    outbox, outbound_tls, monkeypatch, route
):
    value, task = outbox
    runner, _ = await register(value, task, outbound_tls.url + route)
    runner.max_delivery_attempts = 1
    runner.transport.retry_delay = timedelta(milliseconds=10)
    original = OSError("original retirement COMMIT reply lost")
    calls = []

    async def fail_original_ack():
        calls.append(True)
        assert len(calls) == 1, "uncertain retirement COMMIT was retried"
        raise original

    intercept_push_commit_ack(
        value, monkeypatch, fail_original_ack,
        owner=value.configurations, method="retire_delivery",
    )
    with pytest.raises((DatabaseError, ExecutionCommitOutcomeError, BaseExceptionGroup)) as refused:
        await runner.run_once()
    assert is_execution_control_error(refused.value)
    assert execution_commit_outcome(refused.value) == "unknown"
    assert any(error is original for error in _explicit_control_chain(refused.value))
    assert len(calls) == 1 and len(outbound_tls.posts) == 1 and not runner._active
    assert await value.backend.fetch_all(
        "SELECT status FROM durable_signal_deliveries"
    ) == [(LEASED,)]
