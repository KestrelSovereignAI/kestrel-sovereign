"""Real HTTPS callbacks, durable ACK/retry, and original deletion ownership."""

import asyncio
from datetime import timedelta

import pytest

from kestrel_sovereign.a2a.push_transport import (
    NativePushTransport,
    PushDeliveryRevoked,
)
from kestrel_sovereign.a2a.push_worker import (
    NativePushWorker,
    PushDeliveryLeaseLost,
    PushRetirementPending,
)
from kestrel_sovereign.signals.durable import ACKNOWLEDGED, FAILED, LEASED, RETRY
from kestrel_sovereign.storage.db.interface import QueryError
from kestrel_sovereign.storage.db.polling import DatabasePollRetryPolicy
from tests.unit.test_a2a_native_push_outbox import callback
from tests.unit.test_a2a_native_push_outbox import (
    outbox as outbox,  # noqa: PLC0414 - pytest fixture re-export
)
from tests.unit.test_a2a_native_push_outbox import (
    push_store as push_store,  # noqa: PLC0414 - pytest fixture dependency
)
from tests.utils.native_push import fixture_push_authorizer, intercept_push_commit_ack
from tests.utils.outbound_tls import (
    outbound_close_barrier as outbound_close_barrier,  # noqa: PLC0414 - pytest fixture re-export
)
from tests.utils.outbound_tls import (
    outbound_tls as outbound_tls,  # noqa: PLC0414 - pytest fixture re-export
)


def worker(value, *, database_retry_policy=None):
    return NativePushWorker(
        value,
        NativePushTransport(value.configurations, timeout=timedelta(seconds=5)),
        authorize_creator=fixture_push_authorizer(value.configurations.task_store),
        poll_interval=timedelta(milliseconds=10),
        renewal_interval=timedelta(milliseconds=20),
        retry_delay=timedelta(seconds=1),
        retry_max_delay=timedelta(seconds=300),
        max_delivery_attempts=8,
        batch_size=10,
        database_retry_policy=database_retry_policy,
    )


async def test_worker_uses_its_explicit_typed_database_retry_budget(outbox):
    value, _ = outbox
    policy = DatabasePollRetryPolicy(max_attempts=4)
    assert worker(value, database_retry_policy=policy).database_retry_policy is policy
    with pytest.raises(TypeError, match="database retry policy"):
        worker(value, database_retry_policy=True)


async def test_worker_requires_an_explicit_live_recipient_authorizer(outbox):
    value, _ = outbox
    with pytest.raises(TypeError, match="live recipient authorization"):
        NativePushWorker(
            value, NativePushTransport(value.configurations, timeout=timedelta(seconds=5)),
            authorize_creator=None, poll_interval=timedelta(seconds=1),
            renewal_interval=timedelta(seconds=20), retry_delay=timedelta(seconds=1), batch_size=10,
            retry_max_delay=timedelta(seconds=300), max_delivery_attempts=8,
        )


async def register(value, task, url):
    native = callback(task)
    native.url = url
    config = await value.register(native, creator_id="did:creator")
    runner = worker(value)
    # The initial snapshot is future-only work. No historical task prefix is
    # claimed before the actual caller owns its first network delivery.
    assert (
        await value.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts")
        == []
    )
    return runner, config


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
async def test_native_delivery_acknowledges_only_after_actual_https_and_retirement(
    outbox, outbound_tls
):
    value, task = outbox
    runner, config = await register(value, task, outbound_tls.url + "/callback")
    assert await runner.run_once() == 1
    assert len(outbound_tls.posts) == 1
    assert outbound_tls.posts[0][2]["task"]["id"] == task.id
    assert await value.backend.fetch_all(
        "SELECT retired FROM a2a_push_delivery_attempts"
    ) == [(1,)]
    assert await value.backend.fetch_all(
        "SELECT status FROM durable_signal_deliveries"
    ) == [(ACKNOWLEDGED,)]
    restarted = worker(value)
    assert await restarted.run_once() == 0
    assert len(outbound_tls.posts) == 1
    await restarted.delete(task.id, config.config.id, creator_id="did:creator")
    await restarted.delete(task.id, config.config.id, creator_id="did:creator")
    assert (
        await value.configurations.get(
            task.id, config.config.id, creator_id="did:creator"
        )
        is None
    )
    assert await value.backend.fetch_all(
        "SELECT active FROM durable_signal_consumers"
    ) == [(value.signals.to_bool_param(False),)]


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 3}], indirect=True)
async def test_http_failure_uses_the_existing_retry_ledger_without_secret_diagnostics(
    outbox, outbound_tls
):
    value, task = outbox
    runner, _ = await register(
        value, task, outbound_tls.url + "/failure?private=callback-secret"
    )
    before_retry = value.signals.now_utc()
    assert await runner.run_once() == 1
    row = await value.backend.fetch_one(
        "SELECT status, attempts, last_error FROM durable_signal_deliveries WHERE status = ?",
        (RETRY,),
    )
    assert row == (RETRY, 1, "HTTPStatusError")
    retry_time = await value.backend.fetch_one(
        "SELECT next_attempt_at FROM durable_signal_deliveries WHERE status = ?",
        (RETRY,),
    )
    assert (
        value.signals.from_timestamp_field(retry_time[0])
        >= before_retry + runner.retry_delay
    )
    await value.backend.execute(
        "UPDATE durable_signal_deliveries SET next_attempt_at = ? WHERE status = ?",
        (
            value.signals.to_timestamp_param(
                value.signals.now_utc() - timedelta(seconds=1)
            ),
            RETRY,
        ),
    )
    restarted = worker(value)
    assert await restarted.run_once() == 1
    row = await value.backend.fetch_one(
        "SELECT status, attempts, last_error FROM durable_signal_deliveries WHERE status = ?",
        (RETRY,),
    )
    assert row == (RETRY, 2, "HTTPStatusError")
    assert len(outbound_tls.posts) == 2
    assert await value.backend.fetch_all(
        "SELECT retired FROM a2a_push_delivery_attempts"
    ) == [(1,), (1,)]


@pytest.mark.parametrize("check", ["backoff", "exhaustion"])
@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 4}], indirect=True)
async def test_terminal_callback_has_a_persisted_retry_budget_and_capped_backoff(
    outbox, outbound_tls, check
):
    from kestrel_sdk.native_a2a.types import Task, TaskState

    value, original = outbox
    terminal = Task.FromString(original.SerializeToString())
    terminal.status.state = TaskState.TASK_STATE_COMPLETED
    assert await value.configurations.task_store.save_recipient_terminal_outcome(
        terminal, baseline=original,
        recipient_agent_id=value.configurations.scope.recipient_id,
        operation_id="terminal-before-callback",
    )
    runner, _ = await register(value, terminal, outbound_tls.url + "/failure")
    runner.max_delivery_attempts = 3
    runner.retry_max_delay = timedelta(seconds=2)
    for attempt in range(1, 4):
        before = value.signals.now_utc()
        assert await runner.run_once() == 1
        row = await value.backend.fetch_one(
            "SELECT status, attempts, next_attempt_at, acknowledged_at, terminal_at "
            "FROM durable_signal_deliveries"
        )
        assert row[1] == attempt and row[3] is None
        if attempt < 3:
            assert row[0] == RETRY
            if check == "backoff":
                expected = timedelta(seconds=min(2 ** (attempt - 1), 2))
                assert value.signals.from_timestamp_field(row[2]) >= before + expected
            # Advance only a released RETRY row after its real HTTP owner has
            # retired; no live lease or physical request is timed out by this.
            await value.backend.execute(
                "UPDATE durable_signal_deliveries SET next_attempt_at = ? WHERE status = ?",
                (value.signals.to_timestamp_param(value.signals.now_utc() - timedelta(seconds=1)), RETRY),
            )
        else:
            assert row[0] == FAILED and row[2] is None and row[4] is not None
    assert len(outbound_tls.posts) == 3
    assert await value.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(1,), (1,), (1,)]
    assert await worker(value).run_once() == 0
    assert len(outbound_tls.posts) == 3


@pytest.mark.parametrize("attempt,expected", [(1, 1), (2, 2), (3, 3), (10**9, 3)])
async def test_retry_delay_caps_work_even_for_large_persisted_attempt_counts(outbox, attempt, expected):
    value, _ = outbox
    runner = worker(value)
    runner.retry_max_delay = timedelta(seconds=3)
    assert runner._delivery_retry_delay(attempt) == timedelta(seconds=expected)


@pytest.mark.parametrize("attempt", [0, -1, True, 1.5])
async def test_retry_delay_refuses_noncanonical_attempt_counts(outbox, attempt):
    value, _ = outbox
    with pytest.raises(ValueError, match="positive attempt count"):
        worker(value)._delivery_retry_delay(attempt)


async def test_retry_delay_cap_does_not_overflow_a_valid_timedelta(outbox):
    value, _ = outbox
    runner = worker(value)
    runner.retry_delay = timedelta.max - timedelta(days=1)
    runner.retry_max_delay = timedelta.max
    assert runner._delivery_retry_delay(2) == timedelta.max


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
async def test_retry_exhaustion_waits_for_original_http_retirement(
    outbox, outbound_tls, outbound_close_barrier
):
    value, task = outbox
    runner, _ = await register(value, task, outbound_tls.url + "/failure")
    runner.max_delivery_attempts = 1
    running = asyncio.create_task(runner.run_once())
    try:
        await outbound_close_barrier.entered.wait()
        assert not running.done()
        assert await value.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(0,)]
        assert await value.backend.fetch_all("SELECT status FROM durable_signal_deliveries") == [(LEASED,)]
        outbound_close_barrier.release.set()
        assert await running == 1
        assert await value.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(1,)]
        assert await value.backend.fetch_all("SELECT status, acknowledged_at FROM durable_signal_deliveries") == [(FAILED, None)]
    finally:
        outbound_close_barrier.release.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)
@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
async def test_dispatch_gate_checks_the_original_delivery_lease_after_dns(
    outbox, outbound_tls, monkeypatch
):
    from kestrel_sovereign.a2a import push_transport

    value, task = outbox
    runner, _ = await register(value, task, outbound_tls.url + "/callback")
    validate = push_transport.assert_safe_url
    subscription = (await value.subscriptions())[0]
    monitor_renewed = asyncio.Event()
    original_renew = value.signals.renew_delivery_lease
    runner.renewal_interval = 59

    async def observe_renewal(**kwargs):
        result = await original_renew(**kwargs)
        monitor_renewed.set()
        return result

    monkeypatch.setattr(value.signals, "renew_delivery_lease", observe_renewal)

    async def revoke_after_validation(*args, **kwargs):
        approved = await validate(*args, **kwargs)
        await monitor_renewed.wait()
        await value.signals.deactivate_consumer(
            agent_id=value.agent_id, consumer_id=subscription.consumer_id
        )
        return approved

    monkeypatch.setattr(push_transport, "assert_safe_url", revoke_after_validation)
    await runner.run_once()
    assert outbound_tls.dials == [] and outbound_tls.posts == []
    assert await value.backend.fetch_all(
        "SELECT retired FROM a2a_push_delivery_attempts"
    ) == [(1,)]
    assert await value.backend.fetch_all(
        "SELECT status FROM durable_signal_deliveries WHERE status != ?",
        (ACKNOWLEDGED,),
    ) == [(FAILED,)]


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
async def test_delete_physically_joins_original_response_close_before_ack(
    outbox, outbound_tls, monkeypatch, outbound_close_barrier
):
    value, task = outbox
    runner, config = await register(value, task, outbound_tls.url + "/callback")
    entered, release = outbound_close_barrier.entered, outbound_close_barrier.release
    revoked = asyncio.Event()
    original_revoke = value.revoke
    monitor_observed = asyncio.Event()
    original_read = value.configurations.get
    original_renew = runner._renew

    async def observe_renewal(delivery):
        try:
            return await original_renew(delivery)
        except PushDeliveryLeaseLost:
            # Revocation may win between the monitor's config read and its
            # lease renewal. That is the same real retirement boundary.
            monitor_observed.set()
            raise

    monkeypatch.setattr(runner, "_renew", observe_renewal)

    async def observe_revocation(*args, **kwargs):
        result = await original_read(*args, **kwargs)
        if result is None:
            monitor_observed.set()
        return result

    monkeypatch.setattr(value.configurations, "get", observe_revocation)

    async def revoke(*args, **kwargs):
        result = await original_revoke(*args, **kwargs)
        # The real monitor wins before Delete can issue its own cancellation.
        await monitor_observed.wait()
        revoked.set()
        return result

    monkeypatch.setattr(value, "revoke", revoke)
    sending = asyncio.create_task(runner.run_once())
    deleting = None
    try:
        await entered.wait()
        deleting = asyncio.create_task(
            runner.delete(task.id, config.config.id, creator_id="did:creator")
        )
        await revoked.wait()
        assert not deleting.done() and not sending.done()
        assert await value.backend.fetch_all(
            "SELECT retired FROM a2a_push_delivery_attempts"
        ) == [(0,)]
        assert await value.backend.fetch_one(
            "SELECT state, ciphertext IS NOT NULL FROM a2a_push_configs"
        ) == ("deleting", True)
        release.set()
        assert await asyncio.gather(sending, deleting) == [1, None]
        assert await value.backend.fetch_one(
            "SELECT state, ciphertext FROM a2a_push_configs"
        ) == ("deleted", None)
    finally:
        release.set()
        if deleting is not None:
            await asyncio.gather(deleting, return_exceptions=True)
        await asyncio.gather(sending, return_exceptions=True)


async def test_remote_or_failed_physical_owner_blocks_delete_without_erasing_credentials(
    outbox,
):
    value, task = outbox
    config = await value.register(callback(task), creator_id="did:creator")
    attempt = value.configurations.prepare_delivery(
        config, executor_id="original-other-host"
    )
    assert await value.configurations.admit_delivery(attempt)
    runner = worker(value)
    with pytest.raises(PushRetirementPending):
        await runner.delete(task.id, config.config.id, creator_id="did:creator")
    assert await value.backend.fetch_one(
        "SELECT state, ciphertext IS NOT NULL FROM a2a_push_configs"
    ) == ("deleting", True)
    assert await value.configurations.retire_delivery(attempt)
    await runner.delete(task.id, config.config.id, creator_id="did:creator")
    assert await value.backend.fetch_one(
        "SELECT state, ciphertext FROM a2a_push_configs"
    ) == ("deleted", None)


async def test_task_cleanup_requires_config_deletion_even_before_the_first_http_attempt(
    outbox,
):
    value, task = outbox
    config = await value.register(callback(task), creator_id="did:creator")
    with pytest.raises(QueryError, match="active push configuration"):
        await value.configurations.task_store.delete(task.id)
    await worker(value).delete(task.id, config.config.id, creator_id="did:creator")
    assert await value.configurations.task_store.delete(task.id)
    assert await value.subscriptions() == []


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
async def test_caller_cancellation_waits_for_physical_retirement_and_preserves_the_lease(
    outbox, outbound_tls, outbound_close_barrier
):
    value, task = outbox
    runner, _ = await register(value, task, outbound_tls.url + "/callback")
    entered, release = outbound_close_barrier.entered, outbound_close_barrier.release
    sending = asyncio.create_task(runner.run_once())
    try:
        await entered.wait()
        sending.cancel("original host cancellation")
        await asyncio.sleep(0)
        assert not sending.done()
        assert await value.backend.fetch_all(
            "SELECT retired FROM a2a_push_delivery_attempts"
        ) == [(0,)]
        release.set()
        with pytest.raises(asyncio.CancelledError) as failure:
            await sending
        assert failure.value.args == ("original host cancellation",)
        assert await value.backend.fetch_all(
            "SELECT retired FROM a2a_push_delivery_attempts"
        ) == [(1,)]
        assert await value.backend.fetch_one(
            "SELECT status FROM durable_signal_deliveries WHERE status != ?",
            (ACKNOWLEDGED,),
        ) == (LEASED,)
    finally:
        release.set()
        await asyncio.gather(sending, return_exceptions=True)


async def test_lease_transition_commits_before_reporting_original_cancellation(
    outbox, monkeypatch
):
    value, task = outbox
    await value.register(callback(task), creator_id="did:creator")
    await value.forward_pending(limit=10)
    runner = worker(value)
    subscription = (await value.subscriptions())[0]
    delivery = await value.signals.claim_delivery(
        agent_id=value.agent_id,
        consumer_id=subscription.consumer_id,
        executor_id=runner.executor_id,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original_renew = value.signals.renew_delivery_lease

    async def original_transaction(**kwargs):
        async with value.backend.transaction():
            result = await original_renew(**kwargs)
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(value.signals, "renew_delivery_lease", original_transaction)
    renewing = asyncio.create_task(runner._renew(delivery))
    try:
        await entered.wait()
        renewing.cancel("original lease observer cancellation")
        await asyncio.sleep(0)
        assert not renewing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await renewing
        assert caught.value.args == ("original lease observer cancellation",)
        assert await value.backend.fetch_one(
            "SELECT status, lease_token FROM durable_signal_deliveries WHERE delivery_id = ?",
            (delivery.delivery_id,),
        ) == (LEASED, delivery.lease_token)
    finally:
        release.set()
        await asyncio.gather(renewing, return_exceptions=True)


@pytest.mark.parametrize("operation", ["forward", "claim"])
async def test_metadata_commit_remains_owned_through_original_caller_withdrawal(
    outbox, monkeypatch, operation
):
    from kestrel_sovereign.execution_custody import is_execution_control_error
    from tests.unit.test_a2a_native_push_outbox import callback

    value, task = outbox
    runner = worker(value)
    entered, release = asyncio.Event(), asyncio.Event()
    await value.register(callback(task), creator_id="did:creator")
    if operation == "claim":
        await value.forward_pending(limit=10)
        owner, method = value.signals, "claim_delivery"
    else:
        owner, method = value, "forward_pending"

    async def held_ack():
        entered.set()
        await release.wait()

    intercept_push_commit_ack(value, monkeypatch, held_ack, owner=owner, method=method)
    processing = asyncio.create_task(runner.run_once())
    try:
        await entered.wait()
        processing.cancel("original metadata caller withdrawal")
        await asyncio.sleep(0)
        assert not processing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await processing
        assert caught.value.args == ("original metadata caller withdrawal",)
        assert not is_execution_control_error(caught.value)
        if operation == "claim":
            assert await value.backend.fetch_one(
                "SELECT COUNT(*) FROM durable_signal_deliveries WHERE status = ?",
                (LEASED,),
            ) == (1,)
        else:
            assert await value.backend.fetch_one(
                "SELECT COUNT(*) FROM durable_signal_events"
            ) == (1,)
            assert await value.changes.pending(limit=10) == []
        assert (
            await value.backend.fetch_all("SELECT * FROM a2a_push_delivery_attempts")
            == []
        )
    finally:
        release.set()
        await asyncio.gather(processing, return_exceptions=True)


@pytest.mark.parametrize(
    "error_type", [asyncio.CancelledError, PushDeliveryRevoked, PushDeliveryLeaseLost]
)
def test_planned_revocation_cannot_erase_original_commit_control(error_type):
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    original = ExecutionCommitOutcomeError("unknown")
    error = error_type("original delivery control")
    error.__cause__ = original
    assert NativePushWorker._delivery_error(error, planned_stop=True) is error


@pytest.mark.parametrize(
    "error_type", [asyncio.CancelledError, PushDeliveryRevoked, PushDeliveryLeaseLost]
)
def test_plain_delivery_withdrawal_is_local_only_to_its_planned_stop(error_type):
    error = error_type("original delivery withdrawal")
    assert NativePushWorker._delivery_error(error, planned_stop=False) is error
    assert NativePushWorker._delivery_error(error, planned_stop=True) is None


async def test_runner_readiness_follows_its_managed_lifetime(outbox, monkeypatch):
    value, _ = outbox
    runner = worker(value)
    assert not runner.ready
    entered, release = asyncio.Event(), asyncio.Event()
    original_forward = value.forward_pending

    async def forward(**kwargs):
        entered.set()
        await release.wait()
        return await original_forward(**kwargs)

    monkeypatch.setattr(value, "forward_pending", forward)
    resident = asyncio.create_task(runner.run())
    try:
        await entered.wait()
        assert not runner.ready
        release.set()
        await runner.started.wait()
        assert runner.ready
        resident.cancel("original host shutdown")
        with pytest.raises(asyncio.CancelledError):
            await resident
        assert not runner.ready
        assert runner._active == {}
    finally:
        release.set()
        resident.cancel()
        await asyncio.gather(resident, return_exceptions=True)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("poll_interval", timedelta()),
        ("poll_interval", None),
        ("renewal_interval", timedelta(seconds=60)),
        ("renewal_interval", timedelta(seconds=-1)),
        ("retry_delay", timedelta()),
        ("retry_max_delay", timedelta()),
        ("retry_max_delay", timedelta(milliseconds=500)),
        ("max_delivery_attempts", 0),
        ("max_delivery_attempts", True),
        ("batch_size", 0),
        ("batch_size", True),
    ],
)
async def test_invalid_worker_policy_refuses_before_execution(outbox, field, bad):
    value, _ = outbox
    options = {
        "poll_interval": timedelta(seconds=1),
        "renewal_interval": timedelta(seconds=20),
        "retry_delay": timedelta(seconds=1),
        "retry_max_delay": timedelta(seconds=300),
        "max_delivery_attempts": 8,
        "batch_size": 10,
        "authorize_creator": fixture_push_authorizer(value.configurations.task_store),
    }
    options[field] = bad
    with pytest.raises(ValueError):
        NativePushWorker(
            value,
            NativePushTransport(value.configurations, timeout=timedelta(seconds=5)),
            **options,
        )
