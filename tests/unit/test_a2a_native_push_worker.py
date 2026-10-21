"""Real HTTPS callbacks, durable ACK/retry, and original deletion ownership."""

import asyncio
from datetime import timedelta

import httpx
import pytest

from kestrel_sovereign.a2a.push_transport import NativePushTransport
from kestrel_sovereign.a2a.push_worker import NativePushWorker, PushRetirementPending
from kestrel_sovereign.signals.durable import ACKNOWLEDGED, FAILED, LEASED, RETRY
from kestrel_sovereign.storage.db.interface import QueryError
from tests.unit.test_a2a_native_push_outbox import callback
from tests.unit.test_a2a_native_push_outbox import (
    outbox as outbox,  # noqa: PLC0414 - pytest fixture re-export
)
from tests.unit.test_a2a_native_push_outbox import (
    push_store as push_store,  # noqa: PLC0414 - pytest fixture dependency
)
from tests.utils.outbound_tls import (
    outbound_tls as outbound_tls,  # noqa: PLC0414 - pytest fixture re-export
)


def worker(value):
    return NativePushWorker(
        value,
        NativePushTransport(value.configurations, timeout=timedelta(seconds=5)),
        poll_interval=timedelta(milliseconds=10),
        renewal_interval=timedelta(milliseconds=20),
        retry_delay=timedelta(seconds=1),
        batch_size=10,
    )


async def register(value, task, url):
    native = callback(task)
    native.url = url
    config = await value.register(native, creator_id="did:creator")
    runner = worker(value)
    # Registration's older task reference is acknowledged without HTTP.
    assert await runner.run_once() == 1
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
    ) == [(ACKNOWLEDGED,), (ACKNOWLEDGED,)]
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
    outbox, outbound_tls, monkeypatch
):
    value, task = outbox
    runner, config = await register(value, task, outbound_tls.url + "/callback")
    entered, release, revoked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_init = httpx.AsyncClient.__init__
    original_revoke = value.revoke

    async def response_owner(response):
        original_close = response.stream.aclose

        async def close():
            entered.set()
            await release.wait()
            await original_close()

        response.stream.aclose = close

    def client_owner(client, *args, **kwargs):
        original_init(client, *args, **kwargs)
        client.event_hooks["response"].append(response_owner)

    async def revoke(*args, **kwargs):
        result = await original_revoke(*args, **kwargs)
        revoked.set()
        return result

    monkeypatch.setattr(httpx.AsyncClient, "__init__", client_owner)
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
    outbox, outbound_tls, monkeypatch
):
    value, task = outbox
    runner, _ = await register(value, task, outbound_tls.url + "/callback")
    entered, release = asyncio.Event(), asyncio.Event()
    original_init = httpx.AsyncClient.__init__

    async def response_owner(response):
        original_close = response.stream.aclose

        async def close():
            entered.set()
            await release.wait()
            await original_close()

        response.stream.aclose = close

    def client_owner(client, *args, **kwargs):
        original_init(client, *args, **kwargs)
        client.event_hooks["response"].append(response_owner)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", client_owner)
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
        "batch_size": 10,
    }
    options[field] = bad
    with pytest.raises(ValueError):
        NativePushWorker(
            value,
            NativePushTransport(value.configurations, timeout=timedelta(seconds=5)),
            **options,
        )
