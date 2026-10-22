"""Real callback retries cannot monopolize a bounded native delivery batch."""

import asyncio
from datetime import timedelta
from uuid import uuid4

import pytest
from a2a.types import TaskPushNotificationConfig

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.signals.durable import ACKNOWLEDGED
from tests.unit.test_a2a_native_push_worker import worker
from tests.unit.test_a2a_native_task_manager import manager as manager  # noqa: PLC0414
from tests.unit.test_a2a_native_task_manager import request
from tests.utils.native_push import managed_push_service
from tests.utils.outbound_tls import hold_outbound_response_close
from tests.utils.outbound_tls import outbound_tls as outbound_tls  # noqa: PLC0414


async def register(manager, outbox, url, *, creator):
    task = await manager.create_task(request(uuid4().hex), creator_agent_id=creator)
    await outbox.register(
        TaskPushNotificationConfig(
            tenant=manager.task_store.scope.tenant,
            task_id=task.id,
            id="callback",
            url=url,
        ),
        creator_id=creator,
    )
    return task


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 17}], indirect=True)
async def test_one_creators_retries_do_not_starve_older_waiting_deliveries(
    manager, monkeypatch, outbound_tls
):
    async with managed_push_service(manager, monkeypatch, pause_delivery=True) as service:
        outbox = service.worker.outbox
        for _ in range(6):
            await register(manager, outbox, outbound_tls.url + "/drip-headers", creator="did:slow")
        healthy = {
            (await register(manager, outbox, outbound_tls.url + "/healthy", creator=creator)).id
            for creator in ("did:healthy-b", "did:healthy-c")
        }
        await outbox.forward_pending(limit=1000)
        runner = worker(outbox)
        runner.batch_size = 3
        runner.transport.timeout = 0.25
        runner.retry_delay = timedelta(milliseconds=10)
        counts = []
        for _ in range(3):
            counts.append(await runner.run_once())
            assert counts[-1] <= 3, "the worker admitted more than its finite batch"
        received = [body["task"]["id"] for path, _, body in outbound_tls.posts if path == "/healthy"]
        assert set(received) == healthy and len(received) == 2, (
            "retrying callbacks monopolized the batch ahead of healthy waiting creators"
        )
        assert all(count <= 3 for count in counts)
        assert len(outbound_tls.posts) <= 9
        assert not runner._active


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 4}], indirect=True)
@pytest.mark.parametrize("stop_batch", [False, True])
async def test_a_healthy_callback_finishes_while_another_original_http_owner_is_open(
    manager, monkeypatch, outbound_tls, stop_batch
):
    async with managed_push_service(manager, monkeypatch, pause_delivery=True) as service:
        outbox = service.worker.outbox
        # The old serial worker orders by task ID. Make the physically held
        # owner first without replacing the real task-ID producer or transport.
        contracts = sorted([
            (await manager.create_task(request(uuid4().hex), creator_agent_id=creator), creator)
            for creator in ("did:one", "did:two")
        ], key=lambda contract: contract[0].id)
        for (task, creator), path in zip(contracts, ("/held-close", "/healthy"), strict=True):
            await outbox.register(
                TaskPushNotificationConfig(
                    tenant=manager.task_store.scope.tenant,
                    task_id=task.id, id="callback", url=outbound_tls.url + path,
                ), creator_id=creator,
            )
        subscriptions = await outbox.subscriptions()
        healthy_consumer = next(s.consumer_id for s in subscriptions if s.task_id == contracts[1][0].id)
        barrier = hold_outbound_response_close(
            monkeypatch, response_filter=lambda response: response.url.path == "/held-close"
        )
        runner = worker(outbox)

        async def observe_healthy_ack():
            while await outbox.backend.fetch_val(
                "SELECT status FROM durable_signal_deliveries WHERE consumer_id = ?",
                (healthy_consumer,),
            ) != ACKNOWLEDGED:
                await asyncio.sleep(0.01)

        sending = asyncio.create_task(runner.run_once())
        observed = asyncio.create_task(observe_healthy_ack())
        try:
            await asyncio.wait_for(barrier.entered.wait(), 5)
            done, _ = await asyncio.wait((observed,), timeout=2)
            assert done and not sending.done(), (
                "a healthy callback waited for another creator's original HTTP close"
            )
            observed.result()
            if stop_batch:
                sending.cancel()
                ended, _ = await asyncio.wait((sending,), timeout=0.05)
                assert not ended, "batch Stop abandoned an original HTTP close"
        finally:
            barrier.release.set()
            observed.cancel()
            await asyncio.gather(observed, return_exceptions=True)
            outcome = await await_owned_task(sending)
        if stop_batch:
            assert any(isinstance(error, asyncio.CancelledError) for error in _explicit_control_chain(outcome.error))
            assert not is_execution_control_error(outcome.error)
        else:
            assert outcome.error is None and outcome.result == 2
        assert len(outbound_tls.posts) == 2 and not runner._active
        assert await outbox.backend.fetch_all(
            "SELECT retired FROM a2a_push_delivery_attempts"
        ) == [(1,), (1,)]
