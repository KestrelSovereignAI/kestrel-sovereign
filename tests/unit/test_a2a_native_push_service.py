"""Original resident acceptance, startup failure, and service shutdown custody."""

import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import create_owned_task
from kestrel_sovereign.a2a.push_service import NativePushService
from kestrel_sovereign.storage.db.interface import TransactionError
from tests.unit.test_a2a_native_push_outbox import (
    outbox as outbox,  # noqa: PLC0414 - pytest fixture re-export
)
from tests.unit.test_a2a_native_push_outbox import (
    push_store as push_store,  # noqa: PLC0414 - pytest fixture dependency
)
from tests.unit.test_a2a_native_push_outbox import refuse_original_database_write
from tests.unit.test_a2a_native_push_worker import register, worker
from tests.utils.outbound_tls import (
    outbound_close_barrier as outbound_close_barrier,  # noqa: PLC0414 - pytest fixture re-export
)
from tests.utils.outbound_tls import (
    outbound_tls as outbound_tls,  # noqa: PLC0414 - pytest fixture re-export
)


async def test_service_publishes_only_its_original_ready_worker_and_joins_shutdown(
    outbox,
):
    value, _ = outbox
    service = NativePushService(worker(value), task_factory=create_owned_task)
    with pytest.raises(RuntimeError, match="ready managed"):
        service.require_ready()
    await service.start()
    assert service.require_ready() is service.worker
    owner = service.task
    assert owner is not None and not owner.done()
    with pytest.raises(RuntimeError, match="fresh managed"):
        await service.start()
    await service.close()
    assert service.task is owner and owner.done()
    await service.close()
    with pytest.raises(asyncio.CancelledError):
        service.require_ready()


async def test_failed_publication_keeps_and_joins_the_original_accepted_resident(
    outbox,
):
    value, _ = outbox
    original = ValueError("original post-acceptance publication failure")
    accepted = []

    def publisher(operation, *, name, admission):
        owner = create_owned_task(operation, name=name, admission=admission)
        accepted.append(owner)
        raise original

    service = NativePushService(worker(value), task_factory=publisher)
    with pytest.raises(ValueError) as caught:
        await service.start()
    assert caught.value is original
    assert service.task is accepted[0] and service.task.done()
    assert not service.worker.ready


async def test_preacceptance_refusal_closes_the_fresh_operation_without_adopting_a_task(
    outbox,
):
    value, _ = outbox
    original = RuntimeError("original publisher refused before acceptance")
    operations = []

    def publisher(operation, *, name, admission):
        operations.append(operation)
        raise original

    service = NativePushService(worker(value), task_factory=publisher)
    with pytest.raises(RuntimeError) as caught:
        await service.start()
    assert caught.value is original
    assert (
        service.task is None
        and inspect.getcoroutinestate(operations[0]) == inspect.CORO_CLOSED
    )
    await service.close()


async def test_original_worker_storage_failure_wakes_startup_without_a_false_ready_ack(
    outbox,
):
    value, _ = outbox
    await refuse_original_database_write(
        value.backend, table="durable_signal_events", event="INSERT"
    )
    service = NativePushService(worker(value), task_factory=create_owned_task)
    with pytest.raises(TransactionError) as caught:
        await service.start()
    assert service.task.done() and service.task.exception() is caught.value
    assert not service.worker.ready
    with pytest.raises(TransactionError) as repeated:
        service.require_ready()
    assert repeated.value is caught.value


async def test_startup_cancellation_joins_the_original_resident_before_returning(
    outbox, monkeypatch
):
    value, _ = outbox
    entered, release = asyncio.Event(), asyncio.Event()
    original_forward = value.forward_pending

    async def waiting_forward(**kwargs):
        entered.set()
        await release.wait()
        return await original_forward(**kwargs)

    monkeypatch.setattr(value, "forward_pending", waiting_forward)
    service = NativePushService(worker(value), task_factory=create_owned_task)
    starting = asyncio.create_task(service.start())
    try:
        await entered.wait()
        assert not service.worker.ready
        with pytest.raises(RuntimeError, match="ready managed"):
            service.require_ready()
        starting.cancel("original startup observer cancellation")
        with pytest.raises(asyncio.CancelledError) as caught:
            await starting
        assert caught.value.args == ("original startup observer cancellation",)
        assert service.task.done() and not service.worker.ready
    finally:
        release.set()
        await asyncio.gather(starting, return_exceptions=True)
        await service.close()


async def test_waiting_for_readiness_without_an_accepted_owner_is_refused(outbox):
    value, _ = outbox
    service = NativePushService(worker(value), task_factory=create_owned_task)
    with pytest.raises(RuntimeError, match="not accepted"):
        await service.wait_ready()


async def test_invalid_publication_result_still_retires_the_original_accepted_owner(
    outbox,
):
    value, _ = outbox

    def publisher(operation, *, name, admission):
        create_owned_task(operation, name=name, admission=admission)
        # The publisher accepts the operation but fails to publish its handle.

    service = NativePushService(worker(value), task_factory=publisher)
    with pytest.raises(RuntimeError, match="original accepted owner"):
        await service.start()
    assert service.task.done() and not service.worker.ready


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
@pytest.mark.parametrize("cancel_observer", [False, True])
async def test_service_shutdown_joins_the_original_http_close_before_storage_release(
    outbox, outbound_tls, outbound_close_barrier, cancel_observer
):
    value, task = outbox
    runner, _ = await register(value, task, outbound_tls.url + "/callback")
    service = NativePushService(runner, task_factory=create_owned_task)
    closing = None
    try:
        await service.start()
        await outbound_close_barrier.entered.wait()
        closing = asyncio.create_task(service.close())
        await asyncio.sleep(0)
        assert not closing.done() and not service.task.done()
        assert await value.backend.fetch_all(
            "SELECT retired FROM a2a_push_delivery_attempts"
        ) == [(0,)]
        if cancel_observer:
            closing.cancel("original retirement observer cancellation")
            await asyncio.sleep(0)
            assert not closing.done() and not service.task.done()
        outbound_close_barrier.release.set()
        if cancel_observer:
            with pytest.raises(asyncio.CancelledError) as caught:
                await closing
            assert caught.value.args == ("original retirement observer cancellation",)
        else:
            await closing
        assert service.task.done() and not service.worker.ready
        assert await value.backend.fetch_all(
            "SELECT retired FROM a2a_push_delivery_attempts"
        ) == [(1,)]
    finally:
        outbound_close_barrier.release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await service.close()


async def test_publisher_cannot_substitute_a_borrowed_owner_for_its_accepted_worker(
    outbox,
):
    value, _ = outbox
    borrowed = create_owned_task(asyncio.Event().wait())

    def publisher(operation, *, name, admission):
        create_owned_task(operation, name=name, admission=admission)
        return borrowed

    service = NativePushService(worker(value), task_factory=publisher)
    try:
        with pytest.raises(RuntimeError, match="original accepted owner"):
            await service.start()
        assert service.task is not borrowed and service.task.done()
        assert not borrowed.done()
    finally:
        await service.close()
        borrowed.cancel()
        await asyncio.gather(borrowed, return_exceptions=True)


@pytest.mark.parametrize("bad_worker,bad_factory", [(True, False), (False, True)])
async def test_service_requires_the_canonical_worker_and_publisher(
    outbox, bad_worker, bad_factory
):
    value, _ = outbox
    with pytest.raises(TypeError):
        NativePushService(
            None if bad_worker else worker(value),
            task_factory=None if bad_factory else create_owned_task,
        )
