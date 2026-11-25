"""Native tail diagnostics cannot take ownership or expose frame values."""
import asyncio

import pytest

from kestrel_sovereign.async_ownership import OwnedOperation, await_owned_task
from tests.shared.native_task_diagnostics import original_task_tails


@pytest.mark.asyncio
async def test_original_tail_diagnostics_follow_only_explicit_owners_without_mutation(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    private_value = "DO-NOT-LOG-ORIGINAL-FRAME-ARGUMENT"

    async def physical_tail():
        assert private_value
        entered.set()
        await release.wait()

    original = OwnedOperation(physical_tail(), name="diagnostic-original-native-tail")

    async def observer():
        return await original.observe(cancel_on_caller_cancellation=False)

    caller = asyncio.create_task(observer(), name="diagnostic-original-caller")
    try:
        await entered.wait()
        await asyncio.sleep(0)
        counts = caller.cancelling(), original.task.cancelling()
        def refuse_scan(*args, **kwargs):
            pytest.fail("diagnostics scanned or allocated unrelated Tasks")
        with monkeypatch.context() as scoped:
            scoped.setattr(asyncio, "all_tasks", refuse_scan)
            scoped.setattr(asyncio.get_running_loop(), "create_task", refuse_scan)
            detail = "\n".join(original_task_tails((caller,)))
        assert "diagnostic-original-caller" in detail
        assert "diagnostic-original-native-tail" in detail
        assert "physical_tail" in detail
        assert private_value not in detail
        assert (caller.cancelling(), original.task.cancelling()) == counts
        assert not caller.done() and not original.task.done()
    finally:
        release.set()
        await await_owned_task(caller)


@pytest.mark.asyncio
async def test_original_tail_diagnostics_bound_explicit_native_inventory():
    release = asyncio.Event()
    tasks = [asyncio.create_task(release.wait(), name=f"bounded-native-tail-{index}") for index in range(20)]
    try:
        await asyncio.sleep(0)
        detail = original_task_tails(tasks)
        assert len([line for line in detail if line.startswith("original-task ")]) == 16
        assert len(detail) <= 16 * 49
        assert all(not task.done() and not task.cancelling() for task in tasks)
    finally:
        release.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_original_tail_diagnostics_follow_original_source_admissions(monkeypatch):
    from kestrel_sovereign._async_ownership import OwnedAsyncIterator
    entered, release = asyncio.Event(), asyncio.Event()

    async def original_source():
        entered.set()
        await release.wait()
        yield "original item"

    source = OwnedAsyncIterator(lambda: original_source(), operation="diagnostic-original-source")
    consumer = asyncio.create_task(anext(source), name="original-source-consumer")
    try:
        await entered.wait()
        def refuse(*args, **kwargs):
            pytest.fail("diagnostic scanned, allocated or consumed an original")
        with monkeypatch.context() as scoped:
            scoped.setattr(asyncio, "all_tasks", refuse)
            scoped.setattr(asyncio.get_running_loop(), "create_task", refuse)
            detail = "\n".join(original_task_tails((consumer,)))
        assert "owned_async_iterator:diagnostic-original-source" in detail
        assert not consumer.done() and not source._source_admission.producer.task.done()
    finally:
        release.set()
        await consumer
        await source.aclose()


@pytest.mark.asyncio
async def test_diagnostic_timer_uses_real_time_and_physically_joins(caplog):
    import logging
    import threading
    from tests.shared.native_task_diagnostics import original_task_tail_diagnostics
    from tests.utils.virtual_clock import pinned_loop_clock, wait_for_thread_event

    emitted = threading.Event()
    logger = logging.getLogger("tests.shared.native_task_diagnostics")

    class ObserveOriginalReport(logging.Filter):
        def filter(self, record):
            if record.msg.startswith("Original native Task tails before test hang limit"):
                emitted.set()
            return True

    observer = ObserveOriginalReport()
    logger.addFilter(observer)
    current = asyncio.current_task()
    count = current.cancelling()
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            with pinned_loop_clock() as clock:
                original_time = clock.time()
                with original_task_tail_diagnostics((current,), seconds=0) as timer:
                    await wait_for_thread_event(emitted, step="original Task diagnostic callback")
                assert clock.time() == original_time
            assert not timer.is_alive() and current.cancelling() == count
            assert any("original-task" in record.getMessage() for record in caplog.records)
    finally:
        logger.removeFilter(observer)
    caplog.clear()
    with original_task_tail_diagnostics((current,)) as timer:
        pass
    await asyncio.sleep(0)
    assert not timer.is_alive()
    assert not [record for record in caplog.records if record.msg.startswith("Original native Task tails before test hang limit")]
