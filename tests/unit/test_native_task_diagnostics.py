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
