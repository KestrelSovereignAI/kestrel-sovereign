"""Completed original operations remain observable after their source loop closes."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import OwnedOperation, await_owned_task
from kestrel_sovereign.execution_custody import ExecutionAuthorityError


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["result", "authority", "group", "timeout", "cancelled"])
async def test_original_completed_operation_survives_source_loop_closure(native_tool_effect_agent, completion):
    expected = object()
    error = None
    if completion in {"authority", "group", "timeout"}:
        agent = await native_tool_effect_agent(did="did:test:closed-source-loop")
        try:
            agent._get_privacy_transition_lock().require_native_scope(None)
        except ExecutionAuthorityError as authority:
            error = authority
        if completion == "group":
            error = ExceptionGroup("Original source authority", [error])
        elif completion == "timeout":
            try:
                raise TimeoutError("Original authority carrier") from error
            except TimeoutError as carried:
                error = carried

    async def physical_operation():
        if completion == "cancelled":
            await asyncio.Event().wait()
        await asyncio.sleep(0)
        if error is not None:
            raise error
        return expected

    def produce_and_close():
        async def produce():
            operation = OwnedOperation(physical_operation(), retain_originals=True)
            if completion == "cancelled":
                await asyncio.sleep(0)
                operation.task.cancel()
            first = await operation.observe(retain_cancellations=True)
            return operation, first
        # A real separate thread/loop produces the original result, then
        # Runner physically closes that loop before the second observation.
        with asyncio.Runner() as runner:
            return runner.run(produce())

    operation, first = await asyncio.to_thread(produce_and_close)
    assert operation.task.done() and operation.task.get_loop().is_closed()
    final = await operation.observe(retain_cancellations=True)
    assert final.error is first.error, "Closed-loop observation replaced the original terminal outcome"
    assert final.result is first.result
    if completion == "result":
        assert final.error is None and final.result is expected
    elif completion != "cancelled":
        assert final.error is error
    else:
        assert isinstance(final.error, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_pending_original_cancellation_keeps_actual_observer_removal_failure():
    from kestrel_sovereign.execution_custody import _explicit_control_chain

    published, release = asyncio.Event(), asyncio.Event()
    cancellation = asyncio.CancelledError("Original pending source cancellation")
    refusal = RuntimeError("Original pending completion observer removal refused")

    class RefuseRemoval(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            result = super().add_done_callback(callback, *args, **kwargs)
            if getattr(callback, "__name__", "") == "mark_complete":
                published.set()
            return result

        def remove_done_callback(self, callback):
            result = super().remove_done_callback(callback)
            if getattr(callback, "__name__", "") == "mark_complete":
                raise refusal
            return result

    async def physical_operation():
        await release.wait()
        raise cancellation

    source = RefuseRemoval(physical_operation())
    observer = asyncio.create_task(await_owned_task(source))
    try:
        await published.wait()
        assert not source.done(), "Pending-source control did not publish an actual observer"
        release.set()
        outcome = await observer
        assert source.done() and source.cancelled()
        errors = tuple(_explicit_control_chain(outcome.error))
        assert cancellation in errors, "Observer cleanup erased the original source cancellation"
        assert refusal in errors, "Pending-source observation discarded its actual removal failure"
    finally:
        release.set()
        await asyncio.gather(observer, source, return_exceptions=True)
