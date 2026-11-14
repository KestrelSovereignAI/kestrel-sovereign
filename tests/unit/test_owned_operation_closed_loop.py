"""Completed original operations remain observable after their source loop closes."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import OwnedOperation
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
