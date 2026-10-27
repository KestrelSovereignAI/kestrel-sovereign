"""The final awaited Hold lookup cannot lend retired original Work to a handler."""
import asyncio
import inspect
from types import SimpleNamespace

import pytest
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_runtime, is_execution_control_error
from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork
from tests.unit.test_signals_dispatcher import (
    db_path as db_path, dispatcher_components as dispatcher_components, _action_reg, _signal,
)


async def test_original_work_revoked_at_last_hold_await_never_enters_handler(dispatcher_components, monkeypatch):
    c = dispatcher_components
    work = ExecutionCustody(_OriginalWork())
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []
    async def handler(payload):
        effects.append(payload)
        return {"ok": True}
    source = "original.physical.entry"
    c.registry.register(_action_reg(source, handler=handler))
    original = c.dispatcher._held_signal_result
    async def delayed(signal, *args, **kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        value = await original(signal, *args, **kwargs)
        if signal.source == source and caller == "_route_under_locks":
            assert value is None
            entered.set()
            await release.wait()
        return value
    monkeypatch.setattr(c.dispatcher, "_held_signal_result", delayed)
    with bind_execution_runtime(SimpleNamespace(_execution_custody=work)):
        task = asyncio.create_task(c.dispatcher.dispatch_signal(_signal(source)))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        work.revoke("original Work retired after last awaited Hold read")
    finally:
        release.set()
        outcome = await await_owned_task(task)
    assert effects == []
    assert outcome.error is not None and is_execution_control_error(outcome.error)
