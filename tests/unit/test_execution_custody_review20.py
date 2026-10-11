"""Caller cancellation must not turn a successful receipt into permanent debt."""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign.signals.dispatcher import SignalDispatcher
from tests.unit.test_execution_custody_review15 import control_error


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_joined_terminalizer_debt_tracks_child_receipt_not_caller_cancel(carrier, repeat, fails):
    dispatcher = SignalDispatcher.__new__(SignalDispatcher)
    dispatcher._retained_cognition_control_debt = {}
    delivery = SimpleNamespace(delivery_id="original")
    original = control_error("unknown", carrier)
    entered, finish = asyncio.Event(), asyncio.Event()
    children = []

    async def terminalize(*args):
        children.append(asyncio.current_task())
        entered.set()
        await finish.wait()
        if fails:
            raise OSError("receipt unavailable")

    dispatcher._apply_cognition_control_terminal = terminalize
    task = asyncio.create_task(dispatcher._terminalize_failed_cognition(delivery, original))
    try:
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        if repeat:
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(BaseException) as caught:
            await task
        assert caught.value is original
        assert all(child.done() for child in children)
        assert dispatcher._retained_cognition_control_debt == (
            {delivery.delivery_id: (delivery, original)} if fails else {}
        )
    finally:
        finish.set()
        await asyncio.gather(task, *children, return_exceptions=True)
