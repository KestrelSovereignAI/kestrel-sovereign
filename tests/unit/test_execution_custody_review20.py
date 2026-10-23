"""Caller cancellation must not turn a successful receipt into permanent debt."""

import asyncio
from types import SimpleNamespace

import pytest
from tests.utils.receipt_delivery import ReceiptDelivery
from tests.utils.original_failure import OriginalFailure
from kestrel_sovereign.execution_custody import _explicit_control_chain

from kestrel_sovereign.signals.dispatcher import SignalDispatcher
from tests.unit.test_execution_custody_review15 import control_error


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_joined_terminalizer_debt_tracks_child_receipt_not_caller_cancel(carrier, repeat, fails, monkeypatch):
    import contextvars
    dispatcher = SignalDispatcher.__new__(SignalDispatcher)
    dispatcher._cognition_claim_settlement = contextvars.ContextVar('test_claim_settlement', default=None)
    dispatcher._agent = SimpleNamespace(did="did:test:original")
    dispatcher._durable_delivery_owner = "original-owner"
    dispatcher._retained_cognition_control_debt = {}
    delivery = ReceiptDelivery(delivery_id="original", consumer_id="original-consumer", lease_token="original-token")
    original = control_error("unknown", carrier)
    snapshot = OriginalFailure.capture(original)
    from kestrel_sovereign import _async_ownership as ownership
    native_join, withdrawals, failures = ownership.await_owned_task, [], []

    async def observed_join(*args, **kwargs):
        result = await native_join(*args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result

    monkeypatch.setattr(ownership, "await_owned_task", observed_join)
    entered, finish = asyncio.Event(), asyncio.Event()
    children = []

    async def terminalize(*args):
        children.append(asyncio.current_task())
        entered.set()
        await finish.wait()
        if fails:
            failure = OSError("receipt unavailable")
            failures.append(failure)
            raise failure

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
        snapshot.assert_retained(caught.value)
        chain = tuple(_explicit_control_chain(caught.value))
        assert withdrawals and all(error in chain for error in (*withdrawals, *failures))
        assert all(child.done() for child in children)
        assert dispatcher._retained_cognition_control_debt == (
            {delivery.delivery_id: (delivery, original)} if fails else {}
        )
        if fails:
            held = dispatcher._retained_cognition_control_debt[delivery.delivery_id][1]
            snapshot.assert_retained(held)
            assert all(error not in tuple(_explicit_control_chain(held)) for error in withdrawals)
    finally:
        finish.set()
        await asyncio.gather(task, *children, return_exceptions=True)
