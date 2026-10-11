"""Renewal retirement must join its original commit despite caller cancellation."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign.signals.dispatcher import SignalDispatcher
from tests.unit.test_execution_custody_review15 import control_error


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("phase", ["body", "closer"])
async def test_cancelled_renewal_closer_joins_late_original_control(control, carrier, repeat, phase):
    entered, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = control_error(control, carrier)
    owned = []

    async def renew(**kwargs):
        owned.append(asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            draining.set()
            await release.wait()
            raise original

    dispatcher = SignalDispatcher.__new__(SignalDispatcher)
    dispatcher.renew_durable_delivery_lease = renew
    delivery = SimpleNamespace(delivery_id="original-delivery", consumer_id="original-consumer",
                               lease_token="original-token",
                               lease_expires_at=datetime.now(timezone.utc)+timedelta(seconds=0.03))

    async def close():
        async with dispatcher._renew_durable_cognition_lease(delivery):
            await entered.wait()
            if phase == "body":
                await asyncio.Event().wait()

    task = asyncio.create_task(close())
    try:
        if phase == "body":
            await entered.wait()
            task.cancel()
        await draining.wait()
        task.cancel()
        await asyncio.sleep(0)
        if repeat:
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done(), "renewal commit escaped its cancelled closer"
        release.set()
        with pytest.raises(BaseException) as caught:
            await task
        assert caught.value is original
        assert all(child.done() for child in owned)
    finally:
        release.set()
        await asyncio.gather(task, *owned, return_exceptions=True)
