"""One deadline algorithm for scheduler and downstream original owners."""

import asyncio
import time

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ConfirmedLeaseDeadline,
    ExecutionAuthorityError,
)
from kestrel_sovereign.features.scheduler.runner import _LeaseRenewalState


@pytest.mark.parametrize("owner_type", [ConfirmedLeaseDeadline, _LeaseRenewalState])
@pytest.mark.parametrize("deadline", [float("inf"), float("nan"), -1.0])
def test_invalid_or_expired_interval_irrevocably_denies(owner_type, deadline):
    owner = owner_type()
    assert not owner.confirm(deadline)
    with pytest.raises(ExecutionAuthorityError):
        owner.require_work()
    assert not owner.confirm(time.monotonic() + 60)


@pytest.mark.parametrize("owner_type", [ConfirmedLeaseDeadline, _LeaseRenewalState])
def test_sync_guard_denies_even_without_monitor_and_cannot_revive(owner_type, monkeypatch):
    clock = [100.0]
    with monkeypatch.context() as patch:
        patch.setattr(time, "monotonic", lambda: clock[0])
        owner = owner_type()
        assert owner.confirm(110.0)
        owner.require_work()
        clock[0] = 110.0
        with pytest.raises(ExecutionAuthorityError):
            owner.require_work()
        assert not owner.confirm(120.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_type", [ConfirmedLeaseDeadline, _LeaseRenewalState])
async def test_monitor_uses_latest_confirmed_interval_and_expires_without_renewal(owner_type):
    owner = owner_type()
    monitor = create_owned_task(owner.monitor())
    try:
        await asyncio.sleep(0)
        assert owner.confirm(time.monotonic() + 0.05)
        await asyncio.sleep(0)
        assert owner.confirm(time.monotonic() + 30)
        await asyncio.sleep(0.08)
        assert not monitor.done() and owner.is_live()
        owner.mark_lost()
        outcome = await asyncio.wait_for(await_owned_task(monitor), 2)
        assert outcome.error is None
        assert not owner.confirm(time.monotonic() + 60)
    finally:
        owner.mark_lost()
        await await_owned_task(monitor)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_type", [ConfirmedLeaseDeadline, _LeaseRenewalState])
async def test_monitor_expires_while_original_renewal_is_stalled(owner_type):
    owner = owner_type()
    assert owner.confirm(time.monotonic() + 0.02)
    monitor = create_owned_task(owner.monitor())
    try:
        outcome = await asyncio.wait_for(await_owned_task(monitor), 2)
        assert outcome.error is None
        with pytest.raises(ExecutionAuthorityError):
            owner.require_work()
    finally:
        owner.mark_lost()
        await await_owned_task(monitor)
