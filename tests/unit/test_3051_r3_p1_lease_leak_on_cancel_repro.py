"""Round-3 P1 reproduction (#3306): a cancelled projection leaks its privacy lease.

``project_strategy_ledger_assertions`` takes the privacy-transition lease
BEFORE it queues on the per-ledger projection lock, but the ``try/finally``
that releases the lease is written INSIDE the ``async with`` for that lock.
A pass cancelled while waiting for the lock therefore never reaches the
``finally``: the lease counter stays positive for the life of the process and
every subsequent privacy transition is refused with
``PrivacyViolationError``. That is a permanent denial of a sovereign
capability (Amendment IV) caused by an ordinary cancellation.

The fix is to hold the lease release in a ``finally`` that wraps the lock
acquisition, so the queued-and-cancelled path unwinds through it.
"""

from __future__ import annotations

import asyncio

import pytest

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.privacy_wrapper import PrivacyViolationError
from tests.unit import test_strategic_memory_ledger_assertions as base

# Fixtures are re-exported by binding, not by `from ... import`: a test
# signature that takes `governed`/`ledger` would otherwise read as F811
# redefinition of the imported names.
every = base.every
seed = base.seed
tenant_identity = base.tenant_identity
governed = base.governed
ledger = base.ledger


@pytest.mark.asyncio
async def test_cancelled_while_queued_releases_the_privacy_lease(governed, ledger):
    """Cancel pass B while it waits behind pass A, then transition privacy.

    Pass A holds the per-ledger lock. Pass B takes the lease and blocks on the
    lock. B is cancelled there. Once A finishes, the lease counter must be
    back to zero -- observable only through ``set_privacy_mode``, which is the
    capability the leak actually denies.
    """
    storage, _raw, _tenant = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    release_a = asyncio.Event()
    a_inside = asyncio.Event()
    original_read = storage._read_ledger_assertions

    async def slow_read(plan, report):
        # Pass A parks here, holding the per-ledger lock, until we let it go.
        a_inside.set()
        await release_a.wait()
        return await original_read(plan, report)

    storage._read_ledger_assertions = slow_read
    try:
        pass_a = asyncio.create_task(
            storage.project_strategy_ledger_assertions(ledger)
        )
        await asyncio.wait_for(a_inside.wait(), timeout=5)

        # B takes the lease, then queues on the lock A holds.
        pass_b = asyncio.create_task(
            storage.project_strategy_ledger_assertions(ledger)
        )
        for _ in range(50):
            await asyncio.sleep(0)
        pass_b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pass_b
    finally:
        storage._read_ledger_assertions = original_read
        release_a.set()
        await pass_a

    # The only externally visible consequence of the leak.
    try:
        storage.set_privacy_mode(PrivacyMode.ISOLATED)
    except PrivacyViolationError as error:  # pragma: no cover - the defect
        pytest.fail(
            "cancelled projection leaked its privacy-transition lease; every "
            f"later privacy transition is refused: {error}"
        )
    storage.set_privacy_mode(PrivacyMode.NORMAL)
