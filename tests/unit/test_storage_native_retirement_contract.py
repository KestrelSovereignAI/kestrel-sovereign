"""Native lifecycle authority remains distinct from privacy-aware data access."""

import pytest

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import AsyncStorage, PrivacyEnforcingStorage
from tests.utils.aiosqlite_workers import normal_close_schedule


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED])
async def test_native_storage_retirement_contract_preserves_privacy(tmp_path, mode):
    raw_storage = AsyncStorage(str(tmp_path / "retirement-contract.db"))
    storage = PrivacyEnforcingStorage(raw_storage, mode)
    assert raw_storage.native_retirement_owner is None
    assert storage.native_retirement_owner is None
    try:
        await raw_storage.initialize()
        assert raw_storage.native_retirement_owner is raw_storage
        assert storage.native_retirement_owner is storage
        assert storage.native_retirement_owner is not raw_storage.db
    finally:
        async with normal_close_schedule():
            await storage.close()
