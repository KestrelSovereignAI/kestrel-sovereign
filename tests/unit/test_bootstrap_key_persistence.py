"""The OpenRouter bootstrap child key must be reused across restarts.

When a route declares only ``OPENROUTER_MANAGEMENT_API_KEY``, the registry
mints a child key to back the default route. Previously that key lived only in
a process-wide in-memory cache, so every cold start minted a NEW child key and
never reused or cleaned up the old ones — an unbounded key leak on the
OpenRouter account. It is now persisted in ``host_service_keys`` (when a host
DB is available) and reused, keyed by a hash of the management key so separate
accounts stay isolated.
"""

from typing import Iterator

import pytest
import pytest_asyncio

import kestrel_sovereign.llm.provider_registry as pr
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.fixture(autouse=True)
def _kestrel_data_key(monkeypatch) -> Iterator[None]:
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-32-bytes-fixed--")
    pr._reset_bootstrap_openrouter_key_cache()
    yield
    pr._reset_bootstrap_openrouter_key_cache()


class _FakeKeyInfo:
    def __init__(self, key: str) -> None:
        self.key = key


class _FakeProvisioning:
    """Counts mints so tests can assert reuse vs re-mint."""

    calls = 0

    def __init__(self, management_key: str) -> None:
        pass

    async def create_agent_key(self, agent_name, limit_usd, limit_reset):
        _FakeProvisioning.calls += 1
        return _FakeKeyInfo(f"sk-or-minted-{_FakeProvisioning.calls}")

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _fake_provisioning(monkeypatch) -> Iterator[None]:
    _FakeProvisioning.calls = 0
    import kestrel_sovereign.features.llm_keys.openrouter_provisioning as prov
    monkeypatch.setattr(prov, "OpenRouterProvisioningService", _FakeProvisioning)
    yield


@pytest_asyncio.fixture
async def host_db(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "host.db"))
    try:
        yield db
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_bootstrap_key_persisted_and_reused_across_restart(host_db):
    mgmt = "sk-or-mgmt-abc"

    first = await pr._mint_bootstrap_openrouter_key(mgmt, host_db=host_db)
    assert _FakeProvisioning.calls == 1

    # Simulate a new process: clear the in-memory cache; the persisted row
    # survives, so no new child key is minted.
    pr._reset_bootstrap_openrouter_key_cache()
    second = await pr._mint_bootstrap_openrouter_key(mgmt, host_db=host_db)

    assert _FakeProvisioning.calls == 1, "restart must reuse the persisted key, not re-mint"
    assert second == first


@pytest.mark.asyncio
async def test_separate_accounts_get_separate_bootstrap_keys(host_db):
    a = await pr._mint_bootstrap_openrouter_key("sk-or-acct-A", host_db=host_db)
    pr._reset_bootstrap_openrouter_key_cache()
    b = await pr._mint_bootstrap_openrouter_key("sk-or-acct-B", host_db=host_db)

    assert _FakeProvisioning.calls == 2
    assert a != b


@pytest.mark.asyncio
async def test_without_host_db_still_mints_once_per_process(host_db):
    """No host store (e.g. pure-SQLite sovereign): falls back to per-process mint."""
    first = await pr._mint_bootstrap_openrouter_key("sk-or-nohost", host_db=None)
    # same process (cache warm) -> reused
    second = await pr._mint_bootstrap_openrouter_key("sk-or-nohost", host_db=None)
    assert _FakeProvisioning.calls == 1
    assert first == second

    # new process without persistence -> mints again (documents the fallback)
    pr._reset_bootstrap_openrouter_key_cache()
    await pr._mint_bootstrap_openrouter_key("sk-or-nohost", host_db=None)
    assert _FakeProvisioning.calls == 2


@pytest.mark.asyncio
async def test_provider_id_is_stable_and_account_scoped():
    a1 = pr._bootstrap_host_provider_id("sk-or-acct-A")
    a2 = pr._bootstrap_host_provider_id("sk-or-acct-A")
    b1 = pr._bootstrap_host_provider_id("sk-or-acct-B")
    assert a1 == a2 and a1 != b1
    assert a1.startswith("openrouter_bootstrap_")
