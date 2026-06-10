"""Unit tests for USER_MASTER_PROVISIONED (#1646).

Mirrors the HOST_MASTER_PROVISIONED minting tests, but the funding master
belongs to a named user (PayerSpec.master_did) and is read from
UserMasterKeyStorage rather than HostKeyStorage. Also covers
UserMasterKeyStorage's per-user isolation directly.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from kestrel_sdk.payer_policy import (
    PayerKind,
    PayerPolicy,
    PayerPolicyError,
    PayerSpec,
    ResourceClass,
)

from kestrel_sovereign.security.host_key_storage import HostKeyStorage
from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
from kestrel_sovereign.security.user_master_key_storage import (
    UserMasterKeyStorage,
)
from kestrel_sovereign.services.payer_resolver import FoundationPayerResolver
from kestrel_sovereign.storage.async_database import AsyncDatabase

USER_DID = "did:test:user-funding-alice"
OTHER_USER_DID = "did:test:user-funding-bob"


@pytest.fixture(autouse=True)
def _kestrel_data_key(monkeypatch) -> Iterator[None]:
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-32-bytes-fixed--")
    yield


@pytest_asyncio.fixture
async def db(tmp_path) -> AsyncDatabase:
    database = await AsyncDatabase.sqlite(str(tmp_path / "test.db"))
    yield database
    await database.close()


async def _seed_agent_graph_node(db: AsyncDatabase, agent_did: str) -> None:
    await db.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
        "VALUES (?, 'agent', 'test-agent', '{}')",
        (agent_did,),
    )


def _user_master_policy(
    master_did: str = USER_DID, monthly_cap: Decimal = Decimal("25")
) -> PayerPolicy:
    return PayerPolicy(
        llm=PayerSpec(
            vendor="openrouter",
            kind=PayerKind.USER_MASTER_PROVISIONED,
            master_did=master_did,
            monthly_cap_usd=monthly_cap,
        ),
        storage=PayerSpec(vendor="lighthouse", kind=PayerKind.HOST_ENV),
        compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
    )


def _mock_provisioning_service(child_key: str = "sk-or-v1-user-child"):
    mock_class = MagicMock()
    mock_instance = MagicMock()
    mock_instance.close = AsyncMock()
    key_info = MagicMock(key=child_key, key_hash=f"hash-{child_key[-8:]}", limit_usd=25.0)
    mock_instance.create_agent_key = AsyncMock(return_value=key_info)
    mock_class.return_value = mock_instance
    return mock_class, mock_instance


# =============================================================================
# UserMasterKeyStorage — per-user isolation
# =============================================================================

class TestUserMasterKeyStorage:
    @pytest.mark.asyncio
    async def test_round_trip(self, db: AsyncDatabase) -> None:
        storage = UserMasterKeyStorage(db, USER_DID)
        await storage.store_key("openrouter", "sk-or-v1-alice-master")
        assert await storage.has_key("openrouter") is True
        assert await storage.get_key("openrouter") == "sk-or-v1-alice-master"

    @pytest.mark.asyncio
    async def test_users_are_isolated(self, db: AsyncDatabase) -> None:
        alice = UserMasterKeyStorage(db, USER_DID)
        bob = UserMasterKeyStorage(db, OTHER_USER_DID)
        await alice.store_key("openrouter", "sk-alice")
        await bob.store_key("openrouter", "sk-bob")
        # Each reads only its own master; no cross-read.
        assert await alice.get_key("openrouter") == "sk-alice"
        assert await bob.get_key("openrouter") == "sk-bob"
        assert await UserMasterKeyStorage(db, "did:test:user-none").has_key(
            "openrouter"
        ) is False

    @pytest.mark.asyncio
    async def test_get_missing_raises(self, db: AsyncDatabase) -> None:
        from kestrel_sovereign.security.exceptions import KeyNotConfiguredError

        with pytest.raises(KeyNotConfiguredError):
            await UserMasterKeyStorage(db, USER_DID).get_key("openrouter")

    def test_requires_master_did(self) -> None:
        with pytest.raises(ValueError):
            UserMasterKeyStorage(MagicMock(), "")

    @pytest.mark.asyncio
    async def test_delete(self, db: AsyncDatabase) -> None:
        storage = UserMasterKeyStorage(db, USER_DID)
        await storage.store_key("openrouter", "sk-alice")
        assert await storage.delete_key("openrouter") is True
        assert await storage.has_key("openrouter") is False
        assert await storage.delete_key("openrouter") is False


# =============================================================================
# Resolver minting under a user master
# =============================================================================

class TestMintUnderUserMaster:
    @pytest.mark.asyncio
    async def test_mint_creates_child_under_user_master(self, db: AsyncDatabase) -> None:
        # User master configured; agent has no key yet.
        await UserMasterKeyStorage(db, USER_DID).store_key(
            "openrouter", "sk-or-v1-alice-master"
        )
        agent_did = "did:test:agent-user-mint"
        await _seed_agent_graph_node(db, agent_did)

        mock_class, mock_instance = _mock_provisioning_service("sk-or-v1-user-minted")
        with patch(
            "kestrel_sovereign.features.llm_keys.openrouter_provisioning."
            "OpenRouterProvisioningService",
            mock_class,
        ):
            resolver = FoundationPayerResolver(_user_master_policy(), db=db)
            result = await resolver.resolve_for(agent_did, ResourceClass.LLM)

        assert result.enabled is True
        # Provisioning used the USER's master, not a host master.
        assert mock_class.call_args.kwargs.get("management_key") == "sk-or-v1-alice-master"
        ca_kwargs = mock_instance.create_agent_key.await_args.kwargs
        assert ca_kwargs["agent_name"] == agent_did
        assert ca_kwargs["limit_usd"] == 25.0
        # Child is stored in the agent's ServiceKeyStorage.
        assert await ServiceKeyStorage(db, agent_did).get_key("openrouter") == "sk-or-v1-user-minted"

    @pytest.mark.asyncio
    async def test_mint_raises_when_no_user_master(self, db: AsyncDatabase) -> None:
        # graph row present, but the funding user has NOT provisioned a master.
        agent_did = "did:test:agent-no-user-master"
        await _seed_agent_graph_node(db, agent_did)
        resolver = FoundationPayerResolver(_user_master_policy(), db=db)
        with pytest.raises(PayerPolicyError) as excinfo:
            await resolver.resolve_for(agent_did, ResourceClass.LLM)
        assert "user master key" in str(excinfo.value).lower()
        assert await ServiceKeyStorage(db, agent_did).has_key("openrouter") is False

    @pytest.mark.asyncio
    async def test_host_master_not_consulted_for_user_kind(self, db: AsyncDatabase) -> None:
        # A host master EXISTS, but the policy is USER_MASTER → the resolver
        # must use the (absent) user master and fail, not silently use host.
        await HostKeyStorage(db).store_key("openrouter", "sk-or-v1-host-should-not-be-used")
        agent_did = "did:test:agent-user-not-host"
        await _seed_agent_graph_node(db, agent_did)
        resolver = FoundationPayerResolver(_user_master_policy(), db=db)
        with pytest.raises(PayerPolicyError) as excinfo:
            await resolver.resolve_for(agent_did, ResourceClass.LLM)
        assert "user master key" in str(excinfo.value).lower()
        assert await ServiceKeyStorage(db, agent_did).has_key("openrouter") is False

    @pytest.mark.asyncio
    async def test_mint_idempotent_when_agent_already_has_key(self, db: AsyncDatabase) -> None:
        await UserMasterKeyStorage(db, USER_DID).store_key("openrouter", "sk-or-v1-alice-master")
        agent_did = "did:test:agent-user-idempotent"
        await _seed_agent_graph_node(db, agent_did)
        await ServiceKeyStorage(db, agent_did).store_key("openrouter", "sk-or-v1-existing")

        mock_class, _ = _mock_provisioning_service()
        with patch(
            "kestrel_sovereign.features.llm_keys.openrouter_provisioning."
            "OpenRouterProvisioningService",
            mock_class,
        ):
            resolver = FoundationPayerResolver(_user_master_policy(), db=db)
            await resolver.resolve_for(agent_did, ResourceClass.LLM)
        mock_class.assert_not_called()
        assert await ServiceKeyStorage(db, agent_did).get_key("openrouter") == "sk-or-v1-existing"
