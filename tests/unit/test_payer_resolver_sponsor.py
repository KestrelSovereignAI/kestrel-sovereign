"""Unit tests for the SPONSOR payer kind (#1647).

A named sponsor funds a group of agents (beneficiaries). The resolver mints a
capped per-agent child against the sponsor's master — the same delegated-master
mechanism as HOST/USER, with the master read from SponsorKeyStorage. Also covers
the SponsorBeneficiaryStore roster (enroll / lookup / disenroll).
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
from kestrel_sovereign.security.user_master_key_storage import UserMasterKeyStorage
from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
from kestrel_sovereign.security.sponsor_key_storage import (
    SponsorKeyStorage,
    SponsorBeneficiaryStore,
)
from kestrel_sovereign.services.payer_resolver import FoundationPayerResolver
from kestrel_sovereign.storage.async_database import AsyncDatabase

SPONSOR_DID = "did:test:sponsor-acme-org"
OTHER_SPONSOR_DID = "did:test:sponsor-globex"


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


def _sponsor_policy(master_did: str = SPONSOR_DID, cap: Decimal = Decimal("40")) -> PayerPolicy:
    return PayerPolicy(
        llm=PayerSpec(
            vendor="openrouter",
            kind=PayerKind.SPONSOR,
            master_did=master_did,
            monthly_cap_usd=cap,
        ),
        storage=PayerSpec(vendor="lighthouse", kind=PayerKind.HOST_ENV),
        compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
    )


def _mock_provisioning_service(child_key: str = "sk-or-v1-sponsor-child"):
    mock_class = MagicMock()
    mock_instance = MagicMock()
    mock_instance.close = AsyncMock()
    key_info = MagicMock(key=child_key, key_hash=f"hash-{child_key[-8:]}", limit_usd=40.0)
    mock_instance.create_agent_key = AsyncMock(return_value=key_info)
    mock_class.return_value = mock_instance
    return mock_class, mock_instance


# =============================================================================
# SponsorKeyStorage — per-sponsor isolation
# =============================================================================

class TestSponsorKeyStorage:
    @pytest.mark.asyncio
    async def test_round_trip_and_isolation(self, db: AsyncDatabase) -> None:
        acme = SponsorKeyStorage(db, SPONSOR_DID)
        globex = SponsorKeyStorage(db, OTHER_SPONSOR_DID)
        await acme.store_key("openrouter", "sk-acme")
        await globex.store_key("openrouter", "sk-globex")
        assert await acme.get_key("openrouter") == "sk-acme"
        assert await globex.get_key("openrouter") == "sk-globex"
        assert await SponsorKeyStorage(db, "did:test:sponsor-none").has_key("openrouter") is False

    def test_requires_sponsor_did(self) -> None:
        with pytest.raises(ValueError):
            SponsorKeyStorage(MagicMock(), "")

    @pytest.mark.asyncio
    async def test_delete(self, db: AsyncDatabase) -> None:
        s = SponsorKeyStorage(db, SPONSOR_DID)
        await s.store_key("openrouter", "sk-acme")
        assert await s.delete_key("openrouter") is True
        assert await s.has_key("openrouter") is False


# =============================================================================
# SponsorBeneficiaryStore — the roster
# =============================================================================

class TestSponsorBeneficiaryStore:
    @pytest.mark.asyncio
    async def test_enroll_lookup_list(self, db: AsyncDatabase) -> None:
        roster = SponsorBeneficiaryStore(db)
        await roster.enroll(SPONSOR_DID, "did:test:agent-a")
        await roster.enroll(SPONSOR_DID, "did:test:agent-b")
        assert await roster.get_sponsor_for_agent("did:test:agent-a") == SPONSOR_DID
        assert set(await roster.list_beneficiaries(SPONSOR_DID)) == {
            "did:test:agent-a",
            "did:test:agent-b",
        }
        assert await roster.is_enrolled(SPONSOR_DID, "did:test:agent-a") is True
        assert await roster.get_sponsor_for_agent("did:test:unknown") is None

    @pytest.mark.asyncio
    async def test_one_sponsor_per_agent_reenroll_repoints(self, db: AsyncDatabase) -> None:
        roster = SponsorBeneficiaryStore(db)
        await roster.enroll(SPONSOR_DID, "did:test:agent-x")
        await roster.enroll(OTHER_SPONSOR_DID, "did:test:agent-x")  # re-point
        assert await roster.get_sponsor_for_agent("did:test:agent-x") == OTHER_SPONSOR_DID
        assert await roster.list_beneficiaries(SPONSOR_DID) == []
        assert await roster.list_beneficiaries(OTHER_SPONSOR_DID) == ["did:test:agent-x"]

    @pytest.mark.asyncio
    async def test_disenroll(self, db: AsyncDatabase) -> None:
        roster = SponsorBeneficiaryStore(db)
        await roster.enroll(SPONSOR_DID, "did:test:agent-y")
        assert await roster.disenroll("did:test:agent-y") is True
        assert await roster.get_sponsor_for_agent("did:test:agent-y") is None
        assert await roster.disenroll("did:test:agent-y") is False


# =============================================================================
# Resolver minting under a sponsor master
# =============================================================================

class TestMintUnderSponsorMaster:
    @pytest.mark.asyncio
    async def test_mint_creates_child_under_sponsor_master(self, db: AsyncDatabase) -> None:
        await SponsorKeyStorage(db, SPONSOR_DID).store_key("openrouter", "sk-or-v1-acme-master")
        agent_did = "did:test:agent-sponsor-mint"
        await _seed_agent_graph_node(db, agent_did)

        mock_class, mock_instance = _mock_provisioning_service("sk-or-v1-sponsor-minted")
        with patch(
            "kestrel_sovereign.features.llm_keys.openrouter_provisioning."
            "OpenRouterProvisioningService",
            mock_class,
        ):
            resolver = FoundationPayerResolver(_sponsor_policy(), db=db)
            result = await resolver.resolve_for(agent_did, ResourceClass.LLM)

        assert result.enabled is True
        assert mock_class.call_args.kwargs.get("management_key") == "sk-or-v1-acme-master"
        assert mock_instance.create_agent_key.await_args.kwargs["limit_usd"] == 40.0
        assert await ServiceKeyStorage(db, agent_did).get_key("openrouter") == "sk-or-v1-sponsor-minted"

    @pytest.mark.asyncio
    async def test_mint_raises_when_no_sponsor_master(self, db: AsyncDatabase) -> None:
        agent_did = "did:test:agent-no-sponsor-master"
        await _seed_agent_graph_node(db, agent_did)
        resolver = FoundationPayerResolver(_sponsor_policy(), db=db)
        with pytest.raises(PayerPolicyError) as excinfo:
            await resolver.resolve_for(agent_did, ResourceClass.LLM)
        assert "sponsor master key" in str(excinfo.value).lower()
        assert await ServiceKeyStorage(db, agent_did).has_key("openrouter") is False

    @pytest.mark.asyncio
    async def test_host_and_user_masters_not_consulted_for_sponsor_kind(self, db: AsyncDatabase) -> None:
        # Host AND user masters exist, but the policy is SPONSOR → must use the
        # (absent) sponsor master and fail, never silently fall back.
        await HostKeyStorage(db).store_key("openrouter", "sk-host-nope")
        await UserMasterKeyStorage(db, SPONSOR_DID).store_key("openrouter", "sk-user-nope")
        agent_did = "did:test:agent-sponsor-no-fallback"
        await _seed_agent_graph_node(db, agent_did)
        resolver = FoundationPayerResolver(_sponsor_policy(), db=db)
        with pytest.raises(PayerPolicyError) as excinfo:
            await resolver.resolve_for(agent_did, ResourceClass.LLM)
        assert "sponsor master key" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_mint_idempotent_when_agent_already_has_key(self, db: AsyncDatabase) -> None:
        await SponsorKeyStorage(db, SPONSOR_DID).store_key("openrouter", "sk-or-v1-acme-master")
        agent_did = "did:test:agent-sponsor-idempotent"
        await _seed_agent_graph_node(db, agent_did)
        await ServiceKeyStorage(db, agent_did).store_key("openrouter", "sk-or-v1-existing")

        mock_class, _ = _mock_provisioning_service()
        with patch(
            "kestrel_sovereign.features.llm_keys.openrouter_provisioning."
            "OpenRouterProvisioningService",
            mock_class,
        ):
            resolver = FoundationPayerResolver(_sponsor_policy(), db=db)
            await resolver.resolve_for(agent_did, ResourceClass.LLM)
        mock_class.assert_not_called()
        assert await ServiceKeyStorage(db, agent_did).get_key("openrouter") == "sk-or-v1-existing"
