"""Unit tests for FoundationPayerResolver's HOST_MASTER_PROVISIONED
OpenRouter minting side-effect.

Phase 3c of the PayerPolicy foundation work.

When `policy.llm.kind = HOST_MASTER_PROVISIONED` and
`policy.llm.vendor = "openrouter"`, the resolver should:
1. Look up the host master OpenRouter key from HostKeyStorage.
2. Call OpenRouterProvisioningService.create_agent_key with the
   agent's DID and the spec's monthly_cap_usd.
3. Store the resulting child key in the agent's ServiceKeyStorage.
4. Be idempotent — re-resolving on an agent that already has a key
   in ServiceKeyStorage skips the OpenRouter API call entirely.

Tests use a mocked OpenRouterProvisioningService to avoid network and
the OPENROUTER_MANAGEMENT_API_KEY env-var requirement.
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
from kestrel_sovereign.services.payer_resolver import (
    FoundationPayerResolver,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.fixture(autouse=True)
def _kestrel_data_key(monkeypatch) -> Iterator[None]:
    monkeypatch.setenv(
        "KESTREL_DATA_KEY",
        "test-master-key-32-bytes-fixed--",
    )
    yield


@pytest_asyncio.fixture
async def db(tmp_path) -> AsyncDatabase:
    db_path = tmp_path / "test.db"
    database = await AsyncDatabase.sqlite(str(db_path))
    yield database
    await database.close()


async def _seed_agent_graph_node(db: AsyncDatabase, agent_did: str) -> None:
    """Create a minimal graph_nodes row for the agent.

    Tests that exercise the mint path need this row in place because
    _persist_openrouter_key_hash now refuses to mint without it (codex
    round 2 finding: retirement_service reads only graph_nodes.properties).
    Inception_service creates this row in production; tests synthesize.
    """
    await db.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
        "VALUES (?, 'agent', 'test-agent', '{}')",
        (agent_did,),
    )


def _host_master_policy(monthly_cap: Decimal = Decimal("50")) -> PayerPolicy:
    return PayerPolicy(
        llm=PayerSpec(
            vendor="openrouter",
            kind=PayerKind.HOST_MASTER_PROVISIONED,
            monthly_cap_usd=monthly_cap,
        ),
        storage=PayerSpec(vendor="lighthouse", kind=PayerKind.HOST_ENV),
        compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
    )


def _mock_provisioning_service(child_key: str = "sk-or-v1-child-test") -> MagicMock:
    """Patch target: kestrel_sovereign.services.payer_resolver's late
    import of OpenRouterProvisioningService."""
    mock_class = MagicMock()
    mock_instance = MagicMock()
    mock_instance.close = AsyncMock()
    key_info = MagicMock(
        key=child_key,
        key_hash=f"hash-{child_key[-8:]}",
        limit_usd=50.0,
    )
    mock_instance.create_agent_key = AsyncMock(return_value=key_info)
    mock_class.return_value = mock_instance
    return mock_class, mock_instance


class TestMintOpenRouterChild:
    @pytest.mark.asyncio
    async def test_mint_creates_child_when_no_existing_key(
        self, db: AsyncDatabase
    ) -> None:
        # Arrange: host master configured; agent has no key yet.
        host = HostKeyStorage(db)
        await host.store_key("openrouter", "sk-or-v1-host-master")

        agent_did = "did:test:agent-mint"
        await _seed_agent_graph_node(db, agent_did)

        mock_class, mock_instance = _mock_provisioning_service(
            child_key="sk-or-v1-minted"
        )

        with patch(
            "kestrel_sovereign.features.llm_keys.openrouter_provisioning."
            "OpenRouterProvisioningService",
            mock_class,
        ):
            resolver = FoundationPayerResolver(_host_master_policy(), db=db)
            result = await resolver.resolve_for(agent_did, ResourceClass.LLM)

        # Resolver returns enabled.
        assert result.enabled is True
        # Provisioning was called with the master_key (mock_class instantiated
        # with positional arg or management_key=master).
        mock_class.assert_called_once()
        call_kwargs = mock_class.call_args.kwargs
        assert call_kwargs.get("management_key") == "sk-or-v1-host-master"
        # create_agent_key was called with the agent's DID and the
        # policy's cap.
        mock_instance.create_agent_key.assert_awaited_once()
        ca_kwargs = mock_instance.create_agent_key.await_args.kwargs
        assert ca_kwargs["agent_name"] == agent_did
        assert ca_kwargs["limit_usd"] == 50.0
        assert ca_kwargs["limit_reset"] == "monthly"
        # Child key is now in ServiceKeyStorage.
        agent_storage = ServiceKeyStorage(db, agent_did)
        assert (await agent_storage.get_key("openrouter")) == "sk-or-v1-minted"
        # Provisioning client was closed.
        mock_instance.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mint_is_idempotent_when_agent_already_has_key(
        self, db: AsyncDatabase
    ) -> None:
        # Arrange: host master configured AND agent already has a key.
        host = HostKeyStorage(db)
        await host.store_key("openrouter", "sk-or-v1-host-master")
        agent_did = "did:test:agent-already-keyed"
        await _seed_agent_graph_node(db, agent_did)
        agent_storage = ServiceKeyStorage(db, agent_did)
        await agent_storage.store_key("openrouter", "sk-or-v1-existing")

        mock_class, mock_instance = _mock_provisioning_service()

        with patch(
            "kestrel_sovereign.features.llm_keys.openrouter_provisioning."
            "OpenRouterProvisioningService",
            mock_class,
        ):
            resolver = FoundationPayerResolver(_host_master_policy(), db=db)
            await resolver.resolve_for(agent_did, ResourceClass.LLM)

        # No OpenRouter API call. The agent's existing key is preserved.
        mock_class.assert_not_called()
        assert (await agent_storage.get_key("openrouter")) == "sk-or-v1-existing"

    @pytest.mark.asyncio
    async def test_mint_raises_when_no_host_master(
        self, db: AsyncDatabase
    ) -> None:
        # Arrange: graph_nodes row exists but NO host master configured.
        # The resolver must refuse rather than silently falling through
        # to env-var or shared key.
        agent_did = "did:test:agent-no-master"
        await _seed_agent_graph_node(db, agent_did)
        resolver = FoundationPayerResolver(_host_master_policy(), db=db)
        with pytest.raises(PayerPolicyError) as excinfo:
            await resolver.resolve_for(agent_did, ResourceClass.LLM)
        assert "host master key" in str(excinfo.value).lower()
        # No child key was stored.
        agent_storage = ServiceKeyStorage(db, agent_did)
        assert (await agent_storage.has_key("openrouter")) is False

    @pytest.mark.asyncio
    async def test_mint_uses_default_cap_when_spec_has_none(
        self, db: AsyncDatabase
    ) -> None:
        # When monthly_cap_usd is unspecified, mint with $100/mo default
        # (mirrors the deprecated provision_agent_openrouter.py default).
        host = HostKeyStorage(db)
        await host.store_key("openrouter", "sk-or-v1-host-master")
        agent_did = "did:test:agent-default-cap"
        await _seed_agent_graph_node(db, agent_did)

        # Build policy without monthly_cap_usd
        policy = PayerPolicy(
            llm=PayerSpec(
                vendor="openrouter",
                kind=PayerKind.HOST_MASTER_PROVISIONED,
                # no monthly_cap_usd
            ),
            storage=PayerSpec(vendor="lighthouse", kind=PayerKind.HOST_ENV),
            compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        )

        mock_class, mock_instance = _mock_provisioning_service()
        with patch(
            "kestrel_sovereign.features.llm_keys.openrouter_provisioning."
            "OpenRouterProvisioningService",
            mock_class,
        ):
            resolver = FoundationPayerResolver(policy, db=db)
            await resolver.resolve_for(agent_did, ResourceClass.LLM)

        mock_instance.create_agent_key.assert_awaited_once()
        assert (
            mock_instance.create_agent_key.await_args.kwargs["limit_usd"] == 100.0
        )

    @pytest.mark.asyncio
    async def test_mint_persists_key_hash_to_graph_nodes(
        self, db: AsyncDatabase
    ) -> None:
        """Phase 3c codex round 1 finding: retirement_service reads
        openrouter_key_hash from graph_nodes.properties. Without this
        persistence, resolver-minted keys leak — ServiceKeyStorage
        forgets them on retirement, but OpenRouter still bills them."""
        import json

        host = HostKeyStorage(db)
        await host.store_key("openrouter", "sk-or-v1-host-master")
        agent_did = "did:test:agent-hash-persist"
        await _seed_agent_graph_node(db, agent_did)

        mock_class, mock_instance = _mock_provisioning_service(
            child_key="sk-or-v1-minted-hashtest",
        )
        with patch(
            "kestrel_sovereign.features.llm_keys.openrouter_provisioning."
            "OpenRouterProvisioningService",
            mock_class,
        ):
            resolver = FoundationPayerResolver(_host_master_policy(), db=db)
            await resolver.resolve_for(agent_did, ResourceClass.LLM)

        # Verify graph_nodes.properties.openrouter_key_hash now equals
        # the mocked key_info.key_hash.
        rows = await db.fetchall(
            "SELECT properties FROM graph_nodes WHERE node_id = ?",
            (agent_did,),
        )
        assert rows
        properties = json.loads(rows[0][0])
        # _mock_provisioning_service builds key_hash as f"hash-{child_key[-8:]}"
        assert properties["openrouter_key_hash"] == "hash-hashtest"

    @pytest.mark.asyncio
    async def test_mint_raises_when_no_graph_nodes_row(
        self, db: AsyncDatabase
    ) -> None:
        """Codex Phase 3c round 2/3: refuse to mint if no graph_nodes
        row exists, AND fail BEFORE any side effects so retry doesn't
        end up with a working local key but no retirement-visible hash.

        retirement_service.get_agent_info() reads only from
        graph_nodes.properties — without a row, the remote OpenRouter
        child key would leak. inception_service creates the row before
        agent init reaches the resolver, so production paths never
        hit this; tests that want to exercise mint must seed the row.
        """
        host = HostKeyStorage(db)
        await host.store_key("openrouter", "sk-or-v1-host-master")
        agent_did = "did:test:agent-no-graph-row"

        # Deliberately do NOT create a graph_nodes row.

        mock_class, mock_instance = _mock_provisioning_service(
            child_key="sk-or-v1-minted-no-row",
        )
        with patch(
            "kestrel_sovereign.features.llm_keys.openrouter_provisioning."
            "OpenRouterProvisioningService",
            mock_class,
        ):
            resolver = FoundationPayerResolver(_host_master_policy(), db=db)
            with pytest.raises(PayerPolicyError) as excinfo:
                await resolver.resolve_for(agent_did, ResourceClass.LLM)

        assert "graph_nodes" in str(excinfo.value).lower()
        # CRITICAL: the OpenRouter API was NOT called and no local key
        # was stored. Pre-flight short-circuited before any side
        # effects, so a retry has the same starting state and can't
        # land in an inconsistent partially-provisioned state.
        assert mock_instance.create_agent_key.await_count == 0
        # ServiceKeyStorage has no key — confirms no local side effect.
        agent_storage = ServiceKeyStorage(db, agent_did)
        assert (await agent_storage.has_key("openrouter")) is False

    @pytest.mark.asyncio
    async def test_concurrent_mints_for_same_agent_serialize(
        self, db: AsyncDatabase
    ) -> None:
        """Phase 3c codex round 1 finding: two concurrent agent-init
        paths for the same DID would both pass has_key() before either
        store_key() landed, both create remote OpenRouter keys, second
        store_key replaces the local row → first remote key orphaned.
        The per-agent asyncio.Lock serializes first-time mints so the
        second waiter sees the stored key and skips its API call.
        """
        import asyncio as _asyncio

        host = HostKeyStorage(db)
        await host.store_key("openrouter", "sk-or-v1-host-master")
        agent_did = "did:test:agent-concurrent"
        await _seed_agent_graph_node(db, agent_did)

        mock_class, mock_instance = _mock_provisioning_service(
            child_key="sk-or-v1-only-once"
        )

        # Codex Phase 3c round 2 finding: locks must be class-level
        # (not instance-level) because kestrel_agent.py constructs a
        # fresh resolver per init. Use TWO distinct resolver instances
        # for the same agent_did to exercise the cross-instance lock.
        with patch(
            "kestrel_sovereign.features.llm_keys.openrouter_provisioning."
            "OpenRouterProvisioningService",
            mock_class,
        ):
            resolver_a = FoundationPayerResolver(_host_master_policy(), db=db)
            resolver_b = FoundationPayerResolver(_host_master_policy(), db=db)
            assert resolver_a is not resolver_b  # sanity
            # Fire concurrent resolves on two DIFFERENT resolver instances.
            results = await _asyncio.gather(
                resolver_a.resolve_for(agent_did, ResourceClass.LLM),
                resolver_b.resolve_for(agent_did, ResourceClass.LLM),
            )

        # Both calls return enabled.
        assert all(r.enabled for r in results)
        # OpenRouter API was called exactly ONCE despite two concurrent
        # resolves on different resolver instances — the class-level
        # lock prevented the second from minting.
        assert mock_instance.create_agent_key.await_count == 1

    @pytest.mark.asyncio
    async def test_mint_revokes_remote_key_when_graph_row_vanishes_mid_mint(
        self, db: AsyncDatabase
    ) -> None:
        """Codex Phase 3c round 4 finding: the vanishingly-rare race
        where graph_nodes is deleted between pre-flight and persist
        used to log-and-return after the remote/local mint had already
        happened. ServiceKeyStorage.has_key() became True; on retry,
        the resolver skipped mint AND skipped persist — recreating
        the leak.

        Corrected behavior: persist FIRST (before local store). If the
        row vanished, revoke the remote child key via
        OpenRouterProvisioningService.delete_key, raise
        PayerPolicyError, and DO NOT touch ServiceKeyStorage. Retry
        sees a clean slate.

        Simulated by: pre-flight succeeds (row exists), then the test
        deletes the row mid-mint via a side-effect on create_agent_key.
        """
        host = HostKeyStorage(db)
        await host.store_key("openrouter", "sk-or-v1-host-master")
        agent_did = "did:test:agent-vanish"
        await _seed_agent_graph_node(db, agent_did)

        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.close = AsyncMock()
        # Capture revoke calls.
        mock_instance.delete_key = AsyncMock(return_value=True)

        async def _create_then_delete_row(**_kwargs):
            # Mid-mint: simulate the row vanishing (e.g. concurrent
            # agent retirement during init).
            await db.execute(
                "DELETE FROM graph_nodes WHERE node_id = ?",
                (agent_did,),
            )
            return MagicMock(
                key="sk-or-v1-vanished-mid-mint",
                key_hash="hash-vanish-1234",
                limit_usd=50.0,
            )

        mock_instance.create_agent_key = AsyncMock(
            side_effect=_create_then_delete_row
        )
        mock_class.return_value = mock_instance

        with patch(
            "kestrel_sovereign.features.llm_keys.openrouter_provisioning."
            "OpenRouterProvisioningService",
            mock_class,
        ):
            resolver = FoundationPayerResolver(_host_master_policy(), db=db)
            with pytest.raises(PayerPolicyError) as excinfo:
                await resolver.resolve_for(agent_did, ResourceClass.LLM)

        assert "vanished" in str(excinfo.value).lower() or \
               "disappeared" in str(excinfo.value).lower()

        # Critical assertions:
        # 1. The remote key was minted.
        assert mock_instance.create_agent_key.await_count == 1
        # 2. The remote key was REVOKED (delete_key called with the hash).
        mock_instance.delete_key.assert_awaited_once_with("hash-vanish-1234")
        # 3. NO local key was stored (rollback before store_key).
        agent_storage = ServiceKeyStorage(db, agent_did)
        assert (await agent_storage.has_key("openrouter")) is False

    @pytest.mark.asyncio
    async def test_mint_skipped_when_no_db(self) -> None:
        # Defensive: no db means no ServiceKeyStorage to write into.
        # Resolver should warn and skip rather than crash.
        mock_class, mock_instance = _mock_provisioning_service()
        with patch(
            "kestrel_sovereign.features.llm_keys.openrouter_provisioning."
            "OpenRouterProvisioningService",
            mock_class,
        ):
            resolver = FoundationPayerResolver(_host_master_policy(), db=None)
            result = await resolver.resolve_for(
                "did:test:agent-no-db", ResourceClass.LLM
            )
        # No mint attempted; resolver still returns enabled.
        mock_class.assert_not_called()
        assert result.enabled is True
