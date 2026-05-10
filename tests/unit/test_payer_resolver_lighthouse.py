"""Unit tests for Lighthouse SELF_WALLET PayerPolicy resolution."""
from __future__ import annotations

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

from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
from kestrel_sovereign.services.payer_resolver import FoundationPayerResolver
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
    database = await AsyncDatabase.sqlite(str(tmp_path / "test.db"))
    yield database
    await database.close()


def _self_wallet_storage_policy() -> PayerPolicy:
    return PayerPolicy(
        llm=PayerSpec(vendor="openrouter", kind=PayerKind.HOST_ENV),
        storage=PayerSpec(vendor="lighthouse", kind=PayerKind.SELF_WALLET),
        compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
    )


def _mock_lighthouse_client(api_key: str = "lh-agent-key") -> tuple[MagicMock, MagicMock]:
    mock_class = MagicMock()
    mock_instance = MagicMock()
    mock_instance.get_auth_message = AsyncMock(return_value="Sign this Lighthouse challenge")
    mock_instance.create_api_key = AsyncMock(return_value=api_key)
    mock_instance.close = AsyncMock()
    mock_class.return_value = mock_instance
    return mock_class, mock_instance


class TestLighthouseSelfWalletMint:
    @pytest.mark.asyncio
    async def test_mints_key_with_wallet_signature(
        self, db: AsyncDatabase, monkeypatch
    ) -> None:
        agent_did = "did:test:agent-lighthouse-self-wallet"
        mock_class, mock_instance = _mock_lighthouse_client()
        monkeypatch.setattr(
            FoundationPayerResolver,
            "_evm_address_from_private_key",
            lambda _self: "0x1234567890abcdef1234567890abcdef12345678",
        )
        monkeypatch.setattr(
            FoundationPayerResolver,
            "_sign_eth_message",
            lambda _self, message: f"0xsigned-{message}",
        )

        with patch(
            "kestrel_sovereign.storage.providers.lighthouse_rest."
            "LighthouseRestClient",
            mock_class,
        ):
            resolver = FoundationPayerResolver(
                _self_wallet_storage_policy(),
                db=db,
                wallet_private_key="11" * 32,
            )
            result = await resolver.resolve_for(agent_did, ResourceClass.STORAGE)

        assert result.enabled is True
        assert result.key_resolver is not None
        assert (
            await result.key_resolver.resolve_key("lighthouse", require=True)
        ) == "lh-agent-key"
        storage = ServiceKeyStorage(db, agent_did)
        assert await storage.get_key("lighthouse") == "lh-agent-key"
        mock_instance.get_auth_message.assert_awaited_once_with(
            "0x1234567890abcdef1234567890abcdef12345678"
        )
        mock_instance.create_api_key.assert_awaited_once_with(
            "0x1234567890abcdef1234567890abcdef12345678",
            "0xsigned-Sign this Lighthouse challenge",
        )
        mock_instance.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_existing_lighthouse_key_skips_remote_mint(
        self, db: AsyncDatabase, monkeypatch
    ) -> None:
        agent_did = "did:test:agent-lighthouse-existing"
        storage = ServiceKeyStorage(db, agent_did)
        await storage.store_key("lighthouse", "lh-existing")
        mock_class, mock_instance = _mock_lighthouse_client()
        monkeypatch.setattr(
            FoundationPayerResolver,
            "_evm_address_from_private_key",
            lambda _self: "0x1234567890abcdef1234567890abcdef12345678",
        )

        with patch(
            "kestrel_sovereign.storage.providers.lighthouse_rest."
            "LighthouseRestClient",
            mock_class,
        ):
            resolver = FoundationPayerResolver(
                _self_wallet_storage_policy(),
                db=db,
                wallet_private_key="11" * 32,
            )
            result = await resolver.resolve_for(agent_did, ResourceClass.STORAGE)

        assert result.enabled is True
        assert await storage.get_key("lighthouse") == "lh-existing"
        mock_instance.get_auth_message.assert_not_awaited()
        mock_instance.create_api_key.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_wallet_private_key_fails_closed(
        self, db: AsyncDatabase
    ) -> None:
        resolver = FoundationPayerResolver(_self_wallet_storage_policy(), db=db)

        with pytest.raises(PayerPolicyError, match="private key is unavailable"):
            await resolver.resolve_for(
                "did:test:agent-no-wallet", ResourceClass.STORAGE
            )

    @pytest.mark.asyncio
    async def test_missing_db_fails_closed(self) -> None:
        resolver = FoundationPayerResolver(
            _self_wallet_storage_policy(),
            wallet_private_key="11" * 32,
        )

        with pytest.raises(PayerPolicyError, match="no agent database"):
            await resolver.resolve_for(
                "did:test:agent-no-db", ResourceClass.STORAGE
            )
