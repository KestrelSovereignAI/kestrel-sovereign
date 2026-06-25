"""Unit tests for USER_BYOK payer resolution (#1648)."""
from __future__ import annotations

import pytest
import pytest_asyncio

from kestrel_sdk.payer_policy import (
    PayerKind,
    PayerPolicy,
    PayerPolicyError,
    PayerSpec,
    ResourceClass,
    SupportStatus,
    status_for,
)

from kestrel_sovereign.security.exceptions import DecryptionError, PassphraseRequiredError
from kestrel_sovereign.security.user_byok_key_storage import UserBYOKKeyStorage
from kestrel_sovereign.services.payer_resolver import FoundationPayerResolver
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest_asyncio.fixture
async def db(tmp_path) -> AsyncDatabase:
    database = await AsyncDatabase.sqlite(str(tmp_path / "test.db"))
    yield database
    await database.close()


def _byok_policy() -> PayerPolicy:
    return PayerPolicy(
        llm=PayerSpec(vendor="openrouter", kind=PayerKind.USER_BYOK),
        storage=PayerSpec(vendor="lighthouse", kind=PayerKind.HOST_ENV),
        compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
    )


class TestSDKUserBYOK:
    def test_kind_and_support_matrix_exist(self) -> None:
        assert PayerKind.USER_BYOK.value == "user_byok"
        assert (
            status_for(ResourceClass.LLM, "openrouter", PayerKind.USER_BYOK)
            is SupportStatus.READY
        )

    def test_user_byok_forbids_caps(self) -> None:
        with pytest.raises(ValueError, match="caps"):
            PayerSpec(
                vendor="openrouter",
                kind=PayerKind.USER_BYOK,
                monthly_cap_usd="25",
            )


class TestFoundationResolverUserBYOK:
    @pytest.mark.asyncio
    async def test_resolves_with_per_request_passphrase(self, db: AsyncDatabase) -> None:
        agent_did = "did:test:agent-byok-resolve"
        await UserBYOKKeyStorage(db, agent_did).store_key(
            "openrouter",
            "sk-or-v1-user-byok",
            passphrase="correct passphrase",
        )
        resolver = FoundationPayerResolver(_byok_policy(), db=db)

        result = await resolver.resolve_for(
            agent_did,
            ResourceClass.LLM,
            user_passphrase="correct passphrase",
        )

        assert result.enabled is True
        assert result.key_resolver is not None
        assert await result.key_resolver.resolve_key("openrouter") == "sk-or-v1-user-byok"

    @pytest.mark.asyncio
    async def test_missing_passphrase_fails_closed(self, db: AsyncDatabase) -> None:
        agent_did = "did:test:agent-byok-no-pass"
        await UserBYOKKeyStorage(db, agent_did).store_key(
            "openrouter",
            "sk-or-v1-user-byok",
            passphrase="correct passphrase",
        )
        resolver = FoundationPayerResolver(_byok_policy(), db=db)

        with pytest.raises(PassphraseRequiredError):
            await resolver.resolve_for(agent_did, ResourceClass.LLM)

    @pytest.mark.asyncio
    async def test_wrong_passphrase_fails_closed(self, db: AsyncDatabase) -> None:
        agent_did = "did:test:agent-byok-wrong-pass"
        await UserBYOKKeyStorage(db, agent_did).store_key(
            "openrouter",
            "sk-or-v1-user-byok",
            passphrase="correct passphrase",
        )
        resolver = FoundationPayerResolver(_byok_policy(), db=db)
        result = await resolver.resolve_for(
            agent_did,
            ResourceClass.LLM,
            user_passphrase="wrong passphrase",
        )

        with pytest.raises(DecryptionError):
            await result.key_resolver.resolve_key("openrouter")

    @pytest.mark.asyncio
    async def test_missing_byok_key_fails_closed(self, db: AsyncDatabase) -> None:
        resolver = FoundationPayerResolver(_byok_policy(), db=db)

        with pytest.raises(PayerPolicyError, match="no zero-knowledge BYOK key"):
            await resolver.resolve_for(
                "did:test:agent-byok-missing",
                ResourceClass.LLM,
                user_passphrase="correct passphrase",
            )
