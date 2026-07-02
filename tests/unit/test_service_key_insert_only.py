"""Unit tests for insert-only ServiceKeyStorage.store_key (F196).

A plain add_service_key must not silently overwrite an existing key — that is
a rotation, which is approval-gated. Enforcement lives in storage: store_key is
insert-only unless replace=True, which only the rotation path passes.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.security.exceptions import KeyStorageError
from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
from kestrel_sovereign.storage.async_database import AsyncDatabase

TEST_AGENT_DID = "did:pkh:eip155:1:0xServiceKeyF196"
DATA_KEY = "test-encryption-key-for-unit-tests"


@pytest_asyncio.fixture
async def db(tmp_path) -> AsyncDatabase:
    database = await AsyncDatabase.sqlite(str(tmp_path / "test_service_keys.db"))
    yield database
    await database.close()


@pytest_asyncio.fixture
async def storage(db: AsyncDatabase):
    with patch.dict(os.environ, {"KESTREL_DATA_KEY": DATA_KEY}):
        yield ServiceKeyStorage(db, TEST_AGENT_DID)


class TestStoreKeyInsertOnly:
    @pytest.mark.asyncio
    async def test_store_new_key_ok(self, storage: ServiceKeyStorage) -> None:
        key_id = await storage.store_key(provider_id="openai", api_key="sk-first")
        assert key_id
        assert await storage.get_key("openai") == "sk-first"

    @pytest.mark.asyncio
    async def test_store_over_existing_refused(self, storage: ServiceKeyStorage) -> None:
        await storage.store_key(provider_id="openai", api_key="sk-first")

        with pytest.raises(KeyStorageError, match="rotate_service_key"):
            await storage.store_key(provider_id="openai", api_key="sk-second")

        # Stored key unchanged.
        assert await storage.get_key("openai") == "sk-first"

    @pytest.mark.asyncio
    async def test_store_over_inactive_refused(self, storage: ServiceKeyStorage) -> None:
        await storage.store_key(provider_id="openai", api_key="sk-first")
        await storage.deactivate_key(provider_id="openai")

        # Even a deactivated key counts as "exists" — no silent replace.
        with pytest.raises(KeyStorageError, match="rotate_service_key"):
            await storage.store_key(provider_id="openai", api_key="sk-second")

    @pytest.mark.asyncio
    async def test_replace_true_overwrites(self, storage: ServiceKeyStorage) -> None:
        await storage.store_key(provider_id="openai", api_key="sk-first")

        await storage.store_key(
            provider_id="openai", api_key="sk-rotated", replace=True
        )
        assert await storage.get_key("openai") == "sk-rotated"

    @pytest.mark.asyncio
    async def test_race_past_preflight_still_refused(
        self, storage: ServiceKeyStorage
    ) -> None:
        """The plain INSERT — not the preflight — is the enforcement point.

        Simulate the concurrency race where both callers observe no row: force
        ``_key_exists`` to report False even though a key exists. The plain
        INSERT must still hit UNIQUE(agent_did, provider_id) and translate the
        failure into KeyStorageError, so the second add cannot silently rotate
        the credential (F196).
        """
        await storage.store_key(provider_id="openai", api_key="sk-first")

        with patch.object(storage, "_key_exists", AsyncMock(return_value=False)):
            with pytest.raises(KeyStorageError, match="rotate_service_key"):
                await storage.store_key(provider_id="openai", api_key="sk-second")

        # Stored key unchanged despite the bypassed preflight.
        assert await storage.get_key("openai") == "sk-first"


class TestKeyFeatureEnforcement:
    @pytest.fixture
    def mock_agent(self, db: AsyncDatabase):
        agent = MagicMock()
        agent.storage = MagicMock()
        agent.storage.db = db
        agent.did = TEST_AGENT_DID
        # No security feature => rotation proceeds without approval queue.
        agent.get_feature = MagicMock(return_value=None)
        return agent

    @pytest.mark.asyncio
    async def test_add_over_existing_refused(self, mock_agent, db) -> None:
        from kestrel_sovereign.features.keys import KeyManagementFeature

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": DATA_KEY}):
            feature = KeyManagementFeature(mock_agent)
            await feature.initialize()

            first = await feature.add_service_key(provider="openai", api_key="sk-first")
            assert first.status is ToolResultStatus.OK

            second = await feature.add_service_key(provider="openai", api_key="sk-second")
            assert second.status is ToolResultStatus.ERROR
            assert "rotate_service_key" in second.error
            # Stored key unchanged.
            assert await feature.get_key("openai") == "sk-first"

    @pytest.mark.asyncio
    async def test_add_new_ok(self, mock_agent, db) -> None:
        from kestrel_sovereign.features.keys import KeyManagementFeature

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": DATA_KEY}):
            feature = KeyManagementFeature(mock_agent)
            await feature.initialize()

            result = await feature.add_service_key(provider="anthropic", api_key="sk-new")
            assert result.status is ToolResultStatus.OK
            assert await feature.get_key("anthropic") == "sk-new"

    @pytest.mark.asyncio
    async def test_rotate_replaces(self, mock_agent, db) -> None:
        from kestrel_sovereign.features.keys import KeyManagementFeature

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": DATA_KEY}):
            feature = KeyManagementFeature(mock_agent)
            await feature.initialize()

            await feature.add_service_key(provider="github", api_key="ghp-old")
            result = await feature.rotate_service_key(
                provider="github", new_api_key="ghp-rotated"
            )
            assert result.status is ToolResultStatus.OK
            assert await feature.get_key("github") == "ghp-rotated"
