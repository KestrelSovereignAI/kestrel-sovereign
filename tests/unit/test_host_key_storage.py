"""Unit tests for HostKeyStorage.

Phase 2 of the PayerPolicy foundation work.

These tests exercise the host's master credentials storage path via a
real SQLite-backed AsyncDatabase. Coverage:
- store + get round-trip preserves the API key plaintext
- has_key reflects existence and active state
- list_keys returns no secrets
- delete_key reports True only when a row was actually removed
- key isolation: a host key and an agent key for the same provider do
  NOT collide on storage and are NOT cross-decryptable (different
  identity → different HKDF derivation)
"""
from __future__ import annotations

import secrets
from typing import Iterator

import pytest
import pytest_asyncio

from kestrel_sovereign.security.host_key_storage import (
    HostKeyStorage,
    HostKeyInfo,
)
from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
from kestrel_sovereign.security.exceptions import (
    KeyNotConfiguredError,
    DecryptionError,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.fixture(autouse=True)
def _kestrel_data_key(monkeypatch) -> Iterator[None]:
    """Every test sets KESTREL_DATA_KEY to a fixed value so encryption works."""
    # 32 bytes encoded; the SDK accepts a 32-byte raw or base64-encoded master.
    monkeypatch.setenv(
        "KESTREL_DATA_KEY",
        "test-master-key-32-bytes-fixed--",
    )
    yield


@pytest_asyncio.fixture
async def db(tmp_path) -> AsyncDatabase:
    """Per-test SQLite AsyncDatabase with CORE_SCHEMA initialized."""
    db_path = tmp_path / "test.db"
    database = await AsyncDatabase.sqlite(str(db_path))
    yield database
    await database.close()


class TestHostKeyStorageRoundTrip:
    @pytest.mark.asyncio
    async def test_store_then_get_returns_plaintext(self, db: AsyncDatabase) -> None:
        storage = HostKeyStorage(db)
        plaintext = "sk-or-v1-host-master-" + secrets.token_hex(8)

        await storage.store_key("openrouter", plaintext)
        retrieved = await storage.get_key("openrouter")

        assert retrieved == plaintext

    @pytest.mark.asyncio
    async def test_get_unknown_provider_raises(self, db: AsyncDatabase) -> None:
        storage = HostKeyStorage(db)
        with pytest.raises(KeyNotConfiguredError):
            await storage.get_key("openrouter")

    @pytest.mark.asyncio
    async def test_store_replaces_existing_key(self, db: AsyncDatabase) -> None:
        storage = HostKeyStorage(db)
        await storage.store_key("openrouter", "first-key")
        await storage.store_key("openrouter", "second-key")
        assert (await storage.get_key("openrouter")) == "second-key"


class TestHostKeyStorageQueries:
    @pytest.mark.asyncio
    async def test_has_key_reflects_state(self, db: AsyncDatabase) -> None:
        storage = HostKeyStorage(db)
        assert (await storage.has_key("openrouter")) is False
        await storage.store_key("openrouter", "key")
        assert (await storage.has_key("openrouter")) is True

    @pytest.mark.asyncio
    async def test_list_keys_returns_only_metadata(self, db: AsyncDatabase) -> None:
        storage = HostKeyStorage(db)
        await storage.store_key("openrouter", "secret-or")
        await storage.store_key("lighthouse", "secret-lh")

        keys = await storage.list_keys()
        assert len(keys) == 2
        for k in keys:
            assert isinstance(k, HostKeyInfo)
            # Sanity: HostKeyInfo has no plaintext field; this just
            # confirms the dataclass surface stays minimal.
            assert not hasattr(k, "api_key")
            assert not hasattr(k, "encrypted_key")
        providers = {k.provider_id for k in keys}
        assert providers == {"openrouter", "lighthouse"}


class TestHostKeyStorageDelete:
    @pytest.mark.asyncio
    async def test_delete_returns_true_when_present(self, db: AsyncDatabase) -> None:
        storage = HostKeyStorage(db)
        await storage.store_key("openrouter", "key")
        assert (await storage.delete_key("openrouter")) is True
        assert (await storage.has_key("openrouter")) is False

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_absent(self, db: AsyncDatabase) -> None:
        storage = HostKeyStorage(db)
        # Different from the asyncpg "UPDATE 0" pattern that bit Frinz's
        # Prereq-B: this storage explicitly returns False for noop deletes
        # so the upcoming PayerResolver can distinguish "removed" from
        # "never existed" without parsing command tags.
        assert (await storage.delete_key("openrouter")) is False


class TestHostAgentIsolation:
    """Host master key storage and agent key storage must not cross-pollinate.

    They share the encryption module (`encrypt(identity, purpose, ...)`),
    but the host uses identity ``"host"`` and an agent uses its DID.
    Two storages writing to different tables under different identities
    must produce independent ciphertexts AND non-cross-decryptable rows.
    """

    @pytest.mark.asyncio
    async def test_host_and_agent_for_same_provider_are_independent(
        self, db: AsyncDatabase
    ) -> None:
        host = HostKeyStorage(db)
        agent = ServiceKeyStorage(db, agent_did="did:test:agent-isolation")

        await host.store_key("openrouter", "host-master-or-key")
        await agent.store_key("openrouter", "agent-child-or-key")

        # Each side returns its own value.
        assert (await host.get_key("openrouter")) == "host-master-or-key"
        assert (await agent.get_key("openrouter")) == "agent-child-or-key"

    @pytest.mark.asyncio
    async def test_host_ciphertext_does_not_decrypt_under_agent_identity(
        self, db: AsyncDatabase
    ) -> None:
        """If we copy the host's encrypted blob into the agent table and
        try to decrypt it under the agent's identity, decryption MUST
        fail. This guards against any future refactor that accidentally
        unifies the identity used for host and agent paths.
        """
        host = HostKeyStorage(db)
        agent_did = "did:test:agent-isolation-2"
        agent = ServiceKeyStorage(db, agent_did=agent_did)

        plaintext = "host-only-secret"
        await host.store_key("openrouter", plaintext)

        # Pull the encrypted blob the host wrote.
        rows = await db.fetchall(
            "SELECT encrypted_key FROM host_service_keys WHERE provider_id = ?",
            ("openrouter",),
        )
        assert rows, "host_service_keys row should exist"
        encrypted_b64 = rows[0][0]

        # Insert a forged agent_service_keys row using that same blob,
        # then attempt to retrieve it via ServiceKeyStorage.get_key().
        # Decryption should fail because the agent's identity derives a
        # different key than the host's.
        await db.execute(
            """
            INSERT INTO agent_service_keys
            (id, agent_did, provider_id, encrypted_key, key_hash,
             quota_limit, quota_used, is_active, created_at)
            VALUES ('forged', ?, ?, ?, 'forged-hash', NULL, 0, 1, CURRENT_TIMESTAMP)
            """,
            (agent_did, "openrouter", encrypted_b64),
        )

        with pytest.raises(DecryptionError):
            await agent.get_key("openrouter")
