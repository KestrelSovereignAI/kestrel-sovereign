"""Unit tests for zero-knowledge USER_BYOK key storage."""
from __future__ import annotations

import pytest
import pytest_asyncio

from kestrel_sovereign.security.exceptions import (
    DecryptionError,
    PassphraseRequiredError,
)
from kestrel_sovereign.security.user_byok_key_storage import UserBYOKKeyStorage
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest_asyncio.fixture
async def db(tmp_path) -> AsyncDatabase:
    database = await AsyncDatabase.sqlite(str(tmp_path / "test.db"))
    yield database
    await database.close()


class TestUserBYOKKeyStorage:
    @pytest.mark.asyncio
    async def test_round_trip_requires_passphrase(self, db: AsyncDatabase) -> None:
        storage = UserBYOKKeyStorage(db, "did:test:agent-byok")
        await storage.store_key(
            "openrouter",
            "sk-or-v1-user-byok",
            passphrase="correct horse battery staple",
        )

        assert await storage.get_key(
            "openrouter",
            passphrase="correct horse battery staple",
        ) == "sk-or-v1-user-byok"

    @pytest.mark.asyncio
    async def test_wrong_passphrase_fails_closed(self, db: AsyncDatabase) -> None:
        storage = UserBYOKKeyStorage(db, "did:test:agent-byok")
        await storage.store_key(
            "openrouter",
            "sk-or-v1-user-byok",
            passphrase="right passphrase",
        )

        with pytest.raises(DecryptionError):
            await storage.get_key("openrouter", passphrase="wrong passphrase")

    @pytest.mark.asyncio
    async def test_missing_passphrase_rejected(self, db: AsyncDatabase) -> None:
        storage = UserBYOKKeyStorage(db, "did:test:agent-byok")
        with pytest.raises(PassphraseRequiredError):
            await storage.store_key("openrouter", "sk-or-v1-user-byok", passphrase="")

    @pytest.mark.asyncio
    async def test_at_rest_contains_only_ciphertext_salt_nonce(
        self,
        db: AsyncDatabase,
    ) -> None:
        storage = UserBYOKKeyStorage(db, "did:test:agent-byok")
        await storage.store_key(
            "openrouter",
            "sk-or-v1-user-byok",
            passphrase="do not store me",
        )

        rows = await db.fetchall(
            """
            SELECT encrypted_key, key_salt, key_nonce, key_hash
            FROM user_byok_service_keys
            WHERE agent_did = ? AND provider_id = ?
            """,
            ("did:test:agent-byok", "openrouter"),
        )
        assert rows
        stored_text = "|".join(str(value) for value in rows[0])
        assert "sk-or-v1-user-byok" not in stored_text
        assert "do not store me" not in stored_text
        assert rows[0][0]
        assert rows[0][1]
        assert rows[0][2]
        assert rows[0][3]
