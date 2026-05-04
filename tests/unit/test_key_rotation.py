import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.fernet import Fernet

from kestrel_sdk.security.aead import AEADCipher
from kestrel_sovereign.security import encryption
from kestrel_sovereign.security import key_rotation as key_rotation_module
from kestrel_sovereign.security.key_rotation import (
    KeyRotationService,
    RotationRecord,
    RotationStatus,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase


class TestKeyRotationTaskLifecycle:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_tracked_rotation_tasks(self):
        service = KeyRotationService(storage=MagicMock())
        started = asyncio.Event()

        async def never_finishes():
            started.set()
            await asyncio.Event().wait()

        task = service._track_rotation_task(never_finishes(), "rotation-1")
        await started.wait()

        await service.shutdown()

        assert task.done()
        assert task.cancelled()
        assert service._rotation_tasks == set()

    @pytest.mark.asyncio
    async def test_start_rotation_tracks_background_rotation_task(self, monkeypatch):
        service = KeyRotationService(storage=MagicMock())
        service._get_in_progress_rotation = AsyncMock(return_value=None)
        service._count_encrypted_records = AsyncMock(return_value=0)
        service._save_rotation = AsyncMock()

        old_key = Fernet.generate_key().decode()
        new_key = Fernet.generate_key().decode()
        started = asyncio.Event()

        monkeypatch.setattr(
            key_rotation_module,
            "get_fernet",
            lambda: Fernet(old_key),
        )
        monkeypatch.setattr(
            encryption,
            "_get_data_key",
            lambda: old_key,
        )

        async def never_finishes(rotation, old_fernet, new_fernet):
            started.set()
            await asyncio.Event().wait()

        service._execute_rotation = never_finishes

        rotation_id = await service.start_rotation(new_key=new_key)
        await started.wait()

        assert rotation_id
        assert len(service._rotation_tasks) == 1

        task = next(iter(service._rotation_tasks))
        assert task.get_name() == f"key-rotation-{rotation_id}"

        await service.shutdown()

        assert task.done()
        assert task.cancelled()
        assert service._rotation_tasks == set()


async def _fresh_db():
    """Build a temp SQLite-backed AsyncDatabase with the schemas the rotation needs."""
    fd, path = tempfile.mkstemp(suffix=".db")
    import os as _os
    _os.close(fd)
    db = await AsyncDatabase.sqlite(path)
    # AsyncDatabase.sqlite() auto-applies the core schema (conversation_history,
    # files, etc.). We just need to ensure rotation tracking tables exist;
    # storage tables are already there.
    await db.execute(
        "CREATE TABLE IF NOT EXISTS key_rotations ("
        "id TEXT PRIMARY KEY, started_at TEXT, completed_at TEXT, "
        "old_key_hash TEXT, new_key_hash TEXT, status TEXT, "
        "records_processed INTEGER, records_total INTEGER, error_message TEXT)"
    )
    await db.execute(
        "CREATE TABLE IF NOT EXISTS rotation_progress ("
        "rotation_id TEXT, table_name TEXT, record_id TEXT, rotated_at TEXT, "
        "PRIMARY KEY (rotation_id, table_name, record_id))"
    )
    return db, path


class TestRotationEndToEnd:
    """End-to-end rotation tests against a real AsyncDatabase.

    Pre-fix (#932): the rotation code used an
    ``async with database.execute(...) as cursor`` shape that did not match
    the current ``AsyncDatabase.execute(...)`` API. The rotation has been a
    no-op against real DBs. Combined with the Wave 0C prefix-filter fix and
    the table-name correction (``conversations`` → ``conversation_history``),
    this suite proves the rotation actually works end-to-end.
    """

    @pytest.mark.asyncio
    async def test_rotate_picks_up_both_legacy_and_v2_rows(self):
        db, path = await _fresh_db()
        try:
            key_b64 = Fernet.generate_key()
            old_cipher = AEADCipher(key_b64)
            new_cipher = AEADCipher(Fernet.generate_key())

            # 1 legacy Fernet row + 1 v2 row + 1 plaintext (must NOT be touched)
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) VALUES (1, 'a', 'user', ?)",
                (Fernet(key_b64).encrypt(b"legacy-payload").decode(),),
            )
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) VALUES (2, 'a', 'user', ?)",
                (old_cipher.encrypt(b"new-payload").decode(),),
            )
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) VALUES (3, 'a', 'user', 'plaintext-row')"
            )

            storage = MagicMock()
            storage.database = db
            service = KeyRotationService(storage=storage)
            service._save_rotation = AsyncMock()

            rotation = RotationRecord(
                id="rot-1",
                started_at=datetime.now(timezone.utc),
                old_key_hash="old",
                new_key_hash="new",
                status=RotationStatus.IN_PROGRESS,
            )
            await service._rotate_table(
                rotation, old_cipher, new_cipher,
                table="conversation_history",
                content_column="content",
                id_column="id",
            )

            # Both encrypted rows processed; plaintext row left alone
            assert rotation.records_processed == 2

            # Both encrypted rows now under the new cipher
            rows = await db.fetchall(
                "SELECT id, content FROM conversation_history ORDER BY id"
            )
            assert new_cipher.decrypt(rows[0][1].encode()) == b"legacy-payload"
            assert new_cipher.decrypt(rows[1][1].encode()) == b"new-payload"
            # Plaintext row preserved verbatim
            assert rows[2][1] == "plaintext-row"

            # Old cipher must no longer decrypt the rotated rows
            from kestrel_sdk.security.exceptions import DecryptionError
            for _id, ct in rows[:2]:
                with pytest.raises(DecryptionError):
                    old_cipher.decrypt(ct.encode())
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_count_includes_both_prefixes_across_real_tables(self):
        db, path = await _fresh_db()
        try:
            key_b64 = Fernet.generate_key()
            cipher = AEADCipher(key_b64)
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) VALUES (1, 'a', 'user', ?)",
                (Fernet(key_b64).encrypt(b"a").decode(),),
            )
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) VALUES (2, 'a', 'user', ?)",
                (cipher.encrypt(b"b").decode(),),
            )
            await db.execute(
                "INSERT INTO files (content_hash, original_name, content) VALUES ('h1', 'f1.txt', ?)",
                (cipher.encrypt(b"c").decode(),),
            )
            # Plaintext rows in both tables — must NOT be counted
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) VALUES (3, 'a', 'user', 'plain')"
            )
            await db.execute(
                "INSERT INTO files (content_hash, original_name, content) VALUES ('h2', 'f2.txt', 'plain')"
            )

            storage = MagicMock()
            storage.database = db
            service = KeyRotationService(storage=storage)
            count = await service._count_encrypted_records()
            assert count == 3, (
                "Count must include both gAAAAA% and KSAv2:% across all "
                "registered encrypted tables, and exclude plaintext rows."
            )
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_rotation_resumability_skips_already_rotated_rows(self):
        db, path = await _fresh_db()
        try:
            key_b64 = Fernet.generate_key()
            old_cipher = AEADCipher(key_b64)
            new_cipher = AEADCipher(Fernet.generate_key())
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) VALUES (1, 'a', 'user', ?)",
                (old_cipher.encrypt(b"first").decode(),),
            )
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) VALUES (2, 'a', 'user', ?)",
                (old_cipher.encrypt(b"second").decode(),),
            )
            # Pretend row 1 was already rotated in a previous (interrupted) run
            await db.execute(
                "INSERT INTO rotation_progress VALUES (?, ?, ?, ?)",
                ("rot-resume", "conversation_history", "1", datetime.now(timezone.utc).isoformat()),
            )

            storage = MagicMock()
            storage.database = db
            service = KeyRotationService(storage=storage)
            service._save_rotation = AsyncMock()

            rotation = RotationRecord(
                id="rot-resume",
                started_at=datetime.now(timezone.utc),
                old_key_hash="old",
                new_key_hash="new",
                status=RotationStatus.IN_PROGRESS,
            )
            await service._rotate_table(
                rotation, old_cipher, new_cipher,
                table="conversation_history",
                content_column="content",
                id_column="id",
            )

            # Only row 2 should have been processed this run
            assert rotation.records_processed == 1

            # Row 1 still under old_cipher (untouched); row 2 under new_cipher
            row1 = await db.fetchval("SELECT content FROM conversation_history WHERE id = 1")
            row2 = await db.fetchval("SELECT content FROM conversation_history WHERE id = 2")
            assert old_cipher.decrypt(row1.encode()) == b"first"
            assert new_cipher.decrypt(row2.encode()) == b"second"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_upgrade_to_aead_rewrites_legacy_fernet_in_place(self, monkeypatch):
        """The eager AEAD upgrade path: same key, lifts v1 Fernet rows to v2."""
        db, path = await _fresh_db()
        try:
            # Set up a runtime key so _validate_key + get_fernet agree
            passphrase = "the-runtime-passphrase"
            monkeypatch.setenv("KESTREL_DATA_KEY", passphrase)

            # Insert legacy Fernet rows under the runtime-derived key
            runtime_cipher = encryption.get_fernet()
            # The runtime cipher emits v2; force a Fernet row by going around it
            import hashlib, base64 as _b64
            digest = hashlib.sha256(passphrase.encode()).digest()
            legacy_token = Fernet(_b64.urlsafe_b64encode(digest)).encrypt(b"legacy").decode()
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) VALUES (1, 'a', 'user', ?)",
                (legacy_token,),
            )

            storage = MagicMock()
            storage.database = db
            service = KeyRotationService(storage=storage)
            await service.initialize()  # creates tracking tables (idempotent)

            rotation_id = await service.upgrade_to_aead()
            await service.drain_rotations()

            row = await db.fetchval("SELECT content FROM conversation_history WHERE id = 1")
            # Must now be v2 (KSAv2:) and decrypt under runtime cipher
            assert row.startswith("KSAv2:"), (
                f"upgrade_to_aead must rewrite the row as v2, got: {row[:20]!r}"
            )
            assert runtime_cipher.decrypt(row.encode()) == b"legacy"
        finally:
            await db.close()

    def test_validate_key_passphrase_matches_get_fernet(self, monkeypatch):
        """Wave 0C regression: ``_validate_key`` must derive the same AES
        key as ``get_fernet()`` for any given KESTREL_DATA_KEY value.

        The pre-fix ``_validate_key`` called ``AEADCipher(key)`` directly,
        which accepts ANY 32-byte string as a raw AES key. So a 32-char
        passphrase like 'a'*32 was used as-is for the rotation key, while
        ``get_fernet()`` for the same passphrase SHA-256-derives the key.
        Different keys → rotation re-encrypts data under a key the
        runtime can never reproduce → permanent ciphertext rubble after
        the user swaps KESTREL_DATA_KEY.
        """
        # 32-character passphrase — the pathological case
        passphrase = "a" * 32
        monkeypatch.setenv("KESTREL_DATA_KEY", passphrase)

        runtime_cipher = encryption.get_fernet()
        rotation_cipher = key_rotation_module._validate_key(passphrase)

        plaintext = b"data the user expects to keep across rotation"
        rotated = rotation_cipher.encrypt(plaintext)

        # Pre-fix this raised DecryptionError because the keys diverged:
        # rotation_cipher used raw `b'aaaa...'`, runtime used SHA-256(passphrase)
        assert runtime_cipher.decrypt(rotated) == plaintext, (
            "_validate_key must produce the same AES key as get_fernet() — "
            "otherwise rotated rows become unreadable after a KESTREL_DATA_KEY swap"
        )

    def test_validate_key_short_passphrase_matches_get_fernet(self, monkeypatch):
        """Same invariant for short passphrases (the common case)."""
        passphrase = "hunter2"  # 7 chars, won't even base64-decode
        monkeypatch.setenv("KESTREL_DATA_KEY", passphrase)

        runtime = encryption.get_fernet()
        rotation = key_rotation_module._validate_key(passphrase)

        pt = b"x"
        assert runtime.decrypt(rotation.encrypt(pt)) == pt
        assert rotation.decrypt(runtime.encrypt(pt)) == pt

    def test_validate_key_real_fernet_key_matches_get_fernet(self, monkeypatch):
        """Same invariant when the input is a genuine Fernet-shaped key."""
        real_fernet_key = Fernet.generate_key().decode()
        monkeypatch.setenv("KESTREL_DATA_KEY", real_fernet_key)

        runtime = encryption.get_fernet()
        rotation = key_rotation_module._validate_key(real_fernet_key)

        pt = b"real-fernet-key-path"
        assert runtime.decrypt(rotation.encrypt(pt)) == pt
        assert rotation.decrypt(runtime.encrypt(pt)) == pt

    # The earlier SQL-string-only `_SqlCapture` regression test is now
    # subsumed by `test_count_includes_both_prefixes_across_real_tables`
    # which exercises the count against a real AsyncDatabase.
