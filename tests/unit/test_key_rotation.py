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
    _derive_agent_cipher,
    _master_bytes_from_key,
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
    async def test_rotate_per_agent_hkdf_rows(self):
        """The big regression #936 found: conversation_history rows are
        encrypted with per-agent HKDF-derived ciphers (matches production
        ``async_conversation_store.add_conversation``). Rotation must
        derive the matching cipher per row, not use a single global
        cipher. Pre-fix the global cipher couldn't decrypt these rows
        and they were silently skipped — permanent rubble after a swap.
        """
        db, path = await _fresh_db()
        try:
            old_master = _master_bytes_from_key(Fernet.generate_key().decode())
            new_master = _master_bytes_from_key(Fernet.generate_key().decode())

            # Two different agents, each with their own HKDF-derived cipher
            old_alice = _derive_agent_cipher(old_master, "alice")
            old_bob = _derive_agent_cipher(old_master, "bob")

            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) "
                "VALUES (1, 'alice', 'user', ?)",
                (old_alice.encrypt(b"alice-secret").decode(),),
            )
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) "
                "VALUES (2, 'bob', 'user', ?)",
                (old_bob.encrypt(b"bob-secret").decode(),),
            )
            # Plaintext row — must NOT be touched
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) "
                "VALUES (3, 'alice', 'user', 'plaintext-row')"
            )

            storage = MagicMock()
            storage.database = db
            service = KeyRotationService(storage=storage)
            service._save_rotation = AsyncMock()

            rotation = RotationRecord(
                id="rot-per-agent",
                started_at=datetime.now(timezone.utc),
                old_key_hash="old",
                new_key_hash="new",
                status=RotationStatus.IN_PROGRESS,
            )
            await service._rotate_table(
                rotation, old_master, new_master,
                table="conversation_history",
                content_column="content",
                id_column="id",
                agent_id_column="agent_id",
            )

            assert rotation.records_processed == 2, (
                "Both per-agent encrypted rows must rotate. Pre-fix only "
                "the global cipher was tried and both rows would have "
                "raised DecryptionError and been silently skipped."
            )

            # Each row must now decrypt under its own NEW per-agent cipher
            new_alice = _derive_agent_cipher(new_master, "alice")
            new_bob = _derive_agent_cipher(new_master, "bob")
            rows = await db.fetchall(
                "SELECT id, agent_id, content FROM conversation_history ORDER BY id"
            )
            assert new_alice.decrypt(rows[0][2].encode()) == b"alice-secret"
            assert new_bob.decrypt(rows[1][2].encode()) == b"bob-secret"
            # Cross-agent ciphers must NOT decrypt each other's rows
            from kestrel_sdk.security.exceptions import DecryptionError
            with pytest.raises(DecryptionError):
                new_alice.decrypt(rows[1][2].encode())
            with pytest.raises(DecryptionError):
                new_bob.decrypt(rows[0][2].encode())
            # Plaintext row preserved
            assert rows[2][2] == "plaintext-row"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_rotate_files_blob_column(self):
        """``files.content`` is BLOB (bytes). Rotation must read/write
        bytes without crashing on ``.encode()`` like the pre-fix code."""
        db, path = await _fresh_db()
        try:
            old_master = _master_bytes_from_key(Fernet.generate_key().decode())
            new_master = _master_bytes_from_key(Fernet.generate_key().decode())
            old_global = AEADCipher(old_master)
            new_global = AEADCipher(new_master)

            # Insert as BLOB — bytes, not str (matches async_file_store.store_file)
            blob_token = old_global.encrypt(b"file-bytes")
            await db.execute(
                "INSERT INTO files (content_hash, original_name, content) "
                "VALUES ('h1', 'a.bin', ?)",
                (blob_token,),
            )

            storage = MagicMock()
            storage.database = db
            service = KeyRotationService(storage=storage)
            service._save_rotation = AsyncMock()

            rotation = RotationRecord(
                id="rot-blob",
                started_at=datetime.now(timezone.utc),
                old_key_hash="old",
                new_key_hash="new",
                status=RotationStatus.IN_PROGRESS,
            )
            await service._rotate_table(
                rotation, old_master, new_master,
                table="files",
                content_column="content",
                id_column="content_hash",
                agent_id_column=None,
            )

            assert rotation.records_processed == 1, (
                "BLOB row must rotate without crashing. Pre-fix "
                "encrypted_content.encode() raised AttributeError on bytes."
            )
            row = await db.fetchval("SELECT content FROM files WHERE content_hash = 'h1'")
            # The rewritten value must be bytes (preserves BLOB column type)
            assert isinstance(row, bytes), f"BLOB column must store bytes, got {type(row)}"
            assert new_global.decrypt(row) == b"file-bytes"
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
            old_master = _master_bytes_from_key(Fernet.generate_key().decode())
            new_master = _master_bytes_from_key(Fernet.generate_key().decode())
            old_alice = _derive_agent_cipher(old_master, "a")
            new_alice = _derive_agent_cipher(new_master, "a")

            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) VALUES (1, 'a', 'user', ?)",
                (old_alice.encrypt(b"first").decode(),),
            )
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) VALUES (2, 'a', 'user', ?)",
                (old_alice.encrypt(b"second").decode(),),
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
                rotation, old_master, new_master,
                table="conversation_history",
                content_column="content",
                id_column="id",
                agent_id_column="agent_id",
            )

            # Only row 2 should have been processed this run
            assert rotation.records_processed == 1

            # Row 1 still under old_alice (untouched); row 2 under new_alice
            row1 = await db.fetchval("SELECT content FROM conversation_history WHERE id = 1")
            row2 = await db.fetchval("SELECT content FROM conversation_history WHERE id = 2")
            assert old_alice.decrypt(row1.encode()) == b"first"
            assert new_alice.decrypt(row2.encode()) == b"second"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_upgrade_to_aead_rewrites_legacy_fernet_in_place(self, monkeypatch):
        """The eager AEAD upgrade path: same key, lifts v1 Fernet rows to v2.

        Reproduces a mixed corpus: one row written by the OLD global Fernet
        path (key_version 0), one by the per-agent HKDF Fernet path
        (key_version 1, what production uses today). Both must lift to v2;
        per-agent rows must remain readable under the per-agent NEW cipher,
        global rows under the global NEW cipher.
        """
        db, path = await _fresh_db()
        try:
            passphrase = "the-runtime-passphrase"
            monkeypatch.setenv("KESTREL_DATA_KEY", passphrase)

            master = _master_bytes_from_key(passphrase)
            # Row 1: legacy global Fernet (matches pre-v1 / no-agent-id corpus)
            global_b64 = master  # same shape Fernet expects
            global_legacy = Fernet(global_b64).encrypt(b"global-legacy").decode()
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) "
                "VALUES (1, 'a', 'user', ?)",
                (global_legacy,),
            )
            # Row 2: per-agent Fernet (matches what add_conversation writes today
            # for an agent with key_version=1)
            import base64 as _b64
            agent_derived = _derive_agent_cipher(master, "a")
            # Reproduce the per-agent Fernet path by raw HKDF + base64 → Fernet
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            from cryptography.hazmat.primitives import hashes as _h
            hkdf = HKDF(algorithm=_h.SHA256(), length=32, salt=b"a", info=b"kestrel-agent-v1")
            agent_raw = hkdf.derive(master)
            agent_legacy = Fernet(_b64.urlsafe_b64encode(agent_raw)).encrypt(b"agent-legacy").decode()
            await db.execute(
                "INSERT INTO conversation_history (id, agent_id, role, content) "
                "VALUES (2, 'a', 'user', ?)",
                (agent_legacy,),
            )

            storage = MagicMock()
            storage.database = db
            service = KeyRotationService(storage=storage)
            await service.initialize()  # idempotent

            await service.upgrade_to_aead()
            await service.drain_rotations()

            rows = await db.fetchall(
                "SELECT id, content FROM conversation_history ORDER BY id"
            )
            assert all(r[1].startswith("KSAv2:") for r in rows), (
                f"All rows must be lifted to v2; got prefixes "
                f"{[r[1][:10] for r in rows]}"
            )

            # Row 1 (originally global Fernet) decrypts under the per-agent NEW
            # cipher because the upgrade rewrites every per-agent table row
            # under the per-agent cipher (matches production write semantics
            # post-Wave-0C).
            assert agent_derived.decrypt(rows[0][1].encode()) == b"global-legacy"
            assert agent_derived.decrypt(rows[1][1].encode()) == b"agent-legacy"
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
