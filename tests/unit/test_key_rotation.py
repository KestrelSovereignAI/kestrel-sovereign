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


class _SqlCapture:
    """Async-cursor-protocol mock that records the SQL passed to execute()
    and yields a configurable rowset. Captures the production code's
    `async with storage.database.execute(...) as cursor` shape.
    """

    def __init__(self, rows=None, count_value=0):
        self.queries = []
        self._rows = rows or []
        self._count_value = count_value
        self._next_count_first = True

    def execute(self, sql, *args, **kwargs):
        self.queries.append(sql)
        outer = self

        class _CursorCM:
            async def __aenter__(self_inner):
                class _Cursor:
                    def __aiter__(self_):
                        return self_

                    async def __anext__(self_):
                        raise StopAsyncIteration

                    async def fetchall(self_):
                        return outer._rows

                    async def fetchone(self_):
                        if outer._next_count_first:
                            outer._next_count_first = False
                            return (outer._count_value,)
                        return None
                return _Cursor()

            async def __aexit__(self_inner, *exc):
                return False

        return _CursorCM()


class TestRotationCoversBothPrefixes:
    """Wave 0C regression: _rotate_table must select rows encrypted under
    both legacy Fernet (`gAAAAA%`) and v2 AEAD (`KSAv2:%`) prefixes.

    The pre-fix code only matched `gAAAAA%`, which silently excluded any
    post-Wave-0C row from rotation. Once the user swapped KESTREL_DATA_KEY,
    those rows became permanent ciphertext rubble. These tests assert the
    SQL filter includes both prefix patterns; the test would have caught
    the regression.

    A pre-existing issue surfaced while writing these tests: the rotation
    code uses an `async with database.execute(...) as cursor` shape that
    doesn't match the current `AsyncDatabase.execute(...)` API (which
    returns an awaitable int, not a cursor context manager). The rotation
    has therefore been a no-op against real DBs. That is tracked as a
    follow-up; this test stays focused on the prefix-filter regression.
    """

    @pytest.mark.asyncio
    async def test_rotate_table_sql_matches_both_prefixes(self):
        cap = _SqlCapture(rows=[])  # empty rowset, we only care about the SQL
        storage = MagicMock()
        storage.database = cap
        service = KeyRotationService(storage=storage)

        rotation = RotationRecord(
            id="test-rot",
            started_at=datetime.now(timezone.utc),
            old_key_hash="old",
            new_key_hash="new",
            status=RotationStatus.IN_PROGRESS,
        )
        service._save_rotation = AsyncMock()

        old_cipher = AEADCipher(Fernet.generate_key())
        new_cipher = AEADCipher(Fernet.generate_key())

        await service._rotate_table(
            rotation, old_cipher, new_cipher,
            table="conversations",
            content_column="content",
            id_column="rowid",
        )

        # The data SELECT is one of the queries; the first is the rotation_progress
        # bookkeeping query
        joined = " ".join(cap.queries)
        assert "gAAAAA%" in joined, (
            "SELECT must still pick up legacy Fernet rows"
        )
        assert "KSAv2:%" in joined, (
            "SELECT must also pick up v2 AEAD rows; pre-fix this was missing "
            "and post-Wave-0C rows would have been silently skipped during "
            "rotation, leaving them encrypted under the old key (permanent "
            "ciphertext rubble after KESTREL_DATA_KEY swap)."
        )

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

    @pytest.mark.asyncio
    async def test_count_encrypted_records_sql_matches_both_prefixes(self):
        cap = _SqlCapture(count_value=42)
        storage = MagicMock()
        storage.database = cap
        service = KeyRotationService(storage=storage)

        await service._count_encrypted_records()

        joined = " ".join(cap.queries)
        assert "gAAAAA%" in joined
        assert "KSAv2:%" in joined, (
            "_count_encrypted_records must include v2 rows in its sweep total"
        )
