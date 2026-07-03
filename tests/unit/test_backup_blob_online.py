"""Online-backup behaviour for AsyncStorage.create_backup_blob (F266).

create_backup_blob must NOT close the live DB connection. It should snapshot the
running database with SQLite's online backup API and tar+gzip that copy off the
event loop, so concurrent reads/writes keep working during the backup.
"""
import asyncio

import pytest

from kestrel_sovereign.storage import AsyncStorage

TEST_AGENT_ID = "did:pkh:eip155:1:0xabcabcabcabcabcabcabcabcabcabcabcabcabca"


@pytest.fixture(autouse=True)
def data_key(monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-backup-blob-key")


async def _storage(tmp_path, name="kestrel.db"):
    storage = AsyncStorage(str(tmp_path / name), agent_id=TEST_AGENT_ID)
    await storage.initialize()
    return storage


@pytest.mark.asyncio
async def test_connection_stays_open_and_usable_during_and_after_backup(tmp_path):
    storage = await _storage(tmp_path)
    try:
        await storage.add_conversation("user", "hello before backup")

        # Kick off the backup and a concurrent write at the same time. If the
        # implementation closed the live connection (the old behaviour), the
        # concurrent write would fail because the DB is gone.
        blob, _ = await asyncio.gather(
            storage.create_backup_blob(),
            storage.add_conversation("assistant", "written during backup"),
        )

        assert isinstance(blob, (bytes, bytearray)) and len(blob) > 0

        # The connection must be usable *immediately* after — no reopen window.
        history = await storage.get_conversation_history(limit=10)
        contents = [m.get("content") for m in history]
        assert "hello before backup" in contents
        assert "written during backup" in contents
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_backup_blob_restores_into_fresh_storage(tmp_path):
    source = await _storage(tmp_path, name="source.db")
    try:
        await source.add_conversation("user", "restore me please")
        blob = await source.create_backup_blob()
    finally:
        await source.close()

    target = await _storage(tmp_path, name="target.db")
    try:
        stats = await target.restore_from_backup_blob(blob)
        assert stats["messages_restored"] >= 1
        contents = [
            m.get("content") for m in await target.get_conversation_history(limit=10)
        ]
        assert "restore me please" in contents
    finally:
        await target.close()


@pytest.mark.asyncio
async def test_backup_runs_off_event_loop(tmp_path, monkeypatch):
    storage = await _storage(tmp_path)
    try:
        await storage.add_conversation("user", "off-loop archive")

        called = {}
        real_to_thread = asyncio.to_thread

        async def _spy(func, *args, **kwargs):
            called["hit"] = True
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", _spy)
        blob = await storage.create_backup_blob()
        assert called.get("hit") is True
        assert len(blob) > 0
    finally:
        await storage.close()
