"""Native encrypted reads keep cryptography off the shared event loop."""

import asyncio
import json
import threading

import pytest

from kestrel_sovereign.security.exceptions import DecryptionError
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.fixture
async def encrypted_conversations(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "unit-test-owned-decryption-key")
    db = await AsyncDatabase.sqlite(str(tmp_path / "encrypted.db"))
    try:
        store = AsyncConversationStore(db, agent_id="did:test:owned-decryption")
        assert store.encryption_enabled
        yield db, store
    finally:
        await db.close()


@pytest.fixture(params=[
    "get_conversation_history", "get_full_history", "search_history",
    "get_full_history_with_ids", "get_messages_by_ids",
    "get_salient_memory_candidates", "get_lexical_memory_candidates",
    "get_excluded_messages", "get_stashed_messages", "get_all_audit_failures",
    "_normalized_search_row", "_resolve_canonical",
])
async def native_reader(encrypted_conversations, request):
    reader = request.param
    db, store = encrypted_conversations
    content, rendered = "actual native worker token", "actual rendered worker token"
    metadata = {
        "get_excluded_messages": {"excluded_from_context": True},
        "get_stashed_messages": {"stashed": True},
        "get_all_audit_failures": {"audit_failure": True},
    }.get(reader, {})
    await store.add_conversation("user", content, metadata=metadata, rendered_content=rendered)
    row = await db.fetchone(
        "SELECT id, role, content, metadata, created_at, rendered_content FROM conversation_history"
    )
    assert row[2] != content and row[5] != rendered
    return db, store, reader, row, content, rendered


async def _read_native(store, reader, row, content):
    if reader == "_resolve_canonical":
        return await store._resolve_canonical(row[0], row[1], json.loads(row[3]), content, row[5])
    if reader == "_normalized_search_row":
        return await store._normalized_search_row(row)
    args = {
        "get_messages_by_ids": ([row[0]],),
        "search_history": ("worker token",),
        "get_lexical_memory_candidates": ("actual " + " ".join(
            "term" + chr(97 + i // 676) + chr(97 + i // 26 % 26) + chr(97 + i % 26)
            for i in range(760)),),
    }.get(reader, ())
    return await getattr(store, reader)(*args)


@pytest.mark.asyncio
async def test_native_readers_decrypt_in_owned_workers(native_reader, monkeypatch):
    _, store, reader, row, content, rendered = native_reader
    loop_thread, threads = threading.get_ident(), []
    decrypt = store._decrypt_with_fallback

    def observed_decrypt(ciphertext, meta):
        threads.append(threading.get_ident())
        return decrypt(ciphertext, meta)

    monkeypatch.setattr(store, "_decrypt_with_fallback", observed_decrypt)
    result = await _read_native(store, reader, row, content)
    if reader == "_resolve_canonical":
        assert result == (content, rendered)
    elif reader == "_normalized_search_row":
        assert result["content"] == content
    else:
        assert len(result) == 1 and result[0]["content"] == content
        if reader == "get_lexical_memory_candidates":
            assert store._last_lexical_bridge_stats["fallback_reason"] == "query_token_budget"
    assert threads, "The real encrypted row was not read"
    assert loop_thread not in threads, f"{reader} decrypted on the event-loop thread"


@pytest.mark.asyncio
@pytest.mark.parametrize("column", ["content", "rendered_content"])
async def test_native_decryption_error_survives_worker_ownership(encrypted_conversations, column, caplog):
    db, store = encrypted_conversations
    await store.add_conversation("user", "actual encrypted value", rendered_content="actual rendered value")
    row_id = await db.fetchval("SELECT id FROM conversation_history")
    assert column in {"content", "rendered_content"}
    await db.execute_commit("UPDATE conversation_history SET " + column + " = ?", ("invalid ciphertext",))
    with pytest.raises(DecryptionError):
        await store.get_messages_by_ids([row_id])
    if column == "rendered_content":
        assert "Failed to decrypt rendered_content" in caplog.text


@pytest.mark.asyncio
async def test_cancelled_read_joins_its_real_decrypt_worker(native_reader, monkeypatch):
    _, store, reader, row, content, _ = native_reader
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    decrypt = store._decrypt_with_fallback

    def held_decrypt(content, metadata):
        entered.set()
        try:
            assert release.wait(5), "Owned decrypt worker was not released"
            return decrypt(content, metadata)
        finally:
            finished.set()

    monkeypatch.setattr(store, "_decrypt_with_fallback", held_decrypt)
    reading = asyncio.create_task(_read_native(store, reader, row, content))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        reading.cancel()
        await asyncio.sleep(0)
        assert not reading.done(), "Read abandoned its active native decrypt worker"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await reading
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(reading, return_exceptions=True)
