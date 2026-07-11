from time import perf_counter
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_type", "expected_sql"),
    [
        ("sqlite", "json_extract(metadata"),
        ("postgres", "metadata::jsonb"),
    ],
)
async def test_salient_candidate_query_uses_backend_json_dialect(
    backend_type, expected_sql
):
    db = MagicMock()
    db.backend_type = backend_type
    db.fetchall = AsyncMock(return_value=[])
    store = AsyncConversationStore(db, agent_id="did:test:memory")

    assert await store.get_salient_memory_candidates() == []

    sql = db.fetchall.await_args.args[0]
    assert expected_sql in sql
    assert "archived_at IS NULL" in sql


@pytest.mark.asyncio
async def test_lexical_candidates_can_scope_to_rows_outside_active_profile():
    db = MagicMock()
    db.backend_type = "sqlite"
    db.fetchall = AsyncMock(return_value=[])
    store = AsyncConversationStore(db, agent_id="did:test:memory")

    await store.get_lexical_memory_candidates(
        "zirconium", excluded_embedding_profile_id="active-profile"
    )

    sql, params = db.fetchall.await_args.args
    assert "embedding_profile_id IS NULL" in sql
    assert "embedding_profile_id != ?" in sql
    assert params[0:2] == ("did:test:memory", "active-profile")
    assert params[-1] == 1000
    assert "lexical_index_version" in sql


@pytest.mark.asyncio
async def test_blind_index_shortlists_encrypted_exact_match_without_full_scan(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_DATA_KEY", "blind-index-encrypted-test-key")
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    db = await AsyncDatabase.sqlite(str(tmp_path / "encrypted-index.db"))
    try:
        store = AsyncConversationStore(db, agent_id="did:test:encrypted-index")
        await store.add_conversation("user", "routine garden watering")
        await store.add_conversation(
            "user", "My zirconium axolotl is named Quasar-17"
        )

        stored = await db.fetchone(
            "SELECT content FROM conversation_history WHERE id = 2"
        )
        assert stored is not None and "zirconium" not in stored[0]
        digest_rows = await db.fetchall(
            "SELECT token_hash FROM conversation_lexical_tokens"
        )
        assert digest_rows
        assert all("zirconium" not in digest for (digest,) in digest_rows)

        found = await store.get_lexical_memory_candidates("zirconium", limit=5)

        assert [row["id"] for row in found] == [2]
        assert store._last_lexical_bridge_stats["fallback_rows_scanned"] == 0
        health = await store.get_lexical_index_health()
        assert health["indexed_current"] == health["total_live"] == 2
        assert health["coverage"] == 1.0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_partial_blind_index_keeps_unindexed_legacy_fact_recallable(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    db = await AsyncDatabase.sqlite(str(tmp_path / "partial-index.db"))
    try:
        store = AsyncConversationStore(db, agent_id="did:test:partial-index")
        await store.add_conversation("user", "current indexed routine")
        await db.execute(
            "INSERT INTO conversation_history (agent_id, role, content) "
            "VALUES (?, 'user', ?)",
            (store.agent_id, "legacy pelagic octopus fact"),
        )

        found = await store.get_lexical_memory_candidates("pelagic", limit=5)

        assert [row["content"] for row in found] == [
            "legacy pelagic octopus fact"
        ]
        assert store._last_lexical_bridge_stats["fallback_rows_scanned"] == 1
        health = await store.get_lexical_index_health()
        assert health["indexed_current"] == 1
        assert health["unindexed"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_encrypted_backfill_is_resumable_and_eliminates_bridge_scan(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_DATA_KEY", "blind-index-backfill-test-key")
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    db = await AsyncDatabase.sqlite(str(tmp_path / "backfill-index.db"))
    try:
        store = AsyncConversationStore(db, agent_id="did:test:index-backfill")
        for index in range(120):
            content = (
                "rare cobalt observatory fact"
                if index == 3 else f"routine encrypted memory {index}"
            )
            await store.add_conversation("user", content)
        await db.execute("DELETE FROM conversation_lexical_tokens")
        await db.execute(
            "UPDATE conversation_history SET lexical_index_id = NULL, "
            "lexical_index_version = NULL"
        )

        first = await store.backfill_lexical_index(batch_size=17, max_rows=41)
        assert first["indexed"] == 41
        assert first["remaining"] == 79
        second = await store.backfill_lexical_index(batch_size=19)
        assert second["indexed"] == 79
        assert second["remaining"] == 0
        assert second["coverage"] == 1.0

        found = await store.get_lexical_memory_candidates("observatory", limit=5)
        assert [row["content"] for row in found] == [
            "rare cobalt observatory fact"
        ]
        assert store._last_lexical_bridge_stats["fallback_rows_scanned"] == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_index_write_failure_persists_uncovered_recallable_message(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    db = await AsyncDatabase.sqlite(str(tmp_path / "index-write-failure.db"))
    try:
        store = AsyncConversationStore(db, agent_id="did:test:index-failure")
        store._lexical_index.index_message = AsyncMock(
            side_effect=RuntimeError("token table unavailable")
        )

        await store.add_conversation("user", "durable mangosteen fact")

        marker = await db.fetchone(
            "SELECT lexical_index_id, lexical_index_version "
            "FROM conversation_history"
        )
        assert marker == (None, None)
        found = await store.get_lexical_memory_candidates("mangosteen", limit=5)
        assert [row["content"] for row in found] == ["durable mangosteen fact"]
        assert store._last_lexical_bridge_stats["fallback_rows_scanned"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_oversize_query_uses_complete_fallback_instead_of_partial_ranking(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    db = await AsyncDatabase.sqlite(str(tmp_path / "query-budget.db"))
    try:
        store = AsyncConversationStore(db, agent_id="did:test:query-budget")
        query = " ".join(f"term{index}" for index in range(101))
        await store.add_conversation("user", query)

        found = await store.get_lexical_memory_candidates(query, limit=5)

        assert len(found) == 1
        assert store._last_lexical_bridge_stats["fallback_reason"] == (
            "query_token_budget"
        )
        assert store._last_lexical_bridge_stats["fallback_rows_scanned"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_large_corpus_backfill_removes_recall_time_full_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    db = await AsyncDatabase.sqlite(str(tmp_path / "large-index.db"))
    try:
        store = AsyncConversationStore(db, agent_id="did:test:large-index")
        rows = [
            (
                store.agent_id,
                "user",
                "rare zirconium axolotl fact" if index == 17
                else f"routine memory number {index}",
            )
            for index in range(10_000)
        ]
        await db.execute_many(
            "INSERT INTO conversation_history (agent_id, role, content) "
            "VALUES (?, ?, ?)",
            rows,
        )
        before = await store.get_lexical_memory_candidates("zirconium", limit=5)
        assert len(before) == 1
        assert store._last_lexical_bridge_stats["fallback_rows_scanned"] == 10_000

        result = await store.backfill_lexical_index(batch_size=1000)
        started = perf_counter()
        after = await store.get_lexical_memory_candidates("zirconium", limit=5)
        indexed_recall_seconds = perf_counter() - started

        assert result["indexed"] == 10_000
        assert result["coverage"] == 1.0
        assert [row["content"] for row in after] == ["rare zirconium axolotl fact"]
        assert store._last_lexical_bridge_stats["fallback_rows_scanned"] == 0
        assert indexed_recall_seconds < 1.0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_embedding_profile_health_exposes_active_null_and_stale_counts(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    db = await AsyncDatabase.sqlite(str(tmp_path / "profile-health.db"))
    try:
        store = AsyncConversationStore(db, agent_id="did:test:profile-health")
        await db.execute_many(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, embedding_vec, embedding_profile_id) "
            "VALUES (?, 'user', ?, ?, ?)",
            [
                (store.agent_id, "active", b"vector", "active-profile"),
                (store.agent_id, "active missing", None, "active-profile"),
                (store.agent_id, "legacy", None, None),
                (store.agent_id, "stale", b"vector", "old-profile"),
            ],
        )

        health = await store.get_embedding_profile_health("active-profile")

        assert health["total_live"] == 4
        assert health["active_profile_vectors"] == 1
        assert health["active_profile_missing_vector"] == 1
        assert health["null_profile"] == 1
        assert health["other_profile"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_missing_coverage_index_degrades_to_complete_recall(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    db = await AsyncDatabase.sqlite(str(tmp_path / "missing-coverage-index.db"))
    try:
        store = AsyncConversationStore(db, agent_id="did:test:missing-index")
        await store.add_conversation("user", "indexed ordinary fact")
        await db.execute("DROP INDEX idx_conversation_lexical_coverage")
        await db.execute(
            "INSERT INTO conversation_history (agent_id, role, content) "
            "VALUES (?, 'user', ?)",
            (store.agent_id, "legacy exact starfruit fact"),
        )

        found = await store.get_lexical_memory_candidates("starfruit", limit=5)

        assert [row["content"] for row in found] == ["legacy exact starfruit fact"]
        assert store._last_lexical_bridge_stats["fallback_rows_scanned"] == 1
        assert store._lexical_coverage_index_available is False
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_indexed_candidates_use_canonical_sent_form_content(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    db = await AsyncDatabase.sqlite(str(tmp_path / "canonical-index.db"))
    try:
        store = AsyncConversationStore(db, agent_id="did:test:canonical-index")
        rendered = (
            "<retrieved_context>old memory</retrieved_context>\n"
            "<user_input>zirconium is canonical</user_input>"
        )
        await store.add_conversation(
            "user", rendered, metadata={"sent_form": True}
        )

        found = await store.get_lexical_memory_candidates("zirconium", limit=5)

        assert [row["content"] for row in found] == [
            "<user_input>zirconium is canonical</user_input>"
        ]
        assert store._last_lexical_bridge_stats["fallback_rows_scanned"] == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_version_rotation_rebackfills_and_reclaims_obsolete_tokens(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    db = await AsyncDatabase.sqlite(str(tmp_path / "rotated-index.db"))
    try:
        store = AsyncConversationStore(db, agent_id="did:test:rotated-index")
        await store.add_conversation("user", "rotated kumquat fact")
        old_key, old_version = await db.fetchone(
            "SELECT lexical_index_id, lexical_index_version "
            "FROM conversation_history"
        )
        assert old_version == store._lexical_index.version

        store._lexical_index._key = b"rotated-test-key" * 2
        store._lexical_index.version = "v1:keyed:rotated-test"
        result = await store.backfill_lexical_index(batch_size=10)

        marker = await db.fetchone(
            "SELECT lexical_index_id, lexical_index_version "
            "FROM conversation_history"
        )
        orphan = await db.fetchone(
            "SELECT COUNT(*) FROM conversation_lexical_tokens "
            "WHERE lexical_index_id = ?",
            (old_key,),
        )
        found = await store.get_lexical_memory_candidates("kumquat", limit=5)
        assert result["indexed"] == 1
        assert marker[1] == "v1:keyed:rotated-test"
        assert orphan[0] == 0
        assert [row["content"] for row in found] == ["rotated kumquat fact"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_backfill_durable_count_does_not_trust_executemany_return(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    db = await AsyncDatabase.sqlite(str(tmp_path / "backfill-count.db"))
    try:
        store = AsyncConversationStore(db, agent_id="did:test:backfill-count")
        await db.execute_many(
            "INSERT INTO conversation_history (agent_id, role, content) "
            "VALUES (?, 'user', ?)",
            [(store.agent_id, f"legacy {index}") for index in range(3)],
        )
        real_execute_many = db.execute_many

        async def misleading_execute_many(sql, params):
            result = await real_execute_many(sql, params)
            return 999 if sql.startswith("UPDATE conversation_history") else result

        db.execute_many = misleading_execute_many
        result = await store.backfill_lexical_index(batch_size=10)

        assert result["attempted"] == 3
        assert result["indexed"] == 3
        assert result["coverage"] == 1.0
    finally:
        await db.close()
