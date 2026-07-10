from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore


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
    assert params == ("did:test:memory", "active-profile", 1000)
