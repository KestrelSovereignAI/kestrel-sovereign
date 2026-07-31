"""Exact, content-independent retraction of semantic-recall derivatives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kestrel_sovereign.storage.async_conversation_store import (
    AsyncConversationStore,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator


@pytest.fixture
async def storage(tmp_path: Path, monkeypatch):
    """A real SQLite store without embedding-provider I/O."""
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    result = await AsyncStorage.create_sqlite(str(tmp_path / "derivatives.db"))
    result.agent_id = "semantic-derivative-agent"
    result.conversation = AsyncConversationStore(
        result.db, agent_id=result.agent_id
    )
    try:
        yield result
    finally:
        await result.close()


@pytest.mark.asyncio
async def test_exclusion_is_exact_and_scrubbing_keeps_artifact_hidden(storage):
    """Same text is not authority: only exact lineage may be retracted."""
    linked_metadata = {
        "semantic_recall_dependencies": [
            {"assertion_id": "assertion-a", "revision_id": "revision-a"}
        ]
    }
    # The identical text deliberately proves there is no content/string match.
    # This lower-level test exercises selector/scrub mechanics.  Production
    # semantic derivatives enter via AsyncStorage's lifecycle fence, which
    # deliberately refuses unbound test storage; seed the conversation store
    # directly so the test can inspect an otherwise-visible historical row.
    await storage.conversation.add_conversation(
        "assistant", "kite-2748-region-7f3b", metadata=linked_metadata
    )
    await storage.add_conversation("assistant", "kite-2748-region-7f3b")
    rows = await storage.conversation.get_full_history_with_ids(
        include_excluded=True
    )
    linked_row = next(
        row
        for row in rows
        if row["metadata"].get("semantic_recall_dependencies")
    )

    await storage.db.execute(
        "INSERT INTO memory_episodes "
        "(id, agent_id, title, summary, key_message_ids, excluded_from_context) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (
            "episode-linked",
            storage.agent_id,
            "linked episode",
            "derived summary",
            json.dumps([str(linked_row["id"])]),
        ),
    )

    async with storage.transaction():
        message_ids = await storage._exclude_semantic_recall_dependencies(
            assertion_ids=("assertion-a",)
        )
        episode_ids = await storage._exclude_memory_episodes_for_key_message_ids(
            message_ids
        )

    assert message_ids == (linked_row["id"],)
    assert episode_ids == ("episode-linked",)
    # Normal history exclusion and the central memory-retriever filter both
    # consume this sticky marker; the same-text unlinked row remains visible.
    assert [row["id"] for row in await storage.get_conversation_history()] == [
        row["id"] for row in rows if row["id"] != linked_row["id"]
    ]
    assert await storage.db.fetchone(
        "SELECT excluded_from_context FROM memory_episodes WHERE id = ?",
        ("episode-linked",),
    ) == (1,)

    assert await storage._scrub_semantic_recall_dependencies(
        assertion_ids=("assertion-a",)
    ) == 1
    hidden = next(
        row
        for row in await storage.conversation.get_full_history_with_ids(
            include_excluded=True
        )
        if row["id"] == linked_row["id"]
    )
    assert hidden["metadata"]["excluded_from_context"] is True
    assert hidden["metadata"]["semantic_recall_dependencies"] == []


@pytest.mark.asyncio
async def test_sleep_consolidation_never_recreates_episode_from_excluded_rows(storage):
    """The sleep path treats hidden derivatives as non-existent source data."""
    for index in range(3):
        await storage.conversation.add_conversation(
            "assistant",
            f"forgotten semantic answer {index}",
            metadata={
                "excluded_from_context": True,
                "semantic_recall_dependencies": [
                    {
                        "assertion_id": "forgotten-assertion",
                        "revision_id": "forgotten-revision",
                    }
                ],
            },
        )

    consolidator = MemoryConsolidator(storage.db, storage.agent_id)
    report = await consolidator.run_consolidation()

    assert report["episodes_created"] == 0
    assert report["patterns_found"] == 0
    assert await consolidator.create_session_episode(force=True) is None
    assert await storage.db.fetchval(
        "SELECT COUNT(*) FROM memory_episodes WHERE agent_id = ?",
        (storage.agent_id,),
    ) == 0


@pytest.mark.asyncio
async def test_prepared_derivative_drops_retrieval_indexes_when_fence_excludes_it(storage):
    """Pre-fence lexical work cannot linger after a stale recall is hidden."""
    prepared = await storage.conversation._prepare_conversation_write(  # noqa: SLF001 - fence preparation contract
        "assistant",
        "stale semantic result",
        {
            "semantic_recall_dependencies": [
                {"assertion_id": "stale", "revision_id": "stale-revision"}
            ]
        },
        None,
        None,
        None,
        None,
    )
    lexical_index_id = prepared.lexical_index_id

    await storage.conversation._exclude_prepared_conversation_from_retrieval(  # noqa: SLF001 - fence preparation contract
        prepared
    )

    assert prepared.lexical_index_id is None
    assert prepared.embedding is None
    if lexical_index_id is not None:
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_lexical_tokens "
            "WHERE agent_id = ? AND lexical_index_id = ?",
            (storage.agent_id, lexical_index_id),
        ) == 0


@pytest.mark.asyncio
async def test_prepared_derivative_cleans_token_prework_when_final_insert_fails(
    storage,
    monkeypatch,
):
    """A final fenced INSERT error cannot leave token-first residue behind."""
    prepared = await storage.conversation._prepare_conversation_write(  # noqa: SLF001 - fence preparation contract
        "assistant",
        "failed semantic result",
        {
            "semantic_recall_dependencies": [
                {"assertion_id": "failed", "revision_id": "failed-revision"}
            ]
        },
        None,
        None,
        None,
        None,
    )
    lexical_index_id = prepared.lexical_index_id

    async def fail_insert(**_kwargs):
        raise RuntimeError("forced final insert failure")

    monkeypatch.setattr(storage.conversation, "_insert_message", fail_insert)
    with pytest.raises(RuntimeError, match="forced final insert failure"):
        await storage.conversation._persist_prepared_conversation(prepared)  # noqa: SLF001 - fence preparation contract

    if lexical_index_id is not None:
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_lexical_tokens "
            "WHERE agent_id = ? AND lexical_index_id = ?",
            (storage.agent_id, lexical_index_id),
        ) == 0
