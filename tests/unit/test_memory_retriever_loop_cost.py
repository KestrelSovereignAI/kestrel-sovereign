"""A long query cannot stall the event loop in memory retrieval (#3745).

One retrieval scored every candidate row in a single synchronous loop that
derived the query's search forms again for each row. A 26k-character message
against ~700 rows held the event loop for 2.4 s, past the 2.0 s Stop owner
lease every agent on the host shares.
"""

import asyncio
import random
import threading
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.storage import memory_answerability, memory_retriever
from kestrel_sovereign.storage.async_conversation_store import (
    AsyncConversationStore,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.memory_retriever import MemoryRetriever

# The acceptance bound: a retrieval may hold the loop for less than this.
_MAX_EVENT_LOOP_BLOCK_SECONDS = 0.25
_LAG_PROBE_SECONDS = 0.01


def _counting(module, name, query, counts, monkeypatch):
    original = getattr(module, name)

    def counted(text, *args, **kwargs):
        if text == query:
            counts[name] = counts.get(name, 0) + 1
        return original(text, *args, **kwargs)

    monkeypatch.setattr(module, name, counted)


@pytest.mark.asyncio
async def test_query_derived_values_are_computed_once_per_retrieve(monkeypatch):
    query = "ferry harbor departure"
    rows = [
        {"id": 1, "role": "user", "content": "The ferry harbor departure is at Pier 9."},
        {"id": 2, "role": "user", "content": "When is the ferry harbor departure?"},
        {"id": 3, "role": "assistant", "content": "Ferry harbor departure moved to Pier 9."},
    ] + [
        {
            "id": row_id,
            "role": "user" if row_id % 2 else "assistant",
            "content": f"Did we plant the tulips in bed {row_id}?",
        }
        for row_id in range(4, 40)
    ]

    class WeakEmbeddingModel:
        def requires_answerability_gate(self):
            return True

        def current_profile_id(self):
            return "profile"

        def retrieval_similarity_floor(self):
            return 0.0

    store = AsyncMock()
    store.embedding_service = WeakEmbeddingModel()
    store.get_conversation_history.return_value = rows
    store.get_salient_memory_candidates.return_value = []
    store.get_lexical_memory_candidates.return_value = []
    # No judge is configured, so the gate degrades to exact lexical evidence:
    # that path checks the query against every competitive candidate too.
    retriever = MemoryRetriever(store, answerability_gate=None)
    retriever._embed_query = AsyncMock(return_value=[1.0, 0.0])
    retriever._semantic_similarities_via_vector_backend = AsyncMock(
        return_value={str(row["id"]): 0.9 for row in rows}
    )

    counts: dict[str, int] = {}
    for module, name in (
        (memory_retriever, "_strip_search_wrappers"),
        (memory_retriever, "_tokenize_for_search"),
        (memory_retriever, "_normalize_for_echo_check"),
        (memory_answerability, "_strip_search_wrappers"),
        (memory_answerability, "_tokenize_for_search"),
    ):
        _counting(module, name, query, counts, monkeypatch)

    results = await retriever.retrieve(
        query, agent_id="agent", min_score=0.0, read_only=True
    )

    assert [row["id"] for row in results] == [1, 3]
    assert retriever.answerability_stats["failures"] == 1
    # The query carries no search wrappers, so its stripped form is the query.
    assert counts == {
        "_strip_search_wrappers": 1,
        "_tokenize_for_search": 1,
        "_normalize_for_echo_check": 1,
    }


def _words(rng, vocabulary, n_chars):
    words, size = [], 0
    while size < n_chars:
        word = rng.choice(vocabulary)
        roll = rng.random()
        if roll < 0.09:
            word += "?"
        elif roll < 0.2:
            word += "."
        words.append(word)
        size += len(word) + 1
    return " ".join(words)[:n_chars]


async def _max_event_loop_block(operation):
    """Run ``operation`` and return its result and the loop's longest block."""
    loop = asyncio.get_running_loop()
    blocks = []
    done = asyncio.Event()

    async def probe():
        while not done.is_set():
            started = loop.time()
            await asyncio.sleep(_LAG_PROBE_SECONDS)
            blocks.append(loop.time() - started - _LAG_PROBE_SECONDS)

    prober = asyncio.create_task(probe())
    await asyncio.sleep(_LAG_PROBE_SECONDS * 3)
    try:
        result = await operation
    finally:
        done.set()
        await prober
    return result, max(blocks)


@pytest.mark.asyncio
async def test_long_query_retrieval_never_blocks_the_event_loop(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_DATA_KEY", "unit-test-retriever-loop-cost-key")
    rng = random.Random(3745)
    vocabulary = sorted({
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz")
                for _ in range(rng.randint(3, 10)))
        for _ in range(6000)
    })
    query_vocabulary = rng.sample(vocabulary, 760)
    # Production's corpus: ~700 live rows and four rows of 1.33M, 0.84M, 0.69M
    # and 0.49M characters. Even rows are user turns, and every row carries
    # questions, so the user-question echo check reads the megabyte rows too.
    megabyte_rows = {101: 1_330_000, 201: 840_000, 300: 690_000, 401: 490_000}

    db = await AsyncDatabase.sqlite(str(tmp_path / "loop-cost.db"))
    try:
        store = AsyncConversationStore(db, agent_id="agent-3745")
        assert store.encryption_enabled
        for index in range(702):
            if index in megabyte_rows:
                content = _words(rng, vocabulary, megabyte_rows[index])
            elif index % 50 == 7:
                content = _words(rng, query_vocabulary, rng.randint(3000, 9000))
            else:
                content = _words(rng, vocabulary, rng.randint(300, 2500))
            await store.add_conversation(
                "user" if index % 2 == 0 else "assistant",
                content,
                session_id=f"session-{index // 40}",
            )
        query = _words(rng, query_vocabulary, 26_238)
        retriever = MemoryRetriever(store, answerability_enabled=False)
        loop_thread = threading.get_ident()
        decrypt_threads = []
        decrypt = store._decrypt_with_fallback

        def observed_decrypt(content, metadata):
            decrypt_threads.append(threading.get_ident())
            return decrypt(content, metadata)

        monkeypatch.setattr(store, "_decrypt_with_fallback", observed_decrypt)

        results, block = await _max_event_loop_block(
            retriever.retrieve(query, agent_id="agent-3745", read_only=True)
        )

        # The expensive path ran: the query is past the blind index's token
        # budget, so every live row was decrypted and lexically ranked.
        bridge = store._last_lexical_bridge_stats
        assert bridge["fallback_reason"] == "query_token_budget"
        assert bridge["fallback_rows_scanned"] == 702
        assert results
        assert decrypt_threads
        assert loop_thread not in decrypt_threads, (
            "Encrypted retrieval rows were decrypted on the event-loop thread"
        )
        assert block < _MAX_EVENT_LOOP_BLOCK_SECONDS, (
            f"retrieval held the event loop for {block:.3f} s"
        )
    finally:
        await db.close()
