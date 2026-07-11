import importlib.util
from pathlib import Path
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.storage.memory_retriever import MemoryRetriever
from kestrel_sovereign.storage.memory_retriever import _is_user_query_echo


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_memory_quality.py"
SPEC = importlib.util.spec_from_file_location("benchmark_memory_quality", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_summarize_separates_retrieval_from_abstention_quality():
    query_results = [
        ({"relevant": ["a"]}, [{"id": "a"}, {"id": "noise"}]),
        ({"relevant": ["b"]}, [{"id": "noise"}]),
        ({"relevant": [], "expect_empty": True}, []),
        ({"relevant": [], "expect_empty": True}, [{"id": "noise"}]),
    ]

    metrics = benchmark.summarize(query_results, limit=5)

    assert metrics.recall_at_k == pytest.approx(0.5)
    assert metrics.precision_at_k == pytest.approx(0.25)
    assert metrics.mean_reciprocal_rank == pytest.approx(0.5)
    assert metrics.top1_accuracy == pytest.approx(0.5)
    assert metrics.abstention_accuracy == pytest.approx(0.5)


def test_rank_query_uses_production_salience_components_but_relevance_gate():
    memories = [
        {"id": "relevant", "text": "the useful fact", "importance": 0.2},
        {"id": "noise", "text": "unrelated", "importance": 1.0},
    ]
    query = {"query": "useful", "relevant": ["relevant"]}

    ranked = benchmark.rank_query(
        retriever=MemoryRetriever(None),
        memories=memories,
        query=query,
        similarities={"relevant": 0.8, "noise": 0.25},
        cosine_floor=0.2,
        min_score=0.0,
        min_relevance=0.1,
        limit=5,
    )

    assert [row["id"] for row in ranked] == ["relevant"]


def test_fixture_ids_and_labels_are_well_formed():
    import json

    suite = json.loads(benchmark.DEFAULT_SUITE.read_text(encoding="utf-8"))
    memory_ids = {memory["id"] for memory in suite["memories"]}
    assert len(memory_ids) == len(suite["memories"])
    assert any(query.get("expect_empty") for query in suite["queries"])
    for query in suite["queries"]:
        assert set(query.get("relevant", [])) <= memory_ids
        assert set(query.get("forbidden", [])) <= memory_ids


def test_partial_lexical_overlap_below_canonical_threshold_is_not_relevance():
    score = MemoryRetriever(None)._score_semantic(
        content="My favorite breakfast is blueberry oatmeal.",
        query="What is my favorite planet?",
        expanded_concepts=[],
    )

    assert score == 0.0


@pytest.mark.asyncio
async def test_general_recall_excludes_explicitly_superseded_memory():
    store = AsyncMock()
    store.embedding_service = None
    store.get_conversation_history.return_value = [
        {
            "id": 1,
            "role": "user",
            "content": "We considered MongoDB for the ledger.",
            "metadata": {"superseded_by": "decision-2"},
        },
        {
            "id": 2,
            "role": "user",
            "content": "We chose PostgreSQL for the ledger.",
            "metadata": {},
        },
    ]

    results = await MemoryRetriever(store).retrieve(
        "ledger", agent_id="test", min_score=0.0, read_only=True
    )

    assert [row["id"] for row in results] == [2]


@pytest.mark.asyncio
async def test_salience_only_reranks_semantically_competitive_candidates():
    store = AsyncMock()
    store.embedding_service = MagicMock()
    store.embedding_service.aembed = AsyncMock(return_value=[1.0, 0.0])
    store.embedding_service.current_profile_id.return_value = "profile"
    store.embedding_service.retrieval_similarity_floor.return_value = 0.0
    store.get_conversation_history.return_value = [
        {
            "id": 1,
            "role": "user",
            "content": "old exact answer",
            "metadata": {"importance": 0.1},
            "created_at": "2020-01-01T00:00:00+00:00",
        },
        {
            "id": 2,
            "role": "assistant",
            "content": "fresh salient distractor",
            "metadata": {"importance": 1.0, "access_count": 100},
            "created_at": "2026-07-11T00:00:00+00:00",
        },
    ]
    retriever = MemoryRetriever(store)
    retriever._semantic_similarities_via_vector_backend = AsyncMock(
        return_value={"1": 0.9, "2": 0.5}
    )

    results = await retriever.retrieve(
        "answer query", agent_id="test", min_score=0.0, read_only=True
    )

    assert [row["id"] for row in results] == [1]


def test_interrogative_echo_is_dropped_without_hiding_user_fact():
    assert _is_user_query_echo(
        "Which stored memory mentions absentneedle? If none, say NONE.",
        "absentneedle",
    )
    assert _is_user_query_echo(
        "Which stored memory mentions absentneedle? If none, say NONE.",
        "absentneedle-7f3c91",
    )
    assert not _is_user_query_echo("My favorite color is blue.", "blue")
