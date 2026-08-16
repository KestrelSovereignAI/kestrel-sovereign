import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.memory_answerability import (
    AnswerabilityCandidate,
    AnswerabilityDecision,
    LLMAnswerabilityGate,
)
from kestrel_sovereign.storage.memory_retriever import MemoryRetriever
from kestrel_sovereign.storage import memory_system as memory_system_module


@pytest.mark.asyncio
async def test_gate_batches_candidates_and_maps_only_known_opaque_ids():
    service = MagicMock()
    service.generate = AsyncMock(
        return_value=json.dumps({"answerable_ids": ["c1"]})
    )
    gate = LLMAnswerabilityGate(
        service, force_local_only_provider=lambda: True
    )

    decision = await gate.filter(
        "What is my favorite planet?",
        [
            AnswerabilityCandidate("breakfast", "My favorite breakfast is oats."),
            AnswerabilityCandidate("planet", "My favorite planet is Saturn."),
        ],
    )

    assert decision.completed is True
    assert decision.answerable_ids == {"planet"}
    kwargs = service.generate.await_args.kwargs
    assert kwargs["force_local_only"] is True
    assert kwargs["model_override"] is None
    assert "favorite breakfast" in kwargs["user_prompt"]


@pytest.mark.asyncio
async def test_unknown_privacy_state_defaults_to_local_only():
    service = MagicMock(spec=["generate"])
    service.generate = AsyncMock(return_value='{"answerable_ids":[]}')

    await LLMAnswerabilityGate(service).filter(
        "question", [AnswerabilityCandidate("1", "candidate")]
    )

    assert service.generate.await_args.kwargs["force_local_only"] is True


@pytest.mark.asyncio
async def test_gate_rejects_unknown_ids_and_fails_closed():
    service = MagicMock()
    service.generate = AsyncMock(return_value='{"answerable_ids":["memory-42"]}')

    decision = await LLMAnswerabilityGate(service).filter(
        "question", [AnswerabilityCandidate("42", "candidate")]
    )

    assert decision.completed is False
    assert decision.answerable_ids == set()
    assert decision.reason == "invalid_response"


@pytest.mark.asyncio
async def test_gate_timeout_is_bounded_and_fail_closed():
    async def hangs(**_kwargs):
        await asyncio.Event().wait()

    service = MagicMock()
    service.generate = hangs
    decision = await LLMAnswerabilityGate(
        service, timeout_seconds=0.01
    ).filter("question", [AnswerabilityCandidate("1", "candidate")])

    assert decision.completed is False
    assert decision.reason.startswith("timeout:")
    assert decision.latency_ms < 250


@pytest.mark.asyncio
async def test_retriever_weak_model_gate_removes_topical_non_answer():
    store = AsyncMock()
    store.embedding_service = type(
        "EmbeddingStub", (), {"requires_answerability_gate": lambda self: True}
    )()
    gate = MagicMock()
    gate.filter = AsyncMock()
    gate.filter.return_value = AnswerabilityDecision(
        frozenset({"2"}), True, 4.0
    )
    retriever = MemoryRetriever(store, answerability_gate=gate)
    scored = [
        ({"id": 1, "content": "My favorite breakfast is oats."}, 0.5, {"semantic": 0.8}),
        ({"id": 2, "content": "My favorite planet is Saturn."}, 0.4, {"semantic": 0.75}),
    ]

    kept = await retriever._filter_answerable("favorite planet", scored)

    assert [item[0]["id"] for item in kept] == [2]
    assert retriever.answerability_stats["failures"] == 0


@pytest.mark.asyncio
async def test_global_kill_switch_preserves_semantic_recall_without_judge():
    class EmbeddingStub:
        def requires_answerability_gate(self):
            return True

        def current_profile_id(self):
            return "profile"

        def retrieval_similarity_floor(self):
            return 0.0

    store = AsyncMock()
    store.embedding_service = EmbeddingStub()
    store.get_conversation_history.return_value = [
        {"id": 1, "role": "user", "content": "I love the color blue", "metadata": {}}
    ]
    store.get_salient_memory_candidates.return_value = []
    store.get_lexical_memory_candidates.return_value = []
    gate = MagicMock()
    gate.filter = AsyncMock(side_effect=AssertionError("disabled gate was called"))
    retriever = MemoryRetriever(
        store, answerability_gate=gate, answerability_enabled=False
    )
    retriever._embed_query = AsyncMock(return_value=[1.0, 0.0])
    retriever._semantic_similarities_via_vector_backend = AsyncMock(
        return_value={"1": 0.9}
    )

    results = await retriever.retrieve(
        "favorite color", "test", min_score=0.0, read_only=True
    )

    assert [row["id"] for row in results] == [1]
    gate.filter.assert_not_awaited()


@pytest.mark.asyncio
async def test_retriever_failed_judge_preserves_only_strong_lexical_evidence():
    store = AsyncMock()
    gate = MagicMock()
    gate.filter = AsyncMock(
        return_value=AnswerabilityDecision(frozenset(), False, 10.0, "timeout")
    )
    retriever = MemoryRetriever(store, answerability_gate=gate)
    scored = [
        ({"id": 1, "content": "favorite breakfast blueberry oats"}, 0.5, {"semantic": 0.8}),
        ({"id": 2, "content": "favorite planet Saturn"}, 0.4, {"semantic": 0.75}),
    ]

    kept = await retriever._filter_answerable("favorite planet", scored)

    assert [item[0]["id"] for item in kept] == [2]
    assert retriever.answerability_stats["failures"] == 1


@pytest.mark.asyncio
async def test_lexical_only_retrieval_bypasses_answerability_judge():
    class EmbeddingStub:
        async def aembed_query(self, _query, *, instruction):
            return None

        def requires_answerability_gate(self):
            return True

        def retrieval_similarity_floor(self):
            return 0.0

    store = AsyncMock()
    store.embedding_service = EmbeddingStub()
    store.get_conversation_history.return_value = [
        {"id": 1, "role": "user", "content": "My planet is Saturn", "metadata": {}}
    ]
    store.get_salient_memory_candidates.return_value = []
    store.get_lexical_memory_candidates.return_value = []
    gate = MagicMock()
    gate.filter = AsyncMock(side_effect=AssertionError("lexical path called judge"))
    retriever = MemoryRetriever(store, answerability_gate=gate)

    results = await retriever.retrieve(
        "planet Saturn", "test", min_score=0.0, min_relevance=0.0, read_only=True
    )

    assert [row["id"] for row in results] == [1]
    gate.filter.assert_not_awaited()


def test_answerability_settings_expose_kill_switch_timeout_and_model(monkeypatch):
    monkeypatch.setattr(
        memory_system_module,
        "load_section",
        lambda _name: {
            "memory_answerability_gate": False,
            "memory_answerability_timeout_seconds": 3.5,
            "memory_answerability_model": "qualified-judge",
        },
    )

    assert memory_system_module._answerability_settings() == (
        False,
        3.5,
        "qualified-judge",
    )


@pytest.mark.parametrize(
    "config",
    [
        {"memory_answerability_gate": "false"},
        {"memory_answerability_timeout_seconds": 0},
        {"memory_answerability_model": ""},
    ],
)
def test_answerability_settings_reject_invalid_values(monkeypatch, config):
    monkeypatch.setattr(
        memory_system_module, "load_section", lambda _name: config
    )

    with pytest.raises(ValueError):
        memory_system_module._answerability_settings()


def test_answerability_settings_validate_gate_before_timeout(monkeypatch):
    """Consolidation timeout validation must not change legacy error order."""
    monkeypatch.setattr(
        memory_system_module,
        "load_section",
        lambda _name: {
            "memory_answerability_gate": "false",
            "memory_answerability_timeout_seconds": 0,
        },
    )

    with pytest.raises(
        ValueError,
        match="retrieval.memory_answerability_gate must be boolean",
    ):
        memory_system_module._answerability_settings()


def test_answerability_settings_preserve_infinite_timeout(monkeypatch):
    """The new consolidation bound must not tighten answerability config."""
    monkeypatch.setattr(
        memory_system_module,
        "load_section",
        lambda _name: {"memory_answerability_timeout_seconds": float("inf")},
    )

    assert memory_system_module._answerability_settings() == (
        True,
        float("inf"),
        None,
    )
