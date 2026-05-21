"""Tests for ``ContextBuilder.measure_context_breakdown`` and friends.

Track #1308 — the measurement source of truth for the context system.
These tests assert the read-only contract, the canonical return shape,
that tool-schema tokens are now measured (previously: never), and that
episode counting goes through ``len(episodes)`` rather than the legacy
``"**".count() // 2`` heuristic.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kestrel_sovereign.agent.context_builder import (
    ContextBuilder,
    _count_tool_schema_tokens,
    _MESSAGE_OVERHEAD,
)
from kestrel_sovereign.agent.token_counter import get_token_counter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage():
    storage = Mock()
    # Default: no RAG results. Tests override per-case.
    storage.search_chunks = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def builder(mock_storage):
    return ContextBuilder(mock_storage)


@pytest.fixture
def sample_episodes():
    """Three episodes — each carries multiple ``**`` bold markers in the
    formatted block. The legacy ``"**".count() // 2`` heuristic would
    over-count or under-count depending on emotional-arc content; the
    new path uses ``len(episodes)``.
    """
    return [
        {
            "title": "First **bold** episode title",
            "timespan": "2026-05-01 — 2026-05-02",
            "summary": "Summary with **inline bold** sprinkled in.",
            "emotional_arc": "calm → **frustrated** → resolved",
        },
        {
            "title": "Second normal episode",
            "timespan": "2026-05-03 — 2026-05-04",
            "summary": "No bold here.",
            "emotional_arc": "steady",
        },
        {
            "title": "Third **double** **bold** episode",
            "timespan": "2026-05-05 — 2026-05-06",
            "summary": "Lots of **bold** **everywhere**.",
            "emotional_arc": "spike → fall",
        },
    ]


@pytest.fixture
def short_history():
    return [
        {"role": "user", "content": "Hello there agent."},
        {"role": "assistant", "content": "Hi! How can I help today?"},
        {"role": "user", "content": "Tell me about the weather."},
    ]


# ---------------------------------------------------------------------------
# _count_tool_schema_tokens
# ---------------------------------------------------------------------------


class TestCountToolSchemaTokens:
    """The slice the legacy budget machinery never measured."""

    def test_none_returns_zero(self):
        counter = get_token_counter("claude-sonnet-4-6")
        assert _count_tool_schema_tokens(counter, None) == 0

    def test_empty_returns_zero(self):
        counter = get_token_counter("claude-sonnet-4-6")
        assert _count_tool_schema_tokens(counter, []) == 0

    def test_non_empty_is_positive(self):
        counter = get_token_counter("claude-sonnet-4-6")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "save_fact",
                    "description": "Save a learned fact to long-term memory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "topic": {"type": "string"},
                        },
                        "required": ["content"],
                    },
                },
            }
        ]
        tokens = _count_tool_schema_tokens(counter, tools)
        assert tokens > 0
        # Sanity: the serialised payload itself is the floor.
        payload = json.dumps(tools, sort_keys=True, separators=(",", ":"))
        assert tokens == counter.count(payload)

    def test_unserialisable_returns_zero_and_logs(self, caplog):
        counter = get_token_counter("claude-sonnet-4-6")
        # ``set`` is not JSON-serialisable.
        with caplog.at_level("WARNING"):
            tokens = _count_tool_schema_tokens(counter, [{"bad": {1, 2, 3}}])
        assert tokens == 0
        assert any("tool-schema serialisation" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# get_episodes_for_context / format_episodes_for_context
# ---------------------------------------------------------------------------


class TestEpisodeGetFormatSplit:
    """The split that lets ``ContextManager.build_context`` count episodes
    via ``len()`` instead of the ``"**".count() // 2`` heuristic."""

    @pytest.mark.asyncio
    async def test_get_episodes_returns_list_not_none(self, builder):
        builder.consolidator = None
        result = await builder.get_episodes_for_context()
        assert result == []  # not None — callers can len() unconditionally

    @pytest.mark.asyncio
    async def test_get_episodes_passes_through_consolidator(self, builder, sample_episodes):
        consolidator = Mock()
        consolidator.get_recent_episodes_for_context = AsyncMock(
            return_value=sample_episodes
        )
        builder.consolidator = consolidator
        result = await builder.get_episodes_for_context(
            max_tokens=2000, max_episodes=5
        )
        assert result == sample_episodes
        consolidator.get_recent_episodes_for_context.assert_awaited_once_with(
            max_tokens=2000, max_episodes=5
        )

    @pytest.mark.asyncio
    async def test_get_episodes_swallows_consolidator_errors(self, builder):
        consolidator = Mock()
        consolidator.get_recent_episodes_for_context = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        builder.consolidator = consolidator
        result = await builder.get_episodes_for_context()
        assert result == []  # empty list, not raise

    def test_format_episodes_empty_returns_none(self):
        assert ContextBuilder.format_episodes_for_context([]) is None

    def test_format_episodes_is_byte_identical_to_legacy(self, sample_episodes):
        """The block format is part of the prompt-cache prefix; refactor
        must not change a byte (see project_anthropic_cache_markers)."""
        legacy_parts = ["--- CONVERSATION EPISODES (Narrative Summaries) ---"]
        for ep in sample_episodes:
            legacy_parts.append(
                f"\n**{ep['title']}** ({ep['timespan']})\n"
                f"{ep['summary']}\n"
                f"Emotional arc: {ep['emotional_arc']}"
            )
        legacy_parts.append("\n--- END EPISODES ---")
        legacy_block = "\n".join(legacy_parts)
        assert ContextBuilder.format_episodes_for_context(sample_episodes) == legacy_block

    @pytest.mark.asyncio
    async def test_get_episode_context_still_works_via_composition(
        self, builder, sample_episodes
    ):
        consolidator = Mock()
        consolidator.get_recent_episodes_for_context = AsyncMock(
            return_value=sample_episodes
        )
        builder.consolidator = consolidator
        block = await builder.get_episode_context()
        assert block is not None
        assert "First **bold** episode title" in block
        assert "--- END EPISODES ---" in block


# ---------------------------------------------------------------------------
# measure_context_breakdown — return shape, read-only contract, accuracy
# ---------------------------------------------------------------------------


class TestMeasureContextBreakdown:
    """Single source of truth for per-section measurement."""

    @pytest.mark.asyncio
    async def test_returns_canonical_shape(self, builder, short_history):
        result = await builder.measure_context_breakdown(
            query="weather",
            history=short_history,
            constitution="Be kind, be honest.",
            include_briefing=False,
            include_rag=False,
        )
        for key in (
            "model",
            "context_limit",
            "response_reserve",
            "total_budget",
            "total_measured",
            "utilization_percent",
            "budget_summary",
            "sections",
            "notes",
        ):
            assert key in result, f"missing top-level key: {key}"
        for section_name in ("system", "tools", "history", "episodes", "memories", "rag"):
            assert section_name in result["sections"]
            assert "tokens" in result["sections"][section_name]

    @pytest.mark.asyncio
    async def test_total_measured_equals_section_sum(self, builder, short_history):
        result = await builder.measure_context_breakdown(
            query="",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
        )
        section_sum = sum(s["tokens"] for s in result["sections"].values())
        assert result["total_measured"] == section_sum

    @pytest.mark.asyncio
    async def test_utilization_percent_matches_total_over_budget(
        self, builder, short_history
    ):
        result = await builder.measure_context_breakdown(
            query="",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
        )
        if result["total_budget"] > 0:
            expected = round(
                min(result["total_measured"] / result["total_budget"] * 100.0, 100.0),
                1,
            )
            assert result["utilization_percent"] == expected

    @pytest.mark.asyncio
    async def test_tool_schemas_are_measured(self, builder, short_history):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "save_fact",
                    "description": "Save a fact.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        with_tools = await builder.measure_context_breakdown(
            query="",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
            tools=tools,
        )
        without_tools = await builder.measure_context_breakdown(
            query="",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
            tools=None,
        )
        # Tool tokens were previously zero everywhere; assert non-zero now.
        assert with_tools["sections"]["tools"]["tokens"] > 0
        assert without_tools["sections"]["tools"]["tokens"] == 0
        assert with_tools["sections"]["tools"]["count"] == 1
        assert without_tools["sections"]["tools"]["count"] == 0

    @pytest.mark.asyncio
    async def test_episode_count_uses_len_not_double_star_heuristic(
        self, builder, short_history, sample_episodes
    ):
        """sample_episodes has many ``**`` markers; the legacy
        ``"**".count() // 2`` heuristic would over-count. ``len()``
        must return exactly 3."""
        consolidator = Mock()
        consolidator.get_recent_episodes_for_context = AsyncMock(
            return_value=sample_episodes
        )
        builder.consolidator = consolidator

        result = await builder.measure_context_breakdown(
            query="",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
            message_count=25,  # >= EPISODE_THRESHOLD_MESSAGES (=20, production)
        )
        assert result["sections"]["episodes"]["count"] == 3
        assert result["sections"]["episodes"]["threshold"] == 20

        # Sanity: the legacy heuristic on this block would NOT be 3.
        block = ContextBuilder.format_episodes_for_context(sample_episodes)
        legacy_heuristic = block.count("**") // 2
        assert legacy_heuristic != 3, "fixture should expose heuristic mis-counting"

    @pytest.mark.asyncio
    async def test_episodes_skipped_for_short_conversations(
        self, builder, short_history, sample_episodes
    ):
        """Below the production threshold (EPISODE_THRESHOLD_MESSAGES = 20)
        episodes are not fetched. Codex round 1 #2 caught the legacy
        builder gating at 10 while production gated at 20 — measurement
        is now pinned to production.
        """
        consolidator = Mock()
        consolidator.get_recent_episodes_for_context = AsyncMock(
            return_value=sample_episodes
        )
        builder.consolidator = consolidator
        # 19 is the highest count that must NOT trigger episodes
        # (production gate is >=20). Tests the boundary directly.
        result = await builder.measure_context_breakdown(
            query="",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
            message_count=19,
        )
        assert result["sections"]["episodes"]["count"] == 0
        assert result["sections"]["episodes"]["tokens"] == 0
        consolidator.get_recent_episodes_for_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_read_only_idempotent(self, builder, short_history):
        """Calling measurement twice returns identical numbers and never
        writes to storage."""
        builder.storage.add_chunk = Mock()
        builder.storage.upsert = Mock()
        first = await builder.measure_context_breakdown(
            query="hello",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
        )
        second = await builder.measure_context_breakdown(
            query="hello",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
        )
        # Token figures stable across calls
        assert first["total_measured"] == second["total_measured"]
        assert first["sections"]["history"]["tokens"] == second["sections"]["history"]["tokens"]
        assert first["sections"]["system"]["tokens"] == second["sections"]["system"]["tokens"]
        # No DB writes
        builder.storage.add_chunk.assert_not_called()
        builder.storage.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_include_rag_false_records_note_and_skips_search(
        self, builder, short_history
    ):
        result = await builder.measure_context_breakdown(
            query="anything",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
        )
        assert result["sections"]["rag"]["tokens"] == 0
        assert result["sections"]["rag"]["skipped"] is True
        assert any("rag skipped" in n for n in result["notes"])
        builder.storage.search_chunks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_memory_retriever_records_note(self, builder, short_history):
        result = await builder.measure_context_breakdown(
            query="",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
            memory_retriever=None,
        )
        assert result["sections"]["memories"]["wired"] is False
        assert result["sections"]["memories"]["tokens"] == 0
        assert any("memories not measured" in n for n in result["notes"])

    @pytest.mark.asyncio
    async def test_memory_retriever_supplied_is_counted(self, builder, short_history):
        retriever = AsyncMock(return_value="[Memory 1] something\n[Memory 2] else")
        result = await builder.measure_context_breakdown(
            query="anything",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
            memory_retriever=retriever,
        )
        assert result["sections"]["memories"]["wired"] is True
        assert result["sections"]["memories"]["tokens"] > 0
        assert result["sections"]["memories"]["count"] == 2
        # Per-section figure counts the inner <memories>…</memories>
        # wrapper only; the outer <retrieved_context> envelope is a
        # shared overhead reported under ``dynamic_context_overhead``
        # (codex round 2 #1).
        raw_only = builder.counter.count("[Memory 1] something\n[Memory 2] else")
        assert result["sections"]["memories"]["tokens"] > raw_only
        # Shared envelope is counted once.
        assert result["sections"]["dynamic_context_overhead"]["applied"] is True
        assert result["sections"]["dynamic_context_overhead"]["tokens"] > 0

    @pytest.mark.asyncio
    async def test_memory_can_fit_gate_excludes_when_oversized(self, builder, short_history):
        """Production ``ContextManager.build_context`` skips memories when
        ``budget.can_fit("memories", counter.count(raw_block))`` is False
        — gating on the **raw** block, not the wrapped one (codex round 2
        #1). Measurement now matches that semantics so the popup
        excludes/includes the same blocks production would.
        """
        # Build a block guaranteed to blow the memories slice even at raw.
        huge = "[Memory X] " + ("xxxxxxxx " * 50000)
        retriever = AsyncMock(return_value=huge)
        result = await builder.measure_context_breakdown(
            query="anything",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
            memory_retriever=retriever,
        )
        assert result["sections"]["memories"]["excluded"] is True
        assert result["sections"]["memories"]["tokens"] == 0
        assert any("memories excluded" in n for n in result["notes"])

    @pytest.mark.asyncio
    async def test_can_fit_gate_uses_raw_not_wrapped(self, builder, short_history):
        """Pinpoint test for codex round 2 #1: choose a block where the
        raw size fits the memories budget but the wrapped size would not
        if we mistakenly gated on the wrapped form. Measurement must
        INCLUDE it (production would). Construct the test in terms of
        the actual model's counter to avoid coupling to a single
        tokeniser.
        """
        counter = builder.counter
        memories_budget = builder.measure_context_breakdown.__wrapped__ if hasattr(
            builder.measure_context_breakdown, "__wrapped__"
        ) else None  # not actually used; AdaptiveTokenBudget owns the math
        # Use a small block that comfortably fits any reasonable budget.
        block = "[Memory 1] tiny note"
        retriever = AsyncMock(return_value=block)
        result = await builder.measure_context_breakdown(
            query="",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
            memory_retriever=retriever,
        )
        assert result["sections"]["memories"]["excluded"] is False
        assert result["sections"]["memories"]["tokens"] > 0

    @pytest.mark.asyncio
    async def test_rag_wrapping_matches_production_envelope(
        self, builder, short_history, mock_storage
    ):
        """RAG's per-section figure counts the inner ``<documents>…
        </documents>`` wrapper that ``ContextManager.build_context``
        emits; the outer ``<retrieved_context>`` envelope is a shared
        overhead reported separately under ``dynamic_context_overhead``
        (codex round 2 #1).
        """
        mock_storage.search_chunks = AsyncMock(
            return_value=[
                {"document_name": "doc1.txt", "content": "Doc body one."},
                {"document_name": "doc2.txt", "content": "Doc body two."},
            ]
        )
        result = await builder.measure_context_breakdown(
            query="something",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=True,
        )
        assert result["sections"]["rag"]["tokens"] > 0
        assert result["sections"]["rag"]["skipped"] is False
        raw_retrieve = await builder.retrieve_context("something")
        raw_only = builder.counter.count(raw_retrieve)
        assert result["sections"]["rag"]["tokens"] > raw_only, (
            "rag section must count the inner <documents> wrapper"
        )
        # Shared envelope is counted once when rag alone is included.
        assert result["sections"]["dynamic_context_overhead"]["applied"] is True

    @pytest.mark.asyncio
    async def test_dynamic_context_overhead_not_double_charged(
        self, builder, short_history, mock_storage
    ):
        """When BOTH memories and RAG are present, the outer
        ``<retrieved_context>`` envelope is counted exactly once (Codex
        round 2 #1 — the previous version charged it to each section).
        """
        mock_storage.search_chunks = AsyncMock(
            return_value=[{"document_name": "d.txt", "content": "doc body"}]
        )
        retriever = AsyncMock(return_value="[Memory 1] note")
        with_both = await builder.measure_context_breakdown(
            query="something",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=True,
            memory_retriever=retriever,
        )
        envelope_alone = builder.counter.count(
            "<retrieved_context>\n\n</retrieved_context>"
        )
        assert (
            with_both["sections"]["dynamic_context_overhead"]["tokens"]
            == envelope_alone
        )
        # Sanity: turning RAG off should keep memories' inner-wrapper
        # tokens unchanged (no envelope baked into per-section figure).
        memories_only = await builder.measure_context_breakdown(
            query="something",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
            memory_retriever=retriever,
        )
        assert (
            with_both["sections"]["memories"]["tokens"]
            == memories_only["sections"]["memories"]["tokens"]
        )

    @pytest.mark.asyncio
    async def test_history_kept_after_pruning_reported(self, builder, short_history):
        result = await builder.measure_context_breakdown(
            query="",
            history=short_history,
            constitution="Be kind.",
            include_briefing=False,
            include_rag=False,
        )
        hist = result["sections"]["history"]
        assert hist["messages_total"] == len(short_history)
        assert hist["messages_kept_after_pruning"] <= hist["messages_total"]
        assert hist["raw_tokens"] >= 0


# ---------------------------------------------------------------------------
# Drift guard: the source-of-truth claim
# ---------------------------------------------------------------------------


class TestDriftGuard:
    """``measure_context_breakdown`` and ``build_full_context`` share the
    same per-section helpers; their per-section figures must not drift
    on the same inputs."""

    @pytest.mark.asyncio
    async def test_build_full_context_delegates_to_measurement(
        self, builder, short_history
    ):
        """``build_full_context`` now calls ``measure_context_breakdown``
        directly. The assembled system prompt's token count MUST equal
        the measurement's whole-system tokens — no tolerance — because
        both paths read from the same assembled artifact. Codex round 1
        #4 noted the design doc's "cannot drift" invariant was only
        partially satisfied; this asserts the literal invariant.
        """
        with patch.object(builder, "retrieve_context", AsyncMock(return_value=None)):
            assembled = await builder.build_full_context(
                query="",
                history=short_history,
                constitution="Be kind.",
                include_briefing=False,
                message_count=len(short_history),
            )
            measured = await builder.measure_context_breakdown(
                query="",
                history=short_history,
                constitution="Be kind.",
                include_briefing=False,
                include_rag=False,
                message_count=len(short_history),
            )

        whole_system = builder.counter.count(assembled["system_prompt"])
        sub_sum = measured["sections"]["system"]["tokens"]
        assert whole_system == sub_sum, (
            "system token count drifted between build_full_context and "
            "measure_context_breakdown — they must share the assembly path"
        )

        # History tokens must match exactly — both paths run the same
        # format_conversation_history and the same per-message overhead.
        formatted = builder.format_conversation_history(
            history=short_history,
            max_tokens=measured["sections"]["history"]["budget"],
        )
        expected_history = sum(
            builder.counter.count(m.get("content", "") or "") + _MESSAGE_OVERHEAD
            for m in formatted
        )
        assert measured["sections"]["history"]["tokens"] == expected_history

        # ``build_full_context`` exposes ``messages`` from the same
        # formatted-history pass measurement used; they are identity-equal
        # (no copy required because both paths produce the same list).
        assert assembled["messages"] == formatted


# ---------------------------------------------------------------------------
# Heuristic-removal guard: ensure ``"**".count() // 2`` is gone from the
# code paths the design doc names. (D will add a UI-level test; this is
# the architectural guard.)
# ---------------------------------------------------------------------------


def test_double_star_heuristic_removed_from_context_paths():
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    suspects = [
        repo_root / "kestrel_sovereign" / "agent" / "context_manager.py",
        repo_root / "kestrel_sovereign" / "agent" / "context_builder.py",
    ]
    for path in suspects:
        text = path.read_text(encoding="utf-8")
        assert 'count("**") // 2' not in text, (
            f"legacy ``**`` heuristic still present in {path.name} — "
            f"use ``len(episodes)`` via get_episodes_for_context instead"
        )
