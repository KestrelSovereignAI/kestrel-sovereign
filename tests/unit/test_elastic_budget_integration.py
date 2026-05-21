"""Integration tests for the elastic budget (#1309) wired into the
production assembly path.

Covers the gaps codex round 1 #5 flagged:

- ``DegradedModeError`` becomes a degraded ``ContextResult`` (no LLM
  call is made); warnings carry the floor + total-budget figures.
- Released slack from finalized sections actually grows history's
  effective max_tokens — the formatter asks for the extra messages.
- ``measure_mandatory_system_tokens`` counts constitution + SOUL.md
  + AGENTS.md + state-of-mind (and nothing optional).
"""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.context_builder import (
    ContextBuilder,
    MANDATORY_SYSTEM_SUBSECTIONS,
)
from kestrel_sovereign.agent.context_manager import ContextManager, ContextResult
from kestrel_sovereign.agent.token_budget import ElasticTokenBudget, RESPONSE_RESERVE


# ---------------------------------------------------------------------------
# measure_mandatory_system_tokens: only mandatory subsections are counted
# ---------------------------------------------------------------------------


def _real_builder_with_bootstrap(bootstrap: dict, model: str = "gpt-4") -> ContextBuilder:
    """Construct a real ContextBuilder via __new__ with a real token
    counter — needed to exercise ``measure_mandatory_system_tokens``
    against actual byte counts."""
    cb = object.__new__(ContextBuilder)
    cb._llm_service = None
    cb._model_fallback = model
    from kestrel_sovereign.agent.token_counter import get_token_counter
    cb._counter = get_token_counter(model)
    cb._counter_model = model
    cb._bootstrap_loader = MagicMock()
    cb._bootstrap_loader.load = MagicMock(return_value=OrderedDict(bootstrap))
    cb._bootstrap_loader.get_file = MagicMock(
        side_effect=lambda name: bootstrap.get(name)
    )
    cb.storage = MagicMock()
    cb.consolidator = None
    return cb


class _StateOfMindStub:
    def __init__(self, governance_mode: str = "AUTONOMY"):
        self.governance_mode = governance_mode
        self.active_conflicts = []
        self.delegated_principles = []


class TestMeasureMandatoryFloor:
    def test_counts_constitution_only_when_no_bootstrap(self):
        cb = _real_builder_with_bootstrap({})
        floor = cb.measure_mandatory_system_tokens(constitution="Be kind.")
        assert floor > 0

    def test_includes_soul_md_when_present(self):
        cb_no_soul = _real_builder_with_bootstrap({})
        cb_with_soul = _real_builder_with_bootstrap(
            {"SOUL.md": "I am a long identity declaration." * 50}
        )
        floor_a = cb_no_soul.measure_mandatory_system_tokens(constitution="Be kind.")
        floor_b = cb_with_soul.measure_mandatory_system_tokens(constitution="Be kind.")
        assert floor_b > floor_a

    def test_includes_agents_md_when_present(self):
        cb_no_agents = _real_builder_with_bootstrap({})
        cb_with_agents = _real_builder_with_bootstrap(
            {"AGENTS.md": "Operator policy: " + ("x " * 200)}
        )
        floor_a = cb_no_agents.measure_mandatory_system_tokens(constitution="Be kind.")
        floor_b = cb_with_agents.measure_mandatory_system_tokens(constitution="Be kind.")
        assert floor_b > floor_a

    def test_excludes_optional_bootstrap(self):
        """A non-mandatory bootstrap file (e.g. TOOLS.md) must NOT
        change the floor — only items in ``MANDATORY_SYSTEM_SUBSECTIONS``
        are counted."""
        cb_baseline = _real_builder_with_bootstrap({})
        cb_with_tools = _real_builder_with_bootstrap(
            {"TOOLS.md": "Optional tools description " * 100}
        )
        floor_a = cb_baseline.measure_mandatory_system_tokens(constitution="Be kind.")
        floor_b = cb_with_tools.measure_mandatory_system_tokens(constitution="Be kind.")
        assert floor_a == floor_b
        assert "bootstrap_tools" not in MANDATORY_SYSTEM_SUBSECTIONS

    def test_includes_state_of_mind_when_supplied(self):
        cb = _real_builder_with_bootstrap({})
        baseline = cb.measure_mandatory_system_tokens(constitution="Be kind.")
        with_som = cb.measure_mandatory_system_tokens(
            constitution="Be kind.", state_of_mind=_StateOfMindStub()
        )
        assert with_som > baseline


# ---------------------------------------------------------------------------
# DegradedModeError → degraded ContextResult (no LLM call)
# ---------------------------------------------------------------------------


class TestDegradedModeFlow:
    @pytest.mark.asyncio
    async def test_floor_exceeds_window_returns_degraded_result(self):
        """When the mandatory floor cannot fit, ``build_context``
        returns ``ContextResult(degraded_mode=True)`` and warns
        loudly — no LLM call may proceed (Emma's fail-closed
        invariant)."""
        cb = _real_builder_with_bootstrap({})
        # Override the measurement to declare a floor bigger than any
        # real model's context window.
        cb.measure_mandatory_system_tokens = lambda *a, **kw: 10_000_000

        cm = ContextManager(storage=MagicMock(), context_builder=cb)
        cm.conversation_manager = MagicMock()
        cm.conversation_manager.get_conversation_history = AsyncMock(return_value=[])
        cm.llm_service = None
        cm._microcompact_tool_results = lambda hist: 0

        result = await cm.build_context(
            query="anything",
            constitution="Be kind.",
            conversation_history=[],
        )
        assert isinstance(result, ContextResult)
        assert result.degraded_mode is True
        assert result.system_prompt == ""
        assert result.messages == []
        assert result.mandatory_system_tokens == 10_000_000
        assert any("DEGRADED MODE" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Released slack actually grows history's effective max_tokens
# ---------------------------------------------------------------------------


class TestHistoryOverBudgetPrune:
    """Codex round 2 P1: when format_conversation_history overshoots
    its max_tokens (wrap overhead added after the per-message budget
    check), ``budget.use(\"history\", …)`` returns False without
    recording usage. ContextManager must pre-trim the formatted
    history so the LLM call does not send bytes the budget never
    accounted for — and so the legacy ``total_used > total_budget``
    prune isn't relied on for a case it cannot see.
    """

    @pytest.mark.asyncio
    async def test_history_pre_trim_runs_when_use_rejects(self):
        cb = _real_builder_with_bootstrap({})
        cb.get_session_briefing = lambda: ""
        cb.measure_mandatory_system_tokens = lambda *a, **kw: 0
        cb.get_episodes_for_context = AsyncMock(return_value=[])
        cb.retrieve_context = AsyncMock(return_value=None)
        cb.build_system_prompt = lambda **kw: "sys"

        # Force the formatter to return a giant history that blows
        # both the static slice and the elastic pool. Return a fresh
        # copy each call so the production pre-trim's ``.pop(0)``
        # doesn't mutate the fixture.
        original_count = 20
        cb.format_conversation_history = lambda history, max_tokens=None, **kw: [
            {"role": "user", "content": "X" * 50_000} for _ in range(original_count)
        ]

        cm = ContextManager(storage=MagicMock(), context_builder=cb)
        cm.conversation_manager = MagicMock()
        cm.conversation_manager.get_conversation_history = AsyncMock(return_value=[])
        cm.llm_service = None
        cm._microcompact_tool_results = lambda hist: 0
        cm.memory_retriever = None
        cm.memory_manager = None

        result = await cm.build_context(
            query="anything",
            constitution="Be kind.",
            include_memories=False,
            include_rag=False,
            conversation_history=[],
        )
        # Pre-trim must have run — fewer messages than the formatter
        # returned, and a warning surfaced.
        assert len(result.messages) < original_count
        assert any("pre-trimmed" in w or "auto-pruned" in w.lower() for w in result.warnings)


class TestHistoryAbsorbsReleasedSlack:
    @pytest.mark.asyncio
    async def test_history_max_tokens_reflects_pool(self):
        """Production sizes history with ``budget.effective_budget("history")``
        when elastic — codex round 1 #2. Capture the max_tokens the
        formatter actually receives and assert it exceeds the static
        slice once memories/RAG are finalized empty."""
        cb = _real_builder_with_bootstrap({})
        cb.get_session_briefing = lambda: ""
        cb.measure_mandatory_system_tokens = lambda *a, **kw: 0
        cb.get_episodes_for_context = AsyncMock(return_value=[])
        cb.retrieve_context = AsyncMock(return_value=None)
        observed_max: list = []

        def fake_format(history, max_tokens=None, **kw):
            observed_max.append(max_tokens)
            return []

        cb.format_conversation_history = fake_format
        cb.build_system_prompt = lambda **kw: "system body"

        cm = ContextManager(storage=MagicMock(), context_builder=cb)
        cm.conversation_manager = MagicMock()
        cm.conversation_manager.get_conversation_history = AsyncMock(return_value=[])
        cm.llm_service = None
        cm._microcompact_tool_results = lambda hist: 0
        cm.memory_retriever = None  # skip memory section
        cm.memory_manager = None

        result = await cm.build_context(
            query="anything",
            constitution="Be kind.",
            include_memories=False,
            include_rag=False,
            conversation_history=[],
        )
        assert observed_max, "format_conversation_history must be called"
        # Reproduce the elastic budget against the *same model the
        # ContextManager uses* so the comparison is apples-to-apples.
        elastic = ElasticTokenBudget(
            cm.model, message_count=0, mandatory_system_tokens=0
        )
        legacy_static_history = elastic.history
        # Reproduce the post-finalization pool: system has used a few
        # tokens (the "system body" stub); other empty sections fully
        # release their budgets.
        elastic.use("system", 5)
        elastic.mark_section_finalized("system")
        elastic.mark_section_finalized("episodes")
        elastic.mark_section_finalized("memories")
        elastic.mark_section_finalized("rag")
        expected_effective = elastic.effective_budget("history")
        # The bug codex caught (#2): observed_max would equal the
        # legacy static slice because production sized history with
        # ``budget.history`` rather than ``budget.effective_budget("history")``.
        # After the fix, observed_max must exceed the static slice by
        # the released pool from finalized empty sections.
        assert observed_max[0] > legacy_static_history, (
            "elastic budget should let history grow beyond the static "
            "slice when memories/RAG/episodes release their budgets — "
            f"observed {observed_max[0]} vs legacy static {legacy_static_history}"
        )
        # Approximate match to the synthetic-reference figure
        # (system 'use' bytes between this test and the production
        # path differ by a handful of tokens — the contract is
        # "history absorbs the pool," not exact arithmetic equality).
        assert abs(observed_max[0] - expected_effective) <= 20
        assert result.degraded_mode is False
