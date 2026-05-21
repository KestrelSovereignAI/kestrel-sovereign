"""Tests for the elastic token budget (#1309).

Asserts Emma's 2026-05-20 contract:
- Mandatory governance floor is **non-borrowable**.
- Below-floor is a hard-failure / degraded-mode condition
  (``DegradedModeError``), never silent absorption.
- Idle section budget flows to *any* over-demanded eligible section
  by turn-intent priority — not only history.
- Legacy callers (``adaptive=True``, no ``elastic`` kwarg) keep the
  old shape; back-compat invariant.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.agent.token_budget import (
    AdaptiveTokenBudget,
    DEFAULT_ELASTIC_PRIORITY,
    DegradedModeError,
    ElasticTokenBudget,
    RESPONSE_RESERVE,
    TokenBudget,
    create_budget,
)


# ---------------------------------------------------------------------------
# Back-compat: nothing about the legacy API changes by default
# ---------------------------------------------------------------------------


class TestBackCompat:
    def test_create_budget_default_returns_adaptive(self):
        b = create_budget("gpt-4", message_count=5)
        assert isinstance(b, AdaptiveTokenBudget)
        # Not elastic — no mandatory floor concept
        assert not hasattr(b, "mandatory_system_tokens")

    def test_create_budget_adaptive_false_returns_plain(self):
        b = create_budget("gpt-4", adaptive=False)
        assert isinstance(b, TokenBudget) and not isinstance(b, AdaptiveTokenBudget)

    def test_create_budget_elastic_returns_elastic(self):
        b = create_budget("gpt-4", elastic=True)
        assert isinstance(b, ElasticTokenBudget)
        assert b.mandatory_system_tokens == 0


# ---------------------------------------------------------------------------
# Mandatory floor — non-borrowable
# ---------------------------------------------------------------------------


class TestMandatoryFloor:
    def test_zero_floor_keeps_adaptive_allocations(self):
        elastic = create_budget("gpt-4", elastic=True, mandatory_system_tokens=0)
        adaptive = create_budget("gpt-4", adaptive=True)
        assert elastic.system == adaptive.system
        assert elastic.history == adaptive.history

    def test_floor_raises_system_budget_when_needed(self):
        # Compute a floor that exceeds adaptive system but fits the
        # total budget — derived from the model so the test stays
        # honest across context limits.
        adaptive = create_budget("gpt-4", adaptive=True)
        floor = adaptive.system + 500
        assert floor < adaptive.total_budget, "test premise: floor must fit budget"
        elastic = create_budget(
            "gpt-4",
            elastic=True,
            mandatory_system_tokens=floor,
        )
        assert elastic.system >= floor
        # The bump is funded by trimming from elastic-priority sources;
        # the constructor never returns a budget where system < floor.
        assert elastic.mandatory_system_tokens == floor

    def test_floor_exceeds_total_budget_raises_degraded_mode(self):
        """If the floor cannot fit the model's window at all, fail
        closed — Emma's 2026-05-20 hardening invariant."""
        # Use a small model and a huge floor.
        with pytest.raises(DegradedModeError) as exc_info:
            create_budget(
                "gpt-4",
                elastic=True,
                mandatory_system_tokens=10_000_000,  # absurd
            )
        err = exc_info.value
        assert err.mandatory_system_tokens == 10_000_000
        assert err.total_budget > 0
        assert "degraded-mode" in str(err)
        assert err.model == "gpt-4"

    def test_floor_protected_during_mark_section_finalized(self):
        """``mark_section_finalized("system")`` must NOT return floor
        bytes to the elastic pool, even when used < floor."""
        elastic = create_budget(
            "gpt-4",
            elastic=True,
            mandatory_system_tokens=4_000,
        )
        # Use less than the floor.
        elastic.use("system", 100)
        slack = elastic.mark_section_finalized("system")
        # Whatever slack was released, the section's effective budget
        # cannot drop below the floor.
        assert elastic.allocations["system"].budget >= 4_000
        # Pool got the slack above the floor only.
        assert elastic.elastic_pool == slack

    def test_floor_protected_against_zero_use(self):
        """Even when the system slice has 0 used, the floor is still
        preserved on finalization."""
        elastic = create_budget(
            "gpt-4",
            elastic=True,
            mandatory_system_tokens=2_000,
        )
        elastic.mark_section_finalized("system")
        assert elastic.allocations["system"].budget >= 2_000


# ---------------------------------------------------------------------------
# Slack distribution — flows to any over-demanded section by priority
# ---------------------------------------------------------------------------


class TestSlackDistribution:
    def test_history_can_borrow_when_other_sections_finalize_short(self):
        elastic = create_budget("gpt-4", elastic=True)
        # Reserve a known starting state.
        memories_budget = elastic.memories
        rag_budget = elastic.rag
        history_budget = elastic.history

        # Memories and RAG finalize without using their budget.
        elastic.mark_section_finalized("memories")
        elastic.mark_section_finalized("rag")
        assert elastic.elastic_pool == memories_budget + rag_budget

        # History can now use more than its own budget — pool absorbs
        # the deficit.
        oversize = history_budget + memories_budget
        assert elastic.can_fit("history", oversize)
        assert elastic.use("history", oversize) is True
        # Pool drained by exactly the deficit (history_budget was
        # already free).
        expected_pool_after = memories_budget + rag_budget - (oversize - history_budget)
        assert elastic.elastic_pool == expected_pool_after

    def test_history_request_beyond_pool_returns_false(self):
        elastic = create_budget("gpt-4", elastic=True)
        history_budget = elastic.history
        elastic.mark_section_finalized("memories")  # release small slack
        # Ask for way more than budget + pool.
        huge = history_budget + elastic.elastic_pool + 100_000
        assert elastic.can_fit("history", huge) is False
        assert elastic.use("history", huge) is False

    def test_priority_is_documented(self):
        """The default priority order is exposed for the elastic
        contract documentation. History first by design — that's
        where the silent-prune correctness hole hurts most until C ships."""
        assert DEFAULT_ELASTIC_PRIORITY[0] == "history"

    def test_custom_priority_used_when_trimming_for_floor(self):
        """A RAG-heavy turn can demote history's priority. When the
        mandatory floor needs to be funded by trimming, the order is
        ``reverse(priority)`` — so the LAST entry of priority loses
        budget first. We verify that placing ``history`` last keeps
        history's budget at the full pre-trim amount when only a small
        bump is needed."""
        # Bump that requires a small trim.
        adaptive = create_budget("gpt-4", adaptive=True)
        adaptive_system = adaptive.system
        small_bump = 100

        # Default priority: history first → rag/memories/episodes
        # lose budget first → history NEVER loses budget for small bumps.
        default = create_budget(
            "gpt-4",
            elastic=True,
            mandatory_system_tokens=adaptive_system + small_bump,
        )
        assert default.history == adaptive.history

        # Inverted priority: history last → history loses budget first.
        # Confirm by inverting: priority=["rag","memories","episodes","history"]
        # means reversed = history,episodes,memories,rag → history
        # gets trimmed first.
        inverted = create_budget(
            "gpt-4",
            elastic=True,
            mandatory_system_tokens=adaptive_system + small_bump,
            demand_priority=["rag", "memories", "episodes", "history"],
        )
        assert inverted.history < adaptive.history


# ---------------------------------------------------------------------------
# Summary surfacing — the breakdown popup (#1310) reads this
# ---------------------------------------------------------------------------


class TestPriorityValidation:
    """Codex round 1 #3: custom priority must not let ``system`` get
    trimmed, must dedupe, and must fall back to the full borrowable set
    so floor funding can use every option."""

    def test_system_stripped_from_priority(self):
        elastic = ElasticTokenBudget(
            "gpt-4",
            demand_priority=["system", "rag", "history"],
        )
        # ``system`` is non-trimmable — must never appear in the
        # priority list (would let trim-for-floor cut into itself).
        assert "system" not in elastic.get_summary()["elastic_priority"]

    def test_unknown_sources_stripped(self):
        elastic = ElasticTokenBudget(
            "gpt-4",
            demand_priority=["rag", "nonsense", "history"],
        )
        assert "nonsense" not in elastic.get_summary()["elastic_priority"]

    def test_duplicates_collapsed(self):
        elastic = ElasticTokenBudget(
            "gpt-4",
            demand_priority=["history", "rag", "history", "rag"],
        )
        prio = elastic.get_summary()["elastic_priority"]
        assert prio.count("history") == 1
        assert prio.count("rag") == 1

    def test_missing_sources_appended_for_floor_funding(self):
        """A caller that only names ``rag`` must still be able to fund
        the floor by trimming history/episodes/memories — those get
        appended after the caller's explicit priority."""
        elastic = ElasticTokenBudget(
            "gpt-4",
            demand_priority=["rag"],
        )
        prio = elastic.get_summary()["elastic_priority"]
        # Caller-explicit comes first
        assert prio[0] == "rag"
        # All other borrowable sections appended after
        assert set(prio) == {"history", "episodes", "memories", "rag"}


class TestEffectiveBudget:
    """Codex round 1 #2: history must actually grow when slack is
    released into the elastic pool. Sections size against the
    effective ceiling, not the static slice."""

    def test_effective_budget_includes_pool(self):
        elastic = ElasticTokenBudget("gpt-4")
        base = elastic.effective_budget("history")
        assert base == elastic.allocations["history"].remaining
        # Release some slack — effective grows by that amount.
        elastic.mark_section_finalized("memories")
        assert elastic.effective_budget("history") == base + elastic.elastic_pool

    def test_effective_budget_unknown_source_returns_zero(self):
        elastic = ElasticTokenBudget("gpt-4")
        assert elastic.effective_budget("not-a-section") == 0


class TestSummarySurfacing:
    def test_summary_exposes_elastic_fields(self):
        elastic = create_budget(
            "gpt-4",
            elastic=True,
            mandatory_system_tokens=3_000,
            demand_priority=["history", "rag", "memories", "episodes"],
        )
        elastic.mark_section_finalized("memories")
        summary = elastic.get_summary()
        assert summary["mandatory_system_tokens"] == 3_000
        assert summary["elastic_pool_remaining"] >= 0
        assert summary["elastic_priority"] == [
            "history",
            "rag",
            "memories",
            "episodes",
        ]
        assert "memories" in summary["finalized_sections"]
