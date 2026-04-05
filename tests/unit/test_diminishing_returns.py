"""
Tests for diminishing returns detection in orchestrator loops.

Covers:
- IterationTracker dataclass: record(), should_stop()
- Reasoning-only vs tool-call iteration distinction
- Consecutive low-delta counter behavior
- Budget cap enforcement
- Fallback token estimation when output_tokens is None
- Constants defined in kestrel_agent.py
"""

import pytest
from unittest.mock import MagicMock

from kestrel_sovereign.agent.orchestrator_engine import IterationTracker
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(
    content: str = "",
    output_tokens: int = None,
    tool_calls: list = None,
) -> LLMResponse:
    """Build a minimal LLMResponse for testing."""
    return LLMResponse(
        content=content,
        output_tokens=output_tokens,
        tool_calls=tool_calls,
    )


def _make_tool_call(name: str = "some_tool") -> ToolCall:
    return ToolCall(id="tc_1", name=name, arguments={})


# ---------------------------------------------------------------------------
# IterationTracker.record()
# ---------------------------------------------------------------------------

class TestIterationTrackerRecord:
    """Tests for IterationTracker.record()."""

    def test_low_output_increments_counter(self):
        tracker = IterationTracker(threshold=500)
        resp = _make_response(output_tokens=100)
        tracker.record(resp)
        assert tracker.consecutive_low_delta == 1

    def test_high_output_resets_counter(self):
        tracker = IterationTracker(threshold=500)
        # Build up some low-delta
        tracker.consecutive_low_delta = 3
        resp = _make_response(output_tokens=600)
        tracker.record(resp)
        assert tracker.consecutive_low_delta == 0

    def test_exact_threshold_is_not_low(self):
        """output_tokens == threshold should NOT be counted as low-delta."""
        tracker = IterationTracker(threshold=500)
        resp = _make_response(output_tokens=500)
        tracker.record(resp)
        assert tracker.consecutive_low_delta == 0

    def test_tool_call_response_resets_counter(self):
        """Iterations with tool calls are intentionally small -- reset counter."""
        tracker = IterationTracker(threshold=500)
        tracker.consecutive_low_delta = 4
        resp = _make_response(
            content="short",
            output_tokens=10,
            tool_calls=[_make_tool_call()],
        )
        tracker.record(resp)
        assert tracker.consecutive_low_delta == 0

    def test_tool_call_response_not_counted_even_if_low(self):
        """Even zero-token tool-call responses should not increment counter."""
        tracker = IterationTracker(threshold=500)
        resp = _make_response(
            content="",
            output_tokens=0,
            tool_calls=[_make_tool_call()],
        )
        tracker.record(resp)
        assert tracker.consecutive_low_delta == 0

    def test_fallback_estimation_when_output_tokens_is_none(self):
        """When output_tokens is None, fall back to len(content) // 4."""
        tracker = IterationTracker(threshold=500)
        # 2400 chars // 4 = 600 tokens -> above threshold
        resp = _make_response(content="x" * 2400, output_tokens=None)
        tracker.record(resp)
        assert tracker.consecutive_low_delta == 0

    def test_fallback_estimation_low(self):
        """Short content with no output_tokens should count as low."""
        tracker = IterationTracker(threshold=500)
        # 100 chars // 4 = 25 tokens -> below threshold
        resp = _make_response(content="x" * 100, output_tokens=None)
        tracker.record(resp)
        assert tracker.consecutive_low_delta == 1

    def test_fallback_estimation_empty_content(self):
        """None content with None output_tokens -> 0 estimated tokens -> low."""
        tracker = IterationTracker(threshold=500)
        resp = _make_response(content=None, output_tokens=None)
        tracker.record(resp)
        assert tracker.consecutive_low_delta == 1

    def test_consecutive_low_increments(self):
        """Multiple consecutive low-output reasoning iterations accumulate."""
        tracker = IterationTracker(threshold=500)
        for _ in range(5):
            tracker.record(_make_response(output_tokens=50))
        assert tracker.consecutive_low_delta == 5

    def test_high_output_breaks_streak(self):
        """A single high-output iteration resets the streak."""
        tracker = IterationTracker(threshold=500)
        for _ in range(4):
            tracker.record(_make_response(output_tokens=50))
        assert tracker.consecutive_low_delta == 4
        tracker.record(_make_response(output_tokens=1000))
        assert tracker.consecutive_low_delta == 0

    def test_tool_call_breaks_streak(self):
        """A tool-call iteration in the middle resets the streak."""
        tracker = IterationTracker(threshold=500)
        for _ in range(3):
            tracker.record(_make_response(output_tokens=50))
        assert tracker.consecutive_low_delta == 3
        tracker.record(_make_response(output_tokens=10, tool_calls=[_make_tool_call()]))
        assert tracker.consecutive_low_delta == 0


# ---------------------------------------------------------------------------
# IterationTracker.should_stop()
# ---------------------------------------------------------------------------

class TestIterationTrackerShouldStop:
    """Tests for IterationTracker.should_stop()."""

    def test_no_stop_when_fresh(self):
        tracker = IterationTracker(max_low_delta=5, budget_stop_pct=90)
        assert tracker.should_stop(iteration=0, max_iterations=50) is False

    def test_stop_after_max_low_delta(self):
        tracker = IterationTracker(max_low_delta=5, budget_stop_pct=90)
        tracker.consecutive_low_delta = 5
        assert tracker.should_stop(iteration=10, max_iterations=50) is True

    def test_no_stop_below_max_low_delta(self):
        tracker = IterationTracker(max_low_delta=5, budget_stop_pct=90)
        tracker.consecutive_low_delta = 4
        assert tracker.should_stop(iteration=10, max_iterations=50) is False

    def test_stop_at_budget_cap(self):
        """90% of 50 = 45. iteration=45 should trigger stop."""
        tracker = IterationTracker(max_low_delta=5, budget_stop_pct=90)
        assert tracker.should_stop(iteration=45, max_iterations=50) is True

    def test_no_stop_just_below_budget_cap(self):
        """iteration=44 is below 90% of 50 (45)."""
        tracker = IterationTracker(max_low_delta=5, budget_stop_pct=90)
        assert tracker.should_stop(iteration=44, max_iterations=50) is False

    def test_budget_cap_100_percent_never_triggers(self):
        """With budget_stop_pct=100, only the for-loop range limits iterations."""
        tracker = IterationTracker(max_low_delta=5, budget_stop_pct=100)
        # iteration=49, max_iterations=50: 49 < 50 * 100/100 = 50
        assert tracker.should_stop(iteration=49, max_iterations=50) is False

    def test_diminishing_returns_takes_priority_over_budget(self):
        """Both conditions true -> should_stop is True."""
        tracker = IterationTracker(max_low_delta=5, budget_stop_pct=90)
        tracker.consecutive_low_delta = 5
        assert tracker.should_stop(iteration=45, max_iterations=50) is True

    def test_small_max_iterations_budget_cap(self):
        """With max_iterations=10 and 90% cap, stop at iteration 9."""
        tracker = IterationTracker(max_low_delta=5, budget_stop_pct=90)
        assert tracker.should_stop(iteration=9, max_iterations=10) is True
        assert tracker.should_stop(iteration=8, max_iterations=10) is False


# ---------------------------------------------------------------------------
# Integration: full record + should_stop cycle
# ---------------------------------------------------------------------------

class TestIterationTrackerIntegration:
    """End-to-end scenarios combining record() and should_stop()."""

    def test_five_consecutive_low_reasoning_triggers_stop(self):
        """The canonical scenario: 5 consecutive low-output reasoning iterations."""
        tracker = IterationTracker(threshold=500, max_low_delta=5, budget_stop_pct=90)
        max_iter = 50
        for i in range(5):
            resp = _make_response(output_tokens=100)
            tracker.record(resp)
        assert tracker.should_stop(iteration=10, max_iterations=max_iter) is True

    def test_tool_calls_interleaved_prevent_stop(self):
        """Tool calls reset the counter, so interleaving prevents false positives."""
        tracker = IterationTracker(threshold=500, max_low_delta=5, budget_stop_pct=90)
        max_iter = 50
        for i in range(10):
            if i % 3 == 2:
                # Every 3rd iteration is a tool call
                resp = _make_response(output_tokens=10, tool_calls=[_make_tool_call()])
            else:
                resp = _make_response(output_tokens=50)
            tracker.record(resp)
        # Max consecutive without a tool-call reset is 2
        assert tracker.consecutive_low_delta <= 2
        assert tracker.should_stop(iteration=10, max_iterations=max_iter) is False

    def test_high_output_then_low_needs_full_streak(self):
        """High output early, then low -- needs full 5 consecutive low to stop."""
        tracker = IterationTracker(threshold=500, max_low_delta=5, budget_stop_pct=90)
        # 3 high-output iterations
        for _ in range(3):
            tracker.record(_make_response(output_tokens=1000))
        assert tracker.consecutive_low_delta == 0
        # 4 low-output iterations -- not enough
        for _ in range(4):
            tracker.record(_make_response(output_tokens=50))
        assert tracker.should_stop(iteration=7, max_iterations=50) is False
        # 5th low-output iteration -- now triggers
        tracker.record(_make_response(output_tokens=50))
        assert tracker.should_stop(iteration=8, max_iterations=50) is True


# ---------------------------------------------------------------------------
# Constants smoke tests
# ---------------------------------------------------------------------------

class TestDiminishingReturnsConstants:
    """Verify constants are defined and have expected defaults."""

    def test_constants_importable(self):
        from kestrel_sovereign.kestrel_agent import (
            KESTREL_DIMINISHING_THRESHOLD,
            KESTREL_MAX_LOW_DELTA,
            KESTREL_BUDGET_STOP_PCT,
        )
        assert KESTREL_DIMINISHING_THRESHOLD == 500
        assert KESTREL_MAX_LOW_DELTA == 5
        assert KESTREL_BUDGET_STOP_PCT == 90

    def test_constants_are_integers(self):
        from kestrel_sovereign.kestrel_agent import (
            KESTREL_DIMINISHING_THRESHOLD,
            KESTREL_MAX_LOW_DELTA,
            KESTREL_BUDGET_STOP_PCT,
        )
        assert isinstance(KESTREL_DIMINISHING_THRESHOLD, int)
        assert isinstance(KESTREL_MAX_LOW_DELTA, int)
        assert isinstance(KESTREL_BUDGET_STOP_PCT, int)

    def test_tracker_defaults_match_constants(self):
        """IterationTracker field defaults should match the module constants."""
        from kestrel_sovereign.kestrel_agent import (
            KESTREL_DIMINISHING_THRESHOLD,
            KESTREL_MAX_LOW_DELTA,
            KESTREL_BUDGET_STOP_PCT,
        )
        tracker = IterationTracker(
            threshold=KESTREL_DIMINISHING_THRESHOLD,
            max_low_delta=KESTREL_MAX_LOW_DELTA,
            budget_stop_pct=KESTREL_BUDGET_STOP_PCT,
        )
        assert tracker.threshold == 500
        assert tracker.max_low_delta == 5
        assert tracker.budget_stop_pct == 90
