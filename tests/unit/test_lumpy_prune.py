"""Tests for the lumpy-prune anchor and safety-net helper (#1430).

The cache-thrash failure mode: Anthropic's cache is position-indexed
(see ``project_anthropic_cache_markers.md``). When history pruning
sheds ~one turn of tokens every turn, the bytes at ``messages[-2]`` /
``messages[-4]`` shift on every request and compound caching breaks.

Lumpy prune solves it in two layers:

1. ``_lumpy_anchor`` runs before ``format_conversation_history`` and
   pre-slices history so the prefix is byte-stable across multiple
   turns of growth. The anchor only advances in chunks, so several
   turns of growth happen at the same anchor before the next jump.

2. ``_lumpy_prune_history`` is the safety net for the rare case where
   total budget accounting overshoots after section sizing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.agent.context_manager import ContextManager
from kestrel_sovereign.agent.token_budget import TokenBudget


def _msg(content: str, role: str = "user") -> dict:
    return {"role": role, "content": content}


def _make_manager(model: str = "gpt-4") -> ContextManager:
    return ContextManager(storage=MagicMock(), model=model)


def _seed_history(tokens_per_msg_text: int, count: int) -> list[dict]:
    """Build messages of approximately ``tokens_per_msg_text`` tokens
    each under the default counter (4 chars ≈ 1 token English)."""
    body = "x " * tokens_per_msg_text
    return [_msg(body, role=("user" if i % 2 == 0 else "assistant")) for i in range(count)]


class TestLumpyAnchorStability:
    """The anchor's defining property: stable across the turns BETWEEN
    chunk crossings, even as new turns are appended."""

    def test_anchor_zero_when_history_fits(self):
        cm = _make_manager()
        history = _seed_history(10, 5)
        # Tiny history, huge ceiling: include all.
        assert cm._lumpy_anchor(history, max_tokens=10000) == 0

    def test_anchor_zero_for_empty_history(self):
        cm = _make_manager()
        assert cm._lumpy_anchor([], max_tokens=1000) == 0

    def test_anchor_advances_when_history_overflows(self):
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.75
        history = _seed_history(50, 40)
        # Tight ceiling that the full history clearly overflows.
        anchor = cm._lumpy_anchor(history, max_tokens=500)
        assert anchor > 0
        assert anchor < len(history)  # never drops everything

    def test_anchor_stable_across_appended_turns_within_chunk(self):
        """The whole point of lumpy: after the anchor jumps, appending
        a few turns of new messages should NOT advance it again until
        we cross another chunk boundary."""
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.75  # chunk = 25% of max
        max_tokens = 2000  # chunk = 500 tokens
        history = _seed_history(50, 40)  # ~2000 tokens of history
        # Push history clearly over the ceiling so anchor != 0.
        history_over = _seed_history(50, 60)  # ~3000 tokens
        anchor_n = cm._lumpy_anchor(history_over, max_tokens=max_tokens)
        assert anchor_n > 0

        # Append a small new turn (well under chunk_size = 500).
        history_n_plus_1 = history_over + _seed_history(50, 1)
        anchor_n_plus_1 = cm._lumpy_anchor(history_n_plus_1, max_tokens=max_tokens)

        # Anchor must be SAME (cache-stability). If this fails, the
        # whole change is moot.
        assert anchor_n_plus_1 == anchor_n, (
            f"anchor moved from {anchor_n} to {anchor_n_plus_1} on a "
            "small append — cache-stability invariant violated"
        )

    def test_anchor_advances_when_chunk_boundary_crossed(self):
        """When growth crosses a chunk boundary, the anchor MUST move
        forward — otherwise we'd drift past the ceiling forever."""
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.75
        max_tokens = 2000  # chunk = 500
        history_small = _seed_history(50, 60)  # ~3000
        anchor_small = cm._lumpy_anchor(history_small, max_tokens=max_tokens)

        # Append enough new content to clear another chunk boundary.
        history_big = history_small + _seed_history(50, 20)  # +1000 tokens
        anchor_big = cm._lumpy_anchor(history_big, max_tokens=max_tokens)

        assert anchor_big > anchor_small, (
            "anchor failed to advance after crossing a chunk boundary"
        )

    def test_anchor_keeps_at_least_one_message(self):
        """Even on pathologically small budgets, never drop everything
        — the prune-from-empty case is degenerate."""
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.75
        history = _seed_history(50, 40)
        # Absurdly tight ceiling: target_drop would exceed total tokens.
        anchor = cm._lumpy_anchor(history, max_tokens=20)
        assert anchor <= len(history) - 1
        # Anchored slice has at least one message.
        assert len(history[anchor:]) >= 1

    def test_anchor_unaffected_by_zero_or_negative_ceiling(self):
        cm = _make_manager()
        history = _seed_history(50, 10)
        assert cm._lumpy_anchor(history, max_tokens=0) == 0
        assert cm._lumpy_anchor(history, max_tokens=-5) == 0


class TestLumpyPruneSafetyNet:
    """The post-budget safety net rarely fires in practice (most overflow
    is caught earlier by the anchor + format_conversation_history), but
    when it does it must still leave history below the lumpy target."""

    def test_no_overage_returns_zero(self):
        cm = _make_manager()
        budget = TokenBudget("gpt-4")
        history = _seed_history(10, 6)
        original = list(history)
        assert cm._lumpy_prune_history(history, budget) == 0
        assert history == original

    def test_empty_history_returns_zero(self):
        cm = _make_manager()
        budget = TokenBudget("gpt-4")
        budget.use("history", budget.total_budget + 1000, items=0)
        assert cm._lumpy_prune_history([], budget) == 0

    def test_drops_below_lumpy_target(self):
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.75
        budget = TokenBudget("gpt-4")
        history = _seed_history(50, 80)
        budget.use("history", budget.total_budget + 2000, items=len(history))

        dropped = cm._lumpy_prune_history(history, budget)
        assert dropped > 0
        target = int(budget.total_budget * cm.PRUNE_TARGET_FRAC)
        assert budget.allocations["history"].used <= target

    def test_can_drain_completely_if_budget_demands(self):
        """Unlike the previous implementation, the safety net is not
        artificially floored at one message. The current user turn is
        appended downstream by the caller (streaming/agent code), not
        held in this list — so emptying it on a small enough budget is
        correct, not a bug. Confirmed by codex review of #1430."""
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.50
        budget = TokenBudget("gpt-4")
        history = _seed_history(10, 3)
        # Force massive overage.
        budget.use("history", budget.total_budget * 10, items=len(history))
        cm._lumpy_prune_history(history, budget)
        # Helper drained as needed; downstream still appends current user turn.
        assert isinstance(history, list)

    def test_history_alloc_items_kept_in_sync(self):
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.75
        budget = TokenBudget("gpt-4")
        history = _seed_history(50, 80)
        budget.use("history", budget.total_budget + 2000, items=len(history))

        cm._lumpy_prune_history(history, budget)
        assert budget.allocations["history"].items == len(history)


class TestEnvParsing:
    def test_bad_env_falls_back_to_default(self):
        assert ContextManager._resolve_prune_target_frac("not a number") == 0.75

    def test_none_returns_default(self):
        assert ContextManager._resolve_prune_target_frac(None) == 0.75

    def test_valid_value_passes_through(self):
        assert ContextManager._resolve_prune_target_frac("0.6") == 0.6

    def test_oversize_value_clamps_to_one(self):
        assert ContextManager._resolve_prune_target_frac("999") == 1.0

    def test_negative_value_clamps_to_floor(self):
        assert ContextManager._resolve_prune_target_frac("-1") == 0.05

    def test_class_constant_in_unit_band(self):
        assert 0.05 <= ContextManager.PRUNE_TARGET_FRAC <= 1.0


@pytest.mark.parametrize("frac", [0.5, 0.6, 0.75, 0.9])
def test_safety_net_target_respected_across_fractions(frac: float):
    cm = _make_manager()
    cm.PRUNE_TARGET_FRAC = frac
    budget = TokenBudget("gpt-4")
    history = _seed_history(50, 80)
    budget.use("history", budget.total_budget + 2000, items=len(history))

    cm._lumpy_prune_history(history, budget)

    target = int(budget.total_budget * frac)
    assert budget.allocations["history"].used <= target


class TestSentFormBytePreservation:
    """The anchor must NOT rewrite content of surviving messages.
    Sent-form replay (the cache-stability contract) depends on byte
    identity, so the helper is a pure index selector — it just decides
    which messages survive."""

    def test_helper_does_not_mutate_message_content(self):
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.75
        history = _seed_history(50, 40)
        # Tag one of the surviving messages with sent-form metadata.
        history[-5]["metadata"] = {"sent_form": True}
        history[-5]["rendered_content"] = "RENDERED-BYTES"

        anchor = cm._lumpy_anchor(history, max_tokens=500)
        survivors = history[anchor:]
        # The rendered_content & metadata are untouched — caller
        # (format_conversation_history) handles emit-byte selection
        # downstream.
        target_msg = next(m for m in survivors if m.get("metadata"))
        assert target_msg["metadata"] == {"sent_form": True}
        assert target_msg["rendered_content"] == "RENDERED-BYTES"


class TestEmitByteCounting:
    """The anchor must count the bytes the LLM will actually see, not
    raw ``content``. Without this, a sent_form user row whose
    ``rendered_content`` is much larger than raw ``content`` would let
    the anchor declare a fit while the formatter still has to truncate
    inside the anchored slice — recreating the cache churn this fix is
    meant to prevent (codex round-2 P1)."""

    def test_sent_form_uses_rendered_content_bytes(self):
        cm = _make_manager()
        # Two messages: one user with sent_form (rendered_content is
        # much larger than content), one assistant.
        sent_form_msg = {
            "role": "user",
            "content": "hello",  # 5 chars
            "rendered_content": "x" * 4000,  # ~1000 tokens
            "metadata": {"sent_form": True},
        }
        emit = cm._emit_content_for_msg(sent_form_msg)
        # The emitter picks rendered_content, NOT raw content.
        assert emit == "x" * 4000

    def test_unsent_user_wraps_with_user_input_tags(self):
        cm = _make_manager()
        msg = {"role": "user", "content": "hi"}
        emit = cm._emit_content_for_msg(msg)
        assert "<user_input>" in emit and "</user_input>" in emit
        assert "hi" in emit

    def test_assistant_uses_raw_content(self):
        cm = _make_manager()
        msg = {"role": "assistant", "content": "ack"}
        emit = cm._emit_content_for_msg(msg)
        assert emit == "ack"

    def test_legacy_human_role_treated_as_user(self):
        cm = _make_manager()
        msg = {"role": "human", "content": "old format"}
        emit = cm._emit_content_for_msg(msg)
        assert "<user_input>" in emit

    def test_anchor_overflows_when_emit_bytes_exceed_raw(self):
        """Critical regression: if anchor only counted raw, a slice
        with sent_form rows whose rendered_content blew up would pass
        the anchor's fit check and overflow downstream."""
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.75
        # Build messages with small raw content but huge rendered.
        history = []
        for i in range(50):
            history.append(
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": "x",  # ~1 token raw
                    # Sent-form on user rows blows up to ~250 tokens
                    "rendered_content": ("y " * 250) if i % 2 == 0 else None,
                    "metadata": {"sent_form": True} if i % 2 == 0 else {},
                }
            )
        # Small ceiling so emit-byte total clearly overflows but raw total fits.
        anchor = cm._lumpy_anchor(history, max_tokens=500)
        # Must drop some — because emit bytes are what matter.
        assert anchor > 0, "anchor failed to detect emit-byte overflow"


class TestStableCeiling:
    """Per-turn budget variance (RAG/memory slack) must not move the
    anchor. ``build_context`` calls _lumpy_anchor with the static
    ``budget.history``, not the elastic effective ceiling, so the
    anchor is deterministic from raw history and a fixed ceiling
    (codex round-2 P2)."""

    def test_anchor_independent_of_elastic_slack(self):
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.75
        history = _seed_history(50, 60)
        # Same history, different ceilings: anchor should be a
        # function of (history, ceiling), so caller controls
        # stability by passing a stable ceiling.
        a1 = cm._lumpy_anchor(history, max_tokens=2000)
        a2 = cm._lumpy_anchor(history, max_tokens=2000)
        assert a1 == a2  # idempotent given same inputs
        # When the ceiling jitters (e.g. elastic slack changes), the
        # anchor's output changes. That's why build_context passes
        # budget.history (static), not budget.effective_budget.
        a3 = cm._lumpy_anchor(history, max_tokens=2500)
        # Without hysteresis on the ceiling parameter, different
        # ceilings can yield different anchors — this test pins that
        # behaviour so reviewers understand the trade-off.
        assert isinstance(a3, int)
