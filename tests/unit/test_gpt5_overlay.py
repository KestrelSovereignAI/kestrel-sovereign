"""Unit tests for the GPT-5 system-prompt overlay (#807 / #806)."""

import pytest

from kestrel_sovereign.llm.gpt5_overlay import (
    GPT5_BEHAVIOR_CONTRACT,
    is_gpt5_model_id,
    prepend_gpt5_overlay,
)


class TestIsGpt5ModelId:
    @pytest.mark.parametrize(
        "model_id",
        [
            "gpt-5",
            "gpt-5.4",
            "gpt-5.4-codex",
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.4-mini",
            "GPT-5.4",
            "openai/gpt-5.4",
            "openai-codex:gpt-5.5-pro",
            "openai-codex/gpt-5",
        ],
    )
    def test_matches_gpt5_family(self, model_id):
        assert is_gpt5_model_id(model_id)

    @pytest.mark.parametrize(
        "model_id",
        [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4.1",
            "o1-preview",
            "o3-mini",
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "llama3.2",
            "kimi-k2.5",
            "",
            None,
        ],
    )
    def test_rejects_non_gpt5(self, model_id):
        assert not is_gpt5_model_id(model_id)

    def test_no_false_positive_on_substring(self):
        # ``gpt-50`` (hypothetical future) is not gpt-5 family — boundary check.
        assert not is_gpt5_model_id("gpt-50")
        assert not is_gpt5_model_id("gpt-5x")


class TestPrependGpt5Overlay:
    def test_identity_for_non_gpt5(self):
        base = "You are Kestrel."
        assert prepend_gpt5_overlay(base, "gpt-4o") == base
        assert prepend_gpt5_overlay(base, "claude-sonnet-4-5") == base

    def test_identity_for_none_model(self):
        assert prepend_gpt5_overlay("base", None) == "base"

    def test_prepends_for_gpt5(self):
        base = "You are Kestrel."
        result = prepend_gpt5_overlay(base, "gpt-5.4")
        assert result is not None
        assert result.startswith("<persona_latch>")
        assert result.endswith(base)
        assert "<execution_policy>" in result
        assert "<tool_discipline>" in result
        assert "<output_contract>" in result
        assert "<completion_contract>" in result

    def test_returns_contract_alone_when_base_empty(self):
        assert prepend_gpt5_overlay(None, "gpt-5.4") == GPT5_BEHAVIOR_CONTRACT
        assert prepend_gpt5_overlay("", "gpt-5.4") == GPT5_BEHAVIOR_CONTRACT

    def test_byte_stable_across_calls(self):
        # Prefix-cache invariant: contract bytes do not change for the same model id.
        base = "constitutional system prompt with stable bytes"
        a = prepend_gpt5_overlay(base, "gpt-5.4")
        b = prepend_gpt5_overlay(base, "gpt-5.4")
        assert a == b
        # Even across model id variations within the family, the prefix block
        # itself is byte-identical (the contract text is constant).
        c = prepend_gpt5_overlay(base, "gpt-5.5-pro")
        assert c.startswith(GPT5_BEHAVIOR_CONTRACT)
        assert a.startswith(GPT5_BEHAVIOR_CONTRACT)


class TestProactiveElevationDiscipline:
    """#1734 (supersedes #1718's reactive retry): codex 0.138's sandbox
    profile is ``workspace-write`` (#1734) — writes inside the per-agent
    workspace + ``/tmp`` succeed silently, writes outside are blocked.
    The reactive retry from #1718 was correct under ``read-only`` but
    is no longer the dominant case.

    The proactive clause tells GPT-5 agents to request elevation
    BEFORE running a command that writes outside the workspace, not
    after a sandbox failure. Cleaner mental model, fewer round-trips,
    and avoids the stdout-injection vector entirely (no parsing of
    error text required).

    Lives in <tool_discipline> so it inherits the byte-stable
    prefix-cache invariant. Anthropic-backed agents do not see this
    overlay and do not need it — their shell path goes through
    ``_run_gates`` directly (no codex sandbox layer).
    """

    @pytest.mark.parametrize(
        "phrase",
        [
            "workspace",
            "request elevation",
            "host approval queue",
            "Treat operator-denied results as terminal",
            "outside the workspace",
        ],
    )
    def test_contract_carries_proactive_elevation_clause(self, phrase):
        assert phrase in GPT5_BEHAVIOR_CONTRACT, (
            f"GPT5 contract must mention {phrase!r} so the model knows "
            f"to request elevation proactively for out-of-workspace writes"
        )

    def test_proactive_clause_inside_tool_discipline(self):
        # Lives in <tool_discipline>, not floating, so it composes with
        # the rest of the tool-use guidance and stays cache-stable.
        td_start = GPT5_BEHAVIOR_CONTRACT.index("<tool_discipline>")
        td_end = GPT5_BEHAVIOR_CONTRACT.index("</tool_discipline>")
        td_block = GPT5_BEHAVIOR_CONTRACT[td_start:td_end]
        assert "request elevation" in td_block
        assert "outside the workspace" in td_block

    def test_proactive_clause_replaces_reactive_retry(self):
        # #1718's reactive clause keyed off "Operation not permitted"
        # as a retry trigger. Under #1734's workspace-write sandbox,
        # the model proactively requests elevation BEFORE the write,
        # so the reactive trigger is no longer in the contract.
        assert "Operation not permitted" not in GPT5_BEHAVIOR_CONTRACT, (
            "reactive 'Operation not permitted' trigger from #1718 was "
            "superseded by #1734's proactive elevation clause; the "
            "phrase should no longer appear in the contract"
        )
        assert "retry once" not in GPT5_BEHAVIOR_CONTRACT, (
            "the reactive 'retry once' wording was replaced by proactive "
            "elevation — request before, not retry after"
        )

    def test_proactive_clause_excludes_user_denial_as_trigger(self):
        # Carry-forward from #1718 codex review: "rejected by user"
        # means an audit-backed operator denial (#1690). Retrying it
        # ignores the operator's decision.
        assert "rejected by user" not in GPT5_BEHAVIOR_CONTRACT

    def test_proactive_clause_not_in_other_blocks(self):
        # Spot-check: the clause must not have been duplicated into
        # other contract sections by a future edit.
        for tag in ("<execution_policy>", "<output_contract>", "<completion_contract>"):
            start = GPT5_BEHAVIOR_CONTRACT.index(tag)
            end = GPT5_BEHAVIOR_CONTRACT.index(tag.replace("<", "</"))
            block = GPT5_BEHAVIOR_CONTRACT[start:end]
            assert "request elevation" not in block, (
                f"elevation clause must live in <tool_discipline>, "
                f"not duplicated into {tag}"
            )
