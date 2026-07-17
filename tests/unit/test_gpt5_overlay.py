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


class TestHostToolBoundaryDiscipline:
    """#1965: GPT-5 must not seek provider-native tool bypasses."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "sole authority",
            "provider-native execution path",
            "filesystem",
            "report the action as blocked",
            "operator-denied results as terminal",
        ],
    )
    def test_contract_carries_host_tool_boundary(self, phrase):
        assert phrase in GPT5_BEHAVIOR_CONTRACT, (
            f"GPT5 contract must mention {phrase!r} so the model knows "
            f"that host tools are the only authorized action path"
        )

    def test_boundary_clause_inside_tool_discipline(self):
        td_start = GPT5_BEHAVIOR_CONTRACT.index("<tool_discipline>")
        td_end = GPT5_BEHAVIOR_CONTRACT.index("</tool_discipline>")
        td_block = GPT5_BEHAVIOR_CONTRACT[td_start:td_end]
        assert "provider-native execution path" in td_block
        assert "sole authority" in td_block

    def test_boundary_clause_removes_native_sandbox_guidance(self):
        assert "Operation not permitted" not in GPT5_BEHAVIOR_CONTRACT, (
            "provider sandbox errors are not an authorized retry signal"
        )
        assert "retry once" not in GPT5_BEHAVIOR_CONTRACT, (
            "a blocked host action must not be retried through another path"
        )
        assert "request elevation" not in GPT5_BEHAVIOR_CONTRACT
        assert "freely read anywhere" not in GPT5_BEHAVIOR_CONTRACT

    def test_proactive_clause_excludes_user_denial_as_trigger(self):
        # Carry-forward from #1718 codex review: "rejected by user"
        # means an audit-backed operator denial (#1690). Retrying it
        # ignores the operator's decision.
        assert "rejected by user" not in GPT5_BEHAVIOR_CONTRACT

    def test_boundary_clause_not_in_other_blocks(self):
        for tag in ("<execution_policy>", "<output_contract>", "<completion_contract>"):
            start = GPT5_BEHAVIOR_CONTRACT.index(tag)
            end = GPT5_BEHAVIOR_CONTRACT.index(tag.replace("<", "</"))
            block = GPT5_BEHAVIOR_CONTRACT[start:end]
            assert "provider-native execution path" not in block, (
                f"host-tool boundary must live in <tool_discipline>, "
                f"not duplicated into {tag}"
            )
