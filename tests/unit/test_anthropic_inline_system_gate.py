"""The inline mid-conversation system-message gate is a capability matrix (#2846).

Anthropic ships mid-conversation ``{"role": "system"}`` turns on Claude Opus 5,
Opus 4.8, Fable 5 and Mythos 5 — but NOT on Sonnet 5, which postdates Opus 4.8.
Support is therefore an allowlist, not a version comparison: a ">= 4.8" test
would drop Fable/Mythos and would wrongly advertise the capability on Sonnet 5,
whose route rejects the turn.

Before this fix the gate matched only ``claude-opus-4-8``, so every current
production route fell back to relaying operator notices as visible user turns.
"""

import pytest

from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter


SUPPORTED = [
    "claude-opus-5",
    "claude-opus-5-20260601",
    "claude-opus-4-8",
    "claude-opus-4-8-20260115",
    "claude-fable-5",
    "claude-mythos-5",
]

UNSUPPORTED = [
    # Postdates Opus 4.8 but does NOT support inline system turns — the case a
    # version comparison gets wrong.
    "claude-sonnet-5",
    "claude-sonnet-5-20260210",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-haiku-4-5",
    # Legacy Claude 3 naming puts the version BEFORE the family.
    "claude-3-opus-20240229",
    # Mythos 5's invitation-only predecessor.
    "claude-mythos-preview",
    "",
    # Boundary cases: ids that merely START with a supported lineage. An
    # unanchored match would fail OPEN on these and wedge the route.
    "claude-opus-5-1",
    "claude-opus-5-fast",
    "claude-opus-4-80",
    "claude-fable-50",
    "claude-mythos-5-preview",
    "claude-opus-51",
    "not-claude-opus-5",
]


@pytest.mark.parametrize("model", SUPPORTED)
def test_supported_models_pass_the_gate(model):
    assert AnthropicAdapter._model_supports_inline_system(model) is True, (
        f"{model} supports mid-conversation system messages but was gated out"
    )


@pytest.mark.parametrize("model", UNSUPPORTED)
def test_unsupported_models_fail_the_gate(model):
    assert AnthropicAdapter._model_supports_inline_system(model) is False, (
        f"{model} does NOT support mid-conversation system messages; "
        "advertising it would send a role:'system' turn the route rejects"
    )


def test_sonnet_5_is_excluded_despite_postdating_opus_4_8():
    """The single case that proves this is a matrix, not an ordering."""
    assert AnthropicAdapter._model_supports_inline_system("claude-opus-4-8") is True
    assert AnthropicAdapter._model_supports_inline_system("claude-sonnet-5") is False


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-opus-5",
        "ANTHROPIC/claude-opus-5",
        "claude_opus_5",
        "Claude-Opus-5",
        # Route-qualified form (Vendor/Route/Model) — `_resolve_wire_model_id`
        # strips only the bare `anthropic/` prefix, so the gate drops the
        # route segment itself.
        "anthropic:plan/claude-opus-5",
        "anthropic:api/claude-opus-4-8",
        "anthropic:plan/claude-fable-5",
    ],
)
def test_gate_resolves_prefixed_and_normalized_ids(model):
    """Stored ids carry a vendor prefix and inconsistent casing/separators."""
    assert AnthropicAdapter._model_supports_inline_system(model) is True


def test_route_qualified_unsupported_model_still_fails():
    """Dropping the route segment must not smuggle an unsupported model in."""
    assert AnthropicAdapter._model_supports_inline_system(
        "anthropic:plan/claude-sonnet-5"
    ) is False


def test_gate_is_reached_only_on_a_supporting_route():
    """Route and model gates are ANDed — both must agree (#2009)."""
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    assert adapter._should_keep_inline_system("claude-opus-5", True) is True
    # Caller didn't ask for it.
    assert adapter._should_keep_inline_system("claude-opus-5", False) is False
    # Model can't take it.
    assert adapter._should_keep_inline_system("claude-sonnet-5", True) is False
