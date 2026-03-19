"""Contracts for model-selecting operational scripts."""

from unittest.mock import patch

from scripts.verify_character_consistency import resolve_verification_model


def test_character_verification_script_uses_resolved_anthropic_default():
    with patch(
        "scripts.verify_character_consistency.resolve_provider_default",
        return_value="claude-sonnet-4-6-latest",
    ):
        assert resolve_verification_model() == "claude-sonnet-4-6-latest"


def test_character_verification_script_allows_explicit_override():
    with patch.dict(
        "os.environ",
        {"CHARACTER_VERIFY_MODEL": "claude-opus-override"},
        clear=False,
    ):
        assert resolve_verification_model() == "claude-opus-override"
