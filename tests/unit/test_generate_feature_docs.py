"""Tests for the feature-doc generation pipeline."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import generate_feature_docs


def test_generator_uses_canonical_source_path():
    project_root = Path(__file__).resolve().parents[2]
    assert generate_feature_docs.SOURCE_FILE == project_root / "KESTREL_FEATURES.md"


def test_dry_run_returns_expected_output_paths(capsys):
    project_root = Path(__file__).resolve().parents[2]

    developer_path = generate_feature_docs.generate("developer", dry_run=True)
    user_path = generate_feature_docs.generate("user", dry_run=True)
    investor_path = generate_feature_docs.generate("investor", dry_run=True)

    assert developer_path == project_root / "docs/generated/FEATURES_developer.md"
    assert user_path == project_root / "docs/generated/FEATURES_user.md"
    assert investor_path == project_root / "docs/generated/FEATURES_investor.md"

    output = capsys.readouterr().out
    assert "DRY RUN: developer" in output
    assert "DRY RUN: user" in output
    assert "DRY RUN: investor" in output


def test_generator_uses_provider_default_resolution_for_anthropic():
    fake_anthropic = SimpleNamespace(Anthropic=lambda: object())
    with patch.object(generate_feature_docs, "resolve_provider_default", return_value="claude-opus-4-5-20251101"):
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
                _, model_name, provider = generate_feature_docs.get_client_and_model(None)

    assert provider == "anthropic"
    assert model_name == "claude-opus-4-5-20251101"


def test_generator_uses_provider_default_resolution_for_openai():
    fake_openai = SimpleNamespace(OpenAI=lambda: object())
    with patch.object(generate_feature_docs, "resolve_provider_default", return_value="gpt-5.1"):
        with patch.dict("sys.modules", {"openai": fake_openai}):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
                _, model_name, provider = generate_feature_docs.get_client_and_model(None)

    assert provider == "openai"
    assert model_name == "gpt-5.1"
