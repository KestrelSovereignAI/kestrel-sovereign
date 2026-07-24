"""Tests for the feature-doc generation pipeline."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from datetime import datetime, timezone

import pytest
import yaml

from scripts import generate_feature_docs
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo


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


def test_okf_frontmatter_for_generated_docs_is_deterministic():
    header = generate_feature_docs.build_okf_frontmatter(
        "developer",
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        generated_at=datetime(2026, 6, 18, 12, 30, tzinfo=timezone.utc),
    )

    assert header.startswith("---\n")
    frontmatter = header.split("---\n", 2)[1]
    metadata = yaml.safe_load(frontmatter)

    assert metadata["type"] == "Generated Reference"
    assert metadata["audience"] == "developer"
    assert metadata["generated"] is True
    assert metadata["canonical"] is False
    assert metadata["source"] == "/KESTREL_FEATURES.md"
    assert metadata["generator"] == "scripts/generate_feature_docs.py"
    assert metadata["model"] == "anthropic/claude-sonnet-4-6"
    assert metadata["timestamp"] == "2026-06-18T12:30:00Z"


def test_checked_in_generated_docs_have_okf_metadata():
    assert generate_feature_docs.check_generated_docs() == 0


def test_compose_generated_body_preserves_boundary_contract_verbatim():
    source = generate_feature_docs.SOURCE_FILE.read_text(encoding="utf-8")
    contract = generate_feature_docs.extract_boundary_contract(source)
    aliases = generate_feature_docs.extract_non_bundled_surface_aliases(
        contract
    )

    body = generate_feature_docs.compose_generated_body(
        source,
        f"{contract}\n\n# Audience view\n\nA neutral capability summary.",
    )

    assert body.count(contract) == 1
    assert body.startswith(contract)
    assert "# Audience view" in body
    assert aliases["voice"] == ("voice",)
    assert aliases["github integration"] == (
        "github integration",
        "github app",
    )
    assert aliases["github"] == ("github",)


@pytest.mark.parametrize(
    "claim",
    [
        "Voice is a built-in capability.",
        "GitHub is a built-in capability.",
        "The core feature inventory includes Wallet.",
        "RunPod is a native integration.",
        "The kestrel-talon command ships with the base framework.",
    ],
)
def test_generator_rejects_non_bundled_surface_promotions(claim):
    source = generate_feature_docs.SOURCE_FILE.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="contradicts package ownership"):
        generate_feature_docs.compose_generated_body(source, claim)


def test_checked_in_docs_copy_the_canonical_boundary_contract_verbatim():
    source = generate_feature_docs.SOURCE_FILE.read_text(encoding="utf-8")
    contract = generate_feature_docs.extract_boundary_contract(source)

    for audience in generate_feature_docs.AUDIENCES:
        path = (
            generate_feature_docs.OUTPUT_DIR
            / f"FEATURES_{audience}.md"
        )
        assert path.read_text(encoding="utf-8").count(contract) == 1


def test_generated_docs_explain_why_full_regeneration_is_not_deterministic_ci():
    readme = (
        generate_feature_docs.OUTPUT_DIR / "README.md"
    ).read_text(encoding="utf-8")

    assert "cannot be regenerated" in readme
    assert "byte-for-byte in CI" in readme
    assert "requires an external provider credential/model" in readme
    assert "--check" in readme


def test_generator_uses_provider_default_resolution_for_anthropic():
    fake_anthropic = SimpleNamespace(Anthropic=lambda: object())
    with patch.object(generate_feature_docs, "resolve_provider_default", return_value="claude-opus-4-5-20251101"):
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
                _, model_name, provider = generate_feature_docs.get_client_and_model(
                    None,
                    refresh_discovery=False,
                )

    assert provider == "anthropic"
    assert model_name == "claude-opus-4-5-20251101"


def test_generator_uses_provider_default_resolution_for_openai():
    fake_openai = SimpleNamespace(OpenAI=lambda: object())
    with patch.object(generate_feature_docs, "resolve_provider_default", return_value="gpt-5.1"):
        with patch.dict("sys.modules", {"openai": fake_openai}):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
                _, model_name, provider = generate_feature_docs.get_client_and_model(
                    None,
                    refresh_discovery=False,
                )

    assert provider == "openai"
    assert model_name == "gpt-5.1"


def test_generator_refreshes_provider_discovery_before_default_resolution():
    fake_anthropic = SimpleNamespace(Anthropic=lambda: object())
    discovered = [
        ModelInfo(
            id="claude-sonnet-4-6",
            provider="anthropic",
            display_name="Claude Sonnet 4.6",
            category=ModelCategory.CHAT,
        )
    ]

    with patch.object(generate_feature_docs, "_refresh_provider_cache", return_value=discovered) as refresh:
        with patch.object(generate_feature_docs, "resolve_provider_default", return_value="claude-sonnet-4-6") as resolve:
            with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
                with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
                    _, model_name, provider = generate_feature_docs.get_client_and_model(None)

    assert provider == "anthropic"
    assert model_name == "claude-sonnet-4-6"
    refresh.assert_called_once_with("anthropic")
    assert resolve.call_args.kwargs["cached_models"] == discovered
