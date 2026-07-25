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
    context_contract = generate_feature_docs.extract_context_contract(source)
    aliases = generate_feature_docs.extract_non_bundled_surface_aliases(
        contract
    )

    body = generate_feature_docs.compose_generated_body(
        source,
        (
            f"{contract}\n\n{context_contract}\n\n"
            "# Audience view\n\nA neutral capability summary."
        ),
    )

    assert body.count(contract) == 1
    assert body.count(context_contract) == 1
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


@pytest.mark.parametrize(
    "claim",
    [
        "Conversation remains coherent regardless of the selected model.",
        "Context diagnostics exactly reproduce the production prompt.",
        "Automatic durable salvage is the default.",
        "Automatic salvage protects every prune.",
    ],
)
def test_generator_rejects_context_overclaims(claim):
    source = generate_feature_docs.SOURCE_FILE.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="context honesty"):
        generate_feature_docs.compose_generated_body(source, claim)


def test_generated_links_are_rebased_only_when_root_relative():
    text = (
        "[canonical](KESTREL_FEATURES.md) "
        "[architecture](../architecture/CONTEXT_SYSTEM_DESIGN.md) "
        "[external](https://example.com)"
    )

    normalized = generate_feature_docs.normalize_generated_links(text)

    assert "[canonical](../../KESTREL_FEATURES.md)" in normalized
    assert "[architecture](../architecture/CONTEXT_SYSTEM_DESIGN.md)" in normalized
    assert "[external](https://example.com)" in normalized


def test_checked_in_docs_copy_canonical_contracts_verbatim():
    source = generate_feature_docs.SOURCE_FILE.read_text(encoding="utf-8")
    contracts = (
        generate_feature_docs.extract_boundary_contract(source),
        generate_feature_docs.extract_context_contract(source),
    )

    for audience in generate_feature_docs.AUDIENCES:
        path = (
            generate_feature_docs.OUTPUT_DIR
            / f"FEATURES_{audience}.md"
        )
        text = path.read_text(encoding="utf-8")
        for contract in contracts:
            assert text.count(contract) == 1


def test_sync_protected_contracts_preserves_frontmatter_and_audience_prose(
    tmp_path,
    monkeypatch,
):
    source = generate_feature_docs.SOURCE_FILE.read_text(encoding="utf-8")
    source_path = tmp_path / "KESTREL_FEATURES.md"
    source_path.write_text(source, encoding="utf-8")
    output_dir = tmp_path / "docs" / "generated"
    output_dir.mkdir(parents=True)
    stale_contract = generate_feature_docs.extract_boundary_contract(source)
    header = "---\ntype: Generated Reference\n---\n\n"
    prose = "# Audience view\n\nKeep this prose. [source](KESTREL_FEATURES.md)\n"

    for audience in generate_feature_docs.AUDIENCES:
        (output_dir / f"FEATURES_{audience}.md").write_text(
            header + stale_contract + "\n\n" + prose,
            encoding="utf-8",
        )

    monkeypatch.setattr(generate_feature_docs, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generate_feature_docs, "SOURCE_FILE", source_path)
    monkeypatch.setattr(generate_feature_docs, "OUTPUT_DIR", output_dir)

    updated = generate_feature_docs.sync_protected_contracts()

    assert len(updated) == len(generate_feature_docs.AUDIENCES)
    context_contract = generate_feature_docs.extract_context_contract(source)
    for path in updated:
        text = path.read_text(encoding="utf-8")
        assert text.startswith(header)
        assert "# Audience view\n\nKeep this prose." in text
        assert text.count(stale_contract) == 1
        assert text.count(context_contract) == 1
        assert "[source](../../KESTREL_FEATURES.md)" in text


def test_generated_docs_explain_why_full_regeneration_is_not_deterministic_ci():
    readme = (
        generate_feature_docs.OUTPUT_DIR / "README.md"
    ).read_text(encoding="utf-8")

    assert "cannot be regenerated" in readme
    assert "byte-for-byte in CI" in readme
    assert "requires an external provider credential/model" in readme
    assert "--check" in readme
    assert "--sync-protected-contracts" in readme


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
