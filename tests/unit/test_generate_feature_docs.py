"""Tests for the feature-doc generation pipeline."""

from pathlib import Path

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
