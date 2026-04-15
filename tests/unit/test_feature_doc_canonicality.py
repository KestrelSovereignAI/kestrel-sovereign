"""Guardrails for canonical feature-document structure."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_feature_doc_declares_source_of_truth():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text()
    assert "Canonical source of truth" in text
    assert "Do not keep stale marketing counts here." in text


def test_canonical_feature_doc_distinguishes_core_and_package_features():
    """KESTREL_FEATURES.md must document both core and entry_point discovery."""
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text()
    assert "Core features" in text
    assert "Package features" in text
    assert "entry point" in text.lower() or "entry_point" in text


def test_canonical_feature_doc_lists_core_only_inventory():
    """The inventory section should clarify it lists core features only."""
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text()
    assert "core features only" in text.lower() or "core discoverable modules" in text.lower()


def test_legacy_archive_is_marked_historical():
    text = (PROJECT_ROOT / "docs/archive/KESTREL_FEATURES_legacy.md").read_text()
    assert "historical context only" in text
    assert "not the maintained source of truth" in text


def test_generator_prompt_does_not_hardcode_stale_metrics():
    text = (PROJECT_ROOT / "scripts/generate_feature_docs.py").read_text()
    assert "60+ API endpoints" not in text
    assert "28 feature modules" not in text
    assert "do not invent or hardcode counts" in text


def test_progress_review_script_uses_discovered_inventory_language():
    text = (PROJECT_ROOT / "scripts/convene_progress_review.py").read_text()
    assert "discovered feature module inventory" in text
    assert "28 feature modules" not in text


def test_investor_generated_doc_does_not_invent_unverified_metrics():
    text = (PROJECT_ROOT / "docs/generated/FEATURES_investor.md").read_text().lower()
    assert "independently audited" not in text
    assert "externally audited" not in text
    assert "over 100" not in text
    assert "more than 100" not in text
