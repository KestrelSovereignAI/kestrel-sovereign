"""Guardrails for canonical feature-document structure."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_feature_doc_declares_source_of_truth():
    text = (PROJECT_ROOT / "KESTREL_FEATURES.md").read_text()
    assert "Canonical source of truth" in text
    assert "Do not keep stale marketing counts here." in text


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
