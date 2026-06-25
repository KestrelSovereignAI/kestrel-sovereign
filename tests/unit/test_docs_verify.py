"""Tests for documentation verification and render routing."""

import subprocess

import pytest

from scripts import docs_okf, docs_verify


def test_report_links_wrap_space_containing_targets():
    assert (
        docs_verify.report_link_for(
            "docs/research/GPU-Enabled Container Hosting Options for AI Workloads.md"
        )
        == "<../research/GPU-Enabled Container Hosting Options for AI Workloads.md>"
    )


def test_doc_relative_markdown_links_are_not_missing_code_refs():
    doc, error = docs_okf.read_okf_document(docs_okf.PROJECT_ROOT / "docs" / "README.md")

    assert error is None
    assert doc is not None
    _, missing_refs = docs_verify.classify_code_refs(doc)
    assert "demos/DEMO_SCRIPT.md" not in missing_refs


def test_verification_outputs_are_current():
    is_shallow = subprocess.check_output(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=docs_okf.PROJECT_ROOT,
        text=True,
    ).strip()
    if is_shallow == "true":
        pytest.skip("docs verification recent-PR ledger requires full git history")

    items = docs_verify.verify_docs(
        since=docs_verify.DEFAULT_SINCE,
        ignored_prs=docs_verify.DEFAULT_IGNORED_PRS,
    )

    report = docs_verify.render_report(
        items,
        since=docs_verify.DEFAULT_SINCE,
        ignored_prs=docs_verify.DEFAULT_IGNORED_PRS,
    )
    manifest = docs_verify.render_manifest(items)

    assert docs_verify.DEFAULT_REPORT.read_text(encoding="utf-8") == report
    assert docs_verify.DEFAULT_MANIFEST.read_text(encoding="utf-8") == manifest
    assert any(item.path == "docs/README.md" and item.render == "public" for item in items)
