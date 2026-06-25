"""Tests for documentation verification and render routing."""

import json

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
    # The committed ledger/manifest are a pure function of doc content (no git
    # history), so this is deterministic and never needs a full-history clone.
    items = docs_verify.verify_docs(
        since=docs_verify.DEFAULT_SINCE,
        ignored_prs=docs_verify.DEFAULT_IGNORED_PRS,
    )

    report = docs_verify.render_report(items)
    manifest = docs_verify.render_manifest(items)

    assert docs_verify.DEFAULT_REPORT.read_text(encoding="utf-8") == report
    assert docs_verify.DEFAULT_MANIFEST.read_text(encoding="utf-8") == manifest
    assert any(item.path == "docs/README.md" and item.render == "public" for item in items)


def test_committed_outputs_carry_no_head_relative_activity_data():
    # Guard against regressions: the committed artifacts must never embed
    # recent-PR data, which would restale them on every unrelated merge.
    items = docs_verify.verify_docs(
        since=docs_verify.DEFAULT_SINCE,
        ignored_prs=docs_verify.DEFAULT_IGNORED_PRS,
    )
    manifest = docs_verify.render_manifest(items)
    assert "recent_prs" not in manifest
    assert "merged_at" not in manifest
    for route in json.loads(manifest)["routes"]:
        assert "recent_prs" not in route
        assert all(f not in docs_verify.ACTIVITY_FINDINGS for f in route["findings"])


def test_activity_view_is_live_and_separate():
    # The recent-PR review queue is still available, computed live, not committed.
    items = docs_verify.verify_docs(
        since=docs_verify.DEFAULT_SINCE,
        ignored_prs=docs_verify.DEFAULT_IGNORED_PRS,
        with_activity=True,
    )
    rendered = docs_verify.render_activity(items, since=docs_verify.DEFAULT_SINCE)
    assert "recent prs" in rendered.lower()
