"""Tests for documentation verification and render routing."""

import json
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


def test_worktree_inventory_ignores_scratch_and_requires_new_source_to_be_staged(
    tmp_path, monkeypatch
):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    source = tmp_path / "tests" / "unit" / "test_new_source.py"
    source.parent.mkdir(parents=True)
    source.write_text("# new repository evidence\n", encoding="utf-8")
    scratch = tmp_path / "tests" / "unit" / "scratch.py"
    scratch.write_text("# must not affect committed artifacts\n", encoding="utf-8")

    monkeypatch.setattr(docs_verify, "PROJECT_ROOT", tmp_path)
    docs_verify.refresh_worktree_inventory()
    try:
        before_staging = docs_verify.worktree_paths()

        subprocess.run(
            ["git", "add", source.relative_to(tmp_path)],
            cwd=tmp_path,
            check=True,
        )
        docs_verify.refresh_worktree_inventory()
        after_staging = docs_verify.worktree_paths()

        assert before_staging == frozenset()
        assert after_staging == frozenset({"tests/unit/test_new_source.py"})
        assert scratch.relative_to(tmp_path).as_posix() not in after_staging
    finally:
        docs_verify.refresh_worktree_inventory()


def test_worktree_inventory_filters_deletions_and_refreshes_between_verifications(
    tmp_path, monkeypatch
):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source = tmp_path / "scripts" / "evidence.py"
    source.parent.mkdir()
    source.write_text("# tracked evidence\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    monkeypatch.setattr(docs_verify, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(docs_verify, "DOCS_ROOT", docs_dir)
    docs_verify.refresh_worktree_inventory()
    try:
        docs_verify.verify_docs(since=docs_verify.DEFAULT_SINCE)
        assert "scripts/evidence.py" in docs_verify.worktree_paths()

        source.unlink()
        docs_verify.verify_docs(since=docs_verify.DEFAULT_SINCE)
        assert "scripts/evidence.py" not in docs_verify.worktree_paths()
    finally:
        docs_verify.refresh_worktree_inventory()


def test_tracked_symlink_is_inventory_evidence_but_broken_link_is_missing(
    tmp_path, monkeypatch
):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    target = tmp_path / "scripts" / "target.py"
    target.parent.mkdir()
    target.write_text("# target\n", encoding="utf-8")
    link = tmp_path / "scripts" / "linked.py"
    try:
        link.symlink_to(target.name)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    monkeypatch.setattr(docs_verify, "PROJECT_ROOT", tmp_path)
    docs_verify.refresh_worktree_inventory()
    try:
        assert "scripts/linked.py" in docs_verify.worktree_paths()
        target.unlink()
        docs_verify.refresh_worktree_inventory()
        assert "scripts/linked.py" in docs_verify.worktree_paths()
        assert docs_verify.resolve_doc_link(
            docs_dir / "example.md", "../scripts/linked.py"
        ) == target
        assert not target.exists()
        assert not docs_verify.repo_path_exists("scripts/target.py")
        doc = docs_verify.docs_okf.OkfDocument(
            path=docs_dir / "example.md",
            frontmatter={},
            body="[broken](../scripts/linked.py)",
        )
        assert docs_verify.missing_markdown_links(doc) == (
            "../scripts/linked.py",
        )
    finally:
        docs_verify.refresh_worktree_inventory()


def test_verification_outputs_are_current():
    # The committed ledger/manifest depend on documentation and the repository
    # path inventory, not git history, so no full-history clone is required.
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


def test_generated_docs_have_no_volatile_link_or_code_refs():
    # #1971: REPO_MAP.md (and other generated docs) embedded a repo-size-
    # dependent code-ref set (~1400 entries) that grew on every nightly regen
    # and restaled the committed ledger/manifest, redding out main and blocking
    # all merges. Generated docs must stay in render routing but carry empty,
    # stable link/ref sets so out-of-band regeneration never desyncs the
    # committed artifacts.
    items = docs_verify.verify_docs(
        since=docs_verify.DEFAULT_SINCE,
        ignored_prs=docs_verify.DEFAULT_IGNORED_PRS,
    )
    repo_map = next((i for i in items if i.path == "docs/audit/REPO_MAP.md"), None)
    assert repo_map is not None, "REPO_MAP.md should still be verified and routed"
    assert repo_map.render == "public", "generated docs must keep their render routing"
    assert repo_map.existing_code_refs == (), "generated docs must not embed a volatile code-ref set"
    assert repo_map.missing_code_refs == ()
    assert repo_map.missing_links == ()
    assert "missing_code_refs" not in repo_map.findings
    assert "missing_local_links" not in repo_map.findings


def test_activity_view_is_live_and_separate():
    # The recent-PR review queue is still available, computed live, not committed.
    items = docs_verify.verify_docs(
        since=docs_verify.DEFAULT_SINCE,
        ignored_prs=docs_verify.DEFAULT_IGNORED_PRS,
        with_activity=True,
    )
    rendered = docs_verify.render_activity(items, since=docs_verify.DEFAULT_SINCE)
    assert "recent prs" in rendered.lower()
