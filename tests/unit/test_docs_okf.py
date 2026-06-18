"""Tests for OKF documentation metadata tooling."""

from pathlib import Path
from types import SimpleNamespace

from scripts import docs_okf


def test_split_frontmatter_parses_yaml_mapping():
    frontmatter, body, error = docs_okf.split_frontmatter(
        "---\n"
        "type: Demo Script\n"
        "title: Console Demo\n"
        "tags: [demo, docs]\n"
        "---\n\n"
        "# Console Demo\n"
    )

    assert error is None
    assert frontmatter["type"] == "Demo Script"
    assert frontmatter["tags"] == ["demo", "docs"]
    assert body.lstrip().startswith("# Console Demo")


def test_validate_opt_in_okf_file(tmp_path: Path):
    doc = tmp_path / "demo.md"
    doc.write_text(
        "---\n"
        "type: Demo Evidence\n"
        "title: Feature Store Demo Evidence\n"
        "description: Generated screenshots and review evidence.\n"
        "generated: true\n"
        "---\n\n"
        "# Feature Store Demo Evidence\n",
        encoding="utf-8",
    )

    assert docs_okf.validate_files([doc]) == 0


def test_validate_all_requires_frontmatter(tmp_path: Path):
    doc = tmp_path / "plain.md"
    doc.write_text("# Plain Doc\n", encoding="utf-8")

    assert docs_okf.validate_files([doc], include_all=True) == 1


def test_inventory_includes_okf_metadata(tmp_path: Path):
    doc = tmp_path / "plan.md"
    doc.write_text(
        "---\n"
        "type: Migration Plan\n"
        "title: OKF Plan\n"
        "description: Plan metadata.\n"
        "status: proposed\n"
        "canonical: true\n"
        "---\n\n"
        "# OKF Plan\n",
        encoding="utf-8",
    )

    rows = docs_okf.build_inventory([doc])

    assert rows == [
        {
            "path": doc.as_posix(),
            "okf": True,
            "type": "Migration Plan",
            "title": "OKF Plan",
            "description": "Plan metadata.",
            "status": "proposed",
            "owner": "",
            "canonical": True,
            "generated": False,
            "tags": [],
        }
    ]


def test_resolve_input_paths_expands_directories(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    first = docs_dir / "first.md"
    second = docs_dir / "second.md"
    first.write_text("# First\n", encoding="utf-8")
    second.write_text("# Second\n", encoding="utf-8")

    paths = docs_okf.resolve_input_paths(
        SimpleNamespace(paths=[str(docs_dir)], root=docs_dir)
    )

    assert paths == [first.resolve(), second.resolve()]


def test_render_index_lists_okf_and_pending_docs(tmp_path: Path):
    okf = tmp_path / "okf.md"
    pending = tmp_path / "pending.md"
    okf.write_text(
        "---\n"
        "type: Audit Report\n"
        "title: Report\n"
        "description: Report metadata.\n"
        "status: snapshot\n"
        "---\n\n"
        "# Report\n",
        encoding="utf-8",
    )
    pending.write_text("# Pending\n", encoding="utf-8")

    rendered = docs_okf.render_index(tmp_path)

    assert "OKF concepts: 1" in rendered
    assert "Pending markdown files: 1" in rendered
    assert "Audit Report" in rendered
    assert "pending.md" in rendered


def test_render_log_lists_timestamped_okf_docs(tmp_path: Path):
    okf = tmp_path / "okf.md"
    okf.write_text(
        "---\n"
        "type: Audit Report\n"
        "title: Report\n"
        "description: Report metadata.\n"
        "timestamp: 2026-06-18T00:00:00Z\n"
        "status: snapshot\n"
        "---\n\n"
        "# Report\n",
        encoding="utf-8",
    )

    rendered = docs_okf.render_log(tmp_path)

    assert "2026-06-18T00:00:00+00:00" in rendered
    assert "Audit Report" in rendered
    assert "Report" in rendered


def test_write_or_check_generated_indexes(tmp_path: Path):
    okf = tmp_path / "okf.md"
    okf.write_text(
        "---\n"
        "type: Audit Report\n"
        "title: Report\n"
        "description: Report metadata.\n"
        "status: snapshot\n"
        "---\n\n"
        "# Report\n",
        encoding="utf-8",
    )

    assert docs_okf.write_or_check_generated(
        [tmp_path],
        filename="index.md",
        renderer=docs_okf.render_index,
        check=False,
    ) == 0
    assert docs_okf.write_or_check_generated(
        [tmp_path],
        filename="index.md",
        renderer=docs_okf.render_index,
        check=True,
    ) == 0


def test_generated_feature_docs_have_okf_metadata():
    assert docs_okf.check_generated_feature_docs() == 0


def test_may_2026_audit_workspace_is_okf_opted_in():
    root = docs_okf.PROJECT_ROOT / "docs" / "audit" / "documentation-2026-05"
    paths = docs_okf.markdown_files(root)

    assert paths
    assert docs_okf.validate_files(paths, include_all=True) == 0


def test_docs_tree_is_okf_complete():
    paths = docs_okf.markdown_files(docs_okf.PROJECT_ROOT / "docs")

    assert paths
    assert docs_okf.validate_files(paths, include_all=True) == 0


def test_human_indexes_link_okf_surfaces():
    project_root = docs_okf.PROJECT_ROOT
    docs_readme = (project_root / "docs" / "README.md").read_text()
    audit_readme = (project_root / "docs" / "audit" / "README.md").read_text()
    generated_readme = (project_root / "docs" / "generated" / "README.md").read_text()
    inventory = (
        project_root
        / "docs"
        / "audit"
        / "documentation-2026-05"
        / "DOCUMENTATION_INVENTORY.md"
    ).read_text()
    canonical = (
        project_root
        / "docs"
        / "audit"
        / "documentation-2026-05"
        / "CANONICAL_SOURCES.md"
    ).read_text()

    assert "audit/OKF_MIGRATION_PLAN.md" in docs_readme
    assert "audit/index.md" in docs_readme
    assert "generated/DEMO_EVIDENCE.md" in docs_readme
    assert "OKF_MIGRATION_PLAN.md" in audit_readme
    assert "index.md" in audit_readme
    assert "log.md" in audit_readme
    assert "254 OKF documents" in inventory
    assert "0 non-reserved markdown files are missing OKF frontmatter" in inventory
    assert "scripts/docs_okf.py" in canonical
    assert "scripts/generate_demo_evidence_docs.py" in generated_readme
