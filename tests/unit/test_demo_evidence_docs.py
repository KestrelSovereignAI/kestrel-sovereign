"""Tests for generated demo evidence documentation."""

from datetime import datetime, timezone

from scripts import generate_demo_evidence_docs


def test_discovers_existing_demo_evidence():
    demos = generate_demo_evidence_docs.discover_demo_evidence()
    names = {demo.name for demo in demos}

    assert "technical" in names
    assert "feature-store" in names
    assert "TEMPLATE" not in names

    technical = next(demo for demo in demos if demo.name == "technical")
    assert technical.rel_demo_script == "/demos/technical/demo.cjs"
    assert technical.rel_eye_config == "/demos/technical/eye.toml"
    assert technical.screenshot_count > 0


def test_render_demo_evidence_index_has_okf_metadata():
    content = generate_demo_evidence_docs.render(
        generate_demo_evidence_docs.discover_demo_evidence(),
        generated_at=datetime(2026, 6, 18, tzinfo=timezone.utc),
    )

    assert content.startswith("---\ntype: Generated Reference\n")
    assert "title: Kestrel Demo Evidence Index" in content
    assert "generator: scripts/generate_demo_evidence_docs.py" in content
    assert "| `technical` | `/demos/technical/demo.cjs` | `/demos/technical/eye.toml` |" in content
    assert "kestrel-eye review --config demos/technical/eye.toml" in content


def test_checked_in_demo_evidence_index_is_current():
    expected = generate_demo_evidence_docs.render(
        generate_demo_evidence_docs.discover_demo_evidence(),
        generated_at=datetime(2026, 6, 18, tzinfo=timezone.utc),
    )

    assert generate_demo_evidence_docs.check_output(expected) == 0


def test_user_docs_link_to_demo_evidence():
    project_root = generate_demo_evidence_docs.PROJECT_ROOT
    user_index = (project_root / "docs" / "user-documentation" / "README.md").read_text()
    demo_script = (project_root / "docs" / "demos" / "DEMO_SCRIPT.md").read_text()

    assert "../generated/DEMO_EVIDENCE.md" in user_index
    assert "../../demos/privacy-modes/demo.cjs" in user_index
    assert "../../demos/technical/eye.toml" in demo_script
    assert "../generated/DEMO_EVIDENCE.md" in demo_script
