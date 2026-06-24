"""Tests for the OKF -> docs-site projection gate.

The publish gate is default-deny: only curated, public, non-stale docs are
emitted. These tests pin that contract so a future frontmatter change can't
silently leak internal docs onto a public site.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
spec = importlib.util.spec_from_file_location("build_docs_site", SCRIPTS / "build_docs_site.py")
assert spec and spec.loader
build_docs_site = importlib.util.module_from_spec(spec)
sys.modules["build_docs_site"] = build_docs_site
spec.loader.exec_module(build_docs_site)


def _write(root: Path, rel: str, frontmatter: dict, body: str = "# Title\n\nBody.\n") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = "\n".join(f"{k}: {v}" for k, v in frontmatter.items())
    path.write_text(f"---\n{fm}\n---\n\n{body}", encoding="utf-8")


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "guide.md", {"type": "User Guide", "privacy": "public", "status": "active"})
    _write(root, "secret.md", {"type": "User Guide", "privacy": "internal", "status": "active"})
    _write(root, "issue.md", {"type": "Issue Body", "privacy": "public", "status": "active"})
    _write(root, "stale.md", {"type": "Guide", "privacy": "public", "status": "needs-revalidation"})
    return root


def test_public_curated_active_doc_is_published(docs: Path):
    pages, _ = build_docs_site.select_pages(docs, include_stale=False)
    rels = {p["rel"].rsplit("/", 1)[-1] for p in pages}
    assert "guide.md" in rels


def test_internal_privacy_is_blocked(docs: Path):
    pages, skipped = build_docs_site.select_pages(docs, include_stale=False)
    rels = {p["rel"].rsplit("/", 1)[-1] for p in pages}
    assert "secret.md" not in rels
    assert any("secret.md" in s[0] and "privacy=internal" in s[1] for s in skipped)


def test_non_publishable_type_is_blocked(docs: Path):
    pages, skipped = build_docs_site.select_pages(docs, include_stale=False)
    rels = {p["rel"].rsplit("/", 1)[-1] for p in pages}
    assert "issue.md" not in rels
    assert any("issue.md" in s[0] and "type not publishable" in s[1] for s in skipped)


def test_stale_status_blocked_by_default_but_allowed_with_flag(docs: Path):
    default_pages, _ = build_docs_site.select_pages(docs, include_stale=False)
    assert all("stale.md" not in p["rel"] for p in default_pages)

    stale_pages, _ = build_docs_site.select_pages(docs, include_stale=True)
    assert any("stale.md" in p["rel"] for p in stale_pages)


def test_stale_page_renders_a_banner(docs: Path):
    stale_pages, _ = build_docs_site.select_pages(docs, include_stale=True)
    page = next(p for p in stale_pages if "stale.md" in p["rel"])
    rendered = build_docs_site.render_mintlify_page(page)
    assert "<Warning>" in rendered


def test_mint_json_groups_by_type(docs: Path):
    import json

    pages, _ = build_docs_site.select_pages(docs, include_stale=True)
    nav = json.loads(build_docs_site.render_mint_json(pages))["navigation"]
    groups = {g["group"] for g in nav}
    assert "User Guide" in groups
