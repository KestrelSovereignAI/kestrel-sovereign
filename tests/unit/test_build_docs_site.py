"""Tests for the OKF -> docs-site projection gate.

The publish gate is default-deny: only curated, public, non-stale docs are
emitted. These tests pin that contract so a future frontmatter change can't
silently leak internal docs onto a public site.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
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
    # render_channel routing (from docs_verify): public / internal / archive.
    _write(root, "guide.md", {"type": "User Guide", "privacy": "public", "status": "active"})
    _write(root, "secret.md", {"type": "User Guide", "privacy": "internal", "status": "active"})
    _write(root, "issue.md", {"type": "Issue Body", "privacy": "public", "status": "active"})
    _write(root, "unrev.md", {"type": "Guide", "privacy": "public", "status": "needs-revalidation"})
    _write(root, "snap.md", {"type": "Guide", "privacy": "public", "status": "snapshot"})
    return root


def test_public_channel_doc_is_published(docs: Path):
    pages, _ = build_docs_site.select_pages(docs)
    rels = {p["rel"].rsplit("/", 1)[-1] for p in pages}
    assert "guide.md" in rels


def test_internal_privacy_routes_off_public(docs: Path):
    pages, skipped = build_docs_site.select_pages(docs)
    rels = {p["rel"].rsplit("/", 1)[-1] for p in pages}
    assert "secret.md" not in rels
    assert any("secret.md" in s[0] and "render=internal" in s[1] for s in skipped)


def test_internal_type_routes_off_public(docs: Path):
    pages, skipped = build_docs_site.select_pages(docs)
    rels = {p["rel"].rsplit("/", 1)[-1] for p in pages}
    assert "issue.md" not in rels
    assert any("issue.md" in s[0] and "render=internal" in s[1] for s in skipped)


def test_needs_revalidation_routes_internal_not_public(docs: Path):
    pages, skipped = build_docs_site.select_pages(docs)
    assert all("unrev.md" not in p["rel"] for p in pages)
    assert any("unrev.md" in s[0] and "render=internal" in s[1] for s in skipped)


def test_archive_channel_is_opt_in_with_banner(docs: Path):
    # snapshot status routes to the archive channel, withheld by default.
    default_pages, _ = build_docs_site.select_pages(docs)
    assert all("snap.md" not in p["rel"] for p in default_pages)

    archive_pages, _ = build_docs_site.select_pages(docs, channels={"public", "archive"})
    page = next(p for p in archive_pages if "snap.md" in p["rel"])
    assert page["channel"] == "archive"
    rendered = build_docs_site.StarlightEmitter().render_page(page)
    assert ":::note" in rendered  # snapshot banner


def test_starlight_tree_has_config_and_grouped_sidebar(docs: Path):
    import json

    pages, _ = build_docs_site.select_pages(docs)
    tree = build_docs_site.StarlightEmitter().tree(pages)
    assert "package.json" in tree
    assert "src/content.config.ts" in tree
    assert "src/content/docs/index.md" in tree

    config = tree["astro.config.mjs"]
    assert '"User Guide"' in config  # sidebar group label
    # base path flows into the generated Astro config
    assert "/kestrel-sovereign" in config


def test_rewrite_links_published_repo_and_external(tmp_path: Path):
    repo = tmp_path
    (repo / "docs" / "guides").mkdir(parents=True)
    (repo / "server.py").write_text("x", encoding="utf-8")
    src_dir = repo / "docs"
    published = {"docs/guides/building-features.md": "guides/building-features"}

    body = (
        "See [guide](guides/building-features.md#vocab), "
        "[source](../server.py), "
        "[missing](../nope.md), "
        "[site](https://example.com)."
    )
    out = build_docs_site.rewrite_local_links(
        body, src_dir, base="/kestrel-sovereign", published=published, repo_root=repo
    )
    # published doc -> site slug (with anchor preserved)
    assert "(/kestrel-sovereign/guides/building-features/#vocab)" in out
    # real repo file -> GitHub blob URL
    assert f"({build_docs_site.GITHUB_REPO_URL}/blob/main/server.py)" in out
    # unresolvable + external are untouched
    assert "(../nope.md)" in out
    assert "(https://example.com)" in out


def test_rewrite_links_ignores_images(tmp_path: Path):
    # an image embed (leading !) must not be treated as a doc link
    body = "![alt](../img.png)"
    out = build_docs_site.rewrite_local_links(
        body, tmp_path, base="", published={}, repo_root=tmp_path
    )
    assert out == body


def test_mintlify_emitter_groups_by_type(docs: Path):
    import json

    pages, _ = build_docs_site.select_pages(docs)
    tree = build_docs_site.MintlifyEmitter().tree(pages)
    nav = json.loads(tree["mint.json"])["navigation"]
    groups = {g["group"] for g in nav}
    assert "User Guide" in groups


@pytest.mark.parametrize("body", ["no metadata", "---\ninvalid: [\n---\n"])
def test_release_source_requires_valid_metadata(docs: Path, body):
    shutil.copyfile(SCRIPTS.parent / "docs/README.md", docs / "README.md")
    source = docs.parent / build_docs_site.CONSTITUTION_RELEASE_PATH
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match="Published Constitution has invalid OKF metadata"):
        build_docs_site.select_pages(docs)


def test_package_source_links_use_projected_route_and_release_revision(tmp_path: Path):
    source = tmp_path / "CONSTITUTION.md"
    pledge = tmp_path / "PLEDGE.md"
    pledge.write_text("Pledge", encoding="utf-8")
    pages = [{"src": source, "rel": "docs/principles/KESTREL_CONSTITUTION_V2.md"}]
    published = build_docs_site._published_map(pages)
    assert published[build_docs_site.display_path(source)] == "principles/kestrel-constitution-v2"
    rewritten = build_docs_site.rewrite_local_links(
        "[pledge](PLEDGE.md)", tmp_path, base="", published=published,
        repo_root=tmp_path, revision="constitution-v2.0.0",
    )
    assert rewritten == f"[pledge]({build_docs_site.GITHUB_REPO_URL}/blob/constitution-v2.0.0/PLEDGE.md)"


def test_selected_corpus_owns_constitution_release(tmp_path: Path):
    """CLI relocation keeps both editions and links in the selected checkout."""
    root = SCRIPTS.parent
    relocated = tmp_path / "relocated"
    sources = (
        "docs/README.md", "docs/principles/KESTREL_CONSTITUTION.md",
        "docs/proposals/constitution-and-pledge/CONSTITUTION.md",
        "kestrel_sovereign/data/constitutions/v2/KESTREL_CONSTITUTION.md",
        "kestrel_sovereign/data/constitutions/v2/PLEDGE.md",
    )
    for rel in sources:
        target = relocated / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, target)
    before = {rel: (relocated / rel).read_bytes() for rel in sources}
    site = tmp_path / "site"
    subprocess.run(
        [sys.executable, str(SCRIPTS / "build_docs_site.py"), "build",
         "--docs-root", str(relocated / "docs"), "--site-root", str(site)],
        check=True, capture_output=True, text=True,
    )
    content = site / "src/content/docs"
    current = content / "principles/kestrel-constitution-v2.md"
    assert current.is_file(), "a relocated handbook must publish its adjacent released edition"
    assert "published second edition" in current.read_text()
    assert "constitution-v2.0.0/kestrel_sovereign/data/constitutions/v2/PLEDGE.md" in current.read_text()
    assert "first edition" in (content / "principles/kestrel-constitution.md").read_text()
    assert before == {rel: (relocated / rel).read_bytes() for rel in sources}
    pages, _ = build_docs_site.select_pages(relocated / "docs")
    assert all(p["src"].is_relative_to(relocated) for p in pages)
    current_page = next(p for p in pages if p["rel"] == build_docs_site.CONSTITUTION_RELEASE_PAGE)
    assert current_page["src"] == relocated / sources[3]

    # Even identical leaf names and an adjacent package are not enough to
    # turn someone else's corpus into the Kestrel handbook.
    _write(relocated / "docs", "README.md", {
        "title": "Custom handbook", "type": "Documentation Index",
        "resource": "/docs/README.md", "privacy": "public", "status": "active",
    })
    pages, _ = build_docs_site.select_pages(relocated / "docs")
    assert len(pages) == 3
    first = next(p for p in pages if p["src"] == relocated / sources[1])
    assert first["rel"] == build_docs_site.display_path(relocated / sources[1])
    assert first["title"] == "The Kestrel Constitution (first edition)"
    assert "publication_notice" not in first
    assert "repo_root" not in first
    assert not any(p["rel"] == build_docs_site.CONSTITUTION_RELEASE_PAGE for p in pages)


def test_relocated_handbook_never_borrows_another_checkouts_release(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    shutil.copyfile(SCRIPTS.parent / "docs/README.md", docs / "README.md")
    with pytest.raises(FileNotFoundError):
        build_docs_site.select_pages(docs)
