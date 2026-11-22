"""Reader-path regressions over the generated Starlight HTML and Pagefind index.

Install the generated site's npm dependencies first, then run this module.
The fixture builds a fresh site; it never serves HTTP or calls an external API.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

from scripts.docs_okf import split_frontmatter

ROOT = Path(__file__).resolve().parents[2]
CURRENT = "principles/kestrel-constitution-v2"
FIRST = "principles/kestrel-constitution"
PROPOSAL = "proposals/constitution-and-pledge/constitution"
RELEASE_SOURCE = "kestrel_sovereign/data/constitutions/v2/KESTREL_CONSTITUTION.md"
RELEASE_URL = (
    "https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/"
    f"constitution-v2.0.0/{RELEASE_SOURCE}"
)
PROTECTED = (
    "docs/principles/KESTREL_CONSTITUTION.md",
    "docs/proposals/constitution-and-pledge/CONSTITUTION.md",
    "kestrel_sovereign/data/KESTREL_CONSTITUTION.md",
    RELEASE_SOURCE,
    "kestrel_sovereign/data/constitutions/v2/PLEDGE.md",
)


class ReaderPage(HTMLParser):
    def __init__(self, html: str):
        super().__init__()
        self.links = []
        self.headings = []
        self.notices = []
        self.nav = []
        self.anchor = None
        self.heading = None
        self.notice = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "nav":
            self.nav.append(attrs.get("aria-label"))
        elif tag == "a":
            self.anchor = [attrs.get("href", ""), [], "Main" in self.nav]
        elif tag == "h1":
            self.heading = []
        elif tag == "aside" and "starlight-aside" in attrs.get("class", "").split():
            self.notice = []

    def handle_data(self, data):
        for target in (self.heading, self.notice):
            if target is not None:
                target.append(data)
        if self.anchor is not None:
            self.anchor[1].append(data)

    def handle_endtag(self, tag):
        if tag == "nav":
            self.nav.pop()
        elif tag == "a" and self.anchor is not None:
            href, text, in_sidebar = self.anchor
            self.links.append((href, "".join(text).strip(), in_sidebar))
            self.anchor = None
        elif tag == "h1" and self.heading is not None:
            self.headings.append("".join(self.heading).strip())
            self.heading = None
        elif tag == "aside" and self.notice is not None:
            self.notices.append("".join(self.notice).strip())
            self.notice = None


def _digests():
    return {
        rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        for rel in PROTECTED
    }


@pytest.fixture(scope="module", params=["/kestrel-sovereign", ""], ids=["pages", "domain"])
def handbook(tmp_path_factory, request):
    dependencies = Path(os.environ.get("DOCS_SITE_NODE_MODULES", ROOT / "site/node_modules"))
    if not (dependencies / ".bin/astro").is_file():
        message = "Generate site/ and run npm install there before these rendered-site tests."
        if "DOCS_SITE_NODE_MODULES" in os.environ:
            pytest.fail(message)
        pytest.skip(message)
    before = _digests()
    site = tmp_path_factory.mktemp("handbook") / "site"
    subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/build_docs_site.py"), "build",
            "--site-root", str(site), "--base", request.param,
        ],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=60,
    )
    # Astro resolves its own files relative to the project, so dependencies
    # must live inside this fresh project rather than behind a directory link.
    shutil.copytree(dependencies, site / "node_modules", symlinks=True)
    npm = shutil.which("npm")
    assert npm is not None, "npm is required for the rendered-site regression gate"
    result = subprocess.run(
        [npm, "run", "build"], cwd=site, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return site, request.param, before, _digests()


def _read_page(site, slug=""):
    return ReaderPage((site / "dist" / slug / "index.html").read_text(encoding="utf-8"))


def test_home_chooses_second_edition(handbook):
    site, base, _, _ = handbook
    home = _read_page(site)
    choices = [(href, title) for href, title, sidebar in home.links if not sidebar]
    assert (f"{base}/{CURRENT}/", "The Kestrel Constitution") in choices
    assert (f"{base}/{FIRST}/", "The Kestrel Constitution") not in choices


def test_current_edition_precedes_archive_in_home_and_sidebar(handbook):
    site, base, _, _ = handbook
    home = _read_page(site)
    for sidebar in (False, True):
        routes = [href.rstrip("/") for href, _, in_sidebar in home.links if sidebar == in_sidebar]
        assert routes.index(f"{base}/{CURRENT}") < routes.index(f"{base}/{FIRST}")


def test_sidebar_chooses_second_edition(handbook):
    site, base, _, _ = handbook
    for slug in ("", FIRST, PROPOSAL):
        links = [(href.rstrip("/"), title) for href, title, sidebar in _read_page(site, slug).links if sidebar]
        assert (f"{base}/{CURRENT}", "The Kestrel Constitution") in links
        assert (f"{base}/{FIRST}", "The Kestrel Constitution (first edition)") in links
        assert (f"{base}/{FIRST}", "The Kestrel Constitution") not in links


def test_search_indexes_second_edition_under_constitution_title(handbook):
    site, _, _, _ = handbook
    fragments = []
    for path in (site / "dist/pagefind/fragment").glob("*.pf_fragment"):
        raw = gzip.decompress(path.read_bytes())
        assert raw.startswith(b"pagefind_dcd"), "unexpected Pagefind fragment format"
        fragments.append(json.loads(raw.removeprefix(b"pagefind_dcd")))
    current = [entry for entry in fragments if entry["url"].rstrip("/").endswith(f"/{CURRENT}")]
    assert len(current) == 1, "the second edition must be in the generated search index"
    assert current[0]["meta"]["title"] == "The Kestrel Constitution"
    assert "Second constitutional release" in current[0]["content"]
    first = next(entry for entry in fragments if entry["url"].rstrip("/").endswith(f"/{FIRST}"))
    assert first["meta"]["title"] == "The Kestrel Constitution (first edition)"


@pytest.mark.parametrize("query", ["Kestrel Constitution", "Constitution"])
def test_pagefind_queries_prioritize_released_constitution(handbook, query):
    site, base, _, _ = handbook
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("query_pagefind.mjs")),
         str(site / "dist"), base, query],
        check=True, capture_output=True, text=True, timeout=30,
    )
    hits = json.loads(result.stdout)
    urls = [hit["url"].rstrip("/") for hit in hits[query]]
    current = f"{base}/{CURRENT}"
    assert current in urls, f"{query!r} must find the released edition in the top five: {hits[query]}"
    # Leave headroom for corpus growth rather than barely passing at fifth.
    assert urls.index(current) < 2, f"{query!r} should prioritize the current text: {hits[query]}"
    assert hits[query][urls.index(current)]["title"] == "The Kestrel Constitution"
    # Removing the UI adapter must fail too: a passing engine configured only
    # by the test would say nothing about what the reader's dialog searches.
    bundles = "\n".join(path.read_text() for path in (site / "dist/_astro").glob("*.js"))
    assert "released_title" in bundles and "ranking:" in bundles


def test_first_edition_bookmark_has_notice_and_direct_current_link(handbook):
    site, _, _, _ = handbook
    page = _read_page(site, FIRST)
    assert page.headings == ["The Kestrel Constitution (first edition)"]
    assert "first edition" in page.notices[0].lower()
    assert "second edition" in page.notices[0].lower()
    assert any(href == RELEASE_URL for href, _, sidebar in page.links if not sidebar)
    source = (site / f"src/content/docs/{FIRST}.md").read_text(encoding="utf-8")
    assert source.index(":::note") < source.index("## Preamble")


def test_proposal_has_historical_notice_and_direct_current_link(handbook):
    site, _, _, _ = handbook
    page = _read_page(site, PROPOSAL)
    assert "historical draft" in page.headings[0].lower()
    assert "historical drafting material" in page.notices[0].lower()
    assert any(href == RELEASE_URL for href, _, sidebar in page.links if not sidebar)
    source = (site / f"src/content/docs/{PROPOSAL}.md").read_text(encoding="utf-8")
    assert source.index(":::note") < source.index("Unsigned proposal")


def test_published_words_and_protected_source_digests(handbook):
    site, _, before, after = handbook
    assert before == after, "the build must not change protected or historical source bytes"
    assert after[RELEASE_SOURCE] == "d0949dba4b1969bc11098f64d56d38490ea59ddcbe91c40b7775f5d506408908"
    _, body, error = split_frontmatter((ROOT / RELEASE_SOURCE).read_text(encoding="utf-8"))
    assert error is None
    current = site / f"src/content/docs/{CURRENT}.md"
    assert current.is_file(), "publish the released package source in the handbook"
    projected = current.read_text(encoding="utf-8")
    # The theme supplies the H1 and the one relative link needs a public URL.
    # Every governing word otherwise follows the released package verbatim.
    expected = body.lstrip().split("\n", 1)[1].lstrip().rstrip()
    expected = expected.replace(
        "(PLEDGE.md)", f"({RELEASE_URL.rsplit('/', 1)[0]}/PLEDGE.md)"
    )
    assert projected.endswith(expected + "\n")
    page = _read_page(site, CURRENT)
    assert page.headings == ["The Kestrel Constitution"]
    assert "second edition" in page.notices[0].lower()
    assert any(href == RELEASE_URL for href, _, sidebar in page.links if not sidebar)
