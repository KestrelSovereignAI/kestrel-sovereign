"""Tests for the repository-local relative-link checker.

Covers the acceptance criteria of issue #2681 for
``scripts/check_docs_links.py``:

* the pre-move QUICKSTART paths reproduce as a failing fixture and the current
  file passes after correction,
* missing files and missing local anchors are reported with actionable output,
* valid percent-encoded paths and parenthesised destinations are handled,
* external links are classified as external and never fetched,
* generated/vendor directories are explicitly excluded from source scanning, and
* the live repository passes the guard (the CI invariant).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_docs_links as checker


def _write(root: Path, rel: str, text: str = "") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A hermetic repo root; ``PROJECT_ROOT`` is redirected at the module."""

    monkeypatch.setattr(checker, "PROJECT_ROOT", tmp_path)
    return tmp_path


# --- Slug / anchor helpers (pure functions) ----------------------------------


def test_github_slug_matches_github_rules():
    assert checker.github_slug("Hello World") == "hello-world"
    # A leading ``2.`` numbered heading keeps the digit, drops the period.
    assert checker.github_slug("2. Episode participation") == "2-episode-participation"
    # Emoji/punctuation are dropped; spaces are not collapsed first.
    assert checker.github_slug("🚀 Quick Start") == "-quick-start"
    assert checker.github_slug("A — B") == "a--b"


def test_anchors_for_collects_headings_and_html_ids_but_not_fenced():
    text = (
        "# Title\n"
        "## Section One\n"
        '<a id="manual-anchor"></a>\n'
        '<h2 id="html-heading">x</h2>\n'
        "```\n"
        "## fenced heading is not an anchor\n"
        "```\n"
    )
    anchors = checker.anchors_for(text)
    assert {"title", "section-one", "manual-anchor", "html-heading"} <= anchors
    assert "fenced-heading-is-not-an-anchor" not in anchors


def test_duplicate_headings_get_github_style_suffixes():
    assert checker.anchors_for("# Dup\n# Dup\n# Dup\n") == {"dup", "dup-1", "dup-2"}


# --- Link extraction ---------------------------------------------------------


def test_extract_links_handles_angle_brackets_parens_and_ref_defs():
    text = (
        "[a](./a.md)\n"
        "[b](<b file.md>)\n"  # angle-bracket destination with a space
        "[c](./c(1).md)\n"  # balanced parentheses inside the destination
        "[ref]: ./ref.md\n"  # reference-style definition
        "`[code](not-a-link.md)`\n"  # inline code span is masked out
        "[bug]: the tracker is broken\n"  # prose, not a real definition
    )
    targets = [link.raw_target for link in checker.extract_links(text)]
    assert "./a.md" in targets
    assert "b file.md" in targets
    assert "./c(1).md" in targets
    assert "./ref.md" in targets
    assert "not-a-link.md" not in targets
    assert "the" not in targets


# --- Missing files / anchors (actionable reporting) --------------------------


def test_missing_file_is_reported_with_all_fields(repo):
    src = _write(repo, "docs/guide.md", "See [x](./missing.md).\n")
    broken = checker.check_file(src)
    assert len(broken) == 1
    item = broken[0]
    assert item.reason == "missing file"
    assert item.source == "docs/guide.md"
    assert item.line == 1
    assert item.raw_target == "./missing.md"
    assert item.resolved == "docs/missing.md"
    rendered = item.format()
    for token in ("docs/guide.md:1", "missing file", "./missing.md", "docs/missing.md"):
        assert token in rendered


def test_missing_anchor_is_reported(repo):
    _write(repo, "docs/target.md", "# Real Heading\n")
    src = _write(
        repo,
        "docs/src.md",
        "[ok](target.md#real-heading)\n[bad](target.md#nope)\n",
    )
    broken = checker.check_file(src)
    assert len(broken) == 1
    assert broken[0].reason == "missing anchor"
    assert broken[0].raw_target == "target.md#nope"
    assert broken[0].resolved == "docs/target.md#nope"
    assert broken[0].line == 2


def test_valid_file_anchor_and_self_anchor_links_resolve(repo):
    _write(repo, "docs/target.md", "# Heading\n")
    src = _write(
        repo,
        "docs/src.md",
        "# Own Heading\n"
        "[file](target.md)\n"
        "[anchor](target.md#heading)\n"
        "[self](#own-heading)\n",
    )
    assert checker.check_file(src) == []


def test_root_absolute_paths_resolve_from_repo_root(repo):
    _write(repo, "docs/nested/deep/target.md", "# H\n")
    src = _write(repo, "docs/src.md", "[abs](/docs/nested/deep/target.md)\n")
    assert checker.check_file(src) == []


def test_source_code_targets_are_treated_as_cross_references(repo):
    # A link to a source/config file is an informational cross-reference: the
    # checker validates documentation navigation, not code layout, so a missing
    # ``.py`` target is not reported.
    src = _write(repo, "docs/src.md", "[code](../kestrel_sovereign/nope.py#L10)\n")
    assert checker.check_file(src) == []


# --- Percent-encoding and parentheses ----------------------------------------


def test_percent_encoded_and_parenthesised_paths_resolve(repo):
    _write(repo, "docs/a b.md", "# H\n")  # space in the filename
    _write(repo, "docs/c (1).md", "# H\n")  # parentheses and a space
    src = _write(
        repo,
        "docs/src.md",
        "[space](a%20b.md)\n[parens](<c (1).md>)\n",
    )
    assert checker.check_file(src) == []


def test_percent_encoded_missing_target_is_reported_decoded(repo):
    src = _write(repo, "docs/src.md", "[x](a%20b.md)\n")
    broken = checker.check_file(src)
    assert len(broken) == 1
    assert broken[0].resolved == "docs/a b.md"


# --- External links are never resolved or fetched ----------------------------


@pytest.mark.parametrize(
    "target,expected",
    [
        ("https://example.com", True),
        ("http://example.com", True),
        ("mailto:a@b.com", True),
        ("tel:+15555555555", True),
        ("//cdn.example.com/x.js", True),
        ("data:image/png;base64,AAAA", True),
        ("./local.md", False),
        ("../up.md", False),
        ("/root-abs.md", False),
        ("target.md#frag", False),
    ],
)
def test_is_external_classification(target, expected):
    assert checker._is_external(target) is expected


def test_external_links_do_not_touch_the_network(repo, monkeypatch):
    import socket

    def _boom(*args, **kwargs):
        raise AssertionError("the checker attempted network access")

    monkeypatch.setattr(socket, "socket", _boom)
    src = _write(
        repo,
        "docs/src.md",
        "[a](https://example.com/whatever)\n"
        "[b](http://x.y/z)\n"
        "[c](//cdn.example.com/x.js)\n"
        "[d](mailto:a@b.c)\n",
    )
    assert checker.check_file(src) == []


# --- Explicit generated/vendor exclusions ------------------------------------


def test_excluded_directories_are_explicit_and_documented():
    for excluded in (
        "docs/generated",
        "docs/audit",
        "docs/demos",
        "docs/research",
        "node_modules",
        ".venv",
        ".git",
    ):
        assert excluded in checker.EXCLUDED_SOURCE_DIRS


def test_generated_and_vendor_dirs_are_not_scanned(repo):
    _write(repo, "README.md", "root\n")
    _write(repo, "docs/good.md", "ok\n")
    _write(repo, "docs/generated/DERIVED.md", "[broken](./nope.md)\n")
    _write(repo, "docs/audit/SNAP.md", "[broken](./nope.md)\n")
    scanned = {p.relative_to(repo).as_posix() for p in checker.iter_source_files()}
    assert "README.md" in scanned
    assert "docs/good.md" in scanned
    assert "docs/generated/DERIVED.md" not in scanned
    assert "docs/audit/SNAP.md" not in scanned


# --- QUICKSTART regression (issue #2681) -------------------------------------

# The exact pre-move targets QUICKSTART.md linked to before #2681 corrected
# them; the privacy/anchoring docs now live under ``architecture/security/``.
PRE_MOVE_QUICKSTART_TARGETS = (
    "docs/architecture/PRIVACY_MODES.md",
    "docs/architecture/CRYPTOGRAPHIC_ANCHORING.md",
)


def test_pre_move_quickstart_paths_reproduce_as_broken():
    quickstart = checker.PROJECT_ROOT / "QUICKSTART.md"
    text = "\n".join(f"[x]({target})" for target in PRE_MOVE_QUICKSTART_TARGETS) + "\n"
    broken = checker.check_text(quickstart, text)
    assert {item.raw_target for item in broken} == set(PRE_MOVE_QUICKSTART_TARGETS)
    assert all(item.reason == "missing file" for item in broken)


def test_current_quickstart_has_no_broken_links():
    quickstart = checker.PROJECT_ROOT / "QUICKSTART.md"
    assert checker.check_file(quickstart) == []


# --- The live CI guard -------------------------------------------------------


def test_repository_has_no_broken_relative_links():
    # Mirrors ``uv run python scripts/check_docs_links.py`` — the CI invariant
    # that no covered file links to a missing local file or anchor.
    broken = checker.check_paths(checker.iter_source_files())
    assert broken == [], "Broken local links:\n" + "\n".join(
        item.format() for item in broken
    )


def test_main_returns_zero_for_the_clean_repository(capsys):
    assert checker.main([]) == 0
    assert "no broken local links" in capsys.readouterr().out


def test_main_returns_one_and_reports_a_broken_link(repo, capsys):
    src = _write(repo, "docs/src.md", "[x](./missing.md)\n")
    assert checker.main([str(src)]) == 1
    captured = capsys.readouterr()
    assert "missing file" in (captured.out + captured.err)
