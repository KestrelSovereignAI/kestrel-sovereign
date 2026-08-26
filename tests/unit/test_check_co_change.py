"""Tests for the co-change surfacing gate (``scripts/check_co_change.py``, #3124).

Covers gate 4.1 of ``docs/development/DETERMINISTIC_DEV_GATES.md``:

* a modified function body surfaces the call sites the diff did **not** touch,
* a literal *removed* at one site surfaces the siblings still carrying the old
  name — the ``action="tool_execution"`` shape, 5 sites, 1 changed,
* a diff that touches every sharing site reports clean,
* the search pattern is verified against a real ``git grep`` invocation, because
  ``git grep -E`` accepts ``\\b``/``\\s`` and then matches nothing while exiting
  0 — a silent-empty detector reports "all clear" forever,
* ``DetectorBroken`` fires when a lookup finds nothing despite a known site,
* advisory by default (exit 0), blocking only under ``--strict``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import check_co_change as checker


def _run(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repo that the checker treats as the project root."""
    root = tmp_path / "repo"
    root.mkdir()
    _run(root, "init", "-q")
    _run(root, "config", "user.email", "test@example.invalid")
    _run(root, "config", "user.name", "test")
    monkeypatch.setattr(checker, "PROJECT_ROOT", root)
    return root


def _commit(repo: Path, message: str = "baseline") -> None:
    _run(repo, "add", "-A")
    _run(repo, "commit", "-q", "-m", message)


def _findings(repo: Path) -> list[checker.Finding]:
    new_map, old_map = checker.changed_line_map(["HEAD"])
    return checker.collect(new_map, old_map, "HEAD")


def _named(findings: list[checker.Finding], name: str) -> checker.Finding:
    matches = [f for f in findings if f.name == name]
    assert matches, f"{name!r} not surfaced; got {[f.name for f in findings]}"
    return matches[0]


# --------------------------------------------------------------------------
# The detector must actually match. This is the regression guard for the bug
# that made every symbol lookup silently return nothing.
# --------------------------------------------------------------------------

def test_symbol_pattern_matches_call_sites_under_real_git_grep(repo: Path) -> None:
    """``git grep -E`` must find call sites with the generated pattern.

    The first implementation used ``\\bname\\s*\\(``. ``git grep -E`` accepts it,
    matches NOTHING, and exits 0 — so the gate reported "every site was touched"
    for every diff. Asserting through a real ``git grep`` is the only way to
    catch it; a Python ``re`` test would pass on the broken pattern.
    """
    (repo / "mod.py").write_text(
        "def helper(x):\n    return x\n\n\ndef caller():\n    return helper(1)\n"
    )
    _commit(repo)

    result = subprocess.run(
        ["git", "grep", "-n", "-E", "--", checker.symbol_pattern("helper"), "*.py"],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    assert "mod.py:6" in result.stdout, (
        f"pattern {checker.symbol_pattern('helper')!r} did not match the call site; "
        f"git grep said {result.stdout!r}"
    )


def test_detector_broken_raises_when_a_known_symbol_finds_nothing(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lookup that finds nothing must be loud, never a clean report."""
    (repo / "mod.py").write_text("def helper(x):\n    return x\n")
    _commit(repo)
    (repo / "mod.py").write_text("def helper(x):\n    return x + 1\n")

    monkeypatch.setattr(checker, "symbol_pattern", lambda name: "ZZ_NEVER_MATCHES_ZZ")
    with pytest.raises(checker.DetectorBroken, match="helper"):
        _findings(repo)


# --------------------------------------------------------------------------
# Gate 4.1 proper
# --------------------------------------------------------------------------

def test_modified_function_surfaces_untouched_call_sites(repo: Path) -> None:
    (repo / "mod.py").write_text(
        "def shared(x):\n"
        "    return x\n"
        "\n"
        "\n"
        "def door_one():\n"
        "    return shared(1)\n"
        "\n"
        "\n"
        "def door_two():\n"
        "    return shared(2)\n"
        "\n"
        "\n"
        "def door_three():\n"
        "    return shared(3)\n"
    )
    _commit(repo)

    # Change the shared function AND exactly one of its three call sites.
    text = (repo / "mod.py").read_text()
    text = text.replace("def shared(x):\n    return x", "def shared(x):\n    return x * 2")
    text = text.replace("return shared(1)", "return shared(10)")
    (repo / "mod.py").write_text(text)

    finding = _named(_findings(repo), "shared")
    assert len(finding.changed) == 1
    assert {(o.path, o.line) for o in finding.unchanged} == {("mod.py", 10), ("mod.py", 14)}


def test_removed_literal_surfaces_siblings_under_the_old_name(repo: Path) -> None:
    """Renaming one of N sites leaves the rest under a name the new side never mentions."""
    (repo / "mod.py").write_text(
        'A = "tool_execution"\n'
        'B = "tool_execution"\n'
        'C = "tool_execution"\n'
    )
    _commit(repo)
    (repo / "mod.py").write_text(
        'A = "subagent_dispatch"\n'
        'B = "tool_execution"\n'
        'C = "tool_execution"\n'
    )

    finding = _named(_findings(repo), "tool_execution")
    assert {(o.path, o.line) for o in finding.unchanged} == {("mod.py", 2), ("mod.py", 3)}


def test_clean_when_the_diff_touches_every_sharing_site(repo: Path) -> None:
    (repo / "mod.py").write_text(
        'A = "tool_execution"\n'
        'B = "tool_execution"\n'
    )
    _commit(repo)
    (repo / "mod.py").write_text(
        'A = "subagent_dispatch"\n'
        'B = "subagent_dispatch"\n'
    )

    assert _findings(repo) == []
    assert "every site" in checker.render([])


def test_short_and_noise_literals_are_not_surfaced(repo: Path) -> None:
    (repo / "mod.py").write_text('A = "ab"\nB = "ab"\nC = "  "\nD = "  "\n')
    _commit(repo)
    (repo / "mod.py").write_text('A = "cd"\nB = "ab"\nC = "\\t"\nD = "  "\n')

    assert [f.name for f in _findings(repo)] == []


def test_changed_definition_line_is_not_counted_as_a_touched_call_site(
    repo: Path,
) -> None:
    """A signature change puts the ``def`` line itself in the changed set.

    Counting it as a *call* site manufactures a finding out of nothing: the
    definition becomes the "1 of 2 modified" and the only real call site is
    reported as an unreviewed sibling. Every symbol whose signature changes
    would surface a phantom.
    """
    (repo / "mod.py").write_text(
        "def shared(x):\n    return x\n\n\ndef caller():\n    return shared(1)\n"
    )
    _commit(repo)
    # Change the SIGNATURE, so line 1 is a changed line, and leave the one
    # real call site alone.
    (repo / "mod.py").write_text(
        "def shared(x, y=None):\n    return x\n\n\ndef caller():\n    return shared(1)\n"
    )

    findings = _findings(repo)
    assert findings == [], (
        "the changed `def` line was counted as a touched call site, inventing a "
        f"finding whose only real site is unchanged: {findings}"
    )


# --------------------------------------------------------------------------
# Exit-code contract
# --------------------------------------------------------------------------

def _prepare_dirty(repo: Path) -> None:
    (repo / "mod.py").write_text('A = "tool_execution"\nB = "tool_execution"\n')
    _commit(repo)
    (repo / "mod.py").write_text('A = "subagent_dispatch"\nB = "tool_execution"\n')


def test_advisory_by_default_exits_zero(repo: Path, capsys: pytest.CaptureFixture) -> None:
    _prepare_dirty(repo)
    assert checker.main([]) == 0
    assert "did not touch" in capsys.readouterr().out


def test_strict_exits_nonzero_when_anything_is_surfaced(repo: Path) -> None:
    _prepare_dirty(repo)
    assert checker.main(["--strict"]) == 1


def test_strict_exits_zero_when_clean(repo: Path) -> None:
    (repo / "mod.py").write_text('A = "tool_execution"\n')
    _commit(repo)
    (repo / "mod.py").write_text('A = "subagent_dispatch"\n')
    assert checker.main(["--strict"]) == 0


# --------------------------------------------------------------------------
# Ordering: the signal must not sit below unrelated hits
# --------------------------------------------------------------------------

def test_unchanged_sites_in_the_edited_file_rank_above_distant_ones() -> None:
    """Same file first, then same directory, then everywhere else.

    Reproducing the #3107 round-2 mistake surfaced the four sibling sites in the
    file being edited *below* thirty unrelated hits, because ``git grep``
    returns path order. The listing is capped, so ordering decides whether the
    answerable sites are visible at all.
    """
    changed = [checker.Occurrence("pkg/security/queue.py", 310)]
    unchanged = [
        checker.Occurrence("aaa/elsewhere.py", 1),
        checker.Occurrence("pkg/security/hooks.py", 12),
        checker.Occurrence("pkg/security/queue.py", 415),
        checker.Occurrence("pkg/security/queue.py", 327),
    ]

    ordered = checker.rank_unchanged(unchanged, changed, {"pkg/security/queue.py"})

    assert [(o.path, o.line) for o in ordered] == [
        ("pkg/security/queue.py", 327),   # same file, by line
        ("pkg/security/queue.py", 415),
        ("pkg/security/hooks.py", 12),    # same directory
        ("aaa/elsewhere.py", 1),          # everywhere else, despite sorting first
    ]


def test_ranking_uses_the_diffs_files_when_no_site_was_touched(repo: Path) -> None:
    """A removed literal has zero touched sites; the diff's files still frame it."""
    (repo / "near.py").write_text('A = "tool_execution"\nB = "tool_execution"\n')
    (repo / "far.py").write_text('C = "tool_execution"\n')
    _commit(repo)
    (repo / "near.py").write_text('A = "subagent_dispatch"\nB = "tool_execution"\n')

    finding = _named(_findings(repo), "tool_execution")
    ordered = checker.rank_unchanged(finding.unchanged, finding.changed, {"near.py"})
    assert (ordered[0].path, ordered[0].line) == ("near.py", 2)


# --------------------------------------------------------------------------
# Noise control — a gate people suppress is worse than no gate
# --------------------------------------------------------------------------

def test_module_private_names_do_not_match_unrelated_modules(repo: Path) -> None:
    """Every test module defines its own ``_commit``; they are different functions.

    Dogfooding this gate on its own diff reported ``_commit`` as having 33 call
    sites across the repo. Scoping a leading-underscore name to its own module
    is what makes the output readable.
    """
    (repo / "mine.py").write_text(
        "def _helper(x):\n    return x\n\n\ndef a():\n    return _helper(1)\n"
        "\n\ndef b():\n    return _helper(2)\n"
    )
    (repo / "stranger.py").write_text(
        "def _helper(x):\n    return x * 99\n\n\ndef c():\n    return _helper(3)\n"
    )
    _commit(repo)
    text = (repo / "mine.py").read_text().replace(
        "def _helper(x):\n    return x", "def _helper(x):\n    return x + 1"
    ).replace("return _helper(1)", "return _helper(11)")
    (repo / "mine.py").write_text(text)

    finding = _named(_findings(repo), "_helper")
    paths = {o.path for o in finding.unchanged}
    assert paths == {"mine.py"}, f"unrelated module's _helper surfaced: {paths}"


def test_module_private_name_still_reaches_explicit_importers(repo: Path) -> None:
    """Scoping must not hide a private helper someone actually imported."""
    (repo / "mine.py").write_text(
        "def _helper(x):\n    return x\n\n\ndef a():\n    return _helper(1)\n"
    )
    (repo / "user.py").write_text(
        "from mine import _helper\n\n\ndef b():\n    return _helper(2)\n"
    )
    _commit(repo)
    text = (repo / "mine.py").read_text().replace(
        "def _helper(x):\n    return x", "def _helper(x):\n    return x + 1"
    ).replace("return _helper(1)", "return _helper(11)")
    (repo / "mine.py").write_text(text)

    finding = _named(_findings(repo), "_helper")
    assert ("user.py", 5) in {(o.path, o.line) for o in finding.unchanged}


@pytest.mark.parametrize("value", ["has a space", "*.py", "%s/%s", "a b"])
def test_literals_without_an_identifier_shape_are_ignored(
    repo: Path, value: str
) -> None:
    """Prose and pathspecs are not invariants."""
    (repo / "mod.py").write_text(f'A = "{value}"\nB = "{value}"\n')
    _commit(repo)
    (repo / "mod.py").write_text(f'A = "replaced_token"\nB = "{value}"\n')

    assert value not in [f.name for f in _findings(repo)]


def test_identifier_shaped_literals_are_still_surfaced(repo: Path) -> None:
    """The rule must not swallow the case the gate exists for."""
    (repo / "mod.py").write_text('A = "tool_execution"\nB = "tool_execution"\n')
    _commit(repo)
    (repo / "mod.py").write_text('A = "subagent_dispatch"\nB = "tool_execution"\n')

    assert "tool_execution" in [f.name for f in _findings(repo)]
