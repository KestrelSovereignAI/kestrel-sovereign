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
    findings, _, __ = checker.collect(new_map, old_map, "HEAD")
    return findings


def _structural(repo: Path) -> list[tuple[str, int]]:
    new_map, old_map = checker.changed_line_map(["HEAD"])
    _, structural, _unparseable = checker.collect(new_map, old_map, "HEAD")
    return structural


def _unparseable(repo: Path) -> list[str]:
    new_map, old_map = checker.changed_line_map(["HEAD"])
    _, __, unparseable = checker.collect(new_map, old_map, "HEAD")
    return unparseable


def _named(findings: list[checker.Finding], name: str) -> checker.Finding:
    matches = [f for f in findings if f.name == name]
    assert matches, f"{name!r} not surfaced; got {[f.name for f in findings]}"
    return matches[0]


# --------------------------------------------------------------------------
# The detector must actually match. This is the regression guard for the bug
# that made every symbol lookup silently return nothing.
# --------------------------------------------------------------------------

def test_candidate_prefilter_finds_the_file_under_real_git_grep(repo: Path) -> None:
    """The ``git grep`` prefilter must actually match, through a real invocation.

    An earlier implementation searched with ``\\bname\\s*\\(``. ``git grep -E``
    accepts those escapes, matches NOTHING, and exits 0 — so the gate reported
    "every site was touched" for every diff. Asserting through a real ``git
    grep`` is the only way to catch it; a Python ``re`` test passes on the
    broken pattern. grep no longer decides what counts, but if it returns no
    candidate files the AST pass never runs and the result is the same silence.
    """
    (repo / "mod.py").write_text(
        "def helper(x):\n    return x\n\n\ndef caller():\n    return helper(1)\n"
    )
    _commit(repo)

    pattern = r"(^|[^[:alnum:]_])helper([^[:alnum:]_]|$)"
    result = subprocess.run(
        ["git", "grep", "-l", "-E", "--", pattern, "*.py"],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    assert "mod.py" in result.stdout, (
        f"prefilter {pattern!r} matched nothing; git grep said {result.stdout!r}"
    )


def test_detector_broken_raises_when_a_known_symbol_finds_nothing(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lookup that finds nothing must be loud, never a clean report."""
    (repo / "mod.py").write_text("def helper(x):\n    return x\n")
    _commit(repo)
    (repo / "mod.py").write_text("def helper(x):\n    return x + 1\n")

    monkeypatch.setattr(checker, "defines", lambda tree, name: False)
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

    Counting it as a *call* site would report the definition as the site that
    was updated, hiding that no actual caller was visited.

    This test previously asserted ``findings == []`` for this shape and passed.
    That assertion was wrong: it enshrined the false-clean that codex round 2
    found as a P1 — a changed definition with untouched callers is precisely
    what this gate exists to report, not an empty result.
    """
    (repo / "mod.py").write_text(
        "def shared(x):\n    return x\n\n\ndef caller():\n    return shared(1)\n"
    )
    _commit(repo)
    (repo / "mod.py").write_text(
        "def shared(x, y=None):\n    return x\n\n\ndef caller():\n    return shared(1)\n"
    )

    finding = _named(_findings(repo), "shared")
    assert finding.changed == [], (
        f"the changed `def` line was counted as a touched call site: {finding.changed}"
    )
    assert {(o.path, o.line) for o in finding.unchanged} == {("mod.py", 6)}


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


# --------------------------------------------------------------------------
# Codex review round 1 — five P2s, one per test
# --------------------------------------------------------------------------

def test_editing_a_class_body_does_not_abort_the_gate(repo: Path) -> None:
    """An ordinary ``class Worker:`` edit took the whole check down.

    ``modified_symbols`` reported the ClassDef, the search demanded ``Worker(``,
    an uninstantiated class matched nothing, and DetectorBroken aborted every
    remaining finding. The positive control now asserts the *definition* is
    findable, which does not depend on the symbol being used at all.
    """
    (repo / "mod.py").write_text("class Worker:\n    def run(self):\n        return 1\n")
    _commit(repo)
    (repo / "mod.py").write_text("class Worker:\n    def run(self):\n        return 2\n")

    assert _findings(repo) == []  # must not raise


def test_class_references_count_as_sites_not_only_instantiations(repo: Path) -> None:
    (repo / "mod.py").write_text(
        "class Worker:\n    def run(self):\n        return 1\n\n\nw = Worker()\n"
    )
    (repo / "other.py").write_text("from mod import Worker\n\n\ndef f(x: Worker):\n    return x\n")
    _commit(repo)
    (repo / "mod.py").write_text(
        "class Worker:\n    def run(self):\n        return 2\n\n\nw = Worker\n"
    )

    finding = _named(_findings(repo), "Worker")
    assert ("other.py", 4) in {(o.path, o.line) for o in finding.unchanged}


def test_private_method_keeps_external_callers(repo: Path) -> None:
    """``queue._dispatch()`` is reached by importing Queue, never ``_dispatch``.

    Scoping every underscore name to its module plus importers-of-that-name
    dropped exactly the external callers this gate exists to surface.
    """
    (repo / "mod.py").write_text(
        "class Queue:\n"
        "    def _dispatch(self, x):\n"
        "        return x\n"
        "\n"
        "    def go(self):\n"
        "        return self._dispatch(1)\n"
    )
    (repo / "user.py").write_text(
        "from mod import Queue\n\n\ndef run(q: Queue):\n    return q._dispatch(2)\n"
    )
    _commit(repo)
    text = (repo / "mod.py").read_text()
    text = text.replace("    def _dispatch(self, x):\n        return x",
                        "    def _dispatch(self, x):\n        return x + 1")
    text = text.replace("return self._dispatch(1)", "return self._dispatch(11)")
    (repo / "mod.py").write_text(text)

    finding = _named(_findings(repo), "_dispatch")
    assert ("user.py", 5) in {(o.path, o.line) for o in finding.unchanged}, (
        "external caller of a private method was scoped away"
    )


def test_literal_matching_is_exact_not_substring(repo: Path) -> None:
    """``git grep -F "config"`` matches ``base_config`` — 8,241 hits repo-wide."""
    (repo / "mod.py").write_text(
        'A = "config_alpha"\n'
        'B = "config_alpha"\n'
        'base_config_alpha = 1\n'
        'C = "prefix_config_alpha_suffix"\n'
    )
    _commit(repo)
    (repo / "mod.py").write_text(
        'A = "config_beta"\n'
        'B = "config_alpha"\n'
        'base_config_alpha = 1\n'
        'C = "prefix_config_alpha_suffix"\n'
    )

    finding = _named(_findings(repo), "config_alpha")
    assert {(o.path, o.line) for o in finding.unchanged} == {("mod.py", 2)}, (
        "an identifier or a longer string was counted as an occurrence"
    )


def test_a_call_inside_a_string_literal_is_not_a_call_site(repo: Path) -> None:
    """This repository's tests embed Python source as strings constantly."""
    (repo / "mod.py").write_text(
        "def helper(x):\n    return x\n\n\ndef caller():\n    return helper(1)\n"
    )
    (repo / "snippet.py").write_text('SOURCE = "helper(1)"\n# helper() in a comment\n')
    _commit(repo)
    text = (repo / "mod.py").read_text().replace(
        "def helper(x):\n    return x", "def helper(x):\n    return x + 1"
    ).replace("return helper(1)", "return helper(11)")
    (repo / "mod.py").write_text(text)

    findings = _findings(repo)
    if findings:
        paths = {o.path for o in _named(findings, "helper").unchanged}
        assert "snippet.py" not in paths, "a string/comment was reported as a call site"


def test_removed_literal_counts_the_site_it_was_removed_from(repo: Path) -> None:
    """The contract promises ``modified 1 of 3``; grep on the new tree saw 0 of 2."""
    (repo / "mod.py").write_text(
        'A = "tool_execution"\nB = "tool_execution"\nC = "tool_execution"\n'
    )
    _commit(repo)
    (repo / "mod.py").write_text(
        'A = "subagent_dispatch"\nB = "tool_execution"\nC = "tool_execution"\n'
    )

    finding = _named(_findings(repo), "tool_execution")
    assert len(finding.changed) == 1, f"removed site not counted: {finding.changed}"
    assert len(finding.changed) + len(finding.unchanged) == 3
    assert "modified 1 of 3 occurrences" in checker.render([finding])


def test_structural_names_are_named_in_a_footer_not_silently_dropped(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hidden cap reads as 'covered everything'."""
    monkeypatch.setattr(checker, "MAX_TOTAL_SITES", 2)
    body = "def shared(x):\n    return x\n"
    calls = "".join(f"\n\ndef c{i}():\n    return shared({i})\n" for i in range(4))
    (repo / "mod.py").write_text(body + calls)
    _commit(repo)
    (repo / "mod.py").write_text(
        body.replace("return x", "return x + 1") + calls.replace("shared(0)", "shared(9)")
    )

    structural = _structural(repo)
    assert ("shared", 4) in structural
    footer = checker.render([], structural=structural)
    assert "shared (4)" in footer and "not reviewed site-by-site" in footer


# --------------------------------------------------------------------------
# Codex review round 2 — one P1 and six P2s
# --------------------------------------------------------------------------

def test_body_only_change_surfaces_every_caller(repo: Path) -> None:
    """THE motivating case, and it reported clean.

    Change a shared function's implementation, leave every call expression
    alone, and ``touched and untouched`` discarded the symbol entirely — the
    gate answered "every site sharing a changed symbol was touched" to the one
    question it exists to ask. This is the round-7 ``fold_searchable`` shape.
    """
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "a.py").write_text("from mod import shared\n\n\ndef f():\n    return shared(1)\n")
    (repo / "b.py").write_text("from mod import shared\n\n\ndef g():\n    return shared(2)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x + 1\n")

    finding = _named(_findings(repo), "shared")
    assert finding.changed == []
    assert {o.path for o in finding.unchanged} == {"a.py", "b.py"}
    assert "definition changed; 2 call sites untouched" in checker.render([finding])


def test_deletion_only_body_edit_still_surfaces_callers(repo: Path) -> None:
    """A hunk of pure deletions contributes no new-side line at all."""
    (repo / "mod.py").write_text("def shared(x):\n    y = 1\n    return x\n")
    (repo / "a.py").write_text("from mod import shared\n\n\ndef f():\n    return shared(1)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")

    finding = _named(_findings(repo), "shared")
    assert {(o.path, o.line) for o in finding.unchanged} == {("a.py", 5)}


def test_import_alias_resolves_to_the_changed_symbol(repo: Path) -> None:
    """``from mod import shared as alias`` then ``alias()`` is a call to shared."""
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "b.py").write_text(
        "from mod import shared as alias\n\n\ndef g():\n    return alias(2)\n"
    )
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x + 1\n")

    finding = _named(_findings(repo), "shared")
    assert {(o.path, o.line) for o in finding.unchanged} == {("b.py", 5)}


def test_a_multiline_call_is_touched_by_a_change_to_its_arguments(repo: Path) -> None:
    """Only the callee-name line was compared, so an argument edit read untouched."""
    (repo / "mod.py").write_text("def shared(x, y):\n    return x\n")
    (repo / "a.py").write_text(
        "from mod import shared\n\n\ndef f():\n    return shared(\n        1,\n        2,\n    )\n"
    )
    (repo / "b.py").write_text("from mod import shared\n\n\ndef g():\n    return shared(3, 4)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x, y):\n    return x + y\n")
    (repo / "a.py").write_text(
        "from mod import shared\n\n\ndef f():\n    return shared(\n        1,\n        99,\n    )\n"
    )

    finding = _named(_findings(repo), "shared")
    assert [o.path for o in finding.changed] == ["a.py"], (
        f"multiline call not marked touched by its argument change: {finding.changed}"
    )
    assert [o.path for o in finding.unchanged] == ["b.py"]


def test_a_constructor_call_counts_once_not_twice(repo: Path) -> None:
    """``ast.walk`` visits the Call and then its Name child; both matched."""
    (repo / "mod.py").write_text("class Worker:\n    def run(self):\n        return 1\n")
    (repo / "a.py").write_text("from mod import Worker\n\n\nw = Worker()\n")
    _commit(repo)
    (repo / "mod.py").write_text("class Worker:\n    def run(self):\n        return 2\n")

    finding = _named(_findings(repo), "Worker")
    assert len(finding.unchanged) == 1, (
        f"constructor counted more than once: {[(o.path, o.line) for o in finding.unchanged]}"
    )


def test_structural_results_do_not_report_clean(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symbol skipped for being too widely used is an unanswered question."""
    monkeypatch.setattr(checker, "MAX_TOTAL_SITES", 1)
    (repo / "mod.py").write_text(
        "def shared(x):\n    return x\n\n\ndef a():\n    return shared(1)\n"
        "\n\ndef b():\n    return shared(2)\n"
    )
    _commit(repo)
    (repo / "mod.py").write_text(
        "def shared(x):\n    return x + 1\n\n\ndef a():\n    return shared(1)\n"
        "\n\ndef b():\n    return shared(2)\n"
    )

    structural = _structural(repo)
    assert structural, "symbol was not classified structural"
    report = checker.render([], structural=structural)
    assert "every site" not in report, f"structural result claimed clean: {report}"
    assert checker.main(["--strict"]) == 1


def test_an_unparseable_changed_file_is_surfaced_not_swallowed(repo: Path) -> None:
    """A half-written file yielded no symbols and rendered as a clean bill of health."""
    (repo / "mod.py").write_text('A = "tool_execution"\nB = "tool_execution"\n')
    _commit(repo)
    (repo / "mod.py").write_text(
        'A = "subagent_dispatch"\nB = "tool_execution"\ndef broken(\n'
    )

    assert _unparseable(repo) == ["mod.py"]
    report = checker.render([], unparseable=_unparseable(repo))
    assert "every site" not in report
    assert "NOT ANALYSED" in report
    assert checker.main(["--strict"]) == 1


# --------------------------------------------------------------------------
# Codex review round 3 — one P1 and three P2s, all false-clean paths
# --------------------------------------------------------------------------

def test_renaming_a_definition_surfaces_callers_of_the_old_name(repo: Path) -> None:
    """The stranded-caller case, and it reported clean.

    Round 2 added old-side symbol discovery but guarded it on the *working
    tree* still defining the name. A renamed or deleted definition fails that
    guard by construction — so the one shape that always strands callers was
    the one always dropped. The defect was inside the previous round's fix.
    """
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "a.py").write_text("from mod import shared\n\n\ndef f():\n    return shared(1)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def renamed(x):\n    return x\n")

    finding = _named(_findings(repo), "shared")
    # The import binding on line 1 is a site too: after the rename, importing
    # a.py raises ImportError. That is a dependency on the NAME existing, which
    # a body change would not touch and a rename breaks outright.
    assert {(o.path, o.line) for o in finding.unchanged} == {("a.py", 1), ("a.py", 5)}


def test_deleting_a_definition_surfaces_callers_of_the_old_name(repo: Path) -> None:
    (repo / "mod.py").write_text("def shared(x):\n    return x\n\n\ndef kept():\n    return 1\n")
    (repo / "a.py").write_text("from mod import shared\n\n\ndef f():\n    return shared(1)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def kept():\n    return 1\n")

    finding = _named(_findings(repo), "shared")
    assert {(o.path, o.line) for o in finding.unchanged} == {("a.py", 1), ("a.py", 5)}


def test_a_first_class_reference_is_a_dependent_site(repo: Path) -> None:
    """``map(shared, items)`` is never the callee of an ``ast.Call``."""
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "a.py").write_text(
        "from mod import shared\n\n\nitems = [1]\nresult = list(map(shared, items))\n"
    )
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x + 1\n")

    finding = _named(_findings(repo), "shared")
    assert {(o.path, o.line) for o in finding.unchanged} == {("a.py", 5)}


def test_a_property_access_is_a_dependent_site(repo: Path) -> None:
    """``obj.status`` for an ``@property`` is a use with no call node."""
    (repo / "mod.py").write_text(
        "class Thing:\n"
        "    @property\n"
        "    def status(self):\n"
        "        return 1\n"
    )
    (repo / "a.py").write_text("from mod import Thing\n\n\ndef f(t: Thing):\n    return t.status\n")
    _commit(repo)
    (repo / "mod.py").write_text(
        "class Thing:\n"
        "    @property\n"
        "    def status(self):\n"
        "        return 2\n"
    )

    finding = _named(_findings(repo), "status")
    assert ("a.py", 5) in {(o.path, o.line) for o in finding.unchanged}


def test_a_private_function_reached_through_its_module_is_kept(repo: Path) -> None:
    """``import cli`` then ``cli._get_project_dir()`` imports the MODULE, not the name."""
    (repo / "cli.py").write_text("def _get_project_dir():\n    return 1\n")
    (repo / "cli_features.py").write_text(
        "import cli\n\n\ndef f():\n    return cli._get_project_dir()\n"
    )
    _commit(repo)
    (repo / "cli.py").write_text("def _get_project_dir():\n    return 2\n")

    finding = _named(_findings(repo), "_get_project_dir")
    assert {(o.path, o.line) for o in finding.unchanged} == {("cli_features.py", 5)}


def test_a_decorator_only_change_still_changes_the_definition(repo: Path) -> None:
    """``FunctionDef.lineno`` is the ``def`` line; decorators sit above it."""
    (repo / "mod.py").write_text(
        "import functools\n\n\n@functools.cache\ndef shared(x):\n    return x\n"
    )
    (repo / "a.py").write_text("from mod import shared\n\n\ndef f():\n    return shared(1)\n")
    _commit(repo)
    (repo / "mod.py").write_text(
        "import functools\n\n\n@functools.lru_cache\ndef shared(x):\n    return x\n"
    )

    finding = _named(_findings(repo), "shared")
    assert {(o.path, o.line) for o in finding.unchanged} == {("a.py", 5)}


# --------------------------------------------------------------------------
# Codex review round 4 — one P1 and three P2s
# --------------------------------------------------------------------------

def test_a_re_exported_alias_reaches_the_downstream_caller(repo: Path) -> None:
    """The caller's file never contains the original name at all.

    ``bridge.py`` does ``from mod import shared as alias``; ``user.py`` does
    ``from bridge import alias`` and calls ``alias()``. A prefilter on ``shared``
    excludes user.py entirely, so a body change reported clean. One hop is not
    enough — the binding that reaches the caller is the bridge's.
    """
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "bridge.py").write_text("from mod import shared as alias\n")
    (repo / "user.py").write_text("from bridge import alias\n\n\ndef f():\n    return alias(1)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x + 1\n")

    finding = _named(_findings(repo), "shared")
    assert ("user.py", 5) in {(o.path, o.line) for o in finding.unchanged}


def test_a_dotted_module_import_is_still_an_import(repo: Path) -> None:
    """``import pkg.cli as c`` — the prefilter excluded a preceding dot."""
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "cli.py").write_text("def _helper():\n    return 1\n")
    (repo / "user.py").write_text("import pkg.cli as c\n\n\ndef f():\n    return c._helper()\n")
    _commit(repo)
    (repo / "pkg" / "cli.py").write_text("def _helper():\n    return 2\n")

    finding = _named(_findings(repo), "_helper")
    assert {(o.path, o.line) for o in finding.unchanged} == {("user.py", 5)}


def test_a_package_initializer_resolves_to_the_package_name(repo: Path) -> None:
    """``pkg/__init__.py`` has stem ``__init__``; nothing imports ``__init__``."""
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("def _helper():\n    return 1\n")
    (repo / "user.py").write_text("import pkg\n\n\ndef f():\n    return pkg._helper()\n")
    _commit(repo)
    (repo / "pkg" / "__init__.py").write_text("def _helper():\n    return 2\n")

    finding = _named(_findings(repo), "_helper")
    assert {(o.path, o.line) for o in finding.unchanged} == {("user.py", 5)}


def test_a_constant_equal_but_spelled_differently_is_found(repo: Path) -> None:
    """``"tool\\x5fexecution"`` and ``"tool_" "execution"`` equal the value at runtime."""
    (repo / "a.py").write_text('A = "tool_execution"\n')
    (repo / "b.py").write_text('B = "tool\\x5fexecution"\n')
    (repo / "c.py").write_text('C = ("tool_" "execution")\n')
    _commit(repo)
    (repo / "a.py").write_text('A = "subagent_dispatch"\n')

    finding = _named(_findings(repo), "tool_execution")
    assert {o.path for o in finding.unchanged} == {"b.py", "c.py"}


def test_widening_is_skipped_for_an_undistinctive_run(repo: Path) -> None:
    """A short run matches most of the tree; the cost is paid on every run.

    ``"__init__"`` widens to ``init``. Indexing that candidate set took the gate
    from under a second to sixteen. Only a distinctive run earns the extra files.
    """
    assert checker.MIN_WIDENED_RUN >= 8
    (repo / "a.py").write_text('A = "__init__"\nB = "__init__"\n')
    _commit(repo)
    (repo / "a.py").write_text('A = "renamed_marker"\nB = "__init__"\n')

    # Still correct via the exact prefilter — widening is an addition, not the
    # only path to a sibling.
    finding = _named(_findings(repo), "__init__")
    assert {(o.path, o.line) for o in finding.unchanged} == {("a.py", 2)}


# --------------------------------------------------------------------------
# Gaps found by surviving mutants — each guard's ONLY load-bearing case
# --------------------------------------------------------------------------

def test_a_symbol_defined_in_a_new_file_passes_the_positive_control(repo: Path) -> None:
    """The positive control has two halves; only this case needs the first.

    A mutant that broke the working-tree half survived, because every existing
    test defines its symbol in a file the diff also changed — so the old-blob
    half answered instead. A file added by the diff has no old-side entry at
    all, which is the one shape that depends on the working-tree lookup.
    """
    (repo / "a.py").write_text("from mod import shared\n\n\ndef f():\n    return shared(1)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")

    finding = _named(_findings(repo), "shared")  # must not raise DetectorBroken
    assert {(o.path, o.line) for o in finding.unchanged} == {("a.py", 5)}


def test_a_private_helper_re_exported_through_a_bridge_keeps_its_caller(
    repo: Path,
) -> None:
    """Scoping needs the NAME's importers, not only the module's.

    A mutant dropping ``import_sites`` survived: for a direct
    ``from mine import _helper`` the caller also imports the module, so
    ``module_importers`` answered. Through a bridge it does not — ``user.py``
    imports ``bridge``, never ``mine`` — and only the name lookup keeps it.
    """
    (repo / "mine.py").write_text("def _helper(x):\n    return x\n")
    (repo / "bridge.py").write_text("from mine import _helper\n")
    (repo / "user.py").write_text("from bridge import _helper\n\n\ndef f():\n    return _helper(2)\n")
    _commit(repo)
    (repo / "mine.py").write_text("def _helper(x):\n    return x + 1\n")

    finding = _named(_findings(repo), "_helper")
    assert ("user.py", 5) in {(o.path, o.line) for o in finding.unchanged}, (
        "a private helper's caller was scoped away because it imports the bridge, "
        "not the defining module"
    )


def test_an_untracked_new_file_is_analysed(repo: Path) -> None:
    """``git diff`` never mentions an untracked file.

    Found by a surviving mutant: the test meant to cover a new file's symbol
    could not, because a file created and not staged is absent from
    ``git diff HEAD`` entirely — so a brand-new module was invisible and the
    gate reported clean on it.
    """
    (repo / "a.py").write_text("from mod import shared\n\n\ndef f():\n    return shared(1)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")  # never staged

    new_map, _ = checker.changed_line_map(["HEAD"])
    assert "mod.py" in new_map, "an untracked Python file was not analysed at all"

    finding = _named(_findings(repo), "shared")
    assert {(o.path, o.line) for o in finding.unchanged} == {("a.py", 5)}


# --------------------------------------------------------------------------
# Codex review round 5 — three P1s and two P2s, all false-cleans
# --------------------------------------------------------------------------

def test_a_black_formatted_alias_import_is_discovered(repo: Path) -> None:
    """``from mod import (\\n    shared as alias,\\n)`` is how this repo formats.

    The aliasing prefilter required ``import`` and ``shared as`` on ONE line, so
    the parenthesized form was never seen and the closure never learned ``alias``.
    """
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "bridge.py").write_text("from mod import (\n    shared as alias,\n)\n")
    (repo / "user.py").write_text("from bridge import alias\n\n\ndef f():\n    return alias(1)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x + 1\n")

    finding = _named(_findings(repo), "shared")
    assert ("user.py", 5) in {(o.path, o.line) for o in finding.unchanged}


def test_an_untracked_files_occurrence_is_searched_not_only_mapped(repo: Path) -> None:
    """Round 4 taught changed_line_map about untracked files and stopped there.

    ``git grep`` skips untracked files by default, so the new file entered the
    changed map but was never searched: its occurrence could not become
    ``touched``, and the gate reported clean. A boundary has two ends.
    """
    (repo / "a.py").write_text('A = "tool_execution"\n')
    _commit(repo)
    (repo / "newfile.py").write_text('B = "tool_execution"\n')  # never staged

    finding = _named(_findings(repo), "tool_execution")
    assert [(o.path, o.line) for o in finding.changed] == [("newfile.py", 1)]
    assert [(o.path, o.line) for o in finding.unchanged] == [("a.py", 1)]


def test_an_import_only_dependency_on_a_renamed_definition_is_surfaced(
    repo: Path,
) -> None:
    """No call anywhere — but importing the file now raises ImportError."""
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "user.py").write_text("from mod import shared\n")
    _commit(repo)
    (repo / "mod.py").write_text("def renamed(x):\n    return x\n")

    finding = _named(_findings(repo), "shared")
    assert {(o.path, o.line) for o in finding.unchanged} == {("user.py", 1)}


def test_an_import_binding_is_not_a_site_for_a_body_change(repo: Path) -> None:
    """The other direction: a body change does not touch an import.

    Counting import bindings unconditionally made every import an unreviewed
    site and inflated eleven existing tests. The binding depends on the name
    existing, not on what it does.
    """
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "user.py").write_text("from mod import shared\n\n\ndef f():\n    return shared(1)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x + 1\n")

    finding = _named(_findings(repo), "shared")
    assert {(o.path, o.line) for o in finding.unchanged} == {("user.py", 5)}, (
        "the import binding was counted as a site for a body-only change"
    )


def test_scope_follows_importers_of_a_discovered_alias(repo: Path) -> None:
    """The closure can find a name the scope has never heard of."""
    (repo / "mine.py").write_text("def _helper(x):\n    return x\n")
    (repo / "bridge.py").write_text("from mine import _helper as h\n")
    (repo / "user.py").write_text("from bridge import h\n\n\ndef f():\n    return h(2)\n")
    _commit(repo)
    (repo / "mine.py").write_text("def _helper(x):\n    return x + 1\n")

    finding = _named(_findings(repo), "_helper")
    assert ("user.py", 5) in {(o.path, o.line) for o in finding.unchanged}


def test_a_call_removed_by_the_diff_counts_as_touched(repo: Path) -> None:
    """A deleted call is absent from the working tree, so the scan cannot see it."""
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "a.py").write_text("from mod import shared\n\n\ndef f():\n    return shared(1)\n")
    (repo / "b.py").write_text("from mod import shared\n\n\ndef g():\n    return shared(2)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x + 1\n")
    (repo / "a.py").write_text("from mod import shared\n\n\ndef f():\n    return 0\n")

    finding = _named(_findings(repo), "shared")
    assert [(o.path, o.line) for o in finding.changed] == [("a.py", 5)], (
        f"the removed call was not counted: {finding.changed}"
    )
    assert [(o.path, o.line) for o in finding.unchanged] == [("b.py", 5)]


# --------------------------------------------------------------------------
# Codex review round 6 — two P1s and three P2s
# --------------------------------------------------------------------------

def test_a_nul_bearing_literal_does_not_kill_the_gate(repo: Path) -> None:
    """This repository has 26 NUL-bearing string constants.

    One is ``_SCHEDULER_BOOTSTRAP_LOCK_SCOPE = "\\0scheduler-bootstrap"`` in
    ``features/scheduler/runner.py``. The raw value reached ``git grep`` as argv
    and raised ``ValueError: embedded null byte``, taking down the whole
    advisory run rather than producing output.
    """
    (repo / "a.py").write_text(
        'SCOPE = "\\0scheduler-bootstrap"\nOTHER = "\\0scheduler-bootstrap"\n'
    )
    _commit(repo)
    (repo / "a.py").write_text(
        'SCOPE = "\\0renamed-bootstrap"\nOTHER = "\\0scheduler-bootstrap"\n'
    )

    finding = _named(_findings(repo), "\x00scheduler-bootstrap")  # must not raise
    assert len(finding.changed) == 1
    assert [(o.path, o.line) for o in finding.unchanged] == [("a.py", 2)]


def test_import_consumer_surfaced_when_a_same_named_method_survives(
    repo: Path,
) -> None:
    """``from mod import shared`` binds a MODULE-LEVEL name.

    Checking the bare name against every definition in the file left it true
    when the module-level function was deleted and a same-named method
    survived — so the import consumer, which now raises ImportError, was
    silently omitted.
    """
    (repo / "mod.py").write_text(
        "def shared(x):\n    return x\n\n\nclass C:\n    def shared(self):\n        return 1\n"
    )
    (repo / "user.py").write_text("from mod import shared\n")
    _commit(repo)
    (repo / "mod.py").write_text("class C:\n    def shared(self):\n        return 1\n")

    finding = _named(_findings(repo), "shared")
    assert {(o.path, o.line) for o in finding.unchanged} == {("user.py", 1)}


def test_a_parenthesized_import_alias_records_its_own_line(repo: Path) -> None:
    """``ImportFrom.lineno`` is the opening line, not the alias's."""
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "user.py").write_text("from mod import (\n    shared,\n)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def renamed(x):\n    return x\n")

    finding = _named(_findings(repo), "shared")
    assert {(o.path, o.line) for o in finding.unchanged} == {("user.py", 2)}, (
        "the alias was recorded at the import's opening line"
    )


def test_a_shifted_call_is_not_counted_twice(repo: Path) -> None:
    """Editing a call that also moves it recorded new AND old coordinates."""
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "a.py").write_text("from mod import shared\n\n\ndef f():\n    return shared(1)\n")
    (repo / "b.py").write_text("from mod import shared\n\n\ndef g():\n    return shared(2)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x + 1\n")
    (repo / "a.py").write_text(
        "from mod import shared\n\n\ndef f():\n    y = 0\n    return shared(1, y)\n"
    )

    finding = _named(_findings(repo), "shared")
    total = len(finding.changed) + len(finding.unchanged)
    assert total == 2, (
        f"one logical caller counted twice: changed={finding.changed} "
        f"unchanged={finding.unchanged}"
    )
    assert "modified 1 of 2" in checker.render([finding])


def test_base_mode_diffs_against_the_working_tree_not_head(
    repo: Path, capsys: pytest.CaptureFixture
) -> None:
    """``--base`` mapped ``base...HEAD`` while every scan read the working tree.

    An unstaged edit that removed the only untouched caller made it vanish from
    the scan while it still existed in HEAD, so the branch check printed clean.
    """
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "a.py").write_text("from mod import shared\n\n\ndef f():\n    return shared(1)\n")
    _commit(repo, "base")
    _run(repo, "branch", "base-ref")
    (repo / "mod.py").write_text("def shared(x):\n    return x + 1\n")
    _commit(repo, "change on branch")
    # Unstaged: a NEW caller appears only in the working tree.
    (repo / "c.py").write_text("from mod import shared\n\n\ndef h():\n    return shared(3)\n")
    _run(repo, "add", "c.py")

    # Through main(), because the defect is in which spec main CHOOSES —
    # calling changed_line_map directly would pass with main still broken.
    assert checker.main(["--base", "base-ref"]) == 0
    output = capsys.readouterr().out
    # c.py's call is a TOUCHED site, so it shows in the count rather than the
    # unchanged listing. Mapping `base...HEAD` misses it entirely and the line
    # reads "definition changed; 2 call sites untouched" instead.
    assert "modified 1 of 2 call sites" in output, (
        "base mode ignored a working-tree change while scanning the working "
        f"tree; got:\n{output}"
    )


# --------------------------------------------------------------------------
# Codex review round 7 — two of these are defects introduced by round 6
# --------------------------------------------------------------------------

def test_a_multiline_literal_is_touched_by_a_change_to_its_continuation(
    repo: Path,
) -> None:
    """``literals_on`` selected on end_lineno; the index stored only the opening line."""
    (repo / "a.py").write_text('A = ("tool_"\n     "execution")\n')
    (repo / "b.py").write_text('B = "tool_execution"\n')
    _commit(repo)
    (repo / "a.py").write_text('A = ("tool_"\n     "dispatch")\n')

    finding = _named(_findings(repo), "tool_execution")
    assert len(finding.changed) == 1, (
        f"the continuation-line change did not mark the literal touched: {finding.changed}"
    )
    assert [(o.path, o.line) for o in finding.unchanged] == [("b.py", 1)]


def test_a_surviving_same_name_elsewhere_does_not_suppress_the_import_check(
    repo: Path,
) -> None:
    """Symbols merge by bare name, so ANY was letting m2 answer for m1.

    Deleting ``shared`` from m1.py while m2.py still defines its own ``shared``
    left ``module_level_now`` true, and ``from m1 import shared`` — which now
    raises ImportError — went unreported.
    """
    (repo / "m1.py").write_text("def shared(x):\n    return x\n")
    (repo / "m2.py").write_text("def shared(x):\n    return x\n")
    (repo / "u.py").write_text("from m1 import shared\n")
    _commit(repo)
    (repo / "m1.py").write_text("def other(x):\n    return x\n")
    (repo / "m2.py").write_text("def shared(x):\n    return x + 1\n")

    finding = _named(_findings(repo), "shared")
    assert ("u.py", 1) in {(o.path, o.line) for o in finding.unchanged}


def test_a_deleted_call_is_recovered_even_when_the_file_has_an_edited_one(
    repo: Path,
) -> None:
    """Round 6 fixed double-counting with a file-wide skip, which was too coarse.

    One call edited and a second deleted in the same file: the edited call put
    the path in ``touched`` and the skip then suppressed recovery of the deleted
    one, reporting ``modified 1 of 2`` where three sites exist.
    """
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "a.py").write_text(
        "from mod import shared\n\n\ndef f():\n    return shared(1)\n"
        "\n\ndef g():\n    return shared(2)\n"
    )
    (repo / "b.py").write_text("from mod import shared\n\n\ndef h():\n    return shared(3)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x + 1\n")
    (repo / "a.py").write_text(
        "from mod import shared\n\n\ndef f():\n    return shared(11)\n"
        "\n\ndef g():\n    return 0\n"
    )

    finding = _named(_findings(repo), "shared")
    total = len(finding.changed) + len(finding.unchanged)
    assert (len(finding.changed), total) == (2, 3), (
        f"changed={finding.changed} unchanged={finding.unchanged}"
    )
    assert "modified 2 of 3" in checker.render([finding])


def test_base_mode_includes_untracked_files(
    repo: Path, capsys: pytest.CaptureFixture
) -> None:
    """Round 6 pointed --base at the working tree and kept excluding untracked.

    The scans read untracked files either way, so the map and the scan described
    different revisions again — the exact false clean round 6 set out to fix.
    """
    (repo / "a.py").write_text('A = "tool_execution"\n')
    _commit(repo, "base")
    _run(repo, "branch", "base-ref")
    (repo / "mod.py").write_text('M = "tool_execution"\n')
    _commit(repo, "on branch")
    (repo / "newfile.py").write_text('N = "tool_execution"\n')  # never staged

    assert checker.main(["--base", "base-ref"]) == 0
    output = capsys.readouterr().out
    # newfile.py is a CHANGED site, so it shows in the count. Excluding it reads
    # "modified 1 of 3" with two unchanged.
    assert "modified 2 of 3 occurrences" in output, output


# --------------------------------------------------------------------------
# Codex review round 8 — three P2s, no P1s
# --------------------------------------------------------------------------

def test_a_re_export_alias_renamed_by_the_diff_still_reaches_its_caller(
    repo: Path,
) -> None:
    """The closure read only the working tree, where the old alias is gone.

    ``bridge.py`` changes ``as alias`` to ``as alias2``; the downstream
    ``from bridge import alias; alias()`` then mentions neither ``shared`` nor
    ``alias2``, so nothing considered it. Fourth instance in this file of the
    same class — the old side holds the only evidence.
    """
    (repo / "mod.py").write_text("def shared(x):\n    return x\n")
    (repo / "bridge.py").write_text("from mod import shared as alias\n")
    (repo / "user.py").write_text("from bridge import alias\n\n\ndef f():\n    return alias(1)\n")
    _commit(repo)
    (repo / "mod.py").write_text("def shared(x):\n    return x + 1\n")
    (repo / "bridge.py").write_text("from mod import shared as alias2\n")

    finding = _named(_findings(repo), "shared")
    assert ("user.py", 5) in {(o.path, o.line) for o in finding.unchanged}


def test_a_private_re_export_reached_through_the_bridge_module_is_kept(
    repo: Path,
) -> None:
    """``import bridge; bridge.h()`` imports the MODULE, never the alias."""
    (repo / "mine.py").write_text("def _helper(x):\n    return x\n")
    (repo / "bridge.py").write_text("from mine import _helper as h\n")
    (repo / "user.py").write_text("import bridge\n\n\ndef f():\n    return bridge.h(2)\n")
    _commit(repo)
    (repo / "mine.py").write_text("def _helper(x):\n    return x + 1\n")

    finding = _named(_findings(repo), "_helper")
    assert ("user.py", 5) in {(o.path, o.line) for o in finding.unchanged}


def test_a_filename_containing_a_space_is_mapped(repo: Path) -> None:
    """Git appends a tab separator for paths with spaces.

    Slicing ``--- a/`` kept the tab, so the later read targeted a path that does
    not exist and the change was silently skipped — a clean report for a file
    never looked at.
    """
    (repo / "space name.py").write_text('A = "tool_execution"\n')
    (repo / "b.py").write_text('B = "tool_execution"\n')
    _commit(repo)
    (repo / "space name.py").write_text('A = "subagent_dispatch"\n')

    new_map, _ = checker.changed_line_map(["HEAD"])
    assert "space name.py" in new_map, f"path not decoded: {sorted(new_map)}"
    finding = _named(_findings(repo), "tool_execution")
    assert [(o.path, o.line) for o in finding.unchanged] == [("b.py", 1)]


def test_a_c_quoted_filename_is_decoded(repo: Path) -> None:
    """Git C-quotes a path containing non-ASCII, bypassing prefix checks."""
    (repo / "café.py").write_text('A = "tool_execution"\n')
    (repo / "b.py").write_text('B = "tool_execution"\n')
    _commit(repo)
    (repo / "café.py").write_text('A = "subagent_dispatch"\n')

    new_map, _ = checker.changed_line_map(["HEAD"])
    # Assert the EXACT decoded name, and that it names a file that exists.
    # An earlier version asserted `any("caf" in path ...)`, which the undecoded
    # `"caf\303\251.py"` also satisfies — so the test passed with the bug in
    # place and a mutant survived. The decode is only useful if the mapped key
    # can actually be opened, which is what the later file read needs.
    assert "café.py" in new_map, f"quoted path not decoded: {sorted(new_map)}"
    assert (repo / "café.py").exists()

    finding = _named(_findings(repo), "tool_execution")
    assert [(o.path, o.line) for o in finding.unchanged] == [("b.py", 1)]


# --------------------------------------------------------------------------
# A diff BODY line is never a header. With -U0 there are no context lines, but
# removed lines still carry a `-` prefix, so a source line that itself starts
# with `-- ` (a SQL comment at column zero) reads as `--- ...` — the shape of a
# file header. Reading it as one re-keys the old-side map to a bogus path and
# the rename-one-of-N sibling is never surfaced.
# --------------------------------------------------------------------------

def test_a_removed_source_line_that_looks_like_a_header_is_not_one(repo: Path) -> None:
    (repo / "mod.py").write_text(
        'SQL = """\n-- a sql comment at column zero\n"""\n'
        'A = "tool_execution"\nB = "tool_execution"\n'
    )
    (repo / "other.py").write_text('C = "tool_execution"\n')
    _commit(repo)
    # Delete the comment line and rename the literal at both mod.py sites;
    # other.py keeps the old name and must be surfaced.
    (repo / "mod.py").write_text(
        'SQL = """\n"""\nA = "subagent_dispatch"\nB = "subagent_dispatch"\n'
    )

    _, old_map = checker.changed_line_map(["HEAD"])
    assert set(old_map) == {"mod.py"}, f"body line read as a header: {sorted(old_map)}"

    finding = _named(_findings(repo), "tool_execution")
    assert [(o.path, o.line) for o in finding.unchanged] == [("other.py", 1)]


def test_a_removed_line_naming_a_py_file_does_not_become_a_lookup_path(repo: Path) -> None:
    """The spoofed header survives the ``.py`` filter and crashed ``collect``."""
    (repo / "mod.py").write_text(
        'SQL = """\n-- see docs in helper.py\n"""\nA = "tool_execution"\n'
    )
    (repo / "other.py").write_text('C = "tool_execution"\n')
    _commit(repo)
    (repo / "mod.py").write_text('SQL = """\n"""\nA = "subagent_dispatch"\n')

    finding = _named(_findings(repo), "tool_execution")  # must not raise
    assert [(o.path, o.line) for o in finding.unchanged] == [("other.py", 1)]


def test_a_c_quoted_filename_is_searched_for_unchanged_siblings(repo: Path) -> None:
    """Round 9 (`_paths`, NUL-delimited git output) guards the OTHER end of the
    boundary: the non-ASCII file is the untouched sibling, which only the
    `git grep -l` prefilter can find — a C-quoted name there is a path that
    does not exist, and the occurrence silently vanishes."""
    (repo / "a.py").write_text('A = "tool_execution"\n')
    (repo / "café.py").write_text('B = "tool_execution"\n')
    _commit(repo)
    (repo / "a.py").write_text('A = "subagent_dispatch"\n')

    finding = _named(_findings(repo), "tool_execution")
    assert [(o.path, o.line) for o in finding.unchanged] == [("café.py", 1)]
