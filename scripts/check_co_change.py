#!/usr/bin/env python3
"""Surface the sites a diff did **not** touch that share a symbol or literal with it.

Gate 4.1 of ``docs/development/DETERMINISTIC_DEV_GATES.md``. It exists to close a
measured, recurring defect: a fix touches one of N sites sharing an invariant,
the instance goes green, and green terminates the search. Every instance in that
document's evidence table was a ``grep -c`` away — none needed judgement to
*detect*, only to resolve.

So this script only detects. For each function whose body the diff modifies, and
each string literal the diff adds *or removes*, it reports the occurrences
elsewhere in the tree that the diff left alone::

    fold_searchable       modified 1 of 2 call sites
                          unchanged: permissions.py:1346
    "tool_execution"      modified 1 of 5 occurrences
                          unchanged: approval_queue.py:310, :327, :382, :415

Removed literals matter as much as added ones: renaming ``"tool_execution"`` at
one call site is precisely the shape that leaves four siblings behind, and the
new name is not what the siblings still say.

It is **advisory by default and exits 0**. Most of the time some of those sites
are legitimately different and this script cannot know which — that judgement is
the model's, and §7 of the design doc says so explicitly. Failing on every
legitimate co-occurrence would train suppression, and a gate people routinely
suppress is worse than no gate. ``--strict`` exits 1 when anything is surfaced.

Detection is AST and ``git grep``: no network, no model, no side effects.

Known limits, stated so the output is read correctly:

* A **public** symbol is searched repo-wide, so a common name (``collect``,
  ``render``) surfaces unrelated definitions that merely share it. Narrowing
  public names to their importers would be quieter and wrong: a method reached
  as ``self.queue.request_approval(...)`` is never imported by name, and that is
  one of the cases §4.2 exists for. Noise is preferred to a false negative;
  proximity ranking keeps the answerable sites at the top.
* **Module-private** names (leading underscore) are scoped to their own module
  plus explicit importers, by Python convention.
* Only Python is analysed. A literal duplicated into YAML, SQL or TypeScript is
  not seen.

Local usage::

    uv run python scripts/check_co_change.py                  # working tree vs HEAD
    uv run python scripts/check_co_change.py --base main      # whole branch
    uv run python scripts/check_co_change.py --strict         # exit 1 if anything unchanged
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A literal shorter than this matches too much to be a useful signal.
MIN_LITERAL_LEN = 4

# Above this many occurrences a name is structural (``run``, ``get``) rather
# than an invariant. The count stays visible; only the listing is capped.
MAX_SITES_LISTED = 10

# Literals that are punctuation, formatting, or otherwise carry no invariant.
_NOISE_LITERAL = re.compile(r"^[\s\W_]*$|^%[sdrf]$|^\{\}$")

# Dunder plumbing and test scaffolding: shared by construction, not by invariant.
_SKIP_NAMES = frozenset(
    {"__init__", "__repr__", "__str__", "__enter__", "__exit__",
     "__aenter__", "__aexit__", "main", "setUp", "tearDown"}
)


class DetectorBroken(RuntimeError):
    """The search pattern found nothing where a known site must exist.

    ``git grep -E`` accepts ``\\b`` and ``\\s`` and then matches **nothing**,
    exiting 0. A detector that silently finds nothing reports "every site was
    touched" forever, which is worse than no detector at all. Every symbol
    lookup is checked against its own definition to make that failure loud.
    """


@dataclass
class Occurrence:
    path: str
    line: int


@dataclass
class Finding:
    kind: str  # "symbol" or "literal"
    name: str
    changed: list[Occurrence] = field(default_factory=list)
    unchanged: list[Occurrence] = field(default_factory=list)


def _git(*args: str) -> str:
    """Run git in the project root. ``git grep`` exits 1 on no match, which is fine."""
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def changed_line_map(diff_spec: list[str]) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Return (new-side, old-side) maps of file -> changed line numbers.

    ``-U0`` keeps hunk ranges tight; wider context would claim untouched
    neighbours as changed and hide exactly the co-occurrences this looks for.
    The old side is kept because a *removed* literal is what leaves siblings
    behind under the old name.
    """
    diff = _git("diff", "-U0", "--no-color", "--no-ext-diff", *diff_spec)
    new_map: dict[str, set[int]] = defaultdict(set)
    old_map: dict[str, set[int]] = defaultdict(set)
    new_path: str | None = None
    old_path: str | None = None
    for line in diff.splitlines():
        if line.startswith("--- a/"):
            old_path = line[6:]
            continue
        if line.startswith("--- /dev/null"):
            old_path = None
            continue
        if line.startswith("+++ b/"):
            new_path = line[6:]
            continue
        if line.startswith("+++ /dev/null"):
            new_path = None
            continue
        match = _HUNK.match(line)
        if not match:
            continue
        old_start, old_count, new_start, new_count = match.groups()
        if old_path is not None:
            count = 1 if old_count is None else int(old_count)
            for offset in range(count):
                old_map[old_path].add(int(old_start) + offset)
        if new_path is not None:
            count = 1 if new_count is None else int(new_count)
            for offset in range(count):
                new_map[new_path].add(int(new_start) + offset)
    return new_map, old_map


def _parse(source: str, filename: str) -> ast.AST | None:
    try:
        return ast.parse(source, filename=filename)
    except (SyntaxError, ValueError):
        return None


def modified_symbols(source: str, path: str, changed_lines: set[int]) -> set[str]:
    """Names of defs in ``source`` whose body overlaps ``changed_lines``.

    The *definition* is what matters, not the call: changing a function's body
    is what puts its callers in question.
    """
    tree = _parse(source, path)
    if tree is None:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        if node.name in _SKIP_NAMES:
            continue
        if any(node.lineno <= line <= end for line in changed_lines):
            names.add(node.name)
    return names


def literals_on(source: str, path: str, changed_lines: set[int]) -> set[str]:
    """String literals sitting on a changed line — the hardcoded-value shape."""
    tree = _parse(source, path)
    if tree is None:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        if not any(node.lineno <= line <= end for line in changed_lines):
            continue
        value = node.value
        if len(value) < MIN_LITERAL_LEN or _NOISE_LITERAL.match(value):
            continue
        # An invariant carried in a literal is identifier-shaped —
        # ``"tool_execution"``, ``"subagent_dispatch"``, ``"auto_allowed"``.
        # Anything with whitespace is prose or a format fragment; this gate's
        # own report strings were being surfaced against unrelated modules that
        # happened to print the same words.
        if any(character.isspace() for character in value):
            continue
        # And it must carry an actual word: ``"*.py"`` is a pathspec, ``"%s/%s"``
        # a format. Requiring one alphanumeric run of MIN_LITERAL_LEN keeps
        # ``tool_execution`` and drops both.
        if not re.search(rf"[A-Za-z0-9]{{{MIN_LITERAL_LEN},}}", value):
            continue
        found.add(value)
    return found


def symbol_pattern(name: str) -> str:
    """A call-site pattern in POSIX ERE.

    ``git grep -E`` does NOT support ``\\b`` or ``\\s`` — it accepts them and
    matches nothing, exiting 0. Character classes are the portable form.
    """
    return rf"(^|[^[:alnum:]_]){re.escape(name)}[[:space:]]*\("


def import_sites(name: str) -> set[str]:
    """Files that explicitly import ``name``.

    A module-private helper is only shared with modules that import it by name.
    """
    pattern = (
        rf"^[[:space:]]*(from|import)[[:space:]].*[^[:alnum:]_]"
        rf"{re.escape(name)}([^[:alnum:]_]|$)"
    )
    return {o.path for o in _grep(pattern, fixed=False)}


def scope_for(name: str, defining_paths: set[str]) -> set[str] | None:
    """Paths a symbol's call sites may legitimately live in, or None for repo-wide.

    Dogfooding this gate on its own diff reported ``_commit`` as having 33 call
    sites: every test module in the repo defines its own private ``_commit``
    helper, and they are unrelated functions that merely share a name. Treating
    a leading-underscore name as global manufactures dozens of false siblings —
    precisely the noise that teaches people to suppress the gate.

    By Python convention a leading underscore means module-private, so its real
    call sites are its own module plus whatever imports it by name.
    """
    if not name.startswith("_"):
        return None
    return defining_paths | import_sites(name)


def _grep(pattern: str, *, fixed: bool) -> list[Occurrence]:
    output = _git("grep", "-n", "-F" if fixed else "-E", "--", pattern, "*.py")
    found: list[Occurrence] = []
    for line in output.splitlines():
        parts = line.split(":", 2)
        if len(parts) >= 3 and parts[1].isdigit():
            found.append(Occurrence(parts[0], int(parts[1])))
    return found


def _is_definition(occurrence: Occurrence, name: str, blobs: dict[str, list[str]]) -> bool:
    lines = blobs.get(occurrence.path)
    if lines is None:
        try:
            lines = (PROJECT_ROOT / occurrence.path).read_text(
                encoding="utf-8"
            ).splitlines()
        except (OSError, ValueError):
            lines = []
        blobs[occurrence.path] = lines
    if not 0 < occurrence.line <= len(lines):
        return False
    return bool(
        re.match(rf"\s*(async\s+def|def|class)\s+{re.escape(name)}\b", lines[occurrence.line - 1])
    )


def _split(
    occurrences: list[Occurrence], changed: dict[str, set[int]]
) -> tuple[list[Occurrence], list[Occurrence]]:
    touched, untouched = [], []
    for occurrence in occurrences:
        (touched if occurrence.line in changed.get(occurrence.path, ()) else untouched).append(
            occurrence
        )
    return touched, untouched


def collect(
    new_map: dict[str, set[int]],
    old_map: dict[str, set[int]],
    old_ref: str,
) -> list[Finding]:
    symbols: dict[str, set[str]] = defaultdict(set)
    literals: set[str] = set()
    blobs: dict[str, list[str]] = {}

    for path, lines in new_map.items():
        if not path.endswith(".py"):
            continue
        try:
            source = (PROJECT_ROOT / path).read_text(encoding="utf-8")
        except (OSError, ValueError):
            continue
        blobs[path] = source.splitlines()
        for name in modified_symbols(source, path, lines):
            symbols[name].add(path)
        literals |= literals_on(source, path, lines)

    # Removed literals: renaming one of N sites leaves the others under the OLD
    # name, which never appears on the new side at all.
    for path, lines in old_map.items():
        if not path.endswith(".py"):
            continue
        source = _git("show", f"{old_ref}:{path}")
        if source:
            literals |= literals_on(source, path, lines)

    findings: list[Finding] = []

    for name, defining_paths in sorted(symbols.items()):
        sites = _grep(symbol_pattern(name), fixed=False)
        if not sites:
            raise DetectorBroken(
                f"no occurrence of {name!r} found, but its definition was just "
                f"parsed from the diff — the search pattern is not matching."
            )
        scope = scope_for(name, defining_paths)
        if scope is not None:
            sites = [o for o in sites if o.path in scope]
        calls = [o for o in sites if not _is_definition(o, name, blobs)]
        touched, untouched = _split(calls, new_map)
        if touched and untouched:
            findings.append(Finding("symbol", name, touched, untouched))

    for value in sorted(literals):
        sites = _grep(value, fixed=True)
        touched, untouched = _split(sites, new_map)
        # A removed literal has no touched site on the new side; its whole point
        # is the siblings still carrying the old name.
        if untouched and (touched or value not in _new_side_literals(new_map, blobs)):
            findings.append(Finding("literal", value, touched, untouched))

    return findings


def _new_side_literals(new_map: dict[str, set[int]], blobs: dict[str, list[str]]) -> set[str]:
    """Literals present on the new side, used to tell added from removed."""
    present: set[str] = set()
    for path, lines in new_map.items():
        if path.endswith(".py") and path in blobs:
            present |= literals_on("\n".join(blobs[path]), path, lines)
    return present


def rank_unchanged(
    unchanged: list[Occurrence], changed: list[Occurrence], origin: set[str]
) -> list[Occurrence]:
    """Order unchanged sites by proximity to where the diff actually is.

    ``git grep`` returns path order, which buries the signal: reproducing the
    #3107 round-2 mistake surfaced the four sibling sites in the very file being
    edited *below* thirty unrelated hits elsewhere in the tree. A site in the
    file you just changed is the one you can answer immediately; rank on that.
    """
    near = {o.path for o in changed} | origin
    dirs = {str(Path(p).parent) for p in near}

    def key(occurrence: Occurrence) -> tuple[int, str, int]:
        if occurrence.path in near:
            tier = 0
        elif str(Path(occurrence.path).parent) in dirs:
            tier = 1
        else:
            tier = 2
        return (tier, occurrence.path, occurrence.line)

    return sorted(unchanged, key=key)


def render(findings: list[Finding], origin: set[str] | None = None) -> str:
    if not findings:
        return "co-change: every site sharing a changed symbol or literal was touched."

    out = [
        "co-change: the diff shares a symbol or literal with sites it did not touch.",
        "Each line is a QUESTION, not a defect — decide whether the site holds the",
        "same invariant. See docs/development/DETERMINISTIC_DEV_GATES.md §4.1.",
        "",
    ]
    width = min(max(len(_label(f)) for f in findings), 40)
    for finding in findings:
        total = len(finding.changed) + len(finding.unchanged)
        noun = "call site" if finding.kind == "symbol" else "occurrence"
        out.append(
            f"  {_label(finding):<{width}}  modified {len(finding.changed)} of "
            f"{total} {noun}{'s' if total != 1 else ''}"
        )
        ordered = rank_unchanged(finding.unchanged, finding.changed, origin or set())
        shown = ordered[:MAX_SITES_LISTED]
        listing = ", ".join(f"{o.path}:{o.line}" for o in shown)
        extra = len(finding.unchanged) - len(shown)
        if extra > 0:
            listing += f", +{extra} more"
        out.append(f"  {'':<{width}}  unchanged: {listing}")
    return "\n".join(out)


def _label(finding: Finding) -> str:
    return f'"{finding.name}"' if finding.kind == "literal" else finding.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="diff against this ref (e.g. --base main)")
    parser.add_argument(
        "--strict", action="store_true",
        help="exit 1 when any unchanged co-occurrence is surfaced",
    )
    args = parser.parse_args(argv)

    if args.base:
        diff_spec = [f"{args.base}...HEAD"]
        old_ref = _git("merge-base", args.base, "HEAD").strip() or args.base
    else:
        diff_spec = ["HEAD"]
        old_ref = "HEAD"

    new_map, old_map = changed_line_map(diff_spec)
    if not new_map and not old_map:
        print("co-change: no changed lines to analyse.")
        return 0

    findings = collect(new_map, old_map, old_ref)
    # The files the diff touches are the frame of reference for ranking, even
    # for a literal whose every occurrence is elsewhere (the removed-name case).
    print(render(findings, origin=set(new_map) | set(old_map)))
    return 1 if (findings and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
