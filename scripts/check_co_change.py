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

# A name used more times than this cannot be reviewed site-by-site — running
# ``--base main`` over a whole branch reported ``execute`` at 2323 call sites and
# ``initialize`` at 1354. Those are counted and named in a footer, never dropped
# silently: a hidden cap reads as "covered everything".
MAX_TOTAL_SITES = 100

# A constant's source spelling need not contain its value (``"tool\x5fexecution"``,
# or ``"tool_" "execution"`` concatenated), so the fixed-text prefilter can miss a
# sibling. Searching the value's longest alphanumeric run recovers those, but only
# a distinctive run is worth the extra files to index.
MIN_WIDENED_RUN = 8

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
    end_line: int | None = None

    @property
    def span(self) -> range:
        """Every line the use occupies.

        A multiline call whose *argument* changed is a changed call site, but
        only the callee-name line was ever compared, so it read as untouched.
        """
        return range(self.line, (self.end_line or self.line) + 1)


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


def changed_line_map(
    diff_spec: list[str], *, include_untracked: bool = True
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
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
    if include_untracked:
        # `git diff` never mentions an untracked file, so a brand-new module —
        # every line of it new — was invisible and the gate reported clean on
        # it. Treat an untracked Python file as wholly changed.
        for path in untracked_python_files():
            try:
                count = len((PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines())
            except (OSError, ValueError):
                continue
            new_map[path] = set(range(1, count + 1))

    return dict(new_map), dict(old_map)


def _parse(source: str, filename: str) -> ast.AST | None:
    try:
        return ast.parse(source, filename=filename)
    except (SyntaxError, ValueError):
        return None


def _candidate_files(pattern: str, *, fixed: bool) -> set[str]:
    """Files that mention the text at all — a cheap prefilter, never the answer.

    ``git grep`` decides nothing here: it narrows the tree so the AST pass has a
    small set to parse. Every candidate is then verified semantically.
    """
    # ``--untracked`` because round 4 taught changed_line_map about untracked
    # files and left this half behind: a new file entered the changed map but
    # was never searched, so its occurrence could not be counted as touched and
    # the gate reported clean. A boundary has two ends.
    output = _git(
        "grep", "-l", "--untracked", "-F" if fixed else "-E", "--", pattern, "*.py"
    )
    return {line for line in output.splitlines() if line}


_TREES: dict[tuple[str, str], ast.AST | None] = {}


def _tree_for(path: str, ref: str = "") -> ast.AST | None:
    """Parse ``path`` once per run. ``ref`` empty means the working tree."""
    key = (path, ref)
    if key not in _TREES:
        if ref:
            source = _git("show", f"{ref}:{path}")
        else:
            try:
                source = (PROJECT_ROOT / path).read_text(encoding="utf-8")
            except (OSError, ValueError):
                source = ""
        _TREES[key] = _parse(source, path) if source else None
    return _TREES[key]


def _def_start(node: ast.AST) -> int:
    """The first line that belongs to a definition, decorators included.

    ``FunctionDef.lineno`` points at the ``def``, so a diff that edits only
    ``@decorator`` fell outside the span and the function never registered as
    changed — its callers went unasked and the gate reported clean.
    """
    lines = [node.lineno]
    lines += [d.lineno for d in getattr(node, "decorator_list", [])]
    return min(lines)


def _name_line(node: ast.AST) -> int:
    """The line the *name* is on, not where the expression started.

    For ``obj.method(...)`` spanning lines, ``Attribute.lineno`` points at
    ``obj``; the name the reader is looking for is at the end.
    """
    if isinstance(node, ast.Attribute):
        return getattr(node, "end_lineno", node.lineno) or node.lineno
    return node.lineno
_INDEXES: dict[tuple[str, str], FileIndex | None] = {}


@dataclass
class FileIndex:
    """Everything this gate asks of one file, collected in a single AST walk.

    Walking per question was O(files x questions): the literal pass alone made
    3113 walks over 9.1M nodes and took 14 of the run's 20 seconds. Every lookup
    below is now a dict hit against one walk per file.
    """

    strings: dict[str, list[int]] = field(default_factory=dict)
    calls: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    refs: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    import_bindings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    definitions: set[str] = field(default_factory=set)
    aliases: dict[str, set[str]] = field(default_factory=dict)
    imported_modules: set[str] = field(default_factory=set)


def _build_index(tree: ast.AST) -> FileIndex:
    index = FileIndex()
    counted: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            index.strings.setdefault(node.value, []).append(node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            index.definitions.add(node.name)
        elif isinstance(node, ast.Call):
            func = node.func
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            if isinstance(func, ast.Name):
                counted.add(id(func))
                index.calls.setdefault(func.id, []).append((_name_line(func), end))
            elif isinstance(func, ast.Attribute):
                counted.add(id(func))
                index.calls.setdefault(func.attr, []).append((_name_line(func), end))
        elif isinstance(node, ast.ImportFrom):
            for entry in node.names:
                index.aliases.setdefault(entry.name, set()).add(entry.asname or entry.name)
                index.imported_modules.add(entry.name)
                # The binding depends on the name EXISTING, not on what it
                # does. Renaming or deleting the definition makes importing
                # this file raise ImportError — and with no call anywhere,
                # `sites` was empty and the gate called that clean. Kept apart
                # from `refs` because a body change does not touch an import,
                # and counting it there made every import an unreviewed site.
                index.import_bindings.setdefault(entry.name, []).append(
                    (node.lineno, node.lineno)
                )
            if node.module:
                index.imported_modules.add(node.module)
                index.imported_modules.add(node.module.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Import):
            for entry in node.names:
                index.imported_modules.add(entry.name)
                index.imported_modules.add(entry.name.rsplit(".", 1)[-1])

    # Bare references, excluding the callee nodes already recorded as calls, so
    # ``Worker()`` is one use rather than two.
    for node in ast.walk(tree):
        if id(node) in counted:
            continue
        if isinstance(node, ast.Name):
            index.refs.setdefault(node.id, []).append((node.lineno, node.lineno))
        elif isinstance(node, ast.Attribute):
            line = _name_line(node)
            index.refs.setdefault(node.attr, []).append((line, line))
    return index


def index_for(path: str, ref: str = "") -> FileIndex | None:
    key = (path, ref)
    if key not in _INDEXES:
        tree = _tree_for(path, ref)
        _INDEXES[key] = _build_index(tree) if tree is not None else None
    return _INDEXES[key]


def untracked_python_files() -> set[str]:
    """Python files git does not track yet — invisible to ``git diff``."""
    output = _git("ls-files", "--others", "--exclude-standard", "--", "*.py")
    return {line for line in output.splitlines() if line}


def _files_matching_all(*patterns: str) -> set[str]:
    """Files matching EVERY pattern — a cheap conjunction the AST then judges."""
    args = ["grep", "-l", "--untracked", "--all-match", "-E"]
    for pattern in patterns:
        args += ["-e", pattern]
    output = _git(*args, "--", "*.py")
    return {line for line in output.splitlines() if line}


def _word_pattern(word: str) -> str:
    """POSIX ERE matching ``word`` as a whole identifier."""
    return rf"(^|[^[:alnum:]_]){re.escape(word)}([^[:alnum:]_]|$)"


def alias_closure(name: str, *, max_rounds: int = 3) -> set[str]:
    """Every local name that ultimately refers to ``name``, through re-exports.

    A bridge module doing ``from mod import shared as alias`` and a downstream
    ``from bridge import alias; alias()`` means the caller's file never contains
    the string ``shared`` at all — so a prefilter on the original name excluded
    it and the dependency reported clean. Resolving one hop was not enough
    because the binding that reaches the caller is the bridge's, not the
    definition's.

    Bounded rounds: re-export chains are short, and an unbounded walk over a
    large tree is not worth the tail.
    """
    names = {name}
    for _ in range(max_rounds):
        discovered: set[str] = set()
        for current in names:
            # Only a RENAMING import can introduce a name the plain prefilter
            # misses. Requiring `import <current> as` on ONE line missed Black's
            # parenthesized form entirely, which is how this repository formats
            # long import lists. Ask instead for files mentioning both the name
            # and an `as` anywhere, then let the AST decide — that stays cheap
            # (the broad name-only pattern took the gate from 0.8s to 13s) and
            # is not line-oriented.
            for path in _files_matching_all(
                _word_pattern(current), r"[^[:alnum:]_]as[^[:alnum:]_]"
            ):
                index = index_for(path)
                if index is None:
                    continue
                for bound in index.aliases.get(current, ()):
                    if bound not in names:
                        discovered.add(bound)
        if not discovered:
            break
        names |= discovered
    return names


def call_sites(
    index: FileIndex,
    name: str,
    *,
    kind: str,
    local_names: set[str] | None = None,
    include_imports: bool = False,
) -> list[tuple[int, int]]:
    """(start, end) spans where ``name`` is genuinely *used*, per the AST.

    A textual search cannot tell a call from a string that contains one. This
    repository's tests embed Python source as string literals constantly, so
    ``TEXT = "helper(1)"`` was being reported as an unchanged call site of
    ``helper``. Only real syntax nodes count here.

    A bare reference counts too, not only a call. ``map(shared, items)``, a
    callback handed to a registry, and ``obj.status`` for an ``@property`` are
    all real dependencies that are never the callee of an ``ast.Call``; treating
    only calls as uses let a body-only edit report clean. The same is true of a
    class edited in place but never instantiated in the diff's view.

    The span covers the whole call expression, so changing only an argument on a
    later line still marks the site as touched.
    """
    del kind  # every kind counts bare references now
    spans: list[tuple[int, int]] = []
    for local in local_names or {name}:
        spans.extend(index.calls.get(local, ()))
        spans.extend(index.refs.get(local, ()))
        if include_imports:
            spans.extend(index.import_bindings.get(local, ()))
    return spans


def literal_sites(index: FileIndex, value: str) -> list[int]:
    """Lines holding a string constant *equal to* ``value``.

    ``git grep -F "config"`` matches ``base_config``, comments, and every longer
    string containing it — 8,241 hits in this repository against a handful of
    real constants. Equality on an AST constant is the actual question.
    """
    return list(index.strings.get(value, ()))


def defines(index: FileIndex, name: str) -> bool:
    return name in index.definitions


def import_sites(name: str) -> set[str]:
    """Files that import ``name`` by name, per the AST."""
    pattern = rf"(^|[^[:alnum:]_]){re.escape(name)}([^[:alnum:]_]|$)"
    found: set[str] = set()
    for path in _candidate_files(pattern, fixed=False):
        tree = _tree_for(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == name or alias.asname == name for alias in node.names
            ):
                found.add(path)
                break
    return found


def module_importers(defining_paths: set[str]) -> set[str]:
    """Files that import one of the modules in ``defining_paths``.

    A private module function is often reached as ``cli._get_project_dir()`` by a
    file that imports ``cli``, not the name. Scoping on the *name*'s importers
    alone excluded those real callers.
    """
    modules: set[str] = set()
    for path in defining_paths:
        candidate = Path(path)
        # `pkg/__init__.py` IS the package: its importable name is the parent
        # directory, and searching for `__init__` finds nothing real.
        modules.add(candidate.parent.name if candidate.stem == "__init__" else candidate.stem)

    found: set[str] = set()
    for module in modules:
        # The boundary must NOT exclude a preceding dot: `import pkg.cli as c`
        # is exactly the shape being looked for, and excluding `.` discarded
        # the file before the dotted-name check below could ever see it.
        pattern = _word_pattern(module)
        for path in _candidate_files(pattern, fixed=False):
            index = index_for(path)
            if index is None:
                continue
            if module in index.imported_modules or any(
                mod.endswith(f".{module}") for mod in index.imported_modules
            ):
                found.add(path)
    return found


def scope_for(name: str, kind: str, defining_paths: set[str]) -> set[str] | None:
    """Paths a symbol's uses may live in, or None for repo-wide.

    Dogfooding reported ``_commit`` as having 33 call sites: every test module
    defines its own private ``_commit``, and they are unrelated functions that
    merely share a name. A leading underscore means module-private, so its real
    uses are its own module plus whatever imports it by name.

    This applies to **module-level functions only**. A private *method* is
    reached as ``queue._dispatch()`` from modules that import ``Queue``, never
    ``_dispatch`` — scoping those to importers of the method name would drop
    exactly the external callers this gate exists to surface.
    """
    if kind != "function" or not name.startswith("_"):
        return None
    return defining_paths | import_sites(name) | module_importers(defining_paths)


def modified_symbols(
    source: str, path: str, changed_lines: set[int]
) -> dict[str, str]:
    """Defs in ``source`` whose body overlaps ``changed_lines``, mapped to kind.

    The *definition* is what matters, not the call: changing a function's body
    is what puts its callers in question. Kind separates a module-level
    function from a method, which scope differently.
    """
    tree = _parse(source, path)
    if tree is None:
        return {}

    methods: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(id(child))

    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name in _SKIP_NAMES:
            continue
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        if not any(_def_start(node) <= line <= end for line in changed_lines):
            continue
        if isinstance(node, ast.ClassDef):
            kind = "class"
        elif id(node) in methods:
            kind = "method"
        else:
            kind = "function"
        # A method definition outranks a same-named module function: it is the
        # weaker scope, and picking the stronger one would hide call sites.
        if found.get(node.name) != "method":
            found[node.name] = kind
    return found


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
        # "tool_execution", "subagent_dispatch", "auto_allowed". Anything with
        # whitespace is prose or a format fragment.
        if any(character.isspace() for character in value):
            continue
        # And it must carry an actual word: "*.py" is a pathspec, "%s/%s" a
        # format. One alphanumeric run of MIN_LITERAL_LEN keeps the real ones.
        if not re.search(rf"[A-Za-z0-9]{{{MIN_LITERAL_LEN},}}", value):
            continue
        found.add(value)
    return found


def _split(
    occurrences: list[Occurrence], changed: dict[str, set[int]]
) -> tuple[list[Occurrence], list[Occurrence]]:
    touched, untouched = [], []
    for occurrence in occurrences:
        lines = changed.get(occurrence.path, ())
        hit = any(line in lines for line in occurrence.span)
        (touched if hit else untouched).append(occurrence)
    return touched, untouched


def collect(
    new_map: dict[str, set[int]],
    old_map: dict[str, set[int]],
    old_ref: str,
) -> tuple[list[Finding], list[tuple[str, int]], list[str]]:
    """Return (findings, structural, unparseable).

    ``structural`` names are counted but not listed; ``unparseable`` are changed
    Python files that could not be analysed at all.
    """
    _TREES.clear()
    _INDEXES.clear()
    symbols: dict[str, tuple[str, set[str]]] = {}
    literals: set[str] = set()

    unparseable: list[str] = []
    for path, lines in new_map.items():
        if not path.endswith(".py"):
            continue
        try:
            source = (PROJECT_ROOT / path).read_text(encoding="utf-8")
        except (OSError, ValueError):
            continue
        # A changed file that will not parse yields no symbols and no literals,
        # which renders as "every site was touched". On a local iteration gate a
        # half-written file is normal; silently converting it to a clean bill of
        # health is not.
        if _parse(source, path) is None:
            unparseable.append(path)
            continue
        for name, kind in modified_symbols(source, path, lines).items():
            existing_kind, paths = symbols.get(name, (kind, set()))
            symbols[name] = (
                "method" if "method" in (existing_kind, kind) else kind,
                paths | {path},
            )
        literals |= literals_on(source, path, lines)

    # Removed literals: renaming one of N sites leaves the others under the OLD
    # name, which never appears on the new side at all.
    old_sources: dict[str, str] = {}
    for path, lines in old_map.items():
        if not path.endswith(".py"):
            continue
        source = _git("show", f"{old_ref}:{path}")
        if source:
            old_sources[path] = source
            literals |= literals_on(source, path, lines)
            # A body edit made entirely of DELETED lines contributes nothing to
            # the new-side map, so the function never entered `symbols` and its
            # callers were never questioned. The definition still exists in the
            # new tree; only the evidence that it changed lives on the old side.
            for name, kind in modified_symbols(source, path, lines).items():
                # Do NOT require the working tree to still define this name. A
                # DELETED or RENAMED definition is exactly the case that leaves
                # callers stranded under the old name, and requiring the new
                # tree to define it dropped every one of them — a false clean
                # introduced by the previous round's own fix.
                existing_kind, paths = symbols.get(name, (kind, set()))
                symbols[name] = (
                    "method" if "method" in (existing_kind, kind) else kind,
                    paths | {path},
                )

    findings: list[Finding] = []
    structural: list[tuple[str, int]] = []

    for name, (kind, defining_paths) in sorted(symbols.items()):
        # Follow re-export bridges: a caller may only ever mention the alias.
        local_names = alias_closure(name)
        scope = scope_for(name, kind, defining_paths)
        if scope is not None:
            # The closure can discover a name the scope has never heard of. A
            # private helper re-exported as `h` is reached by a module that
            # imports `h`, not `_helper` and not the defining module, so the
            # intersection removed exactly the caller the closure just found.
            for alias in local_names - {name}:
                scope = scope | import_sites(alias)
        candidates: set[str] = set()
        for local in local_names:
            candidates |= _candidate_files(_word_pattern(local), fixed=False)
        if scope is not None:
            candidates &= scope

        # Positive control: the definition we just parsed must be findable.
        # Asserting on *call* sites instead would abort on any symbol that is
        # simply never called — which is how an ordinary `class Worker:` edit
        # took the whole gate down. A renamed or deleted symbol lives only on
        # the old side, so look there too before declaring the pass broken.
        defined_now = any(
            (index := index_for(path)) is not None and defines(index, name)
            for path in defining_paths
        )
        defined_before = any(
            (tree := _parse(source, path)) is not None
            and defines(_build_index(tree), name)
            for path, source in old_sources.items()
        )
        if not (defined_now or defined_before):
            raise DetectorBroken(
                f"the definition of {name!r} was parsed from the diff but cannot "
                f"be found again in {sorted(defining_paths)} — the AST pass is broken."
            )

        sites: list[Occurrence] = []
        for path in sorted(candidates):
            index = index_for(path)
            if index is None:
                continue
            sites.extend(
                Occurrence(path, start, end)
                for start, end in call_sites(
                    index,
                    name,
                    kind=kind,
                    local_names=local_names,
                    include_imports=not defined_now,
                )
            )

        if len(sites) > MAX_TOTAL_SITES:
            structural.append((name, len(sites)))
            continue
        touched, untouched = _split(sites, new_map)

        # A call REMOVED by the diff is gone from the working tree, so the scan
        # above cannot see it and the report understates both counts — the same
        # blind spot already fixed for removed literals.
        for path, source in old_sources.items():
            tree = _parse(source, path)
            if tree is None:
                continue
            old_index = _build_index(tree)
            for start, end in call_sites(
                old_index,
                name,
                kind=kind,
                local_names=local_names,
                include_imports=not defined_now,
            ):
                if any(
                    line in old_map.get(path, ()) for line in range(start, end + 1)
                ) and not any(o.path == path and o.line == start for o in touched):
                    touched.append(Occurrence(path, start, end))

        # `touched and untouched` discarded the gate's whole motivating case:
        # change a shared function's body, leave every caller alone, and it
        # reported CLEAN. The changed *definition* is what puts the callers in
        # question — whether any call expression also moved is beside the point.
        if untouched:
            findings.append(Finding("symbol", name, touched, untouched))

    for value in sorted(literals):
        # `"tool\x5fexecution"` and `"tool_" "execution"` are equal to
        # `"tool_execution"` at runtime but do not contain it in source, so a
        # fixed-text prefilter dropped those files before the AST equality check
        # could see them. Also search the value's longest word run, which
        # survives both escaping and implicit concatenation at a separator.
        runs = re.findall(r"[A-Za-z0-9]+", value)
        longest = max(runs, key=len, default="")
        candidates = _candidate_files(value, fixed=True)
        # Widen only on a run distinctive enough to stay cheap. `"__init__"`
        # widens to `init`, which matches most of the tree: indexing that
        # candidate set took the gate from 0.8s to 16s. A short run buys a rare
        # case at a cost paid on every single run, which is the wrong trade for
        # a gate that runs per iteration.
        if len(longest) >= MIN_WIDENED_RUN and longest != value:
            candidates |= _candidate_files(longest, fixed=True)

        sites: list[Occurrence] = []
        for path in sorted(candidates):
            index = index_for(path)
            if index is None:
                continue
            sites.extend(Occurrence(path, line) for line in literal_sites(index, value))

        touched, untouched = _split(sites, new_map)

        # A literal REMOVED at a site is gone from the new tree entirely, so the
        # grep above cannot see it and the report read "modified 0 of 2" where
        # the contract promises "modified 1 of 3". Count it from the old blob.
        for path, source in old_sources.items():
            tree = _parse(source, path)
            if tree is None:
                continue
            for line in literal_sites(_build_index(tree), value):
                if line in old_map.get(path, ()) and not any(
                    o.path == path and o.line == line for o in touched
                ):
                    touched.append(Occurrence(path, line))

        if len(touched) + len(untouched) > MAX_TOTAL_SITES:
            structural.append((f'"{value}"', len(touched) + len(untouched)))
            continue
        if touched and untouched:
            findings.append(Finding("literal", value, touched, untouched))

    return findings, structural, unparseable


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


def _structural_footer(structural: list[tuple[str, int]]) -> str:
    listed = ", ".join(f"{name} ({count})" for name, count in structural)
    return (
        "  not reviewed site-by-site (used too widely to be an invariant): "
        f"{listed}"
    )


def _unparseable_footer(unparseable: list[str]) -> str:
    return (
        "  NOT ANALYSED — these changed files could not be parsed, so nothing "
        "about them was checked: " + ", ".join(sorted(unparseable))
    )


def render(
    findings: list[Finding],
    origin: set[str] | None = None,
    structural: list[tuple[str, int]] | None = None,
    unparseable: list[str] | None = None,
) -> str:
    structural = structural or []
    unparseable = unparseable or []
    if not findings:
        # "Every site was touched" is a claim about coverage, and it is false
        # when a symbol was skipped for being too widely used or a file could
        # not be parsed at all. Say what was and was not looked at.
        if structural or unparseable:
            out = ["co-change: no reviewable co-occurrence found, but coverage was incomplete."]
            if structural:
                out.append(_structural_footer(structural))
            if unparseable:
                out.append(_unparseable_footer(unparseable))
            return "\n".join(out)
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
        if finding.kind == "symbol" and not finding.changed:
            # The definition changed and no call site did. That is the gate's
            # motivating case, not an empty result — say so plainly.
            headline = f"definition changed; {total} {noun}{'s' if total != 1 else ''} untouched"
        else:
            headline = (
                f"modified {len(finding.changed)} of {total} "
                f"{noun}{'s' if total != 1 else ''}"
            )
        out.append(f"  {_label(finding):<{width}}  {headline}")
        ordered = rank_unchanged(finding.unchanged, finding.changed, origin or set())
        shown = ordered[:MAX_SITES_LISTED]
        listing = ", ".join(f"{o.path}:{o.line}" for o in shown)
        extra = len(finding.unchanged) - len(shown)
        if extra > 0:
            listing += f", +{extra} more"
        out.append(f"  {'':<{width}}  unchanged: {listing}")
    if structural or unparseable:
        out.append("")
        if structural:
            out.append(_structural_footer(structural))
        if unparseable:
            out.append(_unparseable_footer(unparseable))
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

    new_map, old_map = changed_line_map(diff_spec, include_untracked=not args.base)
    if not new_map and not old_map:
        print("co-change: no changed lines to analyse.")
        return 0

    findings, structural, unparseable = collect(new_map, old_map, old_ref)
    # The files the diff touches are the frame of reference for ranking, even
    # for a literal whose every occurrence is elsewhere (the removed-name case).
    print(
        render(
            findings,
            origin=set(new_map) | set(old_map),
            structural=structural,
            unparseable=unparseable,
        )
    )
    # --strict must not pass on incomplete coverage: a symbol skipped for being
    # too widely used, or a file that would not parse, is an unanswered question
    # exactly like a listed one.
    return 1 if (args.strict and (findings or structural or unparseable)) else 0


if __name__ == "__main__":
    sys.exit(main())
