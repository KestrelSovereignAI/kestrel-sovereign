"""Every storage-root environment variable is guarded, by construction (#3286).

A hand-maintained list of "the resolvers that call the operator-root guard" is
a claim, and two review rounds each found one it missed. This census makes it a
test: it walks the package's AST, finds every environment lookup whose key
names a storage-root-shaped variable, and requires each one to be either

* in ``paths.RUNTIME_PATH_ENV_NAMES``, the guarded registry, **and** read inside
  a scope that calls ``paths.guard_storage_path`` (``paths.runtime_path_env``
  is that scope for most callers); or
* in :data:`NOT_STORAGE_ROOTS` below, with the reason it is not one.

The same registry drives the harness's per-test pins
(``tests/shared/host_runtime_isolation.py``), so the lists cannot drift apart.
The guard itself is an allow-list of the test's temporary roots, so it needs
no list of the operator's: this census is what makes sure it is called.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from kestrel_sovereign import paths
from tests.shared.host_runtime_isolation import ISOLATED_LAYOUT

PACKAGE_ROOT = Path(paths.__file__).resolve().parent

#: The shape of a variable that could name a storage root. Keyed on the
#: suffix, not a ``KESTREL_`` prefix: ``LOCAL_MPS_WORKING_DIR`` named a
#: directory Kestrel creates, and a prefix-keyed census never saw it.
STORAGE_ROOT_SHAPE = re.compile(r"[A-Z][A-Z0-9_]*_(DIR|PATH|HOME|ROOT|FILE)")

#: Names of the guard. A read inside a scope that calls one of these is guarded.
GUARD_CALLS = frozenset({"guard_storage_path"})

#: Variables that match :data:`STORAGE_ROOT_SHAPE` but are not storage roots.
NOT_STORAGE_ROOTS: dict[str, str] = {
    # A read-only, operator-pinned trust-root *file*: configuration the
    # constitution tools verify against, never a directory Kestrel writes.
    "KESTREL_SOVEREIGN_TRUST_ROOT_PATH": "read-only trust-root file",
    # The Kite release-evidence tool's pinned signing-key root: read-only
    # verification material the evidence runner exports to its own child.
    "KESTREL_KITE_RELEASE_EVIDENCE_ROOT": "read-only evidence trust root",
    # A launcher-written peer DID registry file the child reads and verifies
    # against a pinned digest; nothing is written there.
    "KESTREL_A2A_PEER_IDENTITY_DOCUMENTS_FILE": "read-only attested registry file",
    # Credential stores owned by other tools (Claude Code, the Codex CLI).
    # Kestrel reads (Codex: symlinks) their existing credentials; it does not
    # create state there, and redirecting them would sign tests out.
    "KESTREL_ANTHROPIC_OAUTH_CREDENTIALS_FILE": "operator credential file",
    "CODEX_HOME": "the Codex CLI's own credential home",
    # Read-only training inputs: a model in diffusers format and the
    # diffusers checkout that holds the training scripts.
    "LOCAL_MPS_MODEL_PATH": "read-only model input",
    "DIFFUSERS_PATH": "read-only diffusers checkout",
    # The uv executor's per-run cache inside its own fresh workspace, read
    # from the environment dict it just built, never from the process's.
    "UV_CACHE_DIR": "per-execution workspace cache",
}

#: Reads. ``pop``/``setdefault`` and item assignment are mutations: a caller
#: saving and restoring a variable resolves nothing.
_ENV_METHODS = frozenset({"get"})


@dataclass(frozen=True)
class Lookup:
    variable: str
    file: Path
    line: int
    guarded: bool

    def where(self) -> str:
        return f"{self.file.relative_to(PACKAGE_ROOT.parent)}:{self.line}"


def _string_constants(trees: dict[Path, ast.Module]) -> dict[str, str]:
    """``NAME = "KESTREL_..."`` bindings anywhere in the package, by name."""
    constants: dict[str, str] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    for target in targets:
                        if isinstance(target, ast.Name):
                            constants.setdefault(target.id, value.value)
    return constants


def _key(node: ast.AST, constants: dict[str, str]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value
    elif isinstance(node, ast.Name):
        value = constants.get(node.id)
    elif isinstance(node, ast.Attribute):
        value = constants.get(node.attr)
    else:
        return None
    if value is not None and (
        STORAGE_ROOT_SHAPE.fullmatch(value) or value in paths.RUNTIME_PATH_ENV_NAMES
    ):
        return value
    return None


def _called_names(owned: list[ast.AST]) -> set[str]:
    names = set()
    for node in owned:
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _lookups_in(
    node: ast.AST, constants: dict[str, str]
) -> Iterator[tuple[str, int]]:
    if isinstance(node, ast.Call):
        func = node.func
        is_env_method = isinstance(func, ast.Attribute) and func.attr in _ENV_METHODS
        is_getenv = (isinstance(func, ast.Attribute) and func.attr == "getenv") or (
            isinstance(func, ast.Name) and func.id == "getenv"
        )
        if (is_env_method or is_getenv) and node.args:
            variable = _key(node.args[0], constants)
            if variable:
                yield variable, node.lineno
    elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        variable = _key(node.slice, constants)
        if variable:
            yield variable, node.lineno


def _scopes(tree: ast.Module) -> Iterator[tuple[ast.AST, list[ast.AST]]]:
    """Each function (and the module) with the nodes it owns directly."""
    pending: list[ast.AST] = [tree]
    while pending:
        scope = pending.pop()
        owned: list[ast.AST] = []
        stack = list(ast.iter_child_nodes(scope))
        while stack:
            child = stack.pop()
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                pending.append(child)
                continue
            owned.append(child)
            stack.extend(ast.iter_child_nodes(child))
        yield scope, owned


def _package_trees() -> dict[Path, ast.Module]:
    return {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
    }


def census(trees: Optional[dict[Path, ast.Module]] = None) -> list[Lookup]:
    package = _package_trees()
    trees = package if trees is None else trees
    # Constants resolve package-wide: a reader may import its key's name.
    constants = _string_constants({**package, **trees})
    found: list[Lookup] = []
    for path, tree in trees.items():
        for scope, owned in _scopes(tree):
            # A module-level read runs at import, before any guard could
            # matter to it; it must go through the accessor.
            guarded = bool(_called_names(owned) & GUARD_CALLS) and not isinstance(
                scope, ast.Module
            )
            for node in owned:
                for variable, line in _lookups_in(node, constants):
                    found.append(Lookup(variable, path, line, guarded))
    return found


def test_every_storage_root_variable_is_registered_or_explained():
    registered = set(paths.RUNTIME_PATH_ENV_NAMES)
    unknown = sorted(
        f"{lookup.variable} at {lookup.where()}"
        for lookup in census()
        if lookup.variable not in registered and lookup.variable not in NOT_STORAGE_ROOTS
    )
    assert not unknown, (
        "These environment variables look like storage roots but are neither "
        "in paths.RUNTIME_PATH_ENV_NAMES (the guarded registry) nor explained "
        "in NOT_STORAGE_ROOTS:\n  " + "\n  ".join(unknown)
    )


def test_every_registered_variable_is_read_through_the_guard():
    registered = set(paths.RUNTIME_PATH_ENV_NAMES)
    unguarded = sorted(
        f"{lookup.variable} at {lookup.where()}"
        for lookup in census()
        if lookup.variable in registered and not lookup.guarded
    )
    assert not unguarded, (
        "These reads of a registered storage-root variable happen in a scope "
        "that never calls paths.guard_storage_path; read them through "
        "paths.runtime_path_env (or guard the resolved path):\n  "
        + "\n  ".join(unguarded)
    )


def test_the_registry_and_the_explanations_do_not_overlap():
    assert not set(paths.RUNTIME_PATH_ENV_NAMES) & set(NOT_STORAGE_ROOTS)


def test_the_explanations_are_not_stale():
    seen = {lookup.variable for lookup in census()}
    assert set(NOT_STORAGE_ROOTS) <= seen


def test_the_census_detects_every_read_form():
    # A census that found nothing would pass vacuously. Every read form, and
    # the constant indirection the package uses, must be seen, and a guard in
    # the reading scope is what makes it guarded.
    source = """
import os
from kestrel_sovereign import paths
SOME_DIR_ENV = "KESTREL_SOME_DIR"

def direct():
    return os.environ.get("KESTREL_DB_PATH")

def item(env):
    return env["KESTREL_HOME"]

def through_constant(runtime_env):
    return runtime_env.get(SOME_DIR_ENV)

def attribute():
    return os.getenv(paths.HOST_DB_PREVIOUS_DEFAULT_ENV)

def unprefixed():
    return os.getenv("LOCAL_PROBE_WORKING_DIR")

def guarded():
    value = os.environ.get("KESTREL_TRASH_DIR")
    return paths.guard_storage_path(value, source="probe")

def not_a_read():
    os.environ.pop("KESTREL_HOME", None)
    os.environ["KESTREL_HOME"] = "x"
"""
    path = PACKAGE_ROOT / "_census_probe.py"
    found = census({path: ast.parse(source)})
    assert {(lookup.variable, lookup.guarded) for lookup in found} == {
        ("KESTREL_DB_PATH", False),
        ("KESTREL_HOME", False),
        ("KESTREL_SOME_DIR", False),
        ("KESTREL_HOST_DB_LAUNCH_PREVIOUS_DEFAULT", False),
        ("LOCAL_PROBE_WORKING_DIR", False),
        ("KESTREL_TRASH_DIR", True),
    }


def test_the_harness_pins_exactly_the_registry():
    assert set(ISOLATED_LAYOUT) == set(paths.RUNTIME_PATH_ENV_NAMES)
