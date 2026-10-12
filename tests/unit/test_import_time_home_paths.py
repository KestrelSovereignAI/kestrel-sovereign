"""Paths derived from the home directory resolve when called, not at import (#3104).

Five module-scope constants used to build an absolute path from the home
directory at import: ``cli_serve``'s state directory, state file and log
directory, the compute trash root, and the MPS training directory (the last
converted earlier, in #3286). Whichever caller imported the module first
decided the path, and no later environment change could move it. Under pytest
that defeated the ``HOME`` redirect, because the modules are imported at
collection, before any fixture runs; the trash root was written that way. In
production the path was fixed by import order rather than by configuration.
Two more were found the same way: the discovered-context-limit cache, which
``tests/conftest.py`` had patched by name, and an unused
``constitutional_profile.UNIFIED_CONFIG_PATH`` alias, now removed.

Two halves:

* **Resolution.** This file imports every one of those modules at module scope,
  so the import happens at collection, before the fixtures move ``HOME`` (the
  ticket's method: a probe that imports inside a test body runs after the
  redirect). Each path is then resolved under two homes in turn.
* **Guard.** A scan of the package fails on any home lookup that runs at
  import, whether written out or reached through a call, so a new one fails
  here rather than being added to a fixture's patch list, the explicit
  non-fix in #3104: that list grows by enumeration, and the entry nobody adds
  names the operator's home again.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import pytest

from kestrel_sovereign import cli_serve, paths
from kestrel_sovereign.agent import token_counter
from kestrel_sovereign.features.compute import destructive_policy, trash_manager
from kestrel_sovereign.features.training.adapters import local_mps_adapter

PACKAGE_ROOT = Path(paths.__file__).resolve().parent

#: The registered variables that choose these paths. Unset, each path falls
#: back to its home-directory default, which is what this file pins down.
_SELECTING_VARIABLES = (
    paths.SERVE_STATE_DIR_ENV,
    paths.TRASH_DIR_ENV,
    paths.TRAINING_WORKING_DIR_ENV,
    paths.DATA_DIR_ENV,
    paths.CONTEXT_LIMITS_CACHE_FILE_ENV,
)

HOME_DERIVED_PATHS = [
    pytest.param(cli_serve.state_dir, ".kestrel", id="cli_serve.state_dir"),
    pytest.param(
        cli_serve.state_file, ".kestrel/serve_state.json", id="cli_serve.state_file"
    ),
    pytest.param(cli_serve.log_dir, ".kestrel/logs", id="cli_serve.log_dir"),
    pytest.param(
        destructive_policy.default_trash_dir,
        ".kestrel/trash",
        id="destructive_policy.default_trash_dir",
    ),
    pytest.param(
        lambda: destructive_policy.DestructiveOperationPolicy().trash_dir,
        ".kestrel/trash",
        id="DestructiveOperationPolicy().trash_dir",
    ),
    pytest.param(
        lambda: trash_manager.TrashManager().trash_dir,
        ".kestrel/trash",
        id="TrashManager().trash_dir",
    ),
    pytest.param(
        lambda: Path(local_mps_adapter.default_working_dir()),
        "kestrel-training",
        id="local_mps_adapter.default_working_dir",
    ),
    pytest.param(
        token_counter.context_limits_cache_file,
        ".kestrel/discovered_context_limits.json",
        id="token_counter.context_limits_cache_file",
    ),
]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resolves ~ from HOME")
@pytest.mark.parametrize(("resolve", "relative"), HOME_DERIVED_PATHS)
def test_each_path_follows_the_home_in_force_when_called(
    resolve, relative, tmp_path, monkeypatch
):
    for name in _SELECTING_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    # The modules were imported at collection, under the session's pins and
    # the operator's HOME, so a path frozen at import names neither home
    # below. A path frozen on its first resolution follows only the first.
    for home in (tmp_path / "first-home", tmp_path / "second-home"):
        monkeypatch.setenv("HOME", str(home))
        assert resolve() == home / relative


def test_cached_context_limits_are_read_from_the_cache_file_in_force(
    tmp_path, monkeypatch
):
    """The one-time read is per cache file, so a redirect is honoured.

    The suite used to reset the read between tests by patching the module;
    with the path resolved on call, a read of an earlier file must not answer
    for a later one.
    """
    monkeypatch.setattr(token_counter, "TIKTOKEN_AVAILABLE", False)
    monkeypatch.setattr(token_counter, "_discovered_context_limits", {})
    model = "context-limit-probe-3104"

    for name, limit in (("first", 1111), ("second", 2222)):
        cache = tmp_path / name / "discovered_context_limits.json"
        cache.parent.mkdir()
        cache.write_text(json.dumps({model: limit}), encoding="utf-8")
        monkeypatch.setenv(paths.CONTEXT_LIMITS_CACHE_FILE_ENV, str(cache))
        assert token_counter.TokenCounter(model).resolved_context_limit() == limit


# ---------------------------------------------------------------------------
# Guard: no home lookup runs at import
# ---------------------------------------------------------------------------

#: Environment variables that name the home directory.
_HOME_VARIABLES = frozenset({"HOME", "USERPROFILE"})

#: A reference to one of them inside a string ``expandvars`` will expand;
#: ``$KESTREL_HOME`` is a different variable.
_HOME_REFERENCE = re.compile(r"\$\{?(?:HOME|USERPROFILE)\b|%(?:HOME|USERPROFILE)%")

#: Functions whose own home lookup only *compares* a path their caller has
#: already chosen, never chooses one: the storage guard expands ``~`` before
#: checking a configured value against the test's roots. Counting that would
#: make every module-scope read of a registered variable look home-derived.
#: A home-resolving function they called would still count.
_PATH_CHECKS = frozenset(
    {
        ("kestrel_sovereign.paths", "_comparable"),
        ("kestrel_sovereign.paths", "guard_storage_path"),
        ("kestrel_sovereign.paths", "runtime_path_env"),
    }
)

#: An import binding: the source module and the imported attribute, or
#: ``None`` for ``import module``.
_Binding = tuple[str, Optional[str]]


@dataclass(frozen=True)
class _Module:
    name: str
    tree: ast.Module
    bindings: dict[str, _Binding]
    #: Top-level functions, and each top-level class's ``__init__`` and
    #: ``__post_init__`` (what instantiating it runs), by name.
    callables: dict[str, tuple[ast.AST, ...]]


def _bindings(tree: ast.Module, name: str, is_package: bool) -> dict[str, _Binding]:
    package = name if is_package else name.rpartition(".")[0]
    bound: dict[str, _Binding] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bound[alias.asname] = (alias.name, None)
                else:
                    top = alias.name.partition(".")[0]
                    bound[top] = (top, None)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                source = ".".join([*base, *([node.module] if node.module else [])])
            else:
                source = node.module or ""
            for alias in node.names:
                bound[alias.asname or alias.name] = (source, alias.name)
    return bound


def _callables(tree: ast.Module) -> dict[str, tuple[ast.AST, ...]]:
    found: dict[str, tuple[ast.AST, ...]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = (node,)
        elif isinstance(node, ast.ClassDef):
            found[node.name] = tuple(
                member
                for member in node.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and member.name in ("__init__", "__post_init__")
            )
    return found


def _module(name: str, source: str, *, is_package: bool = False) -> _Module:
    tree = ast.parse(source)
    return _Module(name, tree, _bindings(tree, name, is_package), _callables(tree))


@lru_cache(maxsize=1)
def _package_modules() -> dict[str, _Module]:
    modules = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        name = ".".join(parts[:-1] if is_package else parts)
        modules[name] = _module(
            name, path.read_text(encoding="utf-8"), is_package=is_package
        )
    return modules


def _dotted(node: ast.AST) -> Optional[list[str]]:
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    names.append(node.id)
    return names[::-1]


def _callee(module: _Module, func: ast.AST) -> Optional[tuple[str, str]]:
    """``(module, name)`` a call's target was imported from or defined as."""
    names = _dotted(func)
    if names is None:
        return None
    head, rest = names[0], names[1:]
    if not rest and head in module.callables:
        return module.name, head
    if head not in module.bindings:
        return None
    source, attribute = module.bindings[head]
    if not rest:
        return None if attribute is None else (source, attribute)
    base = source if attribute is None else f"{source}.{attribute}"
    return ".".join([base, *rest[:-1]]), rest[-1]


def _direct_home_lookup(module: _Module, node: ast.AST) -> Optional[str]:
    """The kind of home lookup ``node`` is in itself, or ``None``."""
    if isinstance(node, ast.Call):
        callee = _callee(module, node.func)
        name = callee[1] if callee else None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name == "home" and isinstance(node.func, ast.Attribute):
            return "Path.home()"
        if name == "expanduser":
            return "expanduser()"
        if name == "expandvars" and any(
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and _HOME_REFERENCE.search(arg.value)
            for arg in node.args
        ):
            return "expandvars() of HOME"
        if name in {"get", "getenv"} and node.args and _names_home(node.args[0]):
            return "HOME variable"
    elif isinstance(node, ast.Subscript) and _names_home(node.slice):
        return "HOME variable"
    elif isinstance(node, ast.Attribute) and node.attr == "pw_dir":
        return "passwd home"
    return None


def _names_home(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in _HOME_VARIABLES


def _home_resolving(modules: dict[str, _Module]) -> frozenset[tuple[str, str]]:
    """Every callable whose call consults the home directory, transitively.

    Calls are followed through imports to ``(module, name)``; a call this
    cannot resolve (a method, a dynamic lookup) is not followed.
    """
    direct: set[tuple[str, str]] = set()
    calls: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for module in modules.values():
        for name, definitions in module.callables.items():
            key = (module.name, name)
            calls[key] = set()
            for definition in definitions:
                for statement in definition.body:
                    for node in ast.walk(statement):
                        if _direct_home_lookup(module, node):
                            direct.add(key)
                        if isinstance(node, ast.Call):
                            callee = _callee(module, node.func)
                            if callee is not None:
                                calls[key].add(callee)
    resolving = direct - _PATH_CHECKS
    grew = True
    while grew:
        grew = False
        for key, callees in calls.items():
            if key not in resolving and callees & resolving:
                resolving.add(key)
                grew = True
    return frozenset(resolving)


def _is_main_guard(node: ast.AST) -> bool:
    """Exactly ``if __name__ == "__main__":``, whose body runs only as a script."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _import_time_nodes(tree: ast.Module) -> Iterator[ast.AST]:
    """Every node evaluated when the module is imported.

    Module and class bodies run at import. A function body waits for a call,
    but its decorators and default values are evaluated when its ``def`` runs;
    a lambda's body waits for a call too, its defaults do not. The ``else`` of
    a ``__main__`` guard runs at import; its body does not.
    """
    pending: list[ast.AST] = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            pending.extend(node.args.defaults)
            pending.extend(d for d in node.args.kw_defaults if d is not None)
            if not isinstance(node, ast.Lambda):
                pending.extend(node.decorator_list)
            continue
        if _is_main_guard(node):
            pending.extend(node.orelse)
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def import_time_home_lookups(
    modules: dict[str, _Module], names: Optional[list[str]] = None
) -> list[tuple[str, int, str]]:
    """``(module, line, kind)`` for every home lookup run at import.

    A direct lookup counts wherever it runs. A call to a home-resolving
    callable counts unless it is a statement of its own: one run for its
    effect, such as the server loading its ``.env`` at boot, stores no path.
    """
    resolving = _home_resolving(modules)
    found = []
    for name in names if names is not None else sorted(modules):
        module = modules[name]
        statements = {
            id(node.value)
            for node in ast.walk(module.tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        }
        for node in _import_time_nodes(module.tree):
            kind = _direct_home_lookup(module, node)
            if kind is None and isinstance(node, ast.Call) and id(node) not in statements:
                callee = _callee(module, node.func)
                if callee in resolving:
                    kind = f"call to {callee[0]}.{callee[1]}()"
            if kind is not None:
                found.append((name, node.lineno, kind))
    return sorted(found)


def test_no_package_module_looks_up_the_home_directory_at_import():
    found = [
        f"{module}:{line}: {kind}"
        for module, line, kind in import_time_home_lookups(_package_modules())
    ]
    assert not found, (
        "These consult the home directory when the module is imported, so the "
        "first importer decides the path and no later environment can move "
        "it. Resolve it on call instead, in a function, as "
        "paths.project_dir() does; do not patch it in a test fixture "
        "(#3104):\n  " + "\n  ".join(found)
    )


_PROBE_HELPERS = '''
import os
from pathlib import Path

def home_root():
    return Path.home() / ".kestrel"

def via_home_root():
    return home_root() / "trash"

def not_home():
    return Path("/srv/kestrel")

class Settings:
    def __init__(self):
        self.root = os.path.expanduser("~/.kestrel")

class Plain:
    def __init__(self):
        self.root = Path("/srv")

    def resolve(self):
        return Path.home()
'''

_PROBE_CONSUMER = '''
import os
import pwd
from os.path import expanduser as expand_user
from pathlib import Path
from kestrel_sovereign import paths
from kestrel_sovereign.paths import project_dir, runtime_path_env
from kestrel_probe import helpers
from kestrel_probe.helpers import home_root, not_home
from . import helpers as sibling

ROOT = Path.home() / ".kestrel"  # import-time
TYPED: str = os.path.expanduser("~/.kestrel")  # import-time
CHAINED = Path("~/.kestrel").expanduser()  # import-time
ALIASED = expand_user("~")  # import-time
ITEM = os.environ["HOME"]  # import-time
GETENV = os.getenv("USERPROFILE")  # import-time
EXPANDED = os.path.expandvars("$HOME/.kestrel")  # import-time
NOT_HOME_EXPANDED = os.path.expandvars("$KESTREL_HOME/x")
PASSWD = pwd.getpwuid(os.getuid()).pw_dir  # import-time
RESOLVER = paths.host_data_dir() / "x"  # import-time
BARE_RESOLVER = project_dir()  # import-time
DEFAULTED = runtime_path_env("KESTREL_TRASH_DIR", str(Path.home()))  # import-time
REGISTERED = runtime_path_env("KESTREL_TRASH_DIR")
HELPER = home_root()  # import-time
TRANSITIVE = helpers.via_home_root()  # import-time
RELATIVE = sibling.home_root()  # import-time
INSTANCE = helpers.Settings()  # import-time
NOT_HOME = not_home()
PLAIN = helpers.Plain()
NOT_HOME_VARIABLE = os.environ.get("KESTREL_HOME")
if os.name == "posix":
    NESTED = os.environ.get("HOME")  # import-time

class Holder:
    ATTRIBUTE = Path.home()  # import-time

    def method(self):
        return home_root()

def default_argument(root=Path.home()):  # import-time
    return Path.home()

def keyword_default(*, root=home_root()):  # import-time
    return os.environ["HOME"]

@decorate(Path.home())  # import-time
def decorated():
    return paths.project_dir()

deferred = lambda: Path.home()
with_default = lambda root=Path.home(): root  # import-time
os.makedirs(home_root(), exist_ok=True)  # import-time
paths.project_dir()

if __name__ != "__main__":
    NOT_MAIN = Path.home()  # import-time

if __name__ == "__main__":
    print(Path.home())
else:
    IMPORTED = Path.home()  # import-time
'''


def test_the_guard_detects_every_import_time_form():
    # A scan that found nothing would pass vacuously. Every line marked
    # "import-time" in the consumer must be found, and nothing else; the
    # helpers module does its lookups only when called.
    modules = {
        **_package_modules(),
        "kestrel_probe": _module("kestrel_probe", "", is_package=True),
        "kestrel_probe.helpers": _module("kestrel_probe.helpers", _PROBE_HELPERS),
        "kestrel_probe.consumer": _module("kestrel_probe.consumer", _PROBE_CONSUMER),
    }
    expected = {
        ("kestrel_probe.consumer", number)
        for number, line in enumerate(_PROBE_CONSUMER.splitlines(), start=1)
        if line.endswith("# import-time")
    }
    found = {
        (module, line)
        for module, line, _kind in import_time_home_lookups(
            modules, ["kestrel_probe.helpers", "kestrel_probe.consumer"]
        )
    }
    assert found == expected
