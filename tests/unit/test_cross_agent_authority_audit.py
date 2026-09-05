"""Contract tests for the checked-in cross-agent authority inventory (#3143)."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_sovereign.auth import AuthMethod, CallerContext
from kestrel_sovereign.command_handler import BUILTIN_COMMAND_SPECS
from kestrel_sovereign.endpoints.models import require_sovereign_host_lifecycle

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "docs/architecture/CROSS_AGENT_AUTHORITY_AUDIT.md"
AUTH_SURFACE_MATRIX_PATH = REPO_ROOT / "docs/audit/AUTH_SURFACE_MATRIX.md"
CONTROL_NAME_TERMS = (
    "agent",
    "peer",
    "a2a",
    "parent",
    "child",
    "descendant",
    "delegate",
    "control",
    "task",
    "cancel",
    "interrupt",
    "stop",
    "hold",
    "terminate",
    "offboard",
    "withdraw",
    "restart",
    "host",
    "fleet",
    # Agent and process adapters often expose lifecycle verbs without an
    # agent-shaped qualifier.  These remain control sinks when provenance is
    # used to decide whether they run (for example ``target.shutdown()`` or
    # ``ProcessManager.kill_process(...)``).
    "shutdown",
    "kill_process",
)
PERMISSION_NAME_TERMS = (
    "authoriz",
    "authority",
    "mandate",
    "owner",
    "permission",
    "allowed",
    "permitted",
    "forbid",
    "denied",
    "access",
    "gate",
    "require",
)
PROVENANCE_TRANSFORM_CALLS = {
    "all",
    "any",
    "bool",
    "copy",
    "deepcopy",
    "dict",
    "enumerate",
    "filter",
    "frozenset",
    "get",
    "getattr",
    "len",
    "list",
    "map",
    "max",
    "min",
    "reversed",
    "set",
    "sorted",
    "sum",
    "tuple",
}
PROVENANCE_ACCESSOR_SUFFIXES = (
    "causation_chain",
    "get_current_chain",
)
HTTP_SEGMENTS = {
    # Every request-routed agent endpoint is addressable through the host's
    # /api/agents/{name}/... alias in multi-agent mode.  Inventory the complete
    # singular namespace rather than guessing which suffixes invoke or control.
    "agent",
    "agents",
    "tasks",
    "stop",
    "restart",
    "a2a",
    "peers",
    "children",
    "webhooks",
    # Feature package lifecycle mutates the host's shared interpreter, while
    # observability can read a shared PostgreSQL event table.  Neither class
    # needs an agent-shaped word in the remainder of its route to cross an
    # authority boundary, so inventory the complete namespaces and classify
    # their self-only/read-only false positives explicitly in the audit.
    "features",
    "observability",
    # Bridge endpoints can invoke the routed agent without an agent-shaped
    # suffix, while app-level /api/host routes issue host-wide UI credentials
    # or describe shared host state.
    "bridge",
    "host",
    "auth",
    # Process-wide credentials/configuration rather than an agent principal.
    "github",
}
HTTP_EXACT_ROUTES = {
    "/docs",
    "/docs/oauth2-redirect",
    "/health",
    "/health/detailed",
    "/metrics",
    "/openapi.json",
    "/phoenix",
    "/phoenix/{path:path}",
    "/redoc",
    "/api/agent/invoke",
    "/api/agent/stream",
    "/api/auth/key",
    "/api/keys/platform",
    "/api/keys/user",
    "/api/keys/user/verify",
    "/api/keys/user/{provider}",
    # These reads cross the selected-agent boundary: IPFS and model discovery
    # use process-wide state, while available-sources combines agent, user, and
    # platform principals.
    "/api/ipfs/status",
    "/api/keys/available-sources",
    "/api/models",
    # These handlers read the process-global sovereignty export cache rather
    # than state owned by the selected agent. Keep the canonical doors in the
    # exact inventory alongside their synthesized request-routed aliases.
    "/api/sovereignty/files",
    "/api/sovereignty/files/{filename}",
    "/api/sovereignty/files/{filename}/preview",
    # Canonical auth-exempt redirects into the host-wide Phoenix asset tree.
    "/assets/{path:path}",
    "/v1/chat/completions",
}
INDIRECT_DISPATCH_CALLS = {
    # Generic tool dispatchers need inventory even when their public name has
    # no agent-shaped word.  These are execution boundaries, not authority:
    # the selected downstream tool must still enforce its own policy.
    "execute_skill",
    "execute_named_tool",
    "_create_schedule",
}
SURFACE_ID = re.compile(
    r"\|\s*`(kestrel_sovereign/"
    r"(?:(?:server|kestrel_agent)\.py::[^`]+|"
    r"(?:agent|features|host_features|endpoints|signals)/[^`]+))`\s*\|"
)
COMMAND_SURFACE_ID = re.compile(
    r"\|\s*`(kestrel_sovereign/command_handler\.py::![^`]+)`\s*\|"
)
CLI_SURFACE_ID = re.compile(
    r"\|\s*`(kestrel_sovereign/cli\.py::kestrel [^`]+)`\s*\|"
)


@lru_cache(maxsize=None)
def _source_text(source_path: Path) -> str:
    """Read checked-in source once for all inventory contracts."""

    return source_path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _parsed_module(source_path: Path) -> ast.Module:
    """Parse a checked-in module once for all inventory contracts."""

    return ast.parse(
        _source_text(source_path),
        filename=str(source_path),
    )


def _resolved_string(
    node: ast.expr,
    constants: dict[str, str] | None = None,
) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return (constants or {}).get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _resolved_string(node.left, constants)
        right = _resolved_string(node.right, constants)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
                continue
            if isinstance(value, ast.FormattedValue):
                resolved = _resolved_string(value.value, constants)
                if resolved is not None:
                    parts.append(resolved)
                    continue
            return None
        return "".join(parts)
    return None


def _imported_module_path(node: ast.ImportFrom, source_path: Path) -> Path | None:
    """Resolve a repository-local ``from ... import`` without importing code."""

    if node.level:
        base = source_path.parent
        for _ in range(node.level - 1):
            base = base.parent
    else:
        base = REPO_ROOT
    candidate = base.joinpath(*(node.module or "").split("."))
    module_file = candidate.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = candidate / "__init__.py"
    return package_file if package_file.is_file() else None


_MODULE_COMPOUND_STATEMENT_TYPES = (
    ast.AsyncFor,
    ast.AsyncWith,
    ast.For,
    ast.If,
    ast.Match,
    ast.Try,
    ast.TryStar,
    ast.While,
    ast.With,
)


def _compound_binding_names(
    statement: ast.stmt,
    collection_names: set[str],
) -> set[str]:
    """Return bindings a module-level compound statement can disturb.

    The evaluator cannot choose a runtime branch. Any assignment under such
    control flow therefore makes a previously resolved scalar ambiguous.
    Mutable collections fail closed on any reference too: a branch may pass or
    alias the live object before mutating it elsewhere. Nested function and
    class bodies are different runtime scopes and are deliberately not
    borrowed into the module-execution replay.
    """

    names: set[str] = set()

    class BindingVisitor(ast.NodeVisitor):
        def visit_Name(self, child: ast.Name) -> None:  # noqa: N802
            if isinstance(child.ctx, (ast.Store, ast.Del)) or (
                isinstance(child.ctx, ast.Load) and child.id in collection_names
            ):
                names.add(child.id)

        def visit_FunctionDef(  # noqa: N802
            self, child: ast.FunctionDef
        ) -> None:
            names.add(child.name)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, child: ast.AsyncFunctionDef
        ) -> None:
            names.add(child.name)

        def visit_ClassDef(self, child: ast.ClassDef) -> None:  # noqa: N802
            names.add(child.name)

        def visit_Lambda(self, child: ast.Lambda) -> None:  # noqa: N802
            return

        def visit_Import(self, child: ast.Import) -> None:  # noqa: N802
            names.update(
                imported.asname or imported.name.split(".", 1)[0]
                for imported in child.names
            )

        def visit_ImportFrom(self, child: ast.ImportFrom) -> None:  # noqa: N802
            names.update(
                imported.asname or imported.name
                for imported in child.names
                if imported.name != "*"
            )

        def visit_ExceptHandler(  # noqa: N802
            self, child: ast.ExceptHandler
        ) -> None:
            if child.name:
                names.add(child.name)
            self.generic_visit(child)

        def visit_MatchAs(self, child: ast.MatchAs) -> None:  # noqa: N802
            if child.name:
                names.add(child.name)
            self.generic_visit(child)

        def visit_MatchStar(self, child: ast.MatchStar) -> None:  # noqa: N802
            if child.name:
                names.add(child.name)

        def visit_MatchMapping(  # noqa: N802
            self, child: ast.MatchMapping
        ) -> None:
            if child.rest:
                names.add(child.rest)
            self.generic_visit(child)

    BindingVisitor().visit(statement)
    return names


def _module_constant_bindings(
    tree: ast.Module,
    source_path: Path | None = None,
    initial_strings: dict[str, str] | None = None,
    initial_collections: dict[str, tuple[str, ...]] | None = None,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Replay scalar-string and string-collection bindings together.

    Collections capture their element values when their assignment executes.
    Resolving them against a separately computed final scalar map would rewrite
    that history when an element name is rebound later.
    """

    strings: dict[str, str] = dict(initial_strings or {})
    collections: dict[str, tuple[str, ...]] = dict(initial_collections or {})
    collection_alias_groups: dict[str, set[str]] = {}

    def detach_collection_alias(name: str) -> None:
        group = collection_alias_groups.pop(name, None)
        if group is not None:
            group.discard(name)

    def invalidate_collection(name: str) -> None:
        group = set(collection_alias_groups.get(name, {name}))
        for alias in group:
            collections.pop(alias, None)
            collection_alias_groups.pop(alias, None)

    def stored_target_names(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.List, ast.Tuple)):
            return {
                name
                for element in target.elts
                for name in stored_target_names(element)
            }
        if isinstance(target, ast.Starred):
            return stored_target_names(target.value)
        return set()

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                strings.clear()
                collections.clear()
                collection_alias_groups.clear()
                continue
            for alias in node.names:
                bound_name = alias.asname or alias.name
                strings.pop(bound_name, None)
                invalidate_collection(bound_name)
            if source_path is not None:
                imported_path = _imported_module_path(node, source_path)
                if imported_path is not None:
                    imported_constants = _cached_local_string_constants(
                        imported_path
                    )
                    for alias in node.names:
                        if alias.name in imported_constants:
                            strings[alias.asname or alias.name] = (
                                imported_constants[alias.name]
                            )
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".", 1)[0]
                strings.pop(bound_name, None)
                invalidate_collection(bound_name)
            continue
        if isinstance(node, _MODULE_COMPOUND_STATEMENT_TYPES):
            for name in _compound_binding_names(node, set(collections)):
                strings.pop(name, None)
                invalidate_collection(name)
            continue
        # A statically resolved binding must never survive an operation whose
        # result this small evaluator does not model. Retaining the old value
        # would make an exact route inventory silently describe a different
        # decorator than Python executes.
        mutated_names: set[str] = set()
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            mutated_names.add(node.target.id)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            function = call.func
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.attr
                in {
                    "add",
                    "append",
                    "clear",
                    "extend",
                    "insert",
                    "pop",
                    "remove",
                    "reverse",
                    "sort",
                    "update",
                }
            ):
                mutated_names.add(function.value.id)
            else:
                # Passing a mutable methods collection across an unmodelled
                # call boundary can change it just as surely as a direct
                # ``METHODS.append(...)``.  The exact route inventory cannot
                # prove that an arbitrary helper is pure, so discard every
                # live collection binding reachable through its arguments and
                # make a later decorator fail closed.
                mutated_names.update(
                    child.id
                    for argument in [
                        *call.args,
                        *(keyword.value for keyword in call.keywords),
                    ]
                    for child in ast.walk(argument)
                    if isinstance(child, ast.Name) and child.id in collections
                )
                if (
                    isinstance(function, ast.Attribute)
                    and isinstance(function.value, ast.Name)
                    and function.value.id in collections
                ):
                    mutated_names.add(function.value.id)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    mutated_names.add(target.id)
                elif isinstance(target, ast.Subscript) and isinstance(
                    target.value, ast.Name
                ):
                    mutated_names.add(target.value.id)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            assignment_targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            for target in assignment_targets:
                if isinstance(target, ast.Subscript) and isinstance(
                    target.value, ast.Name
                ):
                    mutated_names.add(target.value.id)
            if isinstance(node.value, ast.Call):
                mutated_names.update(
                    child.id
                    for argument in [
                        *node.value.args,
                        *(keyword.value for keyword in node.value.keywords),
                    ]
                    for child in ast.walk(argument)
                    if isinstance(child, ast.Name) and child.id in collections
                )
                if (
                    isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id in collections
                ):
                    mutated_names.add(node.value.func.value.id)
                # The result of an unmodelled call cannot preserve an older
                # binding on the assignment target either.
                mutated_names.update(
                    target_name
                    for target in assignment_targets
                    for target_name in stored_target_names(target)
                )
        if mutated_names:
            for name in mutated_names:
                strings.pop(name, None)
                invalidate_collection(name)
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
            value: ast.expr | None = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        resolved_string = (
            _resolved_string(value, strings) if value is not None else None
        )
        resolved_collection: tuple[str, ...] | None = None
        collection_source: str | None = None
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            elements = tuple(
                _resolved_string(element, strings) for element in value.elts
            )
            if all(element is not None for element in elements):
                resolved_collection = tuple(
                    element for element in elements if element is not None
                )
        elif isinstance(value, ast.Name) and value.id in collections:
            resolved_collection = collections[value.id]
            collection_source = value.id
        target_names = [
            target_name
            for target in targets
            for target_name in stored_target_names(target)
        ]
        for target_name in target_names:
            if target_name != collection_source:
                detach_collection_alias(target_name)
            if resolved_string is None:
                strings.pop(target_name, None)
            else:
                strings[target_name] = resolved_string
            if resolved_collection is None:
                collections.pop(target_name, None)
            else:
                collections[target_name] = resolved_collection
        if resolved_collection is not None and target_names:
            if collection_source is not None:
                group = collection_alias_groups.get(collection_source)
                if group is None:
                    group = {collection_source}
                    collection_alias_groups[collection_source] = group
                group.update(target_names)
            else:
                group = set(target_names)
            for target_name in target_names:
                collection_alias_groups[target_name] = group
    return strings, collections


def _module_string_constants(
    tree: ast.Module,
    source_path: Path | None = None,
) -> dict[str, str]:
    """Resolve static strings in module execution order, including imports."""

    return _module_constant_bindings(tree, source_path)[0]


def _module_strings_at_definition(
    tree: ast.Module,
    definition: ast.FunctionDef | ast.AsyncFunctionDef,
    source_path: Path | None = None,
) -> dict[str, str]:
    """Resolve globals as they existed when a decorator executed.

    Top-level functions and methods of a top-level class are decorated while
    that containing module statement executes. A nested function instead runs
    only when its outer function is called, after module initialization, so it
    retains the final module bindings used by the prior scanner.
    """

    for index, statement in enumerate(tree.body):
        executes_during_statement = statement is definition or (
            isinstance(statement, ast.ClassDef)
            and definition in statement.body
        )
        if executes_during_statement:
            preceding = ast.Module(
                body=tree.body[:index],
                type_ignores=[],
            )
            return _module_string_constants(preceding, source_path)
    return _module_string_constants(tree, source_path)


@lru_cache(maxsize=None)
def _cached_local_string_constants(source_path: Path) -> dict[str, str]:
    """Read one imported module's local constants without crawling its imports."""

    return _module_string_constants(_parsed_module(source_path))


def _imported_module_attribute_name(
    tree: ast.Module,
    value: ast.AST,
) -> str | None:
    """Return an attribute selected through a statically imported module."""

    if not isinstance(value, ast.Attribute):
        return None
    module_aliases = {
        (imported.asname or imported.name.split(".", 1)[0])
        for node in tree.body
        if isinstance(node, ast.Import)
        for imported in node.names
    }
    expression = ast.unparse(value)
    if any(
        expression == alias or expression.startswith(f"{alias}.")
        for alias in module_aliases
    ):
        return value.attr
    return None


def _static_assignment_pairs(
    target: ast.AST,
    value: ast.AST,
) -> list[tuple[str, ast.AST]]:
    """Pair statically aligned assignment targets and values.

    Python permits aliases to be introduced by annotated, chained, and
    destructuring assignments. Exact tuple/list shapes can be replayed without
    execution; starred or mismatched shapes remain deliberately unresolved.
    """

    if isinstance(target, ast.Name):
        return [(target.id, value)]
    if (
        isinstance(target, (ast.List, ast.Tuple))
        and isinstance(value, (ast.List, ast.Tuple))
        and len(target.elts) == len(value.elts)
        and not any(isinstance(element, ast.Starred) for element in target.elts)
    ):
        return [
            pair
            for target_element, value_element in zip(target.elts, value.elts)
            for pair in _static_assignment_pairs(target_element, value_element)
        ]
    return []


def _assignment_target_names(target: ast.AST) -> set[str]:
    """Return case-preserving names bound anywhere in one target."""

    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _assignment_target_names(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        return {
            name
            for element in target.elts
            for name in _assignment_target_names(element)
        }
    return set()


def _tool_decorator_aliases(tree: ast.Module) -> set[str]:
    """Resolve import and lexical-scope aliases for the ``tool`` decorator."""

    aliases = {"tool"}
    assignments: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for imported in node.names:
                if imported.name == "tool":
                    aliases.add(imported.asname or imported.name)
        else:
            assignment_pairs: list[tuple[str, ast.AST]] = []
            if isinstance(node, ast.Assign):
                assignment_pairs = [
                    pair
                    for target in node.targets
                    for pair in _static_assignment_pairs(target, node.value)
                ]
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                assignment_pairs = _static_assignment_pairs(
                    node.target,
                    node.value,
                )
            for target, value in assignment_pairs:
                source = (
                    value.id
                    if isinstance(value, ast.Name)
                    else _imported_module_attribute_name(tree, value)
                )
                if source is not None:
                    assignments.append((target, source))
    changed = True
    while changed:
        changed = False
        for target, source in assignments:
            if source in aliases and target not in aliases:
                aliases.add(target)
                changed = True
    return aliases


def test_module_string_constants_follow_source_order_on_reassignment() -> None:
    reassigned = ast.parse(
        'BASE = "/first"\n'
        'PATH = BASE + "/route"\n'
        'BASE = "/second"\n'
        'PATH = BASE + "/route"\n'
    )
    assert _module_string_constants(reassigned) == {
        "BASE": "/second",
        "PATH": "/second/route",
    }

    invalidated = ast.parse(
        'PATH = "/known"\n'
        "PATH = runtime_path()\n"
    )
    assert "PATH" not in _module_string_constants(invalidated)

    mutated = ast.parse(
        'PATH = "/api"\n'
        'PATH += "/agents/{agent}/terminate"\n'
        'METHODS = ["GET"]\n'
        'METHODS.append("DELETE")\n'
    )
    assert "PATH" not in _module_string_constants(mutated)
    assert "METHODS" not in _module_string_collections(mutated)

    mutated_alias = ast.parse(
        'METHODS = ["GET"]\n'
        "ALIAS = METHODS\n"
        'ALIAS.append("DELETE")\n'
    )
    aliased_collections = _module_string_collections(mutated_alias)
    assert "METHODS" not in aliased_collections
    assert "ALIAS" not in aliased_collections

    aliased_route = ast.parse(
        'METHODS = ["GET"]\n'
        "ALIAS = METHODS\n"
        'ALIAS.append("DELETE")\n'
        '@app.api_route("/api/agents/{name}", methods=METHODS)\n'
        "def route():\n    pass\n"
    )
    with pytest.raises(AssertionError, match="Unresolved api_route methods"):
        _route_declarations(
            aliased_route,
            _module_string_constants(aliased_route),
            _module_string_collections(aliased_route),
        )

    helper_mutated = ast.parse(
        'METHODS = ["GET"]\n'
        'mutate(METHODS)\n'
        '@app.api_route("/api/agents/{name}", methods=METHODS)\n'
        'def route():\n    pass\n'
    )
    assert "METHODS" not in _module_string_collections(helper_mutated)
    with pytest.raises(AssertionError, match="Unresolved api_route methods"):
        _route_declarations(
            helper_mutated,
            _module_string_constants(helper_mutated),
            _module_string_collections(helper_mutated),
        )

    assigned_helper_mutation = ast.parse(
        'METHODS = ["GET"]\n'
        'result = mutate(METHODS)\n'
        '@app.api_route("/api/agents/{name}", methods=METHODS)\n'
        'def route():\n    pass\n'
    )
    assert "METHODS" not in _module_string_collections(
        assigned_helper_mutation
    )

    unknown_method_mutation = ast.parse(
        'METHODS = ["GET"]\n'
        'METHODS.rewrite_in_place()\n'
        '@app.api_route("/api/agents/{name}", methods=METHODS)\n'
        'def route():\n    pass\n'
    )
    assert "METHODS" not in _module_string_collections(
        unknown_method_mutation
    )


@pytest.mark.parametrize(
    "compound",
    [
        'if enabled:\n    PATH = "/api/agents/{name}/terminate"\n',
        'try:\n    PATH = "/api/agents/{name}/terminate"\n'
        'except RuntimeError:\n    pass\n',
        'with context():\n    PATH = "/api/agents/{name}/terminate"\n',
        'for item in items:\n    PATH = "/api/agents/{name}/terminate"\n',
        'while enabled:\n    PATH = "/api/agents/{name}/terminate"\n',
    ],
    ids=["if", "try", "with", "for", "while"],
)
def test_module_constants_fail_closed_across_compound_control_flow(
    compound: str,
) -> None:
    tree = ast.parse(
        'PATH = "/health"\n'
        + compound
        + '@app.get(PATH)\n'
        'def route():\n    pass\n'
    )

    assert "PATH" not in _module_string_constants(tree)
    with pytest.raises(AssertionError, match="Unresolved route path"):
        _route_declarations(
            tree,
            _module_string_constants(tree),
            _module_string_collections(tree),
        )


@pytest.mark.parametrize(
    "rebind",
    [
        'PATH, ignored = ("/api/agents/{name}/terminate", None)\n',
        "PATH, ignored = runtime_paths()\n",
        "from extension import PATH\n",
        "import extension as PATH\n",
    ],
    ids=["tuple", "runtime-tuple", "from-import", "import"],
)
def test_module_constants_fail_closed_across_non_simple_rebinding(
    rebind: str,
) -> None:
    tree = ast.parse(
        'PATH = "/health"\n'
        + rebind
        + "@app.get(PATH)\n"
        "def route():\n    pass\n"
    )

    assert "PATH" not in _module_string_constants(tree)
    with pytest.raises(AssertionError, match="Unresolved route path"):
        _route_declarations(
            tree,
            _module_string_constants(tree),
            _module_string_collections(tree),
        )


def test_module_method_collections_fail_closed_under_compound_mutation() -> None:
    tree = ast.parse(
        'METHODS = ["GET"]\n'
        'if enabled:\n'
        '    METHODS[:] = ["DELETE"]\n'
        '@app.api_route("/api/agents/{name}", methods=METHODS)\n'
        'def route():\n    pass\n'
    )

    assert "METHODS" not in _module_string_collections(tree)
    with pytest.raises(AssertionError, match="Unresolved api_route methods"):
        _route_declarations(
            tree,
            _module_string_constants(tree),
            _module_string_collections(tree),
        )

    nested_route = ast.parse(
        'PATH = "/health"\n'
        'router = APIRouter()\n'
        'if enabled:\n'
        '    PATH = "/api/agents/{name}/terminate"\n'
        '    @router.delete(PATH)\n'
        '    def terminate():\n'
        '        pass\n'
    )
    with pytest.raises(AssertionError, match="Unresolved route path"):
        _route_declarations(
            nested_route,
            _module_string_constants(nested_route),
            _module_string_collections(nested_route),
        )


def _public_tool_name(
    decorator: ast.expr,
    fallback: str,
    constants: dict[str, str] | None = None,
    decorator_aliases: set[str] | None = None,
) -> str | None:
    call = decorator if isinstance(decorator, ast.Call) else None
    function = call.func if call is not None else decorator
    decorator_name = (
        function.id
        if isinstance(function, ast.Name)
        else function.attr
        if isinstance(function, ast.Attribute)
        else ""
    )
    if decorator_name not in (decorator_aliases or {"tool"}):
        return None
    if call is None:
        return fallback
    if any(keyword.arg is None for keyword in call.keywords):
        raise AssertionError(
            "Unresolved @tool keyword unpacking: "
            f"{ast.unparse(call)}"
        )
    if call.args:
        name = _resolved_string(call.args[0], constants)
        if name is None:
            raise AssertionError(
                f"Unresolved @tool name expression: {ast.unparse(call.args[0])}"
            )
        return name
    for keyword in call.keywords:
        if keyword.arg == "name":
            name = _resolved_string(keyword.value, constants)
            if name is None:
                raise AssertionError(
                    "Unresolved @tool name expression: "
                    f"{ast.unparse(keyword.value)}"
                )
            return name
    return fallback


def _is_cross_agent_control_name(name: str) -> bool:
    return any(term in name.casefold() for term in CONTROL_NAME_TERMS)


def _is_permission_name(name: str) -> bool:
    return any(term in name.casefold() for term in PERMISSION_NAME_TERMS)


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _is_indirect_tool_dispatcher(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Detect public tools that can select another tool at runtime.

    ``run_workflow`` delegates directly through ``TaskManager.execute_skill``;
    ``signal_dispatch`` selects the contributed workflow runner through
    ``execute_named_tool``. Scheduler creation tools delegate indirectly
    through ``_create_schedule``; the persisted name is later resolved to any
    loaded feature tool. Detect the wiring rather than freezing only today's
    public names, so renaming one of these entry doors cannot make it disappear
    from the authority audit.
    """

    return any(
        isinstance(node, ast.Call) and _call_name(node) in INDIRECT_DISPATCH_CALLS
        for node in ast.walk(function)
    )


@lru_cache(maxsize=None)
def _discovered_tool_surfaces() -> frozenset[str]:
    """Return every core feature tool, including generated dispatch boundaries.

    Cross-agent capability is a property of implementation and deployment,
    not a public-name convention.  Exact inventory of the complete registered
    tool set forces new tools to be classified even when their names do not
    advertise shared-host reach.
    """

    surfaces: set[str] = set()
    feature_root = REPO_ROOT / "kestrel_sovereign/features"
    for path in feature_root.rglob("*.py"):
        tree = _parsed_module(path)
        decorator_aliases = _tool_decorator_aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            string_constants = _module_strings_at_definition(tree, node, path)
            for decorator in node.decorator_list:
                public_name = _public_tool_name(
                    decorator,
                    node.name,
                    string_constants,
                    decorator_aliases,
                )
                if public_name is None:
                    continue
                relative = path.relative_to(REPO_ROOT).as_posix()
                surfaces.add(f"{relative}::{public_name}")
    return frozenset(surfaces | _discovered_runtime_generated_tool_surfaces())


def _cron_task_entry_name(
    item: ast.AST,
    constants: dict[str, str],
) -> str:
    if not isinstance(item, (ast.List, ast.Tuple)) or not item.elts:
        raise AssertionError("CRON_TASKS contains an unsupported entry")
    name = _resolved_string(item.elts[0], constants)
    if name is None:
        raise AssertionError(
            "Unresolved CRON_TASKS name expression: "
            f"{ast.unparse(item.elts[0])}"
        )
    return name


def _cron_task_collection_names(
    value: ast.AST,
    constants: dict[str, str],
    current: set[str] | None,
) -> set[str]:
    if isinstance(value, ast.Name) and value.id == "CRON_TASKS":
        if current is None:
            raise AssertionError("CRON_TASKS referenced before its declaration")
        return set(current)
    if not isinstance(value, (ast.List, ast.Tuple)):
        raise AssertionError(
            "CRON_TASKS must remain a statically resolvable list or tuple: "
            f"{ast.unparse(value)}"
        )

    names: set[str] = set()
    for item in value.elts:
        if isinstance(item, ast.Starred):
            if not (
                isinstance(item.value, ast.Name)
                and item.value.id == "CRON_TASKS"
                and current is not None
            ):
                raise AssertionError(
                    "Unresolved CRON_TASKS collection unpacking: "
                    f"{ast.unparse(item)}"
                )
            names.update(current)
            continue
        names.add(_cron_task_entry_name(item, constants))
    return names


def _cron_task_names(
    tree: ast.Module,
    source_path: Path | None = None,
) -> set[str]:
    """Replay statically knowable module-level ``CRON_TASKS`` mutations."""

    task_names: set[str] | None = None
    for index, node in enumerate(tree.body):
        prefix = ast.Module(body=tree.body[:index], type_ignores=[])
        constants = _module_string_constants(prefix, source_path)

        assignment_target: ast.AST | None = None
        assignment_value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            bound_names = {
                name
                for target in node.targets
                for name in _assignment_target_names(target)
            }
            if "CRON_TASKS" in bound_names:
                if not (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "CRON_TASKS"
                ):
                    raise AssertionError(
                        "Unsupported destructuring assignment to CRON_TASKS"
                    )
                assignment_target = node.targets[0]
                assignment_value = node.value
        elif isinstance(node, ast.AnnAssign):
            if "CRON_TASKS" in _assignment_target_names(node.target):
                if not (
                    isinstance(node.target, ast.Name)
                    and node.target.id == "CRON_TASKS"
                    and node.value is not None
                ):
                    raise AssertionError("Unsupported annotated CRON_TASKS assignment")
                assignment_target = node.target
                assignment_value = node.value

        if assignment_target is not None and assignment_value is not None:
            task_names = _cron_task_collection_names(
                assignment_value,
                constants,
                task_names,
            )
            continue

        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "CRON_TASKS"
        ):
            if task_names is None or not isinstance(node.op, ast.Add):
                raise AssertionError("Unsupported augmented CRON_TASKS mutation")
            task_names.update(
                _cron_task_collection_names(node.value, constants, task_names)
            )
            continue

        direct_call = node.value if isinstance(node, ast.Expr) else None
        if (
            isinstance(direct_call, ast.Call)
            and isinstance(direct_call.func, ast.Attribute)
            and isinstance(direct_call.func.value, ast.Name)
            and direct_call.func.value.id == "CRON_TASKS"
        ):
            if task_names is None or direct_call.keywords:
                raise AssertionError("Unsupported CRON_TASKS method mutation")
            if direct_call.func.attr == "append" and len(direct_call.args) == 1:
                task_names.add(
                    _cron_task_entry_name(direct_call.args[0], constants)
                )
                continue
            if direct_call.func.attr == "extend" and len(direct_call.args) == 1:
                task_names.update(
                    _cron_task_collection_names(
                        direct_call.args[0],
                        constants,
                        task_names,
                    )
                )
                continue
            raise AssertionError(
                "Unsupported CRON_TASKS method mutation: "
                f"{ast.unparse(direct_call)}"
            )

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if any(
            isinstance(child, ast.Name)
            and child.id == "CRON_TASKS"
            and isinstance(child.ctx, (ast.Store, ast.Del))
            for child in ast.walk(node)
        ) or any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "CRON_TASKS"
            for child in ast.walk(node)
        ):
            raise AssertionError(
                "Conditional or indirect CRON_TASKS mutation is not statically "
                f"resolvable: {ast.unparse(node)}"
            )

    if task_names is None:
        raise AssertionError("Could not find the CRON_TASKS declaration")
    return task_names


@lru_cache(maxsize=None)
def _discovered_scheduler_surfaces() -> frozenset[str]:
    """Inventory every cron target and every bespoke handler wired to it."""

    source_path = REPO_ROOT / "kestrel_sovereign/signals/sources/scheduler.py"
    source_tree = ast.parse(
        source_path.read_text(encoding="utf-8"), filename=str(source_path)
    )
    source_constants = _module_string_constants(source_tree, source_path)
    task_names = _cron_task_names(source_tree, source_path)

    scheduler_source_names = _scheduler_registration_source_names(
        source_tree,
        source_constants,
        task_names,
    )

    feature_path = REPO_ROOT / "kestrel_sovereign/features/scheduler/feature.py"
    feature_tree = ast.parse(
        feature_path.read_text(encoding="utf-8"), filename=str(feature_path)
    )
    feature_constants = _module_string_constants(feature_tree, feature_path)
    builtin_handlers: dict[str, str] | None = None
    for node in ast.walk(feature_tree):
        if not (
            isinstance(node, ast.Call)
            and _call_name(node) == "build_cron_registrations"
        ):
            continue
        keyword = next(
            (item for item in node.keywords if item.arg == "builtin_handlers"),
            None,
        )
        if keyword is None or not isinstance(keyword.value, ast.Dict):
            raise AssertionError(
                "build_cron_registrations must expose a literal builtin_handlers map"
            )
        handlers: dict[str, str] = {}
        for key, value in zip(keyword.value.keys, keyword.value.values):
            if key is None:
                raise AssertionError("builtin_handlers contains dictionary unpacking")
            task_name = _resolved_string(key, feature_constants)
            handler_name = (
                value.attr
                if isinstance(value, ast.Attribute)
                else value.id
                if isinstance(value, ast.Name)
                else None
            )
            if task_name is None or handler_name is None:
                raise AssertionError(
                    "Unresolved builtin scheduler handler entry: "
                    f"{ast.unparse(key)}: {ast.unparse(value)}"
                )
            handlers[task_name] = handler_name
        builtin_handlers = handlers
        break
    if builtin_handlers is None:
        raise AssertionError("Could not find the builtin scheduler handler map")
    unknown_handlers = set(builtin_handlers) - task_names
    if unknown_handlers:
        raise AssertionError(
            "Scheduler handlers lack CRON_TASKS entries: "
            + ", ".join(sorted(unknown_handlers))
        )
    function_names = {
        node.name
        for node in ast.walk(feature_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing_functions = set(builtin_handlers.values()) - function_names
    if missing_functions:
        raise AssertionError(
            "Missing builtin scheduler handler functions: "
            + ", ".join(sorted(missing_functions))
        )

    return frozenset({
        *(
            f"kestrel_sovereign/signals/sources/scheduler.py::{name}"
            for name in scheduler_source_names
        ),
        *(
            f"kestrel_sovereign/features/scheduler/feature.py::{name}"
            for name in builtin_handlers.values()
        ),
    })


def _scheduler_registration_source_names(
    tree: ast.Module,
    constants: dict[str, str],
    task_names: set[str],
) -> set[str]:
    """Resolve every scheduler registration against its live constructor.

    The scheduler builds one registration per ``CRON_TASKS`` entry, so merely
    synthesizing ``cron.<task>`` from that table would not prove the
    ``SourceRegistration`` wiring still consumes it.  Validate that loop edge
    structurally, inventory additional literal registrations, and reject any
    constructor whose source name cannot be resolved.
    """

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    names: set[str] = set()
    constructors = _source_registration_constructors(tree)
    if not constructors:
        raise AssertionError("scheduler.py has no SourceRegistration constructor")

    for constructor in constructors:
        name_keyword = next(
            (item for item in constructor.keywords if item.arg == "name"),
            None,
        )
        name_expression = (
            name_keyword.value
            if name_keyword is not None
            else constructor.args[0]
            if constructor.args
            else None
        )
        if name_expression is None:
            raise AssertionError("Scheduler SourceRegistration has no name")

        direct_name = _resolved_string(name_expression, constants)
        if direct_name is not None:
            names.add(direct_name)
            continue

        if not (
            isinstance(name_expression, ast.Call)
            and _call_name(name_expression) == "cron_source_name"
            and len(name_expression.args) == 1
            and not name_expression.keywords
            and isinstance(name_expression.args[0], ast.Name)
        ):
            raise AssertionError(
                "Unresolved scheduler SourceRegistration name: "
                f"{ast.unparse(name_expression)}"
            )

        task_binding = name_expression.args[0].id
        enclosing: ast.AST | None = constructor
        while enclosing is not None and not isinstance(
            enclosing, (ast.For, ast.AsyncFor)
        ):
            enclosing = parents.get(enclosing)
        loop_target_names = (
            {
                child.id
                for child in ast.walk(enclosing.target)
                if isinstance(child, ast.Name)
            }
            if isinstance(enclosing, (ast.For, ast.AsyncFor))
            else set()
        )
        if not (
            isinstance(enclosing, (ast.For, ast.AsyncFor))
            and isinstance(enclosing.iter, ast.Name)
            and enclosing.iter.id == "CRON_TASKS"
            and task_binding in loop_target_names
        ):
            raise AssertionError(
                "cron_source_name must consume the task binding from the "
                "CRON_TASKS registration loop"
            )
        names.update(f"cron.{task_name}" for task_name in task_names)

    return names


def _resolved_source_factory_call_names(
    tree: ast.Module,
    factory: ast.FunctionDef | ast.AsyncFunctionDef,
    parameter_name: str,
    constants: dict[str, str],
    relative: str,
) -> set[str]:
    """Resolve every call's source-name argument or fail that call closed."""

    positional_parameters = [
        *factory.args.posonlyargs,
        *factory.args.args,
    ]
    positional_names = [parameter.arg for parameter in positional_parameters]
    positional_index = (
        positional_names.index(parameter_name)
        if parameter_name in positional_names
        else None
    )
    matched = False
    names: set[str] = set()
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and _call_name(call) == factory.name
        ):
            continue
        matched = True
        argument: ast.expr | None = None
        if positional_index is not None and positional_index < len(call.args):
            argument = call.args[positional_index]
        else:
            keyword = next(
                (item for item in call.keywords if item.arg == parameter_name),
                None,
            )
            if keyword is not None:
                argument = keyword.value
        if argument is None:
            raise AssertionError(
                f"Source factory {factory.name} call omits {parameter_name!r} "
                f"in {relative}: {ast.unparse(call)}"
            )
        resolved = _resolved_string(argument, constants)
        if resolved is None:
            raise AssertionError(
                f"Unresolved source name passed to {factory.name} in "
                f"{relative}: {ast.unparse(argument)}"
            )
        names.add(resolved)
    if not matched:
        raise AssertionError(
            f"SourceRegistration factory {factory.name} has no call in {relative}"
        )
    return names


def _source_registration_constructor_aliases(tree: ast.Module) -> set[str]:
    """Resolve base, subclass, import, and assignment constructor aliases."""

    aliases = {"SourceRegistration"}
    assignments: list[tuple[str, str]] = []
    class_bases: list[tuple[str, set[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for imported in node.names:
                if "SourceRegistration" in imported.name:
                    aliases.add(imported.asname or imported.name)
        elif isinstance(node, ast.ClassDef):
            class_bases.append(
                (
                    node.name,
                    {
                        base.id
                        if isinstance(base, ast.Name)
                        else base.attr
                        if isinstance(base, ast.Attribute)
                        else ""
                        for base in node.bases
                    },
                )
            )
        else:
            assignment_pairs: list[tuple[str, ast.AST]] = []
            if isinstance(node, ast.Assign):
                assignment_pairs = [
                    pair
                    for target in node.targets
                    for pair in _static_assignment_pairs(target, node.value)
                ]
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                assignment_pairs = _static_assignment_pairs(
                    node.target,
                    node.value,
                )
            for target, value in assignment_pairs:
                source = (
                    value.id
                    if isinstance(value, ast.Name)
                    else _imported_module_attribute_name(tree, value)
                )
                if source is not None:
                    assignments.append((target, source))
    changed = True
    while changed:
        changed = False
        for target, source in assignments:
            if source in aliases and target not in aliases:
                aliases.add(target)
                changed = True
        for class_name, bases in class_bases:
            if bases.intersection(aliases) and class_name not in aliases:
                aliases.add(class_name)
                changed = True
    return aliases


def _source_registration_constructors(tree: ast.Module) -> list[ast.Call]:
    """Return source constructors, including neutrally named local aliases."""

    aliases = _source_registration_constructor_aliases(tree)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            "SourceRegistration" in _call_name(node)
            or _call_name(node) in aliases
        )
    ]


def _runtime_signal_publication_surfaces(
    tree: ast.Module,
    relative: str,
) -> set[str]:
    """Return generic seams that publish runtime-contributed signal sources."""

    surfaces: set[str] = set()

    class RuntimeSignalPublisherVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def _visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            self.scope.append(node.name)
            scope_nodes = _walk_lexical_scope(node)
            registry_aliases = {"signal_registry", "source_registry"}
            alias_assignments: list[tuple[set[str], set[str]]] = []
            for child in scope_nodes:
                targets: list[ast.AST] = []
                value: ast.AST | None = None
                if isinstance(child, ast.Assign):
                    targets = list(child.targets)
                    value = child.value
                elif isinstance(child, ast.AnnAssign):
                    targets = [child.target]
                    value = child.value
                elif isinstance(child, ast.NamedExpr):
                    targets = [child.target]
                    value = child.value
                if value is None:
                    continue
                alias_assignments.append(
                    (
                        {
                            name
                            for target in targets
                            for name in _reference_binding_names(target)
                        },
                        set(_identifier_tokens(value)),
                    )
                )
            changed = True
            while changed:
                changed = False
                for targets, sources in alias_assignments:
                    if sources.intersection(registry_aliases):
                        new_aliases = targets - registry_aliases
                        if new_aliases:
                            registry_aliases.update(new_aliases)
                            changed = True
            direct_registry_publish = any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr
                in {"register", "register_batch", "register_with_policy"}
                and bool(
                    _identifier_tokens(child.func.value).intersection(
                        registry_aliases
                    )
                )
                for child in scope_nodes
            )
            imperative_publish = (
                node.name == "_register_signal_sources"
                and {"register_batch", "register_with_policy"}.issubset(
                    {
                        child.value
                        for child in scope_nodes
                        if isinstance(child, ast.Constant)
                        and isinstance(child.value, str)
                    }
                )
            )
            if direct_registry_publish or imperative_publish:
                surfaces.add(f"{relative}::{'.'.join(self.scope)}")
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_function(node)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            self._visit_function(node)

    RuntimeSignalPublisherVisitor().visit(tree)
    return surfaces


@lru_cache(maxsize=None)
def _discovered_runtime_signal_publication_surfaces() -> frozenset[str]:
    surfaces: set[str] = set()
    for path in (REPO_ROOT / "kestrel_sovereign").rglob("*.py"):
        tree = _parsed_module(path)
        surfaces.update(
            _runtime_signal_publication_surfaces(
                tree,
                path.relative_to(REPO_ROOT).as_posix(),
            )
        )
    return frozenset(surfaces)


@lru_cache(maxsize=None)
def _discovered_core_signal_source_surfaces() -> frozenset[str]:
    """Inventory every core ``SourceRegistration`` plus cron handlers.

    Source registrations are execution boundaries even when they are neither
    scheduler targets nor named after agents.  Scan every constructor under
    core instead of maintaining a list of today's always-on modules.  The one
    generic factory currently used by workflow rescue is resolved through its
    string-valued call arguments; unresolved future name expressions fail the
    contract rather than disappearing from the audit.
    """

    surfaces = set(_discovered_scheduler_surfaces())
    surfaces.update(_discovered_runtime_signal_publication_surfaces())
    scheduler_path = (
        REPO_ROOT / "kestrel_sovereign/signals/sources/scheduler.py"
    )
    for path in (REPO_ROOT / "kestrel_sovereign").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        constructors = _source_registration_constructors(tree)
        if not constructors:
            continue
        if path == scheduler_path:
            # The scheduler constructor receives ``cron.<task>`` from the
            # machine-discovered CRON_TASKS table above.
            continue

        constants = _module_string_constants(tree, path)
        parents: dict[ast.AST, ast.AST] = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        relative = path.relative_to(REPO_ROOT).as_posix()
        for constructor in constructors:
            name_keyword = next(
                (item for item in constructor.keywords if item.arg == "name"),
                None,
            )
            name_expression = (
                name_keyword.value
                if name_keyword is not None
                else constructor.args[0]
                if constructor.args
                else None
            )
            if name_expression is None:
                raise AssertionError(
                    f"SourceRegistration without a name in {relative}"
                )
            names: set[str] = set()
            direct_name = _resolved_string(name_expression, constants)
            if direct_name is not None:
                names.add(direct_name)
            elif isinstance(name_expression, ast.Name):
                enclosing: ast.AST | None = constructor
                while enclosing is not None and not isinstance(
                    enclosing, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    enclosing = parents.get(enclosing)
                if isinstance(enclosing, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    parameters = [
                        *enclosing.args.posonlyargs,
                        *enclosing.args.args,
                        *enclosing.args.kwonlyargs,
                    ]
                    parameter_names = [parameter.arg for parameter in parameters]
                    if name_expression.id in parameter_names:
                        names.update(
                            _resolved_source_factory_call_names(
                                tree,
                                enclosing,
                                name_expression.id,
                                constants,
                                relative,
                            )
                        )
            if not names:
                raise AssertionError(
                    "Unresolved SourceRegistration name expression in "
                    f"{relative}: {ast.unparse(name_expression)}"
                )
            surfaces.update(f"{relative}::{name}" for name in names)
    return frozenset(surfaces)


def _direct_tool_writer_surfaces(tree: ast.Module, relative: str) -> set[str]:
    """Return scopes that can publish values into ``_direct_tools``."""

    writers: set[str] = set()

    class DirectToolWriterVisitor(ast.NodeVisitor):
        PUBLISH_METHODS = {"__setitem__", "setdefault", "update"}

        def __init__(self) -> None:
            self.scope: list[str] = []
            self.aliases: list[set[str]] = []

        def _visit_definition(
            self,
            node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> None:
            inherited_aliases = (
                set(self.aliases[-1])
                if self.aliases
                and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                else set()
            )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                inherited_aliases.difference_update(
                    parameter.arg for parameter in parameters
                )
                if node.args.vararg is not None:
                    inherited_aliases.discard(node.args.vararg.arg)
                if node.args.kwarg is not None:
                    inherited_aliases.discard(node.args.kwarg.arg)
            self.scope.append(node.name)
            self.aliases.append(inherited_aliases)
            self.generic_visit(node)
            self.aliases.pop()
            self.scope.pop()

        def _is_registry(self, node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Attribute)
                and node.attr == "_direct_tools"
            ) or (
                isinstance(node, ast.Name)
                and bool(self.aliases)
                and node.id in self.aliases[-1]
            )

        def _record(self) -> None:
            if not self.scope:
                return
            qualified = ".".join(self.scope)
            if self.scope[-1] == "register_dynamic_tools":
                qualified = self.scope[-1]
            writers.add(f"{relative}::{qualified}")

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            self._visit_definition(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_definition(node)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            self._visit_definition(node)

        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
            assignment_pairs = [
                pair
                for target in node.targets
                for pair in _static_assignment_pairs(target, node.value)
            ]
            if self.aliases:
                rebound_names = {
                    name
                    for target in node.targets
                    for name in _assignment_target_names(target)
                }
                self.aliases[-1].difference_update(rebound_names)
                self.aliases[-1].update(
                    target
                    for target, value in assignment_pairs
                    if self._is_registry(value)
                )
            if any(
                isinstance(target, ast.Subscript)
                and self._is_registry(target.value)
                for target in node.targets
            ):
                self._record()
            if any(
                isinstance(target, ast.Attribute)
                and target.attr == "_direct_tools"
                for target in node.targets
            ) and not (
                isinstance(node.value, ast.Dict)
                and not node.value.keys
            ):
                self._record()
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
            if self.aliases:
                self.aliases[-1].difference_update(
                    _assignment_target_names(node.target)
                )
                if node.value is not None:
                    self.aliases[-1].update(
                        target
                        for target, value in _static_assignment_pairs(
                            node.target,
                            node.value,
                        )
                        if self._is_registry(value)
                    )
            if (
                isinstance(node.target, ast.Subscript)
                and self._is_registry(node.target.value)
            ):
                self._record()
            if (
                node.value is not None
                and isinstance(node.target, ast.Attribute)
                and node.target.attr == "_direct_tools"
                and not (
                    isinstance(node.value, ast.Dict)
                    and not node.value.keys
                )
            ):
                self._record()
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
            if self._is_registry(node.target):
                self._record()
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in self.PUBLISH_METHODS
                and self._is_registry(node.func.value)
            ):
                self._record()
            self.generic_visit(node)

    DirectToolWriterVisitor().visit(tree)
    return writers


@lru_cache(maxsize=None)
def _discovered_runtime_generated_tool_surfaces() -> frozenset[str]:
    """Find core execution boundaries whose public names are runtime data.

    ``Feature.get_tools`` creates ``DynamicTool`` wrappers for the statically
    discovered ``@tool`` methods. Isolated features instead advertise arbitrary
    tool names during their child handshake. Finally every visible Feature can
    be exposed as one high-level orchestrator tool through
    ``to_orchestrator_tool``. Their names cannot all be recovered from a
    decorator, so classify the generic core boundaries themselves.
    """

    surfaces: set[str] = set()
    feature_root = REPO_ROOT / "kestrel_sovereign/features"

    def base_name(base: ast.expr) -> str:
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            return base.attr
        return ""

    def walk_statements(
        statements: list[ast.stmt],
        relative: str,
        parents: tuple[str, ...] = (),
    ) -> None:
        for node in statements:
            if isinstance(node, ast.ClassDef):
                qualified = (*parents, node.name)
                if any(base_name(base) == "AgentTool" for base in node.bases):
                    for member in node.body:
                        if (
                            isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and member.name == "execute"
                        ):
                            surfaces.add(
                                f"{relative}::{'.'.join((*qualified, member.name))}"
                            )
                walk_statements(node.body, relative, qualified)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = (*parents, node.name)
                if node.name == "to_orchestrator_tool":
                    surfaces.add(f"{relative}::{'.'.join(qualified)}")
                walk_statements(node.body, relative, qualified)
            else:
                nested_statements = [
                    child
                    for child in ast.iter_child_nodes(node)
                    if isinstance(child, ast.stmt)
                ]
                walk_statements(nested_statements, relative, parents)

    for path in feature_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        walk_statements(
            tree.body,
            path.relative_to(REPO_ROOT).as_posix(),
        )

    # Non-feature providers such as MCP register arbitrary runtime names in
    # ``_direct_tools``.  Their concrete names cannot be recovered from the
    # core checkout, so inventory the registration and both governed execution
    # doors that can reach them.  Keep these explicit expected boundaries in
    # the discovery result: renaming or moving any one fails the exact-set
    # contract instead of silently shrinking its coverage.
    dynamic_boundaries = {
        REPO_ROOT / "kestrel_sovereign/agent/tool_registry.py": {
            "register_dynamic_tools",
        },
        REPO_ROOT / "kestrel_sovereign/agent/orchestrator_engine.py": {
            "execute_named_tool",
            "_dispatch_direct_tool",
        },
    }
    for path, names in dynamic_boundaries.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        discovered_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        }
        missing = names - discovered_names
        if missing:
            raise AssertionError(
                f"Missing dynamic-tool boundary in {path.relative_to(REPO_ROOT)}: "
                + ", ".join(sorted(missing))
            )
        relative = path.relative_to(REPO_ROOT).as_posix()
        surfaces.update(f"{relative}::{name}" for name in discovered_names)

    # Most runtime tools enter through ``register_dynamic_tools``, but any
    # direct write to ``_direct_tools`` is an equally real publication door.
    # Discover writers structurally so a new one cannot bypass this inventory
    # by using a special-purpose tool name or execution path.
    direct_writers: set[str] = set()
    for path in (REPO_ROOT / "kestrel_sovereign").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        direct_writers.update(
            _direct_tool_writer_surfaces(
                tree, path.relative_to(REPO_ROOT).as_posix()
            )
        )
    surfaces.update(direct_writers)

    receipt_writer = (
        "kestrel_sovereign/kestrel_agent.py::"
        "KestrelAgent.register_constitution_receipt_tool"
    )
    if receipt_writer in direct_writers:
        agent_path = REPO_ROOT / "kestrel_sovereign/kestrel_agent.py"
        agent_tree = ast.parse(
            agent_path.read_text(encoding="utf-8"), filename=str(agent_path)
        )
        if not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_handle_constitution_receipt_tool"
            for node in ast.walk(agent_tree)
        ):
            raise AssertionError("Missing constitution-receipt execution handler")
        surfaces.add(
            "kestrel_sovereign/kestrel_agent.py::"
            "KestrelAgent._handle_constitution_receipt_tool"
        )
    return frozenset(surfaces)


@lru_cache(maxsize=None)
def _discovered_builtin_command_surfaces() -> frozenset[str]:
    """Return every built-in command, including apparently local commands.

    Built-in commands bypass feature ``@tool`` discovery.  They therefore need
    an exact, unfiltered inventory or a neutrally named authority-bearing door
    can remain invisible while the feature-tool completeness gate stays green.
    """

    surfaces: set[str] = set()
    for spec in BUILTIN_COMMAND_SPECS:
        command = spec.get("cmd")
        if not isinstance(command, str):
            continue
        surfaces.add(f"kestrel_sovereign/command_handler.py::{command}")
    return frozenset(surfaces)


def _core_cli_command_names(
    tree: ast.Module,
    string_constants: dict[str, str],
) -> set[str]:
    """Replay the canonical dispatch dictionary without dropping later keys."""

    def is_dispatch_lookup(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "commands"
        )

    def scope_contains(
        scope: ast.FunctionDef | ast.AsyncFunctionDef,
        predicate: Callable[[ast.AST], bool],
    ) -> bool:
        found = False

        class DirectScopeVisitor(ast.NodeVisitor):
            def visit(self, node: ast.AST) -> None:
                nonlocal found
                if found:
                    return
                if predicate(node):
                    found = True
                    return
                super().visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                return

            def visit_AsyncFunctionDef(  # noqa: N802
                self, node: ast.AsyncFunctionDef
            ) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
                return

        visitor = DirectScopeVisitor()
        for statement in scope.body:
            visitor.visit(statement)
        return found

    scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    candidates = [
        scope
        for scope in scopes
        if scope_contains(scope, is_dispatch_lookup)
    ]
    if not candidates:
        candidates = [
            scope
            for scope in scopes
            if scope_contains(
                scope,
                lambda node: isinstance(node, (ast.Assign, ast.AnnAssign))
                and any(
                    isinstance(target, ast.Name) and target.id == "commands"
                    for target in (
                        node.targets
                        if isinstance(node, ast.Assign)
                        else [node.target]
                    )
                ),
            )
        ]
    if len(candidates) != 1:
        raise AssertionError(
            "Could not uniquely identify the core CLI command dispatch scope"
        )
    scope = candidates[0]

    def dictionary_keys(node: ast.AST) -> set[str]:
        if not isinstance(node, ast.Dict):
            raise AssertionError(
                "Core CLI commands mutation does not use a dictionary"
            )
        resolved: set[str] = set()
        for key in node.keys:
            if key is None:
                raise AssertionError(
                    "Core CLI command dispatch uses unresolved dictionary unpacking"
                )
            command = _resolved_string(key, string_constants)
            if command is None:
                raise AssertionError(
                    "Unresolved core CLI command key expression: "
                    f"{ast.unparse(key)}"
                )
            resolved.add(command)
        return resolved

    command_names: set[str] | None = None
    for statement in scope.body:
        if any(is_dispatch_lookup(node) for node in ast.walk(statement)):
            if command_names is None:
                raise AssertionError(
                    "Core CLI command dispatch is read before initialization"
                )
            return command_names
        if isinstance(statement, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "commands"
                for target in statement.targets
            ):
                command_names = dictionary_keys(statement.value)
                continue
            command_keys = [
                target.slice
                for target in statement.targets
                if isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "commands"
            ]
            if command_keys:
                if command_names is None:
                    raise AssertionError(
                        "Core CLI command dispatch is mutated before initialization"
                    )
                for key in command_keys:
                    command = _resolved_string(key, string_constants)
                    if command is None:
                        raise AssertionError(
                            "Unresolved core CLI command key expression: "
                            f"{ast.unparse(key)}"
                        )
                    command_names.add(command)
                continue
        if (
            isinstance(statement, ast.AugAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "commands"
        ):
            if command_names is None:
                raise AssertionError(
                    "Core CLI command dispatch is mutated before initialization"
                )
            if not isinstance(statement.op, ast.BitOr):
                raise AssertionError("Unresolved core CLI command-map mutation")
            command_names.update(dictionary_keys(statement.value))
            continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id == "commands"
        ):
            call = statement.value
            if command_names is None:
                raise AssertionError(
                    "Core CLI command dispatch is mutated before initialization"
                )
            if call.func.attr != "update" or len(call.args) > 1:
                raise AssertionError("Unresolved core CLI command-map mutation")
            if call.args:
                command_names.update(dictionary_keys(call.args[0]))
            if any(keyword.arg is None for keyword in call.keywords):
                raise AssertionError("Unresolved core CLI command-map mutation")
            command_names.update(
                keyword.arg for keyword in call.keywords if keyword.arg is not None
            )
            continue
        if command_names is not None and any(
            isinstance(node, ast.Name) and node.id == "commands"
            for node in ast.walk(statement)
        ):
            raise AssertionError("Unresolved core CLI command-map mutation")

    if command_names is not None:
        return command_names
    raise AssertionError("Could not find the core CLI command dispatch dictionary")


@lru_cache(maxsize=None)
def _discovered_core_cli_surfaces() -> frozenset[str]:
    """Return every command dispatched by the canonical core CLI.

    The complete dispatch dictionary is intentionally inventoried, including
    commands that turn out to be self-only or unrelated.  Filtering by names
    such as ``agent`` or ``restart`` would miss authority-bearing verbs such as
    ``ask``, ``create``, and ``update`` and would recreate the blind spot this
    contract exists to close.  Feature entry-point commands are outside core
    and do not appear in this dictionary.
    """

    cli_path = REPO_ROOT / "kestrel_sovereign/cli.py"
    tree = ast.parse(cli_path.read_text(encoding="utf-8"), filename=str(cli_path))
    string_constants = _module_string_constants(tree, cli_path)
    return frozenset({
        f"kestrel_sovereign/cli.py::kestrel {command}"
        for command in _core_cli_command_names(tree, string_constants)
    })


def _dynamic_router_publication_surfaces(
    tree: ast.Module,
    relative: str,
) -> set[str]:
    """Return function-scoped router publications from one parsed module."""

    surfaces: set[str] = set()

    class IncludeRouterVisitor(ast.NodeVisitor):
        def __init__(self, relative: str) -> None:
            self.relative = relative
            self.scope: list[str] = []
            self.scope_counts: list[int] = []
            self.aliases: list[dict[str, tuple[str, str]]] = [{}]

        def _visit_branch(
            self,
            statements: list[ast.stmt],
            inherited: dict[str, tuple[str, str]],
        ) -> dict[str, tuple[str, str]]:
            self.aliases[-1] = dict(inherited)
            self._visit_block(statements)
            return dict(self.aliases[-1])

        def _visit_compound(self, statement: ast.stmt) -> None:
            inherited = dict(self.aliases[-1])
            branches: list[list[ast.stmt]] = []
            expressions: list[ast.AST] = []
            if isinstance(statement, ast.If):
                expressions.append(statement.test)
                branches.extend((statement.body, statement.orelse))
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                expressions.extend((statement.target, statement.iter))
                branches.extend((statement.body, statement.orelse))
            elif isinstance(statement, ast.While):
                expressions.append(statement.test)
                branches.extend((statement.body, statement.orelse))
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                expressions.extend(item.context_expr for item in statement.items)
                expressions.extend(
                    item.optional_vars
                    for item in statement.items
                    if item.optional_vars is not None
                )
                branches.append(statement.body)
            elif isinstance(statement, (ast.Try, ast.TryStar)):
                branches.extend(
                    (
                        statement.body,
                        statement.orelse,
                        statement.finalbody,
                        *(handler.body for handler in statement.handlers),
                    )
                )
                expressions.extend(
                    handler.type
                    for handler in statement.handlers
                    if handler.type is not None
                )
            elif isinstance(statement, ast.Match):
                expressions.append(statement.subject)
                branches.extend(case.body for case in statement.cases)
                expressions.extend(
                    case.guard
                    for case in statement.cases
                    if case.guard is not None
                )
            else:
                self.visit(statement)
                return

            for expression in expressions:
                self.visit(expression)
            branch_states = [
                self._visit_branch(branch, inherited)
                for branch in branches
            ]
            # This inventory asks whether a binding *may* publish a router.
            # Preserve any include_router binding reachable on any branch;
            # an exact runtime path proof is not available to a static audit.
            merged = dict(inherited)
            for state in branch_states:
                merged.update(state)
            self.aliases[-1] = merged

        def _binding_for(self, value: ast.AST) -> tuple[str, str] | None:
            if isinstance(value, ast.Attribute):
                if value.attr.casefold() == "include_router":
                    return ast.unparse(value.value), "include_router"
                return None
            if isinstance(value, ast.Name):
                return self.aliases[-1].get(value.id)
            if isinstance(value, ast.IfExp):
                branch_bindings = [
                    binding
                    for branch in (value.body, value.orelse)
                    if (binding := self._binding_for(branch)) is not None
                ]
                if not branch_bindings:
                    return None
                if (
                    len(branch_bindings) == 2
                    and branch_bindings[0] == branch_bindings[1]
                ):
                    return branch_bindings[0]
                return (
                    branch_bindings[0][0],
                    _UNRESOLVED_ROUTE_REGISTRATION,
                )
            return None

        def _assignment_bindings(
            self,
            targets: list[ast.AST],
            value: ast.AST,
        ) -> dict[str, tuple[str, str]]:
            pairs = [
                pair
                for target in targets
                for pair in _static_assignment_pairs(target, value)
            ]
            if pairs:
                return {
                    name: binding
                    for name, paired_value in pairs
                    if (binding := self._binding_for(paired_value)) is not None
                }
            nested_bindings = [
                binding
                for child in ast.walk(value)
                if (binding := self._binding_for(child)) is not None
            ]
            if not nested_bindings:
                return {}
            receiver = nested_bindings[0][0]
            return {
                name: (receiver, _UNRESOLVED_ROUTE_REGISTRATION)
                for target in targets
                for name in _assignment_target_names(target)
            }

        def _visit_block(self, statements: list[ast.stmt]) -> None:
            for statement in statements:
                if isinstance(statement, _MODULE_COMPOUND_STATEMENT_TYPES):
                    self._visit_compound(statement)
                    continue
                self.visit(statement)
                targets: list[ast.AST] = []
                value: ast.AST | None = None
                if isinstance(statement, ast.Assign):
                    targets = list(statement.targets)
                    value = statement.value
                elif isinstance(statement, ast.AnnAssign):
                    targets = [statement.target]
                    value = statement.value
                target_names = {
                    name
                    for target in targets
                    for name in _assignment_target_names(target)
                }
                for target_name in target_names:
                    self.aliases[-1].pop(target_name, None)
                if value is None:
                    continue
                self.aliases[-1].update(
                    self._assignment_bindings(targets, value)
                )

        def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
            self._visit_block(node.body)

        def _visit_scope(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            self.scope.append(node.name)
            self.scope_counts.append(0)
            self.aliases.append(dict(self.aliases[-1]))
            self._visit_block(node.body)
            self.aliases.pop()
            self.scope_counts.pop()
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_scope(node)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            self._visit_scope(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            self.scope.append(node.name)
            self.aliases.append(dict(self.aliases[-1]))
            self._visit_block(node.body)
            self.aliases.pop()
            self.scope.pop()

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            registration = _route_registration_name(
                node, self.aliases[-1]
            )
            if self.scope_counts and registration == "include_router":
                ordinal = self.scope_counts[-1]
                self.scope_counts[-1] += 1
                qualified = ".".join(self.scope)
                surfaces.add(
                    f"{self.relative}::{qualified}.include_router[{ordinal}]"
                )
            self.generic_visit(node)

    IncludeRouterVisitor(relative).visit(tree)
    return surfaces


@lru_cache(maxsize=None)
def _discovered_dynamic_router_surfaces() -> frozenset[str]:
    """Return every function-scoped ``include_router`` extension boundary.

    Decorators in out-of-tree agent and host features are unavailable to a
    checkout-only scanner.  Their core publication calls are available, so an
    exact inventory of those runtime calls is the fail-closed boundary.  Calls
    at module scope mount checked-in routers whose decorators are already
    enumerated by the HTTP inventory and are intentionally excluded here.
    """

    surfaces: set[str] = set()
    for path in (REPO_ROOT / "kestrel_sovereign").rglob("*.py"):
        tree = _parsed_module(path)
        surfaces.update(
            _dynamic_router_publication_surfaces(
                tree, path.relative_to(REPO_ROOT).as_posix()
            )
        )
    return frozenset(surfaces)


def _router_prefix(
    tree: ast.Module,
    constants: dict[str, str] | None = None,
) -> str:
    """Compatibility helper for tests with one module-level ``router``."""

    return _scope_router_prefixes(tree.body, constants).get("router", "")


def _scope_router_prefixes(
    statements: list[ast.stmt],
    constants: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve every APIRouter variable in one Python lexical scope."""

    prefixes: dict[str, str] = {}

    def collect(node: ast.AST) -> None:
        # A nested function/class has its own names. Its routers are collected
        # when route discovery descends into that lexical scope.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return

        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value

        if isinstance(value, ast.Call) and _call_name(value) == "APIRouter":
            prefix = ""
            for keyword in value.keywords:
                if keyword.arg != "prefix":
                    continue
                resolved = _resolved_string(keyword.value, constants)
                if resolved is None:
                    raise AssertionError(
                        "Unresolved APIRouter prefix expression: "
                        f"{ast.unparse(keyword.value)}"
                    )
                prefix = resolved
            for target in targets:
                if not isinstance(target, ast.Name):
                    raise AssertionError(
                        "APIRouter assignment target is not a simple name: "
                        f"{ast.unparse(target)}"
                    )
                previous = prefixes.get(target.id)
                if previous is not None and previous != prefix:
                    raise AssertionError(
                        f"Ambiguous APIRouter prefix for {target.id!r}: "
                        f"{previous!r} and {prefix!r}"
                    )
                prefixes[target.id] = prefix

        for child in ast.iter_child_nodes(node):
            collect(child)

    for statement in statements:
        collect(statement)
    return prefixes


_ROUTE_REGISTRATION_NAMES = {
    "add_api_route",
    "add_api_websocket_route",
    "add_route",
    "add_websocket_route",
    "api_route",
    "delete",
    "get",
    "head",
    "include_router",
    "mount",
    "options",
    "patch",
    "post",
    "put",
    "route",
    "trace",
    "websocket",
    "websocket_route",
}
_UNRESOLVED_ROUTE_REGISTRATION = "<unresolved>"


def _scope_route_callable_aliases(
    statements: list[ast.stmt],
    prefixes: dict[str, str],
    inherited_aliases: dict[str, tuple[str, str]] | None = None,
) -> dict[str, tuple[str, str]]:
    """Resolve source-ordered aliases of bound FastAPI registrations."""

    aliases = dict(inherited_aliases or {})

    def binding_for(value: ast.AST) -> tuple[str, str] | None:
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            registration = value.attr.casefold()
            receiver = value.value.id
            if registration in _ROUTE_REGISTRATION_NAMES and (
                receiver == "app" or receiver in prefixes
            ):
                return receiver, registration
        if isinstance(value, ast.Name):
            return aliases.get(value.id)
        if isinstance(value, ast.IfExp):
            branch_bindings = [
                binding for branch in (value.body, value.orelse)
                if (binding := binding_for(branch)) is not None
            ]
            if not branch_bindings:
                return None
            if (
                len(branch_bindings) == 2
                and branch_bindings[0] == branch_bindings[1]
            ):
                return branch_bindings[0]
            return branch_bindings[0][0], _UNRESOLVED_ROUTE_REGISTRATION
        return None

    def assignment_bindings(
        targets: list[ast.AST],
        value: ast.AST,
    ) -> dict[str, tuple[str, str]]:
        pairs = [
            pair
            for target in targets
            for pair in _static_assignment_pairs(target, value)
        ]
        if pairs:
            return {
                name: binding
                for name, paired_value in pairs
                if (binding := binding_for(paired_value)) is not None
            }

        # Starred or shape-mismatched destructuring cannot be paired exactly.
        # If any value is a route registration, retain every bound name as an
        # unresolved alias so later use fails closed instead of disappearing.
        nested_bindings = [
            binding
            for child in ast.walk(value)
            if (binding := binding_for(child)) is not None
        ]
        if not nested_bindings:
            return {}
        receiver = nested_bindings[0][0]
        return {
            name: (receiver, _UNRESOLVED_ROUTE_REGISTRATION)
            for target in targets
            for name in _assignment_target_names(target)
        }

    def conditionally_bound_route_aliases(
        statement: ast.stmt,
    ) -> dict[str, tuple[str, str]]:
        bindings: dict[str, tuple[str, str]] = {}

        class RouteBindingVisitor(ast.NodeVisitor):
            def record(self, targets: list[ast.AST], value: ast.AST | None) -> None:
                if value is None:
                    return
                bindings.update(assignment_bindings(targets, value))

            def visit_Assign(self, node: ast.Assign) -> None:
                self.record(list(node.targets), node.value)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                self.record([node.target], node.value)

            def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
                self.record([node.target], node.value)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(
                self, node: ast.AsyncFunctionDef
            ) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

        RouteBindingVisitor().visit(statement)
        return bindings

    for statement in statements:
        if isinstance(statement, _MODULE_COMPOUND_STATEMENT_TYPES):
            conditional_bindings = conditionally_bound_route_aliases(statement)
            for name in _compound_binding_names(statement, set()):
                prior = aliases.get(name) or conditional_bindings.get(name)
                if prior is not None:
                    aliases[name] = (
                        prior[0],
                        _UNRESOLVED_ROUTE_REGISTRATION,
                    )
            continue

        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value
        if not targets:
            continue

        target_names = {
            name
            for target in targets
            for name in _assignment_target_names(target)
        }
        for target_name in target_names:
            aliases.pop(target_name, None)
        if value is None:
            continue

        aliases.update(assignment_bindings(targets, value))
    return aliases


def _route_registration_name(
    call: ast.Call,
    aliases: dict[str, tuple[str, str]] | None = None,
) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr.casefold()
    if isinstance(call.func, ast.Name):
        binding = (aliases or {}).get(call.func.id)
        if (
            binding is not None
            and binding[1] == _UNRESOLVED_ROUTE_REGISTRATION
        ):
            raise AssertionError(
                "Unresolved route registration alias: "
                f"{ast.unparse(call.func)}"
            )
        return binding[1] if binding is not None else None
    return None


def _route_receiver_name(
    decorator: ast.Call,
    aliases: dict[str, tuple[str, str]] | None = None,
) -> str | None:
    if isinstance(decorator.func, ast.Attribute):
        receiver = decorator.func.value
        return receiver.id if isinstance(receiver, ast.Name) else None
    if isinstance(decorator.func, ast.Name):
        binding = (aliases or {}).get(decorator.func.id)
        return binding[0] if binding is not None else None
    return None


def _programmatic_route_path(
    call: ast.Call,
    constants: dict[str, str],
) -> str:
    """Resolve a registered path, retaining a stable marker when dynamic.

    Runtime mount helpers legitimately receive a computed feature path.  Such
    a call is still an entry-door boundary and must not disappear from the
    exact inventory merely because its concrete path is runtime data.  The
    expression marker changes when the registration wiring changes, forcing a
    corresponding audit update.
    """

    path_node: ast.expr | None = call.args[0] if call.args else None
    if path_node is None:
        path_node = next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg in {"path", "path_format"}
            ),
            None,
        )
    if path_node is None:
        raise AssertionError(
            "Programmatic route registration has no path: "
            f"{ast.unparse(call)}"
        )
    resolved = _resolved_string(path_node, constants)
    if resolved is not None:
        return resolved
    return f"<dynamic:{ast.unparse(path_node)}>"


def _fastapi_generated_route_declarations(
    statements: list[ast.stmt],
    constants: dict[str, str],
) -> list[tuple[tuple[str, ...], str]]:
    """Return FastAPI's constructor-generated OpenAPI/documentation doors."""

    declarations: list[tuple[tuple[str, ...], str]] = []

    def optional_path(call: ast.Call, keyword_name: str, default: str) -> str | None:
        keyword = next(
            (item for item in call.keywords if item.arg == keyword_name),
            None,
        )
        if keyword is None:
            return default
        if isinstance(keyword.value, ast.Constant) and keyword.value.value is None:
            return None
        resolved = _resolved_string(keyword.value, constants)
        if resolved is None:
            raise AssertionError(
                f"Unresolved FastAPI {keyword_name} expression: "
                f"{ast.unparse(keyword.value)}"
            )
        return resolved

    for statement in statements:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if not isinstance(value, ast.Call) or _call_name(value) != "FastAPI":
            continue
        openapi_path = optional_path(value, "openapi_url", "/openapi.json")
        if openapi_path is None:
            continue
        declarations.append((("GET", "HEAD"), openapi_path))
        docs_path = optional_path(value, "docs_url", "/docs")
        if docs_path is not None:
            declarations.append((("GET", "HEAD"), docs_path))
            oauth_redirect_path = optional_path(
                value,
                "swagger_ui_oauth2_redirect_url",
                "/docs/oauth2-redirect",
            )
            if oauth_redirect_path is not None:
                declarations.append((("GET", "HEAD"), oauth_redirect_path))
        redoc_path = optional_path(value, "redoc_url", "/redoc")
        if redoc_path is not None:
            declarations.append((("GET", "HEAD"), redoc_path))
    return declarations


def _route_declarations(
    tree: ast.Module,
    string_constants: dict[str, str],
    method_constants: dict[str, tuple[str, ...]],
    source_path: Path | None = None,
) -> list[tuple[tuple[str, ...], str]]:
    """Return methods and canonical paths with receiver/scoped prefixes.

    Module-level decorators and registrations execute in source order. Their
    constants must therefore be resolved from the bindings that existed
    before the containing statement, not from the module's final namespace.
    """

    declarations: list[tuple[tuple[str, ...], str]] = []

    # Function and top-level class-method bodies execute after module setup and
    # therefore resolve global router objects and bound registration callables
    # from the module's final state. Decorators still execute at definition
    # time and are handled from their source-ordered snapshots below.
    module_runtime_prefixes: dict[str, str] = {}
    for index, statement in enumerate(tree.body):
        preceding = ast.Module(body=tree.body[:index], type_ignores=[])
        active_strings = _module_string_constants(preceding, source_path)
        for receiver, prefix in _scope_router_prefixes(
            [statement], active_strings
        ).items():
            previous = module_runtime_prefixes.get(receiver)
            if previous is not None and previous != prefix:
                raise AssertionError(
                    "Ambiguous APIRouter prefix for "
                    f"{receiver!r}: {previous!r} and {prefix!r}"
                )
            module_runtime_prefixes[receiver] = prefix
    module_runtime_route_aliases = _scope_route_callable_aliases(
        tree.body, module_runtime_prefixes
    )

    def module_router_reference_is_static(
        expression: ast.AST | None,
        call: ast.Call,
    ) -> bool:
        if not isinstance(expression, ast.Name):
            return False

        # Imported router names are backed by separately scanned in-tree
        # modules.  A name assigned in this module is safe only when it is the
        # local APIRouter declaration whose decorators this tree exposes.
        for statement in tree.body:
            if getattr(statement, "lineno", 0) >= call.lineno:
                break
            targets: list[ast.AST] = []
            value: ast.AST | None = None
            if isinstance(statement, ast.Assign):
                targets = list(statement.targets)
                value = statement.value
            elif isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
                value = statement.value
            if any(
                isinstance(target, ast.Name)
                and target.id == expression.id
                for target in targets
            ):
                return (
                    isinstance(value, ast.Call)
                    and _call_name(value) == "APIRouter"
                )

        if source_path is None:
            return False
        for statement in tree.body:
            if getattr(statement, "lineno", 0) >= call.lineno:
                break
            if not isinstance(statement, ast.ImportFrom):
                continue
            if not any(
                (imported.asname or imported.name) == expression.id
                for imported in statement.names
            ):
                continue
            imported_path = _resolved_repository_import_path(
                source_path,
                statement.module,
                statement.level,
            )
            if imported_path is None:
                return False
            return any(
                imported_path.is_relative_to(
                    REPO_ROOT / "kestrel_sovereign" / root
                )
                for root in ("endpoints", "features", "host_features")
            )
        return False

    def walk_scope(
        statements: list[ast.stmt],
        inherited_prefixes: dict[str, str],
        inherited_strings: dict[str, str],
        inherited_methods: dict[str, tuple[str, ...]],
        inherited_route_aliases: dict[str, tuple[str, str]],
        *,
        module_scope: bool = False,
        class_body_uses_module_globals: bool = False,
    ) -> None:
        prefixes = dict(inherited_prefixes)

        def visit(
            node: ast.AST,
            active_strings: dict[str, str],
            active_methods: dict[str, tuple[str, ...]],
            active_route_aliases: dict[str, tuple[str, str]],
        ) -> None:
            if isinstance(node, _MODULE_COMPOUND_STATEMENT_TYPES):
                disturbed = _compound_binding_names(
                    node, set(active_methods)
                )
                nested_strings = {
                    name: value
                    for name, value in active_strings.items()
                    if name not in disturbed
                }
                nested_methods = {
                    name: value
                    for name, value in active_methods.items()
                    if name not in disturbed
                }
                for child in ast.iter_child_nodes(node):
                    visit(
                        child,
                        nested_strings,
                        nested_methods,
                        active_route_aliases,
                    )
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call):
                        continue
                    methods = _route_methods(
                        decorator,
                        active_methods,
                        active_route_aliases,
                    )
                    if not methods:
                        continue
                    receiver = _route_receiver_name(
                        decorator, active_route_aliases
                    )
                    if receiver == "app":
                        prefix = ""
                    elif receiver is not None and receiver in prefixes:
                        prefix = prefixes[receiver]
                    else:
                        raise AssertionError(
                            "Unresolved route decorator receiver: "
                            f"{ast.unparse(decorator.func)}"
                        )
                    declarations.append(
                        (methods, prefix + _route_path(decorator, active_strings))
                    )
                # Top-level function bodies run after module initialization,
                # as do methods defined in a top-level class. Nested functions
                # instead close over the bindings visible in their containing
                # execution scope. Replay the body's own assignments from that
                # correct lexical floor in every case.
                child_strings = (
                    string_constants
                    if module_scope or class_body_uses_module_globals
                    else active_strings
                )
                child_methods = (
                    method_constants
                    if module_scope or class_body_uses_module_globals
                    else active_methods
                )
                child_prefixes = (
                    module_runtime_prefixes
                    if module_scope or class_body_uses_module_globals
                    else prefixes
                )
                child_route_aliases = (
                    module_runtime_route_aliases
                    if module_scope or class_body_uses_module_globals
                    else active_route_aliases
                )
                walk_scope(
                    node.body,
                    child_prefixes,
                    child_strings,
                    child_methods,
                    child_route_aliases,
                )
                return
            if isinstance(node, ast.ClassDef):
                walk_scope(
                    node.body,
                    prefixes,
                    active_strings,
                    active_methods,
                    active_route_aliases,
                    class_body_uses_module_globals=(
                        module_scope or class_body_uses_module_globals
                    ),
                )
                return
            if isinstance(node, ast.Call):
                decorator_factory = (
                    node.func if isinstance(node.func, ast.Call) else None
                )
                factory_registration = (
                    _route_registration_name(
                        decorator_factory, active_route_aliases
                    )
                    if decorator_factory is not None
                    else None
                )
                if factory_registration in {
                    "api_route",
                    "delete",
                    "get",
                    "head",
                    "options",
                    "patch",
                    "post",
                    "put",
                    "route",
                    "trace",
                    "websocket",
                    "websocket_route",
                }:
                    receiver = _route_receiver_name(
                        decorator_factory, active_route_aliases
                    )
                    if receiver == "app":
                        prefix = ""
                    elif receiver is not None and receiver in prefixes:
                        prefix = prefixes[receiver]
                    else:
                        raise AssertionError(
                            "Unresolved route decorator-factory receiver: "
                            f"{ast.unparse(decorator_factory.func)}"
                        )
                    declarations.append(
                        (
                            _route_methods(
                                decorator_factory,
                                active_methods,
                                active_route_aliases,
                            ),
                            prefix
                            + _route_path(decorator_factory, active_strings),
                        )
                    )
                registration = _route_registration_name(
                    node, active_route_aliases
                )
                if registration == "include_router":
                    receiver = _route_receiver_name(
                        node, active_route_aliases
                    )
                    prefix_keyword = next(
                        (
                            item
                            for item in node.keywords
                            if item.arg == "prefix"
                        ),
                        None,
                    )
                    include_prefix = ""
                    if prefix_keyword is not None:
                        resolved_prefix = _resolved_string(
                            prefix_keyword.value, active_strings
                        )
                        if resolved_prefix is None:
                            raise AssertionError(
                                "Unresolved include_router prefix expression: "
                                f"{ast.unparse(prefix_keyword.value)}"
                            )
                        include_prefix = resolved_prefix
                    # Checked-in APIRouter composition changes every child
                    # path. Until those child declarations are expanded here,
                    # fail closed rather than leave the old paths green. The
                    # app's existing prefix-free publication calls remain
                    # covered by their source decorators and the separate
                    # dynamic-router boundary inventory.
                    if receiver != "app" or include_prefix:
                        raise AssertionError(
                            "include_router prefix composition requires exact "
                            f"HTTP inventory support: {ast.unparse(node)}"
                        )
                    router_expression = (
                        node.args[0]
                        if node.args
                        else next(
                            (
                                item.value
                                for item in node.keywords
                                if item.arg == "router"
                            ),
                            None,
                        )
                    )
                    if module_scope and not module_router_reference_is_static(
                        router_expression,
                        node,
                    ):
                        raise AssertionError(
                            "Unresolved module-level include_router publication: "
                            f"{ast.unparse(node)}"
                        )
                if registration in {
                    "add_api_route",
                    "add_api_websocket_route",
                    "add_route",
                    "add_websocket_route",
                    "mount",
                }:
                    receiver = _route_receiver_name(
                        node, active_route_aliases
                    )
                    if receiver == "app":
                        prefix = ""
                    elif receiver is not None and receiver in prefixes:
                        prefix = prefixes[receiver]
                    else:
                        raise AssertionError(
                            "Unresolved programmatic route receiver: "
                            f"{ast.unparse(node.func)}"
                        )
                    declarations.append(
                        (
                            _route_methods(
                                node,
                                active_methods,
                                active_route_aliases,
                            ),
                            prefix
                            + _programmatic_route_path(node, active_strings),
                        )
                    )
            for child in ast.iter_child_nodes(node):
                visit(
                    child,
                    active_strings,
                    active_methods,
                    active_route_aliases,
                )

        for index, statement in enumerate(statements):
            preceding = ast.Module(
                body=statements[:index],
                type_ignores=[],
            )
            active_strings, active_methods = _module_constant_bindings(
                preceding,
                source_path if module_scope else None,
                inherited_strings,
                inherited_methods,
            )
            if module_scope:
                declarations.extend(
                    _fastapi_generated_route_declarations(
                        [statement], active_strings
                    )
                )
            new_prefixes = _scope_router_prefixes(
                [statement], active_strings
            )
            for receiver, prefix in new_prefixes.items():
                previous = prefixes.get(receiver)
                if previous is not None and previous != prefix:
                    raise AssertionError(
                        "Ambiguous APIRouter prefix for "
                        f"{receiver!r}: {previous!r} and {prefix!r}"
                    )
                prefixes[receiver] = prefix
            active_route_aliases = _scope_route_callable_aliases(
                statements[:index], prefixes, inherited_route_aliases
            )
            visit(
                statement,
                active_strings,
                active_methods,
                active_route_aliases,
            )

    walk_scope(tree.body, {}, {}, {}, {}, module_scope=True)
    return declarations


def _route_path(
    decorator: ast.Call,
    constants: dict[str, str] | None = None,
) -> str | None:
    """Resolve positional or keyword FastAPI route paths."""

    if decorator.args:
        route = _resolved_string(decorator.args[0], constants)
        if route is None:
            raise AssertionError(
                "Unresolved route path expression: "
                f"{ast.unparse(decorator.args[0])}"
            )
        return route
    for keyword in decorator.keywords:
        if keyword.arg == "path":
            route = _resolved_string(keyword.value, constants)
            if route is None:
                raise AssertionError(
                    "Unresolved route path expression: "
                    f"{ast.unparse(keyword.value)}"
                )
            return route
    raise AssertionError(f"Route decorator has no path: {ast.unparse(decorator)}")


def _module_string_collections(
    tree: ast.Module,
    source_path: Path | None = None,
) -> dict[str, tuple[str, ...]]:
    """Resolve safe module constants used by ``methods=`` declarations."""

    return _module_constant_bindings(tree, source_path)[1]


def _route_methods(
    decorator: ast.Call,
    constants: dict[str, tuple[str, ...]] | None = None,
    route_aliases: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, ...]:
    method = _route_registration_name(decorator, route_aliases)
    if method is None:
        return ()
    if method in {
        "add_api_websocket_route",
        "add_websocket_route",
        "websocket",
        "websocket_route",
    }:
        return ("WEBSOCKET",)
    if method == "mount":
        return ("MOUNT",)
    if method in {
        "get",
        "post",
        "put",
        "patch",
        "delete",
        "head",
        "options",
        "trace",
    }:
        return (method.upper(),)
    if method not in {"add_api_route", "add_route", "api_route", "route"}:
        return ()

    if any(keyword.arg is None for keyword in decorator.keywords):
        raise AssertionError(
            "Unresolved route method keyword unpacking: "
            f"{ast.unparse(decorator)}"
        )

    def normalized(values: tuple[str, ...]) -> tuple[str, ...]:
        methods = tuple(value.upper() for value in values)
        if (
            method in {"add_route", "route"}
            and "GET" in methods
            and "HEAD" not in methods
        ):
            methods = (*methods, "HEAD")
        return methods

    positional_methods: ast.expr | None = None
    if method == "add_route" and len(decorator.args) >= 3:
        positional_methods = decorator.args[2]
    elif method == "route" and len(decorator.args) >= 2:
        positional_methods = decorator.args[1]
    if positional_methods is not None:
        if isinstance(positional_methods, ast.Name):
            values = (constants or {}).get(positional_methods.id)
            if values is None:
                raise AssertionError(
                    f"Unresolved {method} positional methods expression: "
                    f"{ast.unparse(positional_methods)}"
                )
            return normalized(values)
        if not isinstance(positional_methods, (ast.List, ast.Tuple, ast.Set)):
            raise AssertionError(
                f"Unsupported {method} positional methods expression: "
                f"{ast.unparse(positional_methods)}"
            )
        resolved = tuple(
            _resolved_string(element) for element in positional_methods.elts
        )
        if any(value is None for value in resolved):
            raise AssertionError(
                f"Unresolved {method} positional method expression: "
                f"{ast.unparse(positional_methods)}"
            )
        return normalized(tuple(value for value in resolved if value is not None))
    for keyword in decorator.keywords:
        if keyword.arg != "methods":
            continue
        if isinstance(keyword.value, ast.Name):
            values = (constants or {}).get(keyword.value.id)
            if values is None:
                raise AssertionError(
                    "Unresolved api_route methods expression: "
                    f"{ast.unparse(keyword.value)}"
                )
            return normalized(values)
        if isinstance(keyword.value, (ast.List, ast.Tuple, ast.Set)):
            methods: list[str] = []
            for element in keyword.value.elts:
                value = _resolved_string(element)
                if value is None:
                    raise AssertionError(
                        "Unresolved api_route method expression: "
                        f"{ast.unparse(element)}"
                    )
                methods.append(value.upper())
            return normalized(tuple(methods))
        raise AssertionError(
            "Unsupported api_route methods expression: "
            f"{ast.unparse(keyword.value)}"
        )
    return normalized(("GET",))


def _deprecated_agent_alias(route: str) -> str | None:
    """Return the live #871 compatibility spelling for a singular route."""

    if route == "/api/agent":
        return "/agent"
    if route.startswith("/api/agent/"):
        return route.removeprefix("/api")
    return None


@lru_cache(maxsize=None)
def _discovered_http_surfaces() -> frozenset[str]:
    surfaces: set[str] = set()
    roots = (
        REPO_ROOT / "kestrel_sovereign/endpoints",
        REPO_ROOT / "kestrel_sovereign/features",
        REPO_ROOT / "kestrel_sovereign/host_features",
    )
    paths = sorted(
        {path for root in roots for path in root.rglob("*.py")}
        | {REPO_ROOT / "kestrel_sovereign/server.py"}
    )
    for path in paths:
        tree = _parsed_module(path)
        string_constants = _module_string_constants(tree, path)
        method_constants = _module_string_collections(tree, path)
        for methods, route in _route_declarations(
            tree, string_constants, method_constants, path
        ):
            segments = {part for part in route.casefold().split("/") if part}
            if (
                "MOUNT" not in methods
                and route.casefold() not in HTTP_EXACT_ROUTES
                and not segments.intersection(HTTP_SEGMENTS)
            ):
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            for method in methods:
                surfaces.add(f"{relative}::{method} {route}")
                deprecated_alias = _deprecated_agent_alias(route)
                if deprecated_alias is not None:
                    surfaces.add(f"{relative}::{method} {deprecated_alias}")
    return frozenset(surfaces)


def _agent_alias(route: str) -> str:
    """Return the concrete multi-agent spelling after target selection."""

    return f"/api/agents/{{selected_agent_name}}/{route.lstrip('/')}"


@lru_cache(maxsize=None)
def _discovered_request_routed_alias_surfaces() -> frozenset[str]:
    """Synthesize the host alias for every declared core HTTP route.

    The routing middleware accepts ``/api/agents/{name}/{remaining_path}`` and
    rewrites the remainder before FastAPI dispatch.  Consequently every
    canonical or mounted route is an agent-addressable door, even when its
    handler does not consume ``Request`` and its path says only ``security``,
    ``identity``, or ``conversations``.  This complete inventory complements
    the narrower set of intrinsically cross-agent/host routes above.
    """

    surfaces: set[str] = set()
    roots = (
        REPO_ROOT / "kestrel_sovereign/endpoints",
        REPO_ROOT / "kestrel_sovereign/features",
        REPO_ROOT / "kestrel_sovereign/host_features",
    )
    paths = sorted(
        {path for root in roots for path in root.rglob("*.py")}
        | {REPO_ROOT / "kestrel_sovereign/server.py"}
    )
    for path in paths:
        tree = _parsed_module(path)
        string_constants = _module_string_constants(tree, path)
        method_constants = _module_string_collections(tree, path)
        for methods, canonical_route in _route_declarations(
            tree, string_constants, method_constants, path
        ):
            # The host regex requires a non-empty remaining path, so the
            # canonical root has no multi-agent alias.
            if canonical_route == "/":
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            for method in methods:
                surfaces.add(
                    f"{relative}::{method} {_agent_alias(canonical_route)}"
                )
    return frozenset(surfaces)


def _documented_surfaces(section: str) -> set[str]:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    start = audit.index(section)
    next_section = audit.find("\n## ", start + len(section))
    body = audit[start:] if next_section < 0 else audit[start:next_section]
    return set(SURFACE_ID.findall(body))


def _documented_command_surfaces(section: str) -> set[str]:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    start = audit.index(section)
    next_section = audit.find("\n## ", start + len(section))
    body = audit[start:] if next_section < 0 else audit[start:next_section]
    return set(COMMAND_SURFACE_ID.findall(body))


def _documented_cli_surfaces(section: str) -> set[str]:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    start = audit.index(section)
    next_section = audit.find("\n## ", start + len(section))
    body = audit[start:] if next_section < 0 else audit[start:next_section]
    return set(CLI_SURFACE_ID.findall(body))


def test_every_core_tool_is_classified() -> None:
    assert _discovered_tool_surfaces() == _documented_surfaces(
        "## Machine-checked tool inventory"
    )


def test_generic_indirect_dispatch_tools_are_classified() -> None:
    discovered = _discovered_tool_surfaces()
    for surface in (
        "kestrel_sovereign/features/tasks/feature.py::run_workflow",
        "kestrel_sovereign/features/strategic_memory/feature.py::signal_dispatch",
        "kestrel_sovereign/features/scheduler/feature.py::schedule_add",
        "kestrel_sovereign/features/scheduler/feature.py::schedule_add_deadline",
    ):
        assert surface in discovered

        relative, public_name = surface.split("::", maxsplit=1)
        tree = ast.parse(
            (REPO_ROOT / relative).read_text(encoding="utf-8"),
            filename=relative,
        )
        string_constants = _module_string_constants(tree, REPO_ROOT / relative)
        tool_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                _public_tool_name(decorator, node.name, string_constants)
                == public_name
                for decorator in node.decorator_list
            )
        ]
        assert len(tool_nodes) == 1
        assert _is_indirect_tool_dispatcher(tool_nodes[0])


def test_runtime_generated_tool_dispatch_boundaries_are_classified() -> None:
    """Runtime-generated names must not evade the exact tool inventory."""

    expected = {
        "kestrel_sovereign/agent/orchestrator_engine.py::"
        "_dispatch_direct_tool",
        "kestrel_sovereign/agent/orchestrator_engine.py::execute_named_tool",
        "kestrel_sovereign/agent/tool_registry.py::register_dynamic_tools",
        "kestrel_sovereign/kestrel_agent.py::"
        "KestrelAgent._handle_constitution_receipt_tool",
        "kestrel_sovereign/kestrel_agent.py::"
        "KestrelAgent.register_constitution_receipt_tool",
        "kestrel_sovereign/features/base.py::"
        "Feature.get_tools.DynamicTool.execute",
        "kestrel_sovereign/features/base.py::Feature.to_orchestrator_tool",
        "kestrel_sovereign/features/isolated_runtime.py::"
        "IsolatedFeatureTool.execute",
    }
    assert _discovered_runtime_generated_tool_surfaces() == expected
    assert expected <= _discovered_tool_surfaces()

    registry_path = REPO_ROOT / "kestrel_sovereign/agent/tool_registry.py"
    registry_tree = ast.parse(
        registry_path.read_text(encoding="utf-8"),
        filename=str(registry_path),
    )
    assert any(
        isinstance(node, ast.Call)
        and _call_name(node) == "to_orchestrator_tool"
        for node in ast.walk(registry_tree)
    ), "The classified high-level Feature tool must remain wired into registration"


def test_dynamic_tool_registry_mutation_forms_are_inventoried() -> None:
    tree = ast.parse(
        "class Publisher:\n"
        "    def direct(self, tool):\n"
        "        self._direct_tools.update({'x': tool})\n\n"
        "    def default(self, tool):\n"
        "        self._direct_tools.setdefault('x', tool)\n\n"
        "    def alias(self, tool):\n"
        "        registry = self._direct_tools\n"
        "        registry['x'] = tool\n\n"
        "    def union(self, tools):\n"
        "        registry = self._direct_tools\n"
        "        registry |= tools\n\n"
        "    def replace(self, tools):\n"
        "        self._direct_tools = tools\n\n"
        "    def initialize(self):\n"
        "        self._direct_tools = {}\n\n"
        "    def remove(self):\n"
        "        self._direct_tools.pop('x', None)\n"
    )

    assert _direct_tool_writer_surfaces(tree, "example.py") == {
        "example.py::Publisher.alias",
        "example.py::Publisher.default",
        "example.py::Publisher.direct",
        "example.py::Publisher.replace",
        "example.py::Publisher.union",
    }


def test_dynamic_tool_registry_aliases_flow_into_nested_functions() -> None:
    tree = ast.parse(
        "class Publisher:\n"
        "    def outer(self):\n"
        "        registry = self._direct_tools\n"
        "        def publish(tool):\n"
        "            registry['x'] = tool\n"
        "        def shadowed(registry, tool):\n"
        "            registry['x'] = tool\n"
        "        return publish, shadowed\n"
    )

    assert _direct_tool_writer_surfaces(tree, "example.py") == {
        "example.py::Publisher.outer.publish"
    }


def test_dynamic_tool_registry_resolves_destructured_aliases() -> None:
    tree = ast.parse(
        "class Publisher:\n"
        "    def publish(self, tool):\n"
        "        registry, ignored = self._direct_tools, None\n"
        "        registry['x'] = tool\n"
    )

    assert _direct_tool_writer_surfaces(tree, "example.py") == {
        "example.py::Publisher.publish"
    }


def test_every_core_signal_source_and_builtin_handler_is_classified() -> None:
    discovered = _discovered_core_signal_source_surfaces()
    assert discovered == _documented_surfaces(
        "## Machine-checked core signal source inventory"
    )
    for surface in (
        "kestrel_sovereign/signals/sources/a2a_task_submitted.py::"
        "a2a.task_submitted",
        "kestrel_sovereign/signals/sources/a2a_question_answered.py::"
        "a2a.question_answered",
        "kestrel_sovereign/signals/sources/workflow_rescue.py::"
        "a2a_repair_dispatch",
        "kestrel_sovereign/signals/sources/scheduler.py::"
        "cron.restart_coordinator",
        "kestrel_sovereign/features/scheduler/feature.py::"
        "_run_github_pr_watch",
        "kestrel_sovereign/features/scheduler/feature.py::"
        "_run_wait_reconcile",
        "kestrel_sovereign/features/base.py::Feature._register_signal_sources",
        "kestrel_sovereign/features/contribution_runtime.py::"
        "FeatureContributionRuntime.activate",
        "kestrel_sovereign/kestrel_agent.py::"
        "KestrelAgent._boot_phase_a2a_observability_signals",
        "kestrel_sovereign/kestrel_agent.py::"
        "KestrelAgent._boot_phase_periodic_services_readiness",
    ):
        assert surface in discovered


def test_runtime_signal_publication_resolves_registry_spellings_and_aliases() -> None:
    tree = ast.parse(
        "class Publisher:\n"
        "    def signal_registry(self, source):\n"
        "        self.signal_registry.register(source)\n\n"
        "    def local_alias(self, sources):\n"
        "        registry = self.source_registry\n"
        "        registry.register_batch(sources)\n"
    )

    assert _runtime_signal_publication_surfaces(tree, "example.py") == {
        "example.py::Publisher.local_alias",
        "example.py::Publisher.signal_registry",
    }


def test_signal_source_inventory_scans_beyond_scheduler_module() -> None:
    discovered = _discovered_core_signal_source_surfaces()
    non_scheduler_sources = {
        surface
        for surface in discovered
        if "/signals/sources/" in surface
        and "/scheduler.py::" not in surface
    }

    assert len(non_scheduler_sources) == 17
    assert {
        "kestrel_sovereign/signals/sources/a2a.py::a2a.task_complete",
        "kestrel_sovereign/signals/sources/channels.py::channel.message",
        "kestrel_sovereign/signals/sources/heartbeat.py::heartbeat",
        "kestrel_sovereign/signals/sources/restart.py::restart.completed",
        "kestrel_sovereign/signals/sources/system_resumed.py::system.resumed",
        "kestrel_sovereign/signals/sources/wallet.py::"
        "webhook.stripe.deposit_complete",
        "kestrel_sovereign/signals/sources/workflow_rescue.py::"
        "fleet_stalled_sweep",
    } <= non_scheduler_sources


def test_scheduler_registration_constructors_are_structurally_inventoried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wired = ast.parse(
        'CRON_TASKS = [("alpha", mode, resources), '
        '("beta", mode, resources)]\n'
        'for task_name, mode, resources in CRON_TASKS:\n'
        '    registrations.append(\n'
        '        SourceRegistration(name=cron_source_name(task_name))\n'
        '    )\n'
        'SourceRegistration(name="manual.source")\n'
    )
    assert _scheduler_registration_source_names(
        wired,
        _module_string_constants(wired),
        {"alpha", "beta"},
    ) == {"cron.alpha", "cron.beta", "manual.source"}

    unwired = ast.parse(
        'CRON_TASKS = [("alpha", mode, resources)]\n'
        'SourceRegistration(name=cron_source_name(task_name))\n'
    )
    with pytest.raises(AssertionError, match="CRON_TASKS registration loop"):
        _scheduler_registration_source_names(
            unwired,
            _module_string_constants(unwired),
            {"alpha"},
        )

    def reject_unvalidated_scheduler(*_args: object) -> set[str]:
        raise RuntimeError("constructor validator reached")

    _discovered_scheduler_surfaces.cache_clear()
    monkeypatch.setitem(
        globals(),
        "_scheduler_registration_source_names",
        reject_unvalidated_scheduler,
    )
    with pytest.raises(RuntimeError, match="constructor validator reached"):
        _discovered_scheduler_surfaces()
    _discovered_scheduler_surfaces.cache_clear()


@pytest.mark.parametrize(
    "mutation",
    [
        'CRON_TASKS.append(("late", SignalMode.ACTION, frozenset()))',
        'CRON_TASKS += [("late", SignalMode.ACTION, frozenset())]',
        'CRON_TASKS = [*CRON_TASKS, ("late", SignalMode.ACTION, frozenset())]',
    ],
    ids=["append", "augmented-assignment", "reassignment"],
)
def test_scheduler_inventory_replays_cron_tasks_mutations(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = REPO_ROOT / "kestrel_sovereign/signals/sources/scheduler.py"
    original_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        source = original_read_text(path, *args, **kwargs)
        if path != source_path:
            return source
        marker = "\n\n\ndef cron_source_name(task_name: str) -> str:"
        assert source.count(marker) == 1
        return source.replace(marker, f"\n{mutation}{marker}")

    monkeypatch.setattr(Path, "read_text", read_text)
    _discovered_scheduler_surfaces.cache_clear()
    try:
        assert (
            "kestrel_sovereign/signals/sources/scheduler.py::cron.late"
            in _discovered_scheduler_surfaces()
        )
    finally:
        _discovered_scheduler_surfaces.cache_clear()


def test_each_source_factory_call_must_resolve_its_own_name() -> None:
    tree = ast.parse(
        "STATIC_NAME = 'static.source'\n"
        "def make_source(name):\n"
        "    return SourceRegistration(name=name)\n\n"
        "make_source(STATIC_NAME)\n"
        "make_source(runtime_name)\n"
    )
    factory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "make_source"
    )

    with pytest.raises(AssertionError, match="runtime_name"):
        _resolved_source_factory_call_names(
            tree,
            factory,
            "name",
            _module_string_constants(tree),
            "example.py",
        )


def test_signal_source_constructor_import_aliases_are_resolved() -> None:
    tree = ast.parse(
        "from kestrel_sdk.signals import SourceRegistration as Registration\n"
        "Registration(name='aliased.source')\n"
    )

    assert _source_registration_constructor_aliases(tree) == {
        "SourceRegistration",
        "Registration",
    }
    constructors = _source_registration_constructors(tree)
    assert len(constructors) == 1
    assert _call_name(constructors[0]) == "Registration"

    module_alias = ast.parse(
        "import kestrel_sdk.signals as signals\n"
        "Registration = signals.SourceRegistration\n"
        "Registration(name='module.attribute.source')\n"
    )
    assert "Registration" in _source_registration_constructor_aliases(
        module_alias
    )
    module_constructors = _source_registration_constructors(module_alias)
    assert len(module_constructors) == 1
    assert _call_name(module_constructors[0]) == "Registration"


def test_signal_source_constructor_annotated_aliases_are_resolved() -> None:
    tree = ast.parse(
        "Registration: type = SourceRegistration\n"
        "Registration(name='annotated.source')\n"
    )

    assert "Registration" in _source_registration_constructor_aliases(tree)
    constructors = _source_registration_constructors(tree)
    assert len(constructors) == 1
    assert _call_name(constructors[0]) == "Registration"


def test_signal_source_registration_subclasses_are_resolved() -> None:
    local_subclass = ast.parse(
        "class Specialized(SourceRegistration):\n"
        "    pass\n\n"
        "Specialized(name='specialized.source')\n"
    )
    imported_subclass = ast.parse(
        "from kestrel_sovereign.signals import (\n"
        "    SourceRegistrationWithPromptOverride as Specialized,\n"
        ")\n"
        "Specialized(name='overridden.source')\n"
    )

    assert {
        _call_name(constructor)
        for constructor in _source_registration_constructors(local_subclass)
    } == {"Specialized"}
    assert {
        _call_name(constructor)
        for constructor in _source_registration_constructors(imported_subclass)
    } == {"Specialized"}


def test_every_dynamic_router_publication_boundary_is_classified() -> None:
    expected = {
        "kestrel_sovereign/host_features/runtime.py::"
        "mount_host_feature_routers.include_router[0]",
        "kestrel_sovereign/server.py::"
        "_mount_feature_routers._collect_routers_from_agent.include_router[0]",
        "kestrel_sovereign/server.py::"
        "_mount_feature_routers.include_router[0]",
    }
    assert _discovered_dynamic_router_surfaces() == expected
    assert expected == _documented_surfaces(
        "## Machine-checked dynamic router boundary inventory"
    )


def test_dynamic_router_publication_resolves_bound_aliases() -> None:
    tree = ast.parse(
        "def mount(app, feature):\n"
        "    publish = app.include_router\n"
        "    alias = publish\n"
        "    alias(feature.get_router())\n"
    )

    assert _dynamic_router_publication_surfaces(tree, "example.py") == {
        "example.py::mount.include_router[0]"
    }

    conditional = ast.parse(
        "def mount(app, router, enabled):\n"
        "    publish = noop\n"
        "    if enabled:\n"
        "        publish = app.include_router\n"
        "    publish(router)\n"
    )
    nested = ast.parse(
        "def mount(app, router, enabled):\n"
        "    if enabled:\n"
        "        publish = app.include_router\n"
        "        publish(router)\n"
    )

    assert _dynamic_router_publication_surfaces(
        conditional, "conditional.py"
    ) == {"conditional.py::mount.include_router[0]"}
    assert _dynamic_router_publication_surfaces(nested, "nested.py") == {
        "nested.py::mount.include_router[0]"
    }


def test_dynamic_router_publication_resolves_destructured_aliases() -> None:
    tree = ast.parse(
        "def mount(app, router):\n"
        "    publish, ignored = app.include_router, noop\n"
        "    publish(router)\n"
    )

    assert _dynamic_router_publication_surfaces(tree, "example.py") == {
        "example.py::mount.include_router[0]"
    }


def test_relation_free_control_names_are_still_discovered() -> None:
    """A control door need not say ``agent`` or ``task`` to cross a boundary."""

    for name in (
        "hold",
        "interrupt",
        "offboard",
        "terminate",
        "withdraw",
        "stop",
    ):
        assert _is_cross_agent_control_name(name)


def test_every_builtin_command_is_classified() -> None:
    assert _discovered_builtin_command_surfaces() == _documented_command_surfaces(
        "## Machine-checked built-in command inventory"
    )


def test_builtin_agent_creation_command_is_discovered() -> None:
    assert (
        "kestrel_sovereign/command_handler.py::!create-agent"
        in _discovered_builtin_command_surfaces()
    )


def test_every_core_cli_command_is_classified() -> None:
    assert _discovered_core_cli_surfaces() == _documented_cli_surfaces(
        "## Machine-checked core CLI inventory"
    )


def test_core_cli_agent_and_fleet_controls_are_discovered() -> None:
    discovered = _discovered_core_cli_surfaces()
    for command in ("ask", "create", "terminate", "restart", "update"):
        assert f"kestrel_sovereign/cli.py::kestrel {command}" in discovered


def test_core_cli_command_keys_resolve_constants_and_fail_closed() -> None:
    resolved = ast.parse(
        '_CONTROL = "restart"\n'
        "def dispatch():\n"
        "    commands = {_CONTROL: restart_agent}\n"
    )
    assert _core_cli_command_names(
        resolved, _module_string_constants(resolved)
    ) == {"restart"}

    unresolved = ast.parse(
        "def dispatch():\n"
        "    commands = {make_command_name(): restart_agent}\n"
    )
    with pytest.raises(AssertionError, match="Unresolved core CLI command key"):
        _core_cli_command_names(unresolved, {})

    unpacked = ast.parse(
        "def dispatch():\n"
        "    commands = {**extension_commands}\n"
    )
    with pytest.raises(AssertionError, match="unresolved dictionary unpacking"):
        _core_cli_command_names(unpacked, {})


def test_core_cli_command_inventory_replays_dispatch_map_mutations() -> None:
    tree = ast.parse(
        "def helper():\n"
        "    commands = {'irrelevant': noop}\n\n"
        "def dispatch():\n"
        "    commands = {'start': start}\n"
        "    commands['terminate'] = terminate\n"
        "    commands.update({'restart': restart})\n"
        "    handler = commands.get(args.command)\n"
    )

    assert _core_cli_command_names(tree, {}) == {
        "start",
        "terminate",
        "restart",
    }


def test_every_cross_agent_http_route_is_classified() -> None:
    assert _discovered_http_surfaces() == _documented_surfaces(
        "## Machine-checked HTTP inventory"
    )


def test_every_request_routed_agent_alias_is_classified() -> None:
    assert _discovered_request_routed_alias_surfaces() == _documented_surfaces(
        "## Machine-checked request-routed alias inventory"
    )


def test_sensitive_unprefixed_routes_have_synthesized_agent_aliases() -> None:
    discovered = _discovered_request_routed_alias_surfaces()
    for surface in (
        "kestrel_sovereign/endpoints/security.py::"
        "POST /api/agents/{selected_agent_name}/api/security/approve",
        "kestrel_sovereign/endpoints/security.py::"
        "POST /api/agents/{selected_agent_name}/api/security/auto-mode",
        "kestrel_sovereign/endpoints/models.py::"
        "PATCH /api/agents/{selected_agent_name}/api/identity",
        "kestrel_sovereign/endpoints/conversations.py::"
        "DELETE /api/agents/{selected_agent_name}/api/conversations/{session_id}",
        "kestrel_sovereign/endpoints/memories.py::"
        "DELETE /api/agents/{selected_agent_name}/api/memories/{node_id}",
    ):
        assert surface in discovered


def test_feature_contributed_nested_webhook_router_is_discovered() -> None:
    assert (
        "kestrel_sovereign/features/webhooks/receiver.py::"
        "POST /webhooks/{webhook_name}"
    ) in _discovered_http_surfaces()


def test_host_feature_and_shared_observability_routes_are_discovered() -> None:
    discovered = _discovered_http_surfaces()
    for surface in (
        "kestrel_sovereign/endpoints/features.py::"
        "POST /api/features/{name}/install",
        "kestrel_sovereign/endpoints/features.py::"
        "POST /api/features/{name}/remove",
        "kestrel_sovereign/endpoints/observability.py::"
        "GET /api/observability/summary",
        "kestrel_sovereign/endpoints/observability.py::"
        "GET /api/observability/metrics/{metric_name}",
    ):
        assert surface in discovered


def test_every_live_agent_invocation_route_is_discovered() -> None:
    discovered = _discovered_http_surfaces()
    for surface in (
        "kestrel_sovereign/endpoints/agent.py::POST /api/agent/invoke",
        "kestrel_sovereign/endpoints/agent.py::POST /api/agent/stream",
        "kestrel_sovereign/features/bridge/router.py::POST /api/bridge/invoke",
        "kestrel_sovereign/features/bridge/router.py::POST /api/bridge/stream",
        "kestrel_sovereign/endpoints/models.py::POST /v1/chat/completions",
    ):
        assert surface in discovered


def test_app_level_host_authority_routes_are_discovered() -> None:
    discovered = _discovered_http_surfaces()
    for surface in (
        "kestrel_sovereign/server.py::GET /api/auth/key",
        "kestrel_sovereign/server.py::GET /api/host/ui/contributions",
        "kestrel_sovereign/server.py::GET /api/host/csrf",
        "kestrel_sovereign/server.py::POST /api/host/phoenix/session",
        "kestrel_sovereign/server.py::GET /health/detailed",
        "kestrel_sovereign/server.py::GET /phoenix",
        "kestrel_sovereign/server.py::GET /phoenix/{path:path}",
        "kestrel_sovereign/endpoints/github.py::GET /api/github/repos",
        "kestrel_sovereign/endpoints/github.py::GET /api/github/{path:path}",
        "kestrel_sovereign/endpoints/models.py::GET /api/keys/platform",
        "kestrel_sovereign/endpoints/models.py::GET /api/keys/user",
        "kestrel_sovereign/endpoints/models.py::POST /api/keys/user",
        "kestrel_sovereign/endpoints/models.py::POST /api/keys/user/verify",
        "kestrel_sovereign/endpoints/models.py::DELETE /api/keys/user/{provider}",
    ):
        assert surface in discovered

    # The cross-agent audit and the general auth ledger must agree that the
    # bootstrap credential is a narrow public-localhost exception, not an
    # ordinary authenticated agent route.
    auth_matrix = AUTH_SURFACE_MATRIX_PATH.read_text(encoding="utf-8")
    assert "| `Public-Localhost`" in auth_matrix
    assert "/api/auth/key" in auth_matrix


def test_bootstrap_api_key_authority_matches_host_lifecycle_gate() -> None:
    """The audit must not confuse the signing key with runtime API-key power."""

    caller = CallerContext.sovereign(AuthMethod.API_KEY)
    request = SimpleNamespace(state=SimpleNamespace(caller=caller))
    assert require_sovereign_host_lifecycle(request) is caller

    audit = AUDIT_PATH.read_text(encoding="utf-8")
    row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Bootstrap host API credential ")
    )
    assert "CallerRole.SOVEREIGN" in row
    assert "satisfies the current #3149 host-lifecycle gate" in row
    assert "create or withdraw hosted agents" in row
    assert "distinct from the constitutional sovereign signing key" in row


def test_host_github_and_non_agent_key_principals_are_recorded() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")

    github_matrix = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Use the host GitHub credential ")
    )
    assert "process-wide GitHub token" in github_matrix
    assert "do not bind `get_agent`" in github_matrix

    key_matrix = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Manage authenticated-user or platform service keys ")
    )
    assert "request.state.user_id" in key_matrix
    assert "selected agent supplies only PostgreSQL connectivity" in key_matrix

    for route in ("/api/github/repos", "/api/github/{path:path}"):
        canonical = next(
            line
            for line in audit.splitlines()
            if f"endpoints/github.py::GET {route}`" in line
        )
        assert "process-wide GitHub credential" in canonical

        alias = next(
            line
            for line in audit.splitlines()
            if "endpoints/github.py::GET "
            f"/api/agents/{{selected_agent_name}}{route}`" in line
        )
        assert "H —" in alias
        assert "selected-agent prefix" in alias

    for method, route in (
        ("DELETE", "/api/keys/user/{provider}"),
        ("GET", "/api/keys/user"),
        ("POST", "/api/keys/user"),
        ("POST", "/api/keys/user/verify"),
    ):
        canonical = next(
            line
            for line in audit.splitlines()
            if f"endpoints/models.py::{method} {route}`" in line
        )
        assert "request.state.user_id" in canonical

        alias = next(
            line
            for line in audit.splitlines()
            if "endpoints/models.py::"
            f"{method} /api/agents/{{selected_agent_name}}{route}`" in line
        )
        assert "U —" in alias

    platform = next(
        line
        for line in audit.splitlines()
        if "endpoints/models.py::GET /api/keys/platform`" in line
    )
    assert "platform-global" in platform


def test_canonical_host_authentication_routes_are_discovered() -> None:
    discovered = _discovered_http_surfaces()
    for method, route in (
        ("GET", "/auth/login"),
        ("GET", "/auth/callback"),
        ("GET", "/auth/logout"),
        ("POST", "/auth/token"),
        ("GET", "/auth/me"),
        ("GET", "/auth/verify"),
    ):
        assert (
            "kestrel_sovereign/endpoints/auth_oauth.py::"
            f"{method} {route}"
        ) in discovered


def test_deprecated_agent_compatibility_routes_are_discovered() -> None:
    discovered = _discovered_http_surfaces()
    for surface in (
        "kestrel_sovereign/endpoints/agent.py::POST /agent/invoke",
        "kestrel_sovereign/endpoints/agent.py::POST /agent/stream",
        "kestrel_sovereign/endpoints/agent.py::GET /agent/tasks",
        "kestrel_sovereign/endpoints/agent.py::"
        "POST /agent/tasks/{task_id:path}/cancel",
        "kestrel_sovereign/endpoints/files.py::"
        "GET /agent/channels/{channel_type}/link-qr.png",
    ):
        assert surface in discovered


def test_api_route_declarations_expand_every_registered_method() -> None:
    decorator = ast.parse(
        '@router.api_route("/api/tasks", methods=["POST", "PUT"])\n'
        "def route():\n    pass\n"
    ).body[0].decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert _route_methods(decorator) == ("POST", "PUT")


def test_api_route_declarations_resolve_module_method_constants() -> None:
    tree = ast.parse(
        '_METHODS = ["GET", "POST"]\n'
        '@router.api_route("/phoenix", methods=_METHODS)\n'
        "def route():\n    pass\n"
    )
    decorator = tree.body[1].decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert _route_methods(decorator, _module_string_collections(tree)) == (
        "GET",
        "POST",
    )

    reassigned = ast.parse(
        '_METHODS = ["GET"]\n'
        "_METHODS = runtime_methods()\n"
        '@router.api_route("/phoenix", methods=_METHODS)\n'
        "def route():\n    pass\n"
    )
    reassigned_decorator = reassigned.body[2].decorator_list[0]
    assert isinstance(reassigned_decorator, ast.Call)
    method_constants = _module_string_collections(reassigned)
    assert "_METHODS" not in method_constants
    with pytest.raises(AssertionError, match="Unresolved api_route methods"):
        _route_methods(reassigned_decorator, method_constants)


def test_method_collections_capture_scalar_values_in_source_order() -> None:
    tree = ast.parse(
        'METHOD = "POST"\n'
        "METHODS = [METHOD]\n"
        'METHOD = "GET"\n'
        "router = APIRouter()\n"
        '@router.api_route("/api/agents/{agent}/control", methods=METHODS)\n'
        "def control():\n    pass\n"
    )
    strings = _module_string_constants(tree)
    methods = _module_string_collections(tree)

    assert strings["METHOD"] == "GET"
    assert methods["METHODS"] == ("POST",)
    assert _route_declarations(tree, strings, methods) == [
        (("POST",), "/api/agents/{agent}/control")
    ]


def test_route_declarations_resolve_constants_at_decorator_execution() -> None:
    tree = ast.parse(
        'ROUTE = "/api/agents/{agent}/terminate"\n'
        'METHODS = ["POST"]\n'
        "router = APIRouter()\n"
        "@router.api_route(ROUTE, methods=METHODS)\n"
        "def terminate():\n    pass\n"
        'ROUTE = "/benign"\n'
        'METHODS = ["GET"]\n'
    )
    final_strings = _module_string_constants(tree)
    final_methods = _module_string_collections(tree)

    assert final_strings["ROUTE"] == "/benign"
    assert final_methods["METHODS"] == ("GET",)
    assert _route_declarations(tree, final_strings, final_methods) == [
        (("POST",), "/api/agents/{agent}/terminate")
    ]


def test_route_declarations_resolve_bound_registration_aliases() -> None:
    tree = ast.parse(
        'router = APIRouter(prefix="/api")\n'
        'publish = router.delete\n'
        '@publish("/agents/{name}")\n'
        'def delete_agent():\n'
        '    pass\n'
    )

    assert _route_declarations(
        tree,
        _module_string_constants(tree),
        _module_string_collections(tree),
    ) == [(('DELETE',), '/api/agents/{name}')]


def test_route_declarations_resolve_destructured_registration_aliases() -> None:
    tree = ast.parse(
        'router = APIRouter(prefix="/api")\n'
        'read_route, write_route = router.get, router.post\n'
        '@read_route("/agents/{name}")\n'
        'def read_agent():\n'
        '    pass\n'
        '@write_route("/agents/{name}")\n'
        'def write_agent():\n'
        '    pass\n'
    )

    assert _route_declarations(
        tree,
        _module_string_constants(tree),
        _module_string_collections(tree),
    ) == [
        (("GET",), "/api/agents/{name}"),
        (("POST",), "/api/agents/{name}"),
    ]


def test_route_declarations_reject_conditionally_rebound_aliases() -> None:
    tree = ast.parse(
        "router = APIRouter()\n"
        "register = router.get\n"
        "if enabled:\n"
        "    register = router.post\n"
        "@register('/api/agents/{name}')\n"
        "def route():\n"
        "    pass\n"
    )

    with pytest.raises(
        AssertionError, match="Unresolved route registration alias"
    ):
        _route_declarations(tree, {}, {})


@pytest.mark.parametrize(
    "source",
    [
        (
            "router = APIRouter()\n"
            "if enabled:\n"
            "    register = router.post\n"
            "@register('/api/agents/{name}')\n"
            "def route():\n"
            "    pass\n"
        ),
        (
            "router = APIRouter()\n"
            "register = router.post if enabled else noop\n"
            "@register('/api/agents/{name}')\n"
            "def route():\n"
            "    pass\n"
        ),
    ],
    ids=["compound-binding", "conditional-expression"],
)
def test_route_declarations_reject_conditionally_introduced_aliases(
    source: str,
) -> None:
    with pytest.raises(
        AssertionError, match="Unresolved route registration alias"
    ):
        _route_declarations(ast.parse(source), {}, {})


@pytest.mark.parametrize(
    "source",
    [
        (
            "router = APIRouter()\n"
            "register = router.delete\n"
            "def build_router():\n"
            "    @register('/api/agents/{name}')\n"
            "    def terminate():\n"
            "        pass\n"
            "    return router\n"
        ),
        (
            "def build_router():\n"
            "    router = APIRouter()\n"
            "    register = router.delete\n"
            "    def declare_routes():\n"
            "        @register('/api/agents/{name}')\n"
            "        def terminate():\n"
            "            pass\n"
            "    return router\n"
        ),
    ],
    ids=["module-alias", "nested-alias"],
)
def test_route_declarations_inherit_bound_aliases_into_nested_scopes(
    source: str,
) -> None:
    assert _route_declarations(ast.parse(source), {}, {}) == [
        (("DELETE",), "/api/agents/{name}")
    ]


def test_route_declarations_replay_nested_lexical_constants() -> None:
    tree = ast.parse(
        'PATH = "/health"\n'
        'METHODS = ["GET"]\n'
        'def factory():\n'
        '    PATH = "/api/agents/{name}/terminate"\n'
        '    METHODS = ["POST", "DELETE"]\n'
        '    router = APIRouter(prefix="/v1")\n'
        '    @router.api_route(PATH, methods=METHODS)\n'
        '    def route():\n'
        '        pass\n'
        '    return router\n'
    )

    assert _route_declarations(
        tree,
        _module_string_constants(tree),
        _module_string_collections(tree),
    ) == [
        (("POST", "DELETE"), "/v1/api/agents/{name}/terminate")
    ]


def test_class_method_route_registrations_use_final_module_globals() -> None:
    tree = ast.parse(
        'PATH = "/health"\n'
        'METHODS = ["GET"]\n'
        'ALIAS_PATH = "/health"\n'
        'publish = noop\n'
        'class Routes:\n'
        '    @app.get(PATH)\n'
        '    def static_decorator(self):\n'
        '        pass\n\n'
        '    def mount(self, app):\n'
        '        app.add_api_route(PATH, endpoint, methods=METHODS)\n\n'
        '    def mount_alias(self):\n'
        '        publish(ALIAS_PATH)(endpoint)\n\n'
        'PATH = "/api/agents/{name}/terminate"\n'
        'METHODS = ["POST", "DELETE"]\n'
        'ALIAS_PATH = "/api/agents/{name}/restart"\n'
        'publish = app.patch\n'
    )

    assert _route_declarations(
        tree,
        _module_string_constants(tree),
        _module_string_collections(tree),
    ) == [
        (("GET",), "/health"),
        (("POST", "DELETE"), "/api/agents/{name}/terminate"),
        (("PATCH",), "/api/agents/{name}/restart"),
    ]


def test_route_decorator_resolves_positional_methods_and_fails_closed() -> None:
    tree = ast.parse(
        '_MUTATIONS = ["POST", "DELETE"]\n'
        '@router.route("/agents/{agent}/terminate", _MUTATIONS)\n'
        "def terminate():\n    pass\n"
    )
    decorator = tree.body[1].decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert _route_methods(decorator, _module_string_collections(tree)) == (
        "POST",
        "DELETE",
    )

    unresolved = ast.parse(
        '@router.route("/agents/{agent}/terminate", methods_for_target())\n'
        "def terminate():\n    pass\n"
    ).body[0].decorator_list[0]
    assert isinstance(unresolved, ast.Call)
    with pytest.raises(AssertionError, match="Unsupported route positional methods"):
        _route_methods(unresolved)


def test_websocket_declarations_are_inventoried_as_agent_addressable() -> None:
    tree = ast.parse(
        'router = APIRouter(prefix="/api")\n'
        '@router.websocket("/agents/{agent_name}/control")\n'
        "async def control():\n    pass\n\n"
        '@router.websocket_route("/agents/{agent_name}/events")\n'
        "async def events():\n    pass\n"
    )
    assert _route_declarations(tree, {}, {}) == [
        (("WEBSOCKET",), "/api/agents/{agent_name}/control"),
        (("WEBSOCKET",), "/api/agents/{agent_name}/events"),
    ]


def test_programmatic_route_registrations_and_mounts_are_inventoried() -> None:
    tree = ast.parse(
        'router = APIRouter(prefix="/api")\n'
        "def mutate():\n    pass\n"
        'router.add_api_route("/agents/{agent}/terminate", mutate, '
        'methods=["POST", "DELETE"])\n'
        'app.add_route(path="/host/status", route=mutate)\n'
        'app.add_websocket_route("/events", mutate)\n'
        'app.add_api_websocket_route("/api-events", mutate)\n'
        'app.add_route("/host/mutate", mutate, ["PATCH"])\n'
        "app.mount(mount_path, mutate)\n"
    )
    assert _route_declarations(tree, {}, {}) == [
        (("POST", "DELETE"), "/api/agents/{agent}/terminate"),
        (("GET", "HEAD"), "/host/status"),
        (("WEBSOCKET",), "/events"),
        (("WEBSOCKET",), "/api-events"),
        (("PATCH",), "/host/mutate"),
        (("MOUNT",), "<dynamic:mount_path>"),
    ]


def test_decorator_factory_route_registrations_are_inventoried() -> None:
    tree = ast.parse(
        'router = APIRouter(prefix="/api")\n'
        'app.post("/api/agents/{agent}/stop")(handler)\n'
        'router.api_route(\n'
        '    "/agents/{agent}/hold", methods=["POST", "DELETE"]\n'
        ')(handler)\n'
        'router.websocket("/agents/{agent}/events")(socket_handler)\n'
        'publish = app.patch\n'
        'publish("/api/agents/{agent}/restart")(handler)\n'
    )

    assert _route_declarations(tree, {}, {}) == [
        (("POST",), "/api/agents/{agent}/stop"),
        (("POST", "DELETE"), "/api/agents/{agent}/hold"),
        (("WEBSOCKET",), "/api/agents/{agent}/events"),
        (("PATCH",), "/api/agents/{agent}/restart"),
    ]


def test_router_composition_cannot_leave_uncomposed_paths_green() -> None:
    nested_router = ast.parse(
        'child = APIRouter(prefix="/child")\n'
        '@child.get("/status")\n'
        "def status():\n    pass\n"
        'parent = APIRouter(prefix="/parent")\n'
        'parent.include_router(child, prefix="/nested")\n'
    )
    app_prefix = ast.parse(
        'child = APIRouter(prefix="/child")\n'
        '@child.get("/status")\n'
        "def status():\n    pass\n"
        'app.include_router(child, prefix="/v2")\n'
    )

    with pytest.raises(AssertionError, match="prefix composition"):
        _route_declarations(nested_router, {}, {})
    with pytest.raises(AssertionError, match="prefix composition"):
        _route_declarations(app_prefix, {}, {})

    dynamic_module_publication = ast.parse(
        "app.include_router(load_plugin_router())\n"
    )
    with pytest.raises(AssertionError, match="module-level include_router"):
        _route_declarations(dynamic_module_publication, {}, {})

    named_dynamic_module_publication = ast.parse(
        "plugin_router = load_plugin_router()\n"
        "app.include_router(plugin_router)\n"
    )
    with pytest.raises(AssertionError, match="module-level include_router"):
        _route_declarations(named_dynamic_module_publication, {}, {})


def test_fastapi_generated_routes_follow_constructor_configuration() -> None:
    defaults = ast.parse("app = FastAPI()\n")
    assert _route_declarations(defaults, {}, {}) == [
        (("GET", "HEAD"), "/openapi.json"),
        (("GET", "HEAD"), "/docs"),
        (("GET", "HEAD"), "/docs/oauth2-redirect"),
        (("GET", "HEAD"), "/redoc"),
    ]

    disabled = ast.parse(
        "app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)\n"
    )
    assert _route_declarations(disabled, {}, {}) == []

    docs_disabled = ast.parse("app = FastAPI(docs_url=None)\n")
    assert _route_declarations(docs_disabled, {}, {}) == [
        (("GET", "HEAD"), "/openapi.json"),
        (("GET", "HEAD"), "/redoc"),
    ]


def test_decorator_strings_resolve_keywords_and_module_constants() -> None:
    tree = ast.parse(
        '_BASE = "/api"\n'
        '_ROUTE = _BASE + "/agent/constant"\n'
        '_PREFIX = f"{_BASE}"\n'
        '_TOOL_PREFIX = "neutral"\n'
        '_TOOL_NAME = _TOOL_PREFIX + "-control"\n'
        "router = APIRouter(prefix=_PREFIX)\n"
        "@router.post(path=_ROUTE)\n"
        "def route():\n    pass\n\n"
        "@tool(name=_TOOL_NAME)\n"
        "def tool_impl():\n    pass\n"
    )
    constants = _module_string_constants(tree)
    route_decorator = tree.body[6].decorator_list[0]
    tool_decorator = tree.body[7].decorator_list[0]
    assert isinstance(route_decorator, ast.Call)
    assert _router_prefix(tree, constants) == "/api"
    assert _route_path(route_decorator, constants) == "/api/agent/constant"
    assert _public_tool_name(tool_decorator, "tool_impl", constants) == (
        "neutral-control"
    )


def test_tool_names_resolve_constants_at_decorator_execution() -> None:
    tree = ast.parse(
        'NAME = "terminate_child"\n'
        "@tool(NAME)\n"
        "def implementation():\n    pass\n"
        'NAME = "benign"\n'
    )
    function = tree.body[1]
    assert isinstance(function, ast.FunctionDef)
    constants = _module_strings_at_definition(tree, function)

    assert _module_string_constants(tree)["NAME"] == "benign"
    assert _public_tool_name(
        function.decorator_list[0], function.name, constants
    ) == "terminate_child"


def test_route_prefix_is_bound_to_receiver_and_lexical_scope() -> None:
    tree = ast.parse(
        'read_router = APIRouter(prefix="/api/read")\n'
        'admin_router = APIRouter(prefix="/api/admin")\n'
        '@admin_router.post("/agents")\n'
        "def mutate():\n    pass\n\n"
        "def get_router():\n"
        '    router = APIRouter(prefix="/api/nested")\n'
        '    @router.get("/children")\n'
        "    def children():\n        pass\n"
        "    return router\n"
    )
    assert _route_declarations(tree, {}, {}) == [
        (("POST",), "/api/admin/agents"),
        (("GET",), "/api/nested/children"),
    ]


def test_imported_tool_name_constant_is_resolved_without_importing_code() -> None:
    path = REPO_ROOT / "kestrel_sovereign/features/security/feature.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants = _module_string_constants(tree, path)
    assert constants["SEARCH_TOOL_NAME"] == "security_audit_search"

    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "security_audit_search"
    )
    assert (
        _public_tool_name(function.decorator_list[0], function.name, constants)
        == "security_audit_search"
    )


def test_unresolved_decorator_declarations_fail_closed() -> None:
    tool_decorator = ast.parse(
        "@tool(name=make_name())\ndef tool_impl():\n    pass\n"
    ).body[0].decorator_list[0]
    route_decorator = ast.parse(
        "@router.post(make_path())\ndef route():\n    pass\n"
    ).body[0].decorator_list[0]
    prefix_tree = ast.parse("router = APIRouter(prefix=make_prefix())\n")
    methods_decorator = ast.parse(
        '@router.api_route("/route", methods=make_methods())\n'
        "def route():\n    pass\n"
    ).body[0].decorator_list[0]

    with pytest.raises(AssertionError, match="Unresolved @tool name"):
        _public_tool_name(tool_decorator, "tool_impl")
    with pytest.raises(AssertionError, match="Unresolved route path"):
        _route_path(route_decorator)
    with pytest.raises(AssertionError, match="Unresolved APIRouter prefix"):
        _router_prefix(prefix_tree)
    with pytest.raises(AssertionError, match="Unsupported api_route methods"):
        _route_methods(methods_decorator)


def test_tool_import_aliases_and_route_keyword_unpacking_fail_closed() -> None:
    tree = ast.parse(
        "from kestrel_sdk import tool as expose\n"
        '@expose(name="terminate_child")\n'
        "def implementation():\n    pass\n"
    )
    function = tree.body[1]
    assert isinstance(function, ast.FunctionDef)
    assert _public_tool_name(
        function.decorator_list[0],
        function.name,
        {},
        _tool_decorator_aliases(tree),
    ) == "terminate_child"

    module_alias = ast.parse(
        "import kestrel_sdk as sdk\n"
        "expose = sdk.tool\n"
        '@expose(name="terminate_child")\n'
        "def implementation():\n    pass\n"
    )
    module_function = module_alias.body[2]
    assert isinstance(module_function, ast.FunctionDef)
    assert _public_tool_name(
        module_function.decorator_list[0],
        module_function.name,
        {},
        _tool_decorator_aliases(module_alias),
    ) == "terminate_child"

    route = ast.parse(
        'OPTIONS = {"methods": ["DELETE"]}\n'
        '@router.api_route("/api/agents/{name}", **OPTIONS)\n'
        "def terminate():\n    pass\n"
    ).body[1]
    assert isinstance(route, ast.FunctionDef)
    with pytest.raises(AssertionError, match="keyword unpacking"):
        _route_methods(route.decorator_list[0])

    unpacked_tool = ast.parse(
        "@tool(**OPTIONS)\n"
        "def neutral():\n    pass\n"
    ).body[0]
    assert isinstance(unpacked_tool, ast.FunctionDef)
    with pytest.raises(AssertionError, match="@tool keyword unpacking"):
        _public_tool_name(
            unpacked_tool.decorator_list[0], unpacked_tool.name
        )


def test_tool_decorator_aliases_include_class_and_factory_scopes() -> None:
    tree = ast.parse(
        "class Feature:\n"
        "    expose = tool\n"
        "    @expose(name='class_tool')\n"
        "    def class_impl(self):\n"
        "        pass\n\n"
        "def factory():\n"
        "    publish = tool\n"
        "    @publish(name='factory_tool')\n"
        "    def factory_impl():\n"
        "        pass\n"
    )
    aliases = _tool_decorator_aliases(tree)
    decorated = {
        node.name: _public_tool_name(
            node.decorator_list[0], node.name, {}, aliases
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.decorator_list
    }

    assert decorated == {
        "class_impl": "class_tool",
        "factory_impl": "factory_tool",
    }


def test_tool_decorator_aliases_include_annotated_bindings() -> None:
    tree = ast.parse(
        "from collections.abc import Callable\n"
        "expose: Callable = tool\n"
        "@expose(name='annotated_tool')\n"
        "def implementation():\n"
        "    pass\n"
    )
    function = tree.body[2]
    assert isinstance(function, ast.FunctionDef)

    assert _public_tool_name(
        function.decorator_list[0],
        function.name,
        {},
        _tool_decorator_aliases(tree),
    ) == "annotated_tool"


def test_repository_scans_reuse_parsed_trees_and_analysis_summaries() -> None:
    source_path = REPO_ROOT / "kestrel_sovereign/server.py"
    assert _parsed_module(source_path) is _parsed_module(source_path)
    assert _cached_authority_provenance_lines(
        source_path
    ) is _cached_authority_provenance_lines(source_path)

    function = ast.parse(
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        target.shutdown()\n"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    assert _walk_lexical_scope(function) is _walk_lexical_scope(function)
    assert _identifier_tokens(function) is _identifier_tokens(function)


def test_repository_discovery_results_are_cached_and_immutable() -> None:
    discoveries = (
        _discovered_tool_surfaces,
        _discovered_scheduler_surfaces,
        _discovered_core_signal_source_surfaces,
        _discovered_runtime_generated_tool_surfaces,
        _discovered_builtin_command_surfaces,
        _discovered_core_cli_surfaces,
        _discovered_dynamic_router_surfaces,
        _discovered_http_surfaces,
        _discovered_request_routed_alias_surfaces,
    )
    for discover in discoveries:
        first = discover()
        before_second = discover.cache_info()
        second = discover()
        assert isinstance(first, frozenset)
        assert second is first
        assert discover.cache_info().hits == before_second.hits + 1


def test_provenance_return_summaries_skip_authority_body_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = ast.parse(
        "def derive(request):\n"
        "    chain = request.causation_chain\n"
        "    return bool(chain)\n"
    )
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    def forbidden_scan(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("return summaries invoked authority-body analysis")

    monkeypatch.setitem(
        globals(),
        "_contains_cross_agent_control_call",
        forbidden_scan,
    )
    assert _local_provenance_return_helpers(functions) == {"derive"}


def test_repository_scan_prefilters_modules_without_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = REPO_ROOT / "kestrel_sovereign/__init__.py"
    source = _source_text(source_path).casefold()
    assert not any(
        marker in source
        for marker in ("causation", "orchestrator", "current_chain")
    )
    _cached_authority_provenance_lines.cache_clear()

    def forbidden_analysis(_tree: ast.AST) -> set[int]:
        raise AssertionError("marker-free module reached authority analysis")

    monkeypatch.setitem(
        globals(), "_authority_provenance_lines", forbidden_analysis
    )
    assert _cached_authority_provenance_lines(source_path) == frozenset()


def test_audit_records_remediated_authority_paths_as_enforced() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    for issue in (3134, 3144, 3146, 3147, 3148, 3149):
        row = next(line for line in audit.splitlines() if f"[#{issue}]" in line)
        assert "Enforced by" in row
        assert "Defect:" not in row


def test_a2a_cancellation_delegates_to_issue_3134() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    row = next(
        line for line in audit.splitlines() if line.startswith("| Cancel A2A task |")
    )
    assert "[#3134]" in row
    assert "Defect:" not in row


def test_unverified_spawn_authority_is_recorded_as_a_defect() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    for action in (
        "Create child",
        "List/read child work",
        "Delegate work to child",
        "Terminate/offboard child",
    ):
        row = next(
            line for line in audit.splitlines() if line.startswith(f"| {action} |")
        )
        assert "[#3142]" in row
        assert "Defect" in row


def test_newly_discovered_unenforced_surfaces_link_focused_defects() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    expected = {
        "Read observability summaries/metrics": "[#3215]",
        "Install/remove feature package": "[#3214]",
        "Read consent history/statistics": "[#3229]",
        "Anchor/status/verify audit history": "[#3230]",
    }
    for action, issue in expected.items():
        row = next(
            line for line in audit.splitlines() if line.startswith(f"| {action} |")
        )
        assert issue in row
        assert "Defect:" in row

    tool_defects = {
        "features/consent/feature.py::consent_log": "D-3229",
        "features/consent/feature.py::consent_stats": "D-3229",
        "features/audit_anchor/feature.py::audit_anchor": "D-3230",
        "features/audit_anchor/feature.py::audit_anchor_status": "D-3230",
        "features/audit_anchor/feature.py::audit_verify": "D-3230",
    }
    for surface, defect in tool_defects.items():
        row = next(line for line in audit.splitlines() if surface in line)
        assert defect in row
        assert "shared PostgreSQL" in row


def test_agent_invoked_host_shell_lifecycle_escape_is_recorded_as_3233() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    action_row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Run host lifecycle CLI from agent shell |")
    )
    assert "[#3233]" in action_row
    assert "Defect:" in action_row

    for surface in (
        "features/computer_use/feature.py::shell`",
        "cli.py::kestrel start`",
        "cli.py::kestrel terminate`",
        "cli.py::kestrel restart`",
        "cli.py::kestrel update`",
    ):
        tool_row = next(line for line in audit.splitlines() if surface in line)
        assert "D-3233" in tool_row


def test_multi_agent_deployment_control_is_recorded_as_3223() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    action_row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Deploy/teardown shared agent hosting |")
    )
    tool_row = next(
        line
        for line in audit.splitlines()
        if "features/deploy/feature.py::deploy_agent`" in line
    )
    assert "Sovereign/delegated" in action_row
    assert "[#3223]" in action_row
    assert "Defect:" in action_row
    assert "D-3223" in tool_row
    assert "multi-agent" in tool_row


def test_task_reads_record_3145_durable_principal_enforcement() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    action_row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Read task inbox/status/result |")
    )
    assert "[#3145]" in action_row
    assert "Enforced by" in action_row
    assert "durable recipient principal" in action_row

    for surface in (
        "features/tasks/feature.py::check_task_status",
        "features/tasks/feature.py::get_task_result",
        "features/tasks/feature.py::list_my_tasks",
        "command_handler.py::!tasks",
        "endpoints/agent.py::GET /api/agent/tasks",
        "endpoints/agent.py::GET /api/agent/tasks/{task_id}",
        "endpoints/agent.py::GET /api/agent/tasks/{task_id}/subscribe",
    ):
        row = next(line for line in audit.splitlines() if surface in line)
        assert "#3145" in row
        assert "recipient-owned" in row.casefold()

    for surface in (
        "endpoints/agent.py::POST /api/agent/tasks/{task_id:path}/read",
        "endpoints/agent.py::POST /api/agent/tasks/{task_id:path}/subscribe",
        "endpoints/agent.py::POST /agent/tasks/{task_id:path}/read",
        "endpoints/agent.py::POST /agent/tasks/{task_id:path}/subscribe",
    ):
        row = next(line for line in audit.splitlines() if surface in line)
        assert "#3145" in row
        assert "creator-owned" in row.casefold()
        assert "signed" in row.casefold()


def test_webhook_ambiguity_and_open_unlimited_mode_are_recorded() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| General webhook ingress |")
    )
    assert "[#3216]" in row
    assert "Defect:" in row
    assert 'auth_type="none"' in row
    assert "rate_limit=0" in row
    assert "allow_unauthenticated" in row

    alias_row = next(
        line
        for line in audit.splitlines()
        if "features/webhooks/receiver.py::POST "
        "/api/agents/{selected_agent_name}/webhooks/{webhook_name}" in line
    )
    assert "only when configured" in alias_row
    assert "open/unlimited" in alias_row
    assert "source auth/rate limit enforce" not in alias_row


def test_feature_static_alias_auth_exception_is_recorded() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    section_start = audit.index(
        "## Machine-checked request-routed alias inventory"
    )
    introduction = audit[
        section_start : audit.index("Classification codes:", section_start)
    ]
    row = next(
        line
        for line in audit.splitlines()
        if "server.py::MOUNT "
        "/api/agents/{selected_agent_name}/<dynamic:mount_path>" in line
    )

    assert "agent-feature static-asset aliases" in introduction
    assert "host-authentication-exempt" in row
    assert "| S —" in row


def test_shared_local_model_mutations_are_recorded_as_3221() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    action_row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Manage shared local models |")
    )
    assert "Sovereign/delegated" in action_row
    assert "[#3221]" in action_row

    for surface in (
        "features/model/feature.py::pull_model",
        "features/model/feature.py::cleanup_models",
    ):
        row = next(line for line in audit.splitlines() if surface in line)
        assert "D-3221" in row
        assert "shared" in row.casefold()


def test_routed_rasa_target_mismatch_is_recorded() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    row = next(
        line
        for line in audit.splitlines()
        if "endpoints/rasa_shim.py::POST "
        "/api/agents/{selected_agent_name}/webhooks/rest/webhook" in line
    )
    assert "D-3220" in row
    assert "host-default agent" in row
    assert "target binding" in row


def test_shared_sovereignty_cache_reads_are_recorded_as_3225() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    action_row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Browse sovereignty export cache |")
    )
    assert "Self for agent artifacts" in action_row
    assert "sovereign/delegated" in action_row
    assert "[#3225]" in action_row

    for suffix in (
        "api/sovereignty/files`",
        "api/sovereignty/files/{filename}`",
        "api/sovereignty/files/{filename}/preview`",
    ):
        row = next(
            line
            for line in audit.splitlines()
            if "endpoints/sovereignty.py::GET " in line
            and suffix in line
        )
        assert "D-3225" in row
        assert "shared host export-cache" in row


def test_shared_ipfs_pin_read_is_recorded_as_3226() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    action_row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Inspect local IPFS node and pins |")
    )
    assert "Self for agent pins" in action_row
    assert "sovereign/delegated" in action_row
    assert "[#3226]" in action_row

    for route in (
        "/api/ipfs/status",
        "/api/agents/{selected_agent_name}/api/ipfs/status",
    ):
        row = next(
            line
            for line in audit.splitlines()
            if f"endpoints/models.py::GET {route}`" in line
        )
        assert "D-3226" in row
        assert "shared IPFS daemon" in row
        assert "recursive pins" in row


def test_canonical_phoenix_asset_redirects_are_inventoried() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    for method in ("GET", "HEAD"):
        row = next(
            line
            for line in audit.splitlines()
            if line.startswith(
                f"| `kestrel_sovereign/server.py::{method} /assets/{{path:path}}`"
            )
        )
        assert "host-wide Phoenix asset proxy" in row
        assert "no agent relation authority" in row


def test_mixed_model_catalog_and_layered_key_reads_are_recorded() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    key_action = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Inspect layered key availability |")
    )
    model_action = next(
        line
        for line in audit.splitlines()
        if line.startswith("| List available models |")
    )
    assert "authenticated user's BYOK" in key_action
    assert "platform-global" in key_action
    assert "process-wide shared model catalog" in model_action

    for route in (
        "/api/keys/available-sources",
        "/api/agents/{selected_agent_name}/api/keys/available-sources",
    ):
        row = next(
            line
            for line in audit.splitlines()
            if f"endpoints/models.py::GET {route}`" in line
        )
        assert "A/U/H" in row
        assert "authenticated-user" in row
        assert "platform-global" in row

    for route in (
        "/api/models",
        "/api/agents/{selected_agent_name}/api/models",
    ):
        row = next(
            line
            for line in audit.splitlines()
            if f"endpoints/models.py::GET {route}`" in line
        )
        assert "A/H" in row
        assert "process-wide shared model catalog" in row


@lru_cache(maxsize=None)
def _identifier_tokens(node: ast.AST) -> frozenset[str]:
    tokens: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            tokens.add(child.id.casefold())
        elif isinstance(child, ast.Attribute):
            tokens.add(child.attr.casefold())
            tokens.add(ast.unparse(child).casefold())
        elif isinstance(child, ast.Subscript):
            tokens.add(ast.unparse(child).casefold())
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            tokens.add(child.value.casefold())
    return frozenset(tokens)


def _binding_target_names(target: ast.AST) -> set[str]:
    """Return normalized names bound by an assignment or iteration target."""

    if isinstance(target, ast.Name):
        return {target.id.casefold()}
    if isinstance(target, ast.Starred):
        return _binding_target_names(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        return {
            name
            for element in target.elts
            for name in _binding_target_names(element)
        }
    if isinstance(target, (ast.Attribute, ast.Subscript)):
        return {ast.unparse(target).casefold()}
    return set()


_CROSS_AGENT_STATE_COLLECTIONS = {
    "_agents",
    "agents",
    "_children",
    "children",
    "_descendants",
    "descendants",
    "_peers",
    "peers",
    "agent_registry",
    "child_registry",
    "peer_registry",
}


def _is_cross_agent_state_collection_name(name: str) -> bool:
    """Recognize qualified agent registries without matching every registry."""

    normalized = name.casefold().strip("_")
    if normalized in _CROSS_AGENT_STATE_COLLECTIONS:
        return True
    words = set(normalized.split("_"))
    plural_subjects = {"agents", "children", "descendants", "peers"}
    singular_subjects = {"agent", "child", "descendant", "peer"}
    collection_labels = {
        "by",
        "collection",
        "index",
        "map",
        "mapping",
        "pool",
        "registries",
        "registry",
        "store",
    }
    return bool(words.intersection(plural_subjects)) or bool(
        words.intersection(singular_subjects)
        and words.intersection(collection_labels)
    )


_CROSS_AGENT_STATE_OBJECT_LABELS = {
    "agent",
    "child",
    "descendant",
    "peer",
}


_CROSS_AGENT_LIFECYCLE_ACTIONS = frozenset(
    {
        "cancel",
        "close",
        "delete",
        "destroy",
        "disable",
        "enable",
        "hold",
        "interrupt",
        "kill",
        "offboard",
        "pause",
        "remove",
        "reset",
        "restart",
        "resume",
        "shutdown",
        "start",
        "stop",
        "terminate",
        "withdraw",
    }
)


def _cross_agent_state_object_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    initial_aliases: set[str] | None = None,
) -> set[str]:
    """Resolve local bindings that select an agent-shaped mutable object."""

    def semantic_name(name: str) -> bool:
        words = set(name.casefold().split("_"))
        return bool(words.intersection(_CROSS_AGENT_STATE_OBJECT_LABELS)) and not bool(
            words.intersection({"count", "did", "id", "ids", "name", "names"})
        )

    def reference_sources(node: ast.AST) -> set[str]:
        while isinstance(node, (ast.Await, ast.Expr)):
            node = node.value
        if isinstance(node, ast.Name):
            return {node.id.casefold()}
        if isinstance(node, ast.Attribute):
            return {node.attr.casefold()}
        if isinstance(node, ast.Subscript):
            return _control_reference_sources(node)
        if isinstance(node, ast.IfExp):
            return reference_sources(node.body) | reference_sources(node.orelse)
        if isinstance(node, ast.Call):
            call_name = _call_name(node).casefold()
            return {call_name} if semantic_name(call_name) else set()
        return set()

    aliases: set[str] = set(initial_aliases or ())
    parameters = [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ]
    if function.args.vararg is not None:
        parameters.append(function.args.vararg)
    if function.args.kwarg is not None:
        parameters.append(function.args.kwarg)
    for parameter in parameters:
        annotation_tokens = (
            _identifier_tokens(parameter.annotation)
            if parameter.annotation is not None
            else frozenset()
        )
        if semantic_name(parameter.arg) or any(
            any(label in token for label in _CROSS_AGENT_STATE_OBJECT_LABELS)
            for token in annotation_tokens
        ):
            aliases.add(parameter.arg.casefold())

    assignments: list[tuple[set[str], set[str]]] = []
    for node in _walk_lexical_scope(function):
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
            if any(
                any(label in token for label in _CROSS_AGENT_STATE_OBJECT_LABELS)
                for token in _identifier_tokens(node.annotation)
            ):
                aliases.update(
                    name
                    for target in targets
                    for name in _binding_target_names(target)
                )
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
            value = node.value
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
            value = node.iter
        if value is None:
            continue
        target_names = {
            name
            for target in targets
            for name in _binding_target_names(target)
        }
        aliases.update(name for name in target_names if semantic_name(name))
        assignments.append((target_names, reference_sources(value)))

    changed = True
    while changed:
        changed = False
        for targets, sources in assignments:
            selects_agent_object = bool(sources.intersection(aliases)) or any(
                _is_cross_agent_state_collection_name(source)
                or semantic_name(source)
                or source.startswith(("get_agent", "resolve_agent", "select_agent"))
                for source in sources
            )
            if selects_agent_object:
                new_aliases = targets - aliases
                if new_aliases:
                    aliases.update(new_aliases)
                    changed = True
    return aliases


def _is_cross_agent_state_mutation_target(
    node: ast.AST,
    state_object_aliases: frozenset[str] | set[str] | None = None,
) -> bool:
    """Whether a write directly selects an agent/child/peer registry entry."""

    collection_references = {
        child.id.casefold()
        if isinstance(child, ast.Name)
        else child.attr.casefold()
        for child in ast.walk(node)
        if isinstance(child, (ast.Name, ast.Attribute))
        and _is_cross_agent_state_collection_name(
            child.id if isinstance(child, ast.Name) else child.attr
        )
    }
    is_collection = (
        isinstance(node, ast.Name)
        and _is_cross_agent_state_collection_name(node.id)
    ) or (
        isinstance(node, ast.Attribute)
        and _is_cross_agent_state_collection_name(node.attr)
    )
    if collection_references and (
        is_collection
        or any(isinstance(child, ast.Subscript) for child in ast.walk(node))
    ):
        return True
    aliases = set(state_object_aliases or ())
    return isinstance(node, (ast.Attribute, ast.Subscript)) and bool(
        _identifier_tokens(node.value).intersection(aliases)
    )


def _is_cross_agent_state_object_reference(
    node: ast.AST,
    state_object_aliases: frozenset[str] | set[str] | None = None,
) -> bool:
    """Whether an expression resolves to a tracked agent-shaped object."""

    return bool(
        _identifier_tokens(node).intersection(state_object_aliases or set())
    ) or _is_cross_agent_state_mutation_target(node, state_object_aliases)


def _is_cross_agent_state_mutation_call(
    call: ast.Call,
    state_object_aliases: frozenset[str] | set[str] | None = None,
) -> bool:
    """Whether a mutator call writes through an agent registry or object."""

    attribute_mutation = (
        isinstance(call.func, ast.Attribute)
        and call.func.attr.casefold()
        in {
            "__delitem__",
            "__setitem__",
            "add",
            "append",
            "clear",
            "discard",
            "extend",
            "insert",
            "pop",
            "popitem",
            "remove",
            "setdefault",
            "update",
        }
        and _is_cross_agent_state_mutation_target(
            call.func.value, state_object_aliases
        )
    )
    named_mutation = (
        _call_name(call).casefold() in {"delattr", "setattr"}
        and bool(call.args)
        and _is_cross_agent_state_object_reference(
            call.args[0], state_object_aliases
        )
    )
    lifecycle_mutation = (
        isinstance(call.func, ast.Attribute)
        and call.func.attr.casefold()
        in _CROSS_AGENT_LIFECYCLE_ACTIONS
        and _is_cross_agent_state_object_reference(
            call.func.value, state_object_aliases
        )
    )
    return attribute_mutation or named_mutation or lifecycle_mutation


@lru_cache(maxsize=None)
def _walk_lexical_scope(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    """Walk one function body without borrowing nested-scope semantics."""

    nodes: list[ast.AST] = []

    class ScopeVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            return

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

        def generic_visit(self, node: ast.AST) -> None:
            nodes.append(node)
            super().generic_visit(node)

    visitor = ScopeVisitor()
    for statement in function.body:
        visitor.visit(statement)
    return tuple(nodes)


def _block_guaranteed_exits(statements: list[ast.stmt]) -> bool:
    """Whether a simple statement block cannot reach its following sibling."""

    if not statements:
        return False
    terminal = statements[-1]
    if isinstance(terminal, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
        return True
    if isinstance(terminal, ast.If):
        return _block_guaranteed_exits(
            terminal.body
        ) and _block_guaranteed_exits(terminal.orelse)
    if isinstance(terminal, (ast.With, ast.AsyncWith)):
        return _block_guaranteed_exits(terminal.body)
    if isinstance(terminal, (ast.Try, ast.TryStar)):
        if _block_guaranteed_exits(terminal.finalbody):
            return True
        normal_path_exits = _block_guaranteed_exits(
            terminal.body
        ) or _block_guaranteed_exits(terminal.orelse)
        return normal_path_exits and all(
            _block_guaranteed_exits(handler.body)
            for handler in terminal.handlers
        )
    if isinstance(terminal, ast.Match):
        has_catch_all = any(
            case.guard is None
            and isinstance(case.pattern, ast.MatchAs)
            and case.pattern.pattern is None
            for case in terminal.cases
        )
        return has_catch_all and all(
            _block_guaranteed_exits(case.body) for case in terminal.cases
        )
    return False


def _contains_current_loop_break(statements: list[ast.stmt]) -> bool:
    """Whether a block can break its owning loop, excluding nested loops."""

    class BreakVisitor(ast.NodeVisitor):
        found = False

        def visit_Break(self, node: ast.Break) -> None:  # noqa: N802
            self.found = True

        def visit_For(self, node: ast.For) -> None:  # noqa: N802
            return

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
            return

        def visit_While(self, node: ast.While) -> None:  # noqa: N802
            return

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            return

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

    visitor = BreakVisitor()
    for statement in statements:
        visitor.visit(statement)
    return visitor.found


def _loop_else_guards_continuation(
    statement: ast.For | ast.AsyncFor | ast.While,
) -> bool:
    """Whether breaking versus exhausting the loop gates later siblings."""

    return (
        _block_guaranteed_exits(statement.orelse)
        and _contains_current_loop_break(statement.body)
    )


def _child_statement_blocks(statement: ast.stmt) -> list[list[ast.stmt]]:
    """Return same-scope child blocks while excluding nested definitions."""

    if isinstance(statement, (ast.For, ast.AsyncFor, ast.If, ast.While)):
        return [statement.body, statement.orelse]
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return [statement.body]
    if isinstance(statement, (ast.Try, ast.TryStar)):
        return [
            statement.body,
            statement.orelse,
            statement.finalbody,
            *(handler.body for handler in statement.handlers),
        ]
    if isinstance(statement, ast.Match):
        return [case.body for case in statement.cases]
    return []


def _has_provenance_token(node: ast.AST, aliases: set[str] | None = None) -> bool:
    tokens = set(_identifier_tokens(node))
    # The bare string ``"ORCHESTRATOR"`` is also a provider role label in
    # prompts and logs. Treat it as metadata only when it is used as a lookup
    # key; names/attributes called ``orchestrator`` remain provenance-bearing.
    if "orchestrator" in tokens:
        semantic_orchestrator = any(
            (
                isinstance(child, ast.Name)
                and child.id.casefold() == "orchestrator"
            )
            or (
                isinstance(child, ast.Attribute)
                and child.attr.casefold() == "orchestrator"
            )
            or (
                isinstance(child, ast.Subscript)
                and isinstance(child.slice, ast.Constant)
                and isinstance(child.slice.value, str)
                and child.slice.value.casefold() == "orchestrator"
            )
            or (
                isinstance(child, ast.Call)
                and _call_name(child) == "get"
                and bool(child.args)
                and isinstance(child.args[0], ast.Constant)
                and isinstance(child.args[0].value, str)
                and child.args[0].value.casefold() == "orchestrator"
            )
            for child in ast.walk(node)
        )
        if not semantic_orchestrator:
            tokens.discard("orchestrator")
    # Merely retrieving a chain-provider callable is transport wiring, not a
    # provenance value. The callable becomes provenance-bearing only when it
    # is invoked (handled by ``_is_provenance_accessor_call``). Keep ordinary
    # ``getattr(request, "parent_causation_chain")`` value reads tainted.
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and _call_name(child).casefold() == "getattr"
            and len(child.args) > 1
        ):
            attribute = _resolved_string(child.args[1])
            if attribute is not None and "provide" in attribute.casefold().split("_"):
                tokens.discard(attribute.casefold())
    provenance_tokens = {"orchestrator", "kestrel.orchestrator"}
    return any(
        token in provenance_tokens
        or (
            "orchestrator" in token.split("_")
            and not set(token.split("_")).intersection(
                {"feature", "model", "name", "prompt", "role", "slug", "tool"}
            )
        )
        or "causation" in token.split("_")
        or token == "causationframe"
        or token in (aliases or set())
        for token in tokens
    )


def _is_provenance_accessor_call(node: ast.AST) -> bool:
    """Whether an expression actually invokes a canonical chain accessor."""

    return any(
        isinstance(child, ast.Call)
        and _call_name(child).casefold().strip("_").endswith(
            PROVENANCE_ACCESSOR_SUFFIXES
        )
        for child in ast.walk(node)
    )


def _is_provenance_accessor_reference(node: ast.AST) -> bool:
    """Whether a value preserves a canonical chain-accessor callable."""

    return isinstance(node, (ast.Name, ast.Attribute)) and _call_name(
        ast.Call(func=node, args=[], keywords=[])
    ).casefold().strip("_").endswith(PROVENANCE_ACCESSOR_SUFFIXES)


def _is_provenance_accessor_getattr_reference(node: ast.AST) -> bool:
    """Whether a static ``getattr`` retrieves a canonical chain accessor."""

    if not isinstance(node, ast.Call) or _call_name(node).casefold() != "getattr":
        return False
    attribute = (
        node.args[1]
        if len(node.args) > 1
        else next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg in {"attr", "name"}
            ),
            None,
        )
    )
    resolved = _resolved_string(attribute) if attribute is not None else None
    return resolved is not None and resolved.casefold().strip("_").endswith(
        PROVENANCE_ACCESSOR_SUFFIXES
    )


def _function_provenance_accessor_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    inherited_aliases: set[str] | None = None,
) -> set[str]:
    """Return callable aliases whose invocation reads provenance metadata."""

    aliases = set(inherited_aliases or ())
    bindings: list[tuple[set[str], ast.AST]] = []
    for node in _walk_lexical_scope(function):
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        target_names = {
            name
            for target in targets
            for name in _binding_target_names(target)
        }
        target_names.update(
            target.attr.casefold()
            for target in targets
            if isinstance(target, ast.Attribute)
        )
        if target_names:
            bindings.append((target_names, value))

    changed = True
    while changed:
        changed = False
        for target_names, value in bindings:
            preserves_accessor = (
                _is_provenance_accessor_reference(value)
                or _is_provenance_accessor_getattr_reference(value)
                or (
                    isinstance(value, (ast.Name, ast.Attribute))
                    and bool(_identifier_tokens(value).intersection(aliases))
                )
            )
            if preserves_accessor:
                new_aliases = target_names - aliases
                if new_aliases:
                    aliases.update(new_aliases)
                    changed = True
    return aliases


def _has_provenance_value(
    node: ast.AST,
    aliases: set[str] | None = None,
    provenance_return_helpers: set[str] | None = None,
) -> bool:
    """Whether an expression reads provenance directly or via a local helper."""

    return (
        _has_provenance_token(node, aliases)
        or _is_provenance_accessor_call(node)
        or any(
            isinstance(child, ast.Call)
            and _call_name(child).casefold()
            in (provenance_return_helpers or set())
            for child in ast.walk(node)
        )
    )


def _is_non_authority_context_manager(expression: ast.AST) -> bool:
    """Return bounded contexts that cannot grant permission to their body."""

    if not isinstance(expression, ast.Call):
        return False
    return (
        isinstance(expression.func, ast.Attribute)
        and isinstance(expression.func.value, ast.Name)
        and expression.func.value.id == "asyncio"
        and expression.func.attr in {"timeout", "timeout_at"}
    )


_MUTABLE_CONTAINER_WRITE_ARGUMENTS = {
    "__setitem__": (1,),
    "add": (0,),
    "append": (0,),
    "extend": (0,),
    "insert": (1,),
    "setdefault": (1,),
    "update": (0,),
}


def _reference_binding_names(node: ast.AST) -> set[str]:
    """Return exact and callable-selector names for one stored reference."""

    names = _binding_target_names(node)
    if isinstance(node, ast.Attribute):
        names.add(node.attr.casefold())
    elif isinstance(node, ast.Subscript):
        names.update(_control_reference_sources(node))
    return names


def _mutable_container_write(
    node: ast.Call,
) -> tuple[set[str], ast.AST] | None:
    """Return the container binding and values written by a mutator call."""

    receiver: ast.AST | None = None
    value_nodes: list[ast.AST] = []
    attribute_name: str | None = None
    if isinstance(node.func, ast.Attribute):
        method = node.func.attr.casefold()
        indexes = _MUTABLE_CONTAINER_WRITE_ARGUMENTS.get(method)
        if indexes is None:
            return None
        receiver = node.func.value
        value_nodes.extend(
            node.args[index] for index in indexes if index < len(node.args)
        )
        value_nodes.extend(keyword.value for keyword in node.keywords)
    elif _call_name(node).casefold() == "setattr" and len(node.args) >= 3:
        receiver = node.args[0]
        value_nodes.append(node.args[2])
        attribute_name = _resolved_string(node.args[1])
    else:
        return None

    names = _reference_binding_names(receiver)
    if attribute_name is not None:
        names.add(attribute_name.casefold())
    if not names or not value_nodes:
        return None
    value: ast.AST = (
        value_nodes[0]
        if len(value_nodes) == 1
        else ast.Tuple(elts=value_nodes, ctx=ast.Load())
    )
    return names, value


def _mutable_container_alias_snapshots(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[ast.AST, frozenset[str]]:
    """Capture live mutable-container aliases at each individual write."""

    groups: dict[str, set[str]] = {}
    snapshots: dict[ast.AST, frozenset[str]] = {}

    def detach(name: str) -> None:
        group = groups.pop(name, None)
        if group is not None:
            group.discard(name)

    def bind(target_names: set[str], source_names: set[str] | None) -> None:
        for target_name in target_names:
            detach(target_name)
        if not target_names:
            return
        source_group = next(
            (
                groups[source_name]
                for source_name in source_names or set()
                if source_name in groups
            ),
            None,
        )
        group = source_group if source_group is not None else set()
        group.update(target_names)
        for member in group:
            groups[member] = group

    def live_aliases(names: set[str]) -> frozenset[str]:
        expanded = set(names)
        for name in names:
            expanded.update(groups.get(name, ()))
        return frozenset(expanded)

    for node in _walk_lexical_scope(function):
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
            value = node.value

        if targets:
            target_names = {
                name
                for target in targets
                if not isinstance(target, ast.Subscript)
                for name in _reference_binding_names(target)
            }
            source_names = (
                _reference_binding_names(value)
                if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript))
                else set()
            )
            creates_container = isinstance(
                value, (ast.Dict, ast.List, ast.Set)
            ) or (
                isinstance(value, ast.Call)
                and _call_name(value).casefold()
                in {"defaultdict", "deque", "dict", "list", "set"}
            )
            if creates_container or any(
                source_name in groups for source_name in source_names
            ):
                bind(target_names, source_names)
            else:
                for target_name in target_names:
                    detach(target_name)

            for target in targets:
                if isinstance(target, ast.Subscript):
                    receiver_names = _reference_binding_names(target.value)
                    if not any(name in groups for name in receiver_names):
                        bind(receiver_names, None)
            snapshots[node] = live_aliases({
                name
                for target in targets
                for name in _reference_binding_names(target)
            })

        if isinstance(node, ast.Call):
            mutation = _mutable_container_write(node)
            if mutation is not None:
                receiver_names, _written_value = mutation
                if not any(name in groups for name in receiver_names):
                    bind(receiver_names, None)
                snapshots[node] = live_aliases(receiver_names)

    return snapshots


def _cross_agent_control_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    control_helpers: set[str] | None = None,
) -> set[str]:
    """Resolve local names that reference cross-agent control callables."""

    assignments: list[tuple[str, str]] = []
    container_aliases = _mutable_container_alias_snapshots(function)

    def expand_container_aliases(
        names: set[str], node: ast.AST
    ) -> set[str]:
        return names | set(container_aliases.get(node, ()))

    positional_parameters = [
        *function.args.posonlyargs,
        *function.args.args,
    ]
    positional_defaults = (
        zip(
            positional_parameters[-len(function.args.defaults):],
            function.args.defaults,
        )
        if function.args.defaults
        else ()
    )
    default_bindings = [
        *positional_defaults,
        *(
            (parameter, default)
            for parameter, default in zip(
                function.args.kwonlyargs,
                function.args.kw_defaults,
            )
            if default is not None
        ),
    ]
    for parameter, default in default_bindings:
        assignments.extend(
            (parameter.arg.casefold(), source)
            for source in _control_reference_sources(default)
        )

    for node in _walk_lexical_scope(function):
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.Call):
            mutation = _mutable_container_write(node)
            if mutation is not None:
                target_names, value = mutation
                target_names = expand_container_aliases(target_names, node)
                sources = _control_reference_sources(value)
                assignments.extend(
                    (target_name, source)
                    for target_name in target_names
                    for source in sources
                )
            continue
        if value is None:
            continue
        sources = _control_reference_sources(value)
        if not sources:
            continue
        for target in targets:
            target_names = expand_container_aliases(
                _reference_binding_names(target), node
            )
            assignments.extend(
                (target_name, source)
                for target_name in target_names
                for source in sources
            )

    aliases: set[str] = set(control_helpers or ())
    changed = True
    while changed:
        changed = False
        for target, source in assignments:
            if (
                _is_unambiguous_control_token(source) or source in aliases
            ) and target not in aliases:
                aliases.add(target)
                changed = True
    return aliases


def _invoked_lambda_bodies(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.Lambda, ...]:
    """Return locally bound or immediate lambdas that this scope invokes."""

    scope_nodes = _walk_lexical_scope(function)
    invoked_names = {
        _call_name(node).casefold()
        for node in scope_nodes
        if isinstance(node, ast.Call)
    }
    invoked: list[ast.Lambda] = [
        node.func
        for node in scope_nodes
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Lambda)
    ]
    lambda_bindings: dict[str, ast.Lambda] = {}
    callable_aliases: list[tuple[str, str]] = []
    for node in scope_nodes:
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        target_names = {
            target.id.casefold()
            if isinstance(target, ast.Name)
            else target.attr.casefold()
            if isinstance(target, ast.Attribute)
            else ""
            for target in targets
        }
        target_names.discard("")
        if isinstance(value, ast.Lambda):
            for target_name in target_names:
                lambda_bindings[target_name] = value
            continue
        source_name = (
            value.id.casefold()
            if isinstance(value, ast.Name)
            else value.attr.casefold()
            if isinstance(value, ast.Attribute)
            else ""
        )
        if source_name:
            callable_aliases.extend(
                (target_name, source_name) for target_name in target_names
            )
    changed = True
    while changed:
        changed = False
        for target_name, source_name in callable_aliases:
            if target_name in invoked_names and source_name not in invoked_names:
                invoked_names.add(source_name)
                changed = True
    invoked.extend(
        lambda_node
        for name, lambda_node in lambda_bindings.items()
        if name in invoked_names
    )
    return tuple(invoked)


def _control_reference_sources(node: ast.AST) -> set[str]:
    """Return callable names preserved by static control factories.

    A control remains a control when code obtains the bound method through a
    static ``getattr`` or wraps it in ``functools.partial``.  Follow those
    standard callable-producing forms, plus the collection/selector aliases
    already supported by the scanner.  Dynamic attribute strings deliberately
    resolve to no source: the repository contract cannot prove what they name.
    """

    while isinstance(node, (ast.Await, ast.Expr)):
        node = node.value
    if isinstance(node, ast.Name):
        return {node.id.casefold()}
    if isinstance(node, ast.Attribute):
        return {node.attr.casefold()}
    if isinstance(node, ast.Subscript):
        receiver = node.value
        return (
            {receiver.id.casefold()}
            if isinstance(receiver, ast.Name)
            else {receiver.attr.casefold()}
            if isinstance(receiver, ast.Attribute)
            else set()
        )
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        elements = (
            [*node.keys, *node.values]
            if isinstance(node, ast.Dict)
            else list(node.elts)
        )
        return {
            source
            for element in elements
            if element is not None
            for source in _control_reference_sources(element)
        }
    if isinstance(node, ast.Lambda):
        return (
            {"lambda_control"}
            if any(
                isinstance(child, ast.Call)
                and _is_unambiguous_control_sink(child)
                for child in ast.walk(node.body)
            )
            else set()
        )
    if not isinstance(node, ast.Call):
        return set()

    factory_name = _call_name(node).casefold()
    if factory_name == "getattr":
        attribute = (
            node.args[1]
            if len(node.args) > 1
            else next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in {"name", "attr"}
                ),
                None,
            )
        )
        if attribute is None:
            return set()
        resolved = _resolved_string(attribute)
        # This callable is invoked immediately. If its runtime-selected
        # attribute cannot be resolved, the audit cannot prove it is not a
        # lifecycle/control method, so retain a control-shaped fail-closed
        # sentinel instead of silently dropping the call.
        return (
            {resolved.casefold()}
            if resolved is not None
            else {"dynamic_control_attribute"}
        )
    if factory_name == "get" and isinstance(node.func, ast.Attribute):
        sources = _control_reference_sources(node.func.value)
        if node.args:
            key = _resolved_string(node.args[0])
            if key is not None:
                sources.add(key.casefold())
        for default in [
            *node.args[1:],
            *(keyword.value for keyword in node.keywords),
        ]:
            sources.update(_control_reference_sources(default))
        return sources
    if factory_name in {"partial", "partialmethod"} and node.args:
        return _control_reference_sources(node.args[0])
    return set()


def _is_cross_agent_control_call(
    node: ast.Call,
    control_aliases: set[str] | None = None,
) -> bool:
    """Whether ``node`` invokes a known control or a local alias of one."""

    call_name = _call_name(node).casefold()
    # ``_call_name`` covers ordinary names and attributes.  For a mapping or
    # sequence-selected callable, inspect only the selector, not its receiver:
    # ``handlers["terminate_child"]`` is a control sink, while a neutral call
    # on ``task_manager`` must not become one merely because the receiver name
    # contains the broad inventory term ``task``.
    callable_tokens = set(_control_reference_sources(node.func))
    if isinstance(node.func, ast.Subscript):
        callable_tokens.update(_identifier_tokens(node.func.slice))
    higher_order_callable_indexes = {
        "apply_async": (0,),
        "call_at": (1,),
        "call_later": (1,),
        "call_soon": (0,),
        "call_soon_threadsafe": (0,),
        "filter": (0,),
        "map": (0,),
        "run_in_executor": (1,),
        "submit": (0,),
        "to_thread": (0,),
    }
    higher_order_sources = {
        source
        for index in higher_order_callable_indexes.get(call_name, ())
        if index < len(node.args)
        for source in _control_reference_sources(node.args[index])
    }
    return (
        _is_unambiguous_control_sink(node, control_aliases)
        or call_name in (control_aliases or set())
        or any(
            _is_unambiguous_control_token(source)
            or source in (control_aliases or set())
            for source in higher_order_sources
        )
        or any(
            _is_unambiguous_control_token(token)
            or token in (control_aliases or set())
            for token in callable_tokens
        )
    )


def _is_unambiguous_control_token(token: str) -> bool:
    """Recognize authority sinks without broad inventory-only vocabulary."""

    token = token.casefold()
    control_actions = _CROSS_AGENT_LIFECYCLE_ACTIONS | {
        "create",
        "delegate",
        "deploy",
        "invoke",
        "list",
        "read",
        "send",
        "spawn",
        "subscribe",
        "teardown",
        "verify",
    }
    agent_subjects = (
        "a2a",
        "agent",
        "child",
        "descendant",
        "fleet",
        "host",
        "parent",
        "peer",
    )
    return "control" in token.split("_") or token in {
        "cancel_task",
        "delegate",
        "dynamic_control_attribute",
        "hold",
        "kill_process",
        "offboard",
        "restart",
        "shutdown",
        "spawn",
        "stop",
        "terminate",
        "withdraw",
    } or (
        any(action in token for action in control_actions)
        and any(subject in token for subject in agent_subjects)
    )


def _is_unambiguous_control_sink(
    call: ast.Call,
    known_helpers: set[str] | None = None,
) -> bool:
    """Recognize lifecycle calls without broad terms such as local ``task``."""

    call_name = _call_name(call).casefold()
    if call_name in (known_helpers or set()):
        return True
    selector_tokens = set(_control_reference_sources(call.func))
    selector_tokens.update(
        _identifier_tokens(call.func.slice)
        if isinstance(call.func, ast.Subscript)
        else ()
    )
    return any(
        _is_unambiguous_control_token(token)
        for token in {call_name, *selector_tokens}
    )


def _provenance_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    provenance_return_helpers: set[str] | None = None,
    control_helpers: set[str] | None = None,
    initial_aliases: set[str] | None = None,
    *,
    authority_analysis: bool = True,
    state_object_aliases: set[str] | None = None,
    provenance_accessor_aliases: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Resolve provenance aliases and provenance-selected control arguments."""

    def is_provenance_derived(value: ast.AST, aliases: set[str]) -> bool:
        # Awaiting a helper that merely *receives* causation metadata does not
        # make its result provenance (for example ``fired = await
        # emit(..., causation_chain=chain)``).  Follow only direct accessors and
        # transformations whose result still represents or measures the
        # metadata itself. Predicate-shaped helpers are included because a
        # value such as ``has_valid_causation(chain)`` is plainly intended for
        # a later authority decision; arbitrary side-effect call results are
        # not, even when the call carries lineage for propagation.
        while isinstance(value, (ast.Await, ast.Expr)):
            value = value.value
        calls_known_helper = any(
            isinstance(node, ast.Call)
            and _call_name(node).casefold()
            in (provenance_return_helpers or set())
            for node in ast.walk(value)
        )
        if isinstance(
            value,
            (
                ast.Name,
                ast.Attribute,
                ast.Subscript,
                ast.Constant,
                ast.List,
                ast.Tuple,
                ast.Set,
                ast.Dict,
            ),
        ):
            return _is_provenance_accessor_reference(
                value
            ) or _has_provenance_token(value, aliases)
        if (
            isinstance(value, ast.Call)
            and _call_name(value) in PROVENANCE_TRANSFORM_CALLS
        ):
            return _has_provenance_token(value, aliases) or calls_known_helper
        if isinstance(value, ast.Call):
            call_name = _call_name(value).casefold()
            if _is_provenance_accessor_call(value):
                return True
            if call_name in (provenance_return_helpers or set()):
                return True
            if call_name.startswith(("can_", "has_", "is_", "may_")) or (
                _is_permission_name(call_name)
            ):
                return _has_provenance_token(value, aliases) or calls_known_helper
        # Normalization does not erase the authority input. Comparisons,
        # arithmetic, comprehensions, and conditional expressions remain
        # provenance-derived when a later gate consumes their result.
        if isinstance(
            value,
            (
                ast.BinOp,
                ast.BoolOp,
                ast.Compare,
                ast.DictComp,
                ast.GeneratorExp,
                ast.IfExp,
                ast.ListComp,
                ast.SetComp,
                ast.UnaryOp,
            ),
        ):
            return _has_provenance_token(value, aliases) or calls_known_helper
        return False

    aliases: set[str] = set(initial_aliases or ())
    positional_parameters = [
        *function.args.posonlyargs,
        *function.args.args,
    ]
    default_bindings = [
        *zip(
            positional_parameters[-len(function.args.defaults) :],
            function.args.defaults,
        ),
        *(
            (parameter, default)
            for parameter, default in zip(
                function.args.kwonlyargs,
                function.args.kw_defaults,
            )
            if default is not None
        ),
    ]
    for parameter, default in default_bindings:
        if is_provenance_derived(default, aliases):
            aliases.add(parameter.arg.casefold())
    scope_nodes = _walk_lexical_scope(function)
    decision_provenance_helpers = set(provenance_return_helpers or ()) | set(
        provenance_accessor_aliases or ()
    )
    control_aliases = (
        _cross_agent_control_aliases(function, control_helpers)
        if authority_analysis
        else set()
    )
    control_argument_names = (
        {
            token
            for node in scope_nodes
            if isinstance(node, ast.Call)
            and _is_cross_agent_control_call(node, control_aliases)
            for argument in [
                *node.args,
                *(keyword.value for keyword in node.keywords),
            ]
            for token in _identifier_tokens(argument)
        }
        if authority_analysis
        else set()
    )
    selector_control_argument_names = (
        {
            token
            for node in scope_nodes
            if isinstance(node, ast.Call)
            and _is_unambiguous_control_sink(node, control_helpers)
            for argument in [
                *node.args,
                *(keyword.value for keyword in node.keywords),
            ]
            for token in _identifier_tokens(argument)
        }
        if authority_analysis
        else set()
    )
    assignments: list[tuple[set[str], ast.AST, ast.AST]] = []
    container_aliases = _mutable_container_alias_snapshots(function)

    def expand_container_aliases(
        names: set[str], node: ast.AST
    ) -> set[str]:
        return names | set(container_aliases.get(node, ()))

    for node in scope_nodes:
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
            value = node.value
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
            value = node.iter
        elif isinstance(node, ast.Call):
            mutation = _mutable_container_write(node)
            if mutation is not None:
                names, value = mutation
                assignments.append(
                    (expand_container_aliases(names, node), value, node)
                )
            continue
        if value is None:
            continue
        names = expand_container_aliases(
            {
                name
                for target in targets
                for name in _binding_target_names(target)
            },
            node,
        )
        for target in targets:
            names.update(
                expand_container_aliases(
                    _reference_binding_names(target), node
                )
            )
        # A provenance-selected member of a container passed to a control call
        # can choose the target through ``**kwargs`` or a structured argument.
        # Taint that container only when it is actually an argument at this
        # control boundary; do not broadly taint every ``state``/``self`` base.
        names.update(
            control_name
            for control_name in control_argument_names
            if any(
                name.startswith(f"{control_name}.")
                or name.startswith(f"{control_name}[")
                for name in names
            )
        )
        if names:
            assignments.append((names, value, node))

    # An arbitrary helper may compute an authority decision without advertising
    # that fact in its name. Mark assignment targets that later guard a control,
    # then walk local assignment dependencies backwards. Merely passing
    # causation metadata to a helper remains propagation unless its result
    # reaches such a control gate.
    assignment_names = {
        name for names, _value, _node in assignments for name in names
    }
    authority_target_names = assignment_names.intersection(
        control_argument_names
    )
    authority_decision_names: set[str] = set(authority_target_names)
    for node in scope_nodes:
        if not authority_analysis:
            break
        guarded: list[ast.AST] = []
        decision_expression: ast.AST | None = None
        if isinstance(node, (ast.If, ast.While)):
            guarded = [*node.body, *node.orelse]
            decision_expression = node.test
        elif isinstance(node, ast.IfExp):
            guarded = [node.body, node.orelse]
            decision_expression = node.test
        elif isinstance(node, ast.BoolOp):
            guarded = list(node.values)
            decision_expression = node
        elif isinstance(node, ast.Match):
            guarded = [
                statement for case in node.cases for statement in case.body
            ]
            decision_expression = node.subject
            for case in node.cases:
                if (
                    case.guard is not None
                    and _contains_cross_agent_control_call(
                        case.body, control_aliases, state_object_aliases
                    )
                ):
                    authority_decision_names.update(
                        _identifier_tokens(case.guard)
                    )
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            guarded = [*node.body, *node.orelse]
            decision_expression = node.iter
        elif isinstance(
            node,
            (ast.DictComp, ast.GeneratorExp, ast.ListComp, ast.SetComp),
        ):
            decisions = [
                expression
                for generator in node.generators
                for expression in [generator.iter, *generator.ifs]
            ]
            guarded = [node]
            if decisions:
                decision_expression = ast.BoolOp(
                    op=ast.And(), values=decisions
                )
        if (
            guarded
            and decision_expression is not None
            and _contains_cross_agent_control_call(
                guarded, control_aliases, state_object_aliases
            )
        ):
            authority_decision_names.update(
                _identifier_tokens(decision_expression)
            )

    # Guard clauses and assertions govern the statements that follow them,
    # rather than a syntactically nested branch. Mark their conditions as
    # authority decisions before walking assignment dependencies backwards.
    def collect_guard_decisions(
        statements: list[ast.stmt],
        enclosing_continuation_controls: bool = False,
    ) -> None:
        suffix_controls = [False] * (len(statements) + 1)
        for index in range(len(statements) - 1, -1, -1):
            suffix_controls[index] = suffix_controls[index + 1] or (
                _contains_cross_agent_control_call(
                    statements[index], control_aliases, state_object_aliases
                )
            )
        for index, statement in enumerate(statements):
            controls_continuation = (
                suffix_controls[index + 1]
                or enclosing_continuation_controls
            )
            if controls_continuation:
                if isinstance(statement, ast.Assert):
                    authority_decision_names.update(
                        _identifier_tokens(statement.test)
                    )
                elif isinstance(statement, ast.If) and (
                    _block_guaranteed_exits(statement.body)
                    != _block_guaranteed_exits(statement.orelse)
                ):
                    authority_decision_names.update(
                        _identifier_tokens(statement.test)
                    )
                elif isinstance(
                    statement, (ast.For, ast.AsyncFor, ast.While)
                ) and _loop_else_guards_continuation(statement):
                    decision = (
                        statement.iter
                        if isinstance(statement, (ast.For, ast.AsyncFor))
                        else statement.test
                    )
                    authority_decision_names.update(
                        _identifier_tokens(decision)
                    )
            child_continuation_controls = (
                False
                if isinstance(statement, (ast.For, ast.AsyncFor, ast.While))
                else controls_continuation
            )
            for block in _child_statement_blocks(statement):
                collect_guard_decisions(block, child_continuation_controls)

    if authority_analysis:
        collect_guard_decisions(function.body)

    # Keep control-gate dependencies separate from values that merely flow into
    # a control call. A chain accessor carried inside an A2A payload is
    # propagation; the same accessor feeding an ``if`` that gates lifecycle
    # control is authority. Walking those two dependency graphs together would
    # turn ordinary lineage transport into a false permission finding.
    guard_dependency_names = authority_decision_names - authority_target_names
    changed = authority_analysis
    while changed:
        changed = False
        for names, value, _node in assignments:
            if not names.intersection(guard_dependency_names):
                continue
            dependencies = _identifier_tokens(value).intersection(assignment_names)
            new_dependencies = dependencies - guard_dependency_names
            if new_dependencies:
                guard_dependency_names.update(new_dependencies)
                changed = True

    changed = authority_analysis
    while changed:
        changed = False
        for names, value, _node in assignments:
            if not names.intersection(authority_decision_names):
                continue
            dependencies = _identifier_tokens(value).intersection(assignment_names)
            new_dependencies = dependencies - authority_decision_names
            if new_dependencies:
                authority_decision_names.update(new_dependencies)
                changed = True

    def provenance_selected_decisions(
        statements: list[ast.stmt],
        inherited_selection: bool = False,
    ) -> set[str]:
        """Return later control decisions selected by provenance branches."""

        selected: set[str] = set()
        for statement in statements:
            if inherited_selection:
                for names, _value, assignment in assignments:
                    if assignment is statement:
                        selected.update(
                            names.intersection(authority_decision_names)
                        )

            if isinstance(statement, (ast.If, ast.While)):
                branch_selection = inherited_selection or _has_provenance_value(
                    statement.test, aliases, decision_provenance_helpers
                )
                selected.update(
                    provenance_selected_decisions(
                        statement.body, branch_selection
                    )
                )
                selected.update(
                    provenance_selected_decisions(
                        statement.orelse, branch_selection
                    )
                )
                continue
            if isinstance(statement, ast.Match):
                subject_selection = inherited_selection or _has_provenance_value(
                    statement.subject, aliases, decision_provenance_helpers
                )
                for case in statement.cases:
                    case_selection = subject_selection or (
                        case.guard is not None
                        and _has_provenance_value(
                            case.guard, aliases, decision_provenance_helpers
                        )
                    )
                    selected.update(
                        provenance_selected_decisions(
                            case.body, case_selection
                        )
                    )
                continue
            for block in _child_statement_blocks(statement):
                selected.update(
                    provenance_selected_decisions(block, inherited_selection)
                )
        return selected

    changed = True
    provenance_selected_targets: set[str] = set()
    while changed:
        changed = False
        selected_decisions = (
            provenance_selected_decisions(function.body)
            if authority_analysis
            else set()
        )
        provenance_selected_targets.update(
            selected_decisions.intersection(authority_target_names)
        )
        new_selected = selected_decisions - aliases
        if new_selected:
            aliases.update(new_selected)
            changed = True
        for names, value, _node in assignments:
            permission_shaped_target = any(
                _is_permission_name(name) for name in names
            )
            target_names = names.intersection(authority_target_names)
            unwrapped_value = value
            while isinstance(unwrapped_value, (ast.Await, ast.Expr)):
                unwrapped_value = unwrapped_value.value
            selector_target_names = names.intersection(
                selector_control_argument_names
            )
            target_derived = bool(target_names) and (
                is_provenance_derived(value, aliases)
                or bool(selector_target_names)
                and isinstance(unwrapped_value, ast.Call)
                and any(
                    _has_provenance_value(
                        argument, aliases, provenance_return_helpers
                    )
                    for argument in [
                        *unwrapped_value.args,
                        *(keyword.value for keyword in unwrapped_value.keywords),
                    ]
                )
            )
            guard_decision_names = names.intersection(guard_dependency_names)
            derived = is_provenance_derived(value, aliases) or (
                (
                    permission_shaped_target
                    or bool(guard_decision_names)
                )
                and _has_provenance_value(
                    value, aliases, decision_provenance_helpers
                )
            )
            if target_derived:
                provenance_selected_targets.update(target_names)
            if derived and not names.issubset(aliases):
                aliases.update(names)
                changed = True
    return aliases, provenance_selected_targets


def _class_provenance_state_aliases(
    tree: ast.AST,
    provenance_return_helpers: set[str] | None = None,
    module_provenance_aliases: set[str] | None = None,
) -> dict[ast.FunctionDef | ast.AsyncFunctionDef, set[str]]:
    """Share provenance-bearing ``self``/``cls`` state across class methods."""

    if not isinstance(tree, ast.Module):
        return {}
    by_method: dict[ast.FunctionDef | ast.AsyncFunctionDef, set[str]] = {}
    for class_node in (
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ):
        methods = [
            statement
            for statement in class_node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        shared: set[str] = set()
        changed = True
        while changed:
            changed = False
            for method in methods:
                aliases = set(module_provenance_aliases or ()) | shared
                container_aliases = _mutable_container_alias_snapshots(method)
                method_changed = True
                while method_changed:
                    method_changed = False
                    for statement in _walk_lexical_scope(method):
                        targets: list[ast.AST] = []
                        target_names: set[str] = set()
                        value: ast.AST | None = None
                        if isinstance(statement, ast.Assign):
                            targets = list(statement.targets)
                            value = statement.value
                        elif isinstance(statement, ast.AnnAssign):
                            targets = [statement.target]
                            value = statement.value
                        elif isinstance(statement, ast.NamedExpr):
                            targets = [statement.target]
                            value = statement.value
                        elif (
                            isinstance(statement, ast.Call)
                            and _call_name(statement).casefold() == "setattr"
                            and len(statement.args) >= 3
                            and isinstance(statement.args[0], ast.Name)
                            and statement.args[0].id in {"self", "cls"}
                        ):
                            attribute = _resolved_string(statement.args[1])
                            if attribute is not None:
                                target_names.add(
                                    f"{statement.args[0].id}.{attribute}".casefold()
                                )
                                value = statement.args[2]
                        elif isinstance(statement, ast.Call):
                            mutation = _mutable_container_write(statement)
                            if mutation is not None:
                                target_names, value = mutation
                                target_names.update(
                                    container_aliases.get(statement, ())
                                )
                        if value is None:
                            continue
                        tokens = set(_identifier_tokens(value))
                        direct_tokens = {
                            token
                            for token in tokens
                            if token
                            in {
                                "causation",
                                "causation_chain",
                                "causation_frame",
                                "causationframe",
                                "kestrel.orchestrator",
                                "orchestrator",
                            }
                            or token.startswith("orchestrator_")
                            or (
                                token.startswith("causation_")
                                and not token.endswith(
                                    (
                                        "_accessor",
                                        "_callback",
                                        "_factory",
                                        "_provider",
                                    )
                                )
                            )
                        }
                        calls_helper = any(
                            isinstance(child, ast.Call)
                            and _call_name(child).casefold()
                            in (provenance_return_helpers or set())
                            for child in ast.walk(value)
                        )
                        if not (
                            direct_tokens
                            or _is_provenance_accessor_reference(value)
                            or _is_provenance_accessor_call(value)
                            or calls_helper
                            or tokens.intersection(aliases)
                        ):
                            continue
                        target_names.update(
                            name
                            for target in targets
                            for name in _binding_target_names(target)
                        )
                        new_targets = target_names - aliases
                        if new_targets:
                            aliases.update(new_targets)
                            method_changed = True
                discovered = {
                    alias for alias in aliases if alias.startswith(("self.", "cls."))
                }
                new_aliases = discovered - shared
                if new_aliases:
                    shared.update(new_aliases)
                    changed = True
        for method in methods:
            by_method[method] = set(shared)
    return by_method


def _local_control_helpers(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    imported_control_aliases: set[str] | None = None,
) -> set[str]:
    """Find local helpers that eventually invoke a control sink.

    A neutral helper name is still a control boundary when it invokes a
    callback parameter and a caller supplies ``terminate_child`` (or another
    known control callable) for that parameter.  Track that higher-order edge
    explicitly so moving a control behind ``apply(callback, target)`` cannot
    erase it from the causation-as-authority audit.
    """

    helper_names: set[str] = set(imported_control_aliases or ())
    invoked_parameters: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] = {}
    for function in functions:
        parameter_names = {
            argument.arg.casefold()
            for argument in [
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            ]
        }
        invoked_parameters[function] = {
            node.func.id.casefold()
            for node in _walk_lexical_scope(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.casefold() in parameter_names
        }

    call_sites = [
        (caller, node)
        for caller in functions
        for node in _walk_lexical_scope(caller)
        if isinstance(node, ast.Call)
    ]

    def callback_arguments(
        call: ast.Call,
        callee: ast.FunctionDef | ast.AsyncFunctionDef,
        callback_parameters: set[str],
    ) -> list[ast.AST]:
        positional_parameters = [
            *callee.args.posonlyargs,
            *callee.args.args,
        ]
        # ``obj.method(...)`` does not supply the conventional self/cls
        # parameter explicitly.  Free functions and static methods retain the
        # ordinary zero offset.
        if (
            isinstance(call.func, ast.Attribute)
            and positional_parameters
            and positional_parameters[0].arg.casefold() in {"self", "cls"}
        ):
            positional_parameters = positional_parameters[1:]
        values = [
            value
            for parameter, value in zip(positional_parameters, call.args)
            if parameter.arg.casefold() in callback_parameters
        ]
        values.extend(
            keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
            and keyword.arg.casefold() in callback_parameters
        )
        # A starred call prevents an exact position proof.  If this helper
        # invokes any callback parameter, conservatively inspect the expanded
        # value rather than treating the ambiguity as evidence of safety.
        if callback_parameters:
            values.extend(
                argument.value
                for argument in call.args
                if isinstance(argument, ast.Starred)
            )
            values.extend(
                keyword.value for keyword in call.keywords if keyword.arg is None
            )
        return values

    def is_control_reference(node: ast.AST, aliases: set[str]) -> bool:
        sources = set(_control_reference_sources(node))
        if isinstance(node, ast.Subscript):
            sources.update(_identifier_tokens(node.slice))
        return any(
            _is_unambiguous_control_token(source) or source in aliases
            for source in sources
        )

    changed = True
    while changed:
        changed = False
        for function in functions:
            function_name = function.name.casefold()
            if function_name in helper_names:
                continue
            if any(
                isinstance(node, ast.Call)
                and _is_unambiguous_control_sink(node, helper_names)
                # An unresolved immediately-invoked ``getattr`` fails closed
                # when it is itself provenance-guarded. Do not promote every
                # neutral helper containing one into a repository-wide
                # lifecycle helper: configuration predicate loops such as
                # ``getattr(config, name)()`` are not authority sinks.
                and _control_reference_sources(node.func)
                != {"dynamic_control_attribute"}
                for node in _walk_lexical_scope(function)
            ):
                helper_names.add(function_name)
                changed = True
                continue

            callback_parameters = invoked_parameters.get(function, set())
            if not callback_parameters:
                continue
            for caller, call in call_sites:
                if _call_name(call).casefold() != function_name:
                    continue
                caller_aliases = _cross_agent_control_aliases(
                    caller, helper_names
                )
                if any(
                    is_control_reference(argument, caller_aliases)
                    for argument in callback_arguments(
                        call, function, callback_parameters
                    )
                ):
                    helper_names.add(function_name)
                    changed = True
                    break
    return helper_names


def _local_provenance_return_helpers(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    control_helpers: set[str] | None = None,
    module_provenance_aliases: set[str] | None = None,
    imported_provenance_helpers: set[str] | None = None,
    function_initial_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] | None = None,
) -> set[str]:
    """Find visible helpers whose return value is provenance-derived.

    Seed the fixed point with repository-local imported helper summaries so a
    neutral refactor such as ``derive(request)`` does not erase the fact that
    its result came from ``request.causation_chain``.  Local helpers may then
    wrap either imported or local helpers without escaping the contract.
    """

    helper_names: set[str] = set(imported_provenance_helpers or ())
    changed = True
    while changed:
        changed = False
        for function in functions:
            if function.name.casefold() in helper_names:
                continue
            aliases, _selected_targets = _provenance_aliases(
                function,
                helper_names,
                control_helpers,
                set(module_provenance_aliases or ())
                | set((function_initial_aliases or {}).get(function, ())),
                authority_analysis=False,
            )
            returns_provenance = False
            for node in _walk_lexical_scope(function):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                value = node.value
                while isinstance(value, ast.Await):
                    value = value.value
                if isinstance(value, ast.Call):
                    call_name = _call_name(value).casefold()
                    calls_known_helper = any(
                        isinstance(child, ast.Call)
                        and _call_name(child).casefold() in helper_names
                        for child in ast.walk(value)
                    )
                    returns_provenance = (
                        _is_provenance_accessor_call(value)
                        or call_name in helper_names
                        or call_name in PROVENANCE_TRANSFORM_CALLS
                        and (
                            _has_provenance_token(value, aliases)
                            or calls_known_helper
                        )
                        or call_name.startswith(("can_", "has_", "is_", "may_"))
                        and (
                            _has_provenance_token(value, aliases)
                            or calls_known_helper
                        )
                        or _is_permission_name(call_name)
                        and (
                            _has_provenance_token(value, aliases)
                            or calls_known_helper
                        )
                    )
                else:
                    returns_provenance = _has_provenance_token(value, aliases)
                if returns_provenance:
                    break
            if returns_provenance:
                helper_names.add(function.name.casefold())
                changed = True
    return helper_names


def _contains_cross_agent_control_call(
    nodes: ast.AST | list[ast.AST],
    control_aliases: set[str] | None = None,
    state_object_aliases: set[str] | None = None,
) -> bool:
    """Return whether a guarded expression/body invokes an agent control.

    Authority checks are often wrapped by generic adapters named ``execute``
    or ``dispatch``.  In those cases the enclosing function and condition can
    both be neutrally named even though the branch controls another agent.
    Inspect the guarded operation itself, while stopping at nested lexical
    scopes whose calls are not executed merely because the outer branch ran.
    """

    roots = tuple(nodes) if isinstance(nodes, list) else (nodes,)
    return _cached_contains_cross_agent_control_call(
        roots,
        frozenset(control_aliases or ()),
        frozenset(state_object_aliases or ()),
    )


@lru_cache(maxsize=None)
def _cached_contains_cross_agent_control_call(
    roots: tuple[ast.AST, ...],
    control_aliases: frozenset[str],
    state_object_aliases: frozenset[str],
) -> bool:
    """Cache repeated control-body queries made by fixed-point summaries."""

    def is_control_reference(node: ast.AST) -> bool:
        if _is_cross_agent_control_reference(node):
            return True
        reference_name = (
            node.id.casefold()
            if isinstance(node, ast.Name)
            else node.attr.casefold()
            if isinstance(node, ast.Attribute)
            else ""
        )
        return bool(reference_name and reference_name in control_aliases)

    class ControlCallVisitor(ast.NodeVisitor):
        found = False

        def has_control_target(self, targets: list[ast.AST]) -> bool:
            return any(
                _is_cross_agent_state_mutation_target(
                    target, state_object_aliases
                )
                for target in targets
            )

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
            if _is_cross_agent_control_call(
                node, control_aliases
            ) or _is_cross_agent_state_mutation_call(
                node, state_object_aliases
            ):
                self.found = True
                return
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
            if self.has_control_target(list(node.targets)) or is_control_reference(
                node.value
            ):
                self.found = True
                return
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
            if self.has_control_target([node.target]) or (
                node.value is not None and is_control_reference(node.value)
            ):
                self.found = True
                return
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
            if self.has_control_target([node.target]):
                self.found = True
                return
            self.generic_visit(node)

        def visit_Delete(self, node: ast.Delete) -> None:  # noqa: N802
            if self.has_control_target(list(node.targets)):
                self.found = True
                return
            self.generic_visit(node)

        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
            if self.has_control_target([node.target]) or is_control_reference(
                node.value
            ):
                self.found = True
                return
            self.generic_visit(node)

        def visit_FunctionDef(  # noqa: N802 - ast API
            self, node: ast.FunctionDef
        ) -> None:
            return

        def visit_AsyncFunctionDef(  # noqa: N802 - ast API
            self, node: ast.AsyncFunctionDef
        ) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

    visitor = ControlCallVisitor()
    for root in roots:
        visitor.visit(root)
        if visitor.found:
            return True
    return False


def _is_cross_agent_control_reference(node: ast.AST) -> bool:
    """Whether an expression selects a control callable without invoking it."""

    return any(
        _is_unambiguous_control_token(source)
        for source in _control_reference_sources(node)
    )


def _guard_clause_provenance_lines(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    provenance_aliases: set[str],
    control_aliases: set[str],
    provenance_return_helpers: set[str] | None = None,
    state_object_aliases: set[str] | None = None,
) -> set[int]:
    """Find provenance conditions that gate a later control by exiting early."""

    lines: set[int] = set()

    def scan_block(
        statements: list[ast.stmt],
        enclosing_continuation_controls: bool = False,
    ) -> None:
        suffix_controls = [False] * (len(statements) + 1)
        for index in range(len(statements) - 1, -1, -1):
            suffix_controls[index] = suffix_controls[index + 1] or (
                _contains_cross_agent_control_call(
                    statements[index], control_aliases, state_object_aliases
                )
            )
        for index, statement in enumerate(statements):
            controls_continuation = (
                suffix_controls[index + 1]
                or enclosing_continuation_controls
            )
            if controls_continuation and isinstance(statement, ast.Assert):
                if _has_provenance_value(
                    statement.test,
                    provenance_aliases,
                    provenance_return_helpers,
                ):
                    lines.add(statement.lineno)
            elif controls_continuation and isinstance(statement, ast.If):
                body_exits = _block_guaranteed_exits(statement.body)
                orelse_exits = _block_guaranteed_exits(statement.orelse)
                if (
                    body_exits != orelse_exits
                    and _has_provenance_value(
                        statement.test,
                        provenance_aliases,
                        provenance_return_helpers,
                    )
                ):
                    lines.add(statement.lineno)
            elif controls_continuation and isinstance(statement, ast.Match):
                case_exits = [
                    _block_guaranteed_exits(case.body)
                    for case in statement.cases
                ]
                has_catch_all = any(
                    case.guard is None
                    and isinstance(case.pattern, ast.MatchAs)
                    and case.pattern.pattern is None
                    for case in statement.cases
                )
                match_can_continue = (
                    not has_catch_all or not all(case_exits)
                )
                if any(case_exits) and match_can_continue:
                    if _has_provenance_value(
                        statement.subject,
                        provenance_aliases,
                        provenance_return_helpers,
                    ):
                        lines.add(statement.lineno)
                    lines.update(
                        case.guard.lineno
                        for case, exits in zip(statement.cases, case_exits)
                        if exits
                        and case.guard is not None
                        and _has_provenance_value(
                            case.guard,
                            provenance_aliases,
                            provenance_return_helpers,
                        )
                    )
            elif controls_continuation and isinstance(
                statement, (ast.For, ast.AsyncFor, ast.While)
            ):
                decision = (
                    statement.iter
                    if isinstance(statement, (ast.For, ast.AsyncFor))
                    else statement.test
                )
                if _loop_else_guards_continuation(
                    statement
                ) and _has_provenance_value(
                    decision,
                    provenance_aliases,
                    provenance_return_helpers,
                ):
                    lines.add(statement.lineno)
            # ``break``/``continue`` inside a loop do not prevent statements
            # after the loop from running, so do not inherit that outer
            # continuation into loop bodies. Other compound statements retain
            # the enclosing continuation: a return/raise nested under ``with``
            # or ``try`` still gates the later control call.
            child_continuation_controls = (
                False
                if isinstance(statement, (ast.For, ast.AsyncFor, ast.While))
                else controls_continuation
            )
            for block in _child_statement_blocks(statement):
                scan_block(block, child_continuation_controls)

    scan_block(function.body)
    return lines


def _module_imported_control_aliases(
    tree: ast.AST,
    source_path: Path | None = None,
    seen: frozenset[tuple[Path, str]] | None = None,
) -> set[str]:
    """Return neutral local names imported from control-shaped callables."""

    if not isinstance(tree, ast.Module):
        return set()

    def is_control_callable(name: str) -> bool:
        lowered = name.casefold()
        actions = (
            "cancel",
            "delegate",
            "hold",
            "interrupt",
            "kill",
            "offboard",
            "restart",
            "shutdown",
            "spawn",
            "stop",
            "terminate",
            "withdraw",
        )
        subjects = ("a2a", "agent", "child", "descendant", "fleet", "host", "peer")
        words = set(lowered.split("_"))
        return (
            lowered == "kill_process"
            or bool(words.intersection(actions))
            or any(action in lowered for action in (*actions, "remove"))
            and any(subject in lowered for subject in subjects)
        )

    aliases = {
        (imported.asname or imported.name).casefold()
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for imported in node.names
        if is_control_callable(imported.name)
    }

    def include_module_assignment_aliases() -> set[str]:
        assignments: list[tuple[set[str], set[str]]] = []

        class ModuleAssignmentVisitor(ast.NodeVisitor):
            def visit_FunctionDef(  # noqa: N802 - ast API
                self, node: ast.FunctionDef
            ) -> None:
                return

            def visit_AsyncFunctionDef(  # noqa: N802 - ast API
                self, node: ast.AsyncFunctionDef
            ) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
                return

            def _record(
                self, targets: list[ast.AST], value: ast.AST | None
            ) -> None:
                if value is None:
                    return
                assignments.append(
                    (
                        {
                            name
                            for target in targets
                            for name in _reference_binding_names(target)
                        },
                        _control_reference_sources(value),
                    )
                )

            def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
                self._record(list(node.targets), node.value)
                self.generic_visit(node.value)

            def visit_AnnAssign(  # noqa: N802 - ast API
                self, node: ast.AnnAssign
            ) -> None:
                self._record([node.target], node.value)
                if node.value is not None:
                    self.generic_visit(node.value)

            def visit_NamedExpr(  # noqa: N802 - ast API
                self, node: ast.NamedExpr
            ) -> None:
                self._record([node.target], node.value)
                self.generic_visit(node.value)

        ModuleAssignmentVisitor().visit(tree)
        changed = True
        while changed:
            changed = False
            for targets, sources in assignments:
                if any(
                    is_control_callable(source) or source in aliases
                    for source in sources
                ):
                    new_aliases = targets - aliases
                    if new_aliases:
                        aliases.update(new_aliases)
                        changed = True
        return aliases

    if source_path is None:
        return include_module_assignment_aliases()

    source_path = source_path.resolve()
    seen = seen or frozenset()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_names = {_call_name(call).casefold() for call in calls}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                for imported in node.names:
                    child_path = _resolved_repository_import_path(
                        source_path,
                        imported.name,
                        node.level,
                    )
                    if child_path is None:
                        continue
                    module_calls = _module_attribute_call_names(
                        calls,
                        (imported.asname or imported.name).casefold(),
                    )
                    aliases.update(
                        _repository_control_helper_names(
                            child_path,
                            module_calls,
                            seen,
                        )
                    )
                continue

            imported_path = _resolved_repository_import_path(
                source_path,
                node.module,
                node.level,
            )
            if imported_path is None:
                continue
            bindings = {
                (imported.asname or imported.name).casefold(): imported.name.casefold()
                for imported in node.names
                if imported.name != "*"
                and (imported.asname or imported.name).casefold() in called_names
            }
            star_names = called_names if any(
                imported.name == "*" for imported in node.names
            ) else set()
            resolved = _repository_control_helper_names(
                imported_path,
                set(bindings.values()) | star_names,
                seen,
            )
            aliases.update(
                local_name
                for local_name, remote_name in bindings.items()
                if remote_name in resolved
            )
            aliases.update(star_names.intersection(resolved))
            for imported in node.names:
                if imported.name == "*":
                    continue
                qualifier = (imported.asname or imported.name).casefold()
                member_calls = _module_attribute_call_names(calls, qualifier)
                aliases.update(
                    _repository_control_helper_names(
                        imported_path,
                        member_calls,
                        seen,
                    )
                )
        elif isinstance(node, ast.Import):
            for imported in node.names:
                imported_path = _resolved_repository_import_path(
                    source_path,
                    imported.name,
                )
                if imported_path is None:
                    continue
                bound_name = (imported.asname or imported.name).casefold()
                module_calls = _module_attribute_call_names(calls, bound_name)
                aliases.update(
                    _repository_control_helper_names(
                        imported_path,
                        module_calls,
                        seen,
                    )
                )
    return include_module_assignment_aliases()


def _module_provenance_constant_aliases(
    tree: ast.AST,
    source_path: Path | None = None,
) -> set[str]:
    """Return names bound to static causation/display-provenance keys."""

    if not isinstance(tree, ast.Module):
        return set()
    constants = _module_string_constants(tree, source_path)
    aliases = {
        name.casefold()
        for name, value in constants.items()
        if _has_provenance_token(ast.Constant(value=value))
    }
    if source_path is None:
        return aliases

    source_path = source_path.resolve()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                imported_path = _resolved_repository_import_path(
                    source_path,
                    imported.name,
                )
                if imported_path is None:
                    continue
                qualifier = (imported.asname or imported.name).casefold()
                for name, value in _cached_local_string_constants(
                    imported_path
                ).items():
                    if _has_provenance_token(ast.Constant(value=value)):
                        aliases.add(f"{qualifier}.{name}".casefold())
        elif isinstance(node, ast.ImportFrom):
            for imported in node.names:
                child_module = ".".join(
                    part
                    for part in (node.module, imported.name)
                    if part
                )
                imported_path = _resolved_repository_import_path(
                    source_path,
                    child_module,
                    node.level,
                )
                if imported_path is None:
                    continue
                qualifier = (imported.asname or imported.name).casefold()
                for name, value in _cached_local_string_constants(
                    imported_path
                ).items():
                    if _has_provenance_token(ast.Constant(value=value)):
                        aliases.add(f"{qualifier}.{name}".casefold())
    return aliases


def _module_imported_provenance_accessor_aliases(tree: ast.AST) -> set[str]:
    """Return module aliases that preserve canonical chain accessors."""

    if not isinstance(tree, ast.Module):
        return set()
    aliases = {
        (imported.asname or imported.name).casefold()
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for imported in node.names
        if imported.name.casefold().strip("_").endswith(
            PROVENANCE_ACCESSOR_SUFFIXES
        )
    }
    assignments: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.append((target.id.casefold(), value))
    changed = True
    while changed:
        changed = False
        for target, value in assignments:
            source = (
                value.id.casefold()
                if isinstance(value, ast.Name)
                else value.attr.casefold()
                if isinstance(value, ast.Attribute)
                else ""
            )
            if (
                _is_provenance_accessor_reference(value) or source in aliases
            ) and target not in aliases:
                aliases.add(target)
                changed = True
    return aliases


def _module_provenance_state_aliases(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    function_parents: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        ast.FunctionDef | ast.AsyncFunctionDef,
    ],
    provenance_return_helpers: set[str],
    module_provenance_aliases: set[str],
    function_initial_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ],
    function_accessor_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ],
) -> set[str]:
    """Share provenance stored through explicit module-global assignments."""

    aliases = set(module_provenance_aliases)
    changed = True
    while changed:
        changed = False
        function_aliases: dict[
            ast.FunctionDef | ast.AsyncFunctionDef, set[str]
        ] = {}

        def analyze(
            function: ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> set[str]:
            cached = function_aliases.get(function)
            if cached is not None:
                return cached
            parent = function_parents.get(function)
            inherited = analyze(parent) if parent is not None else set()
            resolved, _selected_targets = _provenance_aliases(
                function,
                provenance_return_helpers,
                initial_aliases=(
                    aliases
                    | inherited
                    | function_initial_aliases.get(function, set())
                ),
                authority_analysis=False,
                provenance_accessor_aliases=function_accessor_aliases.get(
                    function, set()
                ),
            )
            function_aliases[function] = resolved
            return resolved

        for function in functions:
            declared_globals = {
                name.casefold()
                for node in _walk_lexical_scope(function)
                if isinstance(node, ast.Global)
                for name in node.names
            }
            discovered = declared_globals.intersection(analyze(function))
            new_aliases = discovered - aliases
            if new_aliases:
                aliases.update(new_aliases)
                changed = True
    return aliases


def _resolved_repository_import_path(
    source_path: Path,
    module_name: str | None,
    level: int = 0,
) -> Path | None:
    """Resolve a Python import to a source module available in this checkout."""

    module_parts = tuple(part for part in (module_name or "").split(".") if part)
    if level:
        base = source_path.parent
        for _ in range(level - 1):
            base = base.parent
        roots = (base,)
    else:
        # Project-qualified imports resolve from the checkout root.  A
        # same-directory fallback also makes isolated synthetic modules model
        # ordinary source-tree imports without manipulating sys.path.
        roots = (REPO_ROOT, source_path.parent)

    for root in roots:
        candidate = root.joinpath(*module_parts)
        module_file = candidate.with_suffix(".py")
        if module_file.is_file():
            return module_file.resolve()
        package_file = candidate / "__init__.py"
        if package_file.is_file():
            return package_file.resolve()
    return None


def _module_attribute_call_names(
    calls: list[ast.Call],
    bound_name: str,
) -> set[str]:
    """Return terminal call names invoked through one imported module binding."""

    prefix = f"{bound_name.casefold()}."
    return {
        _call_name(call).casefold()
        for call in calls
        if ast.unparse(call.func).casefold().startswith(prefix)
    }


@lru_cache(maxsize=None)
def _direct_control_helper_names(source_path: Path) -> frozenset[str]:
    """Summarize control helpers defined within one repository module."""

    tree = _parsed_module(source_path.resolve())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return frozenset(
        _local_control_helpers(
            functions,
            _module_imported_control_aliases(tree),
        )
    )


def _repository_control_helper_names(
    source_path: Path,
    requested_names: set[str],
    seen: frozenset[tuple[Path, str]],
) -> set[str]:
    """Resolve requested control-helper summaries across local import edges."""

    source_path = source_path.resolve()
    requested_names = {
        name.casefold()
        for name in requested_names
        if (source_path, name.casefold()) not in seen
    }
    if not requested_names:
        return set()

    direct_helpers = set(_direct_control_helper_names(source_path))
    resolved = requested_names.intersection(direct_helpers)
    if resolved == requested_names:
        return resolved

    active = seen | {
        (source_path, name) for name in requested_names - resolved
    }
    tree = _parsed_module(source_path)
    imported_helpers = _module_imported_control_aliases(
        tree,
        source_path,
        active,
    )
    if not imported_helpers:
        return resolved
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    all_helpers = _local_control_helpers(functions, imported_helpers)
    return requested_names.intersection(all_helpers)


@lru_cache(maxsize=None)
def _direct_provenance_helper_names(source_path: Path) -> frozenset[str]:
    """Summarize helpers derived within one module, before following imports."""

    tree = _parsed_module(source_path.resolve())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    module_provenance_aliases = (
        _module_provenance_constant_aliases(tree, source_path)
        | _module_imported_provenance_accessor_aliases(tree)
    )
    control_helpers = _local_control_helpers(
        functions,
        _module_imported_control_aliases(tree),
    )
    return frozenset(
        _local_provenance_return_helpers(
            functions,
            control_helpers,
            module_provenance_aliases,
        )
    )


def _repository_provenance_helper_names(
    source_path: Path,
    requested_names: set[str],
    seen: frozenset[tuple[Path, str]],
) -> set[str]:
    """Resolve requested helper summaries through narrow local import edges."""

    source_path = source_path.resolve()
    requested_names = {name.casefold() for name in requested_names}
    requested_names = {
        name
        for name in requested_names
        if (source_path, name) not in seen
    }
    if not requested_names:
        return set()

    direct_helpers = set(_direct_provenance_helper_names(source_path))
    resolved = requested_names.intersection(direct_helpers)
    if resolved == requested_names:
        return resolved

    active = seen | {
        (source_path, name) for name in requested_names - resolved
    }
    tree = _parsed_module(source_path)
    imported_helpers = _module_imported_provenance_return_helper_aliases(
        tree,
        source_path,
        active,
    )
    if not imported_helpers:
        return resolved

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    module_provenance_aliases = (
        _module_provenance_constant_aliases(tree, source_path)
        | _module_imported_provenance_accessor_aliases(tree)
    )
    control_helpers = _local_control_helpers(
        functions,
        _module_imported_control_aliases(tree),
    )
    all_helpers = _local_provenance_return_helpers(
        functions,
        control_helpers,
        module_provenance_aliases,
        imported_helpers,
    )
    return requested_names.intersection(all_helpers)


def _module_imported_provenance_return_helper_aliases(
    tree: ast.AST,
    source_path: Path | None,
    seen: frozenset[tuple[Path, str]] | None = None,
) -> set[str]:
    """Return called local bindings for provenance-returning imports.

    Only follow imports that are actually invoked by this module.  This keeps
    the repository-wide contract proportional to the authority call graph
    rather than recursively summarizing the checkout's entire import graph.
    """

    if not isinstance(tree, ast.Module) or source_path is None:
        return set()

    source_path = source_path.resolve()
    seen = seen or frozenset()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_names = {_call_name(call).casefold() for call in calls}
    aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                for imported in node.names:
                    child_path = _resolved_repository_import_path(
                        source_path,
                        imported.name,
                        node.level,
                    )
                    if child_path is not None:
                        module_calls = _module_attribute_call_names(
                            calls,
                            (imported.asname or imported.name).casefold(),
                        )
                        aliases.update(
                            _repository_provenance_helper_names(
                                child_path,
                                module_calls,
                                seen,
                            )
                        )
                continue

            imported_path = _resolved_repository_import_path(
                source_path,
                node.module,
                node.level,
            )
            if imported_path is None:
                continue
            bindings = {
                (imported.asname or imported.name).casefold(): imported.name.casefold()
                for imported in node.names
                if imported.name != "*"
                and (imported.asname or imported.name).casefold() in called_names
            }
            star_names = called_names if any(
                imported.name == "*" for imported in node.names
            ) else set()
            requested = set(bindings.values()) | star_names
            resolved = _repository_provenance_helper_names(
                imported_path,
                requested,
                seen,
            )
            aliases.update(
                local_name
                for local_name, remote_name in bindings.items()
                if remote_name in resolved
            )
            aliases.update(star_names.intersection(resolved))
        elif isinstance(node, ast.Import):
            for imported in node.names:
                imported_path = _resolved_repository_import_path(
                    source_path,
                    imported.name,
                )
                if imported_path is not None:
                    bound_name = (imported.asname or imported.name).casefold()
                    module_calls = _module_attribute_call_names(calls, bound_name)
                    aliases.update(
                        _repository_provenance_helper_names(
                            imported_path,
                            module_calls,
                            seen,
                        )
                    )
    return aliases


def _nested_function_parents(
    tree: ast.AST,
) -> dict[
    ast.FunctionDef | ast.AsyncFunctionDef,
    ast.FunctionDef | ast.AsyncFunctionDef,
]:
    """Map nested functions to the function whose locals they close over."""

    parents: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        ast.FunctionDef | ast.AsyncFunctionDef,
    ] = {}
    stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    class FunctionParentVisitor(ast.NodeVisitor):
        def _visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            if stack:
                parents[node] = stack[-1]
            stack.append(node)
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_function(node)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            self._visit_function(node)

    FunctionParentVisitor().visit(tree)
    return parents


def _authority_provenance_lines(
    tree: ast.AST,
    source_path: Path | None = None,
) -> set[int]:
    lines: set[int] = set()
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    module_control_aliases = _module_imported_control_aliases(
        tree,
        source_path,
    )
    module_provenance_aliases = (
        _module_provenance_constant_aliases(tree, source_path)
        | _module_imported_provenance_accessor_aliases(tree)
    )
    imported_provenance_helpers = (
        _module_imported_provenance_return_helper_aliases(tree, source_path)
    )
    control_helpers = _local_control_helpers(functions, module_control_aliases)
    function_parents = _nested_function_parents(tree)
    function_accessor_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] = {}

    def analyze_function_accessors(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> set[str]:
        cached = function_accessor_aliases.get(function)
        if cached is not None:
            return cached
        parent = function_parents.get(function)
        inherited = (
            analyze_function_accessors(parent) if parent is not None else set()
        )
        resolved = _function_provenance_accessor_aliases(function, inherited)
        function_accessor_aliases[function] = resolved
        return resolved

    for function in functions:
        analyze_function_accessors(function)

    class_provenance_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] = {}
    provenance_return_helpers = set(imported_provenance_helpers)
    analysis_changed = True
    while analysis_changed:
        previous_module_aliases = set(module_provenance_aliases)
        previous_helper_names = set(provenance_return_helpers)
        previous_class_aliases = {
            function: set(aliases)
            for function, aliases in class_provenance_aliases.items()
        }
        class_provenance_aliases = _class_provenance_state_aliases(
            tree,
            provenance_return_helpers,
            module_provenance_aliases,
        )
        provenance_return_helpers = _local_provenance_return_helpers(
            functions,
            control_helpers,
            module_provenance_aliases,
            provenance_return_helpers,
            class_provenance_aliases,
        )
        module_provenance_aliases = _module_provenance_state_aliases(
            functions,
            function_parents,
            provenance_return_helpers,
            module_provenance_aliases,
            class_provenance_aliases,
            function_accessor_aliases,
        )
        analysis_changed = (
            module_provenance_aliases != previous_module_aliases
            or provenance_return_helpers != previous_helper_names
            or class_provenance_aliases != previous_class_aliases
        )
    function_state_objects: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] = {}

    def analyze_function_state_objects(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> set[str]:
        cached = function_state_objects.get(function)
        if cached is not None:
            return cached
        parent = function_parents.get(function)
        inherited = (
            analyze_function_state_objects(parent)
            if parent is not None
            else set()
        )
        resolved = _cross_agent_state_object_aliases(function, inherited)
        function_state_objects[function] = resolved
        return resolved

    for function in functions:
        analyze_function_state_objects(function)

    function_provenance: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, tuple[set[str], set[str]]
    ] = {}

    def analyze_function_provenance(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[set[str], set[str]]:
        cached = function_provenance.get(function)
        if cached is not None:
            return cached
        parent = function_parents.get(function)
        inherited = (
            analyze_function_provenance(parent)[0]
            if parent is not None
            else set()
        )
        resolved = _provenance_aliases(
            function,
            provenance_return_helpers,
            control_helpers,
            module_provenance_aliases
            | class_provenance_aliases.get(function, set())
            | inherited,
            state_object_aliases=function_state_objects[function],
            provenance_accessor_aliases=function_accessor_aliases[function],
        )
        function_provenance[function] = resolved
        return resolved

    for function in functions:
        analyze_function_provenance(function)
    for function in functions:
        function_name = function.name.casefold()
        decision_provenance_helpers = (
            provenance_return_helpers | function_accessor_aliases[function]
        )
        control_aliases = _cross_agent_control_aliases(function, control_helpers)
        state_object_aliases = function_state_objects[function]
        provenance_aliases, provenance_selected_targets = (
            function_provenance[function]
        )
        lines.update(
            _guard_clause_provenance_lines(
                function,
                provenance_aliases,
                control_aliases,
                decision_provenance_helpers,
                state_object_aliases,
            )
        )
        function_is_permission_boundary = (
            _is_permission_name(function_name)
            or function_name.startswith(("can_", "may_"))
            or _is_unambiguous_control_token(function_name)
        )
        for lambda_node in _invoked_lambda_bodies(function):
            if _has_provenance_value(
                lambda_node.body,
                provenance_aliases,
                decision_provenance_helpers,
            ) and _contains_cross_agent_control_call(
                lambda_node.body, control_aliases, state_object_aliases
            ):
                lines.add(lambda_node.lineno)
        for node in _walk_lexical_scope(function):
            if isinstance(node, ast.Call):
                function_tokens = _identifier_tokens(node.func)
                is_permission_call = any(
                    _is_permission_name(token) for token in function_tokens
                )
                arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
                if is_permission_call and any(
                    _has_provenance_value(
                        argument,
                        provenance_aliases,
                        decision_provenance_helpers,
                    )
                    for argument in arguments
                ):
                    lines.add(node.lineno)
                if _is_cross_agent_control_call(node, control_aliases):
                    # Parameter spelling is not an authority boundary.  A
                    # known control sink may call its target ``candidate``,
                    # ``subject``, or anything else, so inspect every supplied
                    # value rather than maintaining a bypassable name list.
                    direct_control_inputs = arguments
                    if (
                        _has_provenance_value(
                            node.func,
                            provenance_aliases,
                            provenance_return_helpers,
                        )
                        or any(
                            _has_provenance_value(
                                argument,
                                provenance_aliases,
                                provenance_return_helpers,
                            )
                            for argument in direct_control_inputs
                        )
                        or any(
                            bool(
                                _identifier_tokens(argument).intersection(
                                    provenance_selected_targets
                                )
                            )
                            for argument in arguments
                        )
                    ):
                        lines.add(node.lineno)
            assignment_targets: list[ast.AST] = []
            assignment_value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                assignment_targets = list(node.targets)
                assignment_value = node.value
            elif isinstance(node, ast.AnnAssign):
                assignment_targets = [node.target]
                assignment_value = node.value
            elif isinstance(node, ast.NamedExpr):
                assignment_targets = [node.target]
                assignment_value = node.value
            if assignment_value is not None:
                target_tokens = {
                    token
                    for target in assignment_targets
                    for token in _identifier_tokens(target)
                }
                if (
                    any(_is_permission_name(token) for token in target_tokens)
                    and _has_provenance_value(
                        assignment_value,
                        provenance_aliases,
                        decision_provenance_helpers,
                    )
                ):
                    lines.add(node.lineno)
            if isinstance(node, ast.Return):
                if (
                    node.value is not None
                    and _has_provenance_value(
                        node.value,
                        provenance_aliases,
                        decision_provenance_helpers,
                    )
                    and (
                        function_is_permission_boundary
                        or any(
                            _is_permission_name(token)
                            for token in _identifier_tokens(node.value)
                        )
                    )
                ):
                    lines.add(node.lineno)
                continue
            if isinstance(node, ast.Match):
                if _has_provenance_value(
                    node.subject,
                    provenance_aliases,
                    decision_provenance_helpers,
                ) and (
                    function_is_permission_boundary
                    or _contains_cross_agent_control_call(
                        [statement for case in node.cases for statement in case.body],
                        control_aliases,
                        state_object_aliases,
                    )
                ):
                    lines.add(node.lineno)
                for case in node.cases:
                    if (
                        case.guard is not None
                        and _has_provenance_value(
                            case.guard,
                            provenance_aliases,
                            decision_provenance_helpers,
                        )
                        and (
                            function_is_permission_boundary
                            or _contains_cross_agent_control_call(
                                case.body, control_aliases, state_object_aliases
                            )
                        )
                    ):
                        lines.add(case.guard.lineno)
                continue
            if isinstance(node, (ast.For, ast.AsyncFor)):
                if _has_provenance_value(
                    node.iter,
                    provenance_aliases,
                    decision_provenance_helpers,
                ) and (
                    function_is_permission_boundary
                    or _contains_cross_agent_control_call(
                        [*node.body, *node.orelse],
                        control_aliases,
                        state_object_aliases,
                    )
                ):
                    lines.add(node.lineno)
                continue
            if isinstance(node, (ast.With, ast.AsyncWith)):
                if any(
                    (
                        not _is_non_authority_context_manager(item.context_expr)
                        and _has_provenance_value(
                            item.context_expr,
                            provenance_aliases,
                            decision_provenance_helpers,
                        )
                    )
                    for item in node.items
                ) and (
                    function_is_permission_boundary
                    or _contains_cross_agent_control_call(
                        node.body, control_aliases, state_object_aliases
                    )
                ):
                    lines.add(node.lineno)
                continue
            if isinstance(node, ast.BoolOp):
                if (
                    _has_provenance_value(
                        node,
                        provenance_aliases,
                        decision_provenance_helpers,
                    )
                    and _contains_cross_agent_control_call(
                        node.values, control_aliases, state_object_aliases
                    )
                ):
                    lines.add(node.lineno)
                continue
            if isinstance(
                node,
                (ast.DictComp, ast.GeneratorExp, ast.ListComp, ast.SetComp),
            ):
                comprehension_aliases = set(provenance_aliases)
                provenance_driven = False
                for generator in node.generators:
                    if _has_provenance_value(
                        generator.iter,
                        comprehension_aliases,
                        decision_provenance_helpers,
                    ):
                        provenance_driven = True
                        comprehension_aliases.update(
                            _binding_target_names(generator.target)
                        )
                    if any(
                        _has_provenance_value(
                            condition,
                            comprehension_aliases,
                            decision_provenance_helpers,
                        )
                        for condition in generator.ifs
                    ):
                        provenance_driven = True
                if (
                    provenance_driven
                    and _contains_cross_agent_control_call(
                        node, control_aliases, state_object_aliases
                    )
                ):
                    lines.add(node.lineno)
                continue
            if not isinstance(node, (ast.If, ast.IfExp, ast.Assert, ast.While)):
                continue
            tokens = _identifier_tokens(node.test)
            has_provenance = _has_provenance_value(
                node.test,
                provenance_aliases,
                decision_provenance_helpers,
            )
            has_permission = any(_is_permission_name(token) for token in tokens)
            guarded_nodes: list[ast.AST] = [node.test]
            if isinstance(node, (ast.If, ast.While)):
                guarded_nodes.extend([*node.body, *node.orelse])
            elif isinstance(node, ast.IfExp):
                guarded_nodes.extend([node.body, node.orelse])
            selects_control = isinstance(node, ast.IfExp) and any(
                _is_cross_agent_control_reference(branch)
                for branch in (node.body, node.orelse)
            )
            if has_provenance and (
                has_permission
                or function_is_permission_boundary
                or _contains_cross_agent_control_call(
                    guarded_nodes, control_aliases, state_object_aliases
                )
                or selects_control
            ):
                lines.add(node.lineno)
    return lines


@lru_cache(maxsize=None)
def _cached_authority_provenance_lines(source_path: Path) -> frozenset[int]:
    """Reuse one module's parsed tree and complete authority summary."""

    source = _source_text(source_path).casefold()
    tree = _parsed_module(source_path)
    imported_or_local_provenance = _module_provenance_constant_aliases(
        tree, source_path
    )
    imported_provenance_helpers = (
        _module_imported_provenance_return_helper_aliases(tree, source_path)
    )
    if not any(
        marker in source
        for marker in ("causation", "orchestrator", "current_chain")
    ) and not imported_or_local_provenance and not imported_provenance_helpers:
        return frozenset()
    return frozenset(
        _authority_provenance_lines(tree, source_path)
    )


def test_direct_provenance_authority_patterns_are_detected() -> None:
    enclosing = ast.parse(
        "def authorize_child(request):\n"
        "    if request.causation_chain:\n"
        "        return True\n"
    )
    metadata_key = ast.parse(
        "def check(metadata):\n"
        '    return authorize(metadata.get("kestrel.orchestrator"))\n'
    )
    direct_return = ast.parse(
        "def is_authorized(request):\n"
        "    return bool(request.causation_chain)\n"
    )
    ordinary_helpers = ast.parse(
        "def can_control(request):\n"
        "    return bool(request.causation_chain)\n\n"
        "def is_allowed(request):\n"
        "    return bool(request.orchestrator)\n"
    )
    control_handlers = ast.parse(
        "def terminate_child(request):\n"
        "    if request.causation_chain:\n"
        "        return True\n\n"
        "def delegate_task(request):\n"
        "    chain = request.causation_chain\n"
        "    return bool(chain)\n\n"
        "def cancel_task(request):\n"
        "    chain = request.metadata.get('kestrel.orchestrator')\n"
        "    if chain:\n"
        "        return True\n"
    )
    canonical_frame = ast.parse(
        "def authorize_child(request):\n"
        "    if request.causation_frame:\n"
        "        return True\n"
    )
    compared_alias = ast.parse(
        "def authorize_child(request):\n"
        "    allowed = request.causation_chain is not None\n"
        "    if allowed:\n"
        "        return True\n"
    )
    permission_call_alias = ast.parse(
        "def check(request):\n"
        "    chain = request.causation_chain\n"
        "    return authorize(chain)\n"
    )
    collection_wrappers = ast.parse(
        "def authorize_list(request):\n"
        "    chain = list(request.causation_chain)\n"
        "    if chain:\n"
        "        return True\n\n"
        "def authorize_tuple(request):\n"
        "    chain = tuple(request.causation_chain)\n"
        "    return bool(chain)\n\n"
        "def authorize_copy(request):\n"
        "    chain = [*request.causation_chain]\n"
        "    return bool(chain)\n"
    )
    derived_values = ast.parse(
        "def authorize_length(request):\n"
        "    chain_len = len(request.causation_chain)\n"
        "    if chain_len:\n"
        "        return True\n\n"
        "def authorize_comprehension(request):\n"
        "    chain = [frame for frame in request.causation_chain]\n"
        "    return bool(chain)\n\n"
        "def authorize_predicate(request):\n"
        "    trusted = has_valid_causation(request.causation_chain)\n"
        "    if trusted:\n"
        "        return True\n"
    )
    authority_helper = ast.parse(
        "def scheduler_authority_for(request):\n"
        "    return bool(request.causation_chain)\n"
    )
    access_helper = ast.parse(
        "def check_access(request):\n"
        "    if request.causation_frame:\n"
        "        return True\n"
    )
    require_call = ast.parse(
        "def check(request):\n"
        "    chain = request.causation_chain\n"
        "    return require_access(chain)\n"
    )
    dynamic_metadata_key = ast.parse(
        "def check_access(request):\n"
        '    key = "causation_chain"\n'
        "    return bool(request.metadata.get(key))\n"
    )
    mandate_verifier = ast.parse(
        "def verify_mandate(request):\n"
        "    return bool(request.causation_chain)\n"
    )
    helper_return = ast.parse(
        "def check(request):\n"
        "    allowed = bool(request.causation_chain)\n"
        "    return allowed\n"
    )
    permission_assignment = ast.parse(
        "def check(request):\n"
        "    request.state.authorized = bool(request.causation_chain)\n"
    )
    permission_mapping = ast.parse(
        "def check(request):\n"
        '    return {"authorized": bool(request.causation_chain)}\n'
    )
    permission_match = ast.parse(
        "def authorize(request):\n"
        "    match request.causation_chain:\n"
        "        case []:\n"
        "            return False\n"
        "        case _:\n"
        "            return True\n"
    )
    unrelated_attribute_assignment = ast.parse(
        "def authorize_cache(request):\n"
        "    request.state.snapshot = request.causation_chain\n"
        "    return request\n"
    )
    neutral_guarded_control = ast.parse(
        "async def execute(request):\n"
        "    if request.causation_chain:\n"
        "        await terminate_child(request.target)\n\n"
        "def dispatch(request):\n"
        "    return cancel_task(request.target) if request.causation_chain else None\n\n"
        "def adapter(request):\n"
        "    match request.causation_chain:\n"
        "        case []:\n"
        "            return None\n"
        "        case _:\n"
        "            return stop_peer(request.target)\n"
    )
    propagation_only_guard = ast.parse(
        "def dispatch(request, metadata):\n"
        "    if request.causation_chain:\n"
        "        metadata['causation_chain'] = request.causation_chain\n"
    )
    stateful_aliases = ast.parse(
        "def dispatch(request, state):\n"
        "    self.lineage = request.causation_chain\n"
        "    if self.lineage:\n"
        "        terminate_child(request.target)\n\n"
        "def adapter(request, state):\n"
        "    state['lineage'] = request.causation_chain\n"
        "    if state['lineage']:\n"
        "        cancel_task(request.target)\n"
    )
    loop_aliases = ast.parse(
        "def execute(request, target):\n"
        "    for frame in request.causation_chain:\n"
        "        if frame.agent_id == target:\n"
        "            terminate_child(target)\n\n"
        "async def dispatch(request, target):\n"
        "    async for frame in request.orchestrator_frames:\n"
        "        if frame.agent_id == target:\n"
        "            stop_peer(target)\n"
    )
    propagation_only_loop = ast.parse(
        "def dispatch(request, metadata):\n"
        "    for frame in request.causation_chain:\n"
        "        metadata.setdefault('frames', []).append(frame)\n"
    )
    assert _authority_provenance_lines(enclosing) == {2}
    assert _authority_provenance_lines(metadata_key) == {2}
    assert _authority_provenance_lines(direct_return) == {2}
    assert _authority_provenance_lines(ordinary_helpers) == {2, 5}
    assert _authority_provenance_lines(control_handlers) == {2, 7, 11}
    assert _authority_provenance_lines(canonical_frame) == {2}
    assert _authority_provenance_lines(compared_alias) == {2, 3}
    assert _authority_provenance_lines(permission_call_alias) == {3}
    assert _authority_provenance_lines(collection_wrappers) == {3, 8, 12}
    assert _authority_provenance_lines(derived_values) == {3, 8, 12}
    assert _authority_provenance_lines(authority_helper) == {2}
    assert _authority_provenance_lines(access_helper) == {2}
    assert _authority_provenance_lines(require_call) == {3}
    assert _authority_provenance_lines(dynamic_metadata_key) == {3}
    assert _authority_provenance_lines(mandate_verifier) == {2}
    assert _authority_provenance_lines(helper_return) == {2, 3}
    assert _authority_provenance_lines(permission_assignment) == {2}
    assert _authority_provenance_lines(permission_mapping) == {2}
    assert _authority_provenance_lines(permission_match) == {2}
    assert _authority_provenance_lines(unrelated_attribute_assignment) == set()
    assert _authority_provenance_lines(neutral_guarded_control) == {2, 6, 9}
    assert _authority_provenance_lines(propagation_only_guard) == set()
    assert _authority_provenance_lines(stateful_aliases) == {3, 8}
    assert _authority_provenance_lines(loop_aliases) == {2, 3, 7, 8}
    assert _authority_provenance_lines(propagation_only_loop) == set()


def test_provenance_scanner_closes_direct_state_loop_and_comprehension_bypasses(
) -> None:
    direct_state = ast.parse(
        "def disable(request, manager, target):\n"
        "    if request.causation_chain:\n"
        "        manager._agents[target].enabled = False\n\n"
        "def remove(request, manager, target):\n"
        "    if request.causation_chain:\n"
        "        del manager._agents[target]\n\n"
        "def pop(request, manager, target):\n"
        "    if request.causation_chain:\n"
        "        manager._agents.pop(target)\n\n"
        "def setattr_entry(request, manager, target):\n"
        "    if request.causation_chain:\n"
        "        setattr(manager._agents[target], 'enabled', False)\n"
    )
    loop_else = ast.parse(
        "def dispatch(request, target):\n"
        "    for _ in request.causation_chain:\n"
        "        break\n"
        "    else:\n"
        "        return\n"
        "    terminate_child(target)\n"
    )
    comprehension = ast.parse(
        "def dispatch(request):\n"
        "    return [terminate_child(frame.agent_id) "
        "for frame in request.causation_chain]\n"
    )
    nested_comprehension = ast.parse(
        "def dispatch(request):\n"
        "    return [terminate_child(child) "
        "for frame in request.causation_chain "
        "for child in frame.children]\n"
    )
    mapped_control = ast.parse(
        "def dispatch(request):\n"
        "    return list(map(terminate_child, request.causation_chain))\n"
    )

    assert _authority_provenance_lines(direct_state) == {2, 6, 10, 14}
    assert _authority_provenance_lines(loop_else) == {2}
    assert _authority_provenance_lines(comprehension) == {2}
    assert _authority_provenance_lines(nested_comprehension) == {2}
    assert _authority_provenance_lines(mapped_control) == {2}


def test_provenance_scanner_covers_parent_controls_and_boolean_reducers() -> None:
    authority_vocabulary = ast.parse(
        "def control_runtime(request):\n"
        "    if request.causation_chain:\n"
        "        mutate_runtime()\n\n"
        "def verify_parent(request):\n"
        "    if request.orchestrator:\n"
        "        mutate_runtime()\n"
    )
    boolean_reducers = ast.parse(
        "def dispatch(request, target):\n"
        "    decision = any(\n"
        "        frame.agent_id == target\n"
        "        for frame in request.causation_chain\n"
        "    )\n"
        "    if decision:\n"
        "        terminate_child(target)\n\n"
        "def adapter(request, target):\n"
        "    result = all(check(frame) for frame in request.orchestrator_frames)\n"
        "    if not result:\n"
        "        stop_peer(target)\n"
    )
    neutral_predicate = ast.parse(
        "def dispatch(request, target):\n"
        "    decision = compare_lineage(request.causation_chain, target)\n"
        "    if decision:\n"
        "        terminate_child(target)\n"
    )
    chained_neutral_predicate = ast.parse(
        "def dispatch(request, target):\n"
        "    decision = compare_lineage(request.causation_chain, target)\n"
        "    normalized = bool(decision)\n"
        "    if normalized:\n"
        "        terminate_child(target)\n"
    )
    selected_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    callback = (\n"
        "        terminate_child if request.causation_chain else noop\n"
        "    )\n"
        "    callback(target)\n"
    )
    short_circuit_controls = ast.parse(
        "def dispatch(request, target):\n"
        "    request.causation_chain and terminate_child(target)\n\n"
        "def adapter(request, target):\n"
        "    return request.orchestrator or stop_peer(target)\n"
    )
    comprehension_control = ast.parse(
        "def dispatch(request, targets):\n"
        "    return [\n"
        "        terminate_child(target)\n"
        "        for target in targets\n"
        "        if request.causation_chain\n"
        "    ]\n"
    )
    propagation_only_helper = ast.parse(
        "def dispatch(request, metadata):\n"
        "    result = emit_event(causation=request.causation_chain)\n"
        "    if result:\n"
        "        metadata['emitted'] = result\n"
    )
    neutral_control_forms = ast.parse(
        "def match_dispatch(request, target):\n"
        "    decision = compare_lineage(request.causation_chain, target)\n"
        "    match decision:\n"
        "        case True:\n"
        "            terminate_child(target)\n\n"
        "def loop_dispatch(request):\n"
        "    targets = select_targets(request.orchestrator)\n"
        "    for target in targets:\n"
        "        stop_peer(target)\n\n"
        "def comprehension_dispatch(request, targets):\n"
        "    decision = compare_lineage(request.causation_chain)\n"
        "    return [\n"
        "        terminate_child(target)\n"
        "        for target in targets\n"
        "        if decision\n"
        "    ]\n"
    )
    ownership_predicate = ast.parse(
        "def is_owner(request):\n"
        "    return bool(request.causation_chain)\n\n"
        "def dispatch(request, target):\n"
        "    if is_owner(request):\n"
        "        terminate_child(target)\n"
    )
    guard_clauses = ast.parse(
        "def execute(request):\n"
        "    if not request.causation_chain:\n"
        "        return None\n"
        "    terminate_child(request.target)\n\n"
        "def adapter(request):\n"
        "    if request.orchestrator:\n"
        "        raise PermissionError\n"
        "    manager.remove_agent(request.target)\n\n"
        "def asserted(request):\n"
        "    decision = compare_lineage(\n"
        "        request.causation_chain, request.target\n"
        "    )\n"
        "    assert decision\n"
        "    stop_peer(request.target)\n"
    )
    loop_guard_clauses = ast.parse(
        "def dispatch(request, targets):\n"
        "    for target in targets:\n"
        "        if not request.causation_chain:\n"
        "            continue\n"
        "        terminate_child(target)\n\n"
        "def adapter(request, targets):\n"
        "    for target in targets:\n"
        "        if request.orchestrator:\n"
        "            break\n"
        "        stop_peer(target)\n"
    )
    match_guards = ast.parse(
        "def match_dispatch(request, target):\n"
        "    match target:\n"
        "        case _ if request.causation_chain:\n"
        "            terminate_child(target)\n\n"
        "def alias_dispatch(request, target):\n"
        "    decision = compare_lineage(request.causation_chain, target)\n"
        "    match target:\n"
        "        case _ if decision:\n"
        "            stop_peer(target)\n"
    )
    branch_assigned_decision = ast.parse(
        "def dispatch(request, target):\n"
        "    allowed = False\n"
        "    if request.causation_chain:\n"
        "        allowed = True\n"
        "    if allowed:\n"
        "        terminate_child(target)\n"
    )

    assert _authority_provenance_lines(authority_vocabulary) == {2, 6}
    assert _authority_provenance_lines(boolean_reducers) == {6, 11}
    assert _authority_provenance_lines(neutral_predicate) == {3}
    assert _authority_provenance_lines(chained_neutral_predicate) == {4}
    assert _authority_provenance_lines(selected_callback) == {3}
    assert _authority_provenance_lines(short_circuit_controls) == {2, 5}
    assert _authority_provenance_lines(comprehension_control) == {2}
    assert _authority_provenance_lines(propagation_only_helper) == set()
    assert _authority_provenance_lines(neutral_control_forms) == {3, 9, 10, 14}
    assert _authority_provenance_lines(ownership_predicate) == {2, 5}
    assert _authority_provenance_lines(guard_clauses) == {2, 7, 15}
    assert _authority_provenance_lines(loop_guard_clauses) == {3, 9}
    assert _authority_provenance_lines(match_guards) == {3, 9}
    assert _authority_provenance_lines(branch_assigned_decision) == {5}


def test_provenance_scanner_analyzes_nested_scopes_independently() -> None:
    propagation_only_nested_helper = ast.parse(
        "def authorize_request(request):\n"
        "    def serialize_context():\n"
        "        return request.causation_chain\n"
        "    return serialize_context\n"
    )
    nested_authority = ast.parse(
        "def adapter(request):\n"
        "    def control_runtime():\n"
        "        if request.causation_chain:\n"
        "            mutate_runtime()\n"
        "    return control_runtime\n"
    )

    assert _authority_provenance_lines(propagation_only_nested_helper) == set()
    assert _authority_provenance_lines(nested_authority) == {3}


@pytest.mark.parametrize(
    ("inner_signature", "decision"),
    [
        ("target", "lineage"),
        ("target, allowed=lineage", "allowed"),
    ],
    ids=["free-variable", "default-capture"],
)
def test_provenance_scanner_carries_outer_aliases_into_closures(
    inner_signature: str,
    decision: str,
) -> None:
    tree = ast.parse(
        "def outer(request, target):\n"
        "    lineage = request.causation_chain\n"
        f"    def inner({inner_signature}):\n"
        f"        if {decision}:\n"
        "            terminate_child(target)\n"
        "    return inner\n"
    )

    assert _authority_provenance_lines(tree) == {4}


@pytest.mark.parametrize(
    "signature",
    [
        "target, allowed=bool(causation_chain)",
        "target, provider=_get_current_chain",
        "target, *, allowed=bool(causation_chain)",
    ],
    ids=["positional-value", "accessor-callable", "keyword-only-value"],
)
def test_provenance_scanner_seeds_parameter_defaults(signature: str) -> None:
    decision = "provider()" if "provider" in signature else "allowed"
    tree = ast.parse(
        f"def handle({signature}):\n"
        f"    if {decision}:\n"
        "        terminate_child(target)\n"
    )

    assert _authority_provenance_lines(tree) == {2}


def test_provenance_scanner_propagates_object_state_across_methods() -> None:
    tree = ast.parse(
        "class Gate:\n"
        "    def capture(self, request):\n"
        "        self.lineage = request.causation_chain\n\n"
        "    def terminate_child(self, target):\n"
        "        if self.lineage:\n"
        "            target.shutdown()\n"
    )

    assert _authority_provenance_lines(tree) == {6}


def test_provenance_scanner_propagates_static_setattr_state_across_methods() -> None:
    tree = ast.parse(
        "class Gate:\n"
        "    def capture(self, request):\n"
        "        setattr(self, 'ready', bool(request.causation_chain))\n\n"
        "    def run(self, target):\n"
        "        if self.ready:\n"
        "            target.stop()\n"
    )

    assert _authority_provenance_lines(tree) == {6}


def test_provenance_scanner_propagates_container_state_across_methods() -> None:
    tree = ast.parse(
        "class Gate:\n"
        "    def capture(self, request):\n"
        "        self.flags.update(\n"
        "            {'allowed': bool(request.causation_chain)}\n"
        "        )\n\n"
        "    def run(self, target):\n"
        "        if self.flags['allowed']:\n"
        "            target.stop()\n"
    )

    assert _authority_provenance_lines(tree) == {8}


def test_provenance_scanner_propagates_state_through_module_globals() -> None:
    tree = ast.parse(
        "ready = False\n"
        "def capture(request):\n"
        "    global ready\n"
        "    ready = bool(request.causation_chain)\n\n"
        "def run(target):\n"
        "    if ready:\n"
        "        target.stop()\n"
    )

    assert _authority_provenance_lines(tree) == {7}


def test_provenance_scanner_follows_starred_assignment_targets() -> None:
    tree = ast.parse(
        "def run(request, target):\n"
        "    head, *ancestors = request.causation_chain\n"
        "    if ancestors:\n"
        "        target.stop()\n"
    )

    assert _authority_provenance_lines(tree) == {3}


@pytest.mark.parametrize("keyword", ["with", "async with"])
def test_provenance_scanner_inspects_context_manager_guards(
    keyword: str,
) -> None:
    function_prefix = "async " if keyword.startswith("async") else ""
    tree = ast.parse(
        f"{function_prefix}def handle(request, target):\n"
        f"    {keyword} scope(request.causation_chain):\n"
        "        target.shutdown()\n"
    )

    assert _authority_provenance_lines(tree) == {2}


def test_provenance_scanner_does_not_treat_timeout_as_authority_context() -> None:
    tree = ast.parse(
        "ORCHESTRATOR_TURN_TIMEOUT_SECS = 30\n"
        "async def handle(target):\n"
        "    call_timeout = ORCHESTRATOR_TURN_TIMEOUT_SECS\n"
        "    async with asyncio.timeout(call_timeout):\n"
        "        target.shutdown()\n"
    )

    assert _authority_provenance_lines(tree) == set()


def test_provenance_scanner_follows_control_callback_aliases() -> None:
    callbacks = ast.parse(
        "def prebound(request, target):\n"
        "    callback = terminate_child\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n\n"
        "def branch_assigned(request, target):\n"
        "    if request.orchestrator:\n"
        "        callback = stop_peer\n"
        "    else:\n"
        "        callback = noop\n"
        "    callback(target)\n"
    )

    assert _authority_provenance_lines(callbacks) == {3, 7}

    parameter_callback = ast.parse(
        "def apply(callback, target):\n"
        "    callback(target)\n\n"
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        apply(terminate_child, target)\n"
    )
    keyword_callback = ast.parse(
        "def apply(*, callback, target):\n"
        "    callback(target)\n\n"
        "def dispatch(request, target):\n"
        "    if request.orchestrator:\n"
        "        apply(callback=stop_peer, target=target)\n"
    )

    assert _authority_provenance_lines(parameter_callback) == {5}
    assert _authority_provenance_lines(keyword_callback) == {5}


def test_provenance_scanner_follows_mutable_container_writes() -> None:
    append_decision = ast.parse(
        "def dispatch(request, target):\n"
        "    decisions = []\n"
        "    decisions.append(bool(request.causation_chain))\n"
        "    if any(decisions):\n"
        "        terminate_child(target)\n"
    )
    update_decision = ast.parse(
        "def dispatch(request, target):\n"
        "    decisions = {}\n"
        "    decisions.update(\n"
        "        {'allowed': bool(request.causation_chain)}\n"
        "    )\n"
        "    if decisions['allowed']:\n"
        "        terminate_child(target)\n"
    )
    setitem_decision = ast.parse(
        "def dispatch(request, target):\n"
        "    decisions = {}\n"
        "    decisions.__setitem__(\n"
        "        'allowed', bool(request.causation_chain)\n"
        "    )\n"
        "    if decisions['allowed']:\n"
        "        terminate_child(target)\n"
    )
    setattr_decision = ast.parse(
        "def dispatch(request, target, decisions):\n"
        "    setattr(\n"
        "        decisions, 'allowed', bool(request.causation_chain)\n"
        "    )\n"
        "    if decisions.allowed:\n"
        "        terminate_child(target)\n"
    )
    aliased_decision = ast.parse(
        "def dispatch(request, target):\n"
        "    decisions = []\n"
        "    alias = decisions\n"
        "    alias.append(bool(request.causation_chain))\n"
        "    if any(decisions):\n"
        "        terminate_child(target)\n"
    )
    rebound_alias_decision = ast.parse(
        "def dispatch(request, target):\n"
        "    decisions = []\n"
        "    alias = decisions\n"
        "    alias.append(bool(request.causation_chain))\n"
        "    alias = []\n"
        "    if any(decisions):\n"
        "        terminate_child(target)\n"
    )

    assert _authority_provenance_lines(append_decision) == {4}
    assert _authority_provenance_lines(update_decision) == {6}
    assert _authority_provenance_lines(setitem_decision) == {6}
    assert _authority_provenance_lines(setattr_decision) == {5}
    assert _authority_provenance_lines(aliased_decision) == {5}
    assert _authority_provenance_lines(rebound_alias_decision) == {6}


def test_provenance_scanner_follows_higher_order_control_dispatch() -> None:
    callbacks = ast.parse(
        "def dispatch(request, executor, loop, target):\n"
        "    if request.causation_chain:\n"
        "        executor.submit(terminate_child, target)\n"
        "    if request.orchestrator:\n"
        "        loop.call_soon(stop_peer, target)\n"
    )
    delayed_callbacks = ast.parse(
        "def dispatch(request, loop, target):\n"
        "    if request.causation_chain:\n"
        "        loop.call_later(0, terminate_child, target)\n"
        "    if request.orchestrator:\n"
        "        loop.run_in_executor(None, stop_peer, target)\n"
    )

    assert _authority_provenance_lines(callbacks) == {2, 4}
    assert _authority_provenance_lines(delayed_callbacks) == {2, 4}


def test_provenance_scanner_follows_controls_stored_in_containers() -> None:
    subscript_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    callbacks = {}\n"
        "    callbacks['run'] = terminate_child\n"
        "    if request.causation_chain:\n"
        "        callbacks['run'](target)\n"
    )
    updated_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    callbacks = {}\n"
        "    callbacks.update({'run': terminate_child})\n"
        "    if request.causation_chain:\n"
        "        callbacks['run'](target)\n"
    )
    default_callbacks = ast.parse(
        "def dispatch(\n"
        "    request, target, callback=terminate_child, *, fallback=stop_peer\n"
        "):\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n"
        "    if request.orchestrator:\n"
        "        fallback(target)\n"
    )
    aliased_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    callbacks = {}\n"
        "    alias = callbacks\n"
        "    alias['run'] = terminate_child\n"
        "    if request.causation_chain:\n"
        "        callbacks['run'](target)\n"
    )
    attribute_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    self.callbacks.run = terminate_child\n"
        "    if request.causation_chain:\n"
        "        self.callbacks.run(target)\n"
    )
    rebound_alias_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    callbacks = []\n"
        "    alias = callbacks\n"
        "    alias.append(terminate_child)\n"
        "    alias = []\n"
        "    if request.causation_chain:\n"
        "        callbacks[0](target)\n"
    )

    assert _authority_provenance_lines(subscript_callback) == {4}
    assert _authority_provenance_lines(updated_callback) == {4}
    assert _authority_provenance_lines(default_callbacks) == {4, 6}
    assert _authority_provenance_lines(aliased_callback) == {5}
    assert _authority_provenance_lines(attribute_callback) == {3}
    assert _authority_provenance_lines(rebound_alias_callback) == {6}


def test_provenance_scanner_follows_controls_selected_with_mapping_get() -> None:
    assigned_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    callbacks = {'run': terminate_child}\n"
        "    callback = callbacks.get('run')\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n"
    )
    direct_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    callbacks = {'run': stop_peer}\n"
        "    if request.orchestrator:\n"
        "        callbacks.get('run')(target)\n"
    )
    benign_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    callbacks = {'run': record_metric}\n"
        "    callback = callbacks.get('run')\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n"
    )
    fallback_callback = ast.parse(
        "def dispatch(request, callbacks, target):\n"
        "    callback = callbacks.get('run', terminate_child)\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n"
    )

    assert _authority_provenance_lines(assigned_callback) == {4}
    assert _authority_provenance_lines(direct_callback) == {3}
    assert _authority_provenance_lines(benign_callback) == set()
    assert _authority_provenance_lines(fallback_callback) == {3}


def test_provenance_scanner_carries_outer_continuations_into_nested_guards() -> None:
    with_guard = ast.parse(
        "def dispatch(request, target, lock):\n"
        "    with lock:\n"
        "        if not request.causation_chain:\n"
        "            return\n"
        "    terminate_child(target)\n"
    )
    try_guard = ast.parse(
        "def dispatch(request, target):\n"
        "    try:\n"
        "        if not request.orchestrator:\n"
        "            raise PermissionError\n"
        "    finally:\n"
        "        cleanup()\n"
        "    stop_peer(target)\n"
    )
    derived_with_guard = ast.parse(
        "def dispatch(request, target, lock):\n"
        "    decision = compare_lineage(request.causation_chain)\n"
        "    with lock:\n"
        "        if not decision:\n"
        "            return\n"
        "    terminate_child(target)\n"
    )
    guarded_match_exit = ast.parse(
        "def dispatch(request, target):\n"
        "    match target:\n"
        "        case _ if not request.causation_chain:\n"
        "            return\n"
        "    terminate_child(target)\n"
    )

    assert _authority_provenance_lines(with_guard) == {3}
    assert _authority_provenance_lines(try_guard) == {3}
    assert _authority_provenance_lines(derived_with_guard) == {4}
    assert _authority_provenance_lines(guarded_match_exit) == {3}


def test_provenance_scanner_taints_control_targets_selected_by_provenance() -> None:
    branch_selected_target = ast.parse(
        "def dispatch(request, candidate, own):\n"
        "    destination = own\n"
        "    if request.causation_chain:\n"
        "        destination = candidate\n"
        "    terminate_child(destination)\n"
    )
    conditional_target = ast.parse(
        "def dispatch(request, candidate, own):\n"
        "    destination = candidate if request.orchestrator else own\n"
        "    stop_peer(destination)\n"
    )
    conditional_kwargs = ast.parse(
        "def dispatch(request, candidate):\n"
        "    kwargs = {'target': candidate} if request.causation_chain else {}\n"
        "    terminate_child(**kwargs)\n"
    )
    branch_selected_member = ast.parse(
        "def dispatch(request, candidate):\n"
        "    params = {}\n"
        "    if request.orchestrator:\n"
        "        params['target'] = candidate\n"
        "    stop_peer(**params)\n"
    )

    assert _authority_provenance_lines(branch_selected_target) == {5}
    assert _authority_provenance_lines(conditional_target) == {3}
    assert _authority_provenance_lines(conditional_kwargs) == {3}
    assert _authority_provenance_lines(branch_selected_member) == {5}


def test_provenance_scanner_recognizes_generic_lifecycle_control_sinks() -> None:
    target_shutdown = ast.parse(
        "async def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        await target.shutdown()\n"
    )
    process_kill = ast.parse(
        "async def dispatch(request, manager, target):\n"
        "    if request.orchestrator:\n"
        "        await manager.kill_process(target)\n"
    )

    assert _authority_provenance_lines(target_shutdown) == {2}
    assert _authority_provenance_lines(process_kill) == {2}


@pytest.mark.parametrize(
    ("source", "expected_line"),
    [
        (
            "ACTION = 'terminate_child'\n"
            "def handle(request, manager, target):\n"
            "    if request.causation_chain:\n"
            "        getattr(manager, ACTION)(target)\n",
            3,
        ),
        (
            "def handle(request, manager, target, action):\n"
            "    if request.causation_chain:\n"
            "        getattr(manager, action)(target)\n",
            2,
        ),
    ],
    ids=["module-constant", "runtime-name"],
)
def test_provenance_scanner_fails_closed_on_dynamic_control_attributes(
    source: str,
    expected_line: int,
) -> None:
    assert _authority_provenance_lines(ast.parse(source)) == {expected_line}


def test_provenance_scanner_follows_local_helper_return_values() -> None:
    neutral_helper = ast.parse(
        "def derive(request):\n"
        "    return bool(request.causation_chain)\n\n"
        "def dispatch(request, target):\n"
        "    allowed = derive(request)\n"
        "    if allowed:\n"
        "        terminate_child(target)\n"
    )
    wrapped_helper = ast.parse(
        "def extract(context):\n"
        "    return context.orchestrator\n\n"
        "def normalize(context):\n"
        "    return bool(extract(context))\n\n"
        "def dispatch(request, manager, target):\n"
        "    decision = normalize(request)\n"
        "    if decision:\n"
        "        manager.kill_process(target)\n"
    )
    direct_helper_guard = ast.parse(
        "def derive(request):\n"
        "    return bool(request.causation_chain)\n\n"
        "async def dispatch(request, target):\n"
        "    if not derive(request):\n"
        "        return\n"
        "    await target.shutdown()\n"
    )
    propagation_only_helper = ast.parse(
        "def emit(context):\n"
        "    return publish(causation=context.causation_chain)\n\n"
        "def dispatch(request, state):\n"
        "    emitted = emit(request)\n"
        "    if emitted:\n"
        "        state['sent'] = True\n"
    )

    assert _authority_provenance_lines(neutral_helper) == {5, 6}
    assert _authority_provenance_lines(wrapped_helper) == {9}
    assert _authority_provenance_lines(direct_helper_guard) == {5}
    assert _authority_provenance_lines(propagation_only_helper) == set()


def test_provenance_scanner_covers_direct_targets_accessors_and_control_helpers() -> None:
    direct_target = ast.parse(
        "def dispatch(request):\n"
        "    terminate_child(request.causation_chain[-1].agent_id)\n"
    )
    neutral_keyword_target = ast.parse(
        "def dispatch(request):\n"
        "    terminate_child(\n"
        "        candidate=request.causation_chain[-1].agent_id\n"
        "    )\n"
    )
    canonical_accessors = ast.parse(
        "def dispatch(self, target):\n"
        "    if self._get_current_chain():\n"
        "        target.shutdown()\n\n"
        "def adapter(self, target):\n"
        "    chain = self._provide_causation_chain()\n"
        "    if chain:\n"
        "        target.shutdown()\n"
    )
    neutral_control_helpers = ast.parse(
        "def apply(manager, target):\n"
        "    manager.kill_process(target)\n\n"
        "def wrapped(manager, target):\n"
        "    apply(manager, target)\n\n"
        "def dispatch(request, manager, target):\n"
        "    if request.causation_chain:\n"
        "        wrapped(manager, target)\n"
    )
    mapped_callback = ast.parse(
        "def dispatch(request, handlers, target):\n"
        "    if request.orchestrator:\n"
        "        handlers['terminate_child'](target)\n"
    )
    helper_selected_target = ast.parse(
        "def dispatch(request):\n"
        "    target = choose(request.causation_chain)\n"
        "    terminate_child(target)\n"
    )
    container_selected_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    callbacks = [noop, terminate_child]\n"
        "    callback = callbacks[bool(request.causation_chain)]\n"
        "    callback(target)\n"
    )

    assert _authority_provenance_lines(direct_target) == {2}
    assert _authority_provenance_lines(neutral_keyword_target) == {2}
    assert _authority_provenance_lines(canonical_accessors) == {2, 7}
    assert _authority_provenance_lines(neutral_control_helpers) == {8}
    assert _authority_provenance_lines(mapped_callback) == {2}
    assert _authority_provenance_lines(helper_selected_target) == {3}
    assert _authority_provenance_lines(container_selected_callback) == {4}


def test_provenance_scanner_follows_callable_control_factories() -> None:
    getattr_callback = ast.parse(
        "def dispatch(request, manager, target):\n"
        "    callback = getattr(manager, 'terminate_child')\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n"
    )
    partial_callback = ast.parse(
        "from functools import partial\n\n"
        "def dispatch(request, manager, target):\n"
        "    callback = partial(manager.terminate_child)\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n"
    )
    direct_getattr = ast.parse(
        "def dispatch(request, manager, target):\n"
        "    if request.causation_chain:\n"
        "        getattr(manager, 'terminate_child')(target)\n"
    )

    assert _authority_provenance_lines(getattr_callback) == {3}
    assert _authority_provenance_lines(partial_callback) == {5}
    assert _authority_provenance_lines(direct_getattr) == {2}


def test_provenance_scanner_follows_imported_controls_and_control_lambdas() -> None:
    imported_alias = ast.parse(
        "from lifecycle import terminate_child as apply\n\n"
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        apply(target)\n"
    )
    imported_remove_alias = ast.parse(
        "from lifecycle import remove_agent as apply\n\n"
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        apply(target)\n"
    )
    lambda_callback = ast.parse(
        "def dispatch(request, manager, target):\n"
        "    callback = lambda: manager.terminate_child(target)\n"
        "    if request.causation_chain:\n"
        "        callback()\n"
    )
    neutral_remove_helper = ast.parse(
        "def apply(manager, target):\n"
        "    manager.remove_agent(target)\n\n"
        "def dispatch(request, manager, target):\n"
        "    if request.causation_chain:\n"
        "        apply(manager, target)\n"
    )
    lambda_authority = ast.parse(
        "def dispatch(request):\n"
        "    callback = lambda: terminate_child(\n"
        "        request.causation_chain[-1].agent_id\n"
        "    )\n"
        "    callback()\n"
    )
    conditional_lambda_authority = ast.parse(
        "def dispatch(request, target):\n"
        "    callback = lambda: (\n"
        "        terminate_child(target) if request.causation_chain else None\n"
        "    )\n"
        "    callback()\n"
    )
    aliased_lambda_authority = ast.parse(
        "def dispatch(request, target):\n"
        "    callback = lambda: terminate_child(\n"
        "        request.causation_chain[-1].agent_id\n"
        "    )\n"
        "    adapter = callback\n"
        "    adapter()\n"
    )

    assert _authority_provenance_lines(imported_alias) == {4}
    assert _authority_provenance_lines(imported_remove_alias) == {4}
    assert _authority_provenance_lines(lambda_callback) == {3}
    assert _authority_provenance_lines(neutral_remove_helper) == {5}
    assert _authority_provenance_lines(lambda_authority) == {2}
    assert _authority_provenance_lines(conditional_lambda_authority) == {2}
    assert _authority_provenance_lines(aliased_lambda_authority) == {2}


def test_provenance_scanner_resolves_accessor_aliases() -> None:
    assigned_accessor = ast.parse(
        "def dispatch(self, target):\n"
        "    context = self._get_current_chain\n"
        "    if context():\n"
        "        terminate_child(target)\n"
    )
    imported_accessor = ast.parse(
        "from context import get_current_chain as context\n\n"
        "def dispatch(target):\n"
        "    if context():\n"
        "        terminate_child(target)\n"
    )
    module_accessor = ast.parse(
        "context = runtime._get_current_chain\n\n"
        "def dispatch(target):\n"
        "    if context():\n"
        "        terminate_child(target)\n"
    )

    assert _authority_provenance_lines(assigned_accessor) == {3}
    assert _authority_provenance_lines(imported_accessor) == {4}
    assert _authority_provenance_lines(module_accessor) == {4}


def test_provenance_scanner_tracks_getattr_accessor_invocations() -> None:
    assigned_result = ast.parse(
        "def dispatch(self, target):\n"
        "    provider = getattr(self.agent, '_provide_causation_chain', None)\n"
        "    chain = provider()\n"
        "    if chain:\n"
        "        terminate_child(target)\n"
    )
    direct_guard = ast.parse(
        "def dispatch(self, target):\n"
        "    provider = getattr(self.agent, '_get_current_chain', None)\n"
        "    if provider():\n"
        "        terminate_child(target)\n"
    )

    assert _authority_provenance_lines(assigned_result) == {4}
    assert _authority_provenance_lines(direct_guard) == {3}


def test_provenance_scanner_follows_repository_local_imported_helpers(
    tmp_path: Path,
) -> None:
    lineage_path = tmp_path / "lineage.py"
    lineage_path.write_text(
        "def read(request):\n"
        "    return bool(request.causation_chain)\n",
        encoding="utf-8",
    )
    helpers_path = tmp_path / "helpers.py"
    helpers_path.write_text(
        "from .lineage import read\n\n"
        "def derive(request):\n"
        "    return read(request)\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "controller.py"
    source_path.write_text(
        "from .helpers import derive as decide\n\n"
        "def dispatch(request, target):\n"
        "    if decide(request):\n"
        "        terminate_child(target)\n",
        encoding="utf-8",
    )

    assert _cached_authority_provenance_lines(source_path) == frozenset({4})

    package_path = tmp_path / "pkg"
    package_path.mkdir()
    (package_path / "__init__.py").write_text("", encoding="utf-8")
    (package_path / "lineage.py").write_text(
        "def derive(request):\n"
        "    return bool(request.causation_chain)\n",
        encoding="utf-8",
    )
    dotted_path = tmp_path / "dotted_controller.py"
    dotted_path.write_text(
        "import pkg.lineage\n\n"
        "def dispatch(request, target):\n"
        "    if pkg.lineage.derive(request):\n"
        "        terminate_child(target)\n",
        encoding="utf-8",
    )

    assert _cached_authority_provenance_lines(dotted_path) == frozenset({4})


def test_provenance_scanner_follows_repository_local_control_helpers(
    tmp_path: Path,
) -> None:
    controls_path = tmp_path / "controls.py"
    controls_path.write_text(
        "def apply(manager, target):\n"
        "    manager.terminate_child(target)\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "controller.py"
    source_path.write_text(
        "from controls import apply\n\n"
        "def dispatch(request, manager, target):\n"
        "    if request.causation_chain:\n"
        "        apply(manager, target)\n",
        encoding="utf-8",
    )

    assert _cached_authority_provenance_lines(source_path) == frozenset({4})


def test_provenance_scanner_follows_wrapped_guard_clause_exits() -> None:
    wrapped_exits = ast.parse(
        "def with_guard(request, target, lock):\n"
        "    if not request.causation_chain:\n"
        "        with lock:\n"
        "            return None\n"
        "    terminate_child(target)\n\n"
        "def try_guard(request, target):\n"
        "    if not request.causation_chain:\n"
        "        try:\n"
        "            raise PermissionError\n"
        "        finally:\n"
        "            record_denial()\n"
        "    terminate_child(target)\n"
    )
    match_exit = ast.parse(
        "def dispatch(request, target):\n"
        "    match request.causation_chain:\n"
        "        case []:\n"
        "            return None\n"
        "        case _:\n"
        "            pass\n"
        "    terminate_child(target)\n"
    )

    assert _authority_provenance_lines(wrapped_exits) == {2, 8}
    assert _authority_provenance_lines(match_exit) == {2}


def test_provenance_scanner_resolves_module_level_metadata_keys(
    tmp_path: Path,
) -> None:
    module_key = ast.parse(
        'METADATA_KEY = "causation_chain"\n\n'
        "def dispatch(metadata, target):\n"
        "    if metadata.get(METADATA_KEY):\n"
        "        terminate_child(target)\n"
    )

    assert _authority_provenance_lines(module_key) == {4}

    constants_path = tmp_path / "constants.py"
    constants_path.write_text('METADATA_KEY = "causation_chain"\n', encoding="utf-8")
    source_path = tmp_path / "dispatcher.py"
    source = (
        "from .constants import METADATA_KEY\n\n"
        "def dispatch(metadata, target):\n"
        "    if metadata.get(METADATA_KEY):\n"
        "        terminate_child(target)\n"
    )
    source_path.write_text(source, encoding="utf-8")
    assert _authority_provenance_lines(
        ast.parse(source, filename=str(source_path)), source_path
    ) == {4}
    assert _cached_authority_provenance_lines(source_path) == frozenset({4})

    module_import_path = tmp_path / "module_dispatcher.py"
    module_source = (
        "import constants\n\n"
        "def dispatch(metadata, target):\n"
        "    if metadata.get(constants.METADATA_KEY):\n"
        "        terminate_child(target)\n"
    )
    module_import_path.write_text(module_source, encoding="utf-8")
    assert _cached_authority_provenance_lines(
        module_import_path
    ) == frozenset({4})


def test_provenance_scanner_does_not_promote_metadata_transport_to_authority() -> None:
    accessor_reference = ast.parse(
        "def dispatch(self):\n"
        "    provider = getattr(self.agent, '_provide_causation_chain', None)\n"
        "    if callable(provider):\n"
        "        self.send_a2a_task()\n"
    )
    accessor_transport = ast.parse(
        "def dispatch(self, manager, target):\n"
        "    provider = getattr(self.agent, '_provide_causation_chain', None)\n"
        "    chain = provider()\n"
        "    manager.send_a2a_task(target, causation_chain=chain)\n"
    )
    local_task_plumbing = ast.parse(
        "def schedule(work):\n"
        "    asyncio.create_task(work())\n\n"
        "def record_failure():\n"
        "    schedule(write_log)\n\n"
        "def run(request):\n"
        "    reason = request.causation_chain[-1]\n"
        "    if reason:\n"
        "        record_failure()\n"
    )

    assert _authority_provenance_lines(accessor_reference) == set()
    assert _authority_provenance_lines(accessor_transport) == set()
    assert _authority_provenance_lines(local_task_plumbing) == set()


def test_provenance_scanner_tracks_direct_agent_object_mutations_and_aliases() -> None:
    tree = ast.parse(
        "def direct(request, child):\n"
        "    if request.causation_chain:\n"
        "        child.enabled = False\n\n"
        "def annotated(request, target: KestrelAgent):\n"
        "    if request.causation_chain:\n"
        "        del target.session\n\n"
        "def aliased(request, child):\n"
        "    subject = child\n"
        "    if request.causation_chain:\n"
        "        setattr(subject, 'enabled', False)\n\n"
        "def benign(request, metrics):\n"
        "    if request.causation_chain:\n"
        "        metrics.enabled = False\n"
    )

    assert _authority_provenance_lines(tree) == {2, 6, 11}


def test_provenance_scanner_tracks_qualified_agent_registry_mutations() -> None:
    tree = ast.parse(
        "def hosted(request, manager, target):\n"
        "    if request.causation_chain:\n"
        "        manager.hosted_agents[target].enabled = False\n\n"
        "def indexed(request, manager, target):\n"
        "    if request.causation_chain:\n"
        "        manager.agents_by_id[target].enabled = False\n\n"
        "def benign(request, manager, target):\n"
        "    if request.causation_chain:\n"
        "        manager.metric_registry[target].enabled = False\n"
    )

    assert _authority_provenance_lines(tree) == {2, 6}


def test_provenance_scanner_recognizes_qualified_provenance_names() -> None:
    tree = ast.parse(
        "def causation(request, parent_causation_chain, target):\n"
        "    if parent_causation_chain:\n"
        "        terminate_child(target)\n\n"
        "def orchestrator(request, is_orchestrator, target):\n"
        "    if is_orchestrator:\n"
        "        stop_peer(target)\n\n"
        "def benign(request, collaboration, target):\n"
        "    if collaboration:\n"
        "        terminate_child(target)\n"
    )

    assert _authority_provenance_lines(tree) == {2, 6}


@pytest.mark.parametrize(
    "method",
    ["start", "pause", "resume", "disable", "enable", "delete"],
)
def test_provenance_scanner_classifies_lifecycle_methods_on_agent_objects(
    method: str,
) -> None:
    controlled = ast.parse(
        "def dispatch(request, child):\n"
        "    if request.causation_chain:\n"
        f"        child.{method}()\n"
    )
    benign = ast.parse(
        "def dispatch(request, metrics):\n"
        "    if request.causation_chain:\n"
        f"        metrics.{method}()\n"
    )

    assert _authority_provenance_lines(controlled) == {2}
    assert _authority_provenance_lines(benign) == set()


@pytest.mark.parametrize(
    "method",
    [
        "close_agent",
        "delete_agent",
        "destroy_child",
        "disable_agent",
        "enable_peer",
        "pause_agent",
        "reset_child",
        "resume_child",
        "start_peer",
    ],
)
def test_provenance_scanner_classifies_manager_lifecycle_methods(
    method: str,
) -> None:
    controlled = ast.parse(
        "def dispatch(request, manager, target):\n"
        "    if request.causation_chain:\n"
        f"        manager.{method}(target)\n"
    )

    assert _authority_provenance_lines(controlled) == {2}


def test_provenance_scanner_follows_imported_class_control_helpers(
    tmp_path: Path,
) -> None:
    helper_path = tmp_path / "lifecycle.py"
    helper_path.write_text(
        "class Lifecycle:\n"
        "    @staticmethod\n"
        "    def apply(target):\n"
        "        terminate_child(target)\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "controller.py"
    source_path.write_text(
        "from .lifecycle import Lifecycle\n\n"
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        Lifecycle.apply(target)\n",
        encoding="utf-8",
    )

    assert _cached_authority_provenance_lines(source_path) == frozenset({4})


def test_provenance_scanner_follows_module_control_aliases_and_factories(
    tmp_path: Path,
) -> None:
    source = (
        "from lifecycle import terminate_child\n"
        "apply = terminate_child\n"
        "wrapped = partial(apply)\n\n"
        "def direct(request, target):\n"
        "    if request.causation_chain:\n"
        "        apply(target)\n\n"
        "def factory(request, target):\n"
        "    if request.orchestrator:\n"
        "        wrapped(target)\n"
    )
    tree = ast.parse(source)
    source_path = tmp_path / "controller.py"
    source_path.write_text(source, encoding="utf-8")

    assert _authority_provenance_lines(tree) == {6, 10}
    assert _cached_authority_provenance_lines(source_path) == frozenset({6, 10})


def test_provenance_scanner_excludes_benign_task_and_host_calls() -> None:
    tree = ast.parse(
        "def telemetry(request):\n"
        "    if request.causation_chain:\n"
        "        asyncio.create_task(record())\n\n"
        "def inspect_host(request, socket):\n"
        "    if request.causation_chain:\n"
        "        socket.gethostname()\n"
    )

    assert _authority_provenance_lines(tree) == set()


def test_causation_and_orchestrator_metadata_are_not_permission_inputs() -> None:
    """Make a direct causation-as-authority condition fail review loudly."""

    violations: list[str] = []
    for path in (REPO_ROOT / "kestrel_sovereign").rglob("*.py"):
        for line in _cached_authority_provenance_lines(path):
            violations.append(f"{path.relative_to(REPO_ROOT)}:{line}")

    assert not violations, (
        "Causation/orchestrator metadata appeared in a permission condition: "
        + ", ".join(sorted(set(violations)))
    )
