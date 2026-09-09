"""Contract tests for the checked-in cross-agent authority inventory (#3143)."""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from kestrel_sovereign.auth import AuthMethod, CallerContext
from kestrel_sovereign.command_handler import BUILTIN_COMMAND_SPECS
from kestrel_sovereign.endpoints.models import require_sovereign_host_lifecycle

pytestmark = pytest.mark.authority_audit

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "docs/architecture/CROSS_AGENT_AUTHORITY_AUDIT.md"
AUTH_SURFACE_MATRIX_PATH = REPO_ROOT / "docs/audit/AUTH_SURFACE_MATRIX.md"
_AUTHORITY_AUDIT_PATHS = tuple(
    sorted((REPO_ROOT / "kestrel_sovereign").rglob("*.py"))
)
_AUTHORITY_AUDIT_CHUNKS = tuple(
    _AUTHORITY_AUDIT_PATHS[index : index + 48]
    for index in range(0, len(_AUTHORITY_AUDIT_PATHS), 48)
)
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
    "acl",
    "approv",
    "authoriz",
    "authority",
    "capabilit",
    "delegat",
    "entitle",
    "grant",
    "mandate",
    "owner",
    "permit",
    "permission",
    "policy",
    "privilege",
    "role",
    "rule",
    "allowed",
    "permitted",
    "forbid",
    "denied",
    "access",
    "gate",
    "require",
    "rbac",
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
    "partial",
    "partialmethod",
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
TRACE_PARENT_MARKERS = (
    "parent_span",
    "parent-span",
    "parentspan",
    "parent.span",
    "trace_parent",
    "trace-parent",
    "traceparent",
    "trace.parent",
    "parent_trace",
    "parent.trace",
    "span_parent",
    "span.parent",
)
PROVENANCE_SOURCE_MARKERS = (
    "causation",
    "orchestrator",
    "current_chain",
    *TRACE_PARENT_MARKERS,
)
UNVERIFIED_ATTRIBUTION_METADATA_KEYS = frozenset(
    {
        "claimed_sender",
        "requested_by",
        "sender",
        "sender_verified",
        "source_agent",
        "source_agent_id",
    }
)
SOURCE_IDENTIFIER_CHAIN = re.compile(r"[a-z_][a-z0-9_.-]*", re.IGNORECASE)
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
    "/api/ipfs/node",
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


def _clear_authority_analysis_annotations(tree: ast.AST) -> None:
    """Remove transient dataflow facts before a cached AST is reused."""

    for node in ast.walk(tree):
        for attribute in tuple(vars(node)):
            if attribute.startswith("_authority_"):
                delattr(node, attribute)


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
    definition: ast.AST,
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

    member_reference = _static_member_reference(value)
    if member_reference is None:
        return None
    receiver, member = member_reference
    module_aliases = {
        (imported.asname or imported.name.split(".", 1)[0])
        for node in tree.body
        if isinstance(node, ast.Import)
        for imported in node.names
    }
    expression = ast.unparse(receiver)
    if any(
        expression == alias or expression.startswith(f"{alias}.")
        for alias in module_aliases
    ):
        return member
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


def _static_binding_target_names(target: ast.AST) -> set[str]:
    """Return stable keys for local names and object-attribute bindings."""

    if isinstance(target, ast.Attribute):
        return {ast.unparse(target)}
    if isinstance(target, ast.Starred):
        return _static_binding_target_names(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        return {
            name
            for element in target.elts
            for name in _static_binding_target_names(element)
        }
    return _assignment_target_names(target)


_StaticBinding = tuple[str, str]


def _static_member_reference(
    value: ast.AST,
    constants: dict[str, str] | None = None,
) -> tuple[ast.AST, str] | None:
    """Resolve ``obj.member`` and literal ``getattr(obj, member)`` alike."""

    if isinstance(value, ast.Attribute):
        return value.value, value.attr
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "getattr"
        and len(value.args) >= 2
    ):
        member = _resolved_string(value.args[1], constants or {})
        if member is not None:
            return value.args[0], member
    return None


def _compound_flow_parts(
    statement: ast.stmt,
) -> tuple[list[ast.AST], list[list[ast.stmt]]]:
    """Return eagerly evaluated expressions and possible statement branches."""

    expressions: list[ast.AST] = []
    branches: list[list[ast.stmt]] = []
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
    return expressions, branches


class _StaticBindingFlow:
    """Bounded may-analysis for authority inventory aliases.

    The scanner does not try to execute Python. It resolves exact assignment
    shapes and static member references, follows aliases as statements are
    replayed, and joins control-flow branches conservatively.
    Domain-specific resolvers identify the authority-bearing root and decide
    how an ambiguous binding must fail closed.
    """

    def __init__(
        self,
        direct_resolver: Callable[..., _StaticBinding | None],
        ambiguous: Callable[[list[_StaticBinding]], _StaticBinding],
        bindings: dict[str, _StaticBinding] | None = None,
        *,
        normalize_name: Callable[[str], str] | None = None,
        on_resolve: Callable[[ast.AST, _StaticBinding], None] | None = None,
        stateful_resolver: bool = False,
    ) -> None:
        self.direct_resolver = direct_resolver
        self.ambiguous = ambiguous
        self.normalize_name = normalize_name or (lambda name: name)
        self.on_resolve = on_resolve
        self.stateful_resolver = stateful_resolver
        self.bindings = {
            self.normalize_name(name): binding
            for name, binding in (bindings or {}).items()
        }

    def fork(self) -> _StaticBindingFlow:
        return _StaticBindingFlow(
            self.direct_resolver,
            self.ambiguous,
            self.bindings,
            normalize_name=self.normalize_name,
            on_resolve=self.on_resolve,
            stateful_resolver=self.stateful_resolver,
        )

    def resolve(
        self,
        value: ast.AST,
        supplemental_resolver: Callable[
            [ast.AST], _StaticBinding | None
        ] | None = None,
    ) -> _StaticBinding | None:
        binding = (
            supplemental_resolver(value)
            if supplemental_resolver is not None
            else None
        )
        if binding is None:
            binding = (
                self.direct_resolver(self, value)
                if self.stateful_resolver
                else self.direct_resolver(value)
            )
        if binding is None and isinstance(value, ast.NamedExpr):
            binding = self.resolve(
                value.target,
                supplemental_resolver,
            ) or self.resolve(value.value, supplemental_resolver)
        elif binding is None and isinstance(value, ast.Name):
            binding = self.bindings.get(self.normalize_name(value.id))
        elif binding is None and isinstance(value, ast.Attribute):
            binding = self.bindings.get(
                self.normalize_name(ast.unparse(value))
            )
        elif binding is None and isinstance(value, ast.IfExp):
            binding = self._merge_values([
                self.resolve(value.body, supplemental_resolver),
                self.resolve(value.orelse, supplemental_resolver),
            ])
        if binding is not None and self.on_resolve is not None:
            self.on_resolve(value, binding)
        return binding

    def _merge_values(
        self, values: list[_StaticBinding | None]
    ) -> _StaticBinding | None:
        present = [value for value in values if value is not None]
        if not present:
            return None
        if len(present) == len(values) and all(
            value == present[0] for value in present[1:]
        ):
            return present[0]
        return self.ambiguous(present)

    def assignment_bindings(
        self,
        targets: list[ast.AST],
        value: ast.AST,
        supplemental_resolver: Callable[
            [ast.AST], _StaticBinding | None
        ] | None = None,
    ) -> dict[str, _StaticBinding]:
        pairs = [
            pair
            for target in targets
            for pair in _static_assignment_pairs(target, value)
        ]
        if pairs:
            return {
                self.normalize_name(name): binding
                for name, paired_value in pairs
                if (
                    binding := self.resolve(
                        paired_value,
                        supplemental_resolver,
                    )
                )
                is not None
            }

        nested = [
            binding
            for child in ast.walk(value)
            if (
                binding := self.resolve(child, supplemental_resolver)
            ) is not None
        ]
        if not nested:
            return {}
        unresolved = self.ambiguous(nested)
        return {
            self.normalize_name(name): unresolved
            for target in targets
            for name in _static_binding_target_names(target)
        }

    def assign(
        self,
        targets: list[ast.AST],
        value: ast.AST | None,
        supplemental_resolver: Callable[
            [ast.AST], _StaticBinding | None
        ] | None = None,
    ) -> None:
        if value is not None and self.on_resolve is not None:
            self.replay_expression_bindings(value, supplemental_resolver)
        assigned = (
            self.assignment_bindings(
                targets,
                value,
                supplemental_resolver,
            )
            if value is not None
            else {}
        )
        rebound = {
            self.normalize_name(name)
            for target in targets
            for name in _static_binding_target_names(target)
        }
        for name in rebound:
            self.bindings.pop(name, None)
        self.bindings.update(assigned)

    def replay_expression_bindings(
        self,
        expression: ast.AST,
        supplemental_resolver: Callable[
            [ast.AST], _StaticBinding | None
        ] | None = None,
    ) -> None:
        """Replay walrus bindings without descending into nested scopes."""

        flow = self

        class NamedExpressionVisitor(ast.NodeVisitor):
            def generic_visit(self, node: ast.AST) -> None:
                if isinstance(node, ast.expr):
                    flow.resolve(node, supplemental_resolver)
                super().generic_visit(node)

            def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
                self.visit(node.value)
                flow.assign(
                    [node.target],
                    node.value,
                    supplemental_resolver,
                )

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

        NamedExpressionVisitor().visit(expression)

    def disturbed_fork(self, statement: ast.stmt) -> _StaticBindingFlow:
        """Model a path where a compound binding receives unknown data."""

        branch = self.fork()
        for name in _compound_binding_names(statement, set()):
            branch.bindings.pop(self.normalize_name(name), None)
        return branch

    def join(self, branches: list[_StaticBindingFlow]) -> None:
        states = [self.bindings, *(branch.bindings for branch in branches)]
        names = {name for state in states for name in state}
        self.bindings = {
            name: merged
            for name in names
            if (
                merged := self._merge_values(
                    [state.get(name) for state in states]
                )
            )
            is not None
        }

    def declare(
        self,
        statement: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> None:
        """Replay a declaration; specialized domains may preserve semantics."""

        self.bindings.pop(self.normalize_name(statement.name), None)

    def replay(
        self,
        statements: list[ast.stmt],
        supplemental_resolver: Callable[
            [ast.AST], _StaticBinding | None
        ] | None = None,
    ) -> None:
        for statement in statements:
            if isinstance(statement, _MODULE_COMPOUND_STATEMENT_TYPES):
                expressions, blocks = _compound_flow_parts(statement)
                for expression in expressions:
                    self.replay_expression_bindings(
                        expression,
                        supplemental_resolver,
                    )
                branches = []
                for index, block in enumerate(blocks):
                    branch = self.fork()
                    if index == 0:
                        _bind_static_loop_target(
                            branch,
                            statement,
                            supplemental_resolver,
                        )
                    if isinstance(statement, ast.Match):
                        _bind_static_match_pattern(
                            branch,
                            statement.cases[index].pattern,
                            statement.subject,
                            supplemental_resolver,
                        )
                    branch.replay(block, supplemental_resolver)
                    branches.append(branch)
                branches.append(self.disturbed_fork(statement))
                self.join(branches)
                continue
            if isinstance(statement, ast.Assign):
                self.assign(
                    list(statement.targets),
                    statement.value,
                    supplemental_resolver,
                )
            elif isinstance(statement, ast.AnnAssign):
                self.assign(
                    [statement.target],
                    statement.value,
                    supplemental_resolver,
                )
            elif isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                self.declare(statement)
            else:
                self.replay_expression_bindings(
                    statement,
                    supplemental_resolver,
                )


def _bind_static_loop_target(
    flow: _StaticBindingFlow,
    statement: ast.stmt,
    supplemental_resolver: Callable[
        [ast.AST], _StaticBinding | None
    ] | None = None,
) -> None:
    """Bind a loop target when every literal iteration has one exact binding."""

    if not isinstance(statement, (ast.For, ast.AsyncFor)):
        return
    target_names = _assignment_target_names(statement.target)
    for name in target_names:
        flow.bindings.pop(flow.normalize_name(name), None)
    if not isinstance(statement.iter, (ast.List, ast.Tuple, ast.Set)):
        return
    candidates = [
        flow.assignment_bindings(
            [statement.target],
            element,
            supplemental_resolver,
        )
        for element in statement.iter.elts
    ]
    if (
        candidates
        and candidates[0]
        and all(candidate == candidates[0] for candidate in candidates[1:])
    ):
        flow.bindings.update(candidates[0])


def _bind_static_match_pattern(
    flow: _StaticBindingFlow,
    pattern: ast.pattern,
    subject: ast.AST,
    supplemental_resolver: Callable[
        [ast.AST], _StaticBinding | None
    ] | None = None,
) -> None:
    """Bind exact structural-pattern captures through the shared lattice."""

    if isinstance(pattern, ast.MatchAs):
        if pattern.name is not None:
            flow.assign(
                [ast.Name(id=pattern.name, ctx=ast.Store())],
                subject,
                supplemental_resolver,
            )
        if pattern.pattern is not None:
            _bind_static_match_pattern(
                flow, pattern.pattern, subject, supplemental_resolver
            )
        return
    if isinstance(pattern, ast.MatchMapping):
        for key, child in zip(pattern.keys, pattern.patterns):
            selected = ast.Subscript(
                value=subject,
                slice=key,
                ctx=ast.Load(),
            )
            _bind_static_match_pattern(
                flow, child, selected, supplemental_resolver
            )
        if pattern.rest is not None:
            flow.assign(
                [ast.Name(id=pattern.rest, ctx=ast.Store())],
                subject,
                supplemental_resolver,
            )
        return
    if isinstance(pattern, ast.MatchSequence):
        for index, child in enumerate(pattern.patterns):
            selected = ast.Subscript(
                value=subject,
                slice=ast.Constant(value=index),
                ctx=ast.Load(),
            )
            _bind_static_match_pattern(
                flow, child, selected, supplemental_resolver
            )
        return
    if isinstance(pattern, ast.MatchClass):
        for attribute, child in zip(pattern.kwd_attrs, pattern.kwd_patterns):
            selected = ast.Attribute(
                value=subject,
                attr=attribute,
                ctx=ast.Load(),
            )
            _bind_static_match_pattern(
                flow, child, selected, supplemental_resolver
            )


def _unique_module_functions(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return unambiguous module helper definitions addressable by bare name."""

    grouped: dict[
        str, list[ast.FunctionDef | ast.AsyncFunctionDef]
    ] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            grouped.setdefault(node.name, []).append(node)
    return {
        name: definitions[0]
        for name, definitions in grouped.items()
        if len(definitions) == 1
    }


class _RegistryHelperIndex:
    """Resolve module helpers and methods from the helper's lexical class."""

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        self.module_functions = _unique_module_functions(tree)
        self.method_functions: dict[
            tuple[tuple[str, ...], str],
            ast.FunctionDef | ast.AsyncFunctionDef,
        ] = {}
        self.function_owners: dict[
            ast.FunctionDef | ast.AsyncFunctionDef, tuple[str, ...]
        ] = {}

        def collect_class(node: ast.ClassDef, owner: tuple[str, ...]) -> None:
            grouped: dict[
                str, list[ast.FunctionDef | ast.AsyncFunctionDef]
            ] = {}
            for statement in node.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    grouped.setdefault(statement.name, []).append(statement)
                    self.function_owners[statement] = owner
                elif isinstance(statement, ast.ClassDef):
                    collect_class(statement, (*owner, statement.name))
            for name, definitions in grouped.items():
                if len(definitions) == 1:
                    self.method_functions[(owner, name)] = definitions[0]

        for statement in tree.body:
            if isinstance(statement, ast.ClassDef):
                collect_class(statement, (statement.name,))

        self._alias_cache: dict[
            ast.FunctionDef | ast.AsyncFunctionDef | None,
            dict[str, set[ast.FunctionDef | ast.AsyncFunctionDef]],
        ] = {}

    def _method_reference(
        self,
        value: ast.AST,
        caller: ast.FunctionDef | ast.AsyncFunctionDef | None,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        member = _static_member_reference(value)
        if member is None or not isinstance(member[0], ast.Name):
            return None
        receiver, method_name = member
        owner = self.function_owners.get(caller) if caller is not None else None
        if receiver.id in {"self", "cls"} and owner is not None:
            return self.method_functions.get((owner, method_name))
        candidates = [
            function
            for (candidate_owner, name), function in self.method_functions.items()
            if candidate_owner[-1] == receiver.id and name == method_name
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _aliases(
        self,
        caller: ast.FunctionDef | ast.AsyncFunctionDef | None,
    ) -> dict[str, set[ast.FunctionDef | ast.AsyncFunctionDef]]:
        """Return conservative helper aliases in one lexical call scope."""

        cached = self._alias_cache.get(caller)
        if cached is not None:
            return cached
        aliases = {
            name: {function}
            for name, function in self.module_functions.items()
        }
        nodes: tuple[ast.AST, ...] | list[ast.stmt] = (
            self.tree.body if caller is None else _walk_lexical_scope(caller)
        )
        assignments: list[tuple[str, ast.AST]] = []
        for node in nodes:
            if isinstance(node, ast.Assign):
                pairs = [
                    pair
                    for target in node.targets
                    for pair in _static_assignment_pairs(target, node.value)
                ]
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                pairs = _static_assignment_pairs(node.target, node.value)
            elif isinstance(node, ast.NamedExpr):
                pairs = _static_assignment_pairs(node.target, node.value)
            else:
                pairs = []
            assignments.extend(pairs)

        def candidates(value: ast.AST) -> set[
            ast.FunctionDef | ast.AsyncFunctionDef
        ]:
            if isinstance(value, ast.Name):
                return set(aliases.get(value.id, ()))
            if isinstance(value, ast.IfExp):
                return candidates(value.body) | candidates(value.orelse)
            method = self._method_reference(value, caller)
            return {method} if method is not None else set()

        changed = True
        while changed:
            changed = False
            for target, value in assignments:
                resolved = candidates(value)
                if not resolved:
                    continue
                before = len(aliases.setdefault(target, set()))
                aliases[target].update(resolved)
                changed = changed or len(aliases[target]) != before
        self._alias_cache[caller] = aliases
        return aliases

    def resolve(
        self,
        call: ast.Call,
        caller: ast.FunctionDef | ast.AsyncFunctionDef | None,
    ) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, bool] | None:
        """Return a helper definition and whether Python binds its receiver."""

        if isinstance(call.func, ast.Name):
            function = self.module_functions.get(call.func.id)
            if function is not None:
                return function, False
            candidates = self._aliases(caller).get(call.func.id, set())
            return (next(iter(candidates)), False) if len(candidates) == 1 else None

        function = self._method_reference(call.func, caller)
        member = _static_member_reference(call.func)
        if (
            function is None
            or member is None
            or not isinstance(member[0], ast.Name)
        ):
            return None
        receiver = member[0]
        if receiver.id in {"self", "cls"}:
            is_static = any(
                (
                    decorator.id
                    if isinstance(decorator, ast.Name)
                    else decorator.func.id
                    if isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    else ""
                )
                == "staticmethod"
                for decorator in function.decorator_list
            )
            return function, not is_static
        return function, False


def _registry_helper_call_bindings(
    call: ast.Call,
    helpers: _RegistryHelperIndex,
    flow: _StaticBindingFlow,
    registry_binding: _StaticBinding,
    caller: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> tuple[
    ast.FunctionDef | ast.AsyncFunctionDef,
    dict[str, _StaticBinding],
] | None:
    """Bind registry-valued arguments to one local helper's parameters."""

    resolved = helpers.resolve(call, caller)
    if resolved is None:
        return None
    function, receiver_is_bound = resolved
    positional = [*function.args.posonlyargs, *function.args.args]
    if receiver_is_bound and positional:
        positional = positional[1:]
    all_parameters = {parameter.arg for parameter in positional}
    all_parameters.update(parameter.arg for parameter in function.args.kwonlyargs)
    bindings = {
        positional[index].arg: registry_binding
        for index, argument in enumerate(call.args)
        if index < len(positional)
        and flow.resolve(argument) == registry_binding
    }
    bindings.update(
        {
            keyword.arg: registry_binding
            for keyword in call.keywords
            if keyword.arg in all_parameters
            and flow.resolve(keyword.value) == registry_binding
        }
    )
    return (function, bindings) if bindings else None


def _function_publishes_registry(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    initial_bindings: dict[str, _StaticBinding],
    registry_binding: _StaticBinding,
    publish_methods: set[str],
    direct_resolver: Callable[[ast.AST], _StaticBinding | None],
    ambiguous: Callable[[list[_StaticBinding]], _StaticBinding],
    helpers: _RegistryHelperIndex,
    *,
    direct_assignment_member: str | None = None,
    seen: frozenset[ast.AST] = frozenset(),
) -> bool:
    """Whether a local helper publishes a registry passed by its caller."""

    if function in seen:
        return False
    active = seen | {function}
    publish_bindings = {
        method: (registry_binding[0], f"publish:{method}")
        for method in publish_methods
    }

    def bound_method(
        flow: _StaticBindingFlow,
        value: ast.AST,
    ) -> _StaticBinding | None:
        member = _static_member_reference(value)
        if (
            member is not None
            and member[1] in publish_methods
            and flow.resolve(member[0]) == registry_binding
        ):
            return publish_bindings[member[1]]
        return None

    def call_publishes(flow: _StaticBindingFlow, call: ast.Call) -> bool:
        member = _static_member_reference(call.func)
        if (
            member is not None
            and member[1] in publish_methods
            and flow.resolve(member[0]) == registry_binding
        ) or flow.resolve(call.func) in publish_bindings.values():
            return True
        helper = _registry_helper_call_bindings(
            call,
            helpers,
            flow,
            registry_binding,
            function,
        )
        return bool(
            helper is not None
            and _function_publishes_registry(
                helper[0],
                helper[1],
                registry_binding,
                publish_methods,
                direct_resolver,
                ambiguous,
                helpers,
                direct_assignment_member=direct_assignment_member,
                seen=active,
            )
        )

    def node_publishes(flow: _StaticBindingFlow, root: ast.AST) -> bool:
        class PublicationVisitor(ast.NodeVisitor):
            found = False

            def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
                if call_publishes(flow, node):
                    self.found = True
                    return
                self.generic_visit(node)

            def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
                if any(
                    isinstance(target, ast.Subscript)
                    and flow.resolve(target.value) == registry_binding
                    for target in node.targets
                ) or (
                    direct_assignment_member is not None
                    and any(
                        (member := _static_member_reference(target)) is not None
                        and member[1] == direct_assignment_member
                        for target in node.targets
                    )
                    and not (
                        isinstance(node.value, ast.Dict)
                        and not node.value.keys
                    )
                ):
                    self.found = True
                    return
                self.generic_visit(node)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
                if (
                    isinstance(node.target, ast.Subscript)
                    and flow.resolve(node.target.value) == registry_binding
                ):
                    self.found = True
                    return
                self.generic_visit(node)

            def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
                if flow.resolve(node.target) == registry_binding:
                    self.found = True
                    return
                self.generic_visit(node)

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

        visitor = PublicationVisitor()
        visitor.visit(root)
        return visitor.found

    def scan_block(
        statements: list[ast.stmt],
        flow: _StaticBindingFlow,
    ) -> bool:
        for statement in statements:
            if isinstance(statement, _MODULE_COMPOUND_STATEMENT_TYPES):
                expressions, blocks = _compound_flow_parts(statement)
                if any(node_publishes(flow, expression) for expression in expressions):
                    return True
                branches: list[_StaticBindingFlow] = []
                for index, block in enumerate(blocks):
                    branch = flow.fork()
                    supplement = lambda value, current=branch: bound_method(
                        current, value
                    )
                    if index == 0:
                        _bind_static_loop_target(branch, statement, supplement)
                    if scan_block(block, branch):
                        return True
                    branches.append(branch)
                branches.append(flow.disturbed_fork(statement))
                flow.join(branches)
                continue
            if node_publishes(flow, statement):
                return True
            supplement = lambda value, current=flow: bound_method(current, value)
            if isinstance(statement, ast.Assign):
                flow.assign(list(statement.targets), statement.value, supplement)
            elif isinstance(statement, ast.AnnAssign):
                flow.assign([statement.target], statement.value, supplement)
            else:
                flow.replay_expression_bindings(statement, supplement)
        return False

    return scan_block(
        function.body,
        _StaticBindingFlow(direct_resolver, ambiguous, initial_bindings),
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("alias = root.publish", {"alias": ("root", "publish")}),
        ("alias: object = root.publish", {"alias": ("root", "publish")}),
        (
            "left = right = root.publish",
            {
                "left": ("root", "publish"),
                "right": ("root", "publish"),
            },
        ),
        (
            "left, ignored = root.publish, noop",
            {"left": ("root", "publish")},
        ),
        (
            "alias = root.publish\ncopy = alias",
            {
                "alias": ("root", "publish"),
                "copy": ("root", "publish"),
            },
        ),
        (
            "alias = root.publish\nalias = alias",
            {"alias": ("root", "publish")},
        ),
        (
            'alias = getattr(root, "publish")',
            {"alias": ("root", "publish")},
        ),
        (
            "(alias := root.publish)",
            {"alias": ("root", "publish")},
        ),
        (
            "alias = root.publish\nif disabled:\n    alias = noop",
            {"alias": ("root", "<ambiguous>")},
        ),
        (
            "alias = root.publish\nfor alias in values:\n    pass",
            {"alias": ("root", "<ambiguous>")},
        ),
    ],
    ids=[
        "simple",
        "annotated",
        "chained",
        "destructured",
        "alias-chain",
        "self-alias",
        "static-getattr",
        "named-expression",
        "branch-join",
        "loop-target",
    ],
)
def test_static_binding_flow_adversarial_matrix(
    source: str,
    expected: dict[str, _StaticBinding],
) -> None:
    def direct(value: ast.AST) -> _StaticBinding | None:
        member_reference = _static_member_reference(value)
        if member_reference is None:
            return None
        receiver, member = member_reference
        if isinstance(receiver, ast.Name) and receiver.id == "root":
            return receiver.id, member
        return None

    flow = _StaticBindingFlow(
        direct,
        lambda bindings: (bindings[0][0], "<ambiguous>"),
    )
    flow.replay(ast.parse(source).body)

    assert flow.bindings == expected


def _static_alias_edges(tree: ast.Module) -> list[tuple[str, set[str]]]:
    """Collect exact assignment edges for monotone discovery inventories."""

    edges: list[tuple[str, set[str]]] = []
    for node in ast.walk(tree):
        assignment_pairs: list[tuple[str, ast.AST]] = []
        if isinstance(node, ast.Assign):
            assignment_pairs = [
                pair
                for target in node.targets
                for pair in _static_assignment_pairs(target, node.value)
            ]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            assignment_pairs = _static_assignment_pairs(node.target, node.value)
        for target, value in assignment_pairs:
            source = (
                value.id
                if isinstance(value, ast.Name)
                else _imported_module_attribute_name(tree, value)
            )
            if source is not None:
                edges.append((target, {source}))
    return edges


def _static_alias_closure(
    seeds: set[str],
    edges: list[tuple[str, set[str]]],
) -> set[str]:
    """Reach a bounded fixed point across statically resolved alias edges."""

    aliases = set(seeds)
    changed = True
    while changed:
        changed = False
        for target, sources in edges:
            if sources.intersection(aliases) and target not in aliases:
                aliases.add(target)
                changed = True
    return aliases


def _call_produced_decorator_bindings(
    statements: list[ast.stmt],
    is_factory_call: Callable[[ast.Call], bool],
    inherited: dict[str, ast.Call] | None = None,
) -> dict[str, ast.Call]:
    """Resolve names bound to the result of a static decorator-factory call.

    Both FastAPI registrations and ``@tool`` are callable factories: their
    first call returns a decorator that may be stored, aliased, and applied
    later.  Keep that Python binding behavior in one source-ordered flow so
    neither inventory silently depends on the two calls being adjacent.
    """

    calls_by_key: dict[str, ast.Call] = {}

    def binding_for(call: ast.Call) -> _StaticBinding:
        key = str(id(call))
        calls_by_key[key] = call
        return "decorator-factory", key

    initial_bindings = {
        name: binding_for(call) for name, call in (inherited or {}).items()
    }

    def direct(value: ast.AST) -> _StaticBinding | None:
        if isinstance(value, ast.Call) and is_factory_call(value):
            return binding_for(value)
        return None

    def ambiguous(bindings: list[_StaticBinding]) -> _StaticBinding:
        if bindings and all(binding == bindings[0] for binding in bindings[1:]):
            return bindings[0]
        return "decorator-factory", "<unresolved>"

    flow = _StaticBindingFlow(direct, ambiguous, initial_bindings)
    flow.replay(statements)
    unresolved = [
        name for name, binding in flow.bindings.items()
        if binding == ("decorator-factory", "<unresolved>")
    ]
    if unresolved:
        raise AssertionError(
            "Ambiguous stored decorator factory: "
            + ", ".join(sorted(unresolved))
        )
    return {
        name: calls_by_key[binding[1]]
        for name, binding in flow.bindings.items()
        if binding[0] == "decorator-factory" and binding[1] in calls_by_key
    }


def _tool_decorator_aliases(
    tree: ast.Module,
) -> dict[str, ast.Call | None]:
    """Resolve direct, stored, and higher-order ``tool`` decorators."""

    aliases = {"tool"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for imported in node.names:
                if imported.name == "tool":
                    aliases.add(imported.asname or imported.name)
    aliases = _static_alias_closure(aliases, _static_alias_edges(tree))

    produced: dict[str, ast.Call] = {}
    scopes = [tree.body]
    scopes.extend(
        node.body
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
    )
    for statements in scopes:
        local = _call_produced_decorator_bindings(
            statements,
            lambda call: (
                (
                    call.func.id
                    if isinstance(call.func, ast.Name)
                    else call.func.attr
                    if isinstance(call.func, ast.Attribute)
                    else ""
                )
                in aliases
            ),
        )
        for name, call in local.items():
            previous = produced.get(name)
            if previous is not None and ast.dump(
                previous, include_attributes=False
            ) != ast.dump(call, include_attributes=False):
                raise AssertionError(
                    f"Ambiguous stored @tool decorator factory: {name}"
                )
            produced[name] = call
    resolved: dict[str, ast.Call | None] = {
        **dict.fromkeys(aliases),
        **produced,
    }

    # A local decorator may apply a configured ``tool(...)`` factory to its
    # callable parameter. Runtime discovery observes the resulting schema, so
    # static discovery must preserve the same higher-order application rather
    # than keying only on the spelling used at the final ``@decorator`` site.
    changed = True
    while changed:
        changed = False
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            formal_names = {
                parameter.arg.casefold()
                for parameter in [
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                    *(
                        [function.args.vararg]
                        if function.args.vararg is not None
                        else []
                    ),
                    *(
                        [function.args.kwarg]
                        if function.args.kwarg is not None
                        else []
                    ),
                ]
            }
            assignments: dict[str, ast.AST] = {}
            for node in _walk_lexical_scope(function):
                if isinstance(node, ast.Assign):
                    pairs = [
                        pair
                        for target in node.targets
                        for pair in _static_assignment_pairs(
                            target, node.value
                        )
                    ]
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    pairs = _static_assignment_pairs(node.target, node.value)
                else:
                    continue
                assignments.update(pairs)

            def assigned_value(value: ast.AST) -> ast.AST:
                seen: set[str] = set()
                while (
                    isinstance(value, ast.Name)
                    and value.id not in seen
                    and value.id in assignments
                ):
                    seen.add(value.id)
                    value = assignments[value.id]
                return value

            factories: list[ast.Call | None] = []
            for node in _walk_lexical_scope(function):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                application = assigned_value(node.value)
                if not isinstance(application, ast.Call):
                    continue
                supplied = [
                    *application.args,
                    *(keyword.value for keyword in application.keywords),
                ]
                if not any(
                    _identifier_tokens(argument).intersection(formal_names)
                    for argument in supplied
                ):
                    continue
                factory = application.func
                public_name = _public_tool_name(
                    factory,
                    function.name,
                    decorator_aliases=resolved,
                )
                if public_name is None:
                    continue
                factory_name = (
                    factory.id
                    if isinstance(factory, ast.Name)
                    else factory.attr
                    if isinstance(factory, ast.Attribute)
                    else ""
                )
                factories.append(
                    resolved.get(factory_name)
                    if factory_name in resolved
                    else factory
                    if isinstance(factory, ast.Call)
                    else None
                )
            if not factories:
                continue
            concrete = [factory for factory in factories if factory is not None]
            if concrete and any(
                ast.dump(factory, include_attributes=False)
                != ast.dump(concrete[0], include_attributes=False)
                for factory in concrete[1:]
            ):
                raise AssertionError(
                    f"Ambiguous higher-order @tool decorator: {function.name}"
                )
            wrapper_factory = concrete[0] if concrete else None
            previous = resolved.get(function.name)
            if function.name in resolved:
                if (
                    previous is not None
                    and wrapper_factory is not None
                    and ast.dump(previous, include_attributes=False)
                    != ast.dump(wrapper_factory, include_attributes=False)
                ):
                    raise AssertionError(
                        "Ambiguous higher-order @tool decorator: "
                        f"{function.name}"
                    )
                continue
            resolved[function.name] = wrapper_factory
            changed = True
    return resolved


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
    # The conditional value is exact within the same branch that publishes
    # the route, so branch-local source-order replay can inventory it safely.
    assert _route_declarations(
        nested_route,
        _module_string_constants(nested_route),
        _module_string_collections(nested_route),
    ) == [(("DELETE",), "/api/agents/{name}/terminate")]


def _public_tool_name(
    decorator: ast.expr,
    fallback: str,
    constants: dict[str, str] | None = None,
    decorator_aliases: dict[str, ast.Call | None] | None = None,
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
    aliases = decorator_aliases or {"tool": None}
    if decorator_name not in aliases:
        return None
    stored_factory = aliases[decorator_name]
    if call is None and stored_factory is not None:
        call = stored_factory
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
    normalized = name.casefold()
    components = [
        component
        for component in re.split(r"[^a-z0-9]+", normalized)
        if component
    ]
    # Permission collections are commonly named with regular English plurals
    # (``policies``, ``authorities``, ``roles``).  Match the semantic stem
    # rather than growing a second vocabulary of singular/plural spellings.
    singularized = {
        component[:-3] + "y"
        if component.endswith("ies") and len(component) > 3
        else component[:-1]
        if component.endswith("s") and len(component) > 3
        else component
        for component in components
    }
    candidates = {normalized, *components, *singularized}
    return any(
        term in candidate
        for term in PERMISSION_NAME_TERMS
        for candidate in candidates
    )


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


def _imperatively_decorated_tool_names(
    tree: ast.Module,
    decorator_aliases: dict[str, ast.Call | None],
    source_path: Path | None = None,
) -> set[str]:
    """Return SDK tools published by assigning a decorated callable."""

    names: set[str] = set()
    for node in ast.walk(tree):
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
        if not isinstance(value, ast.Call):
            continue

        decorator = value.func
        decorator_name = (
            decorator.id
            if isinstance(decorator, ast.Name)
            else decorator.attr
            if isinstance(decorator, ast.Attribute)
            else ""
        )
        applies_stored_factory = (
            decorator_name in decorator_aliases
            and decorator_aliases[decorator_name] is not None
        )
        if not isinstance(decorator, ast.Call) and not applies_stored_factory:
            continue
        target_names = {
            name
            for target in targets
            for name in (
                {target.id}
                if isinstance(target, ast.Name)
                else {target.attr}
                if isinstance(target, ast.Attribute)
                else set()
            )
        }
        if not target_names:
            continue
        constants = _module_strings_at_definition(tree, node, source_path)
        for target_name in target_names:
            public_name = _public_tool_name(
                decorator,
                target_name,
                constants,
                decorator_aliases,
            )
            if public_name is not None:
                if len(value.args) != 1 or value.keywords:
                    raise AssertionError(
                        "Unresolved imperative @tool application: "
                        f"{ast.unparse(value)}"
                    )
                names.add(public_name)
    return names


def _tool_surfaces_from_module(
    tree: ast.Module,
    relative_path: str,
    source_path: Path | None = None,
) -> set[str]:
    """Inventory declarative and imperative SDK tool registrations."""

    surfaces: set[str] = set()
    decorator_aliases = _tool_decorator_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        string_constants = _module_strings_at_definition(
            tree, node, source_path
        )
        for decorator in node.decorator_list:
            public_name = _public_tool_name(
                decorator,
                node.name,
                string_constants,
                decorator_aliases,
            )
            if public_name is not None:
                surfaces.add(f"{relative_path}::{public_name}")
    imperative_surfaces = {
        f"{relative_path}::{public_name}"
        for public_name in _imperatively_decorated_tool_names(
            tree,
            decorator_aliases,
            source_path,
        )
    }
    surfaces.update(imperative_surfaces)
    return surfaces


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
        relative = path.relative_to(REPO_ROOT).as_posix()
        surfaces.update(_tool_surfaces_from_module(tree, relative, path))
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
    cron_binding = ("CRON_TASKS", "mutable-collection")
    alias_flow = _StaticBindingFlow(
        lambda _value: None,
        lambda bindings: bindings[0],
    )

    def cron_method_binding(value: ast.AST) -> _StaticBinding | None:
        member = _static_member_reference(value)
        if member is None or alias_flow.resolve(member[0]) != cron_binding:
            return None
        return cron_binding[0], f"method:{member[1]}"

    for index, node in enumerate(tree.body):
        prefix = ast.Module(body=tree.body[:index], type_ignores=[])
        constants = _module_string_constants(prefix, source_path)

        mutation_targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            mutation_targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            mutation_targets = [node.target]
        elif isinstance(node, ast.Delete):
            mutation_targets = list(node.targets)
        if any(
            isinstance(target, ast.Subscript)
            and alias_flow.resolve(target.value) == cron_binding
            for target in mutation_targets
        ):
            raise AssertionError("Unsupported subscript CRON_TASKS mutation")

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
            alias_flow.bindings["CRON_TASKS"] = cron_binding
            continue

        if (
            isinstance(node, ast.AugAssign)
            and alias_flow.resolve(node.target) == cron_binding
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
            and (member := _static_member_reference(direct_call.func)) is not None
            and alias_flow.resolve(member[0]) == cron_binding
        ):
            if task_names is None or direct_call.keywords:
                raise AssertionError("Unsupported CRON_TASKS method mutation")
            if member[1] == "append" and len(direct_call.args) == 1:
                task_names.add(
                    _cron_task_entry_name(direct_call.args[0], constants)
                )
                continue
            if member[1] == "extend" and len(direct_call.args) == 1:
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

        if isinstance(direct_call, ast.Call):
            bound_method = alias_flow.resolve(direct_call.func)
            if (
                bound_method is not None
                and bound_method[0] == cron_binding[0]
                and bound_method[1].startswith("method:")
            ):
                raise AssertionError(
                    "Unsupported bound CRON_TASKS method mutation: "
                    f"{ast.unparse(direct_call)}"
                )

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # Replay a compound block before inspecting its descendants.  The
        # shared may-analysis then exposes aliases introduced on any branch,
        # so a later mutation in that same branch cannot disappear merely
        # because its receiver was not bound before the compound statement.
        if isinstance(node, _MODULE_COMPOUND_STATEMENT_TYPES):
            alias_flow.replay([node], cron_method_binding)
        if any(
            isinstance(child, ast.Name)
            and child.id == "CRON_TASKS"
            and isinstance(child.ctx, (ast.Store, ast.Del))
            for child in ast.walk(node)
        ) or any(
            isinstance(child, ast.Call)
            and (child_member := _static_member_reference(child.func)) is not None
            and alias_flow.resolve(child_member[0]) == cron_binding
            for child in ast.walk(node)
        ):
            raise AssertionError(
                "Conditional or indirect CRON_TASKS mutation is not statically "
                f"resolvable: {ast.unparse(node)}"
            )

        if isinstance(node, ast.Assign):
            alias_flow.assign(
                list(node.targets),
                node.value,
                cron_method_binding,
            )
        elif isinstance(node, ast.AnnAssign):
            alias_flow.assign(
                [node.target],
                node.value,
                cron_method_binding,
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
    edges = _static_alias_edges(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for imported in node.names:
                if "SourceRegistration" in imported.name:
                    aliases.add(imported.asname or imported.name)
        elif isinstance(node, ast.ClassDef):
            edges.append(
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
    return _static_alias_closure(aliases, edges)


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
    helpers = _RegistryHelperIndex(tree)

    class RuntimeSignalPublisherVisitor(ast.NodeVisitor):
        PUBLISH_METHODS = {"register", "register_batch", "register_with_policy"}
        REGISTRY_BINDING = ("signal_registry", "mutable-registry")
        PUBLISH_BINDINGS = {
            method: ("signal_registry", f"publish:{method}")
            for method in PUBLISH_METHODS
        }

        def __init__(self) -> None:
            self.scope: list[str] = []
            self.flows: list[_StaticBindingFlow] = []
            self.functions: list[
                ast.FunctionDef | ast.AsyncFunctionDef
            ] = []

        def _ambiguous(
            self, bindings: list[_StaticBinding]
        ) -> _StaticBinding:
            for binding in bindings:
                if binding in self.PUBLISH_BINDINGS.values():
                    return binding
            if self.REGISTRY_BINDING in bindings:
                return self.REGISTRY_BINDING
            return bindings[0]

        @staticmethod
        def _direct_binding(node: ast.AST) -> _StaticBinding | None:
            if isinstance(node, ast.Name) and node.id in {
                "signal_registry",
                "source_registry",
            }:
                return RuntimeSignalPublisherVisitor.REGISTRY_BINDING
            member = _static_member_reference(node)
            if member is not None and member[1] in {
                "signal_registry",
                "source_registry",
            }:
                return RuntimeSignalPublisherVisitor.REGISTRY_BINDING
            return None

        def _new_flow(
            self, bindings: dict[str, _StaticBinding] | None = None
        ) -> _StaticBindingFlow:
            return _StaticBindingFlow(
                self._direct_binding,
                self._ambiguous,
                bindings,
            )

        def _is_registry(self, node: ast.AST) -> bool:
            return bool(self.flows) and (
                self.flows[-1].resolve(node) == self.REGISTRY_BINDING
            )

        def _bound_publish_method(
            self, node: ast.AST
        ) -> _StaticBinding | None:
            member = _static_member_reference(node)
            if (
                member is not None
                and member[1] in self.PUBLISH_METHODS
                and self._is_registry(member[0])
            ):
                return self.PUBLISH_BINDINGS[member[1]]
            return None

        def _assign(
            self,
            targets: list[ast.AST],
            value: ast.AST | None,
        ) -> None:
            self.flows[-1].assign(
                targets,
                value,
                self._bound_publish_method,
            )

        def _record(self) -> None:
            if self.scope:
                surfaces.add(f"{relative}::{'.'.join(self.scope)}")

        def _visit_block(self, statements: list[ast.stmt]) -> None:
            for statement in statements:
                if isinstance(statement, _MODULE_COMPOUND_STATEMENT_TYPES):
                    expressions, blocks = _compound_flow_parts(statement)
                    for expression in expressions:
                        self.visit(expression)
                    inherited = self.flows[-1]
                    for expression in expressions:
                        inherited.replay_expression_bindings(
                            expression,
                            self._bound_publish_method,
                        )
                    branches: list[_StaticBindingFlow] = []
                    for index, block in enumerate(blocks):
                        branch = inherited.fork()
                        self.flows[-1] = branch
                        if index == 0:
                            _bind_static_loop_target(
                                branch,
                                statement,
                                self._bound_publish_method,
                            )
                        self._visit_block(block)
                        branches.append(branch)
                    branches.append(inherited.disturbed_fork(statement))
                    self.flows[-1] = inherited
                    inherited.join(branches)
                    continue
                self.visit(statement)
                if isinstance(statement, ast.Assign):
                    self._assign(list(statement.targets), statement.value)
                elif isinstance(statement, ast.AnnAssign):
                    self._assign([statement.target], statement.value)
                else:
                    self.flows[-1].replay_expression_bindings(
                        statement,
                        self._bound_publish_method,
                    )

        def _visit_definition(
            self,
            node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> None:
            inherited_bindings = (
                dict(self.flows[-1].bindings)
                if self.flows
                and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                else {}
            )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                for parameter in parameters:
                    inherited_bindings.pop(parameter.arg, None)
                if node.args.vararg is not None:
                    inherited_bindings.pop(node.args.vararg.arg, None)
                if node.args.kwarg is not None:
                    inherited_bindings.pop(node.args.kwarg.arg, None)
            self.scope.append(node.name)
            self.flows.append(self._new_flow(inherited_bindings))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.append(node)
            self._visit_block(node.body)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name == "_register_signal_sources"
                and {"register_batch", "register_with_policy"}.issubset(
                    {
                        child.value
                        for child in _walk_lexical_scope(node)
                        if isinstance(child, ast.Constant)
                        and isinstance(child.value, str)
                    }
                )
            ):
                self._record()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.pop()
            self.flows.pop()
            self.scope.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            self._visit_definition(node)

        def _visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            self._visit_definition(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_function(node)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            self._visit_function(node)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            direct_member = _static_member_reference(node.func)
            bound_method = self.flows[-1].resolve(node.func)
            publishes_directly = bool(
                direct_member is not None
                and direct_member[1] in self.PUBLISH_METHODS
                and self._is_registry(direct_member[0])
            ) or bound_method in self.PUBLISH_BINDINGS.values()
            helper = _registry_helper_call_bindings(
                node,
                helpers,
                self.flows[-1],
                self.REGISTRY_BINDING,
                self.functions[-1] if self.functions else None,
            )
            publishes_via_helper = bool(
                helper is not None
                and _function_publishes_registry(
                    helper[0],
                    helper[1],
                    self.REGISTRY_BINDING,
                    self.PUBLISH_METHODS,
                    self._direct_binding,
                    self._ambiguous,
                    helpers,
                )
            )
            if publishes_directly or publishes_via_helper:
                self._record()
            self.generic_visit(node)

    visitor = RuntimeSignalPublisherVisitor()
    visitor.flows.append(visitor._new_flow())
    visitor._visit_block(tree.body)
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
    helpers = _RegistryHelperIndex(tree)
    string_constants = _module_string_constants(tree)

    class DirectToolWriterVisitor(ast.NodeVisitor):
        PUBLISH_METHODS = {"__ior__", "__setitem__", "setdefault", "update"}
        REGISTRY_BINDING = ("_direct_tools", "mutable-registry")
        PUBLISH_BINDINGS = {
            method: ("_direct_tools", f"publish:{method}")
            for method in PUBLISH_METHODS
        }

        def __init__(self) -> None:
            self.scope: list[str] = []
            self.flows: list[_StaticBindingFlow] = []
            self.functions: list[
                ast.FunctionDef | ast.AsyncFunctionDef
            ] = []

        def _ambiguous(
            self, bindings: list[_StaticBinding]
        ) -> _StaticBinding:
            for binding in bindings:
                if binding in self.PUBLISH_BINDINGS.values():
                    return binding
            if self.REGISTRY_BINDING in bindings:
                return self.REGISTRY_BINDING
            return bindings[0]

        def _direct_binding(self, node: ast.AST) -> _StaticBinding | None:
            member_reference = _static_member_reference(node)
            if (
                member_reference is not None
                and member_reference[1] == "_direct_tools"
            ):
                return self.REGISTRY_BINDING
            return None

        def _new_flow(
            self, bindings: dict[str, _StaticBinding] | None = None
        ) -> _StaticBindingFlow:
            return _StaticBindingFlow(
                self._direct_binding,
                self._ambiguous,
                bindings,
            )

        def _visit_definition(
            self,
            node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> None:
            inherited_bindings = (
                dict(self.flows[-1].bindings)
                if self.flows
                and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                else {}
            )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                for parameter in parameters:
                    inherited_bindings.pop(parameter.arg, None)
                if node.args.vararg is not None:
                    inherited_bindings.pop(node.args.vararg.arg, None)
                if node.args.kwarg is not None:
                    inherited_bindings.pop(node.args.kwarg.arg, None)
            self.scope.append(node.name)
            self.flows.append(self._new_flow(inherited_bindings))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.append(node)
            self._visit_block(node.body)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.pop()
            self.flows.pop()
            self.scope.pop()

        def _is_registry(self, node: ast.AST) -> bool:
            return bool(self.flows) and (
                self.flows[-1].resolve(node) == self.REGISTRY_BINDING
            )

        def _bound_publish_method(
            self, node: ast.AST
        ) -> _StaticBinding | None:
            member = _static_member_reference(node)
            if (
                member is not None
                and member[1] in self.PUBLISH_METHODS
                and self._is_registry(member[0])
            ):
                return self.PUBLISH_BINDINGS[member[1]]
            return None

        def _assign(
            self,
            targets: list[ast.AST],
            value: ast.AST | None,
        ) -> None:
            self.flows[-1].assign(
                targets,
                value,
                self._bound_publish_method,
            )

        def _visit_block(self, statements: list[ast.stmt]) -> None:
            for statement in statements:
                if isinstance(statement, _MODULE_COMPOUND_STATEMENT_TYPES):
                    expressions, blocks = _compound_flow_parts(statement)
                    for expression in expressions:
                        self.visit(expression)
                    inherited = self.flows[-1]
                    for expression in expressions:
                        inherited.replay_expression_bindings(
                            expression,
                            self._bound_publish_method,
                        )
                    branches: list[_StaticBindingFlow] = []
                    for index, block in enumerate(blocks):
                        branch = inherited.fork()
                        self.flows[-1] = branch
                        if index == 0:
                            _bind_static_loop_target(
                                branch,
                                statement,
                                self._bound_publish_method,
                            )
                        self._visit_block(block)
                        branches.append(branch)
                    branches.append(inherited.disturbed_fork(statement))
                    self.flows[-1] = inherited
                    inherited.join(branches)
                    continue
                self.visit(statement)
                if isinstance(statement, ast.Assign):
                    self._assign(list(statement.targets), statement.value)
                elif isinstance(statement, ast.AnnAssign):
                    self._assign([statement.target], statement.value)
                else:
                    self.flows[-1].replay_expression_bindings(
                        statement,
                        self._bound_publish_method,
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
            if any(
                isinstance(target, ast.Subscript)
                and self._is_registry(target.value)
                for target in node.targets
            ):
                self._record()
            if any(
                (member := _static_member_reference(target)) is not None
                and member[1] == "_direct_tools"
                for target in node.targets
            ) and not (
                isinstance(node.value, ast.Dict)
                and not node.value.keys
            ):
                self._record()
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
            if (
                isinstance(node.target, ast.Subscript)
                and self._is_registry(node.target.value)
            ):
                self._record()
            if (
                node.value is not None
                and (member := _static_member_reference(node.target)) is not None
                and member[1] == "_direct_tools"
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
            # The callable expression is evaluated before the arguments and
            # may itself bind the registry or one of its publish methods.
            self.flows[-1].replay_expression_bindings(
                node.func,
                self._bound_publish_method,
            )
            setattr_value: ast.AST | None = None
            if (
                _call_name(node) == "setattr"
                and len(node.args) >= 3
                and _resolved_string(node.args[1], string_constants)
                == "_direct_tools"
            ):
                setattr_value = node.args[2]
            replaces_registry = setattr_value is not None and not (
                isinstance(setattr_value, ast.Dict)
                and not setattr_value.keys
            )
            direct_member = _static_member_reference(node.func)
            bound_method = self.flows[-1].resolve(node.func)
            functional_publish = bool(
                node.args
                and _call_name(node).casefold()
                in {
                    "__ior__",
                    "__setitem__",
                    "ior",
                    "setdefault",
                    "setitem",
                    "update",
                }
                and self._is_registry(node.args[0])
            )
            publishes_directly = bool(
                direct_member is not None
                and direct_member[1] in self.PUBLISH_METHODS
                and self._is_registry(direct_member[0])
            ) or (
                bound_method in self.PUBLISH_BINDINGS.values()
                or functional_publish
            )
            helper = _registry_helper_call_bindings(
                node,
                helpers,
                self.flows[-1],
                self.REGISTRY_BINDING,
                self.functions[-1] if self.functions else None,
            )
            publishes_via_helper = bool(
                helper is not None
                and _function_publishes_registry(
                    helper[0],
                    helper[1],
                    self.REGISTRY_BINDING,
                    self.PUBLISH_METHODS,
                    self._direct_binding,
                    self._ambiguous,
                    helpers,
                    direct_assignment_member="_direct_tools",
                )
            )
            if replaces_registry or publishes_directly or publishes_via_helper:
                self._record()
            self.generic_visit(node)

    visitor = DirectToolWriterVisitor()
    visitor.flows.append(visitor._new_flow())
    visitor._visit_block(tree.body)
    return writers


def _runtime_generated_tool_class_surfaces(
    package_root: Path,
    repository_root: Path,
) -> set[str]:
    """Find concrete runtime-tool boundaries in a complete shipped package."""

    surfaces: set[str] = set()

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
                            isinstance(
                                member,
                                (ast.FunctionDef, ast.AsyncFunctionDef),
                            )
                            and member.name == "execute"
                        ):
                            surfaces.add(
                                f"{relative}::"
                                f"{'.'.join((*qualified, member.name))}"
                            )
                walk_statements(node.body, relative, qualified)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = (*parents, node.name)
                if node.name in {"to_orchestrator_tool", "execute_as_subagent"}:
                    surfaces.add(f"{relative}::{'.'.join(qualified)}")
                walk_statements(node.body, relative, qualified)
            else:
                nested_statements = [
                    child
                    for child in ast.iter_child_nodes(node)
                    if isinstance(child, ast.stmt)
                ]
                walk_statements(nested_statements, relative, parents)

    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        walk_statements(
            tree.body,
            path.relative_to(repository_root).as_posix(),
        )
    return surfaces


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

    package_root = REPO_ROOT / "kestrel_sovereign"
    surfaces = _runtime_generated_tool_class_surfaces(
        package_root,
        REPO_ROOT,
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
            "_dispatch_feature_tool",
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


def _core_cli_parser_command_names(
    tree: ast.Module,
    string_constants: dict[str, str],
) -> set[str]:
    """Return literal commands registered on the canonical top-level parser."""

    builders = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_parser"
    ]
    if not builders:
        return set()
    if len(builders) != 1:
        raise AssertionError("Could not uniquely identify the core CLI parser builder")
    builder = builders[0]

    root_names: set[str] = set()
    assignments: list[tuple[set[str], ast.AST | None]] = []
    for node in ast.walk(builder):
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value: ast.AST | None = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        target_names = {
            name
            for target in targets
            for name in _assignment_target_names(target)
        }
        assignments.append((target_names, value))
        if not (
            isinstance(value, ast.Call)
            and (member := _static_member_reference(value.func)) is not None
            and member[1] == "add_subparsers"
        ):
            continue
        destination = next(
            (keyword.value for keyword in value.keywords if keyword.arg == "dest"),
            None,
        )
        if destination is not None and (
            _resolved_string(destination, string_constants) == "command"
        ):
            root_names.update(target_names)

    if not root_names:
        raise AssertionError("Could not find the core CLI top-level subparser")

    # Follow simple aliases so renaming ``subparsers`` or publishing through a
    # second local name cannot make a parser registration disappear.
    changed = True
    while changed:
        changed = False
        for targets, value in assignments:
            if isinstance(value, ast.Name) and value.id in root_names:
                new_names = targets - root_names
                if new_names:
                    root_names.update(new_names)
                    changed = True

    commands: set[str] = set()

    class TopLevelParserVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            member = _static_member_reference(node.func)
            if not (
                member is not None
                and member[1] == "add_parser"
                and isinstance(member[0], ast.Name)
                and member[0].id in root_names
            ):
                self.generic_visit(node)
                return
            if not node.args:
                raise AssertionError("Core CLI add_parser call has no command name")
            command = _resolved_string(node.args[0], string_constants)
            if command is None:
                raise AssertionError(
                    "Unresolved core CLI parser command: "
                    f"{ast.unparse(node.args[0])}"
                )
            commands.add(command)
            if any(keyword.arg is None for keyword in node.keywords):
                raise AssertionError("Core CLI parser registration uses keyword unpacking")
            aliases = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "aliases"),
                None,
            )
            if aliases is not None:
                if not isinstance(aliases, (ast.List, ast.Tuple, ast.Set)):
                    raise AssertionError("Unresolved core CLI parser aliases")
                for alias_node in aliases.elts:
                    alias = _resolved_string(alias_node, string_constants)
                    if alias is None:
                        raise AssertionError("Unresolved core CLI parser alias")
                    commands.add(alias)
            self.generic_visit(node)

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

    parser_visitor = TopLevelParserVisitor()
    for statement in builder.body:
        parser_visitor.visit(statement)
    return commands


def _core_cli_predispatch_command_names(
    statements: list[ast.stmt],
    string_constants: dict[str, str],
) -> set[str]:
    """Return concrete ``args.command`` branches before map dispatch."""

    commands: set[str] = set()

    def is_selector(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "command"
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        )

    def contains_selector(node: ast.AST) -> bool:
        return any(is_selector(child) for child in ast.walk(node))

    def resolved_values(node: ast.AST) -> set[str]:
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = {_resolved_string(element, string_constants) for element in node.elts}
            if None in values:
                raise AssertionError("Unresolved core CLI pre-dispatch command set")
            return {value for value in values if value is not None}
        value = _resolved_string(node, string_constants)
        if value is None:
            raise AssertionError(
                "Unresolved core CLI pre-dispatch command expression: "
                f"{ast.unparse(node)}"
            )
        return {value}

    def condition_values(node: ast.AST) -> set[str]:
        if not contains_selector(node):
            return set()
        if is_selector(node):
            return set()
        if isinstance(node, ast.UnaryOp) and is_selector(node.operand):
            return set()
        if isinstance(node, ast.BoolOp):
            return {
                value
                for expression in node.values
                for value in condition_values(expression)
            }
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            left, right = node.left, node.comparators[0]
            if is_selector(left):
                return resolved_values(right)
            if is_selector(right):
                return resolved_values(left)
        raise AssertionError(
            "Unresolved core CLI pre-dispatch command condition: "
            f"{ast.unparse(node)}"
        )

    def pattern_values(pattern: ast.pattern) -> set[str]:
        if isinstance(pattern, ast.MatchValue):
            return resolved_values(pattern.value)
        if isinstance(pattern, ast.MatchOr):
            return {
                value
                for alternative in pattern.patterns
                for value in pattern_values(alternative)
            }
        if isinstance(pattern, ast.MatchAs):
            return (
                pattern_values(pattern.pattern)
                if pattern.pattern is not None
                else set()
            )
        if isinstance(pattern, ast.MatchSingleton):
            return set()
        raise AssertionError("Unresolved core CLI pre-dispatch match pattern")

    class PredispatchVisitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:  # noqa: N802
            commands.update(condition_values(node.test))
            self.generic_visit(node)

        def visit_Match(self, node: ast.Match) -> None:  # noqa: N802
            if not is_selector(node.subject):
                self.generic_visit(node)
                return
            for case in node.cases:
                commands.update(pattern_values(case.pattern))
                if case.guard is not None:
                    commands.update(condition_values(case.guard))
            self.generic_visit(node)

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

    visitor = PredispatchVisitor()
    for statement in statements:
        visitor.visit(statement)
    return commands


def _core_cli_command_names(
    tree: ast.Module,
    string_constants: dict[str, str],
) -> set[str]:
    """Reconcile parser, pre-dispatch, and command-map entry doors."""

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
    command_binding = ("commands", "mutable-map")
    alias_flow = _StaticBindingFlow(
        lambda _value: None,
        lambda bindings: bindings[0],
    )
    parser_commands = _core_cli_parser_command_names(tree, string_constants)
    for statement_index, statement in enumerate(scope.body):
        if any(is_dispatch_lookup(node) for node in ast.walk(statement)):
            if command_names is None:
                raise AssertionError(
                    "Core CLI command dispatch is read before initialization"
                )
            predispatch_commands = _core_cli_predispatch_command_names(
                scope.body[:statement_index],
                string_constants,
            )
            unknown_predispatch = predispatch_commands - (
                parser_commands | command_names
            )
            if unknown_predispatch:
                raise AssertionError(
                    "Pre-dispatch core CLI commands lack parser or map entries: "
                    + ", ".join(sorted(unknown_predispatch))
                )
            undispatched_parser_commands = parser_commands - (
                predispatch_commands | command_names
            )
            if undispatched_parser_commands:
                raise AssertionError(
                    "Core CLI parser commands lack a dispatch path: "
                    + ", ".join(sorted(undispatched_parser_commands))
                )
            return command_names | predispatch_commands
        if isinstance(statement, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "commands"
                for target in statement.targets
            ):
                command_names = dictionary_keys(statement.value)
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        alias_flow.bindings[target.id] = command_binding
                continue
            command_keys = [
                target.slice
                for target in statement.targets
                if isinstance(target, ast.Subscript)
                and alias_flow.resolve(target.value) == command_binding
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
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "commands"
        ):
            if statement.value is None:
                raise AssertionError(
                    "Core CLI command dispatch has no initialized value"
                )
            command_names = dictionary_keys(statement.value)
            alias_flow.bindings["commands"] = command_binding
            continue
        if (
            isinstance(statement, ast.AugAssign)
            and alias_flow.resolve(statement.target) == command_binding
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
            and (
                member := _static_member_reference(statement.value.func)
            ) is not None
            and alias_flow.resolve(member[0]) == command_binding
        ):
            call = statement.value
            if command_names is None:
                raise AssertionError(
                    "Core CLI command dispatch is mutated before initialization"
                )
            if member[1] != "update" or len(call.args) > 1:
                raise AssertionError("Unresolved core CLI command-map mutation")
            if call.args:
                command_names.update(dictionary_keys(call.args[0]))
            if any(keyword.arg is None for keyword in call.keywords):
                raise AssertionError("Unresolved core CLI command-map mutation")
            command_names.update(
                keyword.arg for keyword in call.keywords if keyword.arg is not None
            )
            continue
        assignment_targets: list[ast.AST] = []
        assignment_value: ast.AST | None = None
        if isinstance(statement, ast.Assign):
            assignment_targets = list(statement.targets)
            assignment_value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            assignment_targets = [statement.target]
            assignment_value = statement.value
        if assignment_targets:
            if assignment_value is not None and any(
                isinstance(node, ast.Call)
                and any(
                    isinstance(argument, ast.Name)
                    and alias_flow.resolve(argument) == command_binding
                    for argument in ast.walk(node)
                )
                for node in ast.walk(assignment_value)
            ):
                raise AssertionError("Unresolved core CLI command-map mutation")
            alias_flow.assign(assignment_targets, assignment_value)
            continue
        if command_names is not None and any(
            isinstance(node, ast.Name)
            and alias_flow.resolve(node) == command_binding
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
            self.flows: list[_StaticBindingFlow] = [self._new_flow()]

        @staticmethod
        def _direct_binding(value: ast.AST) -> _StaticBinding | None:
            member_reference = _static_member_reference(value)
            if (
                member_reference is not None
                and member_reference[1].casefold() == "include_router"
            ):
                return ast.unparse(member_reference[0]), "include_router"
            return None

        @staticmethod
        def _ambiguous(bindings: list[_StaticBinding]) -> _StaticBinding:
            # This inventory asks whether a binding *may* publish a router.
            # Preserve include_router when any reachable branch retains it.
            return bindings[0]

        @classmethod
        def _new_flow(
            cls, bindings: dict[str, _StaticBinding] | None = None
        ) -> _StaticBindingFlow:
            return _StaticBindingFlow(
                cls._direct_binding,
                cls._ambiguous,
                bindings,
            )

        def _visit_compound(self, statement: ast.stmt) -> None:
            inherited = self.flows[-1]
            expressions, blocks = _compound_flow_parts(statement)

            for expression in expressions:
                self.visit(expression)
                inherited.replay_expression_bindings(expression)
            branch_flows: list[_StaticBindingFlow] = []
            for block in blocks:
                branch = inherited.fork()
                self.flows[-1] = branch
                self._visit_block(block)
                branch_flows.append(branch)
            branch_flows.append(inherited.disturbed_fork(statement))
            self.flows[-1] = inherited
            inherited.join(branch_flows)

        def _visit_block(self, statements: list[ast.stmt]) -> None:
            for statement in statements:
                if isinstance(statement, _MODULE_COMPOUND_STATEMENT_TYPES):
                    self._visit_compound(statement)
                    continue
                self.visit(statement)
                if isinstance(statement, ast.Assign):
                    self.flows[-1].assign(
                        list(statement.targets), statement.value
                    )
                elif isinstance(statement, ast.AnnAssign):
                    self.flows[-1].assign([statement.target], statement.value)
                else:
                    self.flows[-1].replay_expression_bindings(statement)

        def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
            self._visit_block(node.body)

        def _visit_scope(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            self.scope.append(node.name)
            self.scope_counts.append(0)
            self.flows.append(self.flows[-1].fork())
            self._visit_block(node.body)
            self.flows.pop()
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
            self.flows.append(self.flows[-1].fork())
            self._visit_block(node.body)
            self.flows.pop()
            self.scope.pop()

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            registration = _route_registration_name(
                node, self.flows[-1].bindings
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

    def direct_binding(value: ast.AST) -> _StaticBinding | None:
        member_reference = _static_member_reference(value)
        if member_reference is not None:
            receiver_expression, member = member_reference
            registration = member.casefold()
            receiver = (
                receiver_expression.id
                if isinstance(receiver_expression, ast.Name)
                else None
            )
            if registration in _ROUTE_REGISTRATION_NAMES and (
                receiver == "app" or receiver in prefixes
            ):
                return receiver, registration
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "getattr"
            and value.args
            and isinstance(value.args[0], ast.Name)
            and (
                value.args[0].id == "app"
                or value.args[0].id in prefixes
            )
        ):
            return value.args[0].id, _UNRESOLVED_ROUTE_REGISTRATION
        return None

    def ambiguous(bindings: list[_StaticBinding]) -> _StaticBinding:
        return bindings[0][0], _UNRESOLVED_ROUTE_REGISTRATION

    flow = _StaticBindingFlow(
        direct_binding,
        ambiguous,
        inherited_aliases,
    )
    flow.replay(statements)
    return flow.bindings


def _scope_route_decorator_factories(
    statements: list[ast.stmt],
    prefixes: dict[str, str],
    route_aliases: dict[str, tuple[str, str]],
    inherited: dict[str, ast.Call] | None = None,
) -> dict[str, ast.Call]:
    """Resolve stored decorators returned by bound FastAPI registrations."""

    return _call_produced_decorator_bindings(
        statements,
        lambda call: (
            _route_registration_name(call, route_aliases)
            in {
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
            }
            and (
                (receiver := _route_receiver_name(call, route_aliases)) == "app"
                or receiver in prefixes
            )
        ),
        inherited,
    )


def _route_registration_name(
    call: ast.Call,
    aliases: dict[str, tuple[str, str]] | None = None,
    string_constants: dict[str, str] | None = None,
    route_receivers: set[str] | None = None,
) -> str | None:
    member_reference = _static_member_reference(call.func, string_constants)
    if member_reference is not None:
        return member_reference[1].casefold()
    if (
        isinstance(call.func, ast.Call)
        and isinstance(call.func.func, ast.Name)
        and call.func.func.id == "getattr"
        and call.func.args
        and isinstance(call.func.args[0], ast.Name)
        and call.func.args[0].id in (route_receivers or {"app"})
    ):
        raise AssertionError(
            "Unresolved immediate route registration: "
            f"{ast.unparse(call.func)}"
        )
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
    string_constants: dict[str, str] | None = None,
) -> str | None:
    member_reference = _static_member_reference(
        decorator.func, string_constants
    )
    if member_reference is not None:
        receiver = member_reference[0]
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


def _route_object_methods_and_path(
    route_object: ast.AST,
    active_strings: dict[str, str],
    active_methods: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, ...], str]:
    """Decode one statically constructed Starlette/FastAPI route object."""

    if not isinstance(route_object, ast.Call):
        raise AssertionError(
            "Unresolved route object publication: "
            f"{ast.unparse(route_object)}"
        )
    constructor = _call_name(route_object)
    if constructor in {
        "APIWebSocketRoute",
        "StarletteWebSocketRoute",
        "WebSocketRoute",
    }:
        methods = ("WEBSOCKET",)
    elif constructor == "Mount":
        methods = ("MOUNT",)
    elif constructor in {"APIRoute", "Route", "StarletteRoute"}:
        method_expression = next(
            (
                keyword.value
                for keyword in route_object.keywords
                if keyword.arg == "methods"
            ),
            None,
        )
        if method_expression is None and len(route_object.args) >= 3:
            method_expression = route_object.args[2]
        registration = "api_route" if constructor == "APIRoute" else "route"
        synthetic = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="router", ctx=ast.Load()),
                attr=registration,
                ctx=ast.Load(),
            ),
            args=[route_object.args[0]] if route_object.args else [],
            keywords=(
                [ast.keyword(arg="methods", value=method_expression)]
                if method_expression is not None
                else []
            ),
        )
        methods = _route_methods(synthetic, active_methods)
    else:
        raise AssertionError(
            "Unsupported route object publication: "
            f"{ast.unparse(route_object)}"
        )
    return methods, _programmatic_route_path(route_object, active_strings)


def _fastapi_generated_route_declarations(
    statements: list[ast.stmt],
    constants: dict[str, str],
    method_constants: dict[str, tuple[str, ...]],
) -> list[tuple[tuple[str, ...], str]]:
    """Return FastAPI constructor-supplied and generated route doors."""

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
        value = (
            statement.value
            if isinstance(
                statement,
                (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return),
            )
            else None
        )
        if not isinstance(value, ast.Call):
            continue
        constructor = _call_name(value)
        if constructor not in {"APIRouter", "FastAPI"}:
            continue
        prefix = (
            optional_path(value, "prefix", "")
            if constructor == "APIRouter"
            else ""
        )
        assert prefix is not None
        routes_keyword = next(
            (item for item in value.keywords if item.arg == "routes"),
            None,
        )
        if routes_keyword is not None:
            route_values = routes_keyword.value
            if isinstance(route_values, ast.Constant) and route_values.value is None:
                route_objects: list[ast.AST] = []
            elif isinstance(route_values, (ast.List, ast.Tuple, ast.Set)):
                route_objects = list(route_values.elts)
            else:
                raise AssertionError(
                    "Unresolved FastAPI constructor routes: "
                    f"{ast.unparse(route_values)}"
                )
            for route_object in route_objects:
                methods, path = _route_object_methods_and_path(
                    route_object,
                    constants,
                    method_constants,
                )
                declarations.append((methods, prefix + path))
        if constructor != "FastAPI":
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
    module_runtime_route_factories = _scope_route_decorator_factories(
        tree.body,
        module_runtime_prefixes,
        module_runtime_route_aliases,
    )

    route_collection_binding = "route_collection"
    unresolved_route_collection = "<unresolved>"

    def route_collection_receiver(
        expression: ast.AST,
        aliases: dict[str, _StaticBinding] | None = None,
    ) -> str | None:
        if isinstance(expression, ast.Name) and aliases is not None:
            binding = aliases.get(expression.id)
            if binding is None or binding[1] != route_collection_binding:
                return None
            if binding[0] == unresolved_route_collection:
                raise AssertionError(
                    "Unresolved route collection alias: "
                    f"{ast.unparse(expression)}"
                )
            return binding[0]
        if not isinstance(expression, ast.Attribute) or expression.attr != "routes":
            return None
        owner = expression.value
        if isinstance(owner, ast.Name):
            return owner.id
        if (
            isinstance(owner, ast.Attribute)
            and owner.attr == "router"
            and isinstance(owner.value, ast.Name)
        ):
            return owner.value.id
        return None

    def scope_route_collection_aliases(
        statements: list[ast.stmt],
        prefixes: dict[str, str],
        inherited: dict[str, _StaticBinding] | None = None,
    ) -> dict[str, _StaticBinding]:
        def direct_binding(value: ast.AST) -> _StaticBinding | None:
            receiver = route_collection_receiver(value)
            if receiver == "app" or receiver in prefixes:
                return receiver, route_collection_binding
            return None

        flow = _StaticBindingFlow(
            direct_binding,
            lambda _bindings: (
                unresolved_route_collection,
                route_collection_binding,
            ),
            inherited,
        )
        flow.replay(statements)
        return flow.bindings

    module_runtime_route_collections = scope_route_collection_aliases(
        tree.body,
        module_runtime_prefixes,
    )

    def route_object_declaration(
        route_object: ast.AST,
        collection: ast.AST,
        active_strings: dict[str, str],
        active_methods: dict[str, tuple[str, ...]],
        prefixes: dict[str, str],
        route_collections: dict[str, _StaticBinding],
    ) -> tuple[tuple[str, ...], str]:
        methods, path = _route_object_methods_and_path(
            route_object,
            active_strings,
            active_methods,
        )

        receiver = route_collection_receiver(collection, route_collections)
        if receiver == "app":
            prefix = ""
        elif receiver is not None and receiver in prefixes:
            prefix = prefixes[receiver]
        else:
            raise AssertionError(
                "Unresolved route object collection: "
                f"{ast.unparse(collection)}"
            )
        return (
            methods,
            prefix + path,
        )

    def appended_route_objects(
        call: ast.Call,
        route_collections: dict[str, _StaticBinding],
    ) -> tuple[ast.AST, list[ast.AST]] | None:
        operation = _call_name(call).casefold()
        functional_collection = call.args[0] if call.args else None
        if (
            functional_collection is not None
            and route_collection_receiver(
                functional_collection, route_collections
            )
            is not None
        ):
            if operation == "append" and len(call.args) == 2 and not call.keywords:
                return functional_collection, [call.args[1]]
            if (
                operation in {"extend", "iadd"}
                and len(call.args) == 2
                and not call.keywords
            ):
                values = call.args[1]
                if isinstance(values, (ast.List, ast.Tuple, ast.Set)):
                    return functional_collection, list(values.elts)
            if operation == "insert" and len(call.args) == 3 and not call.keywords:
                return functional_collection, [call.args[2]]
            if operation in {
                "__delitem__",
                "__iadd__",
                "__setitem__",
                "append",
                "clear",
                "delitem",
                "extend",
                "iadd",
                "insert",
                "pop",
                "remove",
                "reverse",
                "setitem",
                "sort",
            }:
                raise AssertionError(
                    "Unresolved functional route collection mutation: "
                    f"{ast.unparse(call)}"
                )
        member = _static_member_reference(call.func)
        if (
            member is None
            or route_collection_receiver(member[0], route_collections) is None
        ):
            return None
        collection, operation = member
        if operation == "append" and len(call.args) == 1 and not call.keywords:
            return collection, [call.args[0]]
        if operation == "extend" and len(call.args) == 1 and not call.keywords:
            values = call.args[0]
            if isinstance(values, (ast.List, ast.Tuple, ast.Set)):
                return collection, list(values.elts)
        if operation == "insert" and len(call.args) == 2 and not call.keywords:
            return collection, [call.args[1]]
        if operation in {"append", "extend", "insert"}:
            raise AssertionError(
                "Unresolved route collection publication: "
                f"{ast.unparse(call)}"
            )
        if operation in {"__iadd__", "__setitem__"}:
            raise AssertionError(
                "Unresolved route collection mutation: "
                f"{ast.unparse(call)}"
            )
        return None

    def filters_existing_route_objects(value: ast.AST, collection: ast.AST) -> bool:
        """Whether a slice replacement only removes members of this collection."""

        if not isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return False
        if len(value.generators) != 1:
            return False
        generator = value.generators[0]
        return (
            not generator.is_async
            and isinstance(generator.target, ast.Name)
            and isinstance(value.elt, ast.Name)
            and value.elt.id == generator.target.id
            and ast.dump(generator.iter) == ast.dump(collection)
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
        inherited_route_factories: dict[str, ast.Call],
        inherited_route_collections: dict[str, _StaticBinding],
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
            active_route_factories: dict[str, ast.Call],
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
                expressions, blocks = _compound_flow_parts(node)
                for expression in expressions:
                    visit(
                        expression,
                        nested_strings,
                        nested_methods,
                        active_route_aliases,
                        active_route_factories,
                    )
                for block in blocks:
                    # A registration alias can be introduced and consumed in
                    # the same feature-flagged branch. Replay that branch in
                    # source order so its decorators are inventoried (or an
                    # ambiguous alias fails closed) instead of inheriting only
                    # the pre-branch aliases.
                    walk_scope(
                        block,
                        prefixes,
                        nested_strings,
                        nested_methods,
                        active_route_aliases,
                        active_route_factories,
                        active_route_collections,
                        module_scope=module_scope,
                        class_body_uses_module_globals=(
                            class_body_uses_module_globals
                        ),
                    )
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    route_decorator = (
                        decorator
                        if isinstance(decorator, ast.Call)
                        else active_route_factories.get(decorator.id)
                        if isinstance(decorator, ast.Name)
                        else None
                    )
                    if route_decorator is None:
                        continue
                    methods = _route_methods(
                        route_decorator,
                        active_methods,
                        active_route_aliases,
                        active_strings,
                        {"app", *prefixes},
                    )
                    if not methods:
                        continue
                    receiver = _route_receiver_name(
                        route_decorator,
                        active_route_aliases,
                        active_strings,
                    )
                    if receiver == "app":
                        prefix = ""
                    elif receiver is not None and receiver in prefixes:
                        prefix = prefixes[receiver]
                    else:
                        raise AssertionError(
                            "Unresolved route decorator receiver: "
                            f"{ast.unparse(route_decorator.func)}"
                        )
                    declarations.append(
                        (
                            methods,
                            prefix
                            + _route_path(route_decorator, active_strings),
                        )
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
                child_route_factories = (
                    module_runtime_route_factories
                    if module_scope or class_body_uses_module_globals
                    else active_route_factories
                )
                child_route_collections = (
                    module_runtime_route_collections
                    if module_scope or class_body_uses_module_globals
                    else active_route_collections
                )
                walk_scope(
                    node.body,
                    child_prefixes,
                    child_strings,
                    child_methods,
                    child_route_aliases,
                    child_route_factories,
                    child_route_collections,
                )
                return
            if isinstance(node, ast.ClassDef):
                walk_scope(
                    node.body,
                    prefixes,
                    active_strings,
                    active_methods,
                    active_route_aliases,
                    active_route_factories,
                    active_route_collections,
                    class_body_uses_module_globals=(
                        module_scope or class_body_uses_module_globals
                    ),
                )
                return
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    list(node.targets)
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                route_targets: list[tuple[ast.AST, bool]] = []
                for target in targets:
                    if (
                        not isinstance(target, ast.Name)
                        and route_collection_receiver(
                            target, active_route_collections
                        )
                        is not None
                    ):
                        route_targets.append((target, True))
                    elif (
                        isinstance(target, ast.Subscript)
                        and route_collection_receiver(
                            target.value, active_route_collections
                        )
                        is not None
                    ):
                        route_targets.append(
                            (target.value, isinstance(target.slice, ast.Slice))
                        )
                    elif any(
                        route_collection_receiver(
                            candidate, active_route_collections
                        )
                        is not None
                        for candidate in ast.walk(target)
                        if not isinstance(candidate, ast.Name)
                    ):
                        raise AssertionError(
                            "Unresolved route collection assignment: "
                            f"{ast.unparse(node)}"
                        )
                if route_targets:
                    if len(route_targets) != 1 or node.value is None:
                        raise AssertionError(
                            "Unresolved route collection assignment: "
                            f"{ast.unparse(node)}"
                        )
                    collection, requires_iterable = route_targets[0]
                    if requires_iterable:
                        if filters_existing_route_objects(node.value, collection):
                            return
                        if not isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                            raise AssertionError(
                                "Unresolved route collection assignment: "
                                f"{ast.unparse(node)}"
                            )
                        route_objects = list(node.value.elts)
                    else:
                        route_objects = [node.value]
                    declarations.extend(
                        route_object_declaration(
                            route_object,
                            collection,
                            active_strings,
                            active_methods,
                            prefixes,
                            active_route_collections,
                        )
                        for route_object in route_objects
                    )
                    return
            if isinstance(node, ast.AugAssign):
                collection = node.target
                receiver = route_collection_receiver(
                    collection, active_route_collections
                )
                if (
                    receiver is None
                    and isinstance(node.target, ast.Subscript)
                    and isinstance(node.target.slice, ast.Slice)
                ):
                    collection = node.target.value
                    receiver = route_collection_receiver(
                        collection, active_route_collections
                    )
                if receiver is not None:
                    if not isinstance(node.op, ast.Add) or not isinstance(
                        node.value, (ast.List, ast.Tuple, ast.Set)
                    ):
                        raise AssertionError(
                            "Unresolved augmented route collection publication: "
                            f"{ast.unparse(node)}"
                        )
                    declarations.extend(
                        route_object_declaration(
                            route_object,
                            collection,
                            active_strings,
                            active_methods,
                            prefixes,
                            active_route_collections,
                        )
                        for route_object in node.value.elts
                    )
                    return
            if isinstance(node, ast.Call):
                publication = appended_route_objects(
                    node, active_route_collections
                )
                if publication is not None:
                    collection, route_objects = publication
                    declarations.extend(
                        route_object_declaration(
                            route_object,
                            collection,
                            active_strings,
                            active_methods,
                            prefixes,
                            active_route_collections,
                        )
                        for route_object in route_objects
                    )
                decorator_factory = (
                    node.func
                    if isinstance(node.func, ast.Call)
                    else active_route_factories.get(node.func.id)
                    if isinstance(node.func, ast.Name)
                    else None
                )
                factory_registration = (
                    _route_registration_name(
                        decorator_factory,
                        active_route_aliases,
                        active_strings,
                        {"app", *prefixes},
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
                        decorator_factory,
                        active_route_aliases,
                        active_strings,
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
                                active_strings,
                                {"app", *prefixes},
                            ),
                            prefix
                            + _route_path(decorator_factory, active_strings),
                        )
                    )
                registration = _route_registration_name(
                    node,
                    active_route_aliases,
                    active_strings,
                    {"app", *prefixes},
                )
                if registration == "include_router":
                    receiver = _route_receiver_name(
                        node,
                        active_route_aliases,
                        active_strings,
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
                        node,
                        active_route_aliases,
                        active_strings,
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
                                active_strings,
                                {"app", *prefixes},
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
                    active_route_factories,
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
            declarations.extend(
                _fastapi_generated_route_declarations(
                    [statement], active_strings, active_methods
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
            active_route_factories = _scope_route_decorator_factories(
                statements[:index],
                prefixes,
                active_route_aliases,
                inherited_route_factories,
            )
            active_route_collections = scope_route_collection_aliases(
                statements[:index],
                prefixes,
                inherited_route_collections,
            )
            visit(
                statement,
                active_strings,
                active_methods,
                active_route_aliases,
                active_route_factories,
            )

    walk_scope(tree.body, {}, {}, {}, {}, {}, {}, module_scope=True)
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
    string_constants: dict[str, str] | None = None,
    route_receivers: set[str] | None = None,
) -> tuple[str, ...]:
    method = _route_registration_name(
        decorator,
        route_aliases,
        string_constants,
        route_receivers,
    )
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
        "kestrel_sovereign/agent/orchestrator_engine.py::"
        "_dispatch_feature_tool",
        "kestrel_sovereign/agent/orchestrator_engine.py::execute_named_tool",
        "kestrel_sovereign/agent/tool_registry.py::register_dynamic_tools",
        "kestrel_sovereign/kestrel_agent.py::"
        "KestrelAgent._handle_constitution_receipt_tool",
        "kestrel_sovereign/kestrel_agent.py::"
        "KestrelAgent.register_constitution_receipt_tool",
        "kestrel_sovereign/features/base.py::"
        "Feature.execute_as_subagent",
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

    engine_path = REPO_ROOT / "kestrel_sovereign/agent/orchestrator_engine.py"
    engine_tree = ast.parse(
        engine_path.read_text(encoding="utf-8"),
        filename=str(engine_path),
    )
    feature_dispatchers = [
        node
        for node in ast.walk(engine_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_dispatch_feature_tool"
    ]
    assert len(feature_dispatchers) == 1
    assert any(
        isinstance(node, ast.Call)
        and _call_name(node) == "execute_as_subagent"
        for node in ast.walk(feature_dispatchers[0])
    ), "The classified feature dispatcher must call the execution loop"


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
        "    def bound(self, tools):\n"
        "        publish = self._direct_tools.update\n"
        "        publish(tools)\n\n"
        "    def conditional(self, tools, disabled):\n"
        "        publish = consume if disabled else self._direct_tools.update\n"
        "        publish(tools)\n\n"
        "    def rebound(self, tools):\n"
        "        publish = self._direct_tools.update\n"
        "        publish = consume\n"
        "        publish(tools)\n\n"
        "    def replace(self, tools):\n"
        "        self._direct_tools = tools\n\n"
        "    def setattr_replace(self, tools):\n"
        "        setattr(self, '_direct_tools', tools)\n\n"
        "    def initialize(self):\n"
        "        self._direct_tools = {}\n\n"
        "    def setattr_initialize(self):\n"
        "        setattr(self, '_direct_tools', {})\n\n"
        "    def remove(self):\n"
        "        self._direct_tools.pop('x', None)\n"
    )

    assert _direct_tool_writer_surfaces(tree, "example.py") == {
        "example.py::Publisher.alias",
        "example.py::Publisher.bound",
        "example.py::Publisher.conditional",
        "example.py::Publisher.default",
        "example.py::Publisher.direct",
        "example.py::Publisher.replace",
        "example.py::Publisher.setattr_replace",
        "example.py::Publisher.union",
    }


@pytest.mark.parametrize(
    "publication",
    (
        "operator.setitem(self._direct_tools, 'x', tool)",
        "dict.__setitem__(self._direct_tools, 'x', tool)",
        "dict.__ior__(self._direct_tools, {'x': tool})",
        "self._direct_tools.__ior__({'x': tool})",
        "dict.update(self._direct_tools, {'x': tool})",
        "operator.ior(self._direct_tools, {'x': tool})",
    ),
)
def test_functional_dynamic_tool_registry_writes_are_inventoried(
    publication: str,
) -> None:
    tree = ast.parse(
        "class Publisher:\n"
        "    def publish(self, tool):\n"
        f"        {publication}\n"
    )

    assert _direct_tool_writer_surfaces(tree, "example.py") == {
        "example.py::Publisher.publish"
    }


def test_dynamic_tool_registry_flows_through_helpers_and_loops() -> None:
    tree = ast.parse(
        "def publish(registry, tools):\n"
        "    registry.update(tools)\n\n"
        "class Publisher:\n"
        "    def _publish(self, registry, tools):\n"
        "        registry.update(tools)\n\n"
        "    def helper(self, tools):\n"
        "        publish(self._direct_tools, tools)\n\n"
        "    def aliased_helper(self, tools):\n"
        "        helper = publish\n"
        "        helper(self._direct_tools, tools)\n\n"
        "    def method_helper(self, tools):\n"
        "        self._publish(self._direct_tools, tools)\n\n"
        "    def loop(self, tools):\n"
        "        for registry in [self._direct_tools]:\n"
        "            registry.update(tools)\n"
    )

    assert _direct_tool_writer_surfaces(tree, "example.py") == {
        "example.py::Publisher.aliased_helper",
        "example.py::Publisher.helper",
        "example.py::Publisher.loop",
        "example.py::Publisher.method_helper",
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


def test_dynamic_tool_registry_preserves_maybe_live_branch_aliases() -> None:
    tree = ast.parse(
        "class Publisher:\n"
        "    def publish(self, tool, disabled):\n"
        "        registry = self._direct_tools\n"
        "        if disabled:\n"
        "            registry = {}\n"
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
        "        registry.register_batch(sources)\n\n"
        "    def bound_method(self, source):\n"
        "        publish = self.source_registry.register_with_policy\n"
        "        publish(source)\n\n"
        "    def conditional_method(self, source, disabled):\n"
        "        publish = (\n"
        "            consume if disabled else self.source_registry.register\n"
        "        )\n"
        "        publish(source)\n\n"
        "    def rebound_method(self, source):\n"
        "        publish = self.source_registry.register\n"
        "        publish = consume\n"
        "        publish(source)\n"
    )

    assert _runtime_signal_publication_surfaces(tree, "example.py") == {
        "example.py::Publisher.bound_method",
        "example.py::Publisher.conditional_method",
        "example.py::Publisher.local_alias",
        "example.py::Publisher.signal_registry",
    }


def test_runtime_signal_publication_flows_through_helpers_and_loops() -> None:
    tree = ast.parse(
        "def publish(registry, source):\n"
        "    registry.register(source)\n\n"
        "class Publisher:\n"
        "    def _publish(self, registry, source):\n"
        "        registry.register(source)\n\n"
        "    def helper(self, source):\n"
        "        publish(self.source_registry, source)\n\n"
        "    def aliased_helper(self, source):\n"
        "        helper = publish\n"
        "        helper(self.source_registry, source)\n\n"
        "    def method_helper(self, source):\n"
        "        self._publish(self.source_registry, source)\n\n"
        "    def loop(self, source):\n"
        "        for registry in [self.source_registry]:\n"
        "            registry.register(source)\n"
    )

    assert _runtime_signal_publication_surfaces(tree, "example.py") == {
        "example.py::Publisher.aliased_helper",
        "example.py::Publisher.helper",
        "example.py::Publisher.loop",
        "example.py::Publisher.method_helper",
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
        'tasks = CRON_TASKS\ntasks.append('
        '("late", SignalMode.ACTION, frozenset()))',
    ],
    ids=["append", "augmented-assignment", "reassignment", "alias-append"],
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


def test_scheduler_inventory_rejects_compound_alias_mutations() -> None:
    tree = ast.parse(
        "CRON_TASKS = [('base', mode, resources)]\n"
        "if enabled:\n"
        "    tasks = CRON_TASKS\n"
        "    tasks.append(('late', mode, resources))\n"
    )

    with pytest.raises(
        AssertionError,
        match="Conditional or indirect CRON_TASKS mutation",
    ):
        _cron_task_names(tree)


@pytest.mark.parametrize(
    "mutation",
    [
        "CRON_TASKS[:] = [('late', mode, resources)]",
        "tasks = CRON_TASKS\ntasks[0] = ('late', mode, resources)",
        "del CRON_TASKS[0]",
    ],
    ids=["slice", "alias-item", "delete-item"],
)
def test_scheduler_inventory_rejects_subscript_mutations(
    mutation: str,
) -> None:
    tree = ast.parse(
        "CRON_TASKS = [('base', mode, resources)]\n" + mutation + "\n"
    )

    with pytest.raises(
        AssertionError,
        match="Unsupported subscript CRON_TASKS mutation",
    ):
        _cron_task_names(tree)


@pytest.mark.parametrize(
    "binding",
    [
        "mutate = CRON_TASKS.append",
        'mutate = getattr(CRON_TASKS, "append")',
        "if enabled:\n    mutate = CRON_TASKS.append",
    ],
    ids=["attribute", "static-getattr", "conditional"],
)
def test_scheduler_inventory_rejects_bound_method_mutations(
    binding: str,
) -> None:
    tree = ast.parse(
        "CRON_TASKS = [('base', mode, resources)]\n"
        + binding
        + "\nmutate(('late', mode, resources))\n"
    )

    with pytest.raises(
        AssertionError,
        match="Unsupported bound CRON_TASKS method mutation",
    ):
        _cron_task_names(tree)


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


def test_core_cli_predispatch_and_parser_commands_are_discovered() -> None:
    discovered = _discovered_core_cli_surfaces()
    assert "kestrel_sovereign/cli.py::kestrel help" in discovered

    tree = ast.parse(
        "def build_parser():\n"
        "    subparsers = parser.add_subparsers(dest='command')\n"
        "    alias = subparsers\n"
        "    alias.add_parser('help', aliases=['assist'])\n\n"
        "def main():\n"
        "    if not args.command:\n"
        "        return 1\n"
        "    if args.command in {'help', 'assist'}:\n"
        "        return 0\n"
        "    commands = {'start': start}\n"
        "    handler = commands.get(args.command)\n"
    )
    assert _core_cli_command_names(tree, {}) == {
        "assist",
        "help",
        "start",
    }


def test_core_cli_parser_commands_without_dispatch_fail_closed() -> None:
    tree = ast.parse(
        "def build_parser():\n"
        "    subparsers = parser.add_subparsers(dest='command')\n"
        "    subparsers.add_parser('orphan')\n\n"
        "def main():\n"
        "    commands = {'start': start}\n"
        "    handler = commands.get(args.command)\n"
    )
    with pytest.raises(AssertionError, match="lack a dispatch path"):
        _core_cli_command_names(tree, {})


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
        "    registry = commands\n"
        "    registry.update({'create': create})\n"
        "    handler = commands.get(args.command)\n"
    )

    assert _core_cli_command_names(tree, {}) == {
        "start",
        "terminate",
        "restart",
        "create",
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


def test_route_declarations_resolve_static_getattr_registration_aliases() -> None:
    tree = ast.parse(
        'router = APIRouter(prefix="/api")\n'
        'publish = getattr(router, "delete")\n'
        '@publish("/agents/{name}")\n'
        'def delete_agent():\n'
        '    pass\n'
    )

    assert _route_declarations(
        tree,
        _module_string_constants(tree),
        _module_string_collections(tree),
    ) == [(("DELETE",), "/api/agents/{name}")]


def test_route_declarations_reject_dynamic_getattr_registration_aliases() -> None:
    source = (
        "router = APIRouter()\n"
        "publish = getattr(router, method_name)\n"
        "@publish('/api/agents/{name}')\n"
        "def route():\n"
        "    pass\n"
    )

    with pytest.raises(
        AssertionError, match="Unresolved route registration alias"
    ):
        _route_declarations(ast.parse(source), {}, {})


def test_route_declarations_resolve_or_reject_immediate_getattr_registrations() -> None:
    resolved = ast.parse(
        "router = APIRouter()\n"
        "method_name = 'add_api_route'\n"
        "getattr(router, method_name)(\n"
        "    '/api/agents/{name}', endpoint, methods=['DELETE']\n"
        ")\n"
        "method_name = 'get'\n"
        "getattr(router, method_name)('/api/agents/{name}/status')(status)\n"
    )
    unresolved = ast.parse(
        "router = APIRouter()\n"
        "getattr(router, method_name)(\n"
        "    '/api/agents/{name}', endpoint, methods=['DELETE']\n"
        ")\n"
    )

    assert _route_declarations(
        resolved,
        _module_string_constants(resolved),
        _module_string_collections(resolved),
    ) == [
        (('DELETE',), '/api/agents/{name}'),
        (('GET',), '/api/agents/{name}/status'),
    ]
    with pytest.raises(
        AssertionError, match="Unresolved immediate route registration"
    ):
        _route_declarations(unresolved, {}, {})


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


def test_route_declarations_replay_aliases_inside_compound_blocks() -> None:
    tree = ast.parse(
        "router = APIRouter(prefix='/api')\n"
        "if enabled:\n"
        "    register = router.post\n"
        "    @register('/agents/{name}/terminate')\n"
        "    def terminate():\n"
        "        pass\n"
    )

    assert _route_declarations(tree, {}, {}) == [
        (("POST",), "/api/agents/{name}/terminate")
    ]


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


def test_route_objects_appended_to_route_collections_are_inventoried() -> None:
    tree = ast.parse(
        'router = APIRouter(prefix="/api")\n'
        'router.routes.append(APIRoute(\n'
        '    "/agents/{name}/terminate", endpoint, methods=["DELETE"]\n'
        '))\n'
        'app.router.routes.extend([\n'
        '    Route("/host/restart", endpoint, methods=["POST"]),\n'
        '    WebSocketRoute("/agents/{name}/events", socket_endpoint),\n'
        '])\n'
    )

    assert _route_declarations(tree, {}, {}) == [
        (("DELETE",), "/api/agents/{name}/terminate"),
        (("POST",), "/host/restart"),
        (("WEBSOCKET",), "/agents/{name}/events"),
    ]


def test_functional_route_collection_publications_are_inventoried() -> None:
    tree = ast.parse(
        "app = Starlette()\n"
        "list.append(app.routes, Route(\n"
        "    '/api/agents/{name}/stop', endpoint, methods=['POST']\n"
        "))\n"
        "operator.iadd(app.routes, [\n"
        "    Route('/host/restart', endpoint, methods=['POST'])\n"
        "])\n"
    )
    unresolved = ast.parse(
        "app = Starlette()\n"
        "operator.setitem(app.routes, index, build_route())\n"
    )

    assert _route_declarations(tree, {}, {}) == [
        (("POST",), "/api/agents/{name}/stop"),
        (("POST",), "/host/restart"),
    ]
    with pytest.raises(
        AssertionError, match="functional route collection mutation"
    ):
        _route_declarations(unresolved, {}, {})


def test_route_collection_alias_publications_are_inventoried() -> None:
    aliased = ast.parse(
        "app = Starlette()\n"
        "routes = app.routes\n"
        "published = routes\n"
        "published.append(Route(\n"
        "    '/api/agents/{name}/terminate', endpoint, methods=['DELETE']\n"
        "))\n"
    )
    rebound = ast.parse(
        "app = Starlette()\n"
        "routes = app.routes\n"
        "routes = []\n"
        "routes.append(Route('/local', endpoint))\n"
    )

    assert _route_declarations(aliased, {}, {}) == [
        (("DELETE",), "/api/agents/{name}/terminate"),
    ]
    assert _route_declarations(rebound, {}, {}) == []


def test_direct_route_collection_mutations_are_inventoried_or_rejected() -> None:
    tree = ast.parse(
        'router = APIRouter(prefix="/api")\n'
        'app.routes.insert(0, Route(\n'
        '    "/api/agents/{name}/stop", endpoint, methods=["POST"]\n'
        '))\n'
        'router.routes[:] = [\n'
        '    APIRoute("/agents/{name}/terminate", endpoint, methods=["DELETE"]),\n'
        '    WebSocketRoute("/agents/{name}/events", socket_endpoint),\n'
        ']\n'
    )
    unresolved = ast.parse(
        "app.routes.insert(index=position, object=build_route())\n"
    )
    unsupported = ast.parse("app.routes.__setitem__(slice(None), build_routes())\n")
    destructured = ast.parse(
        "app.routes, marker = [Route('/hidden', endpoint)], 1\n"
    )
    deletion = ast.parse("del app.routes[:]\n")
    filtered = ast.parse(
        "app.routes[:] = [route for route in app.routes if keep(route)]\n"
    )

    assert _route_declarations(tree, {}, {}) == [
        (("POST",), "/api/agents/{name}/stop"),
        (("DELETE",), "/api/agents/{name}/terminate"),
        (("WEBSOCKET",), "/api/agents/{name}/events"),
    ]
    with pytest.raises(AssertionError, match="route collection publication"):
        _route_declarations(unresolved, {}, {})
    with pytest.raises(AssertionError, match="route collection mutation"):
        _route_declarations(unsupported, {}, {})
    with pytest.raises(AssertionError, match="route collection assignment"):
        _route_declarations(destructured, {}, {})
    assert _route_declarations(deletion, {}, {}) == []
    assert _route_declarations(filtered, {}, {}) == []


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
        'stored = router.delete("/agents/{agent}/terminate")\n'
        'stored_alias = stored\n'
        'stored_alias(handler)\n'
    )

    assert _route_declarations(tree, {}, {}) == [
        (("POST",), "/api/agents/{agent}/stop"),
        (("POST", "DELETE"), "/api/agents/{agent}/hold"),
        (("WEBSOCKET",), "/api/agents/{agent}/events"),
        (("PATCH",), "/api/agents/{agent}/restart"),
        (("DELETE",), "/api/agents/{agent}/terminate"),
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

    supplied_routes = ast.parse(
        'app = FastAPI(routes=[\n'
        '    Route("/api/agents/{name}/terminate", endpoint, '
        'methods=["DELETE"]),\n'
        '    WebSocketRoute("/api/agents/{name}/events", socket_endpoint),\n'
        '])\n'
    )
    assert _route_declarations(supplied_routes, {}, {}) == [
        (("DELETE",), "/api/agents/{name}/terminate"),
        (("WEBSOCKET",), "/api/agents/{name}/events"),
        (("GET", "HEAD"), "/openapi.json"),
        (("GET", "HEAD"), "/docs"),
        (("GET", "HEAD"), "/docs/oauth2-redirect"),
        (("GET", "HEAD"), "/redoc"),
    ]

    router_routes = ast.parse(
        'router = APIRouter(prefix="/api", routes=['
        'APIRoute("/agents/{name}/stop", endpoint, methods=["POST"]), '
        'WebSocketRoute("/agents/{name}/events", socket_endpoint)])\n'
    )
    assert _route_declarations(router_routes, {}, {}) == [
        (("POST",), "/api/agents/{name}/stop"),
        (("WEBSOCKET",), "/api/agents/{name}/events"),
    ]

    nested_router_routes = ast.parse(
        "class Feature:\n"
        "    def get_router(self):\n"
        '        return APIRouter(prefix="/api", routes=['
        'APIRoute("/agents/{name}/hold", endpoint, methods=["POST"]), '
        'WebSocketRoute("/agents/{name}/events", socket_endpoint)])\n'
    )
    assert _route_declarations(nested_router_routes, {}, {}) == [
        (("POST",), "/api/agents/{name}/hold"),
        (("WEBSOCKET",), "/api/agents/{name}/events"),
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


def test_call_produced_tool_decorators_are_inventoried() -> None:
    tree = ast.parse(
        'publish = tool("terminate_child", "Terminate a child")\n'
        "publish_alias = publish\n"
        "@publish_alias\n"
        "def implementation(target):\n"
        "    pass\n"
    )
    function = tree.body[2]
    assert isinstance(function, ast.FunctionDef)

    assert _public_tool_name(
        function.decorator_list[0],
        function.name,
        {},
        _tool_decorator_aliases(tree),
    ) == "terminate_child"


def test_imperatively_decorated_tools_are_inventoried() -> None:
    tree = ast.parse(
        "class Feature:\n"
        "    def direct(self, target):\n"
        "        pass\n"
        "    direct = tool('terminate_child', 'Terminate child')(direct)\n\n"
        "    publish = tool('stop_peer', 'Stop peer')\n"
        "    def aliased(self, target):\n"
        "        pass\n"
        "    aliased = publish(aliased)\n\n"
        "    def ordinary(self):\n"
        "        pass\n"
        "    ordinary = unrelated(ordinary)\n"
    )

    assert _imperatively_decorated_tool_names(
        tree,
        _tool_decorator_aliases(tree),
    ) == {"stop_peer", "terminate_child"}
    assert _tool_surfaces_from_module(tree, "feature.py") == {
        "feature.py::stop_peer",
        "feature.py::terminate_child",
    }


def test_repository_scans_reuse_parsed_trees_and_analysis_summaries(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "controller.py"
    source_path.write_text(
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        target.shutdown()\n",
        encoding="utf-8",
    )
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
        assert discover.cache_parameters() == {
            "maxsize": None,
            "typed": False,
        }

    # Exercise the cache/immutability contract on the bounded built-in command
    # inventory. The individual completeness tests exercise every expensive
    # repository discovery; repeating all of them cold in this meta-test makes
    # xdist duplicate the entire checkout scan in an otherwise isolated worker.
    first = _discovered_builtin_command_surfaces()
    before_second = _discovered_builtin_command_surfaces.cache_info()
    second = _discovered_builtin_command_surfaces()
    assert isinstance(first, frozenset)
    assert second is first
    assert (
        _discovered_builtin_command_surfaces.cache_info().hits
        == before_second.hits + 1
    )


def test_exhaustive_repository_scan_is_an_explicit_ci_gate() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        )
        and ast.unparse(node.value) == "pytest.mark.authority_audit"
        for node in tree.body
    )

    conftest = (REPO_ROOT / "tests/conftest.py").read_text(encoding="utf-8")
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )
    assert '"--run-authority-audit"' in conftest
    assert '"authority_audit" in item.keywords' in conftest
    assert "-m authority_audit --run-authority-audit" in workflow


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


def test_provenance_summaries_follow_generator_yields() -> None:
    tree = ast.parse(
        "def derive(request):\n"
        "    yield from request.causation_chain\n\n"
        "def dispatch(request, target):\n"
        "    if any(derive(request)):\n"
        "        terminate_child(target)\n"
    )

    assert _authority_provenance_lines(tree) == {5}


def test_repository_scan_prefilters_modules_without_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "kestrel_sovereign/__init__.py"
    source = _source_text(source_path).casefold()
    assert not any(
        marker in source
        for marker in ("causation", "orchestrator", "current_chain")
    )
    _cached_authority_provenance_lines.cache_clear()

    def forbidden_analysis(*_args: object, **_kwargs: object) -> set[int]:
        raise AssertionError("marker-free module reached recursive analysis")

    for analyzer in (
        "_module_provenance_constant_aliases",
        "_module_imported_provenance_return_helper_aliases",
        "_authority_provenance_lines",
    ):
        monkeypatch.setitem(globals(), analyzer, forbidden_analysis)
    assert _cached_authority_provenance_lines(source_path) == frozenset()

    unused_helper = tmp_path / "unused_helper.py"
    unused_helper.write_text(
        "def context(request):\n"
        "    return request.causation_chain\n",
        encoding="utf-8",
    )
    unused_import = tmp_path / "unused_import.py"
    unused_import.write_text(
        "from unused_helper import context\n\n"
        "def ordinary(value):\n"
        "    return value\n",
        encoding="utf-8",
    )
    assert _cached_authority_provenance_lines(unused_import) == frozenset()


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


def test_spawn_authority_distinguishes_issuance_from_control_boundaries() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    assert "signs while `child_did` is unset" not in audit

    creation_row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Create child |")
    )
    assert "[#3133]" in creation_row
    assert "Enforced by" in creation_row
    assert "final child DID" in creation_row

    spawn_surface_row = next(
        line
        for line in audit.splitlines()
        if "features/spawn/feature.py::spawn_agent`" in line
    )
    assert "Enforced by #3133" in spawn_surface_row
    assert "#3142" not in spawn_surface_row
    assert "Signature is invalidated" not in spawn_surface_row

    for action in (
        "List/read child work",
        "Delegate work to child",
        "Terminate/offboard child",
    ):
        row = next(
            line for line in audit.splitlines() if line.startswith(f"| {action} |")
        )
        assert "[#3142]" in row
        assert "Defect" in row


def test_main_fixed_surfaces_record_current_authority_enforcement() -> None:
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
        assert "Defect:" not in row

    scoped_tools = {
        "features/consent/feature.py::consent_log": "#3229",
        "features/consent/feature.py::consent_stats": "#3229",
        "features/audit_anchor/feature.py::audit_anchor": "#3230",
        "features/audit_anchor/feature.py::audit_anchor_status": "#3230",
        "features/audit_anchor/feature.py::audit_verify": "#3230",
    }
    for surface, issue in scoped_tools.items():
        row = next(line for line in audit.splitlines() if surface in line)
        assert issue in row
        assert re.search(r"\bD-\d+", row) is None
        assert "scoped" in row

    for suffix in ("install", "remove"):
        canonical = next(
            line
            for line in audit.splitlines()
            if f"endpoints/features.py::POST /api/features/{{name}}/{suffix}`" in line
        )
        alias = next(
            line
            for line in audit.splitlines()
            if "endpoints/features.py::POST "
            f"/api/agents/{{selected_agent_name}}/api/features/{{name}}/{suffix}`"
            in line
        )
        assert "require_sovereign_host_lifecycle" in canonical
        assert "| H —" in alias
        assert "#3214" in canonical and "#3214" in alias

    for suffix in ("summary", "metrics/{metric_name}"):
        canonical = next(
            line
            for line in audit.splitlines()
            if "endpoints/observability.py::GET "
            f"/api/observability/{suffix}`" in line
        )
        alias = next(
            line
            for line in audit.splitlines()
            if "endpoints/observability.py::GET "
            f"/api/agents/{{selected_agent_name}}/api/observability/{suffix}`"
            in line
        )
        assert "#3215" in canonical and "#3215" in alias
        assert "scoped" in canonical and "| A —" in alias


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
        "cli.py::kestrel create`",
        "cli.py::kestrel start`",
        "cli.py::kestrel terminate`",
        "cli.py::kestrel restart`",
        "cli.py::kestrel update`",
    ):
        tool_row = next(line for line in audit.splitlines() if surface in line)
        assert "D-3233" in tool_row


def test_multi_agent_deployment_control_records_3223_enforcement() -> None:
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
    assert "Defect:" not in action_row
    assert "#3223" in tool_row
    assert "Sovereign-gated" in tool_row
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


def test_webhook_collision_refusal_and_open_unlimited_mode_are_recorded() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| General webhook ingress |")
    )
    assert "[#3216]" in row
    assert "refuses duplicate ownership" in row
    assert "Defect:" not in row
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


def test_shared_local_model_mutations_record_3221_enforcement() -> None:
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
        assert "#3221" in row
        assert "Sovereign-gated" in row
        assert "shared" in row.casefold()


def test_routed_rasa_target_binding_is_recorded_after_3220() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    row = next(
        line
        for line in audit.splitlines()
        if "endpoints/rasa_shim.py::POST "
        "/api/agents/{selected_agent_name}/webhooks/rest/webhook" in line
    )
    assert "| W —" in row
    assert "#3220" in row
    assert "bound to the trusted routed agent" in row
    assert "per-agent" in row


def test_sovereignty_cache_reads_are_owner_scoped_after_3225() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    action_row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Browse sovereignty export cache |")
    )
    assert "| Self |" in action_row
    assert "durable receipts" in action_row
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
        assert "| A —" in row
        assert "receipt-scoped" in row
        assert "routed agent's own" in row


def test_ipfs_reads_are_split_by_owner_and_host_authority_after_3226() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    action_row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Inspect local IPFS node and pins |")
    )
    assert "Self for agent pins" in action_row
    assert "sovereign/delegated" in action_row
    assert "[#3226]" in action_row

    assert (
        "kestrel_sovereign/endpoints/models.py::GET /api/ipfs/node"
        in _discovered_http_surfaces()
    )

    for route in (
        "/api/ipfs/status",
        "/api/agents/{selected_agent_name}/api/ipfs/status",
    ):
        row = next(
            line
            for line in audit.splitlines()
            if f"endpoints/models.py::GET {route}`" in line
        )
        assert "| A —" in row
        assert "receipt-owned pins" in row

    for route, boundary in {
        "/api/ipfs/node": "canonical route has no selected-agent context",
        "/api/agents/{selected_agent_name}/api/ipfs/node": (
            "selected-agent prefix grants no host authority"
        ),
    }.items():
        host_row = next(
            line
            for line in audit.splitlines()
            if f"endpoints/models.py::GET {route}`" in line
        )
        assert "| H —" in host_row
        assert "sovereign/delegated" in host_row
        assert boundary in host_row


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


def _match_pattern_provenance_marker(pattern: ast.pattern) -> ast.AST | None:
    """Return one static provenance marker named by a match pattern."""

    if isinstance(pattern, ast.MatchValue):
        return pattern.value if _has_provenance_token(pattern.value) else None
    if isinstance(pattern, ast.MatchMapping):
        for key in pattern.keys:
            if _has_provenance_token(key):
                return key
        children = pattern.patterns
    elif isinstance(pattern, ast.MatchClass):
        if _has_provenance_token(pattern.cls):
            return pattern.cls
        for attribute in pattern.kwd_attrs:
            marker = ast.Constant(value=attribute)
            if _has_provenance_token(marker):
                return marker
        children = [*pattern.patterns, *pattern.kwd_patterns]
    elif isinstance(pattern, ast.MatchAs):
        children = [pattern.pattern] if pattern.pattern is not None else []
    elif isinstance(pattern, (ast.MatchOr, ast.MatchSequence)):
        children = pattern.patterns
    else:
        children = []
    return next(
        (
            marker
            for child in children
            if (marker := _match_pattern_provenance_marker(child)) is not None
        ),
        None,
    )


def _match_pattern_provenance_assignments(
    pattern: ast.pattern,
    subject_source: ast.AST | None,
) -> list[tuple[set[str], ast.AST]]:
    """Model names bound from provenance-bearing match components."""

    assignments: list[tuple[set[str], ast.AST]] = []
    if isinstance(pattern, ast.MatchAs):
        binding_source = (
            _match_pattern_provenance_marker(pattern.pattern)
            if pattern.pattern is not None
            else None
        )
        if binding_source is None:
            binding_source = subject_source
        if pattern.name is not None and binding_source is not None:
            assignments.append(({pattern.name.casefold()}, binding_source))
        if pattern.pattern is not None:
            assignments.extend(
                _match_pattern_provenance_assignments(
                    pattern.pattern,
                    subject_source,
                )
            )
        return assignments
    if isinstance(pattern, ast.MatchStar):
        if pattern.name is not None and subject_source is not None:
            assignments.append(({pattern.name.casefold()}, subject_source))
        return assignments
    if isinstance(pattern, ast.MatchMapping):
        for key, child_pattern in zip(pattern.keys, pattern.patterns):
            child_source = (
                key if _has_provenance_token(key) else subject_source
            )
            assignments.extend(
                _match_pattern_provenance_assignments(
                    child_pattern,
                    child_source,
                )
            )
        if pattern.rest is not None and subject_source is not None:
            assignments.append(({pattern.rest.casefold()}, subject_source))
        return assignments
    if isinstance(pattern, ast.MatchClass):
        class_source = (
            pattern.cls
            if _has_provenance_token(pattern.cls)
            else subject_source
        )
        for child_pattern in pattern.patterns:
            assignments.extend(
                _match_pattern_provenance_assignments(
                    child_pattern,
                    class_source,
                )
            )
        for attribute, child_pattern in zip(
            pattern.kwd_attrs,
            pattern.kwd_patterns,
        ):
            child_source = (
                ast.Constant(value=attribute)
                if _has_provenance_token(ast.Constant(value=attribute))
                else class_source
            )
            assignments.extend(
                _match_pattern_provenance_assignments(
                    child_pattern,
                    child_source,
                )
            )
        return assignments
    if isinstance(pattern, ast.MatchSequence):
        for child_pattern in pattern.patterns:
            assignments.extend(
                _match_pattern_provenance_assignments(
                    child_pattern,
                    subject_source,
                )
            )
        return assignments
    if isinstance(pattern, ast.MatchOr):
        for child_pattern in pattern.patterns:
            assignments.extend(
                _match_pattern_provenance_assignments(
                    child_pattern,
                    subject_source,
                )
            )
    return assignments


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
        "deactivate",
        "destroy",
        "disable",
        "enable",
        "hold",
        "interrupt",
        "kill",
        "offboard",
        "pause",
        "register",
        "remove",
        "reset",
        "restart",
        "retire",
        "revoke",
        "resume",
        "shutdown",
        "start",
        "stop",
        "suspend",
        "terminate",
        "unregister",
        "withdraw",
    }
)
_KESTREL_CLI_LIFECYCLE_ACTIONS = _CROSS_AGENT_LIFECYCLE_ACTIONS | {
    "create",
    "update",
}


_CROSS_AGENT_TARGETED_CONTROL_ACTIONS = frozenset(
    {
        "ask",
        "delegate",
        "deploy",
        "dispatch",
        "execute",
        "invoke",
        "list",
        "message",
        "read",
        "send",
        "spawn",
        "submit",
        "subscribe",
        "teardown",
        "verify",
    }
)


def _is_cross_agent_lifecycle_action(name: str) -> bool:
    """Recognize exact lifecycle verbs and conventional async variants."""

    normalized = name.casefold()
    if normalized in _CROSS_AGENT_LIFECYCLE_ACTIONS:
        return True
    candidates = []
    if normalized.startswith("async_"):
        candidates.append(normalized.removeprefix("async_"))
    if normalized.startswith("a"):
        candidates.append(normalized[1:])
    for suffix in ("_async", "_now", "_sync"):
        if normalized.endswith(suffix):
            candidates.append(normalized.removesuffix(suffix))
    parts = normalized.split("_")
    if len(parts) > 1:
        candidates.extend((parts[0], parts[-1]))
    return any(candidate in _CROSS_AGENT_LIFECYCLE_ACTIONS for candidate in candidates)


def _is_cross_agent_targeted_control_action(name: str) -> bool:
    """Recognize operations that cross authority only with an agent target."""

    normalized = name.casefold()
    candidates = {normalized}
    if normalized.startswith("async_"):
        candidates.add(normalized.removeprefix("async_"))
    if normalized.startswith("a"):
        candidates.add(normalized[1:])
    for suffix in ("_async", "_now", "_sync"):
        if normalized.endswith(suffix):
            candidates.add(normalized.removesuffix(suffix))
    return bool(candidates.intersection(_CROSS_AGENT_TARGETED_CONTROL_ACTIONS))


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
            sources = {call_name} if semantic_name(call_name) else set()
            # A method call on a tracked registry may select one of its agent
            # objects. Preserve the receiver as may-flow instead of relying on
            # an ever-growing list of selector spellings such as ``get`` and
            # ``lookup``.
            if isinstance(node.func, ast.Attribute):
                sources.update(reference_sources(node.func.value))
            return sources
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
    resolved_inline = any(
        isinstance(child, ast.Call)
        and any(
            _call_name(child).casefold().startswith(f"{action}_{subject}")
            for action in ("find", "get", "lookup", "resolve", "select")
            for subject in ("agent", "child", "descendant", "peer")
        )
        for child in ast.walk(node.value)
    ) if isinstance(node, (ast.Attribute, ast.Subscript)) else False
    return isinstance(node, (ast.Attribute, ast.Subscript)) and (
        bool(_identifier_tokens(node.value).intersection(aliases))
        or resolved_inline
    )


def _is_cross_agent_state_object_reference(
    node: ast.AST,
    state_object_aliases: frozenset[str] | set[str] | None = None,
) -> bool:
    """Whether an expression resolves to a tracked agent-shaped object."""

    return bool(
        _identifier_tokens(node).intersection(state_object_aliases or set())
    ) or _is_cross_agent_state_mutation_target(node, state_object_aliases)


def _provenance_selected_cross_agent_read_lines(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    state_object_aliases: set[str],
    provenance_aliases: set[str],
    provenance_return_helpers: set[str],
) -> set[int]:
    """Locate reads through agent objects selected by provenance metadata."""

    def selects_agent_state(value: ast.AST) -> bool:
        if not _has_provenance_value(
            value, provenance_aliases, provenance_return_helpers
        ):
            return False
        for child in ast.walk(value):
            if isinstance(child, ast.Call):
                call_name = _call_name(child).casefold()
                if any(
                    call_name.startswith(f"{action}_{subject}")
                    for action in ("find", "get", "lookup", "resolve", "select")
                    for subject in ("agent", "child", "descendant", "peer")
                ):
                    return True
                if (
                    isinstance(child.func, ast.Attribute)
                    and child.func.attr.casefold() in {"get", "lookup", "resolve"}
                    and any(
                        _is_cross_agent_state_collection_name(token)
                        for token in _identifier_tokens(child.func.value)
                    )
                ):
                    return True
            if isinstance(child, ast.Subscript) and any(
                _is_cross_agent_state_collection_name(token)
                for token in _identifier_tokens(child.value)
            ):
                return True
        return False

    assignments: list[tuple[set[str], ast.AST]] = []
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
        }.intersection(state_object_aliases)
        if target_names:
            assignments.append((target_names, value))

    selected_aliases: set[str] = set()
    changed = True
    while changed:
        changed = False
        for target_names, value in assignments:
            if not (
                selects_agent_state(value)
                or _identifier_tokens(value).intersection(selected_aliases)
            ):
                continue
            new_aliases = target_names - selected_aliases
            if new_aliases:
                selected_aliases.update(new_aliases)
                changed = True

    lines: set[int] = set()
    for node in _walk_lexical_scope(function):
        if not isinstance(node, (ast.Attribute, ast.Subscript)):
            continue
        if _identifier_tokens(node.value).intersection(
            selected_aliases
        ) or selects_agent_state(node.value):
            lines.add(node.lineno)
    return lines


def _is_cross_agent_state_mutation_call(
    call: ast.Call,
    state_object_aliases: frozenset[str] | set[str] | None = None,
) -> bool:
    """Whether a mutator call writes through an agent registry or object."""

    item_mutation_actions = {"delitem", "setitem"}
    attribute_mutation = (
        isinstance(call.func, ast.Attribute)
        and call.func.attr.casefold().strip("_")
        in {
            *item_mutation_actions,
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
    call_name = _call_name(call).casefold()
    functional_item_receiver = (
        call.args[0]
        if call.args
        else next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg in {"container", "mapping", "obj", "object"}
            ),
            None,
        )
    )
    functional_item_form = call_name.strip("_") in item_mutation_actions and (
        not isinstance(call.func, ast.Attribute)
        or not _is_cross_agent_state_mutation_target(
            call.func.value, state_object_aliases
        )
    )
    functional_item_mutation = functional_item_form and (
        functional_item_receiver is not None
    ) and (
        _is_cross_agent_state_mutation_target(
            functional_item_receiver, state_object_aliases
        )
    )
    named_mutation_receiver = (
        call.args[0]
        if call.args
        else next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg in {"obj", "object"}
            ),
            None,
        )
    )
    named_mutation_attribute = (
        call.args[1]
        if len(call.args) > 1
        else next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg in {"attr", "name"}
            ),
            None,
        )
    )
    resolved_mutation_attribute = (
        _resolved_string(named_mutation_attribute)
        if named_mutation_attribute is not None
        else None
    )
    mutates_named_registry = (
        resolved_mutation_attribute is not None
        and _is_cross_agent_state_collection_name(
            resolved_mutation_attribute
        )
    )
    named_mutation = (
        call_name in {"__delattr__", "__setattr__", "delattr", "setattr"}
        and named_mutation_receiver is not None
        and (
            _is_cross_agent_state_object_reference(
                named_mutation_receiver, state_object_aliases
            )
            or mutates_named_registry
        )
    )
    bound_mutation_attribute = (
        call.args[0]
        if call.args
        else next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg in {"attr", "name"}
            ),
            None,
        )
    )
    resolved_bound_mutation_attribute = (
        _resolved_string(bound_mutation_attribute)
        if bound_mutation_attribute is not None
        else None
    )
    mutates_bound_named_registry = (
        resolved_bound_mutation_attribute is not None
        and _is_cross_agent_state_collection_name(
            resolved_bound_mutation_attribute
        )
    )
    bound_dunder_mutation = (
        isinstance(call.func, ast.Attribute)
        and call.func.attr.casefold() in {"__delattr__", "__setattr__"}
        and (
            _is_cross_agent_state_object_reference(
                call.func.value, state_object_aliases
            )
            or mutates_bound_named_registry
        )
    )
    lifecycle_target_arguments = [
        *call.args,
        *(keyword.value for keyword in call.keywords),
    ]
    lifecycle_mutation = _is_cross_agent_lifecycle_action(
        _call_name(call)
    ) and (
        (
            isinstance(call.func, ast.Attribute)
            and _is_cross_agent_state_object_reference(
                call.func.value, state_object_aliases
            )
        )
        or any(
            _is_cross_agent_state_object_reference(
                argument, state_object_aliases
            )
            for argument in lifecycle_target_arguments
        )
    )
    return (
        attribute_mutation
        or functional_item_mutation
        or named_mutation
        or bound_dunder_mutation
        or lifecycle_mutation
    )


def _is_cross_agent_state_mutation_node(
    node: ast.AST,
    state_object_aliases: frozenset[str] | set[str] | None = None,
) -> bool:
    """Apply the central registry/object mutation matchers to one AST node."""

    if isinstance(node, ast.Call):
        return _is_cross_agent_state_mutation_call(node, state_object_aliases)
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        targets = [node.target]
    elif isinstance(node, ast.Delete):
        targets = list(node.targets)
    else:
        return False
    return any(
        _is_cross_agent_state_mutation_target(target, state_object_aliases)
        for target in targets
    )


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


@lru_cache(maxsize=None)
def _lexical_statement_owners(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[ast.AST, ast.stmt]:
    """Map each same-scope expression to its nearest owning statement."""

    owners: dict[ast.AST, ast.stmt] = {}
    statements: list[ast.stmt] = []

    class OwnerVisitor(ast.NodeVisitor):
        def visit(self, node: ast.AST) -> None:
            is_statement = isinstance(node, ast.stmt)
            if is_statement:
                statements.append(node)
            if statements:
                owners[node] = statements[-1]
            super().visit(node)
            if is_statement:
                statements.pop()

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

    visitor = OwnerVisitor()
    for statement in function.body:
        visitor.visit(statement)
    return owners


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


def _block_guaranteed_function_exit(statements: list[ast.stmt]) -> bool:
    """Whether a block cannot reach code following its current loop.

    ``break`` and ``continue`` exit a local block, but unlike ``return`` and
    ``raise`` they may still reach controls after the loop.  Keeping this
    distinction prevents causation-dependent loop exits from hiding authority.
    """

    if not statements:
        return False
    terminal = statements[-1]
    if isinstance(terminal, (ast.Raise, ast.Return)):
        return True
    if isinstance(terminal, (ast.Break, ast.Continue)):
        return False
    if isinstance(terminal, ast.If):
        return _block_guaranteed_function_exit(
            terminal.body
        ) and _block_guaranteed_function_exit(terminal.orelse)
    if isinstance(terminal, (ast.With, ast.AsyncWith)):
        return _block_guaranteed_function_exit(terminal.body)
    if isinstance(terminal, (ast.Try, ast.TryStar)):
        if _block_guaranteed_function_exit(terminal.finalbody):
            return True
        normal_path_exits = _block_guaranteed_function_exit(
            terminal.body
        ) or _block_guaranteed_function_exit(terminal.orelse)
        return normal_path_exits and all(
            _block_guaranteed_function_exit(handler.body)
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
            _block_guaranteed_function_exit(case.body)
            for case in terminal.cases
        )
    return False


def _contains_current_loop_transfer(
    statements: list[ast.stmt],
    *,
    include_break: bool = True,
    include_continue: bool = False,
) -> bool:
    """Whether a block can transfer its owning loop, excluding nested loops."""

    class LoopTransferVisitor(ast.NodeVisitor):
        found = False

        def visit_Break(self, node: ast.Break) -> None:  # noqa: N802
            if include_break:
                self.found = True

        def visit_Continue(self, node: ast.Continue) -> None:  # noqa: N802
            if include_continue:
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

    visitor = LoopTransferVisitor()
    for statement in statements:
        visitor.visit(statement)
    return visitor.found


def _contains_current_loop_break(statements: list[ast.stmt]) -> bool:
    """Whether a block can break its owning loop, excluding nested loops."""

    return _contains_current_loop_transfer(statements)


def _block_then_suffix_guaranteed_function_exit(
    block: list[ast.stmt],
    suffix: list[ast.stmt],
) -> bool:
    """Whether one loop branch must exit the function before post-loop code."""

    if _contains_current_loop_transfer(
        block,
        include_continue=True,
    ):
        return False
    if _block_guaranteed_exits(block):
        return _block_guaranteed_function_exit(block)
    return _block_guaranteed_function_exit([*block, *suffix])


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


def _is_trace_parent_token(token: str) -> bool:
    """Recognize common W3C and Python spellings of trace-parent metadata."""

    words = set(token.casefold().replace("-", "_").replace(".", "_").split("_"))
    return (
        bool(words.intersection({"parentspan", "traceparent"}))
        or {"trace", "parent"} <= words
        or {"span", "parent"} <= words
    )


def _source_text_may_expose_provenance(source: str) -> bool:
    """Apply the same trace-parent grammar to the cheap source prefilter."""

    source = source.casefold()
    return (
        "metadata" in source
        and any(key in source for key in UNVERIFIED_ATTRIBUTION_METADATA_KEYS)
    ) or any(
        marker in source for marker in PROVENANCE_SOURCE_MARKERS
    ) or any(
        _is_trace_parent_token(token)
        for token in SOURCE_IDENTIFIER_CHAIN.findall(source)
    )


def _is_unverified_attribution_metadata_lookup(node: ast.AST) -> bool:
    """Recognize attribution that has not been authorized for the target.

    Signature verification authenticates a principal but does not grant that
    principal relation authority.  Both a raw metadata claim and an
    authenticated-yet-unauthorized verdict field therefore remain provenance
    when code tries to use them as an authority decision.
    """

    descendants = tuple(ast.walk(node))
    has_explicit_unverified = any(
        getattr(child, "_authority_unverified_attribution", False)
        for child in descendants
    )
    has_verified = any(
        getattr(child, "_authority_verified_attribution", False)
        for child in descendants
    )
    has_authenticated = any(
        getattr(child, "_authority_authenticated_attribution", False)
        for child in descendants
    )
    has_witness = any(
        getattr(child, "_authority_identity_witness", False)
        for child in descendants
    )
    if (
        isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.And)
        and (has_verified or has_authenticated)
        and has_witness
        and not has_explicit_unverified
    ):
        return False
    if (
        isinstance(node, ast.Call)
        and _call_name(node).casefold() == "_a2a_sender_witness_unchanged"
        and has_witness
        and not has_explicit_unverified
    ):
        return False
    if (
        has_authenticated
        and has_witness
        and not has_explicit_unverified
        and any(
            isinstance(child, ast.Call)
            and _call_name(child).casefold()
            == "_a2a_sender_witness_unchanged"
            for child in descendants
        )
    ):
        return False

    def visit(current: ast.AST) -> bool:
        if (
            isinstance(current, ast.Call)
            and _call_name(current).casefold()
            == "_a2a_sender_witness_unchanged"
            and any(
                getattr(child, "_authority_identity_witness", False)
                for child in ast.walk(current)
            )
            and not any(
                getattr(child, "_authority_unverified_attribution", False)
                for child in ast.walk(current)
            )
        ):
            return False
        if getattr(current, "_authority_unverified_attribution", False):
            return True
        # Authentication is tracked separately below when it directly selects
        # a protected control. Shield the verification-result receiver here so
        # ordinary authorization plumbing is not reclassified as a raw claim.
        if getattr(current, "_authority_authenticated_attribution", False):
            return False
        # A value returned by a target authorization seam shields the receiver
        # marker beneath it. Walk recursively rather than through ``ast.walk``
        # so descendants of this authorized value are not re-tainted.
        if getattr(current, "_authority_verified_attribution", False):
            return False
        if getattr(current, "_authority_identity_witness", False):
            return True
        if getattr(current, "_authority_verification_result", False):
            return True
        receiver: ast.AST | None = None
        key: str | None = None
        if isinstance(current, ast.Subscript):
            receiver = current.value
            key = _resolved_string(current.slice)
        elif (
            isinstance(current, ast.Call)
            and isinstance(current.func, ast.Attribute)
            and current.func.attr.casefold() in {"get", "pop", "setdefault"}
            and current.args
        ):
            receiver = current.func.value
            key = _resolved_string(current.args[0])
        elif isinstance(current, ast.Attribute):
            receiver = current.value
            key = current.attr
        if (
            receiver is not None
            and key is not None
            and any(
                token == "metadata" or token.endswith(".metadata")
                for token in _identifier_tokens(receiver)
            )
            and key.casefold() in UNVERIFIED_ATTRIBUTION_METADATA_KEYS
        ):
            return True
        return any(visit(child) for child in ast.iter_child_nodes(current))

    return visit(node)


_ATTRIBUTION_UNTRUSTED_MAPPING = ("identity", "untrusted-mapping")
_ATTRIBUTION_UNTRUSTED_CLAIM = ("identity", "untrusted-claim")
_ATTRIBUTION_VERIFIED_MAPPING = ("identity", "verified-mapping")
_ATTRIBUTION_VERIFIED_VALUE = ("identity", "verified-value")
_ATTRIBUTION_AUTHENTICATED_VALUE = ("identity", "authenticated-value")
_ATTRIBUTION_VERIFICATION_RESULT = ("identity", "verification-result")
_ATTRIBUTION_ENVELOPE_VERIFIER = ("identity", "envelope-verifier")
_ATTRIBUTION_IDENTITY_WITNESS = ("identity", "identity-witness")
_ATTRIBUTION_WITNESS_FACTORY = ("identity", "witness-factory")
_ATTRIBUTION_VALIDATOR = ("identity", "validator")
_ATTRIBUTION_UNKNOWN = ("identity", "unknown")

_TRUSTED_ATTRIBUTION_VALIDATOR_IMPORTS = {
    "kestrel_sovereign.a2a.envelope_signing": {"verify_inbound_envelope"},
}
_TRUSTED_ATTRIBUTION_VALIDATOR_MEMBERS = {
    "authorize_a2a_legacy_unsigned_sender",
}
_TRUSTED_ATTRIBUTION_WITNESS_MEMBERS = {
    "a2a_sender_identity_witness",
}


def _trusted_attribution_validator_aliases(tree: ast.AST) -> set[str]:
    """Return bindings proven to import the canonical envelope verifier."""

    aliases: set[str] = set()
    nodes: list[ast.AST] = (
        list(tree.body)
        if isinstance(tree, ast.Module)
        else _walk_lexical_scope(tree)
    )
    for node in nodes:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        trusted = _TRUSTED_ATTRIBUTION_VALIDATOR_IMPORTS.get(node.module, set())
        aliases.update(
            (imported.asname or imported.name).casefold()
            for imported in node.names
            if imported.name in trusted
        )
    return aliases


def _attribution_projection_helpers(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> set[str]:
    """Return local helpers that extract an identity field from a parameter."""

    helpers: set[str] = set()
    for function in functions:
        aliases = {
            parameter.arg.casefold()
            for parameter in [
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            ]
        }
        strings: dict[str, str] = {}
        changed = True
        while changed:
            changed = False
            for node in _walk_lexical_scope(function):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                value = node.value
                if value is None:
                    continue
                resolved = _resolved_string(value, strings)
                for target in targets:
                    for name in _assignment_target_names(target):
                        normalized = name.casefold()
                        if resolved is not None:
                            current = strings.get(normalized)
                            if current is None:
                                strings[normalized] = resolved
                                changed = True
                            elif current not in {
                                resolved,
                                "<ambiguous>",
                            }:
                                strings[normalized] = "<ambiguous>"
                                changed = True
                        if (
                            _identifier_tokens(value).intersection(aliases)
                            and normalized not in aliases
                        ):
                            aliases.add(normalized)
                            changed = True

        for returned in (
            node.value
            for node in _walk_lexical_scope(function)
            if isinstance(node, ast.Return) and node.value is not None
        ):
            for child in ast.walk(returned):
                receiver: ast.AST | None = None
                key: str | None = None
                if isinstance(child, ast.Subscript):
                    receiver = child.value
                    key = _resolved_string(child.slice, strings)
                elif (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr.casefold() in {"get", "pop", "setdefault"}
                    and child.args
                ):
                    receiver = child.func.value
                    key = _resolved_string(child.args[0], strings)
                if (
                    receiver is not None
                    and key is not None
                    and key.casefold() in UNVERIFIED_ATTRIBUTION_METADATA_KEYS
                    and _identifier_tokens(receiver).intersection(aliases)
                ):
                    helpers.add(function.name.casefold())
                    break
            if function.name.casefold() in helpers:
                break
    return helpers


def _local_attribution_authorization_helpers(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> set[str]:
    """Find local seams that combine envelope and legacy sender authorization."""

    helpers: set[str] = set()
    for function in functions:
        nodes = _walk_lexical_scope(function)
        calls = {
            _call_name(node).casefold()
            for node in nodes
            if isinstance(node, ast.Call)
        }
        constants = {
            node.value.casefold()
            for node in nodes
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }
        has_verified_branch = any(
            isinstance(node, ast.Attribute)
            and node.attr.casefold() == "verified"
            for node in nodes
        )
        if (
            "verify_inbound_envelope" in calls
            and "authorize_legacy" in calls
            and "authorize_a2a_legacy_unsigned_sender" in constants
            and has_verified_branch
        ):
            helpers.add(function.name.casefold())
    return helpers


def _unverified_attribution_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    trusted_validator_aliases: set[str] | None = None,
    string_constants: dict[str, str] | None = None,
    parameter_return_flows: dict[str, _ParameterReturnFlow] | None = None,
    projection_helpers: set[str] | None = None,
) -> set[str]:
    """Track untrusted and verified identity in the shared binding lattice."""

    # Parsed repository modules are cached and may be visited first as an
    # imported helper, then as their own audit target. Identity annotations are
    # analysis results, not permanent AST facts; discard the prior pass before
    # replaying this function so collection order cannot change the verdict.
    for node in _walk_lexical_scope(function):
        for attribute in (
            "_authority_attribution_validator",
            "_authority_static_string",
            "_authority_unverified_attribution",
            "_authority_authenticated_attribution",
            "_authority_identity_witness",
            "_authority_verification_result",
            "_authority_verified_attribution",
        ):
            if hasattr(node, attribute):
                delattr(node, attribute)

    static_strings = {
        name: ("static-string", value)
        for name, value in (string_constants or {}).items()
    }
    positional_parameters = [
        *function.args.posonlyargs,
        *function.args.args,
    ]
    for parameter, default in [
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
    ]:
        resolved = _resolved_string(default, string_constants or {})
        if resolved is not None:
            static_strings[parameter.arg] = ("static-string", resolved)

    def static_string_direct(value: ast.AST) -> _StaticBinding | None:
        resolved = _resolved_string(value)
        return ("static-string", resolved) if resolved is not None else None

    def annotate_static_string(
        value: ast.AST, binding: _StaticBinding
    ) -> None:
        if binding[0] == "static-string" and binding[1] != "<ambiguous>":
            setattr(value, "_authority_static_string", binding[1])

    string_flow = _StaticBindingFlow(
        static_string_direct,
        lambda bindings: (
            bindings[0]
            if bindings and all(binding == bindings[0] for binding in bindings[1:])
            else ("static-string", "<ambiguous>")
        ),
        static_strings,
        normalize_name=str.casefold,
        on_resolve=annotate_static_string,
    )
    string_flow.replay(function.body)

    def resolved_key(value: ast.AST) -> str | None:
        return _resolved_string(value) or getattr(
            value, "_authority_static_string", None
        )

    untrusted = {
        _ATTRIBUTION_UNTRUSTED_MAPPING,
        _ATTRIBUTION_UNTRUSTED_CLAIM,
        _ATTRIBUTION_AUTHENTICATED_VALUE,
    }
    verified = {
        _ATTRIBUTION_VERIFIED_MAPPING,
        _ATTRIBUTION_VERIFIED_VALUE,
    }
    initial: dict[str, _StaticBinding] = dict.fromkeys(
        trusted_validator_aliases or set(), _ATTRIBUTION_ENVELOPE_VERIFIER
    )
    initial.update({
        parameter.arg: _ATTRIBUTION_UNTRUSTED_MAPPING
        for parameter in [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
        if parameter.arg.casefold() == "metadata"
    })
    def lookup_binding(
        flow: _StaticBindingFlow, value: ast.AST
    ) -> _StaticBinding | None:
        if isinstance(value, ast.Subscript):
            receiver = value.value
            if flow.resolve(receiver) == _ATTRIBUTION_IDENTITY_WITNESS:
                return _ATTRIBUTION_AUTHENTICATED_VALUE
            field = resolved_key(value.slice)
            if field is None:
                return None
        else:
            member = _static_member_reference(value)
            if member is None:
                return None
            receiver, field = member
        field = field.casefold()
        if field == "metadata":
            return _ATTRIBUTION_UNTRUSTED_MAPPING
        receiver_binding = flow.resolve(receiver)
        if receiver_binding == _ATTRIBUTION_IDENTITY_WITNESS:
            return _ATTRIBUTION_AUTHENTICATED_VALUE
        if receiver_binding == _ATTRIBUTION_VERIFICATION_RESULT:
            if field in {
                "nonce",
                "sender",
                "verification_document_fingerprint",
                "verified",
            }:
                # Signature verification authenticates an identity. It does
                # not authorize that identity to control the selected target.
                return _ATTRIBUTION_AUTHENTICATED_VALUE
            return _ATTRIBUTION_UNTRUSTED_CLAIM
        if field in UNVERIFIED_ATTRIBUTION_METADATA_KEYS:
            if receiver_binding in untrusted:
                return _ATTRIBUTION_UNTRUSTED_CLAIM
            if receiver_binding in verified:
                return _ATTRIBUTION_VERIFIED_VALUE
        if (
            field in _TRUSTED_ATTRIBUTION_VALIDATOR_MEMBERS
            and any(
                token in {"manager", "agent_manager"}
                or token.endswith(".manager")
                for token in _identifier_tokens(receiver)
            )
        ):
            return _ATTRIBUTION_VALIDATOR
        if (
            field in _TRUSTED_ATTRIBUTION_WITNESS_MEMBERS
            and any(
                token in {"manager", "agent_manager"}
                or token.endswith(".manager")
                for token in _identifier_tokens(receiver)
            )
        ):
            return _ATTRIBUTION_WITNESS_FACTORY
        return None

    def direct(
        flow: _StaticBindingFlow, value: ast.AST
    ) -> _StaticBinding | None:
        while isinstance(value, (ast.Await, ast.Expr)):
            value = value.value
        member_binding = lookup_binding(flow, value)
        if member_binding is not None:
            return member_binding
        if isinstance(value, ast.Call):
            callable_binding = flow.resolve(value.func)
            if callable_binding == _ATTRIBUTION_ENVELOPE_VERIFIER:
                return _ATTRIBUTION_VERIFICATION_RESULT
            if callable_binding == _ATTRIBUTION_WITNESS_FACTORY:
                return _ATTRIBUTION_IDENTITY_WITNESS
            if callable_binding == _ATTRIBUTION_VALIDATOR:
                return _ATTRIBUTION_VERIFIED_VALUE
            if (
                _call_name(value).casefold()
                == "_a2a_sender_witness_unchanged"
                and any(
                    flow.resolve(argument) == _ATTRIBUTION_IDENTITY_WITNESS
                    for argument in value.args
                )
            ):
                return _ATTRIBUTION_AUTHENTICATED_VALUE
            parameter_flow = (parameter_return_flows or {}).get(
                _call_name(value).casefold()
            )
            if parameter_flow is not None:
                returned = [
                    flow.resolve(argument)
                    for argument in _bound_parameter_flow_arguments(
                        value, parameter_flow
                    )
                ]
                if _ATTRIBUTION_UNTRUSTED_CLAIM in returned:
                    return _ATTRIBUTION_UNTRUSTED_CLAIM
                if _ATTRIBUTION_UNTRUSTED_MAPPING in returned:
                    if _call_name(value).casefold() in (
                        projection_helpers or set()
                    ):
                        return _ATTRIBUTION_UNTRUSTED_CLAIM
                    return _ATTRIBUTION_UNTRUSTED_MAPPING
                if _ATTRIBUTION_AUTHENTICATED_VALUE in returned:
                    return _ATTRIBUTION_AUTHENTICATED_VALUE
                if returned and all(binding in verified for binding in returned):
                    return _ATTRIBUTION_VERIFIED_VALUE
            if isinstance(value.func, ast.Attribute):
                receiver_binding = flow.resolve(value.func.value)
                if (
                    value.func.attr.casefold() in {"get", "pop", "setdefault"}
                    and value.args
                    and (key := resolved_key(value.args[0])) is not None
                    and key.casefold() in UNVERIFIED_ATTRIBUTION_METADATA_KEYS
                ):
                    if receiver_binding in untrusted:
                        return _ATTRIBUTION_UNTRUSTED_CLAIM
                    if receiver_binding in verified:
                        return _ATTRIBUTION_VERIFIED_VALUE
                if value.func.attr.casefold() == "copy":
                    if receiver_binding == _ATTRIBUTION_UNTRUSTED_MAPPING:
                        return receiver_binding
                    if receiver_binding == _ATTRIBUTION_VERIFIED_MAPPING:
                        return receiver_binding
            supplied = [
                flow.resolve(argument)
                for argument in [
                    *value.args,
                    *(keyword.value for keyword in value.keywords),
                ]
            ]
            # An opaque consumer of the entire metadata mapping does not, by
            # itself, turn its result into a claimed sender identity.  Only a
            # value already extracted as an identity claim propagates through
            # an unknown call; parameter-return flow above handles proven
            # passthrough helpers.
            if _ATTRIBUTION_UNTRUSTED_CLAIM in supplied:
                return _ATTRIBUTION_UNTRUSTED_CLAIM
            if (
                _is_provenance_transform_call(value)
                and any(
                    binding in verified
                    or binding == _ATTRIBUTION_AUTHENTICATED_VALUE
                    for binding in supplied
                )
            ):
                return (
                    _ATTRIBUTION_AUTHENTICATED_VALUE
                    if _ATTRIBUTION_AUTHENTICATED_VALUE in supplied
                    else _ATTRIBUTION_VERIFIED_VALUE
                )
        if isinstance(value, ast.Dict):
            identity_entries = [
                flow.resolve(item)
                for key, item in zip(value.keys, value.values)
                if key is not None
                and (resolved := resolved_key(key)) is not None
                and resolved.casefold() in UNVERIFIED_ATTRIBUTION_METADATA_KEYS
            ]
            if any(binding in untrusted for binding in identity_entries):
                return _ATTRIBUTION_UNTRUSTED_MAPPING
            if identity_entries and all(
                binding in verified for binding in identity_entries
            ):
                return _ATTRIBUTION_VERIFIED_MAPPING
        if isinstance(
            value,
            (
                ast.BinOp,
                ast.BoolOp,
                ast.Compare,
                ast.List,
                ast.Set,
                ast.Tuple,
                ast.UnaryOp,
            ),
        ):
            nested = [flow.resolve(child) for child in ast.iter_child_nodes(value)]
            if _ATTRIBUTION_UNTRUSTED_CLAIM in nested:
                return _ATTRIBUTION_UNTRUSTED_CLAIM
            if _ATTRIBUTION_AUTHENTICATED_VALUE in nested:
                return _ATTRIBUTION_AUTHENTICATED_VALUE
            if (
                isinstance(value, ast.BoolOp)
                and isinstance(value.op, ast.And)
                and _ATTRIBUTION_VERIFIED_VALUE in nested
                and all(
                    binding in verified
                    or binding == _ATTRIBUTION_IDENTITY_WITNESS
                    for binding in nested
                )
            ):
                return _ATTRIBUTION_VERIFIED_VALUE
            if (
                isinstance(value, ast.Compare)
                and _ATTRIBUTION_IDENTITY_WITNESS in nested
            ):
                return _ATTRIBUTION_IDENTITY_WITNESS
            if nested and all(binding in verified for binding in nested):
                return _ATTRIBUTION_VERIFIED_VALUE
        return None

    def ambiguous(bindings: list[_StaticBinding]) -> _StaticBinding:
        if _ATTRIBUTION_UNTRUSTED_CLAIM in bindings:
            return _ATTRIBUTION_UNTRUSTED_CLAIM
        # Preserve the distinction between an entire caller-controlled
        # metadata container and an extracted identity claim across branches.
        # Promoting the container to a claim taints every unrelated field and
        # every helper that consumes metadata.
        if _ATTRIBUTION_UNTRUSTED_MAPPING in bindings:
            return _ATTRIBUTION_UNTRUSTED_MAPPING
        # Authentication cannot be promoted to target authorization merely by
        # merging it with a stronger value on another control-flow path.
        if _ATTRIBUTION_AUTHENTICATED_VALUE in bindings:
            return _ATTRIBUTION_AUTHENTICATED_VALUE
        if bindings and all(binding == bindings[0] for binding in bindings[1:]):
            return bindings[0]
        # Mixed or partially-bound validators are not proof of identity.
        return _ATTRIBUTION_UNKNOWN

    def annotate(value: ast.AST, binding: _StaticBinding) -> None:
        if binding == _ATTRIBUTION_UNTRUSTED_CLAIM:
            setattr(value, "_authority_unverified_attribution", True)
        elif binding == _ATTRIBUTION_AUTHENTICATED_VALUE:
            setattr(value, "_authority_authenticated_attribution", True)
        elif binding == _ATTRIBUTION_VERIFIED_VALUE:
            setattr(value, "_authority_verified_attribution", True)
        elif binding == _ATTRIBUTION_VERIFICATION_RESULT:
            setattr(value, "_authority_verification_result", True)
        elif binding == _ATTRIBUTION_IDENTITY_WITNESS:
            setattr(value, "_authority_identity_witness", True)
        elif binding in {
            _ATTRIBUTION_ENVELOPE_VERIFIER,
            _ATTRIBUTION_WITNESS_FACTORY,
            _ATTRIBUTION_VALIDATOR,
        }:
            setattr(value, "_authority_attribution_validator", True)

    identity_flow = _StaticBindingFlow(
        direct,
        ambiguous,
        initial,
        normalize_name=str.casefold,
        on_resolve=annotate,
        stateful_resolver=True,
    )
    identity_flow.replay(function.body)
    return {
        name
        for name, binding in identity_flow.bindings.items()
        if binding == _ATTRIBUTION_UNTRUSTED_CLAIM
    }


def _is_attribution_validation_call(
    call: ast.Call,
    attribution_aliases: set[str] | None = None,
) -> bool:
    """Return only calls resolved to an audited identity-verification seam."""

    return bool(
        _call_name(call).casefold() in (attribution_aliases or set())
        or getattr(call, "_authority_local_attribution_authorizer", False)
        or getattr(call, "_authority_verification_result", False)
        or getattr(call, "_authority_verified_attribution", False)
        or getattr(call, "_authority_identity_witness", False)
    )


def _has_provenance_token(
    node: ast.AST,
    aliases: set[str] | None = None,
    *,
    include_unverified_attribution: bool = True,
) -> bool:
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
    invoked_accessor_names = {
        ast.unparse(child.func).casefold()
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and _is_provenance_accessor_reference(child.func)
    }
    referenced_accessor_names = {
        ast.unparse(child).casefold()
        for child in ast.walk(node)
        if isinstance(child, (ast.Name, ast.Attribute))
        and "provide" in ast.unparse(child).casefold().split("_")
        and _is_provenance_accessor_reference(child)
    }
    for accessor_name in referenced_accessor_names - invoked_accessor_names:
        tokens.discard(accessor_name)
        tokens.discard(accessor_name.rsplit(".", maxsplit=1)[-1])
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
    return (
        include_unverified_attribution
        and _is_unverified_attribution_metadata_lookup(node)
    ) or any(
        token in provenance_tokens
        or _is_trace_parent_token(token)
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
        and (
            _call_name(child).casefold().strip("_").endswith(
                PROVENANCE_ACCESSOR_SUFFIXES
            )
            or _is_provenance_accessor_getattr_reference(child.func)
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
    aliases.update(
        parameter.arg.casefold()
        for parameter, default in default_bindings
        if _is_provenance_accessor_reference(default)
        or _is_provenance_accessor_getattr_reference(default)
    )
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
    *,
    include_unverified_attribution: bool = True,
) -> bool:
    """Whether an expression reads provenance directly or via a local helper."""

    return (
        _has_provenance_token(
            node,
            aliases,
            include_unverified_attribution=include_unverified_attribution,
        )
        or _is_provenance_accessor_call(node)
        or any(
            isinstance(child, ast.Call)
            and (
                _call_name(child).casefold()
                in (provenance_return_helpers or set())
                or _CALLABLE_PROVENANCE in _callable_semantics(child)
            )
            for child in ast.walk(node)
        )
        or any(
            isinstance(child, ast.Attribute)
            and child.attr.casefold()
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
    inherited_bindings: set[str] | None = None,
) -> dict[ast.AST, frozenset[str]]:
    """Capture live aliases, including shared bindings from outer scopes."""

    groups: dict[str, set[str]] = {}
    snapshots: dict[ast.AST, frozenset[str]] = {}

    for binding in inherited_bindings or set():
        root = _state_alias_root(binding)
        group = groups.setdefault(root, {root})
        group.add(binding)
        for member in group:
            groups[member] = group

    def detach(name: str) -> None:
        group = groups.pop(name, None)
        if group is not None:
            group.discard(name)

    def bind(target_names: set[str], source_names: set[str] | None) -> None:
        for target_name in target_names:
            detach(target_name)
        if not target_names:
            return
        source_groups = [
            groups[source_name]
            for source_name in source_names or set()
            if source_name in groups
        ]
        group = set(source_names or set())
        for source_group in source_groups:
            group.update(source_group)
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
            # Python reference assignment aliases a mutable object only after
            # this analysis has evidence that the source is a tracked
            # container. Outer shared bindings are seeded above, so module,
            # class, and closure aliases participate without treating every
            # ordinary object assignment as a container alias.
            aliases_tracked_container = any(
                source_name in groups for source_name in source_names
            )
            if creates_container or aliases_tracked_container:
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
    control_return_helpers: set[str] | None = None,
    parameter_return_flows: dict[str, _ParameterReturnFlow] | None = None,
) -> set[str]:
    """Resolve local names that reference cross-agent control callables."""

    assignments: list[tuple[str, str]] = []
    container_aliases = _mutable_container_alias_snapshots(function)
    callable_alias_edges = _scope_callable_alias_edges(function)

    def reference_sources(value: ast.AST) -> set[str]:
        sources = set(
            _control_reference_sources(value, control_return_helpers)
        )
        while isinstance(value, (ast.Await, ast.Expr)):
            value = value.value
        if not isinstance(value, ast.Call):
            return sources
        for source_name in _expanded_callable_sources(
            _call_name(value),
            callable_alias_edges,
        ):
            flow = (parameter_return_flows or {}).get(source_name)
            if flow is None:
                continue
            for argument in _bound_parameter_flow_arguments(value, flow):
                sources.update(reference_sources(argument))
        return sources

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
            for source in reference_sources(default)
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
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            # Iteration is assignment too: a callable stored in the iterable
            # remains a control callable after it is bound to the loop target.
            targets = [node.target]
            value = node.iter
        elif isinstance(node, ast.Call):
            mutation = _mutable_container_write(node)
            if mutation is not None:
                target_names, value = mutation
                target_names = expand_container_aliases(target_names, node)
                sources = reference_sources(value)
                assignments.extend(
                    (target_name, source)
                    for target_name in target_names
                    for source in sources
                )
            continue
        if value is None:
            continue
        sources = reference_sources(value)
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

    aliases: set[str] = set(control_helpers or ()) | set(
        control_return_helpers or ()
    )
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


def _local_control_return_helpers(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    control_helpers: set[str] | None = None,
    function_control_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] | None = None,
) -> set[str]:
    """Find local helpers whose returned value is a control callable."""

    return_helpers: set[str] = set()
    changed = True
    while changed:
        changed = False
        for function in functions:
            function_name = function.name.casefold()
            if function_name in return_helpers:
                continue
            visible_controls = set(control_helpers or ()) | set(
                (function_control_aliases or {}).get(function, ())
            )
            aliases = _cross_agent_control_aliases(
                function,
                visible_controls,
                return_helpers,
            )
            returned_values = [
                node.value
                for node in _walk_lexical_scope(function)
                if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom))
                and node.value is not None
            ]
            if any(
                any(
                    _is_unambiguous_control_token(source)
                    or source in aliases
                    for source in _control_reference_sources(
                        value, return_helpers
                    )
                )
                for value in returned_values
            ):
                return_helpers.add(function_name)
                changed = True
    return return_helpers


def _invoked_lambda_bodies(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.Lambda, ...]:
    """Return lambdas invoked locally or allowed to escape this scope."""

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
    escaping_names: set[str] = set()
    for node in scope_nodes:
        escaping_values: list[ast.AST] = []
        if isinstance(node, ast.Return) and node.value is not None:
            escaping_values.append(node.value)
        elif isinstance(node, ast.Call):
            escaping_values.extend(node.args)
            escaping_values.extend(keyword.value for keyword in node.keywords)
        for value in escaping_values:
            invoked.extend(
                child for child in ast.walk(value) if isinstance(child, ast.Lambda)
            )
            escaping_names.update(_reference_binding_names(value))
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
    invoked_names.update(escaping_names)
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


def _control_reference_sources(
    node: ast.AST,
    control_return_helpers: set[str] | None = None,
) -> set[str]:
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
            else _control_reference_sources(
                receiver, control_return_helpers
            )
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
            for source in _control_reference_sources(
                element, control_return_helpers
            )
        }
    if isinstance(
        node,
        (ast.DictComp, ast.GeneratorExp, ast.ListComp, ast.SetComp),
    ):
        elements = (
            [node.key, node.value]
            if isinstance(node, ast.DictComp)
            else [node.elt]
        )
        elements.extend(
            expression
            for generator in node.generators
            for expression in [generator.iter, *generator.ifs]
        )
        return {
            source
            for element in elements
            for source in _control_reference_sources(
                element, control_return_helpers
            )
        }
    if isinstance(node, ast.IfExp):
        return _control_reference_sources(
            node.body, control_return_helpers
        ) | _control_reference_sources(
            node.orelse, control_return_helpers
        )
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
    if factory_name in (control_return_helpers or set()):
        return {factory_name}
    if factory_name in {"getattr", "__getattribute__", "methodcaller"}:
        if factory_name == "getattr":
            attribute = node.args[1] if len(node.args) > 1 else None
        elif factory_name == "__getattribute__":
            # Bound ``peer.__getattribute__('stop')`` has one argument while
            # unbound ``object.__getattribute__(peer, 'stop')`` has two.
            attribute = (
                node.args[1]
                if len(node.args) > 1
                else node.args[0]
                if node.args
                else None
            )
        else:
            attribute = node.args[0] if node.args else None
        if attribute is None:
            attribute = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in {"name", "attr"}
                ),
                None,
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
        sources = _control_reference_sources(
            node.func.value, control_return_helpers
        )
        if node.args:
            key = _resolved_string(node.args[0])
            if key is not None:
                sources.add(key.casefold())
        for default in [
            *node.args[1:],
            *(keyword.value for keyword in node.keywords),
        ]:
            sources.update(
                _control_reference_sources(default, control_return_helpers)
            )
        return sources
    if factory_name in {"partial", "partialmethod"} and node.args:
        return _control_reference_sources(node.args[0], control_return_helpers)
    # Collection/iterator adapters preserve or select their input members.
    # Follow the input references so a control callback does not become opaque
    # merely because ordinary iteration adds an index, view, or lazy wrapper.
    iterable_adapter_names = {
        "chain",
        "deque",
        "enumerate",
        "filter",
        "frozenset",
        "iter",
        "list",
        "map",
        "next",
        "reversed",
        "set",
        "sorted",
        "tuple",
        "zip",
    }
    collection_view_names = {"items", "keys", "values"}
    if factory_name in iterable_adapter_names:
        return {
            source
            for argument in [
                *node.args,
                *(keyword.value for keyword in node.keywords),
            ]
            for source in _control_reference_sources(
                argument, control_return_helpers
            )
        }
    if (
        factory_name in collection_view_names
        and isinstance(node.func, ast.Attribute)
    ):
        return _control_reference_sources(
            node.func.value, control_return_helpers
        )
    # Type adapters and decorator helpers preserve a callable argument in
    # their return value.  Do not generalize this to every call: an ordinary
    # consumer may accept a callback without returning it, and treating that
    # result as callable creates fleet-wide false positives.
    wrapper_names = {
        "cache",
        "cast",
        "identity",
        "lru_cache",
        "singledispatch",
        "singledispatchmethod",
        "update_wrapper",
        "wraps",
    }
    nested_factory_name = (
        _call_name(node.func).casefold()
        if isinstance(node.func, ast.Call)
        else ""
    )
    if factory_name in wrapper_names or nested_factory_name in wrapper_names:
        return {
            source
            for argument in [
                *(
                    [*node.func.args, *node.args]
                    if isinstance(node.func, ast.Call)
                    else node.args
                ),
                *(
                    keyword.value
                    for call in (
                        [node.func, node]
                        if isinstance(node.func, ast.Call)
                        else [node]
                    )
                    for keyword in call.keywords
                ),
            ]
            for source in _control_reference_sources(
                argument, control_return_helpers
            )
        }
    return set()


def _is_cross_agent_control_call(
    node: ast.Call,
    control_aliases: set[str] | None = None,
    state_object_aliases: frozenset[str] | set[str] | None = None,
) -> bool:
    """Whether ``node`` invokes a known control or a local alias of one."""

    if _is_attribution_validation_call(node):
        return False
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
        "add_done_callback": (0,),
        "apply_async": (0,),
        "call_at": (1,),
        "call_later": (1,),
        "call_soon": (0,),
        "call_soon_threadsafe": (0,),
        "filter": (0,),
        "map": (0,),
        "process": (1,),
        "reduce": (0,),
        "register": (0,),
        "run_in_executor": (1,),
        "starmap": (0,),
        "starmap_async": (0,),
        "submit": (0,),
        "thread": (1,),
        "timer": (1,),
        "to_thread": (0,),
    }
    higher_order_callable_keywords = {
        "add_done_callback": {"fn"},
        "apply_async": {"callback", "error_callback", "func"},
        "call_at": {"callback"},
        "call_later": {"callback"},
        "call_soon": {"callback"},
        "call_soon_threadsafe": {"callback"},
        "filter": {"function"},
        "map": {"function"},
        "process": {"target"},
        "reduce": {"function"},
        "register": {"func"},
        "run_in_executor": {"func"},
        "starmap": {"func"},
        "starmap_async": {"func"},
        "submit": {"fn"},
        "thread": {"target"},
        "timer": {"function"},
        "to_thread": {"func"},
    }
    higher_order_sources = {
        source
        for index in higher_order_callable_indexes.get(call_name, ())
        if index < len(node.args)
        for source in _control_reference_sources(node.args[index])
    }
    higher_order_sources.update(
        source
        for keyword in node.keywords
        if keyword.arg in higher_order_callable_keywords.get(call_name, set())
        for source in _control_reference_sources(keyword.value)
    )
    kills_agent_process = (
        call_name == "kill"
        and bool(node.args)
        and _is_cross_agent_state_object_reference(
            node.args[0], state_object_aliases
        )
    )
    targeted_agent_control = _is_cross_agent_targeted_control_action(
        call_name
    ) and (
        (
            isinstance(node.func, ast.Attribute)
            and _is_cross_agent_state_object_reference(
                node.func.value, state_object_aliases
            )
        )
        or any(
            _is_cross_agent_state_object_reference(
                argument, state_object_aliases
            )
            for argument in [
                *node.args,
                *(keyword.value for keyword in node.keywords),
            ]
        )
    )
    return (
        _CALLABLE_CONTROL in _callable_semantics(node.func)
        or _is_unambiguous_control_sink(node, control_aliases)
        or _shell_adapter_invokes_kestrel_lifecycle(node)
        or kills_agent_process
        or targeted_agent_control
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


def _shell_adapter_invokes_kestrel_lifecycle(call: ast.Call) -> bool:
    """Recognize statically visible host-lifecycle CLI re-entry.

    The shell/subprocess adapter is merely the execution mechanism; the
    authority-bearing operation lives in its command payload.  Restrict the
    check to known adapters and a literal ``kestrel <lifecycle-verb>`` command
    prefix so logging or echoing the same words does not become a control sink.
    """

    def reference_path(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id.casefold()
        if isinstance(node, ast.Attribute):
            prefix = reference_path(node.value)
            return (
                f"{prefix}.{node.attr.casefold()}"
                if prefix
                else node.attr.casefold()
            )
        return ""

    adapter = reference_path(call.func)
    adapter_name = adapter.rsplit(".", 1)[-1]
    positional_command = call.args[0] if call.args else None
    keyword_command = next(
        (
            keyword.value
            for keyword in call.keywords
            if keyword.arg in {"args", "cmd", "command"}
        ),
        None,
    )
    command = (
        ast.List(elts=call.args, ctx=ast.Load())
        if adapter == "asyncio.create_subprocess_exec"
        else positional_command or keyword_command
    )
    if command is None:
        return False

    shell_adapters = {"os.system", "shell"}
    subprocess_adapters = {
        "asyncio.create_subprocess_exec",
        "asyncio.create_subprocess_shell",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.popen",
        "subprocess.run",
    }
    if (
        adapter not in shell_adapters
        and adapter_name != "shell"
        and adapter not in subprocess_adapters
    ):
        return False

    return _is_kestrel_lifecycle_command(command)


def _is_kestrel_lifecycle_command(command: ast.AST) -> bool:
    """Whether an expression statically denotes a Kestrel lifecycle command."""

    if getattr(command, "_authority_shell_lifecycle", False):
        return True

    def executable_name(word: str) -> str:
        basename = word.strip("'\"").replace("\\", "/").rsplit("/", 1)[-1]
        for suffix in (".exe", ".cmd", ".bat"):
            if basename.casefold().endswith(suffix):
                basename = basename[: -len(suffix)]
                break
        return basename.casefold()

    def denotes_lifecycle(words: list[str]) -> bool:
        """Unwrap supported launchers and match one canonical CLI operation."""

        if not words:
            return False
        normalized = [word.strip("'\"") for word in words]
        executable = executable_name(normalized[0])
        operation_index = 1
        # Launchers compose (``uv run python -m ...``), so consume them as a
        # grammar instead of making their alternatives mutually exclusive.
        if executable == "uv":
            if len(normalized) < 3 or normalized[1].casefold() != "run":
                return False
            normalized = normalized[2:]
            executable = executable_name(normalized[0])
        if executable == "py" or executable.startswith("python"):
            if (
                len(normalized) < 4
                or normalized[1] != "-m"
                or normalized[2].casefold() != "kestrel_sovereign.cli"
            ):
                return False
            executable = "kestrel"
            operation_index = 3
        return (
            executable == "kestrel"
            and len(normalized) > operation_index
            and executable_name(normalized[operation_index])
            in _KESTREL_CLI_LIFECYCLE_ACTIONS
        )

    def literal_tokens(node: ast.AST) -> list[str]:
        return [
            token
            for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
            for token in SOURCE_IDENTIFIER_CHAIN.findall(child.value.casefold())
        ]

    if isinstance(command, (ast.List, ast.Tuple)):
        words: list[str] = []
        for element in command.elts:
            resolved = _resolved_string(element)
            if resolved is None and (
                isinstance(element, ast.Attribute)
                and isinstance(element.value, ast.Name)
                and element.value.id == "sys"
                and element.attr == "executable"
            ):
                resolved = "python"
            if resolved is not None:
                words.append(resolved)
        if len(words) == len(command.elts) and denotes_lifecycle(words):
            return True
        return denotes_lifecycle(literal_tokens(command))

    resolved_command = _resolved_string(command)
    if resolved_command is None:
        # Preserve a statically visible command prefix when later f-string
        # fields are dynamic (for example ``f"kestrel restart {name}"``).
        prefix = ""
        if isinstance(command, ast.JoinedStr):
            prefix = "".join(
                value.value
                for value in command.values
                if isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            )
        if prefix:
            for posix in (True, False):
                try:
                    words = shlex.split(prefix, posix=posix)
                except ValueError:
                    continue
                if denotes_lifecycle(words):
                    return True
        tokens = literal_tokens(command)
        return denotes_lifecycle(tokens)
    for posix in (True, False):
        try:
            words = shlex.split(resolved_command, posix=posix)
        except ValueError:
            continue
        if denotes_lifecycle(words):
            return True
    return False


def _annotate_shell_lifecycle_command_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    inherited_aliases: set[str] | None = None,
) -> set[str]:
    """Propagate and kill lifecycle values with the shared binding lattice."""

    local_bindings = _scope_local_binding_names(function)
    inherited = {
        alias
        for alias in inherited_aliases or set()
        if _state_alias_root(alias) not in local_bindings
    }
    lifecycle = ("command-effect", "kestrel-lifecycle")
    def direct(
        current: _StaticBindingFlow, value: ast.AST
    ) -> _StaticBinding | None:
        if _is_kestrel_lifecycle_command(value):
            return lifecycle
        if isinstance(value, ast.Attribute):
            return current.bindings.get(value.attr.casefold())
        return None

    def annotate(value: ast.AST, binding: _StaticBinding) -> None:
        if binding == lifecycle:
            setattr(value, "_authority_shell_lifecycle", True)

    flow = _StaticBindingFlow(
        direct,
        lambda bindings: lifecycle,
        dict.fromkeys(inherited, lifecycle),
        normalize_name=str.casefold,
        on_resolve=annotate,
        stateful_resolver=True,
    )
    flow.replay(function.body)
    return {
        name for name, binding in flow.bindings.items() if binding == lifecycle
    }


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
        "revoke",
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

    if _is_attribution_validation_call(call):
        return False
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


def _is_recipient_scoped_task_read(call: ast.Call) -> bool:
    """Recognize task reads whose explicit recipient is an authority input."""

    return _call_name(call).casefold() in {"get_task", "list_tasks"} and any(
        keyword.arg == "recipient_agent_id" for keyword in call.keywords
    )


def _is_provenance_transform_call(call: ast.Call) -> bool:
    """Whether a call's result retains or serializes its input value."""

    call_name = _call_name(call).casefold()
    return call_name in PROVENANCE_TRANSFORM_CALLS or call_name in {
        "join",
        "repr",
        "str",
    } or any(
        marker in call_name
        for marker in (
            "canonical",
            "dump",
            "encode",
            "format",
            "normaliz",
            "serializ",
        )
    )


def _provenance_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    provenance_return_helpers: set[str] | None = None,
    control_helpers: set[str] | None = None,
    initial_aliases: set[str] | None = None,
    *,
    control_return_helpers: set[str] | None = None,
    authority_analysis: bool = True,
    state_object_aliases: set[str] | None = None,
    provenance_accessor_aliases: set[str] | None = None,
    parameter_mutation_flows: dict[
        str, tuple[_ParameterMutationFlow, ...]
    ] | None = None,
    control_parameter_return_flows: dict[
        str, _ParameterReturnFlow
    ] | None = None,
    shared_state_bindings: set[str] | None = None,
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
            and (
                _call_name(node).casefold()
                in (provenance_return_helpers or set())
                or _CALLABLE_PROVENANCE in _callable_semantics(node)
            )
            for node in ast.walk(value)
        )
        calls_known_accessor = _is_provenance_accessor_call(value) or any(
            isinstance(node, ast.Call)
            and _call_name(node).casefold()
            in (provenance_accessor_aliases or set())
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
            return _has_provenance_token(
                value,
                aliases,
                include_unverified_attribution=False,
            )
        if (
            isinstance(value, ast.Call)
            and _is_provenance_transform_call(value)
        ):
            return (
                _has_provenance_token(
                    value,
                    aliases,
                    include_unverified_attribution=False,
                )
                or calls_known_helper
                or calls_known_accessor
            )
        if isinstance(value, ast.Call):
            call_name = _call_name(value).casefold()
            if _is_provenance_accessor_call(value):
                return True
            if (
                isinstance(value.func, ast.Attribute)
                and _has_provenance_value(
                    value.func.value,
                    aliases,
                    provenance_return_helpers,
                    include_unverified_attribution=False,
                )
            ):
                # Selection/transform methods retain authority provenance from
                # their receiver (pop/get/index helpers included).  Arguments
                # alone are insufficient for method-call dataflow.
                return True
            if (
                call_name in (provenance_return_helpers or set())
                or _CALLABLE_PROVENANCE in _callable_semantics(value)
            ):
                return True
            if call_name.startswith(("can_", "has_", "is_", "may_")) or (
                _is_permission_name(call_name)
            ):
                return _has_provenance_token(
                    value,
                    aliases,
                    include_unverified_attribution=False,
                ) or calls_known_helper
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
            return _has_provenance_token(
                value,
                aliases,
                include_unverified_attribution=False,
            ) or calls_known_helper
        return False

    aliases: set[str] = set(initial_aliases or ())
    positional_parameters = [
        *function.args.posonlyargs,
        *function.args.args,
    ]
    annotated_parameters = [
        *positional_parameters,
        *function.args.kwonlyargs,
        *(
            [function.args.vararg]
            if function.args.vararg is not None
            else []
        ),
        *(
            [function.args.kwarg]
            if function.args.kwarg is not None
            else []
        ),
    ]
    aliases.update(
        parameter.arg.casefold()
        for parameter in annotated_parameters
        if parameter.annotation is not None
        and _has_provenance_token(parameter.annotation, aliases)
    )
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
    statement_owners = _lexical_statement_owners(function)
    decision_provenance_helpers = set(provenance_return_helpers or ()) | set(
        provenance_accessor_aliases or ()
    )
    control_aliases = (
        _cross_agent_control_aliases(
            function,
            control_helpers,
            control_return_helpers,
            control_parameter_return_flows,
        )
        if authority_analysis
        else set()
    )
    control_argument_names = (
        {
            token
            for node in scope_nodes
            if isinstance(node, ast.Call)
            and _is_cross_agent_control_call(
                node, control_aliases, state_object_aliases
            )
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
    container_aliases = _mutable_container_alias_snapshots(
        function,
        set(shared_state_bindings or ()),
    )
    callable_alias_edges = _scope_callable_alias_edges(function)

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
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
            value = ast.BinOp(left=node.target, op=node.op, right=node.value)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
            value = node.iter
        elif isinstance(node, ast.Call):
            mutation = _mutable_container_write(node)
            if mutation is not None:
                names, value = mutation
                assignments.append(
                    (
                        expand_container_aliases(names, node),
                        value,
                        statement_owners.get(node, node),
                    )
                )
            source_names = _expanded_callable_sources(
                _call_name(node), callable_alias_edges
            )
            for source_name in source_names:
                for flow in (parameter_mutation_flows or {}).get(
                    source_name, ()
                ):
                    targets = _bound_parameter_flow_arguments(
                        node, flow.targets
                    )
                    values = _bound_parameter_flow_arguments(
                        node, flow.sources
                    )
                    if flow.direct_provenance:
                        values.append(ast.Constant(value="causation_chain"))
                    target_names = {
                        name
                        for target in targets
                        for name in expand_container_aliases(
                            _reference_binding_names(target), node
                        )
                    }
                    if target_names and values:
                        assignments.append(
                            (
                                target_names,
                                values[0]
                                if len(values) == 1
                                else ast.Tuple(
                                    elts=values, ctx=ast.Load()
                                ),
                                statement_owners.get(node, node),
                            )
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
            assignments.append((names, value, statement_owners.get(node, node)))

    for node in scope_nodes:
        if not isinstance(node, ast.Match):
            continue
        for case in node.cases:
            assignments.extend(
                (
                    names,
                    source,
                    node,
                )
                for names, source in _match_pattern_provenance_assignments(
                    case.pattern,
                    node.subject,
                )
            )

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
        enclosing_loop_continuation_controls: bool = False,
    ) -> None:
        suffix_controls = [False] * (len(statements) + 1)
        for index in range(len(statements) - 1, -1, -1):
            suffix_controls[index] = suffix_controls[index + 1] or (
                _contains_cross_agent_control_call(
                    statements[index], control_aliases, state_object_aliases
                )
            )
        for index, statement in enumerate(statements):
            local_continuation_controls = (
                suffix_controls[index + 1]
                or enclosing_continuation_controls
            )
            controls_continuation = (
                local_continuation_controls
                or enclosing_loop_continuation_controls
            )
            if controls_continuation:
                if isinstance(statement, ast.Assert):
                    authority_decision_names.update(
                        _identifier_tokens(statement.test)
                    )
                elif isinstance(statement, ast.If):
                    local_exit_differs = (
                        local_continuation_controls
                        and _block_guaranteed_exits(statement.body)
                        != _block_guaranteed_exits(statement.orelse)
                    )
                    loop_exit_differs = (
                        enclosing_loop_continuation_controls
                        and _block_then_suffix_guaranteed_function_exit(
                            statement.body,
                            statements[index + 1 :],
                        )
                        != _block_then_suffix_guaranteed_function_exit(
                            statement.orelse,
                            statements[index + 1 :],
                        )
                    )
                    if local_exit_differs or loop_exit_differs:
                        authority_decision_names.update(
                            _identifier_tokens(statement.test)
                        )
                elif isinstance(statement, ast.While) or (
                    isinstance(statement, (ast.For, ast.AsyncFor))
                    and _loop_else_guards_continuation(statement)
                ):
                    decision = (
                        statement.iter
                        if isinstance(statement, (ast.For, ast.AsyncFor))
                        else statement.test
                    )
                    authority_decision_names.update(
                        _identifier_tokens(decision)
                    )
            is_loop = isinstance(statement, (ast.For, ast.AsyncFor, ast.While))
            child_continuation_controls = (
                False if is_loop else local_continuation_controls
            )
            child_loop_continuation_controls = (
                (
                    local_continuation_controls
                    or enclosing_loop_continuation_controls
                )
                if is_loop
                else enclosing_loop_continuation_controls
            )
            for block in _child_statement_blocks(statement):
                collect_guard_decisions(
                    block,
                    child_continuation_controls,
                    child_loop_continuation_controls,
                )

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
                    value,
                    aliases,
                    decision_provenance_helpers,
                    include_unverified_attribution=False,
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
    function_imported_provenance_helpers: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] | None = None,
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
        class_name = class_node.name.casefold()

        def is_container_creation(value: ast.AST | None) -> bool:
            return isinstance(value, (ast.Dict, ast.List, ast.Set)) or (
                isinstance(value, ast.Call)
                and _call_name(value).casefold()
                in {"defaultdict", "deque", "dict", "list", "set"}
            )

        mutable_members: set[str] = set()
        declaration_nodes: list[ast.AST] = [
            *class_node.body,
            *(node for method in methods for node in _walk_lexical_scope(method)),
        ]
        for declaration in declaration_nodes:
            if isinstance(declaration, ast.Assign):
                targets = declaration.targets
                value = declaration.value
            elif isinstance(declaration, ast.AnnAssign):
                targets = [declaration.target]
                value = declaration.value
            else:
                continue
            if not is_container_creation(value):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and declaration in class_node.body:
                    member = target.id.casefold()
                elif (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in {"self", "cls", class_node.name}
                ):
                    member = target.attr.casefold()
                else:
                    continue
                mutable_members.update(
                    {
                        member,
                        f"self.{member}",
                        f"cls.{member}",
                        f"{class_name}.{member}",
                    }
                )
        shared: set[str] = set()
        changed = True
        while changed:
            changed = False
            for method in methods:
                aliases = set(module_provenance_aliases or ()) | shared
                visible_helpers = set(provenance_return_helpers or ()) | set(
                    (function_imported_provenance_helpers or {}).get(
                        method, ()
                    )
                )
                container_aliases = _mutable_container_alias_snapshots(
                    method, mutable_members
                )
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
                        calls_helper = any(
                            isinstance(child, ast.Call)
                            and (
                                _call_name(child).casefold()
                                in visible_helpers
                                or _CALLABLE_PROVENANCE
                                in _callable_semantics(child)
                            )
                            for child in ast.walk(value)
                        )
                        if not (
                            _has_provenance_token(
                                value,
                                aliases,
                                include_unverified_attribution=False,
                            )
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
                        target_names.update(
                            container_aliases.get(statement, ())
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


def _class_control_state_aliases(
    tree: ast.AST,
    control_helpers: set[str],
    module_control_aliases: set[str],
    function_control_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ],
    control_return_helpers: set[str],
) -> dict[ast.FunctionDef | ast.AsyncFunctionDef, set[str]]:
    """Share statically stored control callables across class methods."""

    if not isinstance(tree, ast.Module):
        return {}

    def expanded_member_names(
        names: set[str], class_name: str
    ) -> set[str]:
        expanded = set(names)
        prefixes = ("self.", "cls.", f"{class_name.casefold()}.")
        for name in names:
            member = next(
                (
                    name.removeprefix(prefix)
                    for prefix in prefixes
                    if name.startswith(prefix)
                ),
                name,
            )
            member = member.split("[", maxsplit=1)[0]
            expanded.update(
                {
                    member,
                    f"self.{member}",
                    f"cls.{member}",
                    f"{class_name.casefold()}.{member}",
                }
            )
        return expanded

    def stored_names(
        scope: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        class_body: bool,
    ) -> set[str]:
        names: set[str] = set()
        for node in _walk_lexical_scope(scope):
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.NamedExpr):
                targets = [node.target]
            for target in targets:
                target_source = ast.unparse(target).casefold()
                if class_body or target_source.startswith(("self.", "cls.")):
                    names.update(_reference_binding_names(target))
            if not isinstance(node, ast.Call):
                continue
            if (
                _call_name(node).casefold() == "setattr"
                and len(node.args) >= 3
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id.casefold() in {"self", "cls"}
            ):
                attribute = _resolved_string(node.args[1])
                if attribute is not None:
                    names.update(
                        {
                            attribute.casefold(),
                            f"{node.args[0].id.casefold()}.{attribute.casefold()}",
                        }
                    )
        return names

    by_method: dict[ast.FunctionDef | ast.AsyncFunctionDef, set[str]] = {}
    for class_node in (
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ):
        methods = [
            statement
            for statement in class_node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        class_scope = ast.parse("def __audit_class_body__():\n    pass\n").body[0]
        assert isinstance(class_scope, ast.FunctionDef)
        class_scope.body = class_node.body
        ast.copy_location(class_scope, class_node)
        class_targets = stored_names(class_scope, class_body=True)
        class_aliases = _cross_agent_control_aliases(
            class_scope,
            control_helpers | module_control_aliases,
            control_return_helpers | module_control_aliases,
        )
        shared = expanded_member_names(
            class_targets.intersection(class_aliases), class_node.name
        )
        changed = True
        while changed:
            changed = False
            for method in methods:
                method_targets = stored_names(method, class_body=False)
                aliases = _cross_agent_control_aliases(
                    method,
                    control_helpers
                    | function_control_aliases.get(method, set())
                    | shared,
                    control_return_helpers
                    | function_control_aliases.get(method, set()),
                )
                discovered = expanded_member_names(
                    method_targets.intersection(aliases), class_node.name
                )
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
    function_imported_control_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] | None = None,
    function_callback_control_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] | None = None,
    function_parameter_return_flows: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        dict[str, _ParameterReturnFlow],
    ] | None = None,
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
    reflective_lifecycle_parameters: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[tuple[str, str]]
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
        reflective_lifecycle_parameters[function] = {
            (node.func.args[0].id.casefold(), node.func.args[1].id.casefold())
            for node in _walk_lexical_scope(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Call)
            and _call_name(node.func).casefold() == "getattr"
            and len(node.func.args) > 1
            and isinstance(node.func.args[0], ast.Name)
            and isinstance(node.func.args[1], ast.Name)
            and node.func.args[0].id.casefold() in parameter_names
            and node.func.args[1].id.casefold() in parameter_names
        }

    call_sites = [
        (caller, node)
        for caller in functions
        for node in _walk_lexical_scope(caller)
        if isinstance(node, ast.Call)
    ]
    callable_alias_edges = {
        function: _scope_callable_alias_edges(function)
        for function in functions
    }
    callback_aliases = function_callback_control_aliases
    if callback_aliases is None:
        callback_aliases = {function: set() for function in functions}

    def callback_parameter_bindings(
        call: ast.Call,
        callee: ast.FunctionDef | ast.AsyncFunctionDef,
        callback_parameters: set[str],
    ) -> list[tuple[str, ast.AST]]:
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
        bindings = [
            (parameter.arg.casefold(), value)
            for parameter, value in zip(positional_parameters, call.args)
            if parameter.arg.casefold() in callback_parameters
        ]
        bindings.extend(
            (keyword.arg.casefold(), keyword.value)
            for keyword in call.keywords
            if keyword.arg is not None
            and keyword.arg.casefold() in callback_parameters
        )
        # A starred call prevents an exact position proof.  If this helper
        # invokes any callback parameter, conservatively inspect the expanded
        # value rather than treating the ambiguity as evidence of safety.
        if callback_parameters:
            ambiguous_values = [
                argument.value
                for argument in call.args
                if isinstance(argument, ast.Starred)
            ]
            ambiguous_values.extend(
                keyword.value for keyword in call.keywords if keyword.arg is None
            )
            bindings.extend(
                (parameter, value)
                for parameter in callback_parameters
                for value in ambiguous_values
            )
        return bindings

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
            visible_control_aliases = (
                helper_names
                | set(
                    (function_imported_control_aliases or {}).get(function, ())
                )
                | callback_aliases[function]
            )
            scope_nodes = _walk_lexical_scope(function)
            state_object_aliases = _cross_agent_state_object_aliases(function)
            invokes_control = any(
                isinstance(node, ast.Call)
                and _is_cross_agent_control_call(
                    node,
                    visible_control_aliases,
                    state_object_aliases,
                )
                # An unresolved immediately-invoked ``getattr`` fails closed
                # when it is itself provenance-guarded. Do not promote every
                # neutral helper containing one into a repository-wide
                # lifecycle helper: configuration predicate loops such as
                # ``getattr(config, name)()`` are not authority sinks.
                and _control_reference_sources(node.func)
                != {"dynamic_control_attribute"}
                for node in scope_nodes
            )
            mutates_cross_agent_state = any(
                _is_cross_agent_state_mutation_node(
                    node,
                    state_object_aliases,
                )
                for node in scope_nodes
            )
            if invokes_control or mutates_cross_agent_state:
                if function_name not in helper_names:
                    helper_names.add(function_name)
                    changed = True

            reflective_parameters = reflective_lifecycle_parameters.get(
                function, set()
            )
            for caller, call in call_sites:
                if not reflective_parameters or function_name not in (
                    _expanded_callable_sources(
                        _call_name(call), callable_alias_edges[caller]
                    )
                ):
                    continue
                caller_state_objects = _cross_agent_state_object_aliases(caller)
                for object_parameter, name_parameter in reflective_parameters:
                    bound_arguments = dict(
                        callback_parameter_bindings(
                            call,
                            function,
                            {object_parameter, name_parameter},
                        )
                    )
                    object_argument = bound_arguments.get(object_parameter)
                    name_argument = bound_arguments.get(name_parameter)
                    resolved_name = (
                        _resolved_string(name_argument)
                        if name_argument is not None
                        else None
                    )
                    if (
                        object_argument is not None
                        and _is_cross_agent_state_object_reference(
                            object_argument, caller_state_objects
                        )
                        and resolved_name is not None
                        and _is_unambiguous_control_token(resolved_name)
                        and function_name not in helper_names
                    ):
                        helper_names.add(function_name)
                        changed = True

            callback_parameters = invoked_parameters.get(function, set())
            if not callback_parameters:
                continue
            for caller, call in call_sites:
                if function_name not in _expanded_callable_sources(
                    _call_name(call), callable_alias_edges[caller]
                ):
                    continue
                caller_aliases = _cross_agent_control_aliases(
                    caller,
                    helper_names
                    | set(
                        (function_imported_control_aliases or {}).get(
                            caller, ()
                        )
                    )
                    | callback_aliases[caller],
                    parameter_return_flows=(
                        (function_parameter_return_flows or {}).get(caller)
                    ),
                )
                discovered_parameters = {
                    parameter
                    for parameter, argument in callback_parameter_bindings(
                        call, function, callback_parameters
                    )
                    if is_control_reference(argument, caller_aliases)
                }
                new_parameters = (
                    discovered_parameters - callback_aliases[function]
                )
                if new_parameters:
                    callback_aliases[function].update(new_parameters)
                    changed = True
                if discovered_parameters and function_name not in helper_names:
                    helper_names.add(function_name)
                    changed = True
    return helper_names


class _ParameterReturnFlow(NamedTuple):
    """Call-shape summary for formal parameters that influence a return."""

    positional: frozenset[int]
    keywords: frozenset[str]
    vararg_from: int | None
    kwarg: bool
    accepted_keywords: frozenset[str]
    implicit_receiver: bool


class _ParameterMutationFlow(NamedTuple):
    """Call-shape summary for provenance written through a parameter."""

    targets: _ParameterReturnFlow
    sources: _ParameterReturnFlow
    direct_provenance: bool


def _parameter_return_flow(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    flowed_parameters: set[str],
) -> _ParameterReturnFlow:
    """Convert flowed formal names into a caller-facing immutable summary."""

    positional = [*function.args.posonlyargs, *function.args.args]
    implicit_receiver = bool(
        positional and positional[0].arg.casefold() in {"self", "cls"}
    )
    positional_only = {
        parameter.arg.casefold() for parameter in function.args.posonlyargs
    }
    keyword_parameters = [
        *function.args.args,
        *function.args.kwonlyargs,
    ]
    accepted_keywords = {
        parameter.arg.casefold() for parameter in keyword_parameters
    }
    return _ParameterReturnFlow(
        positional=frozenset(
            index
            for index, parameter in enumerate(positional)
            if parameter.arg.casefold() in flowed_parameters
        ),
        keywords=frozenset(accepted_keywords.intersection(flowed_parameters)),
        vararg_from=(
            len(positional)
            if function.args.vararg is not None
            and function.args.vararg.arg.casefold() in flowed_parameters
            else None
        ),
        kwarg=(
            function.args.kwarg is not None
            and function.args.kwarg.arg.casefold() in flowed_parameters
        ),
        accepted_keywords=frozenset(accepted_keywords - positional_only),
        implicit_receiver=implicit_receiver,
    )


def _bound_parameter_flow_arguments(
    call: ast.Call,
    flow: _ParameterReturnFlow,
) -> list[ast.AST]:
    """Return actual values bound to return-influencing formal parameters."""

    positional_indexes = set(flow.positional)
    if flow.implicit_receiver:
        positional_indexes.update(
            index - 1 for index in flow.positional if index > 0
        )
        if 0 in flow.positional and isinstance(call.func, ast.Attribute):
            positional_indexes.add(-1)
    values = [
        call.func.value
        if index == -1 and isinstance(call.func, ast.Attribute)
        else call.args[index].value
        if isinstance(call.args[index], ast.Starred)
        else call.args[index]
        for index in positional_indexes
        if index == -1 and isinstance(call.func, ast.Attribute)
        or 0 <= index < len(call.args)
    ]
    values.extend(
        keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
        and keyword.arg.casefold() in flow.keywords
    )
    if flow.positional:
        values.extend(
            argument.value
            for argument in call.args
            if isinstance(argument, ast.Starred)
        )
    if flow.keywords:
        values.extend(
            keyword.value for keyword in call.keywords if keyword.arg is None
        )
    if flow.vararg_from is not None:
        vararg_from = (
            max(0, flow.vararg_from - 1)
            if flow.implicit_receiver
            else flow.vararg_from
        )
        values.extend(
            argument.value if isinstance(argument, ast.Starred) else argument
            for argument in call.args[vararg_from:]
        )
        values.extend(
            argument.value
            for argument in call.args[:vararg_from]
            if isinstance(argument, ast.Starred)
        )
    if flow.kwarg:
        values.extend(
            keyword.value
            for keyword in call.keywords
            if keyword.arg is None
            or keyword.arg.casefold() not in flow.accepted_keywords
        )
    return values


def _local_parameter_mutation_flows(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> dict[str, tuple[_ParameterMutationFlow, ...]]:
    """Summarize provenance-bearing writes through local helper arguments.

    A helper can return ``None`` yet still place an authority decision in a
    caller-owned mapping or object.  Record both direct provenance reads and
    ordinary formal-parameter flow so the caller can replay that side effect
    as a normal assignment in its own provenance graph.
    """

    grouped: dict[str, set[_ParameterMutationFlow]] = {}
    for function in functions:
        parameters = {
            parameter.arg.casefold()
            for parameter in [
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
                *(
                    [function.args.vararg]
                    if function.args.vararg is not None
                    else []
                ),
                *(
                    [function.args.kwarg]
                    if function.args.kwarg is not None
                    else []
                ),
            ]
        }
        if not parameters:
            continue

        value_assignments: list[tuple[set[str], ast.AST]] = []
        identity_assignments: list[tuple[set[str], ast.AST]] = []
        writes: list[tuple[set[str], ast.AST]] = []

        def mutation_target_names(target: ast.AST) -> set[str]:
            """Name the mutated reference and its receiver chain, not keys."""

            names = {ast.unparse(target).casefold()}
            receiver = target
            while isinstance(receiver, (ast.Attribute, ast.Subscript)):
                receiver = receiver.value
                names.add(ast.unparse(receiver).casefold())
            return names

        for node in _walk_lexical_scope(function):
            targets: list[ast.AST] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
                value = node.value
            elif isinstance(node, ast.NamedExpr):
                targets = [node.target]
                value = node.value

            if value is not None:
                bound_names = {
                    name
                    for target in targets
                    if not isinstance(target, (ast.Attribute, ast.Subscript))
                    for name in _binding_target_names(target)
                }
                if bound_names:
                    value_assignments.append((bound_names, value))
                    if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript)):
                        identity_assignments.append((bound_names, value))
                mutated_names = {
                    name
                    for target in targets
                    if isinstance(target, (ast.Attribute, ast.Subscript))
                    for name in mutation_target_names(target)
                }
                if mutated_names:
                    writes.append((mutated_names, value))
                continue

            if isinstance(node, ast.Call):
                mutation = _mutable_container_write(node)
                if mutation is not None:
                    mutated_names, written_value = mutation
                    writes.append(
                        (
                            mutated_names,
                            written_value,
                        )
                    )

        parameter_object_aliases = {
            parameter: {parameter} for parameter in parameters
        }
        for aliases in parameter_object_aliases.values():
            changed = True
            while changed:
                changed = False
                for target_names, value in identity_assignments:
                    if not _identifier_tokens(value).intersection(aliases):
                        continue
                    new_aliases = target_names - aliases
                    if new_aliases:
                        aliases.update(new_aliases)
                        changed = True

        parameter_value_aliases = {
            parameter: {parameter} for parameter in parameters
        }
        for aliases in parameter_value_aliases.values():
            changed = True
            while changed:
                changed = False
                for target_names, value in value_assignments:
                    if not _identifier_tokens(value).intersection(aliases):
                        continue
                    new_aliases = target_names - aliases
                    if new_aliases:
                        aliases.update(new_aliases)
                        changed = True

        provenance_aliases: set[str] = set()
        changed = True
        while changed:
            changed = False
            for target_names, value in value_assignments:
                if not _has_provenance_token(value, provenance_aliases):
                    continue
                new_aliases = target_names - provenance_aliases
                if new_aliases:
                    provenance_aliases.update(new_aliases)
                    changed = True

        flows = grouped.setdefault(function.name.casefold(), set())
        for target_names, value in writes:
            target_parameters = {
                parameter
                for parameter, aliases in parameter_object_aliases.items()
                if target_names.intersection(aliases)
            }
            if not target_parameters:
                continue
            value_tokens = set(_identifier_tokens(value))
            source_parameters = {
                parameter
                for parameter, aliases in parameter_value_aliases.items()
                if value_tokens.intersection(aliases)
            }
            direct_provenance = _has_provenance_token(
                value, provenance_aliases
            )
            if not source_parameters and not direct_provenance:
                continue
            flows.add(
                _ParameterMutationFlow(
                    targets=_parameter_return_flow(
                        function, target_parameters
                    ),
                    sources=_parameter_return_flow(
                        function, source_parameters
                    ),
                    direct_provenance=direct_provenance,
                )
            )

    return {
        name: tuple(sorted(flows, key=repr))
        for name, flows in grouped.items()
        if flows
    }


def _scope_callable_alias_edges(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, set[str]]]:
    """Collect same-scope aliases that can rename a called local/imported helper."""

    edges: list[tuple[str, set[str]]] = []
    nodes = (
        ast.walk(scope)
        if isinstance(scope, ast.Module)
        else _walk_lexical_scope(scope)
    )
    for node in nodes:
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
        sources = (
            {value.id.casefold()}
            if isinstance(value, ast.Name)
            else {
                value.attr.casefold(),
                ast.unparse(value).casefold(),
            }
            if isinstance(value, ast.Attribute)
            else set()
        )
        if sources:
            edges.extend(
                (target_name, sources)
                for target in targets
                for target_name in _binding_target_names(target)
            )
    return edges


def _expanded_callable_sources(
    name: str,
    edges: list[tuple[str, set[str]]],
) -> set[str]:
    """Walk a bounded alias chain backwards from the invoked local name."""

    sources = {name.casefold()}
    changed = True
    while changed:
        changed = False
        for target, candidates in edges:
            if target in sources:
                new_sources = candidates - sources
                if new_sources:
                    sources.update(new_sources)
                    changed = True
    return sources


_CALLABLE_CONTROL = "control"
_CALLABLE_PROVENANCE = "provenance"
_CALLABLE_CYCLE_BOUNDED = "cycle_bounded"
_CALLABLE_VALUE = "callable"
_CALLABLE_CLASS = "class"
_CALLABLE_FACTORY = "factory"

_CYCLE_BOUNDED_UNIVERSAL_CALLS = {
    "ask_agent",
    "peer_stop",
    "send_a2a_message",
    "send_a2a_question",
    "send_a2a_task",
    "stop_peer",
}

_CYCLE_BOUNDED_UNIVERSAL_MODULE = (
    REPO_ROOT / "kestrel_sovereign/features/peers/feature.py"
).resolve()


def _resolved_cycle_bounded_helpers(
    tree: ast.AST,
    source_path: Path | None,
) -> set[str]:
    """Name universal rails only when their defining module is verified."""

    if (
        source_path is None
        or source_path.resolve() != _CYCLE_BOUNDED_UNIVERSAL_MODULE
    ):
        return set()
    return {
        node.name.casefold()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.casefold() in _CYCLE_BOUNDED_UNIVERSAL_CALLS
    }


def _callable_semantics(node: ast.AST) -> frozenset[str]:
    """Return semantic effects attached by the source-order callable flow."""

    return getattr(node, "_authority_callable_semantics", frozenset())


def _mark_callable_semantics(node: ast.AST, semantics: set[str]) -> None:
    """Attach monotone semantic evidence to one expression node."""

    setattr(
        node,
        "_authority_callable_semantics",
        _callable_semantics(node) | frozenset(semantics),
    )


def _semantic_binding(
    semantics: set[str] | frozenset[str], kind: str
) -> _StaticBinding:
    return "+".join(sorted(semantics)), kind


def _semantic_binding_flags(binding: _StaticBinding) -> set[str]:
    return set(binding[0].split("+")) - {""}


def _merge_semantic_bindings(
    bindings: list[_StaticBinding],
) -> _StaticBinding:
    """Conservatively join callable effects across possible flow paths."""

    semantics = {
        semantic
        for binding in bindings
        for semantic in _semantic_binding_flags(binding)
    }
    kind = (
        bindings[0][1]
        if bindings and all(binding[1] == bindings[0][1] for binding in bindings)
        else _CALLABLE_VALUE
    )
    return _semantic_binding(semantics, kind)


class _CallableSemanticFlow(_StaticBindingFlow):
    """Source-order identity flow for summarized callable behavior.

    Function aliases and instances of callable classes are the same semantic
    problem: Python changed the expression that names a callable, not what an
    invocation does.  Keep that identity in one small abstract domain and
    annotate call sites for the provenance and control scanners to consume.
    """

    def __init__(
        self,
        declarations: dict[str, _StaticBinding],
        classes: dict[str, _StaticBinding],
        bindings: dict[str, _StaticBinding] | None = None,
    ) -> None:
        super().__init__(
            lambda _node: None,
            _merge_semantic_bindings,
            {
                name.casefold(): binding
                for name, binding in (bindings or {}).items()
            },
        )
        self.declarations = declarations
        self.classes = classes

    def fork(self) -> _CallableSemanticFlow:
        return _CallableSemanticFlow(
            self.declarations,
            self.classes,
            self.bindings,
        )

    def resolve(
        self,
        value: ast.AST,
        supplemental_resolver: Callable[
            [ast.AST], _StaticBinding | None
        ]
        | None = None,
    ) -> _StaticBinding | None:
        while isinstance(value, (ast.Await, ast.Expr)):
            value = value.value
        if isinstance(value, ast.Name):
            return self.bindings.get(value.id.casefold())
        if isinstance(value, ast.Attribute):
            return self.bindings.get(
                ast.unparse(value).casefold()
            ) or self.bindings.get(value.attr.casefold()) or self.resolve(
                value.value, supplemental_resolver
            )
        if isinstance(value, ast.Subscript):
            # A may-analysis must preserve the joined semantics of values
            # stored in a container when a later static selector retrieves
            # one.  The container binding is already the conservative join of
            # its literal members, so no key-specific execution is needed.
            return self.resolve(value.value, supplemental_resolver)
        if isinstance(value, ast.Dict):
            return self._merge_values([
                self.resolve(member, supplemental_resolver)
                for member in value.values
            ])
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            return self._merge_values([
                self.resolve(member, supplemental_resolver)
                for member in value.elts
            ])
        if isinstance(value, ast.IfExp):
            return self._merge_values([
                self.resolve(value.body),
                self.resolve(value.orelse),
            ])
        if not isinstance(value, ast.Call):
            return None

        callee = self.resolve(value.func)
        if callee is not None and callee[1] in {
            _CALLABLE_CLASS,
            _CALLABLE_FACTORY,
        }:
            return _semantic_binding(
                _semantic_binding_flags(callee), _CALLABLE_VALUE
            )
        if _call_name(value).casefold() in {
            "cache",
            "cast",
            "identity",
            "lru_cache",
            "partial",
            "partialmethod",
            "singledispatch",
            "singledispatchmethod",
            "update_wrapper",
            "wraps",
        } and value.args:
            return self.resolve(value.args[0])
        return None

    def declare(
        self,
        statement: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> None:
        name = statement.name.casefold()
        self.bindings.pop(name, None)
        binding = (
            self.classes.get(name)
            if isinstance(statement, ast.ClassDef)
            else self.declarations.get(name)
        )
        if binding is not None:
            self.bindings[name] = binding

    def assign(
        self,
        targets: list[ast.AST],
        value: ast.AST | None,
        supplemental_resolver: Callable[
            [ast.AST], _StaticBinding | None
        ]
        | None = None,
    ) -> None:
        """Replay bindings in the same normalized namespace as ``resolve``."""

        assigned = (
            self.assignment_bindings(
                targets,
                value,
                supplemental_resolver,
            )
            if value is not None
            else {}
        )
        rebound = {
            name.casefold()
            for target in targets
            for name in _assignment_target_names(target)
        }
        for name in rebound:
            self.bindings.pop(name, None)
        self.bindings.update(
            {name.casefold(): binding for name, binding in assigned.items()}
        )
        if value is None:
            return
        stored = self.resolve(value, supplemental_resolver)
        if stored is None:
            return
        for target in targets:
            receiver = (
                target.value
                if isinstance(target, (ast.Attribute, ast.Subscript))
                else None
            )
            if receiver is None:
                continue
            for name in _reference_binding_names(receiver):
                normalized = name.casefold()
                existing = self.bindings.get(normalized)
                self.bindings[normalized] = self._merge_values(
                    [existing, stored]
                ) or stored

    def store_container_value(
        self,
        names: set[str],
        value: ast.AST,
    ) -> None:
        """Join a mutable-container write into its callable may-binding."""

        stored = self.resolve(value)
        if stored is None:
            return
        for name in names:
            normalized = name.casefold()
            existing = self.bindings.get(normalized)
            self.bindings[normalized] = self._merge_values(
                [existing, stored]
            ) or stored

    def disturbed_fork(self, statement: ast.stmt) -> _CallableSemanticFlow:
        """Invalidate branch-local names in the normalized namespace."""

        branch = self.fork()
        for name in _compound_binding_names(statement, set()):
            branch.bindings.pop(name.casefold(), None)
        return branch

    def _annotate_expression(self, expression: ast.AST) -> None:
        flow = self

        class SemanticCallVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
                self.generic_visit(node)
                for operand in [
                    node.func,
                    *node.args,
                    *(keyword.value for keyword in node.keywords),
                ]:
                    binding = flow.resolve(operand)
                    if binding is not None:
                        _mark_callable_semantics(
                            operand, _semantic_binding_flags(binding)
                        )
                callee = flow.resolve(node.func)
                if callee is None:
                    return
                semantics = _semantic_binding_flags(callee)
                if callee[1] == _CALLABLE_VALUE:
                    _mark_callable_semantics(node.func, semantics)
                    if _CALLABLE_PROVENANCE in semantics:
                        _mark_callable_semantics(
                            node, {_CALLABLE_PROVENANCE}
                        )
                elif callee[1] in {_CALLABLE_CLASS, _CALLABLE_FACTORY}:
                    _mark_callable_semantics(node, semantics)

            def visit_FunctionDef(  # noqa: N802
                self, node: ast.FunctionDef
            ) -> None:
                return

            def visit_AsyncFunctionDef(  # noqa: N802
                self, node: ast.AsyncFunctionDef
            ) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
                return

        SemanticCallVisitor().visit(expression)

    def replay(
        self,
        statements: list[ast.stmt],
        supplemental_resolver: Callable[
            [ast.AST], _StaticBinding | None
        ]
        | None = None,
    ) -> None:
        for statement in statements:
            if isinstance(statement, _MODULE_COMPOUND_STATEMENT_TYPES):
                expressions, _blocks = _compound_flow_parts(statement)
                for expression in expressions:
                    self._annotate_expression(expression)
            elif not isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                self._annotate_expression(statement)
                for node in ast.walk(statement):
                    if not isinstance(node, ast.Call):
                        continue
                    mutation = _mutable_container_write(node)
                    if mutation is not None:
                        self.store_container_value(*mutation)
            super().replay([statement])


def _callable_class_semantics(
    tree: ast.AST,
    provenance_return_helpers: set[str],
    control_helpers: set[str],
    module_control_aliases: set[str],
    function_control_imports: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ],
    class_control_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ],
    module_provenance_aliases: set[str],
    class_provenance_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ],
    function_imported_provenance_helpers: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ],
) -> dict[str, frozenset[str]]:
    """Summarize each class whose instances are semantically callable."""

    summarized: dict[str, frozenset[str]] = {}
    if not isinstance(tree, ast.Module):
        return summarized
    for class_node in (
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ):
        call_method = next(
            (
                statement
                for statement in class_node.body
                if isinstance(
                    statement, (ast.FunctionDef, ast.AsyncFunctionDef)
                )
                and statement.name == "__call__"
            ),
            None,
        )
        if call_method is None:
            continue
        semantics: set[str] = set()
        visible_controls = (
            control_helpers
            | module_control_aliases
            | function_control_imports.get(call_method, set())
            | class_control_aliases.get(call_method, set())
        )
        if _contains_cross_agent_control_call(
            call_method.body, visible_controls
        ):
            semantics.add(_CALLABLE_CONTROL)

        visible_provenance_helpers = (
            provenance_return_helpers
            | function_imported_provenance_helpers.get(call_method, set())
        ) - {"__call__"}
        provenance_aliases, _selected = _provenance_aliases(
            call_method,
            visible_provenance_helpers,
            initial_aliases=(
                module_provenance_aliases
                | class_provenance_aliases.get(call_method, set())
            ),
            authority_analysis=False,
        )
        if any(
            isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom))
            and node.value is not None
            and _has_provenance_value(
                node.value,
                provenance_aliases,
                visible_provenance_helpers,
            )
            for node in _walk_lexical_scope(call_method)
        ):
            semantics.add(_CALLABLE_PROVENANCE)
        if semantics:
            summarized[class_node.name.casefold()] = frozenset(semantics)
    return summarized


@lru_cache(maxsize=None)
def _direct_repository_callable_class_semantics(
    source_path: Path,
) -> tuple[tuple[str, frozenset[str]], ...]:
    """Summarize classes defined in one module before following re-exports."""

    source_path = source_path.resolve()
    tree = _parsed_module(source_path)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    call_methods = {
        statement
        for class_node in ast.walk(tree)
        if isinstance(class_node, ast.ClassDef)
        for statement in class_node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == "__call__"
    }
    if not call_methods:
        return ()
    calls = [
        call
        for method in call_methods
        for call in _lexical_scope_calls(method)
    ]
    module_controls = _module_imported_control_aliases(
        tree,
        source_path,
        calls=calls,
    )
    function_controls = _function_control_import_aliases(
        tree,
        functions,
        module_controls,
        source_path,
        eligible_functions=call_methods,
    )
    control_helpers = _local_control_helpers(
        functions,
        module_controls,
        function_controls,
    )
    imported_provenance = _module_imported_provenance_return_helper_aliases(
        tree,
        source_path,
        calls=calls,
    )
    function_provenance = _function_provenance_helper_import_aliases(
        tree,
        functions,
        imported_provenance,
        source_path,
        eligible_functions=call_methods,
    )
    module_provenance = (
        _module_provenance_constant_aliases(tree, source_path)
        | _module_imported_provenance_accessor_aliases(tree)
    )
    summarized = _callable_class_semantics(
        tree,
        set(imported_provenance),
        control_helpers,
        module_controls,
        function_controls,
        {},
        module_provenance,
        {},
        function_provenance,
    )
    return tuple(sorted(summarized.items()))


def _repository_callable_class_semantics(
    source_path: Path,
    requested_names: set[str],
    seen: frozenset[tuple[Path, str]] = frozenset(),
) -> dict[str, frozenset[str]]:
    """Resolve callable-class effects through repository re-export edges."""

    source_path = source_path.resolve()
    requested = {
        name.casefold()
        for name in requested_names
        if (source_path, name.casefold()) not in seen
    }
    if not requested:
        return {}
    direct = dict(_direct_repository_callable_class_semantics(source_path))
    resolved = {
        name: direct[name] for name in requested.intersection(direct)
    }
    active = seen | {
        (source_path, name) for name in requested - resolved.keys()
    }
    for local_name, imported_path, remote_name in _repository_reexport_bindings(
        source_path,
        requested - resolved.keys(),
    ):
        remote = _repository_callable_class_semantics(
            imported_path,
            {remote_name},
            active,
        )
        if remote_name in remote:
            resolved[local_name] = remote[remote_name]
    return resolved


def _scope_imported_callable_class_semantics(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
    source_path: Path | None,
) -> dict[str, frozenset[str]]:
    """Bind invoked imported classes to their summarized ``__call__`` effects."""

    if source_path is None:
        return {}
    source_path = source_path.resolve()
    visible_calls = _lexical_scope_calls(scope)
    called_names = {_call_name(call).casefold() for call in visible_calls}
    imported_classes: dict[str, frozenset[str]] = {}
    for node in _lexical_scope_imports(scope):
        if isinstance(node, ast.ImportFrom):
            imported_path = _resolved_repository_import_path(
                source_path,
                node.module,
                node.level,
            )
            if imported_path is None:
                continue
            bindings = {
                (imported.asname or imported.name).casefold(): (
                    imported.name.casefold()
                )
                for imported in node.names
                if imported.name != "*"
                and (imported.asname or imported.name).casefold()
                in called_names
            }
            star_names = called_names if any(
                imported.name == "*" for imported in node.names
            ) else set()
            remote = _repository_callable_class_semantics(
                imported_path,
                set(bindings.values()) | star_names,
            )
            imported_classes.update(
                {
                    local_name: remote[remote_name]
                    for local_name, remote_name in bindings.items()
                    if remote_name in remote
                }
            )
            imported_classes.update(
                {
                    name: remote[name]
                    for name in star_names.intersection(remote)
                }
            )
            continue
        for imported in node.names:
            imported_path = _resolved_repository_import_path(
                source_path,
                imported.name,
            )
            if imported_path is None:
                continue
            qualifier = (imported.asname or imported.name).casefold()
            member_calls = _module_attribute_call_names(
                visible_calls,
                qualifier,
                scope if isinstance(scope, ast.Module) else None,
            )
            remote = _repository_callable_class_semantics(
                imported_path,
                member_calls,
            )
            imported_classes.update(
                {
                    f"{qualifier}.{name}": semantics
                    for name, semantics in remote.items()
                }
            )
    return imported_classes


def _function_imported_callable_class_semantics(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    source_path: Path | None,
) -> dict[
    ast.FunctionDef | ast.AsyncFunctionDef,
    dict[str, frozenset[str]],
]:
    """Resolve callable-class imports visible at every function scope."""

    # Enclosing/module bindings arrive through the ordinary callable flow.
    # Seeding them again as classes would overwrite an already-constructed
    # instance with its constructor semantics.
    return {
        function: _scope_imported_callable_class_semantics(
            function,
            source_path,
        )
        for function in functions
    }


def _annotate_static_callable_semantics(
    tree: ast.AST,
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    function_parents: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        ast.FunctionDef | ast.AsyncFunctionDef,
    ],
    provenance_return_helpers: set[str],
    control_helpers: set[str],
    control_return_helpers: set[str],
    callable_classes: dict[str, frozenset[str]],
    imported_callable_classes: dict[str, frozenset[str]],
    function_imported_callable_classes: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        dict[str, frozenset[str]],
    ],
    imported_provenance_helpers: set[str],
    module_control_aliases: set[str],
    function_imported_provenance_helpers: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ],
    function_control_imports: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ],
    function_accessor_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ],
    cycle_bounded_helpers: set[str],
) -> None:
    """Annotate invocations after resolving aliases and callable instances."""

    declaration_semantics: dict[str, set[str]] = {}
    for name in provenance_return_helpers:
        declaration_semantics.setdefault(name, set()).add(_CALLABLE_PROVENANCE)
    for name in control_helpers:
        declaration_semantics.setdefault(name, set()).add(_CALLABLE_CONTROL)
        if name in cycle_bounded_helpers:
            declaration_semantics[name].add(_CALLABLE_CYCLE_BOUNDED)
    declarations = {
        name: _semantic_binding(semantics, _CALLABLE_VALUE)
        for name, semantics in declaration_semantics.items()
    }
    for name in control_return_helpers - set(declarations):
        declarations[name] = _semantic_binding(
            {_CALLABLE_CONTROL}, _CALLABLE_FACTORY
        )
    classes = {
        name: _semantic_binding(semantics, _CALLABLE_CLASS)
        for name, semantics in callable_classes.items()
    }

    module_bindings: dict[str, _StaticBinding] = {}
    module_bindings.update({
        name: _semantic_binding(semantics, _CALLABLE_CLASS)
        for name, semantics in imported_callable_classes.items()
    })
    for name in imported_provenance_helpers:
        module_bindings[name] = _semantic_binding(
            {_CALLABLE_PROVENANCE}, _CALLABLE_VALUE
        )
    for name in module_control_aliases:
        existing = module_bindings.get(name)
        semantics = (
            _semantic_binding_flags(existing) if existing is not None else set()
        )
        semantics.add(_CALLABLE_CONTROL)
        module_bindings[name] = _semantic_binding(
            semantics, _CALLABLE_VALUE
        )
    module_flow = _CallableSemanticFlow(
        declarations, classes, module_bindings
    )
    if isinstance(tree, ast.Module):
        module_flow.replay(tree.body)

    functions_by_name: dict[
        str, list[ast.FunctionDef | ast.AsyncFunctionDef]
    ] = {}
    for function in functions:
        functions_by_name.setdefault(function.name.casefold(), []).append(function)
    callable_alias_edges = {
        function: _scope_callable_alias_edges(function)
        for function in functions
    }
    parameter_semantics: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, dict[str, set[str]]
    ] = {function: {} for function in functions}
    method_descriptors: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, tuple[str, str]
    ] = {}
    for class_node in (
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ):
        for statement in class_node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = {
                _decorator_terminal_name(decorator).casefold()
                for decorator in statement.decorator_list
            }
            descriptor = (
                "static"
                if "staticmethod" in decorators
                else "class"
                if "classmethod" in decorators
                else "instance"
            )
            method_descriptors[statement] = (
                class_node.name,
                descriptor,
            )

    def bound_arguments(
        call: ast.Call,
        callee: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> dict[str, list[ast.AST]]:
        """Bind supplied values to parameters for higher-order may-flow."""

        positional = [*callee.args.posonlyargs, *callee.args.args]
        method_descriptor = method_descriptors.get(callee)
        if (
            positional
            and isinstance(call.func, ast.Attribute)
            and method_descriptor is not None
            and method_descriptor[1] != "static"
        ):
            owner, descriptor = method_descriptor
            receiver = call.func.value
            explicit_instance = (
                descriptor == "instance"
                and isinstance(receiver, ast.Name)
                and receiver.id == owner
            )
            if not explicit_instance:
                positional = positional[1:]
        bound: dict[str, list[ast.AST]] = {}
        for parameter, argument in zip(positional, call.args):
            bound.setdefault(parameter.arg.casefold(), []).append(argument)
        if callee.args.vararg is not None:
            for argument in call.args[len(positional) :]:
                bound.setdefault(
                    callee.args.vararg.arg.casefold(), []
                ).append(argument)
        named = {
            parameter.arg.casefold(): parameter
            for parameter in [*positional, *callee.args.kwonlyargs]
        }
        for keyword in call.keywords:
            if keyword.arg is not None and keyword.arg.casefold() in named:
                bound.setdefault(keyword.arg.casefold(), []).append(
                    keyword.value
                )
            elif callee.args.kwarg is not None:
                bound.setdefault(
                    callee.args.kwarg.arg.casefold(), []
                ).append(keyword.value)
        return bound

    # Callable identity crosses the same ordinary dataflow edges as values.
    # Iterate call-site-to-parameter bindings to a fixed point so callbacks can
    # traverse wrappers, aliases, and mixed positional/keyword call shapes.
    changed = True
    while changed:
        changed = False
        completed: dict[
            ast.FunctionDef | ast.AsyncFunctionDef, dict[str, _StaticBinding]
        ] = {}

        def analyze(
            function: ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> dict[str, _StaticBinding]:
            cached = completed.get(function)
            if cached is not None:
                return cached
            parent = function_parents.get(function)
            inherited = (
                analyze(parent) if parent is not None else module_flow.bindings
            )
            bindings = dict(inherited)
            for name in cycle_bounded_helpers:
                if name in declarations:
                    bindings[name] = declarations[name]
            bindings.update({
                name: _semantic_binding(semantics, _CALLABLE_CLASS)
                for name, semantics in function_imported_callable_classes[
                    function
                ].items()
            })
            local_semantics: dict[str, set[str]] = {}
            for name in (
                function_imported_provenance_helpers.get(function, set())
                | function_accessor_aliases.get(function, set())
            ):
                local_semantics.setdefault(name, set()).add(
                    _CALLABLE_PROVENANCE
                )
            for name in function_control_imports.get(function, set()):
                local_semantics.setdefault(name, set()).add(_CALLABLE_CONTROL)
            for name, semantics in local_semantics.items():
                bindings[name] = _semantic_binding(
                    semantics, _CALLABLE_VALUE
                )
            parameters = {
                parameter.arg.casefold()
                for parameter in [
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                    *(
                        [function.args.vararg]
                        if function.args.vararg is not None
                        else []
                    ),
                    *(
                        [function.args.kwarg]
                        if function.args.kwarg is not None
                        else []
                    ),
                ]
            }
            for parameter in parameters:
                bindings.pop(parameter, None)
            for parameter, semantics in parameter_semantics[function].items():
                bindings[parameter] = _semantic_binding(
                    semantics, _CALLABLE_VALUE
                )
            flow = _CallableSemanticFlow(declarations, classes, bindings)
            flow.replay(function.body)
            completed[function] = dict(flow.bindings)
            return completed[function]

        for function in functions:
            analyze(function)

        for caller in functions:
            for call in _lexical_scope_calls(caller):
                source_names = _expanded_callable_sources(
                    _call_name(call), callable_alias_edges[caller]
                )
                for source_name in source_names:
                    for callee in functions_by_name.get(source_name, ()):
                        for parameter, arguments in bound_arguments(
                            call, callee
                        ).items():
                            semantics = {
                                semantic
                                for argument in arguments
                                for semantic in _callable_semantics(argument)
                            }
                            known = parameter_semantics[callee].setdefault(
                                parameter, set()
                            )
                            new_semantics = semantics - known
                            if new_semantics:
                                known.update(new_semantics)
                                changed = True


def _local_parameter_return_flows(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    function_imported_flows: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        dict[str, _ParameterReturnFlow],
    ] | None = None,
) -> dict[ast.FunctionDef | ast.AsyncFunctionDef, _ParameterReturnFlow]:
    """Summarize local parameter-to-return flow, including assignment aliases."""

    functions_by_name: dict[
        str, list[ast.FunctionDef | ast.AsyncFunctionDef]
    ] = {}
    for function in functions:
        functions_by_name.setdefault(function.name.casefold(), []).append(function)
    all_parameter_names = {
        function: {
            parameter.arg.casefold()
            for parameter in [
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
                *(
                    [function.args.vararg]
                    if function.args.vararg is not None
                    else []
                ),
                *(
                    [function.args.kwarg]
                    if function.args.kwarg is not None
                    else []
                ),
            ]
        }
        for function in functions
    }
    flowed_names = {function: set() for function in functions}
    summaries = {
        function: _parameter_return_flow(function, set()) for function in functions
    }
    callable_alias_edges = {
        function: _scope_callable_alias_edges(function)
        for function in functions
    }
    container_aliases = {
        function: _mutable_container_alias_snapshots(function)
        for function in functions
    }

    def call_flows(
        call: ast.Call,
        caller: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[_ParameterReturnFlow]:
        call_name = _call_name(call).casefold()
        source_names = _expanded_callable_sources(
            call_name,
            callable_alias_edges[caller],
        )
        flows = [
            summaries[callee]
            for source_name in source_names
            for callee in functions_by_name.get(source_name, ())
        ]
        imported_flows = (function_imported_flows or {}).get(caller, {})
        flows.extend(
            imported_flows[source_name]
            for source_name in source_names.intersection(imported_flows)
        )
        return flows

    def expression_flows_from_aliases(
        value: ast.AST,
        aliases: set[str],
        caller: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        while isinstance(value, (ast.Await, ast.Expr)):
            value = value.value
        if isinstance(value, ast.Name):
            return value.id.casefold() in aliases
        if isinstance(value, ast.Constant):
            return False
        if isinstance(value, ast.Call):
            if any(
                expression_flows_from_aliases(argument, aliases, caller)
                for flow in call_flows(value, caller)
                for argument in _bound_parameter_flow_arguments(value, flow)
            ):
                return True
            callable_sources = _expanded_callable_sources(
                _call_name(value).casefold(),
                callable_alias_edges[caller],
            )
            if callable_sources.intersection(all_parameter_names[caller]):
                # The callee is itself supplied by the caller.  A returned
                # invocation can therefore retain both the selected callback
                # and any value handed to it.  With no callable body available
                # in this scope, conservatively summarize all operands at the
                # same parameter-flow boundary used for ordinary wrappers.
                callback_inputs: list[ast.AST] = [
                    value.func,
                    *value.args,
                    *(keyword.value for keyword in value.keywords),
                ]
                if any(
                    expression_flows_from_aliases(argument, aliases, caller)
                    for argument in callback_inputs
                ):
                    return True
            if not (
                _is_provenance_transform_call(value)
                or _is_provenance_accessor_call(value)
            ):
                return False
            inputs: list[ast.AST] = [
                *value.args,
                *(keyword.value for keyword in value.keywords),
            ]
            if isinstance(value.func, ast.Attribute):
                inputs.append(value.func.value)
            return any(
                expression_flows_from_aliases(argument, aliases, caller)
                for argument in inputs
            )
        return any(
            expression_flows_from_aliases(child, aliases, caller)
            for child in ast.iter_child_nodes(value)
        )

    changed = True
    while changed:
        changed = False
        for function in functions:
            assignments: list[tuple[set[str], ast.AST]] = []
            returned_values: list[ast.AST] = []
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
                elif isinstance(node, (ast.For, ast.AsyncFor)):
                    targets = [node.target]
                    value = node.iter
                elif isinstance(node, ast.Call):
                    mutation = _mutable_container_write(node)
                    if mutation is not None:
                        target_names, value = mutation
                        target_names.update(
                            container_aliases[function].get(node, ())
                        )
                        assignments.append((target_names, value))
                        continue
                elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
                    if node.value is not None:
                        returned_values.append(node.value)
                if value is not None:
                    assignments.append(
                        (
                            {
                                name
                                for target in targets
                                for name in _reference_binding_names(target)
                            }
                            | set(container_aliases[function].get(node, ())),
                            value,
                        )
                    )

            discovered: set[str] = set()
            for parameter in all_parameter_names[function]:
                aliases = {parameter}
                alias_changed = True
                while alias_changed:
                    alias_changed = False
                    for targets, value in assignments:
                        if expression_flows_from_aliases(value, aliases, function):
                            new_aliases = targets - aliases
                            if new_aliases:
                                aliases.update(new_aliases)
                                alias_changed = True
                if any(
                    expression_flows_from_aliases(value, aliases, function)
                    for value in returned_values
                ):
                    discovered.add(parameter)
            new_names = discovered - flowed_names[function]
            if new_names:
                flowed_names[function].update(new_names)
                summaries[function] = _parameter_return_flow(
                    function,
                    flowed_names[function],
                )
                changed = True
    return summaries


def _local_provenance_return_helpers(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    control_helpers: set[str] | None = None,
    module_provenance_aliases: set[str] | None = None,
    imported_provenance_helpers: set[str] | None = None,
    function_initial_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] | None = None,
    function_imported_provenance_helpers: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] | None = None,
    function_imported_parameter_return_flows: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        dict[str, _ParameterReturnFlow],
    ] | None = None,
    function_parents: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        ast.FunctionDef | ast.AsyncFunctionDef,
    ] | None = None,
) -> set[str]:
    """Find visible helpers whose returned or yielded value is provenance-derived.

    Seed the fixed point with repository-local imported helper summaries so a
    neutral refactor such as ``derive(request)`` does not erase the fact that
    its result came from ``request.causation_chain``.  Local helpers may then
    wrap either imported or local helpers without escaping the contract.
    """

    functions_by_name: dict[
        str, list[ast.FunctionDef | ast.AsyncFunctionDef]
    ] = {}
    for function in functions:
        functions_by_name.setdefault(function.name.casefold(), []).append(function)
    callable_alias_edges = {
        function: _scope_callable_alias_edges(function)
        for function in functions
    }
    parameter_return_flows = _local_parameter_return_flows(
        functions,
        function_imported_parameter_return_flows,
    )

    def call_returns_supplied_provenance(
        call: ast.Call,
        aliases: set[str],
        visible_helpers: set[str],
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        source_names = _expanded_callable_sources(
            _call_name(call),
            callable_alias_edges[function],
        )
        flows = [
            parameter_return_flows[callee]
            for source_name in source_names
            for callee in functions_by_name.get(source_name, ())
        ]
        imported_flows = (function_imported_parameter_return_flows or {}).get(
            function, {}
        )
        flows.extend(
            imported_flows[source_name]
            for source_name in source_names.intersection(imported_flows)
        )
        return any(
            _has_provenance_value(argument, aliases, visible_helpers)
            for flow in flows
            for argument in _bound_parameter_flow_arguments(call, flow)
        )

    helper_names: set[str] = set(imported_provenance_helpers or ())
    changed = True
    while changed:
        changed = False
        lexical_aliases: dict[
            ast.FunctionDef | ast.AsyncFunctionDef, set[str]
        ] = {}

        def initial_aliases_for(
            function: ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> set[str]:
            """Carry enclosing provenance values into a closure summary."""

            cached = lexical_aliases.get(function)
            if cached is not None:
                return cached
            parent = (function_parents or {}).get(function)
            if parent is None:
                inherited = set(module_provenance_aliases or ())
            else:
                inherited = initial_aliases_for(parent)
                parent_helpers = helper_names | set(
                    (function_imported_provenance_helpers or {}).get(
                        parent, ()
                    )
                )
                inherited, _selected = _provenance_aliases(
                    parent,
                    parent_helpers,
                    control_helpers,
                    inherited
                    | set((function_initial_aliases or {}).get(parent, ())),
                    authority_analysis=False,
                )

            # A parameter is a new local binding, not a capture of the same-
            # named value in the enclosing function. Annotated/defaulted
            # provenance is reintroduced by ``_provenance_aliases`` itself.
            shadowed = {
                parameter.arg.casefold()
                for parameter in [
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                    *(
                        [function.args.vararg]
                        if function.args.vararg is not None
                        else []
                    ),
                    *(
                        [function.args.kwarg]
                        if function.args.kwarg is not None
                        else []
                    ),
                ]
            }
            resolved = (inherited - shadowed) | set(
                (function_initial_aliases or {}).get(function, ())
            )
            lexical_aliases[function] = resolved
            return resolved

        for function in functions:
            if function.name.casefold() in helper_names:
                continue
            visible_helpers = helper_names | set(
                (function_imported_provenance_helpers or {}).get(
                    function, ()
                )
            )
            aliases, _selected_targets = _provenance_aliases(
                function,
                visible_helpers,
                control_helpers,
                initial_aliases_for(function),
                authority_analysis=False,
            )
            returns_provenance = False
            for node in _walk_lexical_scope(function):
                if not isinstance(
                    node,
                    (ast.Return, ast.Yield, ast.YieldFrom),
                ) or node.value is None:
                    continue
                value = node.value
                while isinstance(value, ast.Await):
                    value = value.value
                if isinstance(value, ast.Call):
                    call_name = _call_name(value).casefold()
                    calls_known_helper = any(
                        isinstance(child, ast.Call)
                        and (
                            _call_name(child).casefold() in visible_helpers
                            or _CALLABLE_PROVENANCE
                            in _callable_semantics(child)
                        )
                        for child in ast.walk(value)
                    )
                    returns_provenance = (
                        _is_provenance_accessor_call(value)
                        or call_name in visible_helpers
                        or _CALLABLE_PROVENANCE in _callable_semantics(value)
                        or call_returns_supplied_provenance(
                            value,
                            aliases,
                            visible_helpers,
                            function,
                        )
                        or _is_provenance_transform_call(value)
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
                node, control_aliases, state_object_aliases
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

        def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
            if node.value is not None and is_control_reference(node.value):
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

    return _CALLABLE_CONTROL in _callable_semantics(node) or any(
        _is_unambiguous_control_token(source)
        for source in _control_reference_sources(node)
    )


def _causation_membership_polarity(
    condition: ast.AST,
    provenance_aliases: set[str],
    provenance_return_helpers: set[str] | None = None,
) -> str | None:
    """Return whether a target-membership test selects seen or unseen peers."""

    inverted = False
    while isinstance(condition, ast.UnaryOp) and isinstance(
        condition.op, ast.Not
    ):
        inverted = not inverted
        condition = condition.operand
    if (
        not isinstance(condition, ast.Compare)
        or len(condition.ops) != 1
        or len(condition.comparators) != 1
        or not isinstance(condition.ops[0], (ast.In, ast.NotIn))
    ):
        return None
    member = condition.left
    collection = condition.comparators[0]
    if _has_provenance_value(
        member, provenance_aliases, provenance_return_helpers
    ) or not _has_provenance_value(
        collection, provenance_aliases, provenance_return_helpers
    ):
        return None
    selects_seen = isinstance(condition.ops[0], ast.In) ^ inverted
    return "seen" if selects_seen else "unseen"


def _is_cycle_bounded_universal_call(call: ast.Call) -> bool:
    return _CALLABLE_CYCLE_BOUNDED in _callable_semantics(call.func)


def _contains_only_cycle_bounded_controls(
    nodes: ast.AST | list[ast.AST],
    control_aliases: set[str],
    state_object_aliases: set[str] | None = None,
) -> bool:
    """Whether every guarded authority sink is an explicit universal rail."""

    roots = tuple(nodes) if isinstance(nodes, list) else (nodes,)

    class CycleBoundedControlVisitor(ast.NodeVisitor):
        found_control = False
        invalid_control = False

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if _is_cross_agent_state_mutation_call(
                node, state_object_aliases
            ):
                self.invalid_control = True
                return
            if _is_cross_agent_control_call(
                node, control_aliases, state_object_aliases
            ):
                self.found_control = True
                if not _is_cycle_bounded_universal_call(node):
                    self.invalid_control = True
                    return
            self.generic_visit(node)

        def visit_FunctionDef(  # noqa: N802
            self, node: ast.FunctionDef
        ) -> None:
            return

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

    visitor = CycleBoundedControlVisitor()
    for root in roots:
        visitor.visit(root)
    return visitor.found_control and not visitor.invalid_control


def _if_only_suppresses_causation_cycle(
    statement: ast.If,
    provenance_aliases: set[str],
    control_aliases: set[str],
    provenance_return_helpers: set[str] | None = None,
    state_object_aliases: set[str] | None = None,
) -> bool:
    """Whether an if removes a repeated peer from a universal operation."""

    polarity = _causation_membership_polarity(
        statement.test,
        provenance_aliases,
        provenance_return_helpers,
    )
    if polarity == "unseen":
        permitted_branch, suppressed_branch = statement.body, statement.orelse
    elif polarity == "seen":
        permitted_branch, suppressed_branch = statement.orelse, statement.body
    else:
        return False
    return (
        _contains_only_cycle_bounded_controls(
            permitted_branch, control_aliases, state_object_aliases
        )
        and not _contains_cross_agent_control_call(
            suppressed_branch, control_aliases, state_object_aliases
        )
    )


def _try_flow_uses_provenance_as_control(
    statement: ast.Try | ast.TryStar,
    provenance_aliases: set[str],
    control_aliases: set[str],
    provenance_return_helpers: set[str] | None = None,
    state_object_aliases: set[str] | None = None,
) -> bool:
    """Whether try success/failure selects a control using provenance."""

    branch_controls = _contains_cross_agent_control_call(
        [
            *statement.orelse,
            *(child for handler in statement.handlers for child in handler.body),
        ],
        control_aliases,
        state_object_aliases,
    )
    suffix_controls = [False] * (len(statement.body) + 1)
    for index in range(len(statement.body) - 1, -1, -1):
        suffix_controls[index] = suffix_controls[index + 1] or (
            _contains_cross_agent_control_call(
                statement.body[index],
                control_aliases,
                state_object_aliases,
            )
        )
    return any(
        (suffix_controls[index] or branch_controls)
        and _has_provenance_value(
            body_statement,
            provenance_aliases,
            provenance_return_helpers,
        )
        for index, body_statement in enumerate(statement.body)
    )


_A2A_IDENTITY_PLUMBING_CALLS = frozenset(
    {
        "_a2a_inbound_current_scope_is_valid",
        "_a2a_inbound_requires_verified_sender",
        "_a2a_inbound_scope_snapshot",
        "_a2a_inbound_scope_unchanged",
        "_a2a_sender_witness_unchanged",
        "_authorize_verified_a2a_sender",
        "a2a_hosted_policy_for",
        "a2a_sender_identity_witness",
        "authorize_a2a_legacy_unsigned_sender",
        "authorize_legacy",
    }
)


def _authenticated_identity_selects_protected_control(
    roots: ast.AST | list[ast.AST],
    control_aliases: set[str],
    state_object_aliases: set[str],
) -> bool:
    """Whether authenticated identity directly selects a governed effect.

    Authentication and identity-stability plumbing may choose how target
    authorization is established. It may not itself choose an unrelated
    lifecycle or state mutation: key ownership is not relation authority.
    """

    class ProtectedControlVisitor(ast.NodeVisitor):
        found = False

        def visit_Call(self, call: ast.Call) -> None:  # noqa: N802
            if _call_name(call).casefold() in _A2A_IDENTITY_PLUMBING_CALLS:
                return
            if _is_cross_agent_control_call(
                call, control_aliases, state_object_aliases
            ) or _is_cross_agent_state_mutation_call(
                call, state_object_aliases
            ):
                self.found = True
                return
            self.generic_visit(call)

        def generic_visit(self, current: ast.AST) -> None:
            if self.found:
                return
            if _is_cross_agent_state_mutation_node(
                current, state_object_aliases
            ):
                self.found = True
                return
            super().generic_visit(current)

        def visit_FunctionDef(self, _node: ast.FunctionDef) -> None:  # noqa: N802
            return

        def visit_AsyncFunctionDef(  # noqa: N802
            self, _node: ast.AsyncFunctionDef
        ) -> None:
            return

        def visit_ClassDef(self, _node: ast.ClassDef) -> None:  # noqa: N802
            return

        def visit_Lambda(self, _node: ast.Lambda) -> None:  # noqa: N802
            return

    visitor = ProtectedControlVisitor()
    for root in roots if isinstance(roots, list) else [roots]:
        visitor.visit(root)
        if visitor.found:
            return True
    return False


def _is_fail_closed_envelope_acceptance_guard(
    node: ast.AST,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    local_attribution_authorizers: set[str],
    control_aliases: set[str],
    state_object_aliases: set[str],
) -> bool:
    """Whether a negative ``verdict.ok`` guard is dominated by sender auth."""

    if not (
        function.name.casefold() in local_attribution_authorizers
        and isinstance(node, ast.If)
        and not node.orelse
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and _block_guaranteed_function_exit(node.body)
    ):
        return False

    def contains_protected_control(
        statements: list[ast.stmt],
        *,
        allowed_calls: frozenset[str] = frozenset(),
    ) -> bool:
        """Find executed controls without treating callable wiring as one."""

        class ProtectedControlVisitor(ast.NodeVisitor):
            found = False

            def visit_Call(self, call: ast.Call) -> None:  # noqa: N802
                if _call_name(call).casefold() in (
                    _A2A_IDENTITY_PLUMBING_CALLS | allowed_calls
                ):
                    return
                if _is_cross_agent_control_call(
                    call, control_aliases, state_object_aliases
                ) or _is_cross_agent_state_mutation_call(
                    call, state_object_aliases
                ):
                    self.found = True
                    return
                self.generic_visit(call)

            def generic_visit(self, current: ast.AST) -> None:
                if self.found:
                    return
                if _is_cross_agent_state_mutation_node(
                    current, state_object_aliases
                ):
                    self.found = True
                    return
                super().generic_visit(current)

            def visit_FunctionDef(self, _node: ast.FunctionDef) -> None:  # noqa: N802
                return

            def visit_AsyncFunctionDef(  # noqa: N802
                self, _node: ast.AsyncFunctionDef
            ) -> None:
                return

            def visit_ClassDef(self, _node: ast.ClassDef) -> None:  # noqa: N802
                return

            def visit_Lambda(self, _node: ast.Lambda) -> None:  # noqa: N802
                return

        visitor = ProtectedControlVisitor()
        for statement in statements:
            visitor.visit(statement)
            if visitor.found:
                return True
        return False

    accepted_field = next(
        (
            child
            for child in ast.walk(node.test)
            if isinstance(child, ast.Attribute)
            and child.attr.casefold() == "ok"
            and getattr(child, "_authority_unverified_attribution", False)
        ),
        None,
    )
    if accepted_field is None or contains_protected_control(node.body):
        return False
    verdict_receiver = ast.unparse(accepted_field.value).casefold()

    def lexical_nodes(root: ast.AST) -> tuple[ast.AST, ...]:
        """Walk one statement while excluding deferred nested scopes."""

        nodes: list[ast.AST] = []

        class ScopeVisitor(ast.NodeVisitor):
            def generic_visit(self, current: ast.AST) -> None:
                nodes.append(current)
                super().generic_visit(current)

            def visit_FunctionDef(  # noqa: N802
                self, _node: ast.FunctionDef
            ) -> None:
                return

            def visit_AsyncFunctionDef(  # noqa: N802
                self, _node: ast.AsyncFunctionDef
            ) -> None:
                return

            def visit_ClassDef(self, _node: ast.ClassDef) -> None:  # noqa: N802
                return

            def visit_Lambda(self, _node: ast.Lambda) -> None:  # noqa: N802
                return

        visitor = ScopeVisitor()
        visitor.visit(root)
        return tuple(nodes)

    def authorization_result_names(
        statement: ast.AST,
        accepted_call_names: frozenset[str],
    ) -> set[str]:
        names: set[str] = set()
        for candidate in lexical_nodes(statement):
            targets: list[ast.AST] = []
            value: ast.AST | None = None
            if isinstance(candidate, (ast.Assign, ast.AnnAssign)):
                targets = (
                    list(candidate.targets)
                    if isinstance(candidate, ast.Assign)
                    else [candidate.target]
                )
                value = candidate.value
            elif isinstance(candidate, ast.NamedExpr):
                targets = [candidate.target]
                value = candidate.value
            if value is None:
                continue
            calls = [child for child in ast.walk(value) if isinstance(child, ast.Call)]
            if not any(
                _call_name(call).casefold() in accepted_call_names
                or getattr(call, "_authority_verified_attribution", False)
                for call in calls
            ):
                continue
            names.update(
                name.casefold()
                for target in targets
                for name in _binding_target_names(target)
            )
        return names

    def rejects_authorization_result(test: ast.AST, name: str) -> bool:
        """Whether truth of ``test`` proves ``name`` is unauthorized."""

        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return name in _identifier_tokens(test.operand)
        if isinstance(test, ast.Compare):
            values = [test.left, *test.comparators]
            return name in {
                token
                for value in values
                for token in _identifier_tokens(value)
            } and any(
                isinstance(value, ast.Constant)
                and value.value in {None, False, ""}
                for value in values
            )
        if isinstance(test, ast.BoolOp):
            checks = [
                rejects_authorization_result(value, name)
                for value in test.values
            ]
            # For OR, one disjunct that is necessarily true for an empty/false
            # result makes the whole rejection true. For AND, every conjunct
            # must reject it; otherwise another flag could bypass the guard.
            return any(checks) if isinstance(test.op, ast.Or) else all(checks)
        return False

    def branch_has_fail_closed_authorization(
        statements: list[ast.stmt],
        accepted_call_names: frozenset[str],
    ) -> bool:
        """Prove an authorizer result is checked before branch continuation."""

        result_names = {
            name
            for statement in statements
            for name in authorization_result_names(
                statement, accepted_call_names
            )
        }
        for name in result_names:
            assignment_lines = [
                getattr(candidate, "lineno", 0)
                for statement in statements
                for candidate in lexical_nodes(statement)
                if isinstance(
                    candidate, (ast.Assign, ast.AnnAssign, ast.NamedExpr)
                )
                and name
                in {
                    bound.casefold()
                    for target in (
                        list(candidate.targets)
                        if isinstance(candidate, ast.Assign)
                        else [candidate.target]
                    )
                    for bound in _binding_target_names(target)
                }
                and authorization_result_names(candidate, accepted_call_names)
            ]
            for statement in lexical_nodes(
                ast.Module(body=statements, type_ignores=[])
            ):
                if (
                    isinstance(statement, ast.If)
                    and rejects_authorization_result(statement.test, name)
                    and _block_guaranteed_function_exit(statement.body)
                    and assignment_lines
                    and max(assignment_lines) < statement.lineno
                ):
                    return True
        return False

    def branch_all_continuing_paths_authorized(
        statements: list[ast.stmt],
        accepted_call_names: frozenset[str],
    ) -> bool:
        """Conservatively prove every path past ``statements`` authorized."""

        def assignment_kind(
            assignment: ast.Assign | ast.AnnAssign | ast.NamedExpr,
            name: str,
        ) -> str | None:
            targets = (
                list(assignment.targets)
                if isinstance(assignment, ast.Assign)
                else [assignment.target]
            )
            if name not in {
                bound.casefold()
                for target in targets
                for bound in _binding_target_names(target)
            }:
                return None
            value = assignment.value
            if isinstance(value, ast.Constant) and value.value in {None, False, ""}:
                return "invalid"
            calls = [child for child in ast.walk(value) if isinstance(child, ast.Call)]
            if any(
                _call_name(call).casefold() in accepted_call_names
                or getattr(call, "_authority_verified_attribution", False)
                for call in calls
            ):
                return "authorizer"
            return "unknown"

        result_names = {
            name
            for statement in statements
            for name in authorization_result_names(
                statement, accepted_call_names
            )
        }
        for guard_index, guard in enumerate(statements):
            if not (
                isinstance(guard, ast.If)
                and _block_guaranteed_function_exit(guard.body)
            ):
                continue
            for name in result_names:
                if not rejects_authorization_result(guard.test, name):
                    continue
                kinds = [
                    kind
                    for statement in statements[:guard_index]
                    for candidate in lexical_nodes(statement)
                    if isinstance(
                        candidate, (ast.Assign, ast.AnnAssign, ast.NamedExpr)
                    )
                    if (kind := assignment_kind(candidate, name)) is not None
                ]
                if "authorizer" in kinds and set(kinds) <= {
                    "authorizer",
                    "invalid",
                }:
                    return True

        # A branch tree is complete only when each continuing arm authorizes;
        # an omitted ``else`` is a real unauthenticated continuation path.
        for statement in statements:
            if not isinstance(statement, ast.If) or not statement.orelse:
                continue
            body_safe = _block_guaranteed_function_exit(
                statement.body
            ) or branch_all_continuing_paths_authorized(
                statement.body, accepted_call_names
            )
            orelse_safe = _block_guaranteed_function_exit(
                statement.orelse
            ) or branch_all_continuing_paths_authorized(
                statement.orelse, accepted_call_names
            )
            if body_safe and orelse_safe:
                return True
        return _block_guaranteed_function_exit(statements)

    def authorization_partition_strength(statement: ast.stmt) -> str | None:
        if not isinstance(statement, ast.If):
            return None
        if not (
            isinstance(statement.test, ast.Attribute)
            and statement.test.attr.casefold() == "verified"
            and ast.unparse(statement.test.value).casefold()
            == verdict_receiver
            and getattr(
                statement.test, "_authority_authenticated_attribution", False
            )
        ):
            return None
        # Authentication is not relation authority. Each branch must invoke a
        # target authorizer and fail closed on its result before continuation.
        verified_call_names = frozenset({"_authorize_verified_a2a_sender"})
        legacy_call_names = frozenset(
            {
                "authorize_a2a_legacy_unsigned_sender",
                "authorize_legacy",
            }
        )
        if not (
            branch_has_fail_closed_authorization(
                statement.body,
                verified_call_names,
            )
            and branch_has_fail_closed_authorization(
                statement.orelse,
                legacy_call_names,
            )
            and not contains_protected_control(statement.body)
            and not contains_protected_control(statement.orelse)
        ):
            return None
        return (
            "strong"
            if branch_all_continuing_paths_authorized(
                statement.body, verified_call_names
            )
            and branch_all_continuing_paths_authorized(
                statement.orelse, legacy_call_names
            )
            else "a2a-only"
        )

    def containing_block(
        statements: list[ast.stmt],
    ) -> tuple[list[ast.stmt], int] | None:
        for index, statement in enumerate(statements):
            if statement is node:
                return statements, index
            for block in _child_statement_blocks(statement):
                found = containing_block(block)
                if found is not None:
                    return found
        return None

    location = containing_block(function.body)
    if location is None:
        return False
    statements, guard_index = location
    suffix = statements[guard_index + 1 :]
    partition = next(
        (
            (index, strength)
            for index, statement in enumerate(suffix)
            if (
                strength := authorization_partition_strength(statement)
            ) is not None
        ),
        None,
    )
    if partition is None:
        return False
    partition_index, partition_strength = partition
    return (
        not contains_protected_control(suffix[:partition_index])
        and (
            partition_strength == "strong"
            or not contains_protected_control(
                suffix[partition_index + 1 :],
                allowed_calls=frozenset({"commit", "create_task"}),
            )
        )
    )


def _guard_clause_provenance_lines(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    provenance_aliases: set[str],
    control_aliases: set[str],
    provenance_return_helpers: set[str] | None = None,
    state_object_aliases: set[str] | None = None,
    local_attribution_authorizers: set[str] | None = None,
) -> set[int]:
    """Find provenance conditions that gate a later control by exiting early."""

    lines: set[int] = set()

    def scan_block(
        statements: list[ast.stmt],
        enclosing_continuation_controls: bool = False,
        enclosing_loop_continuation_controls: bool = False,
    ) -> None:
        suffix_controls = [False] * (len(statements) + 1)
        for index in range(len(statements) - 1, -1, -1):
            suffix_controls[index] = suffix_controls[index + 1] or (
                _contains_cross_agent_control_call(
                    statements[index], control_aliases, state_object_aliases
                )
            )
        for index, statement in enumerate(statements):
            local_continuation_controls = (
                suffix_controls[index + 1]
                or enclosing_continuation_controls
            )
            controls_continuation = (
                local_continuation_controls
                or enclosing_loop_continuation_controls
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
                body_function_exits = _block_guaranteed_function_exit(
                    statement.body
                )
                orelse_function_exits = _block_guaranteed_function_exit(
                    statement.orelse
                )
                if enclosing_loop_continuation_controls:
                    body_function_exits = (
                        _block_then_suffix_guaranteed_function_exit(
                            statement.body,
                            statements[index + 1 :],
                        )
                    )
                    orelse_function_exits = (
                        _block_then_suffix_guaranteed_function_exit(
                            statement.orelse,
                            statements[index + 1 :],
                        )
                    )
                membership_polarity = _causation_membership_polarity(
                    statement.test,
                    provenance_aliases,
                    provenance_return_helpers,
                )
                seen_branch_exits = (
                    body_exits
                    if membership_polarity == "seen"
                    else orelse_exits
                    if membership_polarity == "unseen"
                    else False
                )
                unseen_branch_exits = (
                    orelse_exits
                    if membership_polarity == "seen"
                    else body_exits
                    if membership_polarity == "unseen"
                    else False
                )
                only_suppresses_cycle = (
                    suffix_controls[index + 1]
                    and not enclosing_continuation_controls
                    and not enclosing_loop_continuation_controls
                    and seen_branch_exits
                    and not unseen_branch_exits
                    and not _contains_cross_agent_control_call(
                        [*statement.body, *statement.orelse],
                        control_aliases,
                        state_object_aliases,
                    )
                    and _contains_only_cycle_bounded_controls(
                        statements[index + 1 :],
                        control_aliases,
                        state_object_aliases,
                    )
                )
                if (
                    (
                        local_continuation_controls
                        and body_exits != orelse_exits
                    )
                    or (
                        enclosing_loop_continuation_controls
                        and body_function_exits != orelse_function_exits
                    )
                ) and _has_provenance_value(
                    statement.test,
                    provenance_aliases,
                    provenance_return_helpers,
                ) and not only_suppresses_cycle and not (
                    _is_fail_closed_envelope_acceptance_guard(
                        statement,
                        function,
                        local_attribution_authorizers or set(),
                        control_aliases,
                        state_object_aliases or set(),
                    )
                ):
                    lines.add(statement.lineno)
            elif controls_continuation and isinstance(statement, ast.Match):
                has_catch_all = any(
                    case.guard is None
                    and isinstance(case.pattern, ast.MatchAs)
                    and case.pattern.pattern is None
                    for case in statement.cases
                )
                case_exit_sets = []
                if local_continuation_controls:
                    case_exit_sets.append([
                        _block_guaranteed_exits(case.body)
                        for case in statement.cases
                    ])
                if enclosing_loop_continuation_controls:
                    case_exit_sets.append([
                        _block_then_suffix_guaranteed_function_exit(
                            case.body,
                            statements[index + 1 :],
                        )
                        for case in statement.cases
                    ])
                governing_exit_sets = [
                    case_exits
                    for case_exits in case_exit_sets
                    if any(case_exits)
                    and (not has_catch_all or not all(case_exits))
                ]
                if governing_exit_sets:
                    if _has_provenance_value(
                        statement.subject,
                        provenance_aliases,
                        provenance_return_helpers,
                    ):
                        lines.add(statement.lineno)
                    lines.update(
                        case.guard.lineno
                        for case_index, case in enumerate(statement.cases)
                        if any(
                            case_exits[case_index]
                            for case_exits in governing_exit_sets
                        )
                        and case.guard is not None
                        and _has_provenance_value(
                            case.guard,
                            provenance_aliases,
                            provenance_return_helpers,
                        )
                    )
            elif controls_continuation and isinstance(
                statement, (ast.Try, ast.TryStar)
            ):
                try_exits = _block_guaranteed_exits(statement.body)
                handler_exits = [
                    _block_guaranteed_exits(handler.body)
                    for handler in statement.handlers
                ]
                try_function_exits = _block_guaranteed_function_exit(
                    statement.body
                )
                handler_function_exits = [
                    _block_guaranteed_function_exit(handler.body)
                    for handler in statement.handlers
                ]
                if enclosing_loop_continuation_controls:
                    try_function_exits = (
                        _block_then_suffix_guaranteed_function_exit(
                            statement.body,
                            statements[index + 1 :],
                        )
                    )
                    handler_function_exits = [
                        _block_then_suffix_guaranteed_function_exit(
                            handler.body,
                            statements[index + 1 :],
                        )
                        for handler in statement.handlers
                    ]
                if (
                    handler_exits
                    and (
                        local_continuation_controls
                        and any(
                            exit_state != try_exits
                            for exit_state in handler_exits
                        )
                        or enclosing_loop_continuation_controls
                        and any(
                            exit_state != try_function_exits
                            for exit_state in handler_function_exits
                        )
                    )
                    and any(
                        _has_provenance_value(
                            node,
                            provenance_aliases,
                            provenance_return_helpers,
                        )
                        for node in statement.body
                    )
                ):
                    lines.add(statement.lineno)
            elif controls_continuation and isinstance(
                statement, (ast.For, ast.AsyncFor, ast.While)
            ):
                decision = (
                    statement.iter
                    if isinstance(statement, (ast.For, ast.AsyncFor))
                    else statement.test
                )
                guards_continuation = isinstance(
                    statement, ast.While
                ) or _loop_else_guards_continuation(statement)
                if guards_continuation and _has_provenance_value(
                    decision,
                    provenance_aliases,
                    provenance_return_helpers,
                ):
                    lines.add(statement.lineno)
            # Loop-local exits need their own continuation state: ``break`` may
            # reach a control after the loop while ``return``/``raise`` cannot.
            # Other compound statements preserve both ordinary sibling and
            # enclosing-loop continuation dependencies.
            is_loop = isinstance(statement, (ast.For, ast.AsyncFor, ast.While))
            child_continuation_controls = (
                False if is_loop else local_continuation_controls
            )
            child_loop_continuation_controls = (
                (
                    local_continuation_controls
                    or enclosing_loop_continuation_controls
                )
                if is_loop
                else enclosing_loop_continuation_controls
            )
            for block in _child_statement_blocks(statement):
                scan_block(
                    block,
                    child_continuation_controls,
                    child_loop_continuation_controls,
                )

    scan_block(function.body)
    return lines


def _lexical_scope_imports(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.Import | ast.ImportFrom, ...]:
    """Return imports executed in exactly one lexical scope.

    Module imports are limited to the module body. Function-local imports may
    live under ordinary compound statements, but imports in nested functions
    or classes belong to those child scopes and are deliberately excluded.
    """

    nodes: tuple[ast.AST, ...] = (
        tuple(scope.body)
        if isinstance(scope, ast.Module)
        else _walk_lexical_scope(scope)
    )
    return tuple(
        node for node in nodes if isinstance(node, (ast.Import, ast.ImportFrom))
    )


def _lexical_scope_calls(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.Call]:
    """Return calls that can consume bindings visible in one import scope."""

    nodes: tuple[ast.AST, ...] = (
        tuple(ast.walk(scope))
        if isinstance(scope, ast.Module)
        else _walk_lexical_scope(scope)
    )
    return [node for node in nodes if isinstance(node, ast.Call)]


def _scope_imported_control_aliases(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
    source_path: Path | None = None,
    seen: frozenset[tuple[Path, str]] | None = None,
    calls: list[ast.Call] | None = None,
) -> set[str]:
    """Return control-callable imports visible within one lexical scope."""

    import_nodes = _lexical_scope_imports(scope)

    def is_control_callable(name: str) -> bool:
        # Import aliasing must recognize exactly the same control vocabulary
        # as direct calls.  A second verb list inevitably lets ``as apply``
        # erase whichever lifecycle action was added only to the main matcher.
        return _is_unambiguous_control_token(name)

    aliases = {
        (imported.asname or imported.name).casefold()
        for node in import_nodes
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

        for statement in scope.body:
            ModuleAssignmentVisitor().visit(statement)
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
    visible_calls = calls if calls is not None else _lexical_scope_calls(scope)
    called_names = {_call_name(call).casefold() for call in visible_calls}
    for node in import_nodes:
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
                        visible_calls,
                        (imported.asname or imported.name).casefold(),
                        scope if isinstance(scope, ast.Module) else None,
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
                member_calls = _module_attribute_call_names(
                    visible_calls,
                    qualifier,
                    scope if isinstance(scope, ast.Module) else None,
                )
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
                module_calls = _module_attribute_call_names(
                    visible_calls,
                    bound_name,
                    scope if isinstance(scope, ast.Module) else None,
                )
                aliases.update(
                    _repository_control_helper_names(
                        imported_path,
                        module_calls,
                        seen,
                    )
                )
    return include_module_assignment_aliases()


def _module_imported_control_aliases(
    tree: ast.AST,
    source_path: Path | None = None,
    seen: frozenset[tuple[Path, str]] | None = None,
    calls: list[ast.Call] | None = None,
) -> set[str]:
    """Return control imports visible to every function in one module."""

    if not isinstance(tree, ast.Module):
        return set()
    return _scope_imported_control_aliases(tree, source_path, seen, calls)


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


def _module_imported_provenance_annotation_aliases(tree: ast.AST) -> set[str]:
    """Return local aliases for imported causation/provenance annotation types."""

    if not isinstance(tree, ast.Module):
        return set()
    return {
        (imported.asname or imported.name).casefold()
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for imported in node.names
        if imported.name != "*"
        and _has_provenance_token(ast.Name(id=imported.name))
    }


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


def _state_alias_root(alias: str) -> str:
    """Return the lexical binding that owns an attribute/container alias."""

    return alias.casefold().split(".", 1)[0].split("[", 1)[0]


def _scope_local_binding_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """Return names whose Python binding belongs to exactly this function."""

    names = {
        parameter.arg.casefold()
        for parameter in [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
            *(
                [function.args.vararg]
                if function.args.vararg is not None
                else []
            ),
            *(
                [function.args.kwarg]
                if function.args.kwarg is not None
                else []
            ),
        ]
    }
    external = {
        name.casefold()
        for node in _walk_lexical_scope(function)
        if isinstance(node, (ast.Global, ast.Nonlocal))
        for name in node.names
    }
    for node in _walk_lexical_scope(function):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            targets = [
                item.optional_vars
                for item in node.items
                if item.optional_vars is not None
            ]
        names.update(
            name.casefold()
            for target in targets
            for name in _assignment_target_names(target)
        )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                (imported.asname or imported.name.split(".", 1)[0]).casefold()
                for imported in node.names
                if imported.name != "*"
            )
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            names.add(node.name.casefold())
    return names - external


def _module_shared_binding_names(tree: ast.AST) -> set[str]:
    """Return bindings owned by the module rather than a nested scope."""

    if not isinstance(tree, ast.Module):
        return set()
    scope = ast.parse("def __audit_module_bindings__():\n    pass\n").body[0]
    assert isinstance(scope, ast.FunctionDef)
    scope.body = tree.body
    return _scope_local_binding_names(scope)


def _scope_mutable_container_bindings(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """Resolve bindings proven to hold a mutable container in one scope."""

    container = ("object-kind", "mutable-container")

    def direct(value: ast.AST) -> _StaticBinding | None:
        if isinstance(value, (ast.Dict, ast.List, ast.Set)) or (
            isinstance(value, ast.Call)
            and _call_name(value).casefold()
            in {"defaultdict", "deque", "dict", "list", "set"}
        ):
            return container
        return None

    flow = _StaticBindingFlow(
        direct,
        lambda bindings: container,
        normalize_name=str.casefold,
    )
    flow.replay(function.body)
    return {
        name for name, binding in flow.bindings.items() if binding == container
    }


def _lexical_provenance_state_aliases(
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
    function_imported_provenance_helpers: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] | None = None,
) -> dict[ast.FunctionDef | ast.AsyncFunctionDef, set[str]]:
    """Share provenance through Python closure cells and captured containers."""

    local_bindings = {
        function: _scope_local_binding_names(function)
        for function in functions
    }

    def binding_owner(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        root: str,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        current: ast.FunctionDef | ast.AsyncFunctionDef | None = function
        while current is not None:
            if root in local_bindings[current]:
                return current
            current = function_parents.get(current)
        return None

    captured_cells: set[
        tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]
    ] = set()
    for function in functions:
        ancestor = function_parents.get(function)
        while ancestor is not None:
            for root in local_bindings[ancestor]:
                if binding_owner(function, root) is ancestor:
                    captured_cells.add((ancestor, root))
            ancestor = function_parents.get(ancestor)
    mutable_cells = {
        (owner, root)
        for owner, root in captured_cells
        if root in _scope_mutable_container_bindings(owner)
    }

    shared_by_owner: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] = {}

    def visible_shared(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> set[str]:
        visible: set[str] = set()
        for owner, aliases in shared_by_owner.items():
            for alias in aliases:
                root = _state_alias_root(alias)
                if function is owner or binding_owner(function, root) is owner:
                    visible.add(alias)
        return visible

    changed = True
    while changed:
        changed = False
        for function in functions:
            resolved, _selected = _provenance_aliases(
                function,
                provenance_return_helpers
                | set(
                    (function_imported_provenance_helpers or {}).get(
                        function, ()
                    )
                ),
                initial_aliases=(
                    module_provenance_aliases
                    | function_initial_aliases.get(function, set())
                    | visible_shared(function)
                ),
                authority_analysis=False,
                provenance_accessor_aliases=function_accessor_aliases.get(
                    function, set()
                ),
                shared_state_bindings={
                    root
                    for owner, root in mutable_cells
                    if function is owner
                    or binding_owner(function, root) is owner
                },
            )
            for alias in resolved:
                root = _state_alias_root(alias)
                owner = binding_owner(function, root)
                if owner is None or (owner, root) not in captured_cells:
                    continue
                owner_aliases = shared_by_owner.setdefault(owner, set())
                if alias not in owner_aliases:
                    owner_aliases.add(alias)
                    changed = True

    return {
        function: visible_shared(function)
        for function in functions
    }


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
    function_imported_provenance_helpers: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] | None = None,
    module_shared_bindings: set[str] | None = None,
    module_mutable_bindings: set[str] | None = None,
) -> set[str]:
    """Share provenance stored through module bindings and mutable objects."""

    aliases = set(module_provenance_aliases)
    local_bindings = {
        function: _scope_local_binding_names(function)
        for function in functions
    }

    def resolves_module_binding(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        root: str,
        declared_globals: set[str],
    ) -> bool:
        if root in declared_globals:
            return True
        if root not in (module_shared_bindings or set()):
            return False
        current: ast.FunctionDef | ast.AsyncFunctionDef | None = function
        while current is not None:
            if root in local_bindings[current]:
                return False
            current = function_parents.get(current)
        return True

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
                provenance_return_helpers
                | set(
                    (function_imported_provenance_helpers or {}).get(
                        function, ()
                    )
                ),
                initial_aliases=(
                    aliases
                    | inherited
                    | function_initial_aliases.get(function, set())
                ),
                authority_analysis=False,
                provenance_accessor_aliases=function_accessor_aliases.get(
                    function, set()
                ),
                shared_state_bindings=set(module_mutable_bindings or ()),
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
            discovered = {
                alias
                for alias in analyze(function)
                if resolves_module_binding(
                    function,
                    _state_alias_root(alias),
                    declared_globals,
                )
            }
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


@lru_cache(maxsize=None)
def _scope_repository_import_paths(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
    source_path: Path,
) -> frozenset[Path]:
    """Resolve referenced repository imports in exactly one lexical scope."""

    scope_nodes: tuple[ast.AST, ...] = (
        tuple(ast.walk(scope))
        if isinstance(scope, ast.Module)
        else _walk_lexical_scope(scope)
    )
    referenced_names = {
        node.id.casefold()
        for node in scope_nodes
        if isinstance(node, ast.Name)
    }
    paths: set[Path] = set()
    for node in _lexical_scope_imports(scope):
        candidates: list[tuple[str | None, int]] = []
        if isinstance(node, ast.Import):
            candidates.extend(
                (imported.name, 0)
                for imported in node.names
                if (
                    imported.asname or imported.name.split(".", 1)[0]
                ).casefold()
                in referenced_names
            )
        elif isinstance(node, ast.ImportFrom):
            referenced_imports = [
                imported
                for imported in node.names
                if imported.name == "*"
                or (imported.asname or imported.name).casefold()
                in referenced_names
            ]
            if referenced_imports:
                candidates.append((node.module, node.level))
            candidates.extend(
                (
                    ".".join(
                        part
                        for part in (node.module, imported.name)
                        if part
                    ),
                    node.level,
                )
                for imported in referenced_imports
                if imported.name != "*"
            )
        for module_name, level in candidates:
            imported_path = _resolved_repository_import_path(
                source_path,
                module_name,
                level,
            )
            if imported_path is not None:
                paths.add(imported_path)
    return frozenset(paths)


@lru_cache(maxsize=None)
def _repository_import_paths(source_path: Path) -> frozenset[Path]:
    """Return repository modules reachable by one static import edge."""

    source_path = source_path.resolve()
    tree = _parsed_module(source_path)
    return _scope_repository_import_paths(tree, source_path)


def _repository_reexport_bindings(
    source_path: Path,
    requested_names: set[str],
) -> tuple[tuple[str, Path, str], ...]:
    """Resolve requested facade names to their repository import bindings.

    Python exposes module-level imports as module attributes whether or not the
    facade consumes them locally.  Repository helper summaries therefore need
    to follow both direct ``from x import y`` exports and simple module-level
    aliases of those imports.
    """

    source_path = source_path.resolve()
    requested_names = {name.casefold() for name in requested_names}
    tree = _parsed_module(source_path)
    imported_bindings: dict[str, list[tuple[Path, str]]] = {}
    star_imports: list[Path] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        imported_path = _resolved_repository_import_path(
            source_path,
            node.module,
            node.level,
        )
        if imported_path is None:
            continue
        for imported in node.names:
            if imported.name == "*":
                star_imports.append(imported_path)
                continue
            local_name = (imported.asname or imported.name).casefold()
            imported_bindings.setdefault(local_name, []).append(
                (imported_path, imported.name.casefold())
            )

    alias_edges: list[tuple[str, str]] = []
    for node in tree.body:
        assignment_pairs: list[tuple[str, ast.AST]] = []
        if isinstance(node, ast.Assign):
            assignment_pairs = [
                pair
                for target in node.targets
                for pair in _static_assignment_pairs(target, node.value)
            ]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            assignment_pairs = _static_assignment_pairs(node.target, node.value)
        alias_edges.extend(
            (target.casefold(), value.id.casefold())
            for target, value in assignment_pairs
            if isinstance(value, ast.Name)
        )

    resolved: list[tuple[str, Path, str]] = []
    for requested_name in requested_names:
        candidate_bindings = {requested_name}
        changed = True
        while changed:
            changed = False
            for target, source in alias_edges:
                if target in candidate_bindings and source not in candidate_bindings:
                    candidate_bindings.add(source)
                    changed = True
        resolved.extend(
            (requested_name, imported_path, remote_name)
            for candidate in candidate_bindings
            for imported_path, remote_name in imported_bindings.get(candidate, ())
        )
        resolved.extend(
            (requested_name, imported_path, requested_name)
            for imported_path in star_imports
        )
    return tuple(resolved)


@lru_cache(maxsize=None)
def _source_references_repository_reexport(source_path: Path) -> bool:
    """Whether invoked imports enter a repository facade re-export edge."""

    source_path = source_path.resolve()
    tree = _parsed_module(source_path)
    visible_calls = _lexical_scope_calls(tree)
    called_names = {_call_name(call).casefold() for call in visible_calls}
    for node in _lexical_scope_imports(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_path = _resolved_repository_import_path(
                source_path,
                node.module,
                node.level,
            )
            if imported_path is None:
                continue
            requested = {
                imported.name.casefold()
                for imported in node.names
                if imported.name != "*"
                and (imported.asname or imported.name).casefold() in called_names
            }
            if any(imported.name == "*" for imported in node.names):
                requested.update(called_names)
            if _repository_reexport_bindings(imported_path, requested):
                return True
        elif isinstance(node, ast.ImportFrom):
            for imported in node.names:
                child_path = _resolved_repository_import_path(
                    source_path,
                    imported.name,
                    node.level,
                )
                if child_path is None:
                    continue
                member_calls = _module_attribute_call_names(
                    visible_calls,
                    (imported.asname or imported.name).casefold(),
                    tree,
                )
                if _repository_reexport_bindings(child_path, member_calls):
                    return True
        elif isinstance(node, ast.Import):
            for imported in node.names:
                imported_path = _resolved_repository_import_path(
                    source_path,
                    imported.name,
                )
                if imported_path is None:
                    continue
                member_calls = _module_attribute_call_names(
                    visible_calls,
                    (imported.asname or imported.name).casefold(),
                    tree,
                )
                if _repository_reexport_bindings(imported_path, member_calls):
                    return True
    return False


@lru_cache(maxsize=None)
def _source_or_imports_may_expose_provenance(source_path: Path) -> bool:
    """Cheaply over-approximate provenance reachability through imports."""

    pending = [source_path.resolve()]
    seen: set[Path] = set()
    while pending:
        candidate = pending.pop()
        if candidate in seen:
            continue
        seen.add(candidate)
        source = _source_text(candidate)
        if _source_text_may_expose_provenance(source):
            return True
        pending.extend(_repository_import_paths(candidate) - seen)
    return False


def _scope_or_imports_may_expose_provenance(
    scope: ast.FunctionDef | ast.AsyncFunctionDef,
    source_path: Path | None,
) -> bool:
    """Bound deep helper resolution to scopes that can consume provenance."""

    if _has_provenance_token(scope):
        return True
    if source_path is None:
        return False
    return any(
        _source_or_imports_may_expose_provenance(imported_path)
        for imported_path in _scope_repository_import_paths(
            scope,
            source_path.resolve(),
        )
    )


def _scope_contains_provenance_marker(
    scope: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Whether one lexical function scope directly names provenance."""

    return _scope_or_imports_may_expose_provenance(scope, None)


@lru_cache(maxsize=None)
def _function_local_imports_expose_provenance(source_path: Path) -> bool:
    """Check only invoked local imports that resolve as provenance helpers."""

    source_path = source_path.resolve()
    tree = _parsed_module(source_path)
    return any(
        _scope_imported_provenance_return_helper_aliases(
            function,
            source_path,
        )
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _module_attribute_call_names(
    calls: list[ast.Call],
    bound_name: str,
    tree: ast.Module | None = None,
) -> set[str]:
    """Return terminal call names invoked through one imported module binding."""

    bound_name = bound_name.casefold()
    qualifiers = {bound_name}
    if tree is not None:
        qualifiers = _static_alias_closure(
            qualifiers,
            [
                (
                    target.casefold(),
                    {source.casefold() for source in sources},
                )
                for target, sources in _static_alias_edges(tree)
            ],
        )
    names: set[str] = set()
    for call in calls:
        member_reference = _static_member_reference(call.func)
        if member_reference is None:
            continue
        receiver, member = member_reference
        receiver_name = ast.unparse(receiver).casefold()
        if any(
            receiver_name == qualifier
            or receiver_name.startswith(f"{qualifier}.")
            for qualifier in qualifiers
        ):
            names.add(member.casefold())
    return names


def _functions_reachable_by_local_calls(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    requested_names: set[str],
) -> set[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Bound import resolution to requested helpers and their local callees."""

    by_name: dict[
        str, set[ast.FunctionDef | ast.AsyncFunctionDef]
    ] = {}
    for function in functions:
        by_name.setdefault(function.name.casefold(), set()).add(function)
    reachable = {
        function
        for name in requested_names
        for function in by_name.get(name.casefold(), ())
    }
    pending = list(reachable)
    while pending:
        function = pending.pop()
        callable_alias_edges = _scope_callable_alias_edges(function)
        called_names = {
            source_name
            for call in _lexical_scope_calls(function)
            for source_name in _expanded_callable_sources(
                _call_name(call),
                callable_alias_edges,
            )
        }
        for name in called_names:
            for callee in by_name.get(name, ()):
                if callee not in reachable:
                    reachable.add(callee)
                    pending.append(callee)
    return reachable


@lru_cache(maxsize=None)
def _direct_control_helper_names(source_path: Path) -> frozenset[str]:
    """Summarize control helpers defined within one repository module."""

    tree = _parsed_module(source_path.resolve())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    # "Direct" summaries must not recursively traverse imports: repository
    # resolution layers the cycle-aware import graph on top of this result.
    module_aliases = _module_imported_control_aliases(tree)
    function_aliases = _function_control_import_aliases(
        tree,
        functions,
        module_aliases,
        None,
    )
    control_helpers = _local_control_helpers(
        functions,
        module_aliases,
        function_aliases,
    )
    return frozenset(
        control_helpers
        | _local_control_return_helpers(
            functions,
            control_helpers,
            function_aliases,
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
    for local_name, imported_path, remote_name in _repository_reexport_bindings(
        source_path,
        requested_names - resolved,
    ):
        if remote_name in _repository_control_helper_names(
            imported_path,
            {remote_name},
            active,
        ):
            resolved.add(local_name)
    if resolved == requested_names:
        return resolved

    tree = _parsed_module(source_path)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    relevant_functions = _functions_reachable_by_local_calls(
        functions,
        requested_names,
    )
    relevant_calls = [
        call
        for function in relevant_functions
        for call in _lexical_scope_calls(function)
    ]
    imported_helpers = _module_imported_control_aliases(
        tree,
        source_path,
        active,
        relevant_calls,
    )
    all_helpers = _local_control_helpers(
        functions,
        imported_helpers,
        _function_control_import_aliases(
            tree,
            functions,
            imported_helpers,
            source_path,
            seen=active,
            eligible_functions=relevant_functions,
        ),
    )
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
        | _module_imported_provenance_annotation_aliases(tree)
        | _module_imported_provenance_accessor_aliases(tree)
    )
    module_control_aliases = _module_imported_control_aliases(tree)
    control_helpers = _local_control_helpers(
        functions,
        module_control_aliases,
        _function_control_import_aliases(
            tree,
            functions,
            module_control_aliases,
            None,
        ),
    )
    imported_provenance_helpers: set[str] = set()
    return frozenset(
        _local_provenance_return_helpers(
            functions,
            control_helpers,
            module_provenance_aliases,
            imported_provenance_helpers,
            function_imported_provenance_helpers=(
                _function_provenance_helper_import_aliases(
                    tree,
                    functions,
                    imported_provenance_helpers,
                    None,
                )
            ),
            function_parents=_nested_function_parents(tree),
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
    for local_name, imported_path, remote_name in _repository_reexport_bindings(
        source_path,
        requested_names - resolved,
    ):
        if remote_name in _repository_provenance_helper_names(
            imported_path,
            {remote_name},
            active,
        ):
            resolved.add(local_name)
    if resolved == requested_names:
        return resolved

    tree = _parsed_module(source_path)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    relevant_functions = _functions_reachable_by_local_calls(
        functions,
        requested_names,
    )
    relevant_calls = [
        call
        for function in relevant_functions
        for call in _lexical_scope_calls(function)
    ]
    imported_helpers = _module_imported_provenance_return_helper_aliases(
        tree,
        source_path,
        active,
        relevant_calls,
    )
    module_provenance_aliases = (
        _module_provenance_constant_aliases(tree, source_path)
        | _module_imported_provenance_annotation_aliases(tree)
        | _module_imported_provenance_accessor_aliases(tree)
    )
    module_control_aliases = _module_imported_control_aliases(
        tree,
        source_path,
        active,
        relevant_calls,
    )
    control_helpers = _local_control_helpers(
        functions,
        module_control_aliases,
        _function_control_import_aliases(
            tree,
            functions,
            module_control_aliases,
            source_path,
            seen=active,
            eligible_functions=relevant_functions,
        ),
    )
    module_parameter_flows = _scope_imported_parameter_return_flows(
        tree,
        source_path,
        active,
        relevant_calls,
    )
    function_parameter_flows = _function_parameter_return_flow_imports(
        tree,
        functions,
        module_parameter_flows,
        source_path,
        seen=active,
        eligible_functions=relevant_functions,
    )
    all_helpers = _local_provenance_return_helpers(
        functions,
        control_helpers,
        module_provenance_aliases,
        imported_helpers,
        function_imported_provenance_helpers=(
            _function_provenance_helper_import_aliases(
                tree,
                functions,
                imported_helpers,
                source_path,
                seen=active,
                eligible_functions=relevant_functions,
            )
        ),
        function_imported_parameter_return_flows=function_parameter_flows,
        function_parents=_nested_function_parents(tree),
    )
    return requested_names.intersection(all_helpers)


def _merge_parameter_return_flows(
    flows: list[_ParameterReturnFlow],
) -> _ParameterReturnFlow:
    """Conservatively combine same-name summaries from ambiguous definitions."""

    vararg_starts = [
        flow.vararg_from for flow in flows if flow.vararg_from is not None
    ]
    accepted = (
        set.intersection(*(set(flow.accepted_keywords) for flow in flows))
        if flows
        else set()
    )
    return _ParameterReturnFlow(
        positional=frozenset(
            index for flow in flows for index in flow.positional
        ),
        keywords=frozenset(name for flow in flows for name in flow.keywords),
        vararg_from=min(vararg_starts) if vararg_starts else None,
        kwarg=any(flow.kwarg for flow in flows),
        accepted_keywords=frozenset(accepted),
        implicit_receiver=any(flow.implicit_receiver for flow in flows),
    )


def _merged_parameter_flow_map(
    entries: list[tuple[str, _ParameterReturnFlow]],
) -> dict[str, _ParameterReturnFlow]:
    grouped: dict[str, list[_ParameterReturnFlow]] = {}
    for name, flow in entries:
        grouped.setdefault(name.casefold(), []).append(flow)
    merged = {
        name: _merge_parameter_return_flows(flows)
        for name, flows in grouped.items()
    }
    return {
        name: flow
        for name, flow in merged.items()
        if flow.positional
        or flow.keywords
        or flow.vararg_from is not None
        or flow.kwarg
    }


@lru_cache(maxsize=None)
def _direct_parameter_return_flow_summaries(
    source_path: Path,
) -> tuple[tuple[str, _ParameterReturnFlow], ...]:
    """Summarize direct parameter-to-return flow without traversing imports."""

    tree = _parsed_module(source_path.resolve())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    flows = _local_parameter_return_flows(functions)
    merged = _merged_parameter_flow_map(
        [(function.name, flows[function]) for function in functions]
    )
    return tuple(sorted(merged.items()))


def _local_callback_invocation_flows(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    imported_flows: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        dict[str, _ParameterReturnFlow],
    ]
    | None = None,
) -> dict[ast.FunctionDef | ast.AsyncFunctionDef, _ParameterReturnFlow]:
    """Summarize formal parameters eventually invoked as callbacks."""

    functions_by_name: dict[
        str, list[ast.FunctionDef | ast.AsyncFunctionDef]
    ] = {}
    for function in functions:
        functions_by_name.setdefault(function.name.casefold(), []).append(function)
    parameters = {
        function: {
            parameter.arg.casefold()
            for parameter in [
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
                *(
                    [function.args.vararg]
                    if function.args.vararg is not None
                    else []
                ),
                *(
                    [function.args.kwarg]
                    if function.args.kwarg is not None
                    else []
                ),
            ]
        }
        for function in functions
    }
    invoked: dict[ast.FunctionDef | ast.AsyncFunctionDef, set[str]] = {
        function: set() for function in functions
    }
    alias_edges = {
        function: _scope_callable_alias_edges(function) for function in functions
    }

    changed = True
    while changed:
        changed = False
        summaries = {
            function: _parameter_return_flow(function, invoked[function])
            for function in functions
        }
        for function in functions:
            discovered = set(invoked[function])
            for call in _lexical_scope_calls(function):
                source_names = _expanded_callable_sources(
                    _call_name(call), alias_edges[function]
                )
                if isinstance(call.func, ast.Name):
                    discovered.update(
                        source_names.intersection(parameters[function])
                    )
                flows = [
                    summaries[callee]
                    for source_name in source_names
                    for callee in functions_by_name.get(source_name, ())
                ]
                visible_imports = (imported_flows or {}).get(function, {})
                flows.extend(
                    visible_imports[source_name]
                    for source_name in source_names.intersection(visible_imports)
                )
                for flow in flows:
                    for argument in _bound_parameter_flow_arguments(call, flow):
                        discovered.update(
                            _identifier_tokens(argument).intersection(
                                parameters[function]
                            )
                        )
            new_parameters = discovered - invoked[function]
            if new_parameters:
                invoked[function].update(new_parameters)
                changed = True
    return {
        function: _parameter_return_flow(function, invoked[function])
        for function in functions
    }


@lru_cache(maxsize=None)
def _direct_callback_invocation_flow_summaries(
    source_path: Path,
) -> tuple[tuple[str, _ParameterReturnFlow], ...]:
    """Summarize callback-parameter effects within one repository module."""

    tree = _parsed_module(source_path.resolve())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    flows = _local_callback_invocation_flows(functions)
    merged = _merged_parameter_flow_map(
        [(function.name, flows[function]) for function in functions]
    )
    return tuple(sorted(merged.items()))


def _repository_callback_invocation_flows(
    source_path: Path,
    requested_names: set[str],
    seen: frozenset[tuple[Path, str]],
) -> dict[str, _ParameterReturnFlow]:
    """Resolve callback-invocation summaries through local reexports."""

    source_path = source_path.resolve()
    requested = {
        name.casefold()
        for name in requested_names
        if (source_path, name.casefold()) not in seen
    }
    if not requested:
        return {}
    direct = dict(_direct_callback_invocation_flow_summaries(source_path))
    resolved = {name: direct[name] for name in requested.intersection(direct)}
    active = seen | {(source_path, name) for name in requested - set(resolved)}
    for local_name, imported_path, remote_name in _repository_reexport_bindings(
        source_path,
        requested - set(resolved),
    ):
        imported = _repository_callback_invocation_flows(
            imported_path,
            {remote_name},
            active,
        )
        if remote_name in imported:
            resolved[local_name] = imported[remote_name]
    return resolved


def _scope_imported_callback_invocation_flows(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
    source_path: Path | None,
    seen: frozenset[tuple[Path, str]] | None = None,
    calls: list[ast.Call] | None = None,
) -> dict[str, _ParameterReturnFlow]:
    """Return called repository-local callback summaries in one scope."""

    if source_path is None:
        return {}
    source_path = source_path.resolve()
    seen = seen or frozenset()
    visible_calls = calls if calls is not None else _lexical_scope_calls(scope)
    alias_edges = _scope_callable_alias_edges(scope)
    called_names = {
        source_name
        for call in visible_calls
        for source_name in _expanded_callable_sources(
            _call_name(call), alias_edges
        )
    }
    entries: list[tuple[str, _ParameterReturnFlow]] = []
    for node in _lexical_scope_imports(scope):
        if isinstance(node, ast.ImportFrom) and node.module is None:
            for imported in node.names:
                child_path = _resolved_repository_import_path(
                    source_path,
                    imported.name,
                    node.level,
                )
                if child_path is None:
                    continue
                qualifier = (imported.asname or imported.name).casefold()
                members = _module_attribute_call_names(
                    visible_calls,
                    qualifier,
                    scope if isinstance(scope, ast.Module) else None,
                )
                remote = _repository_callback_invocation_flows(
                    child_path,
                    members,
                    seen,
                )
                entries.extend(remote.items())
            continue
        if isinstance(node, ast.ImportFrom):
            imported_path = _resolved_repository_import_path(
                source_path,
                node.module,
                node.level,
            )
            if imported_path is None:
                continue
            bindings = {
                (imported.asname or imported.name).casefold(): (
                    imported.name.casefold()
                )
                for imported in node.names
                if imported.name != "*"
                and (imported.asname or imported.name).casefold() in called_names
            }
            star_names = called_names if any(
                imported.name == "*" for imported in node.names
            ) else set()
            remote = _repository_callback_invocation_flows(
                imported_path,
                set(bindings.values()) | star_names,
                seen,
            )
            entries.extend(
                (local_name, remote[remote_name])
                for local_name, remote_name in bindings.items()
                if remote_name in remote
            )
            entries.extend(
                (name, remote[name]) for name in star_names.intersection(remote)
            )
            continue
        if not isinstance(node, ast.Import):
            continue
        for imported in node.names:
            imported_path = _resolved_repository_import_path(
                source_path,
                imported.name,
            )
            if imported_path is None:
                continue
            qualifier = (imported.asname or imported.name).casefold()
            members = _module_attribute_call_names(
                visible_calls,
                qualifier,
                scope if isinstance(scope, ast.Module) else None,
            )
            remote = _repository_callback_invocation_flows(
                imported_path,
                members,
                seen,
            )
            entries.extend(remote.items())
    return _merged_parameter_flow_map(entries)


def _function_callback_invocation_flow_imports(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    module_flows: dict[str, _ParameterReturnFlow],
    source_path: Path | None,
    function_parents: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        ast.FunctionDef | ast.AsyncFunctionDef,
    ],
    seen: frozenset[tuple[Path, str]] | None = None,
) -> dict[
    ast.FunctionDef | ast.AsyncFunctionDef,
    dict[str, _ParameterReturnFlow],
]:
    """Resolve callback summaries visible to each lexical function."""

    visible: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        dict[str, _ParameterReturnFlow],
    ] = {}

    def analyze(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> dict[str, _ParameterReturnFlow]:
        cached = visible.get(function)
        if cached is not None:
            return cached
        parent = function_parents.get(function)
        inherited = analyze(parent) if parent is not None else module_flows
        local = _scope_imported_callback_invocation_flows(
            function,
            source_path,
            seen,
        )
        resolved = _merged_parameter_flow_map(
            [*inherited.items(), *local.items()]
        )
        visible[function] = resolved
        return resolved

    for function in functions:
        analyze(function)
    return visible


def _repository_parameter_return_flows(
    source_path: Path,
    requested_names: set[str],
    seen: frozenset[tuple[Path, str]],
) -> dict[str, _ParameterReturnFlow]:
    """Resolve parameter-return summaries through repository-local imports."""

    source_path = source_path.resolve()
    requested = {
        name.casefold()
        for name in requested_names
        if (source_path, name.casefold()) not in seen
    }
    if not requested:
        return {}
    direct = dict(_direct_parameter_return_flow_summaries(source_path))
    resolved = {name: direct[name] for name in requested.intersection(direct)}
    active = seen | {(source_path, name) for name in requested - set(resolved)}

    for local_name, imported_path, remote_name in _repository_reexport_bindings(
        source_path,
        requested - set(resolved),
    ):
        imported = _repository_parameter_return_flows(
            imported_path,
            {remote_name},
            active,
        )
        if remote_name in imported:
            resolved[local_name] = imported[remote_name]
    if requested <= set(resolved):
        return resolved

    tree = _parsed_module(source_path)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    relevant_functions = _functions_reachable_by_local_calls(
        functions,
        requested - set(resolved),
    )
    relevant_calls = [
        call
        for function in relevant_functions
        for call in _lexical_scope_calls(function)
    ]
    module_imports = _scope_imported_parameter_return_flows(
        tree,
        source_path,
        active,
        relevant_calls,
    )
    function_imports = _function_parameter_return_flow_imports(
        tree,
        functions,
        module_imports,
        source_path,
        seen=active,
        eligible_functions=relevant_functions,
    )
    local_flows = _local_parameter_return_flows(functions, function_imports)
    local_by_name = _merged_parameter_flow_map(
        [
            (function.name, local_flows[function])
            for function in relevant_functions
        ]
    )
    resolved.update(
        {
            name: local_by_name[name]
            for name in requested.intersection(local_by_name)
        }
    )
    return resolved


def _scope_imported_parameter_return_flows(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
    source_path: Path | None,
    seen: frozenset[tuple[Path, str]] | None = None,
    calls: list[ast.Call] | None = None,
) -> dict[str, _ParameterReturnFlow]:
    """Return called repository-local parameter-flow imports in one scope."""

    if source_path is None:
        return {}
    source_path = source_path.resolve()
    seen = seen or frozenset()
    visible_calls = calls if calls is not None else _lexical_scope_calls(scope)
    callable_alias_edges = _scope_callable_alias_edges(scope)
    called_names = {
        source_name
        for call in visible_calls
        for source_name in _expanded_callable_sources(
            _call_name(call),
            callable_alias_edges,
        )
    }
    entries: list[tuple[str, _ParameterReturnFlow]] = []

    def aliased_module_members(qualifier: str) -> set[str]:
        prefix = f"{qualifier.casefold()}."
        return {
            source_name.rsplit(".", 1)[-1]
            for source_name in called_names
            if source_name.startswith(prefix)
        }

    for node in _lexical_scope_imports(scope):
        if isinstance(node, ast.ImportFrom) and node.module is None:
            for imported in node.names:
                child_path = _resolved_repository_import_path(
                    source_path,
                    imported.name,
                    node.level,
                )
                if child_path is None:
                    continue
                qualifier = (imported.asname or imported.name).casefold()
                member_calls = _module_attribute_call_names(
                    visible_calls,
                    qualifier,
                    scope if isinstance(scope, ast.Module) else None,
                ) | aliased_module_members(qualifier)
                entries.extend(
                    _repository_parameter_return_flows(
                        child_path,
                        member_calls,
                        seen,
                    ).items()
                )
            continue
        if isinstance(node, ast.ImportFrom):
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
            remote = _repository_parameter_return_flows(
                imported_path,
                set(bindings.values()) | star_names,
                seen,
            )
            entries.extend(
                (local_name, remote[remote_name])
                for local_name, remote_name in bindings.items()
                if remote_name in remote
            )
            entries.extend(
                (name, remote[name]) for name in star_names.intersection(remote)
            )
            continue
        if not isinstance(node, ast.Import):
            continue
        for imported in node.names:
            imported_path = _resolved_repository_import_path(
                source_path,
                imported.name,
            )
            if imported_path is None:
                continue
            qualifier = (imported.asname or imported.name).casefold()
            member_calls = _module_attribute_call_names(
                visible_calls,
                qualifier,
                scope if isinstance(scope, ast.Module) else None,
            ) | aliased_module_members(qualifier)
            entries.extend(
                _repository_parameter_return_flows(
                    imported_path,
                    member_calls,
                    seen,
                ).items()
            )
    return _merged_parameter_flow_map(entries)


def _decorator_terminal_name(decorator: ast.AST) -> str:
    while isinstance(decorator, ast.Call):
        decorator = decorator.func
    if isinstance(decorator, ast.Name):
        return decorator.id.casefold()
    if isinstance(decorator, ast.Attribute):
        return decorator.attr.casefold()
    return ""


@lru_cache(maxsize=None)
def _direct_provenance_properties(
    source_path: Path,
) -> tuple[tuple[str, frozenset[str]], ...]:
    """Summarize provenance-returning descriptors by their owner class."""

    source_path = source_path.resolve()
    tree = _parsed_module(source_path)
    provenance_helpers = set(_direct_provenance_helper_names(source_path))
    summarized: dict[str, frozenset[str]] = {}
    for class_node in (
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ):
        properties = {
            method.name.casefold()
            for method in class_node.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            and method.name.casefold() in provenance_helpers
            and any(
                marker in _decorator_terminal_name(decorator)
                for decorator in method.decorator_list
                for marker in ("property", "descriptor")
            )
        }
        if properties:
            summarized[class_node.name.casefold()] = frozenset(properties)
    return tuple(sorted(summarized.items()))


def _repository_provenance_property_names(
    source_path: Path,
    requested_classes: set[str],
    seen: frozenset[tuple[Path, str]] = frozenset(),
) -> set[str]:
    """Resolve descriptor summaries through repository class re-exports."""

    source_path = source_path.resolve()
    requested = {
        name.casefold()
        for name in requested_classes
        if (source_path, name.casefold()) not in seen
    }
    if not requested:
        return set()
    direct = dict(_direct_provenance_properties(source_path))
    properties = {
        property_name
        for class_name in requested.intersection(direct)
        for property_name in direct[class_name]
    }
    unresolved = requested - set(direct)
    active = seen | {(source_path, name) for name in unresolved}
    for _local_name, imported_path, remote_name in (
        _repository_reexport_bindings(source_path, unresolved)
    ):
        properties.update(
            _repository_provenance_property_names(
                imported_path, {remote_name}, active
            )
        )
    return properties


def _scope_imported_provenance_property_names(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
    source_path: Path | None,
) -> set[str]:
    """Return imported descriptor names whose reads expose provenance."""

    if source_path is None:
        return set()
    source_path = source_path.resolve()
    scope_nodes = (
        tuple(ast.walk(scope))
        if isinstance(scope, ast.Module)
        else _walk_lexical_scope(scope)
    )
    referenced_names = {
        node.id.casefold()
        for node in scope_nodes
        if isinstance(node, ast.Name)
    }
    properties: set[str] = set()
    for node in _lexical_scope_imports(scope):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_path = _resolved_repository_import_path(
                source_path, node.module, node.level
            )
            if imported_path is None:
                continue
            requested = {
                imported.name.casefold()
                for imported in node.names
                if imported.name != "*"
                and (imported.asname or imported.name).casefold()
                in referenced_names
            }
            if any(imported.name == "*" for imported in node.names):
                requested.update(
                    name for name, _values in _direct_provenance_properties(
                        imported_path
                    )
                )
            properties.update(
                _repository_provenance_property_names(
                    imported_path, requested
                )
            )
            continue
        if not isinstance(node, ast.Import):
            continue
        for imported in node.names:
            imported_path = _resolved_repository_import_path(
                source_path, imported.name
            )
            if imported_path is None:
                continue
            qualifier = (
                imported.asname or imported.name.split(".", 1)[0]
            ).casefold()
            requested = {
                expression[len(qualifier) + 1 :].split(".", 1)[0]
                for child in scope_nodes
                if isinstance(child, ast.Attribute)
                and (
                    expression := ast.unparse(child).casefold()
                ).startswith(f"{qualifier}.")
            }
            properties.update(
                _repository_provenance_property_names(
                    imported_path, requested
                )
            )
    return properties


def _scope_imported_provenance_return_helper_aliases(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
    source_path: Path | None,
    seen: frozenset[tuple[Path, str]] | None = None,
    calls: list[ast.Call] | None = None,
) -> set[str]:
    """Return provenance-helper imports visible in one lexical scope.

    Only follow imports that are actually invoked by this scope.  This keeps
    the repository-wide contract proportional to the authority call graph
    rather than recursively summarizing the checkout's entire import graph.
    """

    if source_path is None:
        return set()

    source_path = source_path.resolve()
    seen = seen or frozenset()
    visible_calls = calls if calls is not None else _lexical_scope_calls(scope)
    called_names = {_call_name(call).casefold() for call in visible_calls}
    aliases: set[str] = set()
    for node in _lexical_scope_imports(scope):
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
                        visible_calls,
                        (imported.asname or imported.name).casefold(),
                        scope if isinstance(scope, ast.Module) else None,
                    )
                    if not (
                        _source_or_imports_may_expose_provenance(child_path)
                        or _repository_reexport_bindings(
                            child_path,
                            module_calls,
                        )
                    ):
                        continue
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
            if not (
                _source_or_imports_may_expose_provenance(imported_path)
                or _repository_reexport_bindings(imported_path, requested)
            ):
                continue
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
            for imported in node.names:
                if imported.name == "*":
                    continue
                qualifier = (imported.asname or imported.name).casefold()
                member_calls = _module_attribute_call_names(
                    visible_calls,
                    qualifier,
                    scope if isinstance(scope, ast.Module) else None,
                )
                aliases.update(
                    _repository_provenance_helper_names(
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
                if imported_path is not None and (
                    _source_or_imports_may_expose_provenance(imported_path)
                    or _repository_reexport_bindings(
                        imported_path,
                        _module_attribute_call_names(
                            visible_calls,
                            (imported.asname or imported.name).casefold(),
                            scope if isinstance(scope, ast.Module) else None,
                        ),
                    )
                ):
                    bound_name = (imported.asname or imported.name).casefold()
                    module_calls = _module_attribute_call_names(
                        visible_calls,
                        bound_name,
                        scope if isinstance(scope, ast.Module) else None,
                    )
                    aliases.update(
                        _repository_provenance_helper_names(
                            imported_path,
                            module_calls,
                            seen,
                        )
                    )
    aliases.update(
        _scope_imported_provenance_property_names(scope, source_path)
    )
    return aliases


def _module_imported_provenance_return_helper_aliases(
    tree: ast.AST,
    source_path: Path | None,
    seen: frozenset[tuple[Path, str]] | None = None,
    calls: list[ast.Call] | None = None,
) -> set[str]:
    """Return provenance-helper imports visible to the module body."""

    if not isinstance(tree, ast.Module):
        return set()
    return _scope_imported_provenance_return_helper_aliases(
        tree,
        source_path,
        seen,
        calls,
    )


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


def _function_class_owners(
    tree: ast.AST,
) -> dict[ast.FunctionDef | ast.AsyncFunctionDef, ast.ClassDef]:
    """Map functions to the nearest class whose lexical body contains them."""

    owners: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, ast.ClassDef
    ] = {}
    classes: list[ast.ClassDef] = []

    class ClassOwnerVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            classes.append(node)
            self.generic_visit(node)
            classes.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            if classes:
                owners[node] = classes[-1]
            self.generic_visit(node)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            if classes:
                owners[node] = classes[-1]
            self.generic_visit(node)

    ClassOwnerVisitor().visit(tree)
    return owners


def _visible_import_aliases_by_function(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    module_aliases: set[str],
    source_path: Path | None,
    resolver: Callable[
        [ast.FunctionDef | ast.AsyncFunctionDef, Path | None], set[str]
    ],
    function_parents: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        ast.FunctionDef | ast.AsyncFunctionDef,
    ],
) -> dict[ast.FunctionDef | ast.AsyncFunctionDef, set[str]]:
    """Resolve imported bindings visible at each function's lexical floor."""

    visible: dict[ast.FunctionDef | ast.AsyncFunctionDef, set[str]] = {}

    def analyze(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> set[str]:
        cached = visible.get(function)
        if cached is not None:
            return cached
        parent = function_parents.get(function)
        inherited = analyze(parent) if parent is not None else module_aliases
        resolved = set(inherited) | resolver(function, source_path)
        visible[function] = resolved
        return resolved

    for function in functions:
        analyze(function)
    return visible


def _function_control_import_aliases(
    tree: ast.AST,
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    module_aliases: set[str],
    source_path: Path | None,
    function_parents: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        ast.FunctionDef | ast.AsyncFunctionDef,
    ] | None = None,
    seen: frozenset[tuple[Path, str]] | None = None,
    eligible_functions: set[
        ast.FunctionDef | ast.AsyncFunctionDef
    ] | None = None,
) -> dict[ast.FunctionDef | ast.AsyncFunctionDef, set[str]]:
    return _visible_import_aliases_by_function(
        functions,
        module_aliases,
        source_path,
        lambda function, path: (
            _scope_imported_control_aliases(function, path, seen)
            if eligible_functions is None or function in eligible_functions
            else set()
        ),
        function_parents or _nested_function_parents(tree),
    )


def _function_provenance_helper_import_aliases(
    tree: ast.AST,
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    module_aliases: set[str],
    source_path: Path | None,
    function_parents: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        ast.FunctionDef | ast.AsyncFunctionDef,
    ] | None = None,
    seen: frozenset[tuple[Path, str]] | None = None,
    eligible_functions: set[
        ast.FunctionDef | ast.AsyncFunctionDef
    ] | None = None,
) -> dict[ast.FunctionDef | ast.AsyncFunctionDef, set[str]]:
    return _visible_import_aliases_by_function(
        functions,
        module_aliases,
        source_path,
        lambda function, path: (
            _scope_imported_provenance_return_helper_aliases(
                function,
                path,
                seen,
            )
            if eligible_functions is None or function in eligible_functions
            else set()
        ),
        function_parents or _nested_function_parents(tree),
    )


def _function_parameter_return_flow_imports(
    tree: ast.AST,
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    module_flows: dict[str, _ParameterReturnFlow],
    source_path: Path | None,
    function_parents: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        ast.FunctionDef | ast.AsyncFunctionDef,
    ] | None = None,
    seen: frozenset[tuple[Path, str]] | None = None,
    eligible_functions: set[
        ast.FunctionDef | ast.AsyncFunctionDef
    ] | None = None,
) -> dict[
    ast.FunctionDef | ast.AsyncFunctionDef,
    dict[str, _ParameterReturnFlow],
]:
    """Resolve module, local, and enclosing parameter-flow imports per function."""

    parents = function_parents or _nested_function_parents(tree)
    visible: dict[
        ast.FunctionDef | ast.AsyncFunctionDef,
        dict[str, _ParameterReturnFlow],
    ] = {}

    def analyze(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> dict[str, _ParameterReturnFlow]:
        cached = visible.get(function)
        if cached is not None:
            return cached
        parent = parents.get(function)
        inherited = analyze(parent) if parent is not None else module_flows
        resolved = dict(inherited)
        if eligible_functions is None or function in eligible_functions:
            local = _scope_imported_parameter_return_flows(
                function,
                source_path,
                seen,
            )
            resolved = _merged_parameter_flow_map(
                [*resolved.items(), *local.items()]
            )
        visible[function] = resolved
        return resolved

    for function in functions:
        analyze(function)
    return visible


def _executable_body_functions(
    tree: ast.AST,
) -> list[ast.FunctionDef]:
    """Wrap import-time module/class bodies for the function-scope analyzers."""

    bodies: list[tuple[str, ast.AST, list[ast.stmt]]] = []
    if isinstance(tree, ast.Module):
        bodies.append(("module", tree, tree.body))
    bodies.extend(
        (f"class_{node.name}", node, node.body)
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    )
    scopes: list[ast.FunctionDef] = []
    for name, owner, body in bodies:
        scope = ast.parse(f"def __audit_{name}_body__():\n    pass\n").body[0]
        assert isinstance(scope, ast.FunctionDef)
        scope.body = body
        ast.copy_location(scope, owner)
        scopes.append(scope)
    return scopes


def _permission_mutation_uses_provenance(
    targets: list[ast.AST],
    values: list[ast.AST],
    provenance_aliases: set[str],
    provenance_return_helpers: set[str],
    permission_store_aliases: set[str] | None = None,
) -> bool:
    """Whether provenance selects a permission write's subject or value."""

    if not any(
        _is_permission_name(token)
        or (
            isinstance(target, (ast.Attribute, ast.Subscript))
            and token in (permission_store_aliases or set())
        )
        for target in targets
        for token in _identifier_tokens(target)
    ):
        return False
    target_selectors: list[ast.AST] = []
    for target in targets:
        if isinstance(target, ast.Subscript):
            target_selectors.append(target.slice)
            if not isinstance(target.value, ast.Name):
                target_selectors.append(target.value)
        elif isinstance(target, ast.Attribute):
            target_selectors.append(target.value)
    candidates = [*values, *target_selectors]
    return any(
        _has_provenance_value(
            candidate,
            provenance_aliases,
            provenance_return_helpers,
        )
        for candidate in candidates
    )


def _permission_store_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """Propagate permission-store identity through ordinary local aliases."""

    aliases = {
        token
        for node in _walk_lexical_scope(function)
        for token in _identifier_tokens(node)
        if _is_permission_name(token)
    }
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
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        assignments.append(
            (
                {
                    name
                    for target in targets
                    for name in _binding_target_names(target)
                },
                set(_identifier_tokens(value)),
            )
        )

    changed = True
    while changed:
        changed = False
        for targets, sources in assignments:
            if not sources.intersection(aliases):
                continue
            new_aliases = targets - aliases
            if new_aliases:
                aliases.update(new_aliases)
                changed = True
    return aliases


def _permission_store_call_uses_provenance(
    call: ast.Call,
    permission_store_aliases: set[str],
    provenance_aliases: set[str],
    provenance_return_helpers: set[str],
) -> bool:
    """Whether a mutator call writes provenance through a permission alias."""

    receiver: ast.AST | None = None
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr.casefold() in _MUTABLE_CONTAINER_WRITE_ARGUMENTS
    ):
        receiver = call.func.value
    elif _call_name(call).casefold().strip("_") in {
        "delattr",
        "delitem",
        "setattr",
        "setitem",
    }:
        receiver = call.args[0] if call.args else None
    if receiver is None or not _identifier_tokens(receiver).intersection(
        permission_store_aliases
    ):
        return False
    return any(
        _has_provenance_value(
            candidate,
            provenance_aliases,
            provenance_return_helpers,
        )
        for candidate in [*call.args, *(item.value for item in call.keywords)]
    )


def _authority_provenance_lines(
    tree: ast.AST,
    source_path: Path | None = None,
) -> set[int]:
    lines: set[int] = set()
    real_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    executable_scopes = _executable_body_functions(tree)
    functions = [*real_functions, *executable_scopes]
    local_attribution_authorizers = _local_attribution_authorization_helpers(
        functions
    )
    for call in (
        node
        for function in functions
        for node in _walk_lexical_scope(function)
        if isinstance(node, ast.Call)
        and _call_name(node).casefold() in local_attribution_authorizers
    ):
        setattr(call, "_authority_local_attribution_authorizer", True)
    function_parents = _nested_function_parents(tree)
    module_attribution_validators = _trusted_attribution_validator_aliases(tree)
    function_attribution_validators: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] = {}

    def attribution_validators_for(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> set[str]:
        cached = function_attribution_validators.get(function)
        if cached is not None:
            return cached
        parent = function_parents.get(function)
        inherited = (
            attribution_validators_for(parent)
            if parent is not None
            else module_attribution_validators
        )
        local_bindings = _scope_local_binding_names(function)
        resolved = {
            alias
            for alias in inherited
            if _state_alias_root(alias) not in local_bindings
        } | _trusted_attribution_validator_aliases(function)
        function_attribution_validators[function] = resolved
        return resolved

    module_shell_aliases: set[str] = set()
    class_shell_aliases: dict[ast.ClassDef, set[str]] = {}
    if isinstance(tree, ast.Module) and executable_scopes:
        module_shell_aliases = _annotate_shell_lifecycle_command_aliases(
            executable_scopes[0]
        )
        class_nodes = [
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]
        classes_by_name = {
            class_node.name.casefold(): class_node for class_node in class_nodes
        }
        for class_node, scope in zip(class_nodes, executable_scopes[1:]):
            inherited_class_aliases = {
                alias
                for base in class_node.bases
                for token in _identifier_tokens(base)
                if (base_class := classes_by_name.get(token)) is not None
                for alias in class_shell_aliases.get(base_class, set())
            }
            class_shell_aliases[class_node] = (
                _annotate_shell_lifecycle_command_aliases(
                    scope, module_shell_aliases | inherited_class_aliases
                )
            )
    function_class_owners = _function_class_owners(tree)
    function_shell_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] = {}

    def annotate_function_shell_aliases(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> set[str]:
        cached = function_shell_aliases.get(function)
        if cached is not None:
            return cached
        parent = function_parents.get(function)
        inherited = (
            annotate_function_shell_aliases(parent)
            if parent is not None
            else module_shell_aliases
            | class_shell_aliases.get(
                function_class_owners.get(function), set()
            )
        )
        resolved = _annotate_shell_lifecycle_command_aliases(
            function, inherited
        )
        function_shell_aliases[function] = resolved
        return resolved

    for function in real_functions:
        annotate_function_shell_aliases(function)

    module_control_aliases = _module_imported_control_aliases(tree)
    parameter_flow_functions = {
        function
        for function in functions
        if _scope_contains_provenance_marker(function)
    }
    parameter_flow_functions.update(
        _functions_reachable_by_local_calls(
            functions,
            {function.name.casefold() for function in parameter_flow_functions},
        )
    )
    for function in tuple(parameter_flow_functions):
        parent = function_parents.get(function)
        while parent is not None:
            parameter_flow_functions.add(parent)
            parent = function_parents.get(parent)
    parameter_flow_calls = [
        call
        for function in parameter_flow_functions
        for call in _lexical_scope_calls(function)
    ]
    module_parameter_return_flows = _scope_imported_parameter_return_flows(
        tree,
        source_path,
        calls=parameter_flow_calls,
    )
    function_parameter_return_flows = _function_parameter_return_flow_imports(
        tree,
        functions,
        module_parameter_return_flows,
        source_path,
        function_parents,
        eligible_functions=parameter_flow_functions,
    )
    local_parameter_return_flows = _local_parameter_return_flows(
        functions,
        function_parameter_return_flows,
    )
    local_parameter_return_flows_by_name = _merged_parameter_flow_map(
        [
            (function.name, local_parameter_return_flows[function])
            for function in functions
        ]
    )
    function_control_parameter_return_flows = {
        function: _merged_parameter_flow_map(
            [
                *local_parameter_return_flows_by_name.items(),
                *function_parameter_return_flows[function].items(),
            ]
        )
        for function in functions
    }
    attribution_projection_helpers = _attribution_projection_helpers(
        functions
    )
    function_attribution_aliases = {
        function: _unverified_attribution_aliases(
            function,
            trusted_validator_aliases=attribution_validators_for(function),
            string_constants=(
                _module_strings_at_definition(tree, function, source_path)
                if isinstance(tree, ast.Module)
                else {}
            ),
            parameter_return_flows=(
                function_control_parameter_return_flows[function]
            ),
            projection_helpers=attribution_projection_helpers,
        )
        for function in functions
    }
    module_provenance_aliases = (
        _module_provenance_constant_aliases(tree, source_path)
        | _module_imported_provenance_annotation_aliases(tree)
        | _module_imported_provenance_accessor_aliases(tree)
    )
    module_shared_bindings = _module_shared_binding_names(tree)
    module_mutable_bindings = (
        _scope_mutable_container_bindings(executable_scopes[0])
        if isinstance(tree, ast.Module) and executable_scopes
        else set()
    )
    imported_provenance_helpers = (
        _module_imported_provenance_return_helper_aliases(tree, source_path)
    )
    function_imported_provenance_helpers = (
        _function_provenance_helper_import_aliases(
            tree,
            functions,
            imported_provenance_helpers,
            source_path,
            function_parents,
        )
    )
    control_import_functions = {
        function
        for function in functions
        if _scope_contains_provenance_marker(function)
        or any(
            _call_name(call).casefold()
            in function_imported_provenance_helpers[function]
            for call in _lexical_scope_calls(function)
        )
    }
    for function in tuple(control_import_functions):
        parent = function_parents.get(function)
        while parent is not None:
            control_import_functions.add(parent)
            parent = function_parents.get(parent)
    control_import_functions.update(
        _functions_reachable_by_local_calls(
            functions,
            {function.name.casefold() for function in control_import_functions},
        )
    )
    relevant_control_calls = [
        call
        for function in control_import_functions
        for call in _lexical_scope_calls(function)
    ]
    module_control_aliases = _module_imported_control_aliases(
        tree,
        source_path,
        calls=relevant_control_calls,
    )
    function_control_imports = _function_control_import_aliases(
        tree,
        functions,
        module_control_aliases,
        source_path,
        function_parents,
        eligible_functions=control_import_functions,
    )
    module_callback_flows = _scope_imported_callback_invocation_flows(
        tree,
        source_path,
        calls=relevant_control_calls,
    )
    function_callback_flows = _function_callback_invocation_flow_imports(
        functions,
        module_callback_flows,
        source_path,
        function_parents,
    )
    for function in functions:
        alias_edges = _scope_callable_alias_edges(function)
        visible_aliases = module_control_aliases | function_control_imports[function]
        discovered: set[str] = set()
        for call in _lexical_scope_calls(function):
            source_names = _expanded_callable_sources(
                _call_name(call), alias_edges
            )
            for source_name in source_names.intersection(
                function_callback_flows[function]
            ):
                flow = function_callback_flows[function][source_name]
                if any(
                    any(
                        _is_unambiguous_control_token(token)
                        or token in visible_aliases
                        for token in _control_reference_sources(argument)
                    )
                    for argument in _bound_parameter_flow_arguments(call, flow)
                ):
                    discovered.update(source_names)
                    discovered.add(_call_name(call).casefold())
        function_control_imports[function].update(discovered)
    function_callback_control_aliases = {
        function: set() for function in functions
    }
    control_helpers = _local_control_helpers(
        functions,
        module_control_aliases,
        function_control_imports,
        function_callback_control_aliases,
        function_control_parameter_return_flows,
    )
    control_return_helpers: set[str] = set()
    class_control_aliases: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, set[str]
    ] = {}
    control_changed = True
    while control_changed:
        previous_helpers = set(control_helpers)
        previous_return_helpers = set(control_return_helpers)
        previous_class_aliases = {
            function: set(aliases)
            for function, aliases in class_control_aliases.items()
        }
        visible_control_aliases = {
            function: function_control_imports[function]
            | class_control_aliases.get(function, set())
            | function_callback_control_aliases[function]
            for function in functions
        }
        control_return_helpers.update(
            _local_control_return_helpers(
                functions,
                control_helpers,
                visible_control_aliases,
            )
        )
        control_helpers.update(
            _local_control_helpers(
                functions,
                module_control_aliases | control_helpers,
                visible_control_aliases,
                function_callback_control_aliases,
                function_control_parameter_return_flows,
            )
        )
        class_control_aliases = _class_control_state_aliases(
            tree,
            control_helpers,
            module_control_aliases,
            visible_control_aliases,
            control_return_helpers,
        )
        control_changed = (
            control_helpers != previous_helpers
            or control_return_helpers != previous_return_helpers
            or class_control_aliases != previous_class_aliases
        )
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
    lexical_provenance_aliases: dict[
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
        previous_lexical_aliases = {
            function: set(aliases)
            for function, aliases in lexical_provenance_aliases.items()
        }
        class_provenance_aliases = _class_provenance_state_aliases(
            tree,
            provenance_return_helpers,
            module_provenance_aliases,
            function_imported_provenance_helpers,
        )
        function_state_aliases = {
            function: class_provenance_aliases.get(function, set())
            | lexical_provenance_aliases.get(function, set())
            for function in functions
        }
        provenance_return_helpers = _local_provenance_return_helpers(
            functions,
            control_helpers,
            module_provenance_aliases,
            provenance_return_helpers,
            function_state_aliases,
            function_imported_provenance_helpers,
            function_parameter_return_flows,
            function_parents,
        )
        module_provenance_aliases = _module_provenance_state_aliases(
            functions,
            function_parents,
            provenance_return_helpers,
            module_provenance_aliases,
            function_state_aliases,
            function_accessor_aliases,
            function_imported_provenance_helpers,
            module_shared_bindings,
            module_mutable_bindings,
        )
        lexical_provenance_aliases = _lexical_provenance_state_aliases(
            functions,
            function_parents,
            provenance_return_helpers,
            module_provenance_aliases,
            function_state_aliases,
            function_accessor_aliases,
            function_imported_provenance_helpers,
        )
        analysis_changed = (
            module_provenance_aliases != previous_module_aliases
            or provenance_return_helpers != previous_helper_names
            or class_provenance_aliases != previous_class_aliases
            or lexical_provenance_aliases != previous_lexical_aliases
        )

    # Callable identity is a shared semantic layer, not a growing inventory of
    # AST spellings.  Reconcile helper summaries after annotating aliases and
    # callable instances so wrappers discovered through either domain feed the
    # same fixed point as directly named functions.
    callable_semantics_changed = True
    while callable_semantics_changed:
        previous_control_helpers = set(control_helpers)
        previous_control_return_helpers = set(control_return_helpers)
        previous_class_control_aliases = {
            function: set(aliases)
            for function, aliases in class_control_aliases.items()
        }
        previous_provenance_helpers = set(provenance_return_helpers)
        previous_module_provenance_aliases = set(module_provenance_aliases)
        previous_class_provenance_aliases = {
            function: set(aliases)
            for function, aliases in class_provenance_aliases.items()
        }
        previous_lexical_provenance_aliases = {
            function: set(aliases)
            for function, aliases in lexical_provenance_aliases.items()
        }
        function_state_aliases = {
            function: class_provenance_aliases.get(function, set())
            | lexical_provenance_aliases.get(function, set())
            for function in functions
        }

        imported_callable_classes = _scope_imported_callable_class_semantics(
            tree,
            source_path,
        )
        function_imported_callable_classes = (
            _function_imported_callable_class_semantics(
                functions,
                source_path,
            )
        )
        callable_classes = dict(imported_callable_classes)
        callable_classes.update(_callable_class_semantics(
            tree,
            provenance_return_helpers,
            control_helpers,
            module_control_aliases,
            function_control_imports,
            class_control_aliases,
            module_provenance_aliases,
            function_state_aliases,
            function_imported_provenance_helpers,
        ))
        _annotate_static_callable_semantics(
            tree,
            functions,
            function_parents,
            provenance_return_helpers,
            control_helpers,
            control_return_helpers,
            callable_classes,
            imported_callable_classes,
            function_imported_callable_classes,
            imported_provenance_helpers,
            module_control_aliases,
            function_imported_provenance_helpers,
            function_control_imports,
            function_accessor_aliases,
            _resolved_cycle_bounded_helpers(tree, source_path),
        )
        _cached_contains_cross_agent_control_call.cache_clear()

        visible_control_aliases = {
            function: function_control_imports[function]
            | class_control_aliases.get(function, set())
            | function_callback_control_aliases[function]
            for function in functions
        }
        control_return_helpers.update(
            _local_control_return_helpers(
                functions,
                control_helpers,
                visible_control_aliases,
            )
        )
        control_helpers.update(
            _local_control_helpers(
                functions,
                module_control_aliases | control_helpers,
                visible_control_aliases,
                function_callback_control_aliases,
                function_control_parameter_return_flows,
            )
        )
        class_control_aliases = _class_control_state_aliases(
            tree,
            control_helpers,
            module_control_aliases,
            visible_control_aliases,
            control_return_helpers,
        )
        class_provenance_aliases = _class_provenance_state_aliases(
            tree,
            provenance_return_helpers,
            module_provenance_aliases,
            function_imported_provenance_helpers,
        )
        function_state_aliases = {
            function: class_provenance_aliases.get(function, set())
            | lexical_provenance_aliases.get(function, set())
            for function in functions
        }
        provenance_return_helpers = _local_provenance_return_helpers(
            functions,
            control_helpers,
            module_provenance_aliases,
            provenance_return_helpers,
            function_state_aliases,
            function_imported_provenance_helpers,
            function_parameter_return_flows,
            function_parents,
        )
        module_provenance_aliases = _module_provenance_state_aliases(
            functions,
            function_parents,
            provenance_return_helpers,
            module_provenance_aliases,
            function_state_aliases,
            function_accessor_aliases,
            function_imported_provenance_helpers,
            module_shared_bindings,
            module_mutable_bindings,
        )
        lexical_provenance_aliases = _lexical_provenance_state_aliases(
            functions,
            function_parents,
            provenance_return_helpers,
            module_provenance_aliases,
            function_state_aliases,
            function_accessor_aliases,
            function_imported_provenance_helpers,
        )
        callable_semantics_changed = (
            control_helpers != previous_control_helpers
            or control_return_helpers != previous_control_return_helpers
            or class_control_aliases != previous_class_control_aliases
            or provenance_return_helpers != previous_provenance_helpers
            or module_provenance_aliases
            != previous_module_provenance_aliases
            or class_provenance_aliases
            != previous_class_provenance_aliases
            or lexical_provenance_aliases
            != previous_lexical_provenance_aliases
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

    parameter_mutation_flows = _local_parameter_mutation_flows(functions)

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
            provenance_return_helpers
            | function_imported_provenance_helpers[function],
            control_helpers
            | function_control_imports[function]
            | class_control_aliases.get(function, set())
            | function_callback_control_aliases[function],
            module_provenance_aliases
            | class_provenance_aliases.get(function, set())
            | lexical_provenance_aliases.get(function, set())
            | inherited,
            control_return_helpers=(
                control_return_helpers
                | function_control_imports[function]
            ),
            state_object_aliases=function_state_objects[function],
            provenance_accessor_aliases=function_accessor_aliases[function],
            parameter_mutation_flows=parameter_mutation_flows,
            control_parameter_return_flows=(
                function_control_parameter_return_flows[function]
            ),
        )
        function_provenance[function] = resolved
        return resolved

    for function in functions:
        analyze_function_provenance(function)
    for function in functions:
        function_name = function.name.casefold()
        decision_provenance_helpers = (
            provenance_return_helpers
            | function_imported_provenance_helpers[function]
            | function_accessor_aliases[function]
        )
        control_aliases = _cross_agent_control_aliases(
            function,
            control_helpers
            | function_control_imports[function]
            | class_control_aliases.get(function, set())
            | function_callback_control_aliases[function],
            control_return_helpers | function_control_imports[function],
            function_control_parameter_return_flows[function],
        )
        state_object_aliases = function_state_objects[function]
        permission_store_aliases = _permission_store_aliases(function)
        provenance_aliases, provenance_selected_targets = (
            function_provenance[function]
        )
        attribution_aliases = function_attribution_aliases[function]
        lines.update(
            _provenance_selected_cross_agent_read_lines(
                function,
                state_object_aliases,
                provenance_aliases,
                decision_provenance_helpers,
            )
        )
        lines.update(
            _guard_clause_provenance_lines(
                function,
                provenance_aliases,
                control_aliases,
                decision_provenance_helpers,
                state_object_aliases,
                local_attribution_authorizers,
            )
        )
        function_is_permission_boundary = (
            _is_permission_name(function_name)
            or function_name.startswith(("can_", "may_"))
            or _is_unambiguous_control_token(function_name)
        )
        for decorator in function.decorator_list:
            if not _has_provenance_value(
                decorator,
                provenance_aliases,
                decision_provenance_helpers,
            ):
                continue
            decorator_tokens = _identifier_tokens(decorator)
            decorator_is_authority_boundary = any(
                _is_permission_name(token)
                or _is_unambiguous_control_token(token)
                for token in decorator_tokens
            )
            if (
                decorator_is_authority_boundary
                or function_is_permission_boundary
                or _contains_cross_agent_control_call(
                    function.body,
                    control_aliases,
                    state_object_aliases,
                )
            ):
                lines.add(decorator.lineno)
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
            if isinstance(node, (ast.Try, ast.TryStar)) and (
                _try_flow_uses_provenance_as_control(
                    node,
                    provenance_aliases,
                    control_aliases,
                    decision_provenance_helpers,
                    state_object_aliases,
                )
            ):
                lines.add(node.lineno)
            if isinstance(node, ast.Call):
                function_tokens = _identifier_tokens(node.func)
                is_permission_call = any(
                    _is_permission_name(token) for token in function_tokens
                )
                validates_attribution = _is_attribution_validation_call(
                    node, local_attribution_authorizers
                ) or _call_name(node).casefold() in _A2A_IDENTITY_PLUMBING_CALLS
                arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
                if (
                    is_permission_call
                    and not validates_attribution
                    and any(
                        _has_provenance_value(
                            candidate,
                            provenance_aliases,
                            decision_provenance_helpers,
                        )
                        for candidate in [node.func, *arguments]
                    )
                ):
                    lines.add(node.lineno)
                if _permission_store_call_uses_provenance(
                    node,
                    permission_store_aliases,
                    provenance_aliases,
                    decision_provenance_helpers,
                ):
                    lines.add(node.lineno)
                is_control_call = _is_cross_agent_control_call(
                    node, control_aliases, state_object_aliases
                )
                is_direct_control_call = (
                    is_control_call or _is_recipient_scoped_task_read(node)
                )
                is_state_mutation_call = _is_cross_agent_state_mutation_call(
                    node, state_object_aliases
                )
                if (
                    is_direct_control_call or is_state_mutation_call
                ) and not validates_attribution:
                    # Parameter spelling is not an authority boundary.  A
                    # known control sink may call its target ``candidate``,
                    # ``subject``, or anything else, so inspect every supplied
                    # value rather than maintaining a bypassable name list.
                    evaluated_arguments = [
                        argument
                        for argument in arguments
                        # A lambda body is not evaluated when the callable is
                        # registered. Escaping lambdas are audited at their
                        # own execution site above, so counting their body as
                        # a direct input duplicates and mislocates the finding.
                        if not isinstance(argument, ast.Lambda)
                    ]
                    direct_control_inputs = [node.func, *evaluated_arguments]
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
                            for argument in evaluated_arguments
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
                if _permission_mutation_uses_provenance(
                    assignment_targets,
                    [assignment_value],
                    provenance_aliases,
                    decision_provenance_helpers,
                    permission_store_aliases,
                ):
                    lines.add(node.lineno)
                if any(
                    _is_cross_agent_state_mutation_target(
                        target,
                        state_object_aliases,
                    )
                    for target in assignment_targets
                ) and any(
                    _has_provenance_value(
                        candidate,
                        provenance_aliases,
                        decision_provenance_helpers,
                    )
                    for candidate in [*assignment_targets, assignment_value]
                ):
                    lines.add(node.lineno)
            if isinstance(node, ast.AugAssign):
                if (
                    isinstance(node.target, (ast.Attribute, ast.Subscript))
                    and _permission_mutation_uses_provenance(
                        [node.target],
                        [],
                        provenance_aliases,
                        decision_provenance_helpers,
                        permission_store_aliases,
                    )
                ) or (
                    _is_cross_agent_state_mutation_target(
                        node.target,
                        state_object_aliases,
                    )
                    and any(
                        _has_provenance_value(
                            candidate,
                            provenance_aliases,
                            decision_provenance_helpers,
                        )
                        for candidate in (node.target, node.value)
                    )
                ):
                    lines.add(node.lineno)
            if isinstance(node, ast.Delete):
                structured_targets = [
                    target
                    for target in node.targets
                    if isinstance(target, (ast.Attribute, ast.Subscript))
                ]
                if _permission_mutation_uses_provenance(
                    structured_targets,
                    [],
                    provenance_aliases,
                    decision_provenance_helpers,
                    permission_store_aliases,
                ) or any(
                    _is_cross_agent_state_mutation_target(
                        target,
                        state_object_aliases,
                    )
                    and isinstance(target, (ast.Attribute, ast.Subscript))
                    and _has_provenance_value(
                        target, provenance_aliases, decision_provenance_helpers
                    )
                    for target in node.targets
                ):
                    lines.add(node.lineno)
            if isinstance(node, ast.Return):
                if (
                    node.value is not None
                    and not (
                        (
                            isinstance(node.value, ast.Call)
                            or (
                                isinstance(node.value, ast.Await)
                                and isinstance(node.value.value, ast.Call)
                            )
                        )
                        and _is_attribution_validation_call(
                            node.value.value
                            if isinstance(node.value, ast.Await)
                            else node.value,
                            local_attribution_authorizers,
                        )
                    )
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
                subject_or_pattern_has_provenance = _has_provenance_value(
                    node.subject,
                    provenance_aliases,
                    decision_provenance_helpers,
                ) or any(
                    _match_pattern_provenance_marker(case.pattern) is not None
                    for case in node.cases
                )
                if subject_or_pattern_has_provenance and (
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
            ) or bool(tokens.intersection(attribution_aliases))
            has_authenticated_attribution = any(
                getattr(child, "_authority_authenticated_attribution", False)
                for child in ast.walk(node.test)
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
            only_selects_attribution_validation = (
                isinstance(node, ast.IfExp)
                and any(
                    isinstance(child, ast.Call)
                    and _is_attribution_validation_call(
                        child, local_attribution_authorizers
                    )
                    for branch in (node.body, node.orelse)
                    for child in ast.walk(branch)
                )
                and not _contains_cross_agent_control_call(
                    [node.body, node.orelse],
                    control_aliases,
                    state_object_aliases,
                )
            )
            only_suppresses_cycle = isinstance(
                node, ast.If
            ) and _if_only_suppresses_causation_cycle(
                node,
                provenance_aliases,
                control_aliases,
                decision_provenance_helpers,
                state_object_aliases,
            )
            authenticated_identity_selects_control = (
                has_authenticated_attribution
                and _authenticated_identity_selects_protected_control(
                    guarded_nodes, control_aliases, state_object_aliases
                )
            )
            if authenticated_identity_selects_control or (
                has_provenance
                and not only_suppresses_cycle
                and not only_selects_attribution_validation
                and not _is_fail_closed_envelope_acceptance_guard(
                    node,
                    function,
                    local_attribution_authorizers,
                    control_aliases,
                    state_object_aliases,
                )
                and (
                    has_permission
                    or function_is_permission_boundary
                    or _contains_cross_agent_control_call(
                        guarded_nodes, control_aliases, state_object_aliases
                    )
                    or selects_control
                )
            ):
                lines.add(node.lineno)
    result = set(lines)
    _clear_authority_analysis_annotations(tree)
    return result


@lru_cache(maxsize=None)
def _imported_callable_classes_expose_provenance(
    source_path: Path,
) -> bool:
    """Whether an invoked imported ``__call__`` can return provenance."""

    source_path = source_path.resolve()
    tree = _parsed_module(source_path)
    module_classes = _scope_imported_callable_class_semantics(
        tree,
        source_path,
    )
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    function_classes = _function_imported_callable_class_semantics(
        functions,
        source_path,
    )
    return any(
        _CALLABLE_PROVENANCE in semantics
        for semantics in [
            *module_classes.values(),
            *(
                semantics
                for classes in function_classes.values()
                for semantics in classes.values()
            ),
        ]
    )


@lru_cache(maxsize=None)
def _cached_authority_provenance_lines(source_path: Path) -> frozenset[int]:
    """Reuse one module's parsed tree and complete authority summary."""

    source_path = source_path.resolve()
    source = _source_text(source_path)
    if not _source_text_may_expose_provenance(
        source
    ) and not _source_or_imports_may_expose_provenance(
        source_path
    ) and not _source_references_repository_reexport(
        source_path
    ) and not _function_local_imports_expose_provenance(
        source_path
    ) and not _imported_callable_classes_expose_provenance(source_path):
        return frozenset()
    tree = _parsed_module(source_path)
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


def test_provenance_scanner_covers_module_and_class_body_decisions() -> None:
    tree = ast.parse(
        "if get_current_chain():\n"
        "    terminate_agent(target)\n\n"
        "class Bootstrap:\n"
        "    if get_current_chain():\n"
        "        stop_peer(target)\n"
    )

    assert _authority_provenance_lines(tree) == {1, 5}


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
    transformed_accessor = ast.parse(
        "def dispatch(target):\n"
        "    allowed = bool(get_current_chain())\n"
        "    if allowed:\n"
        "        terminate_child(target)\n"
    )

    assert _authority_provenance_lines(authority_vocabulary) == {2, 6}
    assert _authority_provenance_lines(boolean_reducers) == {6, 11}
    assert _authority_provenance_lines(neutral_predicate) == {3}
    assert _authority_provenance_lines(chained_neutral_predicate) == {4}
    assert _authority_provenance_lines(selected_callback) == {3, 5}
    assert _authority_provenance_lines(short_circuit_controls) == {2, 5}
    assert _authority_provenance_lines(comprehension_control) == {2}
    assert _authority_provenance_lines(propagation_only_helper) == set()
    assert _authority_provenance_lines(neutral_control_forms) == {3, 9, 10, 14}
    assert _authority_provenance_lines(ownership_predicate) == {2, 5}
    assert _authority_provenance_lines(guard_clauses) == {2, 7, 15}
    assert _authority_provenance_lines(loop_guard_clauses) == {3, 9}
    assert _authority_provenance_lines(match_guards) == {3, 9}
    assert _authority_provenance_lines(branch_assigned_decision) == {5}
    assert _authority_provenance_lines(transformed_accessor) == {2, 3}


def test_provenance_scanner_taints_match_pattern_bindings() -> None:
    tree = ast.parse(
        "def mapping(request, target):\n"
        "    match request:\n"
        "        case {'causation_chain': chain}:\n"
        "            if chain:\n"
        "                terminate_child(target)\n\n"
        "def class_pattern(frame, target):\n"
        "    match frame:\n"
        "        case CausationFrame(parent=parent):\n"
        "            if parent:\n"
        "                terminate_child(target)\n\n"
        "def mapping_as(request, target):\n"
        "    match request:\n"
        "        case {'causation_chain': chain} as payload:\n"
        "            if payload:\n"
        "                terminate_child(target)\n"
    )

    assert _authority_provenance_lines(tree) == {2, 4, 8, 10, 14, 16}


def test_provenance_scanner_inspects_authorization_decorators() -> None:
    tree = ast.parse(
        "@authorized_by(get_current_chain())\n"
        "def terminate_agent(target):\n"
        "    target.stop()\n"
    )

    assert _authority_provenance_lines(tree) == {1}


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
        "target, allowed=bool(get_current_chain())",
        "target, provider=_get_current_chain",
        "target, *, allowed=bool(causation_chain)",
    ],
    ids=[
        "positional-value",
        "transformed-accessor-value",
        "accessor-callable",
        "keyword-only-value",
    ],
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


@pytest.mark.parametrize(
    "source",
    (
        "FLAGS = {}\n"
        "def capture(request):\n"
        "    FLAGS.update({'allowed': bool(request.causation_chain)})\n\n"
        "def run(target):\n"
        "    if FLAGS['allowed']:\n"
        "        target.stop()\n",
        "class Box: pass\n"
        "BOX = Box()\n"
        "def capture(request):\n"
        "    BOX.value = bool(request.causation_chain)\n\n"
        "def run(target):\n"
        "    if BOX.value:\n"
        "        target.stop()\n",
        "def build():\n"
        "    ready = False\n"
        "    def capture(request):\n"
        "        nonlocal ready\n"
        "        ready = bool(request.causation_chain)\n"
        "    def run(target):\n"
        "        if ready:\n"
        "            target.stop()\n",
        "def build():\n"
        "    flags = {}\n"
        "    def capture(request):\n"
        "        flags.update({'allowed': bool(request.causation_chain)})\n"
        "    def run(target):\n"
        "        if flags['allowed']:\n"
        "            target.stop()\n",
        "FLAGS = {}\n"
        "def capture(request):\n"
        "    alias = FLAGS\n"
        "    alias['ready'] = bool(request.causation_chain)\n\n"
        "def run(target):\n"
        "    if FLAGS['ready']:\n"
        "        target.stop()\n",
        "def build():\n"
        "    flags = {}\n"
        "    def capture(request):\n"
        "        alias = flags\n"
        "        alias['ready'] = bool(request.causation_chain)\n"
        "    def run(target):\n"
        "        if flags['ready']:\n"
        "            target.stop()\n",
        "class Gate:\n"
        "    flags = {}\n"
        "    def capture(self, request):\n"
        "        alias = self.flags\n"
        "        alias['ready'] = bool(request.causation_chain)\n"
        "    def run(self, target):\n"
        "        if self.flags['ready']:\n"
        "            target.stop()\n",
    ),
)
def test_provenance_scanner_propagates_shared_mutable_state(
    source: str,
) -> None:
    tree = ast.parse(source)
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If))

    assert _authority_provenance_lines(tree) == {guard.lineno}


def test_local_state_does_not_taint_same_named_module_binding() -> None:
    tree = ast.parse(
        "FLAGS = {}\n"
        "def capture(request):\n"
        "    FLAGS = {}\n"
        "    FLAGS.update({'allowed': bool(request.causation_chain)})\n\n"
        "def run(target):\n"
        "    if FLAGS.get('allowed'):\n"
        "        target.stop()\n"
    )

    assert _authority_provenance_lines(tree) == set()


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


def test_provenance_scanner_follows_registered_control_callbacks() -> None:
    tree = ast.parse(
        "def named(request, child):\n"
        "    action = child.stop\n"
        "    if request.causation_chain:\n"
        "        register(action)\n\n"
        "def wrapped(request, child):\n"
        "    action = partial(child.stop)\n"
        "    if request.causation_chain:\n"
        "        register(action)\n\n"
        "def captured(request, child):\n"
        "    action = lambda: child.stop()\n"
        "    if request.causation_chain:\n"
        "        register(action)\n\n"
        "def inline(request, child):\n"
        "    if request.causation_chain:\n"
        "        register(lambda: child.stop())\n\n"
        "def benign(request):\n"
        "    action = lambda: record_metric()\n"
        "    if request.causation_chain:\n"
        "        register(action)\n"
    )

    assert _authority_provenance_lines(tree) == {3, 8, 13, 17}


def test_provenance_scanner_follows_loop_bound_control_callbacks() -> None:
    tree = ast.parse(
        "def direct(request, target):\n"
        "    if request.causation_chain:\n"
        "        for callback in [terminate_child]:\n"
        "            callback(target)\n\n"
        "async def aliased(request, target):\n"
        "    callbacks = [stop_peer]\n"
        "    if request.orchestrator:\n"
        "        async for callback in callbacks:\n"
        "            callback(target)\n"
    )

    assert _authority_provenance_lines(tree) == {2, 8}


@pytest.mark.parametrize(
    "loop",
    (
        "for _, callback in enumerate(callbacks):",
        "for callback in iter(callbacks):",
        "for callback in map(identity, callbacks):",
        "for callback in callbacks.values():",
        "for _, callback in callbacks.items():",
        "for callback in (item for item in callbacks):",
    ),
)
def test_provenance_scanner_preserves_controls_through_iterable_adapters(
    loop: str,
) -> None:
    setup = (
        "callbacks = {'stop': target.shutdown}"
        if ".values()" in loop or ".items()" in loop
        else "callbacks = [target.shutdown]"
    )
    tree = ast.parse(
        "def dispatch(request, target):\n"
        f"    {setup}\n"
        f"    {loop}\n"
        "        if request.causation_chain:\n"
        "            callback()\n"
    )
    benign_tree = ast.parse(
        "def dispatch(request, target):\n"
        f"    {setup.replace('target.shutdown', 'record_metric')}\n"
        f"    {loop}\n"
        "        if request.causation_chain:\n"
        "            callback()\n"
    )

    assert _authority_provenance_lines(tree) == {4}
    assert _authority_provenance_lines(benign_tree) == set()


def test_provenance_scanner_follows_conditionally_selected_controls() -> None:
    tree = ast.parse(
        "def dispatch(request, target, enabled):\n"
        "    callback = terminate_child if enabled else noop\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n"
    )

    assert _authority_provenance_lines(tree) == {3}


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
    selected_target = ast.parse(
        "def dispatch(request, target):\n"
        "    selected = []\n"
        "    if request.causation_chain:\n"
        "        selected.append(target)\n"
        "    for item in selected:\n"
        "        item.shutdown()\n"
    )

    assert _authority_provenance_lines(append_decision) == {4}
    assert _authority_provenance_lines(update_decision) == {6}
    assert _authority_provenance_lines(setitem_decision) == {6}
    assert _authority_provenance_lines(setattr_decision) == {5}
    assert _authority_provenance_lines(aliased_decision) == {5}
    assert _authority_provenance_lines(rebound_alias_decision) == {6}
    assert _authority_provenance_lines(selected_target) == {5, 6}


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


def test_provenance_scanner_follows_future_control_callbacks() -> None:
    tree = ast.parse(
        "def stop_child(_future):\n"
        "    terminate_child('victim')\n\n"
        "def dispatch(request, future):\n"
        "    if request.causation_chain:\n"
        "        future.add_done_callback(stop_child)\n"
    )

    assert _authority_provenance_lines(tree) == {5}


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


def test_provenance_scanner_follows_controls_selected_from_literals() -> None:
    mapping = ast.parse(
        "def dispatch(request, target):\n"
        "    {True: terminate_child, False: record_metric}[\n"
        "        bool(request.causation_chain)\n"
        "    ](target)\n"
    )
    sequence = ast.parse(
        "def dispatch(request, target):\n"
        "    (record_metric, terminate_child)[\n"
        "        bool(request.causation_chain)\n"
        "    ](target)\n"
    )
    benign = ast.parse(
        "def dispatch(request, target):\n"
        "    {True: record_metric, False: record_event}[\n"
        "        bool(request.causation_chain)\n"
        "    ](target)\n"
    )

    assert _authority_provenance_lines(mapping) == {2}
    assert _authority_provenance_lines(sequence) == {2}
    assert _authority_provenance_lines(benign) == set()


def test_provenance_scanner_retains_partial_bound_provenance() -> None:
    controlled = ast.parse(
        "from functools import partial\n"
        "def dispatch(request):\n"
        "    operation = partial(\n"
        "        terminate_child, request.causation_chain[-1].agent_id\n"
        "    )\n"
        "    operation()\n"
    )
    qualified = ast.parse(
        "import functools\n"
        "def dispatch(request):\n"
        "    operation = functools.partial(\n"
        "        terminate_child, target=request.causation_chain[-1].agent_id\n"
        "    )\n"
        "    operation()\n"
    )
    benign = ast.parse(
        "from functools import partial\n"
        "def dispatch(request):\n"
        "    operation = partial(\n"
        "        record_metric, request.causation_chain[-1].agent_id\n"
        "    )\n"
        "    operation()\n"
    )

    assert _authority_provenance_lines(controlled) == {6}
    assert _authority_provenance_lines(qualified) == {6}
    assert _authority_provenance_lines(benign) == set()


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
    "invocation",
    (
        'subprocess.run(["kestrel", "terminate", target.name])',
        'asyncio.create_subprocess_exec("kestrel", "restart", target.name)',
        'os.system(f"kestrel restart {target.name}")',
        'shell(command=f"kestrel stop {target.name}")',
    ),
)
def test_provenance_scanner_recognizes_shell_lifecycle_commands(
    invocation: str,
) -> None:
    tree = ast.parse(
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        f"        {invocation}\n"
    )

    assert _authority_provenance_lines(tree) == {2}


def test_shell_payload_words_without_lifecycle_execution_are_benign() -> None:
    tree = ast.parse(
        "def dispatch(request):\n"
        "    if request.causation_chain:\n"
        '        subprocess.run(["echo", "kestrel terminate"])\n'
    )

    assert _authority_provenance_lines(tree) == set()


def test_provenance_scanner_recognizes_delegation_revocation_control_sink() -> None:
    tree = ast.parse(
        "def dispatch(request, delegation):\n"
        "    if request.causation_chain:\n"
        "        delegation.revoke()\n"
    )

    assert _authority_provenance_lines(tree) == {2}


def test_provenance_scanner_recognizes_agent_process_kills() -> None:
    controlled = ast.parse(
        "def dispatch(request, child):\n"
        "    if request.causation_chain:\n"
        "        os.kill(child.pid, signal.SIGTERM)\n"
    )
    benign = ast.parse(
        "def dispatch(request, pid):\n"
        "    if request.causation_chain:\n"
        "        os.kill(pid, signal.SIGTERM)\n"
    )

    assert _authority_provenance_lines(controlled) == {2}
    assert _authority_provenance_lines(benign) == set()


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
    serialized_helper = ast.parse(
        "def derive(request):\n"
        '    return "".join(\n'
        "        frame.source for frame in request.causation_chain\n"
        "    )\n\n"
        "def dispatch(request, target):\n"
        "    if derive(request):\n"
        "        terminate_child(target)\n"
    )
    normalized_helper = ast.parse(
        "def derive(request):\n"
        "    return normalize(request.causation_chain)\n\n"
        "def dispatch(request, target):\n"
        "    if derive(request):\n"
        "        terminate_child(target)\n"
    )

    assert _authority_provenance_lines(neutral_helper) == {5, 6}
    assert _authority_provenance_lines(wrapped_helper) == {9}
    assert _authority_provenance_lines(direct_helper_guard) == {5}
    assert _authority_provenance_lines(propagation_only_helper) == set()
    assert _authority_provenance_lines(serialized_helper) == {7}
    assert _authority_provenance_lines(normalized_helper) == {5}


def test_provenance_scanner_follows_neutral_parameter_return_wrappers() -> None:
    wrapped_parameter = ast.parse(
        "def wrap(value):\n"
        "    return bool(value)\n\n"
        "def derive(request):\n"
        "    return wrap(request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n"
    )
    ordinary_parameter = ast.parse(
        "def wrap(value):\n"
        "    return bool(value)\n\n"
        "def dispatch(request, child, configured):\n"
        "    if wrap(configured):\n"
        "        child.stop()\n"
    )

    assert _authority_provenance_lines(wrapped_parameter) == {8}
    assert _authority_provenance_lines(ordinary_parameter) == set()


def test_provenance_scanner_follows_provenance_helper_side_effects() -> None:
    direct = ast.parse(
        "def populate(state, request):\n"
        "    state['flag'] = bool(request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    state = {}\n"
        "    populate(state, request)\n"
        "    if state['flag']:\n"
        "        child.stop()\n"
    )
    parameter_flow = ast.parse(
        "def populate(state, value):\n"
        "    alias = state\n"
        "    alias['flag'] = bool(value)\n\n"
        "def dispatch(request, child):\n"
        "    state = {}\n"
        "    populate(state, request.causation_chain)\n"
        "    if state['flag']:\n"
        "        child.stop()\n"
    )
    attribute_write = ast.parse(
        "def populate(context, request):\n"
        "    context.flag = bool(request.causation_chain)\n\n"
        "def dispatch(request, child, context):\n"
        "    populate(context, request)\n"
        "    if context.flag:\n"
        "        child.stop()\n"
    )
    benign = ast.parse(
        "def populate(state, configured):\n"
        "    state['flag'] = bool(configured)\n\n"
        "def dispatch(request, child, configured):\n"
        "    state = {}\n"
        "    populate(state, configured)\n"
        "    if state['flag']:\n"
        "        child.stop()\n"
    )

    assert _authority_provenance_lines(direct) == {7}
    assert _authority_provenance_lines(parameter_flow) == {8}
    assert _authority_provenance_lines(attribute_write) == {6}
    assert _authority_provenance_lines(benign) == set()


def test_provenance_scanner_follows_assignment_inside_return_wrapper() -> None:
    tree = ast.parse(
        "def wrap(value):\n"
        "    result = bool(value)\n"
        "    return result\n\n"
        "def derive(request):\n"
        "    return wrap(request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n"
    )

    assert _authority_provenance_lines(tree) == {9}


def test_provenance_scanner_follows_returned_callback_parameters() -> None:
    positional = ast.parse(
        "def invoke(callback, value):\n"
        "    return callback(value)\n\n"
        "def derive(request):\n"
        "    return invoke(bool, request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n"
    )
    aliased_keyword = ast.parse(
        "def invoke(callback, value):\n"
        "    apply = callback\n"
        "    return apply(value)\n\n"
        "def derive(request):\n"
        "    return invoke(callback=bool, value=request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n"
    )

    assert _authority_provenance_lines(positional) == {8}
    assert _authority_provenance_lines(aliased_keyword) == {9}


def test_provenance_scanner_follows_container_writes_in_return_wrappers() -> None:
    subscript_write = ast.parse(
        "def wrap(value):\n"
        "    result = {}\n"
        "    result['decision'] = value\n"
        "    return result['decision']\n\n"
        "def derive(request):\n"
        "    return wrap(request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n"
    )
    aliased_mutator = ast.parse(
        "def wrap(value):\n"
        "    result = []\n"
        "    alias = result\n"
        "    alias.append(value)\n"
        "    return result[0]\n\n"
        "def derive(request):\n"
        "    return wrap(request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n"
    )

    assert _authority_provenance_lines(subscript_write) == {10}
    assert _authority_provenance_lines(aliased_mutator) == {11}


def test_provenance_scanner_binds_variadic_return_wrapper_arguments() -> None:
    positional = ast.parse(
        "def wrap(*values):\n"
        "    return bool(values)\n\n"
        "def derive(request):\n"
        "    return wrap(request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n"
    )
    keyword = ast.parse(
        "def wrap(**values):\n"
        "    return bool(values)\n\n"
        "def derive(request):\n"
        "    return wrap(value=request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n"
    )

    assert _authority_provenance_lines(positional) == {8}
    assert _authority_provenance_lines(keyword) == {8}


def test_provenance_scanner_binds_unpacked_fixed_wrapper_arguments() -> None:
    positional = ast.parse(
        "def wrap(prefix, value):\n"
        "    return bool(value)\n\n"
        "def derive(request):\n"
        "    return wrap(*[None, request.causation_chain])\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n"
    )
    keyword = ast.parse(
        "def wrap(*, value):\n"
        "    return bool(value)\n\n"
        "def derive(request):\n"
        "    return wrap(**{'value': request.causation_chain})\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n"
    )

    assert _authority_provenance_lines(positional) == {8}
    assert _authority_provenance_lines(keyword) == {8}


def test_provenance_scanner_binds_explicit_and_implicit_wrapper_receivers() -> None:
    explicit_receiver = ast.parse(
        "def wrap(self, value):\n"
        "    return bool(value)\n\n"
        "def derive(request, wrapper):\n"
        "    return wrap(wrapper, request.causation_chain)\n\n"
        "def dispatch(request, child, wrapper):\n"
        "    if derive(request, wrapper):\n"
        "        child.stop()\n"
    )
    implicit_receiver = ast.parse(
        "def wrap(self, value):\n"
        "    return bool(value)\n\n"
        "def derive(request, wrapper):\n"
        "    return wrapper.wrap(request.causation_chain)\n\n"
        "def dispatch(request, child, wrapper):\n"
        "    if derive(request, wrapper):\n"
        "        child.stop()\n"
    )
    static_receiver_name = ast.parse(
        "class Wrapper:\n"
        "    @staticmethod\n"
        "    def wrap(self):\n"
        "        return bool(self)\n\n"
        "def derive(request):\n"
        "    return Wrapper.wrap(request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n"
    )

    assert _authority_provenance_lines(explicit_receiver) == {8}
    assert _authority_provenance_lines(implicit_receiver) == {8}
    assert _authority_provenance_lines(static_receiver_name) == {10}


def test_provenance_scanner_follows_imported_parameter_return_wrappers(
    tmp_path: Path,
) -> None:
    helper_path = tmp_path / "helper.py"
    helper_path.write_text(
        "def wrap(value):\n"
        "    return bool(value)\n",
        encoding="utf-8",
    )
    controller_path = tmp_path / "controller.py"
    controller_path.write_text(
        "from .helper import wrap\n\n"
        "def derive(request):\n"
        "    return wrap(request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n",
        encoding="utf-8",
    )

    assert _cached_authority_provenance_lines(controller_path) == frozenset({7})


def test_provenance_scanner_resolves_parameter_return_wrapper_call_aliases(
    tmp_path: Path,
) -> None:
    local_alias = ast.parse(
        "def wrap(value):\n"
        "    return bool(value)\n\n"
        "def derive(request):\n"
        "    project = wrap\n"
        "    return project(request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n"
    )
    helper_path = tmp_path / "helper.py"
    helper_path.write_text(
        "def wrap(value):\n"
        "    return bool(value)\n",
        encoding="utf-8",
    )
    controller_path = tmp_path / "controller.py"
    controller_path.write_text(
        "from .helper import wrap\n\n"
        "def derive(request):\n"
        "    project = wrap\n"
        "    return project(request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n",
        encoding="utf-8",
    )
    module_controller_path = tmp_path / "module_controller.py"
    module_controller_path.write_text(
        "from . import helper\n\n"
        "def derive(request):\n"
        "    project = helper.wrap\n"
        "    return project(request.causation_chain)\n\n"
        "def dispatch(request, child):\n"
        "    if derive(request):\n"
        "        child.stop()\n",
        encoding="utf-8",
    )

    assert _authority_provenance_lines(local_alias) == {9}
    assert _cached_authority_provenance_lines(controller_path) == frozenset({8})
    assert _cached_authority_provenance_lines(module_controller_path) == frozenset({8})


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
    wrapped_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    callback = cast(Callable, terminate_child)\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n"
    )
    keyword_wrapped_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    callback = identity(value=terminate_child)\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n"
    )
    decorator_wrapped_callback = ast.parse(
        "def dispatch(request, target):\n"
        "    callback = wraps(terminate_child)(terminate_child)\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n"
    )

    assert _authority_provenance_lines(getattr_callback) == {3}
    assert _authority_provenance_lines(partial_callback) == {5}
    assert _authority_provenance_lines(direct_getattr) == {2}
    assert _authority_provenance_lines(wrapped_callback) == {3}
    assert _authority_provenance_lines(keyword_wrapped_callback) == {3}
    assert _authority_provenance_lines(decorator_wrapped_callback) == {3}


def test_provenance_scanner_follows_reflective_lifecycle_dispatch() -> None:
    tree = ast.parse(
        "def via_dunder(request, peer):\n"
        "    if request.causation_chain:\n"
        "        peer.__getattribute__('terminate')()\n\n"
        "def via_method_factory(request, peer):\n"
        "    if request.orchestrator:\n"
        "        operator.methodcaller('stop')(peer)\n\n"
        "def benign(request, peer):\n"
        "    if request.causation_chain:\n"
        "        operator.methodcaller('format')(peer)\n"
    )

    assert _authority_provenance_lines(tree) == {2, 6}


def test_provenance_scanner_recognizes_peer_targeted_control_calls() -> None:
    tree = ast.parse(
        "def route_request(request, router, requester, peer, message):\n"
        "    if request.causation_chain:\n"
        "        router.invoke(requester, peer, message)\n\n"
        "def deliver(request, peer):\n"
        "    if request.orchestrator:\n"
        "        peer.send('continue')\n\n"
        "def inspect(request, store, peer):\n"
        "    if request.causation_chain:\n"
        "        store.read(peer)\n\n"
        "def benign(request, metrics):\n"
        "    if request.causation_chain:\n"
        "        metrics.read()\n\n"
        "def first(request, peer):\n"
        "    if request.causation_chain:\n"
        "        peer.deploy()\n\n"
        "def second(request, peer):\n"
        "    if request.orchestrator:\n"
        "        peer.teardown()\n"
    )

    assert _authority_provenance_lines(tree) == {2, 6, 10, 18, 22}


def test_provenance_scanner_recognizes_provenance_selected_agent_reads() -> None:
    tree = ast.parse(
        "def status(request, agent_manager):\n"
        "    agent = agent_manager.get_agent(\n"
        "        request.causation_chain[-1].agent_id\n"
        "    )\n"
        "    return agent.status\n\n"
        "def indexed(request, peers):\n"
        "    selected = peers[request.causation_chain[-1].agent_id]\n"
        "    return selected.config\n"
    )

    assert _authority_provenance_lines(tree) == {5, 9}


def test_provenance_scanner_propagates_reflective_lifecycle_helper_names() -> None:
    tree = ast.parse(
        "def apply(obj, name):\n"
        "    getattr(obj, name)()\n\n"
        "def dispatch(request, child):\n"
        "    if request.causation_chain:\n"
        "        apply(child, 'terminate')\n"
    )

    assert _authority_provenance_lines(tree) == {5}


def test_provenance_scanner_resolves_imported_annotation_aliases() -> None:
    tree = ast.parse(
        "from kestrel_sdk.signals import CausationFrame as Frame\n\n"
        "def dispatch(context: Frame, child):\n"
        "    if context:\n"
        "        child.stop()\n"
    )

    assert _authority_provenance_lines(tree) == {4}


def test_provenance_scanner_follows_control_return_helpers(
    tmp_path: Path,
) -> None:
    local_factory = ast.parse(
        "def factory():\n"
        "    return terminate_child\n\n"
        "def dispatch(request, child):\n"
        "    action = factory()\n"
        "    if request.causation_chain:\n"
        "        action(child)\n"
    )
    chained_factory = ast.parse(
        "def first():\n"
        "    action = terminate_child\n"
        "    return action\n\n"
        "def second():\n"
        "    return first()\n\n"
        "def dispatch(request, child):\n"
        "    action = second()\n"
        "    if request.causation_chain:\n"
        "        action(child)\n"
    )
    helper_path = tmp_path / "control_factory.py"
    helper_path.write_text(
        "def factory():\n"
        "    return terminate_child\n",
        encoding="utf-8",
    )
    controller_path = tmp_path / "controller.py"
    controller_path.write_text(
        "from .control_factory import factory\n\n"
        "def dispatch(request, child):\n"
        "    action = factory()\n"
        "    if request.causation_chain:\n"
        "        action(child)\n",
        encoding="utf-8",
    )

    assert _authority_provenance_lines(local_factory) == {6}
    assert _authority_provenance_lines(chained_factory) == {10}
    assert _cached_authority_provenance_lines(controller_path) == frozenset({5})


def test_provenance_scanner_detects_guarded_control_callable_returns() -> None:
    selected = ast.parse(
        "def choose(request):\n"
        "    if request.causation_chain:\n"
        "        return terminate_child\n"
        "    return record_metric\n\n"
        "def dispatch(request, target):\n"
        "    choose(request)(target)\n"
    )
    benign = ast.parse(
        "def choose(configured):\n"
        "    if configured:\n"
        "        return terminate_child\n"
        "    return record_metric\n"
    )

    assert _authority_provenance_lines(selected) == {2}
    assert _authority_provenance_lines(benign) == set()


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


def test_provenance_scanner_reuses_central_lifecycle_for_imports() -> None:
    imported_disable_alias = ast.parse(
        "from lifecycle import disable_agent as apply\n\n"
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        apply(target)\n"
    )
    imported_create_alias = ast.parse(
        "from lifecycle import create_agent as apply\n\n"
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        apply(target)\n"
    )

    assert _authority_provenance_lines(imported_disable_alias) == {4}
    assert _authority_provenance_lines(imported_create_alias) == {4}


def test_provenance_scanner_analyzes_escaping_control_lambdas() -> None:
    tree = ast.parse(
        "def returned(request, target):\n"
        "    return lambda: (\n"
        "        terminate_child(target) if request.causation_chain else None\n"
        "    )\n\n"
        "def registered(request, target):\n"
        "    register(\n"
        "        lambda: stop_peer(target) if request.orchestrator else None\n"
        "    )\n"
    )

    assert _authority_provenance_lines(tree) == {2, 8}


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
    immediately_invoked = ast.parse(
        "def dispatch(self, target):\n"
        "    if getattr(self.agent, '_provide_causation_chain', None)():\n"
        "        terminate_child(target)\n"
    )

    assert _authority_provenance_lines(assigned_result) == {4}
    assert _authority_provenance_lines(direct_guard) == {3}
    assert _authority_provenance_lines(immediately_invoked) == {2}


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


def test_provenance_scanner_follows_helpers_reexported_by_facades(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "pkg"
    package_path.mkdir()
    (package_path / "lineage.py").write_text(
        "def derive(request):\n"
        "    return bool(request.causation_chain)\n",
        encoding="utf-8",
    )
    (package_path / "controls.py").write_text(
        "def apply(target):\n"
        "    terminate_child(target)\n",
        encoding="utf-8",
    )
    (package_path / "__init__.py").write_text(
        "from .lineage import derive as _derive\n"
        "decide = _derive\n"
        "from .controls import apply\n",
        encoding="utf-8",
    )
    provenance_path = tmp_path / "provenance_controller.py"
    provenance_path.write_text(
        "from pkg import decide\n\n"
        "def dispatch(request, target):\n"
        "    if decide(request):\n"
        "        terminate_child(target)\n",
        encoding="utf-8",
    )
    control_path = tmp_path / "control_controller.py"
    control_path.write_text(
        "from pkg import apply\n\n"
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        apply(target)\n",
        encoding="utf-8",
    )

    assert _cached_authority_provenance_lines(provenance_path) == frozenset({4})
    assert _cached_authority_provenance_lines(control_path) == frozenset({4})


def test_provenance_scanner_follows_imported_class_provenance_helpers(
    tmp_path: Path,
) -> None:
    lineage_path = tmp_path / "lineage.py"
    lineage_path.write_text(
        "class Context:\n"
        "    @staticmethod\n"
        "    def derive(request):\n"
        "        return bool(request.causation_chain)\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "controller.py"
    source_path.write_text(
        "from .lineage import Context\n\n"
        "def dispatch(request, manager, target):\n"
        "    if Context.derive(request):\n"
        "        manager.kill_process(target)\n",
        encoding="utf-8",
    )

    assert _cached_authority_provenance_lines(source_path) == frozenset({4})

    aliased_path = tmp_path / "aliased_controller.py"
    aliased_path.write_text(
        "from .lineage import Context\n"
        "LineageContext = Context\n\n"
        "def dispatch(request, manager, target):\n"
        "    if LineageContext.derive(request):\n"
        "        manager.kill_process(target)\n",
        encoding="utf-8",
    )

    assert _cached_authority_provenance_lines(aliased_path) == frozenset({5})


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


def test_provenance_scanner_follows_function_local_imports_without_scope_leaks(
    tmp_path: Path,
) -> None:
    (tmp_path / "controls.py").write_text(
        "def apply(target):\n"
        "    terminate_child(target)\n",
        encoding="utf-8",
    )
    (tmp_path / "lineage.py").write_text(
        "def decide(request):\n"
        "    return bool(request.causation_chain)\n",
        encoding="utf-8",
    )
    control_path = tmp_path / "local_control.py"
    control_path.write_text(
        "def dispatch(request, target):\n"
        "    from controls import apply\n"
        "    if request.causation_chain:\n"
        "        apply(target)\n\n"
        "def sibling(request, target):\n"
        "    if request.causation_chain:\n"
        "        apply(target)\n",
        encoding="utf-8",
    )
    provenance_path = tmp_path / "local_provenance.py"
    provenance_path.write_text(
        "def dispatch(request, target):\n"
        "    from lineage import decide\n"
        "    allowed = decide(request)\n"
        "    if allowed:\n"
        "        terminate_child(target)\n",
        encoding="utf-8",
    )

    assert _cached_authority_provenance_lines(control_path) == frozenset({3})
    assert _cached_authority_provenance_lines(provenance_path) == frozenset(
        {3, 4}
    )


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


def test_provenance_scanner_preserves_post_loop_control_reachability() -> None:
    break_reaches_control = ast.parse(
        "def dispatch(request, target):\n"
        "    while True:\n"
        "        if request.causation_chain:\n"
        "            break\n"
        "        return\n"
        "    terminate_child(target)\n"
    )
    both_paths_return = ast.parse(
        "def dispatch(request, target):\n"
        "    while True:\n"
        "        if request.causation_chain:\n"
        "            return\n"
        "        return\n"
        "    terminate_child(target)\n"
    )

    assert _authority_provenance_lines(break_reaches_control) == {3}
    assert _authority_provenance_lines(both_paths_return) == set()


def test_provenance_scanner_tracks_augmented_assignment_decisions() -> None:
    local_decision = ast.parse(
        "def dispatch(request, child):\n"
        "    allowed = True\n"
        "    allowed &= bool(request.causation_chain)\n"
        "    if allowed:\n"
        "        child.terminate()\n"
    )
    attribute_decision = ast.parse(
        "def dispatch(self, request, child):\n"
        "    self.allowed |= bool(request.causation_chain)\n"
        "    if self.allowed:\n"
        "        child.terminate()\n"
    )

    assert _authority_provenance_lines(local_decision) == {4}
    assert _authority_provenance_lines(attribute_decision) == {3}


def test_provenance_scanner_tracks_post_while_control_reachability() -> None:
    negated_condition = ast.parse(
        "def dispatch(request, child):\n"
        "    while not request.causation_chain:\n"
        "        pass\n"
        "    child.terminate()\n"
    )
    positive_condition = ast.parse(
        "def dispatch(request, child):\n"
        "    while request.causation_chain:\n"
        "        pass\n"
        "    child.terminate()\n"
    )
    finite_iteration = ast.parse(
        "def dispatch(request, child):\n"
        "    for frame in request.causation_chain:\n"
        "        record(frame)\n"
        "    child.terminate()\n"
    )

    assert _authority_provenance_lines(negated_condition) == {2}
    assert _authority_provenance_lines(positive_condition) == {2}
    assert _authority_provenance_lines(finite_iteration) == set()


def test_provenance_scanner_follows_exception_guard_clause_exits() -> None:
    tree = ast.parse(
        "def dispatch(request, child):\n"
        "    try:\n"
        "        validate(request.causation_chain)\n"
        "    except ValueError:\n"
        "        return\n"
        "    child.stop()\n"
    )

    assert _authority_provenance_lines(tree) == {2}


def test_provenance_scanner_detects_try_selected_controls() -> None:
    tree = ast.parse(
        "def success(request, target):\n"
        "    try:\n"
        "        request.causation_chain[-1]\n"
        "        terminate_child(target)\n"
        "    except IndexError:\n"
        "        pass\n\n"
        "def failure(request, target):\n"
        "    try:\n"
        "        request.causation_chain[-1]\n"
        "    except IndexError:\n"
        "        stop_peer(target)\n"
    )

    assert _authority_provenance_lines(tree) == {2, 9}


def test_provenance_scanner_recognizes_provenance_properties() -> None:
    tree = ast.parse(
        "class Context:\n"
        "    @property\n"
        "    def lineage(self):\n"
        "        return self.request.causation_chain\n\n"
        "def dispatch(context, target):\n"
        "    if context.lineage:\n"
        "        terminate_child(target)\n"
    )

    assert _authority_provenance_lines(tree) == {7}


@pytest.mark.parametrize(
    "lookup",
    (
        "task.metadata.get('sender')",
        "task.metadata['claimed_sender']",
        "metadata.get('source_agent_id')",
    ),
)
def test_unverified_attribution_metadata_is_provenance(
    lookup: str,
) -> None:
    tree = ast.parse(
        "def dispatch(task, metadata, target):\n"
        f"    if {lookup}:\n"
        "        target.shutdown()\n"
    )

    assert _authority_provenance_lines(tree) == {2}


def test_verified_sender_principals_require_verified_verdict_fields() -> None:
    verified = ast.parse(
        "from kestrel_sovereign.a2a.envelope_signing import (\n"
        "    verify_inbound_envelope,\n"
        ")\n\n"
        "async def dispatch(task, target):\n"
        "    sender_verdict = await verify_inbound_envelope(task.metadata)\n"
        "    if sender_verdict.sender:\n"
        "        target.shutdown()\n"
    )
    verification_flag = ast.parse(
        "def dispatch(task, target):\n"
        "    if task.metadata.get('sender_verified'):\n"
        "        target.shutdown()\n"
    )
    accepted_only = ast.parse(
        "from kestrel_sovereign.a2a.envelope_signing import (\n"
        "    verify_inbound_envelope,\n"
        ")\n\n"
        "async def dispatch(task, target):\n"
        "    sender_verdict = await verify_inbound_envelope(task.metadata)\n"
        "    if sender_verdict.ok:\n"
        "        target.shutdown()\n"
    )
    verified_only = ast.parse(
        "from kestrel_sovereign.a2a.envelope_signing import (\n"
        "    verify_inbound_envelope,\n"
        ")\n\n"
        "async def dispatch(task, target):\n"
        "    sender_verdict = await verify_inbound_envelope(task.metadata)\n"
        "    if sender_verdict.verified:\n"
        "        target.shutdown()\n"
    )

    assert _authority_provenance_lines(verified) == {7}
    assert _authority_provenance_lines(verification_flag) == {2}
    assert _authority_provenance_lines(accepted_only) == {7}
    assert _authority_provenance_lines(verified_only) == {7}


def test_envelope_acceptance_requires_dominating_sender_authorization() -> None:
    authorization = (
        "    if verdict.verified:\n"
        "        verified_sender = await _authorize_verified_a2a_sender(\n"
        "            manager, verdict.sender\n"
        "        )\n"
        "        if not verified_sender:\n"
        "            raise PermissionError\n"
        "    else:\n"
        "        authorize_legacy = getattr(\n"
        "            manager,\n"
        "            'authorize_a2a_legacy_unsigned_sender',\n"
        "        )\n"
        "        authorized = await authorize_legacy(task)\n"
        "        if not authorized:\n"
        "            raise PermissionError\n"
    )
    header = (
        "from kestrel_sovereign.a2a.envelope_signing import (\n"
        "    verify_inbound_envelope,\n"
        ")\n\n"
        "async def dispatch(task, target, manager):\n"
        "    verdict = await verify_inbound_envelope(task.metadata)\n"
    )
    prefix = (
        header
        + "    if not verdict.ok:\n"
        "        raise PermissionError\n"
    )
    safe = ast.parse(prefix + authorization + "    target.shutdown()\n")
    unsafe = ast.parse(
        prefix + "    target.shutdown()\n" + authorization
    )
    unsafe_rejection = ast.parse(
        header
        + "    if not verdict.ok:\n"
        "        target.shutdown()\n"
        "        raise PermissionError\n"
        + authorization
    )
    unauthenticated_verified_arm = ast.parse(
        prefix
        + "    if verdict.verified:\n"
        "        pass\n"
        "    else:\n"
        "        authorize_legacy = getattr(\n"
        "            manager,\n"
        "            'authorize_a2a_legacy_unsigned_sender',\n"
        "        )\n"
        "        authorized = await authorize_legacy(task)\n"
        "        if not authorized:\n"
        "            raise PermissionError\n"
        "    target.shutdown()\n"
    )
    unrelated_legacy_source = (
        prefix
        + "    if verdict.verified:\n"
        "        verified_sender = await _authorize_verified_a2a_sender(\n"
        "            manager, verdict.sender\n"
        "        )\n"
        "        if not verified_sender:\n"
        "            raise PermissionError\n"
        "    else:\n"
        "        authorize_legacy = getattr(\n"
        "            manager,\n"
        "            'authorize_a2a_legacy_unsigned_sender',\n"
        "        )\n"
        "        authorized = await authorize_legacy(task)\n"
        "        if debug:\n"
        "            raise PermissionError\n"
    )
    unrelated_legacy_raise = ast.parse(
        unrelated_legacy_source + "    target.shutdown()\n"
    )
    unrelated_legacy_a2a_commit = ast.parse(
        unrelated_legacy_source + "    await manager.create_task()\n"
    )
    conditional_legacy_authorizer = ast.parse(
        prefix
        + "    if verdict.verified:\n"
        "        verified_sender = await _authorize_verified_a2a_sender(\n"
        "            manager, verdict.sender\n"
        "        )\n"
        "        if not verified_sender:\n"
        "            raise PermissionError\n"
        "    else:\n"
        "        authorize_legacy = getattr(\n"
        "            manager,\n"
        "            'authorize_a2a_legacy_unsigned_sender',\n"
        "        )\n"
        "        if debug:\n"
        "            authorized = await authorize_legacy(task)\n"
        "        else:\n"
        "            raise PermissionError\n"
        "    target.shutdown()\n"
    )

    assert _authority_provenance_lines(safe) == set()
    assert _authority_provenance_lines(unsafe) == {7}
    assert _authority_provenance_lines(unsafe_rejection) == {7}
    assert _authority_provenance_lines(unauthenticated_verified_arm) == {7}
    assert _authority_provenance_lines(unrelated_legacy_raise) == {7}
    assert _authority_provenance_lines(unrelated_legacy_a2a_commit) == {7}
    assert _authority_provenance_lines(conditional_legacy_authorizer) == {7}


def test_unverified_sender_alias_remains_provenance_until_validation() -> None:
    guarded = ast.parse(
        "def dispatch(task, target):\n"
        "    claimed = str(task.metadata.get('sender') or '')\n"
        "    if claimed:\n"
        "        target.shutdown()\n"
    )
    validation = ast.parse(
        "async def validate(task, target, manager):\n"
        "    claimed = str(task.metadata.get('sender') or '')\n"
        "    authorize_legacy = getattr(\n"
        "        manager,\n"
        "        'authorize_a2a_legacy_unsigned_sender',\n"
        "        None,\n"
        "    )\n"
        "    authorized = await authorize_legacy(target, claimed)\n"
        "    if authorized:\n"
        "        target.shutdown()\n"
    )
    inline_validation = ast.parse(
        "from kestrel_sovereign.a2a.envelope_signing import (\n"
        "    verify_inbound_envelope,\n"
        ")\n\n"
        "async def dispatch(task, target):\n"
        "    verdict = await verify_inbound_envelope(task.metadata)\n"
        "    if verdict.sender:\n"
        "        target.shutdown()\n"
    )
    manager_witness = ast.parse(
        "def dispatch(task, target, manager):\n"
        "    claimed = task.metadata['sender']\n"
        "    witness = manager.a2a_sender_identity_witness(claimed)\n"
        "    if witness:\n"
        "        target.shutdown()\n"
    )
    fake_validation = ast.parse(
        "def dispatch(task, target, verify_sender, authorize_agent, peer):\n"
        "    claimed = task.metadata['sender']\n"
        "    if verify_sender(claimed):\n"
        "        target.shutdown()\n"
        "    if authorize_agent(claimed):\n"
        "        target.shutdown()\n"
        "    if peer.a2a_sender_identity_witness(claimed):\n"
        "        target.shutdown()\n"
    )

    assert _authority_provenance_lines(guarded) == {3}
    assert _authority_provenance_lines(validation) == set()
    assert _authority_provenance_lines(inline_validation) == {7}
    assert _authority_provenance_lines(manager_witness) == {4}
    assert _authority_provenance_lines(fake_validation) == {3, 5, 7}


@pytest.mark.parametrize(
    "source",
    (
        "def dispatch(task, target):\n"
        "    attrs = task.metadata\n"
        "    claimed = attrs['sender']\n"
        "    if claimed:\n"
        "        target.shutdown()\n",
        "SENDER_KEY = 'sender'\n\n"
        "def dispatch(task, target):\n"
        "    attrs = task.metadata\n"
        "    claimed = attrs[SENDER_KEY]\n"
        "    if claimed:\n"
        "        target.shutdown()\n",
        "def dispatch(task, target, sender_key='sender'):\n"
        "    attrs = task.metadata\n"
        "    claimed = attrs[sender_key]\n"
        "    if claimed:\n"
        "        target.shutdown()\n",
        "def dispatch(task, target):\n"
        "    sender_key = 'sender'\n"
        "    attrs = task.metadata\n"
        "    claimed = attrs[sender_key]\n"
        "    if claimed:\n"
        "        target.shutdown()\n",
        "def dispatch(task, target):\n"
        "    match task.metadata:\n"
        "        case {'sender': claimed}:\n"
        "            if claimed:\n"
        "                target.shutdown()\n",
    ),
)
def test_unverified_attribution_preserves_mapping_and_key_aliases(
    source: str,
) -> None:
    tree = ast.parse(source)
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If))

    assert _authority_provenance_lines(tree) == {guard.lineno}


def test_unverified_attribution_flows_through_neutral_return_helpers() -> None:
    tree = ast.parse(
        "def select(attrs):\n"
        "    return attrs['sender']\n\n"
        "def dispatch(task, target):\n"
        "    claimed = select(task.metadata)\n"
        "    if claimed:\n"
        "        target.shutdown()\n"
    )
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If))

    assert _authority_provenance_lines(tree) == {guard.lineno}


@pytest.mark.parametrize(
    "invocation",
    (
        "task_manager.list_tasks(recipient_agent_id=params.metadata['sender'])",
        "task_manager.get_task(\n"
        "        params.id, recipient_agent_id=params.metadata['sender']\n"
        "    )",
    ),
)
def test_unverified_recipient_scoped_task_reads_are_authority_sinks(
    invocation: str,
) -> None:
    tree = ast.parse(
        "async def read(params, task_manager):\n"
        f"    return await {invocation}\n"
    )

    assert _authority_provenance_lines(tree) == {2}


def test_cached_a2a_sender_claim_guard_reaches_the_ci_gate(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sender_gate.py"
    source_path.write_text(
        "def dispatch(task, target):\n"
        "    if task.metadata.get('sender'):\n"
        "        target.shutdown()\n",
        encoding="utf-8",
    )

    assert _cached_authority_provenance_lines(source_path) == frozenset({2})


def test_attribution_analysis_discards_annotations_from_prior_passes() -> None:
    unverified_tree = ast.parse(
        "def dispatch(task, target):\n"
        "    claimed = task.metadata.get('sender')\n"
        "    if claimed:\n"
        "        target.shutdown()\n"
    )
    unverified_guard = next(
        node for node in ast.walk(unverified_tree) if isinstance(node, ast.If)
    )

    assert _authority_provenance_lines(unverified_tree) == {
        unverified_guard.lineno
    }
    setattr(unverified_guard.test, "_authority_verified_attribution", True)
    assert _authority_provenance_lines(unverified_tree) == {
        unverified_guard.lineno
    }

    verified_tree = ast.parse(
        "async def dispatch(task, target, manager):\n"
        "    claimed = task.metadata['sender']\n"
        "    authorized = await manager.authorize_a2a_legacy_unsigned_sender(\n"
        "        target, claimed\n"
        "    )\n"
        "    if authorized:\n"
        "        target.shutdown()\n"
    )
    verified_guard = next(
        node for node in ast.walk(verified_tree) if isinstance(node, ast.If)
    )

    assert _authority_provenance_lines(verified_tree) == set()
    setattr(verified_guard.test, "_authority_unverified_attribution", True)
    assert _authority_provenance_lines(verified_tree) == set()


def test_unverified_attribution_dominates_conflicting_annotation() -> None:
    value = ast.parse("claimed_sender", mode="eval").body
    setattr(value, "_authority_verified_attribution", True)
    setattr(value, "_authority_unverified_attribution", True)

    assert _is_unverified_attribution_metadata_lookup(value)


def test_cached_ast_and_import_summaries_ignore_analysis_annotations() -> None:
    verifier_path = (
        REPO_ROOT / "kestrel_sovereign/a2a/envelope_signing.py"
    ).resolve()
    _direct_provenance_helper_names.cache_clear()
    try:
        cached_tree = _parsed_module(verifier_path)
        _authority_provenance_lines(
            cached_tree, verifier_path
        )
        assert not any(
            any(name.startswith("_authority_") for name in vars(node))
            for node in ast.walk(cached_tree)
        )
        assert "verify_inbound_envelope" not in (
            _direct_provenance_helper_names(verifier_path)
        )
    finally:
        _direct_provenance_helper_names.cache_clear()


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
    direct_accessor_reference = ast.parse(
        "def dispatch(request, target):\n"
        "    provider = request._provide_causation_chain\n"
        "    if callable(provider):\n"
        "        target.shutdown()\n"
    )
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
    configured_transport = ast.parse(
        "class Agent:\n"
        "    def configure(self):\n"
        "        self.manager = TaskManager(\n"
        "            causation_chain_provider=self._provide_causation_chain\n"
        "        )\n\n"
        "    async def shutdown(self):\n"
        "        if self.manager:\n"
        "            await self.manager.close()\n"
    )

    assert _authority_provenance_lines(direct_accessor_reference) == set()
    assert _authority_provenance_lines(accessor_reference) == set()
    assert _authority_provenance_lines(accessor_transport) == set()
    assert _authority_provenance_lines(local_task_plumbing) == set()
    assert _authority_provenance_lines(configured_transport) == set()


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


def test_provenance_scanner_tracks_reflective_registry_mutations() -> None:
    guarded = ast.parse(
        "def dispatch(request, manager):\n"
        "    if request.causation_chain:\n"
        "        setattr(manager, '_agents', {})\n"
    )
    direct = ast.parse(
        "def dispatch(request, manager):\n"
        "    setattr(manager, '_children', request.causation_chain)\n"
    )
    bound = ast.parse(
        "def dispatch(request, manager):\n"
        "    if request.orchestrator:\n"
        "        manager.__setattr__('_peers', {})\n"
    )
    benign = ast.parse(
        "def dispatch(request, manager):\n"
        "    if request.causation_chain:\n"
        "        setattr(manager, '_metrics', {})\n"
    )

    assert _authority_provenance_lines(guarded) == {2}
    assert _authority_provenance_lines(direct) == {2}
    assert _authority_provenance_lines(bound) == {2}
    assert _authority_provenance_lines(benign) == set()


def test_provenance_scanner_tracks_dunder_agent_object_mutations() -> None:
    tree = ast.parse(
        "def unbound_set(request, child):\n"
        "    if request.causation_chain:\n"
        "        object.__setattr__(child, 'enabled', False)\n\n"
        "def unbound_delete(request, child):\n"
        "    if request.orchestrator:\n"
        "        object.__delattr__(child, 'session')\n\n"
        "def bound_set(request, child):\n"
        "    if request.causation_chain:\n"
        "        child.__setattr__('enabled', False)\n"
    )

    assert _authority_provenance_lines(tree) == {2, 6, 10}


def test_provenance_scanner_tracks_inline_resolved_agent_mutations() -> None:
    tree = ast.parse(
        "def assign(request, manager, name):\n"
        "    if request.causation_chain:\n"
        "        manager.get_agent(name).enabled = False\n\n"
        "def mutate(request, manager, name):\n"
        "    if request.orchestrator:\n"
        "        manager.resolve_child(name).permissions.clear()\n\n"
        "def benign(request, manager, name):\n"
        "    if request.causation_chain:\n"
        "        manager.get_feature(name).enabled = False\n"
    )

    assert _authority_provenance_lines(tree) == {2, 6}


def test_provenance_scanner_tracks_direct_provenance_state_mutations() -> None:
    tree = ast.parse(
        "def assign(request, child):\n"
        "    child.enabled = bool(request.causation_chain)\n\n"
        "def delete(request, manager):\n"
        "    del manager._agents[request.causation_chain[-1].agent_id]\n\n"
        "def augment(request, manager):\n"
        "    manager._agents[request.causation_chain[-1].agent_id] |= FLAG\n\n"
        "def set_attribute(request, child):\n"
        "    setattr(child, 'enabled', bool(request.causation_chain))\n\n"
        "def pop_entry(request, manager):\n"
        "    manager._agents.pop(request.causation_chain[-1].agent_id)\n"
    )

    assert _authority_provenance_lines(tree) == {2, 5, 8, 11, 14}


def test_provenance_scanner_summarizes_registry_mutation_helpers() -> None:
    tree = ast.parse(
        "def pop_entry(manager, agent_id):\n"
        "    manager._agents.pop(agent_id, None)\n\n"
        "def delete_entry(manager, agent_id):\n"
        "    del manager._agents[agent_id]\n\n"
        "def replace_entry(manager, agent_id):\n"
        "    manager._agents[agent_id] = None\n\n"
        "def dispatch(request, manager, agent_id):\n"
        "    if request.causation_chain:\n"
        "        pop_entry(manager, agent_id)\n"
        "    if request.orchestrator:\n"
        "        delete_entry(manager, agent_id)\n"
        "    if request.causation_chain:\n"
        "        replace_entry(manager, agent_id)\n"
    )

    assert _authority_provenance_lines(tree) == {11, 13, 15}


def test_provenance_scanner_classifies_agent_registration_lifecycle() -> None:
    tree = ast.parse(
        "def unregister(request, task_manager, target):\n"
        "    if request.causation_chain:\n"
        "        task_manager.unregister_agent(target)\n\n"
        "def register(request, task_manager, target):\n"
        "    if request.orchestrator:\n"
        "        task_manager.register_agent(target)\n"
    )

    assert _authority_provenance_lines(tree) == {2, 6}


def test_provenance_scanner_seeds_causation_typed_parameters() -> None:
    tree = ast.parse(
        "def positional(context: CausationFrame, child):\n"
        "    if context:\n"
        "        child.stop()\n\n"
        "def keyword(child, *, context: list[CausationFrame]):\n"
        "    if context:\n"
        "        child.stop()\n\n"
        "def forward(*context: 'CausationFrame', child):\n"
        "    if context:\n"
        "        child.stop()\n"
    )

    assert _authority_provenance_lines(tree) == {2, 6, 10}


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


def test_provenance_scanner_classifies_trace_parent_metadata(
    tmp_path: Path,
) -> None:
    source = (
        "def direct(request, target):\n"
        "    if request.trace_parent:\n"
        "        target.shutdown()\n\n"
        "def w3c(metadata, target):\n"
        "    if metadata.get('traceparent'):\n"
        "        terminate_child(target)\n\n"
        "def inverse(request, target):\n"
        "    if request.parent_trace_id:\n"
        "        target.shutdown()\n"
    )
    tree = ast.parse(source)
    source_path = tmp_path / "controller.py"
    source_path.write_text(source, encoding="utf-8")

    assert _authority_provenance_lines(tree) == {2, 6, 10}
    assert _cached_authority_provenance_lines(source_path) == frozenset(
        {2, 6, 10}
    )

    parent_span_source = (
        "def dispatch(request, target):\n"
        "    if request.parent_span_id:\n"
        "        target.shutdown()\n"
    )
    parent_span_tree = ast.parse(parent_span_source)
    parent_span_path = tmp_path / "parent_span_controller.py"
    parent_span_path.write_text(parent_span_source, encoding="utf-8")

    assert _authority_provenance_lines(parent_span_tree) == {2}
    assert _cached_authority_provenance_lines(
        parent_span_path
    ) == frozenset({2})

    dotted_source = (
        "def nested(request, target):\n"
        "    if request.trace.parent:\n"
        "        target.shutdown()\n\n"
        "def inverse(request, target):\n"
        "    if request.parent.span:\n"
        "        terminate_child(target)\n"
    )
    dotted_tree = ast.parse(dotted_source)
    dotted_path = tmp_path / "dotted_trace_parent_controller.py"
    dotted_path.write_text(dotted_source, encoding="utf-8")

    assert _authority_provenance_lines(dotted_tree) == {2, 6}
    assert _cached_authority_provenance_lines(dotted_path) == frozenset(
        {2, 6}
    )

    contextual_source = (
        "def nested(request, target):\n"
        "    if request.trace_context.parent:\n"
        "        target.shutdown()\n\n"
        "def inverse(request, target):\n"
        "    if request.parent_context.span:\n"
        "        terminate_child(target)\n"
    )
    contextual_path = tmp_path / "contextual_lineage_controller.py"
    contextual_path.write_text(contextual_source, encoding="utf-8")

    assert _cached_authority_provenance_lines(
        contextual_path
    ) == frozenset({2, 6})


def test_provenance_scanner_classifies_delegation_and_approval_boundaries() -> None:
    tree = ast.parse(
        "def grant_restart_delegation(request):\n"
        "    return bool(request.causation_chain)\n\n"
        "def approve_request(request):\n"
        "    return bool(request.causation_chain)\n\n"
        "def request_approval(request):\n"
        "    return bool(request.causation_chain)\n\n"
        "def permit_child(request):\n"
        "    return bool(request.causation_chain)\n"
    )

    assert _authority_provenance_lines(tree) == {2, 5, 8, 11}


@pytest.mark.parametrize(
    "method",
    [
        "start",
        "pause",
        "resume",
        "deactivate",
        "disable",
        "enable",
        "delete",
        "retire",
        "suspend",
    ],
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
    "method", ["aclose", "terminate_async", "stop_now", "shutdown_runtime"]
)
def test_provenance_scanner_classifies_async_lifecycle_method_variants(
    method: str,
) -> None:
    controlled = ast.parse(
        "async def dispatch(request, child):\n"
        "    if request.causation_chain:\n"
        f"        await child.{method}()\n"
    )
    benign = ast.parse(
        "async def dispatch(request, metrics):\n"
        "    if request.causation_chain:\n"
        f"        await metrics.{method}()\n"
    )

    assert _authority_provenance_lines(controlled) == {2}
    assert _authority_provenance_lines(benign) == set()


@pytest.mark.parametrize(
    "dispatch",
    [
        "manager.start(child)",
        "runtime.deactivate(child)",
        "manager.suspend(target=peer)",
        "restart(child)",
    ],
)
def test_provenance_scanner_classifies_generic_lifecycle_target_arguments(
    dispatch: str,
) -> None:
    controlled = ast.parse(
        "def dispatch(request, manager, child, peer):\n"
        "    if request.causation_chain:\n"
        f"        {dispatch}\n"
    )
    benign = ast.parse(
        "def dispatch(request, manager, metrics):\n"
        "    if request.causation_chain:\n"
        "        manager.start(metrics)\n"
    )

    assert _authority_provenance_lines(controlled) == {2}
    assert _authority_provenance_lines(benign) == set()


@pytest.mark.parametrize(
    "dispatch",
    [
        "threading.Thread(target=terminate_fleet).start()",
        "multiprocessing.Process(target=terminate_fleet).start()",
        "executor.submit(fn=terminate_fleet)",
        "itertools.starmap(terminate_fleet, targets)",
        "functools.reduce(function=terminate_fleet, iterable=targets)",
    ],
)
def test_provenance_scanner_classifies_deferred_callback_dispatch(
    dispatch: str,
) -> None:
    tree = ast.parse(
        "def dispatch(request, targets):\n"
        "    if request.causation_chain:\n"
        f"        {dispatch}\n"
    )

    assert _authority_provenance_lines(tree) == {2}


def test_provenance_scanner_binds_local_control_callback_parameters() -> None:
    positional = ast.parse(
        "def choose(request, callback, target):\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n\n"
        "def dispatch(request, target):\n"
        "    choose(request, terminate_child, target)\n"
    )
    keyword = ast.parse(
        "def choose(request, callback, target):\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n\n"
        "def dispatch(request, target):\n"
        "    choose(request, callback=terminate_child, target=target)\n"
    )
    benign = ast.parse(
        "def choose(request, callback, target):\n"
        "    if request.causation_chain:\n"
        "        callback(target)\n\n"
        "def dispatch(request, target):\n"
        "    choose(request, record_metric, target)\n"
    )

    assert _authority_provenance_lines(positional) == {2}
    assert _authority_provenance_lines(keyword) == {2}
    assert _authority_provenance_lines(benign) == set()


def test_provenance_scanner_follows_local_control_helper_call_aliases() -> None:
    controlled = ast.parse(
        "def apply(callback, target):\n"
        "    callback(target)\n\n"
        "def dispatch(request, target):\n"
        "    alias = apply\n"
        "    if request.causation_chain:\n"
        "        alias(terminate_child, target)\n"
    )
    benign = ast.parse(
        "def apply(callback, target):\n"
        "    callback(target)\n\n"
        "def dispatch(request, target):\n"
        "    alias = apply\n"
        "    if request.causation_chain:\n"
        "        alias(record_metric, target)\n"
    )

    assert _authority_provenance_lines(controlled) == {6}
    assert _authority_provenance_lines(benign) == set()


def test_provenance_scanner_follows_neutral_control_return_wrappers() -> None:
    direct = ast.parse(
        "def passthrough(fn):\n"
        "    return fn\n\n"
        "def dispatch(request, target):\n"
        "    operation = passthrough(terminate_child)\n"
        "    if request.causation_chain:\n"
        "        operation(target)\n"
    )
    aliased = ast.parse(
        "def passthrough(fn):\n"
        "    return fn\n\n"
        "def dispatch(request, target):\n"
        "    alias = passthrough\n"
        "    operation = alias(terminate_child)\n"
        "    if request.causation_chain:\n"
        "        operation(target)\n"
    )
    benign = ast.parse(
        "def passthrough(fn):\n"
        "    return fn\n\n"
        "def dispatch(request, target):\n"
        "    operation = passthrough(record_metric)\n"
        "    if request.causation_chain:\n"
        "        operation(target)\n"
    )

    assert _authority_provenance_lines(direct) == {6}
    assert _authority_provenance_lines(aliased) == {7}
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
        "retire_agent",
        "retire_persisted_child",
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


def test_provenance_scanner_shares_class_stored_control_aliases() -> None:
    class_attribute = ast.parse(
        "class Controller:\n"
        "    action = terminate_child\n\n"
        "    def dispatch(self, request, child):\n"
        "        if request.causation_chain:\n"
        "            self.action(child)\n"
    )
    instance_attribute = ast.parse(
        "class Controller:\n"
        "    def __init__(self):\n"
        "        self.action = terminate_child\n\n"
        "    def dispatch(self, request, child):\n"
        "        if request.causation_chain:\n"
        "            self.action(child)\n"
    )
    setattr_attribute = ast.parse(
        "class Controller:\n"
        "    def __init__(self):\n"
        "        setattr(self, 'action', terminate_child)\n\n"
        "    def dispatch(self, request, child):\n"
        "        if request.causation_chain:\n"
        "            self.action(child)\n"
    )

    assert _authority_provenance_lines(class_attribute) == {5}
    assert _authority_provenance_lines(instance_attribute) == {6}
    assert _authority_provenance_lines(setattr_attribute) == {6}


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


@pytest.mark.parametrize(
    "binding",
    (
        "check = derive",
        "renamed = derive\ncheck = renamed",
    ),
)
def test_provenance_helper_semantics_survive_callable_aliases(
    binding: str,
) -> None:
    """Renaming a summarized helper must not erase its return semantics."""

    tree = ast.parse(
        "def derive(request):\n"
        "    return request.causation_chain\n\n"
        f"{binding}\n\n"
        "def dispatch(request, target):\n"
        "    if check(request):\n"
        "        terminate_child(target)\n"
    )
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If))

    assert _authority_provenance_lines(tree) == {guard.lineno}


def test_control_semantics_survive_callable_instance_construction() -> None:
    """A class instance keeps the control semantics of its ``__call__``."""

    tree = ast.parse(
        "class Controller:\n"
        "    def __call__(self, target):\n"
        "        target.shutdown()\n\n"
        "controller = Controller()\n\n"
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        controller(target)\n"
    )
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If))

    assert _authority_provenance_lines(tree) == {guard.lineno}


def test_callable_semantics_preserve_constant_style_aliases() -> None:
    """Python's case-sensitive aliases must retain summarized effects."""

    tree = ast.parse(
        "def derive(request):\n"
        "    return request.causation_chain\n\n"
        "def apply(target):\n"
        "    target.shutdown()\n\n"
        "HANDLER = derive\n"
        "CONTROL = apply\n\n"
        "def dispatch(request, first, second):\n"
        "    if HANDLER(request):\n"
        "        terminate_child(first)\n"
        "    if request.causation_chain:\n"
        "        CONTROL(second)\n"
    )
    guards = {
        node.lineno for node in ast.walk(tree) if isinstance(node, ast.If)
    }

    assert _authority_provenance_lines(tree) == guards


@pytest.mark.parametrize("callable_kind", ("control", "provenance"))
@pytest.mark.parametrize("import_scope", ("module", "function"))
def test_provenance_scanner_follows_imported_callable_classes(
    tmp_path: Path,
    callable_kind: str,
    import_scope: str,
) -> None:
    """An imported class instance keeps the semantics of ``__call__``."""

    helper_path = tmp_path / "authority.py"
    if callable_kind == "control":
        helper_path.write_text(
            "class Provider:\n"
            "    def __call__(self, target):\n"
            "        target.shutdown()\n",
            encoding="utf-8",
        )
        guarded_expression = "request.causation_chain"
        guarded_action = "provider(target)"
    else:
        helper_path.write_text(
            "class Provider:\n"
            "    def __call__(self, request):\n"
            "        return request.causation_chain\n",
            encoding="utf-8",
        )
        guarded_expression = "provider(request)"
        guarded_action = "target.shutdown()"
    source_path = tmp_path / "dispatch.py"
    source = (
        (
            "from .authority import Provider\n"
            "provider = Provider()\n\n"
            "def dispatch(request, target):\n"
        )
        if import_scope == "module"
        else (
            "def dispatch(request, target):\n"
            "    from .authority import Provider\n"
            "    provider = Provider()\n"
        )
    ) + (
        f"    if {guarded_expression}:\n"
        f"        {guarded_action}\n"
    )
    source_path.write_text(source, encoding="utf-8")
    guard = next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.If)
    )

    assert _cached_authority_provenance_lines(source_path) == frozenset({guard.lineno})


@pytest.mark.parametrize(
    "mutation",
    (
        "permissions[request.causation_chain[-1].agent_id] = True",
        "permissions[request.causation_chain[-1].agent_id] |= {'stop'}",
        "del permissions[request.causation_chain[-1].agent_id]",
    ),
)
def test_permission_mutation_targets_are_provenance_inputs(
    mutation: str,
) -> None:
    tree = ast.parse(
        "def configure(request, permissions):\n"
        f"    {mutation}\n"
    )

    assert _authority_provenance_lines(tree) == {2}


@pytest.mark.parametrize(
    "mutation",
    (
        "setattr(permissions, 'allowed', bool(request.causation_chain))",
        "delattr(permissions, request.causation_chain[-1].agent_id)",
    ),
)
def test_functional_permission_attribute_mutations_use_provenance(
    mutation: str,
) -> None:
    tree = ast.parse(
        "def configure(request, permissions):\n"
        f"    {mutation}\n"
    )

    assert _authority_provenance_lines(tree) == {2}


@pytest.mark.parametrize(
    "store_name",
    (
        "acl",
        "rbac",
        "policy",
        "policies",
        "authority",
        "authorities",
        "roles",
        "capabilities",
    ),
)
def test_common_permission_store_mutation_targets_are_provenance_inputs(
    store_name: str,
) -> None:
    tree = ast.parse(
        f"def configure(request, {store_name}):\n"
        f"    {store_name}[request.causation_chain[-1].agent_id] = True\n"
    )

    assert _authority_provenance_lines(tree) == {2}


@pytest.mark.parametrize(
    "invocation",
    (
        "choose(derive, request, target)",
        "choose(check=derive, request=request, target=target)",
    ),
)
def test_provenance_callable_semantics_flow_through_parameters(
    invocation: str,
) -> None:
    tree = ast.parse(
        "def derive(request):\n"
        "    return request.causation_chain\n\n"
        "def choose(check, request, target):\n"
        "    if check(request):\n"
        "        target.shutdown()\n\n"
        "def dispatch(request, target):\n"
        f"    {invocation}\n"
    )

    assert _authority_provenance_lines(tree) == {5}


@pytest.mark.parametrize(
    "setup, invocation",
    (
        ("", "Controller().choose(derive, request, target)"),
        (
            "controller = Controller()\n    ",
            "controller.choose(derive, request, target)",
        ),
        ("", "Controller.choose(Controller(), derive, request, target)"),
    ),
)
def test_provenance_callable_parameter_binding_accounts_for_method_receivers(
    setup: str,
    invocation: str,
) -> None:
    tree = ast.parse(
        "def derive(request):\n"
        "    return request.causation_chain\n\n"
        "class Controller:\n"
        "    def choose(self, check, request, target):\n"
        "        if check(request):\n"
        "            target.shutdown()\n\n"
        "def dispatch(request, target):\n"
        f"    {setup}{invocation}\n"
    )

    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If))
    assert _authority_provenance_lines(tree) == {guard.lineno}


@pytest.mark.parametrize(
    "selection",
    (
        "agent_registry.get(agent_name)",
        "manager._agents.lookup(agent_name)",
        "child_registry.resolve(agent_name)",
    ),
)
@pytest.mark.parametrize(
    "mutation",
    (
        "target.enabled = False",
        "target.delete()",
    ),
)
def test_registry_receiver_semantics_flow_through_agent_selection_calls(
    selection: str,
    mutation: str,
) -> None:
    tree = ast.parse(
        "def govern(request, agent_registry, child_registry, manager, agent_name):\n"
        f"    target = {selection}\n"
        "    if request.causation_chain:\n"
        f"        {mutation}\n"
    )

    assert _authority_provenance_lines(tree) == {3}


@pytest.mark.parametrize(
    "mutation",
    (
        "operator.setitem(agent_registry, agent_name, replacement)",
        "operator.delitem(agent_registry, agent_name)",
    ),
)
def test_functional_item_mutators_preserve_registry_write_semantics(
    mutation: str,
) -> None:
    tree = ast.parse(
        "def govern(request, agent_registry, agent_name, replacement):\n"
        "    if request.causation_chain:\n"
        f"        {mutation}\n"
    )

    assert _authority_provenance_lines(tree) == {2}


@pytest.mark.parametrize(
    "command",
    (
        "/usr/local/bin/kestrel restart Bob",
        "'/opt/Kestrel Tools/kestrel' stop Bob",
        "kestrel shutdown Bob",
    ),
)
def test_shell_lifecycle_recognizes_kestrel_executable_basenames(
    command: str,
) -> None:
    tree = ast.parse(
        "def govern(request):\n"
        "    if request.causation_chain:\n"
        f"        subprocess.run({command!r}, shell=True)\n"
    )

    assert _authority_provenance_lines(tree) == {2}


@pytest.mark.parametrize(
    "command",
    (
        "uv run kestrel terminate Bob",
        "uv run python -m kestrel_sovereign.cli restart Bob",
        ["uv", "run", "python", "-m", "kestrel_sovereign.cli", "stop", "Bob"],
        "python -m kestrel_sovereign.cli restart Bob",
        [r"C:\\venv\\Scripts\\kestrel.exe", "stop", "Bob"],
    ),
)
def test_shell_lifecycle_unwraps_supported_cli_launchers(
    command: str | list[str],
) -> None:
    tree = ast.parse(
        "def govern(request):\n"
        "    if request.causation_chain:\n"
        f"        subprocess.run({command!r}, shell=True)\n"
    )

    assert _authority_provenance_lines(tree) == {2}


@pytest.mark.parametrize("operation", ("create", "update"))
def test_shell_lifecycle_includes_all_cli_state_transitions(operation: str) -> None:
    tree = ast.parse(
        "def govern(request):\n"
        "    if request.causation_chain:\n"
        f"        subprocess.run('kestrel {operation} Bob', shell=True)\n"
    )

    assert _authority_provenance_lines(tree) == {2}


@pytest.mark.parametrize(
    "invocation",
    (
        "subprocess.run([sys.executable, '-m', "
        "'kestrel_sovereign.cli', 'restart', 'Bob'])",
        "subprocess.run(['uv', 'run', sys.executable, '-m', "
        "'kestrel_sovereign.cli', 'stop', 'Bob'])",
        "asyncio.create_subprocess_exec(sys.executable, '-m', "
        "'kestrel_sovereign.cli', 'terminate', 'Bob')",
    ),
)
def test_shell_lifecycle_resolves_the_active_python_executable(
    invocation: str,
) -> None:
    tree = ast.parse(
        "def govern(request):\n"
        "    if request.causation_chain:\n"
        f"        {invocation}\n"
    )

    assert _authority_provenance_lines(tree) == {2}


@pytest.mark.parametrize(
    "setup, mutation",
    (
        (
            "store = permissions",
            "store[request.causation_chain[-1].agent_id] = True",
        ),
        (
            "store = policies\n    alias = store",
            "alias.update({request.causation_chain[-1].agent_id: True})",
        ),
        (
            "store = authorities",
            "operator.setitem("
            "store, request.causation_chain[-1].agent_id, True)",
        ),
    ),
)
def test_permission_store_identity_survives_aliases(
    setup: str,
    mutation: str,
) -> None:
    tree = ast.parse(
        "def configure(request, permissions, policies, authorities):\n"
        f"    {setup}\n"
        f"    {mutation}\n"
    )

    mutation_line = max(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.Call))
    )
    assert _authority_provenance_lines(tree) == {mutation_line}


@pytest.mark.parametrize(
    "setup, command",
    (
        ('command = ["kestrel", "restart", target]', "command"),
        (
            'command = ["kestrel", "stop", target]\n    alias = command',
            "alias",
        ),
        ('command = f"/usr/local/bin/kestrel shutdown {target}"', "command"),
    ),
)
def test_shell_lifecycle_command_values_survive_aliases(
    setup: str,
    command: str,
) -> None:
    tree = ast.parse(
        "def govern(request, target):\n"
        f"    {setup}\n"
        "    if request.causation_chain:\n"
        f"        subprocess.run({command})\n"
    )

    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If))
    assert _authority_provenance_lines(tree) == {guard.lineno}


@pytest.mark.parametrize(
    "source",
    (
        "COMMAND = 'kestrel terminate Bob'\n"
        "def govern(request):\n"
        "    if request.causation_chain:\n"
        "        subprocess.run(COMMAND, shell=True)\n",
        "class Gate:\n"
        "    COMMAND = 'kestrel restart Bob'\n"
        "    def govern(self, request):\n"
        "        if request.causation_chain:\n"
        "            subprocess.run(self.COMMAND, shell=True)\n",
        "def build():\n"
        "    command = 'kestrel stop Bob'\n"
        "    def govern(request):\n"
        "        if request.causation_chain:\n"
        "            subprocess.run(command, shell=True)\n",
    ),
)
def test_shell_lifecycle_command_aliases_inherit_outer_scopes(
    source: str,
) -> None:
    tree = ast.parse(source)
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If))

    assert _authority_provenance_lines(tree) == {guard.lineno}


def test_shell_lifecycle_command_aliases_follow_rebinding_and_inheritance() -> None:
    reassigned = ast.parse(
        "COMMAND = 'kestrel restart Bob'\n"
        "COMMAND = 'echo safe'\n\n"
        "def govern(request):\n"
        "    command = 'kestrel stop Bob'\n"
        "    command = 'echo safe'\n"
        "    if request.causation_chain:\n"
        "        subprocess.run(COMMAND, shell=True)\n"
        "        subprocess.run(command, shell=True)\n"
    )
    inherited = ast.parse(
        "class Base:\n"
        "    COMMAND = 'kestrel restart Bob'\n\n"
        "class Child(Base):\n"
        "    def govern(self, request):\n"
        "        if request.causation_chain:\n"
        "            subprocess.run(self.COMMAND, shell=True)\n"
    )
    guard = next(
        node for node in ast.walk(inherited) if isinstance(node, ast.If)
    )

    assert _authority_provenance_lines(reassigned) == set()
    assert _authority_provenance_lines(inherited) == {guard.lineno}


@pytest.mark.parametrize(
    "wrapper_body",
    (
        "    callback(target)\n",
        "    invoke = callback\n    invoke(target)\n",
        "    inner(callback, target)\n",
    ),
)
def test_imported_wrapper_callback_effects_reach_control_callers(
    tmp_path: Path,
    wrapper_body: str,
) -> None:
    wrappers_path = tmp_path / "wrappers.py"
    prefix = (
        "def inner(callback, target):\n"
        "    callback(target)\n\n"
        if "inner(" in wrapper_body
        else ""
    )
    wrappers_path.write_text(
        prefix
        + "def apply(callback, target):\n"
        + wrapper_body,
        encoding="utf-8",
    )
    source_path = tmp_path / "controller.py"
    source_path.write_text(
        "from .wrappers import apply as invoke\n\n"
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        invoke(terminate_child, target)\n",
        encoding="utf-8",
    )

    assert _cached_authority_provenance_lines(source_path) == frozenset({4})


def test_imported_wrapper_callback_summary_requires_invocation(
    tmp_path: Path,
) -> None:
    wrappers_path = tmp_path / "wrappers.py"
    wrappers_path.write_text(
        "def retain(callback, target):\n"
        "    return target\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "controller.py"
    source_path.write_text(
        "from .wrappers import retain\n\n"
        "def dispatch(request, target):\n"
        "    if request.causation_chain:\n"
        "        retain(terminate_child, target)\n",
        encoding="utf-8",
    )

    assert _cached_authority_provenance_lines(source_path) == frozenset()


def test_nested_return_helper_inherits_enclosing_provenance_value_flow() -> None:
    tree = ast.parse(
        "def dispatch(request, target):\n"
        "    lineage = request.causation_chain\n"
        "    def derive():\n"
        "        return lineage\n"
        "    if derive():\n"
        "        terminate_child(target)\n"
    )
    shadowed = ast.parse(
        "def dispatch(request, target):\n"
        "    lineage = request.causation_chain\n"
        "    def derive(lineage):\n"
        "        return lineage\n"
        "    if derive(False):\n"
        "        terminate_child(target)\n"
    )

    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If))
    assert _authority_provenance_lines(tree) == {guard.lineno}
    assert _authority_provenance_lines(shadowed) == set()


@pytest.mark.parametrize(
    "wrapper_body",
    (
        "    return tool(name='terminate_child', description='x')(fn)\n",
        "    decorated = tool(name='terminate_child', description='x')(fn)\n"
        "    return decorated\n",
        "    return configured(fn)\n",
    ),
)
def test_tool_inventory_follows_higher_order_decorator_applications(
    wrapper_body: str,
) -> None:
    prefix = (
        "configured = tool(name='terminate_child', description='x')\n\n"
        if "configured" in wrapper_body
        else ""
    )
    tree = ast.parse(
        prefix
        + "def expose(fn):\n"
        + wrapper_body
        + "\nclass Controls:\n"
        "    @expose\n"
        "    def terminate(self, target):\n"
        "        pass\n"
    )

    assert _tool_surfaces_from_module(tree, "example.py") == {
        "example.py::terminate_child"
    }


@pytest.mark.parametrize(
    "publication",
    (
        "self.registry['x'] = tool",
        "self.registry.update({'x': tool})",
        "self.publish({'x': tool})",
    ),
)
def test_dynamic_tool_registry_identity_flows_through_attributes(
    publication: str,
) -> None:
    setup = (
        "        self.publish = self._direct_tools.update\n"
        if "self.publish" in publication
        else "        self.registry = self._direct_tools\n"
    )
    tree = ast.parse(
        "class Publisher:\n"
        "    def publish_tool(self, tool):\n"
        + setup
        + f"        {publication}\n"
    )

    assert _direct_tool_writer_surfaces(tree, "example.py") == {
        "example.py::Publisher.publish_tool"
    }


def test_runtime_agent_tool_scan_uses_complete_core_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "kestrel_sovereign"
    provider = package_root / "agent" / "provider.py"
    provider.parent.mkdir(parents=True)
    provider.write_text(
        "class OutsideFeatureTool(AgentTool):\n"
        "    async def execute(self, **kwargs):\n"
        "        return kwargs\n",
        encoding="utf-8",
    )

    assert _runtime_generated_tool_class_surfaces(package_root, tmp_path) == {
        "kestrel_sovereign/agent/provider.py::OutsideFeatureTool.execute"
    }

    scanned_roots: list[tuple[Path, Path]] = []

    def record_root(root: Path, repository_root: Path) -> set[str]:
        scanned_roots.append((root, repository_root))
        return set()

    monkeypatch.setitem(
        globals(),
        "_runtime_generated_tool_class_surfaces",
        record_root,
    )
    _discovered_runtime_generated_tool_surfaces.cache_clear()
    try:
        _discovered_runtime_generated_tool_surfaces()
    finally:
        _discovered_runtime_generated_tool_surfaces.cache_clear()
    assert scanned_roots == [
        (REPO_ROOT / "kestrel_sovereign", REPO_ROOT)
    ]


@pytest.mark.parametrize(
    "storage",
    (
        "checks = {'lineage': derive}\n    check = checks['lineage']",
        "checks = [derive]\n    check = checks[0]",
        "checks = {}\n    checks['lineage'] = derive\n"
        "    check = checks['lineage']",
        "checks = []\n    checks.append(derive)\n    check = checks[0]",
    ),
)
def test_provenance_callable_semantics_survive_container_selection(
    storage: str,
) -> None:
    tree = ast.parse(
        "def derive(request):\n"
        "    return request.causation_chain\n\n"
        "def dispatch(request, target):\n"
        f"    {storage}\n"
        "    if check(request):\n"
        "        target.shutdown()\n"
    )

    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If))
    assert _authority_provenance_lines(tree) == {guard.lineno}


@pytest.mark.parametrize(
    "selection",
    (
        "request.causation_chain.pop()",
        "request.causation_chain.__getitem__(-1)",
    ),
)
def test_provenance_method_receivers_taint_selected_control_targets(
    selection: str,
) -> None:
    tree = ast.parse(
        "def dispatch(request, manager):\n"
        f"    frame = {selection}\n"
        "    manager.terminate_agent(frame.agent_id)\n"
    )

    assert _authority_provenance_lines(tree) == {3}


@pytest.mark.parametrize("decorator", ("property", "cached_property"))
def test_imported_provenance_descriptors_guarding_control_are_detected(
    tmp_path: Path,
    decorator: str,
) -> None:
    authority_path = tmp_path / "authority.py"
    authority_path.write_text(
        "class Context:\n"
        f"    @{decorator}\n"
        "    def lineage(self):\n"
        "        return self.request.causation_chain\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "dispatcher.py"
    source = (
        "from .authority import Context\n\n"
        "def dispatch(context: Context, target):\n"
        "    if context.lineage:\n"
        "        target.shutdown()\n"
    )
    source_path.write_text(source, encoding="utf-8")

    assert _cached_authority_provenance_lines(source_path) == frozenset({4})


def test_permission_callable_selector_is_a_provenance_input() -> None:
    tree = ast.parse(
        "def dispatch(request, grantors, target):\n"
        "    grantors[request.causation_chain[-1].agent_id](target)\n"
    )

    assert _authority_provenance_lines(tree) == {2}


def test_cycle_exemption_requires_resolved_universal_rail_semantics() -> None:
    """A lookalike helper name is not evidence of a bounded signal rail."""

    tree = ast.parse(
        "def stop_peer(subject):\n"
        "    subject.shutdown()\n\n"
        "def dispatch(request, target):\n"
        "    if target not in request.causation_chain:\n"
        "        stop_peer(target)\n"
    )
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If))

    assert _authority_provenance_lines(tree) == {guard.lineno}

    verified_rail = ast.parse(
        "class PeersFeature:\n"
        "    def send_a2a_message(self, subject):\n"
        "        subject.shutdown()\n\n"
        "    def dispatch(self, request, target):\n"
        "        if target not in request.causation_chain:\n"
        "            self.send_a2a_message(target)\n"
    )
    assert _authority_provenance_lines(
        verified_rail,
        _CYCLE_BOUNDED_UNIVERSAL_MODULE,
    ) == set()


@pytest.mark.parametrize(
    "publication",
    (
        "(publish := self._direct_tools.update)(tools)",
        "(registry := self._direct_tools).update(tools)",
    ),
)
def test_dynamic_tool_registry_replays_inline_walrus_publications(
    publication: str,
) -> None:
    tree = ast.parse(
        "class Publisher:\n"
        "    def publish(self, tools):\n"
        f"        {publication}\n"
    )

    assert _direct_tool_writer_surfaces(tree, "example.py") == {
        "example.py::Publisher.publish"
    }


def test_authority_audit_chunks_fit_the_per_test_timeout_budget() -> None:
    """Cold-worker scan units stay bounded beneath CI's 60-second guard."""

    assert max(map(len, _AUTHORITY_AUDIT_CHUNKS)) <= 48


@pytest.mark.parametrize(
    ("operator", "guard_exits", "expected_violation"),
    (
        ("not in", False, False),
        ("in", False, True),
        ("in", True, False),
        ("not in", True, True),
    ),
)
def test_causation_cycle_suppression_is_not_an_authority_grant(
    operator: str,
    guard_exits: bool,
    expected_violation: bool,
) -> None:
    """Only the polarity that suppresses a repeated peer is non-authority."""

    guarded = (
        "        return\n"
        if guard_exits
        else "        send_a2a_message(peer)\n"
    )
    continuation = "    send_a2a_message(peer)\n" if guard_exits else ""
    tree = ast.parse(
        "def dispatch(request, peer):\n"
        "    visited = {frame.agent_id for frame in request.causation_chain}\n"
        f"    if peer.agent_id {operator} visited:\n"
        f"{guarded}"
        f"{continuation}"
    )
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If))
    for call in (
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
    ):
        if _call_name(call) == "send_a2a_message":
            _mark_callable_semantics(
                call.func,
                {_CALLABLE_CONTROL, _CALLABLE_CYCLE_BOUNDED},
            )

    assert (guard.lineno in _authority_provenance_lines(tree)) is (
        expected_violation
    )


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


@pytest.mark.parametrize(
    "paths",
    _AUTHORITY_AUDIT_CHUNKS,
    ids=lambda paths: (
        f"{paths[0].relative_to(REPO_ROOT).as_posix()}.."
        f"{paths[-1].relative_to(REPO_ROOT).as_posix()}"
    ),
)
def test_causation_and_orchestrator_metadata_are_not_permission_inputs(
    paths: tuple[Path, ...],
) -> None:
    """Make a direct causation-as-authority condition fail review loudly."""

    violations: list[str] = []
    for path in paths:
        for line in _cached_authority_provenance_lines(path):
            violations.append(f"{path.relative_to(REPO_ROOT)}:{line}")

    assert not violations, (
        "Causation/orchestrator metadata appeared in a permission condition: "
        + ", ".join(sorted(set(violations)))
    )
