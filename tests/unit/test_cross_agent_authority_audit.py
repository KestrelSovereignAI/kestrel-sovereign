"""Contract tests for the checked-in cross-agent authority inventory (#3143)."""

from __future__ import annotations

import ast
import re
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
    "child",
    "descendant",
    "delegate",
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
)
PERMISSION_NAME_TERMS = (
    "authoriz",
    "authority",
    "mandate",
    "permission",
    "allowed",
    "permitted",
    "forbid",
    "denied",
    "access",
    "gate",
    "require",
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
    "/health",
    "/health/detailed",
    "/metrics",
    "/phoenix",
    "/phoenix/{path:path}",
    "/api/agent/invoke",
    "/api/agent/stream",
    "/api/auth/key",
    "/api/keys/platform",
    "/api/keys/user",
    "/api/keys/user/verify",
    "/api/keys/user/{provider}",
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
    r"(?:server\.py::[^`]+|(?:features|endpoints)/[^`]+))`\s*\|"
)
COMMAND_SURFACE_ID = re.compile(
    r"\|\s*`(kestrel_sovereign/command_handler\.py::![^`]+)`\s*\|"
)
CLI_SURFACE_ID = re.compile(
    r"\|\s*`(kestrel_sovereign/cli\.py::kestrel [^`]+)`\s*\|"
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


def _module_string_constants(
    tree: ast.Module,
    source_path: Path | None = None,
) -> dict[str, str]:
    """Resolve static strings used in decorators, including local imports."""

    constants: dict[str, str] = {}
    if source_path is not None:
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            imported_path = _imported_module_path(node, source_path)
            if imported_path is None:
                continue
            imported_constants = _cached_local_string_constants(imported_path)
            for alias in node.names:
                if alias.name in imported_constants:
                    constants[alias.asname or alias.name] = imported_constants[
                        alias.name
                    ]

    unresolved: list[tuple[list[ast.expr], ast.expr]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            unresolved.append((node.targets, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            unresolved.append(([node.target], node.value))

    changed = True
    while changed:
        changed = False
        for targets, value in unresolved:
            resolved = _resolved_string(value, constants)
            if resolved is None:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and constants.get(target.id) != resolved
                ):
                    constants[target.id] = resolved
                    changed = True
    return constants


@lru_cache(maxsize=None)
def _cached_local_string_constants(source_path: Path) -> dict[str, str]:
    """Read one imported module's local constants without crawling its imports."""

    tree = ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=str(source_path),
    )
    return _module_string_constants(tree)


def _public_tool_name(
    decorator: ast.expr,
    fallback: str,
    constants: dict[str, str] | None = None,
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
    if decorator_name != "tool":
        return None
    if call is None:
        return fallback
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


def _discovered_tool_surfaces() -> set[str]:
    """Return every core feature tool, including generated dispatch boundaries.

    Cross-agent capability is a property of implementation and deployment,
    not a public-name convention.  Exact inventory of the complete registered
    tool set forces new tools to be classified even when their names do not
    advertise shared-host reach.
    """

    surfaces: set[str] = set()
    feature_root = REPO_ROOT / "kestrel_sovereign/features"
    for path in feature_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        string_constants = _module_string_constants(tree, path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                public_name = _public_tool_name(
                    decorator, node.name, string_constants
                )
                if public_name is None:
                    continue
                relative = path.relative_to(REPO_ROOT).as_posix()
                surfaces.add(f"{relative}::{public_name}")
    return surfaces | _discovered_runtime_generated_tool_surfaces()


def _discovered_runtime_generated_tool_surfaces() -> set[str]:
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
    return surfaces


def _discovered_builtin_command_surfaces() -> set[str]:
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
    return surfaces


def _discovered_core_cli_surfaces() -> set[str]:
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
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "commands"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        command_names = {
            str(key.value)
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        return {
            f"kestrel_sovereign/cli.py::kestrel {command}"
            for command in command_names
        }
    raise AssertionError("Could not find the core CLI command dispatch dictionary")


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


def _route_receiver_name(decorator: ast.Call) -> str | None:
    if not isinstance(decorator.func, ast.Attribute):
        return None
    receiver = decorator.func.value
    return receiver.id if isinstance(receiver, ast.Name) else None


def _route_declarations(
    tree: ast.Module,
    string_constants: dict[str, str],
    method_constants: dict[str, tuple[str, ...]],
) -> list[tuple[tuple[str, ...], str]]:
    """Return methods and canonical paths with receiver/scoped prefixes."""

    declarations: list[tuple[tuple[str, ...], str]] = []

    def walk_scope(
        statements: list[ast.stmt],
        inherited_prefixes: dict[str, str],
    ) -> None:
        prefixes = dict(inherited_prefixes)
        prefixes.update(_scope_router_prefixes(statements, string_constants))

        def visit(node: ast.AST) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call):
                        continue
                    methods = _route_methods(decorator, method_constants)
                    if not methods:
                        continue
                    receiver = _route_receiver_name(decorator)
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
                        (methods, prefix + _route_path(decorator, string_constants))
                    )
                walk_scope(node.body, prefixes)
                return
            if isinstance(node, ast.ClassDef):
                walk_scope(node.body, prefixes)
                return
            for child in ast.iter_child_nodes(node):
                visit(child)

        for statement in statements:
            visit(statement)

    walk_scope(tree.body, {})
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
    string_constants: dict[str, str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Resolve safe module constants used by ``methods=`` declarations."""

    collections: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(
            node.value, (ast.List, ast.Tuple, ast.Set)
        ):
            continue
        resolved_values = [
            _resolved_string(element, string_constants) for element in node.value.elts
        ]
        if any(value is None for value in resolved_values):
            continue
        values = tuple(value for value in resolved_values if value is not None)
        for target in node.targets:
            if isinstance(target, ast.Name):
                collections[target.id] = values
    return collections


def _route_methods(
    decorator: ast.Call,
    constants: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    if not isinstance(decorator.func, ast.Attribute):
        return ()
    method = decorator.func.attr.lower()
    if method in {"websocket", "websocket_route"}:
        return ("WEBSOCKET",)
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
    if method != "api_route":
        return ()
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
            return tuple(value.upper() for value in values)
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
            return tuple(methods)
        raise AssertionError(
            "Unsupported api_route methods expression: "
            f"{ast.unparse(keyword.value)}"
        )
    return ("GET",)


def _deprecated_agent_alias(route: str) -> str | None:
    """Return the live #871 compatibility spelling for a singular route."""

    if route == "/api/agent":
        return "/agent"
    if route.startswith("/api/agent/"):
        return route.removeprefix("/api")
    return None


def _discovered_http_surfaces() -> set[str]:
    surfaces: set[str] = set()
    roots = (
        REPO_ROOT / "kestrel_sovereign/endpoints",
        REPO_ROOT / "kestrel_sovereign/features",
    )
    paths = sorted(
        {path for root in roots for path in root.rglob("*.py")}
        | {REPO_ROOT / "kestrel_sovereign/server.py"}
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        string_constants = _module_string_constants(tree, path)
        method_constants = _module_string_collections(tree, string_constants)
        for methods, route in _route_declarations(
            tree, string_constants, method_constants
        ):
            segments = {part for part in route.casefold().split("/") if part}
            if (
                route.casefold() not in HTTP_EXACT_ROUTES
                and not segments.intersection(HTTP_SEGMENTS)
            ):
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            for method in methods:
                surfaces.add(f"{relative}::{method} {route}")
                deprecated_alias = _deprecated_agent_alias(route)
                if deprecated_alias is not None:
                    surfaces.add(f"{relative}::{method} {deprecated_alias}")
    return surfaces


def _agent_alias(route: str) -> str:
    """Return the concrete multi-agent spelling after target selection."""

    return f"/api/agents/{{selected_agent_name}}/{route.lstrip('/')}"


def _discovered_request_routed_alias_surfaces() -> set[str]:
    """Synthesize the host alias for every decorated core HTTP route.

    The routing middleware accepts ``/api/agents/{name}/{remaining_path}`` and
    rewrites the remainder before FastAPI dispatch.  Consequently every
    canonical route is an agent-addressable door, even when its handler
    does not consume ``Request`` and its path says only ``security``,
    ``identity``, or ``conversations``.  This complete inventory complements
    the narrower set of intrinsically cross-agent/host routes above.
    """

    surfaces: set[str] = set()
    roots = (
        REPO_ROOT / "kestrel_sovereign/endpoints",
        REPO_ROOT / "kestrel_sovereign/features",
    )
    paths = sorted(
        {path for root in roots for path in root.rglob("*.py")}
        | {REPO_ROOT / "kestrel_sovereign/server.py"}
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        string_constants = _module_string_constants(tree, path)
        method_constants = _module_string_collections(tree, string_constants)
        for methods, canonical_route in _route_declarations(
            tree, string_constants, method_constants
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
    return surfaces


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


def test_audit_records_remediated_authority_paths_as_enforced() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    for issue in (3134, 3144, 3146, 3147, 3149):
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
    }
    for action, issue in expected.items():
        row = next(
            line for line in audit.splitlines() if line.startswith(f"| {action} |")
        )
        assert issue in row
        assert "Defect:" in row


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


def test_task_reads_remain_labeled_unscoped_until_3145_lands() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    action_row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Read task inbox/status/result |")
    )
    assert "[#3145]" in action_row
    assert "unscoped" in action_row.casefold()

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
        assert "[#3145]" in row
        assert "unscoped" in row.casefold()


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


def _identifier_tokens(node: ast.AST) -> set[str]:
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
    return tokens


def _has_provenance_token(node: ast.AST, aliases: set[str] | None = None) -> bool:
    tokens = _identifier_tokens(node)
    provenance_tokens = {"orchestrator", "kestrel.orchestrator"}
    return any(
        token in provenance_tokens
        or token == "causation"
        or token.startswith("causation_")
        or token == "causationframe"
        or token in (aliases or set())
        for token in tokens
    )


def _provenance_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """Resolve simple local aliases of provenance metadata to a fixed point."""

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
            return _has_provenance_token(value, aliases)
        if isinstance(value, ast.Call) and _call_name(value) in {
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
        }:
            return _has_provenance_token(value, aliases)
        if isinstance(value, ast.Call):
            call_name = _call_name(value).casefold()
            if call_name.startswith(("can_", "has_", "is_", "may_")) or (
                _is_permission_name(call_name)
                and _has_provenance_token(value, aliases)
            ):
                return _has_provenance_token(value, aliases)
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
            return _has_provenance_token(value, aliases)
        return False

    def target_names(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id.casefold()}
        if isinstance(target, (ast.List, ast.Tuple)):
            return {
                name
                for element in target.elts
                for name in target_names(element)
            }
        if isinstance(target, (ast.Attribute, ast.Subscript)):
            # Use the full access path rather than tainting broad bases such as
            # ``self`` or ``state``. The same normalized path is emitted by
            # ``_identifier_tokens`` when a later condition reads it.
            return {ast.unparse(target).casefold()}
        return set()

    aliases: set[str] = set()
    assignments: list[tuple[set[str], ast.AST]] = []
    for node in ast.walk(function):
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
        names = {name for target in targets for name in target_names(target)}
        if names:
            assignments.append((names, value))

    changed = True
    while changed:
        changed = False
        for names, value in assignments:
            if is_provenance_derived(value, aliases) and not names.issubset(
                aliases
            ):
                aliases.update(names)
                changed = True
    return aliases


def _contains_cross_agent_control_call(nodes: ast.AST | list[ast.AST]) -> bool:
    """Return whether a guarded expression/body invokes an agent control.

    Authority checks are often wrapped by generic adapters named ``execute``
    or ``dispatch``.  In those cases the enclosing function and condition can
    both be neutrally named even though the branch controls another agent.
    Inspect the guarded operation itself, while stopping at nested lexical
    scopes whose calls are not executed merely because the outer branch ran.
    """

    roots = nodes if isinstance(nodes, list) else [nodes]

    class ControlCallVisitor(ast.NodeVisitor):
        found = False

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
            if _is_cross_agent_control_name(_call_name(node)):
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


def _authority_provenance_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for function in functions:
        function_name = function.name.casefold()
        provenance_aliases = _provenance_aliases(function)
        function_is_permission_boundary = (
            _is_permission_name(function_name)
            or function_name.startswith(("can_", "may_"))
            or _is_cross_agent_control_name(function_name)
        )
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                function_tokens = _identifier_tokens(node.func)
                is_permission_call = any(
                    _is_permission_name(token) for token in function_tokens
                )
                arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
                if is_permission_call and any(
                    _has_provenance_token(argument, provenance_aliases)
                    for argument in arguments
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
                    and _has_provenance_token(
                        assignment_value, provenance_aliases
                    )
                ):
                    lines.add(node.lineno)
            if isinstance(node, ast.Return):
                if (
                    node.value is not None
                    and _has_provenance_token(node.value, provenance_aliases)
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
                if _has_provenance_token(node.subject, provenance_aliases) and (
                    function_is_permission_boundary
                    or _contains_cross_agent_control_call(
                        [statement for case in node.cases for statement in case.body]
                    )
                ):
                    lines.add(node.lineno)
                continue
            if not isinstance(node, (ast.If, ast.IfExp, ast.Assert, ast.While)):
                continue
            tokens = _identifier_tokens(node.test)
            has_provenance = _has_provenance_token(
                node.test, provenance_aliases
            )
            has_permission = any(_is_permission_name(token) for token in tokens)
            guarded_nodes: list[ast.AST] = [node.test]
            if isinstance(node, (ast.If, ast.While)):
                guarded_nodes.extend([*node.body, *node.orelse])
            elif isinstance(node, ast.IfExp):
                guarded_nodes.extend([node.body, node.orelse])
            if has_provenance and (
                has_permission
                or function_is_permission_boundary
                or _contains_cross_agent_control_call(guarded_nodes)
            ):
                lines.add(node.lineno)
    return lines


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


def test_causation_and_orchestrator_metadata_are_not_permission_inputs() -> None:
    """Make a direct causation-as-authority condition fail review loudly."""

    violations: list[str] = []
    for path in (REPO_ROOT / "kestrel_sovereign").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line in _authority_provenance_lines(tree):
            violations.append(f"{path.relative_to(REPO_ROOT)}:{line}")

    assert not violations, (
        "Causation/orchestrator metadata appeared in a permission condition: "
        + ", ".join(sorted(set(violations)))
    )
