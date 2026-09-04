"""Contract tests for the checked-in cross-agent authority inventory (#3143)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from kestrel_sovereign.command_handler import BUILTIN_COMMAND_SPECS

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


def _public_tool_name(decorator: ast.expr, fallback: str) -> str | None:
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
    if call.args and isinstance(call.args[0], ast.Constant):
        return str(call.args[0].value)
    for keyword in call.keywords:
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
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
    """Return every core feature tool, including apparently local tools.

    Cross-agent capability is a property of implementation and deployment,
    not a public-name convention.  Exact inventory of the complete registered
    tool set forces new tools to be classified even when their names do not
    advertise shared-host reach.
    """

    surfaces: set[str] = set()
    feature_root = REPO_ROOT / "kestrel_sovereign/features"
    for path in feature_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                public_name = _public_tool_name(decorator, node.name)
                if public_name is None:
                    continue
                relative = path.relative_to(REPO_ROOT).as_posix()
                surfaces.add(f"{relative}::{public_name}")
    return surfaces


def _discovered_builtin_command_surfaces() -> set[str]:
    """Return built-ins whose public names indicate cross-agent reach.

    Built-in commands bypass feature ``@tool`` discovery.  They therefore need
    their own inventory source or a host-control door such as ``!create-agent``
    can remain invisible while the feature-tool completeness gate stays green.
    """

    surfaces: set[str] = set()
    for spec in BUILTIN_COMMAND_SPECS:
        command = spec.get("cmd")
        if not isinstance(command, str):
            continue
        if _is_cross_agent_control_name(command):
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


def _router_prefix(tree: ast.Module) -> str:
    # Feature routers are commonly built inside ``get_router`` factories, so
    # the APIRouter assignment is not necessarily at module scope.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "router" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        for keyword in node.value.keywords:
            if keyword.arg == "prefix" and isinstance(keyword.value, ast.Constant):
                return str(keyword.value.value)
    return ""


def _module_string_collections(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    """Resolve safe module constants used by ``methods=`` declarations."""

    collections: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(
            node.value, (ast.List, ast.Tuple, ast.Set)
        ):
            continue
        values = tuple(
            str(element.value)
            for element in node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        )
        if len(values) != len(node.value.elts):
            continue
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
            return tuple(
                value.upper()
                for value in (constants or {}).get(keyword.value.id, ())
            )
        if isinstance(keyword.value, (ast.List, ast.Tuple, ast.Set)):
            methods = []
            for element in keyword.value.elts:
                if isinstance(element, ast.Constant) and isinstance(
                    element.value, str
                ):
                    methods.append(element.value.upper())
            return tuple(methods)
    return ()


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
        prefix = _router_prefix(tree)
        method_constants = _module_string_collections(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not isinstance(
                    decorator.func, ast.Attribute
                ):
                    continue
                methods = _route_methods(decorator, method_constants)
                if not methods:
                    continue
                if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
                    continue
                route = prefix + str(decorator.args[0].value)
                segments = {part for part in route.casefold().split("/") if part}
                if (
                    route.casefold() not in HTTP_EXACT_ROUTES
                    and not segments.intersection(HTTP_SEGMENTS)
                ):
                    continue
                relative = path.relative_to(REPO_ROOT).as_posix()
                for method in methods:
                    surfaces.add(f"{relative}::{method} {route}")
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
        prefix = _router_prefix(tree)
        method_constants = _module_string_collections(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                methods = _route_methods(decorator, method_constants)
                if not methods:
                    continue
                if not decorator.args or not isinstance(
                    decorator.args[0], ast.Constant
                ):
                    continue
                canonical_route = prefix + str(decorator.args[0].value)
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
        tool_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                _public_tool_name(decorator, node.name) == public_name
                for decorator in node.decorator_list
            )
        ]
        assert len(tool_nodes) == 1
        assert _is_indirect_tool_dispatcher(tool_nodes[0])


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


def test_every_cross_agent_named_builtin_command_is_classified() -> None:
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
    ):
        assert surface in discovered

    # The cross-agent audit and the general auth ledger must agree that the
    # bootstrap credential is a narrow public-localhost exception, not an
    # ordinary authenticated agent route.
    auth_matrix = AUTH_SURFACE_MATRIX_PATH.read_text(encoding="utf-8")
    assert "| `Public-Localhost`" in auth_matrix
    assert "/api/auth/key" in auth_matrix


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
        # boolean wrappers whose result still represents the metadata itself.
        while isinstance(value, (ast.Await, ast.Expr)):
            value = value.value
        if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript, ast.Constant)):
            return _has_provenance_token(value, aliases)
        if isinstance(value, ast.Call) and _call_name(value) in {
            "bool",
            "get",
            "getattr",
        }:
            return _has_provenance_token(value, aliases)
        # Boolean normalization does not erase the authority input. Common
        # forms such as ``allowed = chain is not None`` and ``has_chain = not
        # not chain`` remain provenance-derived when used by a later gate.
        if isinstance(value, (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.IfExp)):
            return _has_provenance_token(value, aliases)
        return False

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
        names = {
            child.id.casefold()
            for target in targets
            for child in ast.walk(target)
            if isinstance(child, ast.Name)
        }
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
            if isinstance(node, ast.Return):
                if (
                    function_is_permission_boundary
                    and node.value is not None
                    and _has_provenance_token(node.value, provenance_aliases)
                ):
                    lines.add(node.lineno)
                continue
            if not isinstance(node, (ast.If, ast.IfExp, ast.Assert)):
                continue
            tokens = _identifier_tokens(node.test)
            has_provenance = _has_provenance_token(
                node.test, provenance_aliases
            )
            has_permission = any(_is_permission_name(token) for token in tokens)
            if has_provenance and (
                has_permission or function_is_permission_boundary
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
    assert _authority_provenance_lines(enclosing) == {2}
    assert _authority_provenance_lines(metadata_key) == {2}
    assert _authority_provenance_lines(direct_return) == {2}
    assert _authority_provenance_lines(ordinary_helpers) == {2, 5}
    assert _authority_provenance_lines(control_handlers) == {2, 7, 11}
    assert _authority_provenance_lines(canonical_frame) == {2}
    assert _authority_provenance_lines(compared_alias) == {3}
    assert _authority_provenance_lines(permission_call_alias) == {3}
    assert _authority_provenance_lines(authority_helper) == {2}
    assert _authority_provenance_lines(access_helper) == {2}
    assert _authority_provenance_lines(require_call) == {3}
    assert _authority_provenance_lines(dynamic_metadata_key) == {3}


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
