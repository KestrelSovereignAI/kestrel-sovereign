"""Contract tests for the checked-in cross-agent authority inventory (#3143)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "docs/architecture/CROSS_AGENT_AUTHORITY_AUDIT.md"
TOOL_KEYWORDS = ("agent", "peer", "a2a", "child", "restart", "task")
HTTP_SEGMENTS = {"agents", "tasks", "stop", "restart", "a2a", "peers", "children"}
HTTP_EXACT_ROUTES = {"/api/agent/invoke"}
SURFACE_ID = re.compile(
    r"\|\s*`(kestrel_sovereign/(?:features|endpoints)/[^`]+)`\s*\|"
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


def _discovered_tool_surfaces() -> set[str]:
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
                if any(term in public_name.casefold() for term in TOOL_KEYWORDS):
                    relative = path.relative_to(REPO_ROOT).as_posix()
                    surfaces.add(f"{relative}::{public_name}")
    return surfaces


def _router_prefix(tree: ast.Module) -> str:
    for node in tree.body:
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


def _route_methods(decorator: ast.Call) -> tuple[str, ...]:
    if not isinstance(decorator.func, ast.Attribute):
        return ()
    method = decorator.func.attr.lower()
    if method in {"get", "post", "put", "patch", "delete"}:
        return (method.upper(),)
    if method != "api_route":
        return ()
    for keyword in decorator.keywords:
        if keyword.arg != "methods" or not isinstance(
            keyword.value, (ast.List, ast.Tuple, ast.Set)
        ):
            continue
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
    endpoint_root = REPO_ROOT / "kestrel_sovereign/endpoints"
    for path in endpoint_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        prefix = _router_prefix(tree)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not isinstance(
                    decorator.func, ast.Attribute
                ):
                    continue
                methods = _route_methods(decorator)
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


def _documented_surfaces(section: str) -> set[str]:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    start = audit.index(section)
    next_section = audit.find("\n## ", start + len(section))
    body = audit[start:] if next_section < 0 else audit[start:next_section]
    return set(SURFACE_ID.findall(body))


def test_every_cross_agent_named_tool_is_classified() -> None:
    assert _discovered_tool_surfaces() == _documented_surfaces(
        "## Machine-checked tool inventory"
    )


def test_every_cross_agent_http_route_is_classified() -> None:
    assert _discovered_http_surfaces() == _documented_surfaces(
        "## Machine-checked HTTP inventory"
    )


def test_api_route_declarations_expand_every_registered_method() -> None:
    decorator = ast.parse(
        '@router.api_route("/api/tasks", methods=["POST", "PUT"])\n'
        "def route():\n    pass\n"
    ).body[0].decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert _route_methods(decorator) == ("POST", "PUT")


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


def _authority_provenance_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for function in functions:
        function_is_permission_boundary = any(
            term in function.name.casefold()
            for term in ("authoriz", "permission")
        )
        for node in ast.walk(function):
            if not isinstance(node, (ast.If, ast.IfExp, ast.Assert)):
                continue
            tokens = _identifier_tokens(node.test)
            has_provenance = any(
                "causation" in token or "orchestrator" in token
                for token in tokens
            )
            has_permission = any(
                "authoriz" in token or "permission" in token
                for token in tokens
            )
            if has_provenance and (
                has_permission or function_is_permission_boundary
            ):
                lines.add(node.lineno)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_tokens = _identifier_tokens(node.func)
        is_permission_call = any(
            "authoriz" in token or "permission" in token
            for token in function_tokens
        )
        argument_tokens = set().union(
            *(_identifier_tokens(argument) for argument in node.args),
            *(_identifier_tokens(keyword.value) for keyword in node.keywords),
        )
        has_provenance_argument = any(
            "causation" in token or "orchestrator" in token
            for token in argument_tokens
        )
        if is_permission_call and has_provenance_argument:
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
    assert _authority_provenance_lines(enclosing) == {2}
    assert _authority_provenance_lines(metadata_key) == {2}


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
