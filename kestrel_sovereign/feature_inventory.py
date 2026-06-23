"""Canonical feature/tool/endpoint inventory discovery and rendering."""

from __future__ import annotations

import ast
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kestrel_sovereign.command_handler import BUILTIN_COMMAND_SPECS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_ROOT = PROJECT_ROOT / "kestrel_sovereign" / "features"
ENDPOINTS_ROOT = PROJECT_ROOT / "kestrel_sovereign" / "endpoints"
CANONICAL_INVENTORY = PROJECT_ROOT / "KESTREL_FEATURES.md"
GENERATED_START = "<!-- BEGIN AUTO-GENERATED FEATURE INVENTORY -->"
GENERATED_END = "<!-- END AUTO-GENERATED FEATURE INVENTORY -->"


@dataclass(frozen=True)
class ToolInventory:
    name: str
    description: str
    category: str
    command_prefix: str | None
    parameters: list[dict[str, Any]]
    token_cost_estimate: int
    enablement_state: str
    source: str


@dataclass(frozen=True)
class FeatureInventory:
    module: str
    class_name: str | None
    source: str
    enablement_state: str
    tools: list[ToolInventory] = field(default_factory=list)


@dataclass(frozen=True)
class RouteInventory:
    method: str
    path: str
    source: str


@dataclass(frozen=True)
class CommandInventory:
    command: str
    description: str
    args: str | None
    category: str
    feature: str | None
    source: str


@dataclass(frozen=True)
class Inventory:
    feature_modules: list[str]
    feature_classes: list[str]
    entrypoint_feature_classes: list[str]
    features: list[FeatureInventory]
    app_routes: list[RouteInventory]
    router_routes: list[RouteInventory]
    commands: list[CommandInventory]


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def discover_core_feature_modules() -> list[str]:
    """Return discoverable in-tree feature module names."""
    try:
        from kestrel_sovereign.features import discover_feature_modules
    except ModuleNotFoundError:
        discover_feature_modules = None

    if discover_feature_modules is None:
        return sorted(_discover_feature_class_by_module())

    modules = []
    for module_path in discover_feature_modules():
        parts = module_path.split(".")
        modules.append(parts[-2] if module_path.endswith(".feature") else parts[-1])
    return sorted(modules)


def discover_exported_feature_classes() -> list[str]:
    """Return exported in-tree Feature subclass names using source inspection."""
    names: list[str] = []
    pattern = re.compile(r"^class\s+(\w+)\([^)]*\bFeature\b[^)]*\):", re.M | re.S)
    for path in FEATURES_ROOT.rglob("*.py"):
        if path.name == "base.py":
            continue
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            class_name = match.group(1)
            if class_name != "ProxyFeature":
                names.append(class_name)
    return sorted(set(names))


def discover_entrypoint_feature_class_names() -> list[str]:
    """Return installed external feature class names when runtime deps exist."""
    try:
        from kestrel_sovereign.features import discover_entrypoint_feature_classes

        return sorted(discover_entrypoint_feature_classes())
    except ModuleNotFoundError:
        return []


def discover_endpoint_router_files() -> list[str]:
    return sorted(
        path.name
        for path in ENDPOINTS_ROOT.glob("*.py")
        if path.name not in {"__init__.py", "agent_helpers.py"}
    )


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _decorator_call(node: ast.AST) -> ast.Call | None:
    if isinstance(node, ast.Call):
        return node
    return None


def _decorator_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _docstring_param_descriptions(docstring: str | None) -> dict[str, str]:
    if not docstring:
        return {}
    descriptions: dict[str, str] = {}
    args_match = re.search(
        r"(?:Args|Arguments|Parameters):\s*\n((?:\s+.+\n?)+)",
        docstring,
        re.IGNORECASE | re.MULTILINE,
    )
    if args_match:
        block = args_match.group(1)
        pattern = re.compile(r"^\s+(\w+)\s*(?:\([^)]+\))?\s*:\s*(.+?)(?=\n\s+\w+|\Z)", re.M | re.S)
        for match in pattern.finditer(block):
            descriptions[match.group(1)] = re.sub(r"\s+", " ", match.group(2)).strip()
    for match in re.finditer(r":param\s+(\w+)\s*:\s*(.+?)(?=\n\s*:|$)", docstring, re.S):
        descriptions.setdefault(match.group(1), re.sub(r"\s+", " ", match.group(2)).strip())
    return descriptions


def _tool_call(func: ast.AsyncFunctionDef | ast.FunctionDef) -> ast.Call | None:
    for decorator in func.decorator_list:
        call = _decorator_call(decorator)
        if call and _decorator_name(call) == "tool":
            return call
    return None


def _tool_value(call: ast.Call, keyword: str, position: int) -> str | None:
    value = _literal_str(_keyword(call, keyword))
    if value is not None:
        return value
    if len(call.args) > position:
        return _literal_str(call.args[position])
    return None


def _category_value(call: ast.Call) -> str:
    category = _keyword(call, "category")
    if isinstance(category, ast.Attribute):
        return category.attr.lower()
    if isinstance(category, ast.Constant):
        return str(category.value)
    return _safe_unparse(category) or "unknown"


def _function_parameters(func: ast.AsyncFunctionDef | ast.FunctionDef) -> list[dict[str, Any]]:
    args = func.args.posonlyargs + func.args.args
    defaults = [None] * (len(args) - len(func.args.defaults)) + list(func.args.defaults)
    descriptions = _docstring_param_descriptions(ast.get_docstring(func))
    params: list[dict[str, Any]] = []
    for arg, default in zip(args, defaults):
        if arg.arg == "self":
            continue
        param = {
            "name": arg.arg,
            "type": _safe_unparse(arg.annotation) or "Any",
            "required": default is None,
        }
        if default is not None:
            param["default"] = _safe_unparse(default)
        if arg.arg in descriptions:
            param["description"] = descriptions[arg.arg]
        params.append(param)
    return params


def estimate_tool_token_cost(description: str, parameters: list[dict[str, Any]]) -> int:
    """Estimate schema prompt cost from description plus JSON parameter schema."""
    payload = {"description": description, "parameters": parameters}
    return max(1, math.ceil(len(json.dumps(payload, sort_keys=True)) / 4))


def _feature_key_for_path(path: Path) -> str:
    rel = path.relative_to(FEATURES_ROOT)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def _discover_tools_by_feature() -> dict[str, list[ToolInventory]]:
    tools_by_feature: dict[str, list[ToolInventory]] = {}
    for path in sorted(FEATURES_ROOT.rglob("*.py")):
        if path.name == "base.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        feature_key = _feature_key_for_path(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            call = _tool_call(node)
            if call is None:
                continue
            name = _tool_value(call, "name", 0) or node.name
            description = _tool_value(call, "description", 1) or ""
            params = _function_parameters(node)
            command_prefix = _literal_str(_keyword(call, "command_prefix"))
            tools_by_feature.setdefault(feature_key, []).append(
                ToolInventory(
                    name=name,
                    description=description,
                    category=_category_value(call),
                    command_prefix=command_prefix,
                    parameters=params,
                    token_cost_estimate=estimate_tool_token_cost(description, params),
                    enablement_state="enabled",
                    source=f"{_rel(path)}:{node.lineno}",
                )
            )
    for tools in tools_by_feature.values():
        tools.sort(key=lambda tool: tool.name)
    return tools_by_feature


def _discover_feature_class_by_module() -> dict[str, str]:
    classes: dict[str, str] = {}
    pattern = re.compile(r"^class\s+(\w+)\([^)]*\bFeature\b[^)]*\):", re.M | re.S)
    for path in sorted(FEATURES_ROOT.rglob("*.py")):
        if path.name == "base.py":
            continue
        key = _feature_key_for_path(path)
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            class_name = match.group(1)
            if class_name != "ProxyFeature":
                classes.setdefault(key, class_name)
    return classes


def discover_router_routes() -> list[RouteInventory]:
    routes: list[RouteInventory] = []
    for path in sorted(ENDPOINTS_ROOT.glob("*.py")):
        if path.name in {"__init__.py", "agent_helpers.py"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        prefix = ""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call = node.value
                if _decorator_name(call) == "APIRouter":
                    prefix = _literal_str(_keyword(call, "prefix")) or ""
                    break
        for node in ast.walk(tree):
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            for decorator in node.decorator_list:
                call = _decorator_call(decorator)
                if call is None:
                    continue
                method = _decorator_name(call).upper()
                if method not in {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}:
                    continue
                route_path = _literal_str(call.args[0] if call.args else _keyword(call, "path"))
                if route_path is not None:
                    routes.append(RouteInventory(method, f"{prefix}{route_path}", _rel(path)))
    return sorted(routes, key=lambda route: (route.source, route.path, route.method))


def discover_app_routes() -> list[RouteInventory]:
    path = PROJECT_ROOT / "kestrel_sovereign" / "server.py"
    routes: list[RouteInventory] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        for decorator in node.decorator_list:
            call = _decorator_call(decorator)
            if call is None:
                continue
            method = _decorator_name(call).upper()
            if method not in {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}:
                continue
            route_path = _literal_str(call.args[0] if call.args else _keyword(call, "path"))
            if route_path is not None:
                routes.append(RouteInventory(method, route_path, _rel(path)))
    return sorted(routes, key=lambda route: (route.path, route.method))


def build_inventory() -> Inventory:
    modules = discover_core_feature_modules()
    classes = discover_exported_feature_classes()
    class_by_module = _discover_feature_class_by_module()
    tools_by_feature = _discover_tools_by_feature()
    disabled = _get_disabled_features()

    features = [
        FeatureInventory(
            module=module,
            class_name=class_by_module.get(module),
            source=_module_source(module),
            enablement_state="disabled" if class_by_module.get(module) in disabled else "enabled",
            tools=tools_by_feature.get(module, []),
        )
        for module in modules
    ]

    commands = [
        CommandInventory(
            command=spec["cmd"],
            description=spec.get("description", ""),
            args=spec.get("args"),
            category=spec.get("category", "Built-in"),
            feature=None,
            source="kestrel_sovereign/command_handler.py",
        )
        for spec in BUILTIN_COMMAND_SPECS
    ]
    for feature in features:
        for tool in feature.tools:
            if tool.command_prefix:
                commands.append(
                    CommandInventory(
                        command=tool.command_prefix,
                        description=tool.description,
                        args=_parameters_summary(tool.parameters),
                        category=feature.module,
                        feature=feature.module,
                        source=tool.source,
                    )
                )
    commands.sort(key=lambda cmd: (cmd.category, cmd.command))
    return Inventory(
        feature_modules=modules,
        feature_classes=classes,
        entrypoint_feature_classes=discover_entrypoint_feature_class_names(),
        features=features,
        app_routes=discover_app_routes(),
        router_routes=discover_router_routes(),
        commands=commands,
    )


def _get_disabled_features() -> set[str]:
    try:
        from kestrel_sovereign.features import get_disabled_features

        return get_disabled_features()
    except ModuleNotFoundError:
        env_val = os.environ.get("KESTREL_DISABLED_FEATURES", "")
        return {name.strip() for name in env_val.split(",") if name.strip()}


def _module_source(module: str) -> str:
    package_feature = FEATURES_ROOT / module / "feature.py"
    package_init = FEATURES_ROOT / module / "__init__.py"
    single_file = FEATURES_ROOT / f"{module}.py"
    for path in (package_feature, package_init, single_file):
        if path.exists():
            return _rel(path)
    return f"kestrel_sovereign/features/{module}"


def _parameters_summary(parameters: list[dict[str, Any]]) -> str | None:
    if not parameters:
        return None
    parts = []
    for param in parameters:
        wrapper = "<{}>" if param.get("required") else "[{}]"
        parts.append(wrapper.format(param["name"]))
    return " ".join(parts)


def _table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def inventory_to_dict(inventory: Inventory) -> dict[str, Any]:
    return asdict(inventory)


def render_inventory_markdown(inventory: Inventory) -> str:
    lines = [
        GENERATED_START,
        "",
        "## Feature Module Inventory",
        "",
        "Features come from two sources:",
        "",
        "1. **Core features** — discovered from `kestrel_sovereign/features/` via `discover_feature_modules()`.",
        "2. **Package features** — installed packages registered with the `kestrel_sovereign.features` entry point group at runtime.",
        "",
        "The generated inventory below lists core features only: the in-tree surface discoverable from this checkout.",
        "Installed entry point feature classes are included in JSON output when present in the active environment.",
        "Runtime security policy can still deny a discovered tool at call time; static generation marks source-discovered tools as enabled unless their feature is disabled.",
        "",
        f"- Current audited snapshot: `{len(inventory.feature_modules)}` discoverable modules and `{len(inventory.feature_classes)}` exported `Feature` subclasses.",
        "",
    ]

    for feature in inventory.features:
        class_suffix = f" ({feature.class_name})" if feature.class_name else ""
        lines.append(f"### `{feature.module}`{class_suffix}")
        lines.append("")
        lines.append(f"- Source: [`{feature.source}`]({feature.source})")
        lines.append(f"- Enablement state: `{feature.enablement_state}`")
        if not feature.tools:
            lines.append("- Tools: none discovered")
            lines.append("")
            continue
        lines.append("")
        lines.append("| Tool | Command | Category | Params | Token cost | State |")
        lines.append("|---|---|---|---|---:|---|")
        for tool in feature.tools:
            command = f"`{tool.command_prefix}`" if tool.command_prefix else ""
            params = ", ".join(f"`{param['name']}`" for param in tool.parameters) or ""
            lines.append(
                f"| `{tool.name}` | {command} | `{tool.category}` | {params} | "
                f"{tool.token_cost_estimate} | `{tool.enablement_state}` |"
            )
        lines.append("")

    lines.extend([
        "## Public HTTP Surface",
        "",
        "### App-level routes in `kestrel_sovereign/server.py`",
        "",
    ])
    for route in inventory.app_routes:
        lines.append(f"- `{route.method} {route.path}`")

    lines.extend(["", "### Router families mounted by `kestrel_sovereign/server.py`", ""])
    routes_by_source: dict[str, list[RouteInventory]] = {}
    for route in inventory.router_routes:
        routes_by_source.setdefault(route.source, []).append(route)
    for source, routes in routes_by_source.items():
        lines.append(f"- [`{source}`]({source})")
        for route in routes:
            lines.append(f"  - `{route.method} {route.path}`")

    lines.extend(["", "## Command Surface", ""])
    lines.append("| Command | Source | Args | Description |")
    lines.append("|---|---|---|---|")
    for command in inventory.commands:
        args = f"`{command.args}`" if command.args else ""
        source = command.feature or "built-in"
        lines.append(
            f"| `{command.command}` | `{source}` | {args} | "
            f"{_table_cell(command.description)} |"
        )

    lines.extend(["", GENERATED_END, ""])
    return "\n".join(lines)


def render_inventory_json(inventory: Inventory) -> str:
    return json.dumps(inventory_to_dict(inventory), indent=2, sort_keys=True) + "\n"


def replace_generated_inventory(existing: str, generated: str) -> str:
    if GENERATED_START in existing and GENERATED_END in existing:
        before, rest = existing.split(GENERATED_START, 1)
        _, after = rest.split(GENERATED_END, 1)
        return before.rstrip() + "\n\n" + generated.rstrip() + "\n" + after

    insertion_heading = "\n## Authentication Surface\n"
    if insertion_heading in existing:
        before, after = existing.split(insertion_heading, 1)
        prefix = re.split(r"\n## Feature Module Inventory\n", before, maxsplit=1)[0].rstrip()
        return prefix + "\n\n" + generated.rstrip() + "\n" + insertion_heading + after

    return existing.rstrip() + "\n\n" + generated


def write_canonical_inventory(path: Path = CANONICAL_INVENTORY) -> None:
    inventory = build_inventory()
    generated = render_inventory_markdown(inventory)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(replace_generated_inventory(existing, generated), encoding="utf-8")
