"""Scaffold generator for forged features.

Turns a declarative :class:`ForgeSpec` into the source files of a loadable
feature package. The generated ``feature.py`` subclasses the sovereign
``Feature`` base, exposes the requested tools as ``@tool`` methods returning
``ToolResult``, and therefore satisfies the same feature-registry contract every
in-tree feature does (discoverable class, ``tool_description`` property,
``initialize`` coroutine, ToolResult-returning tools).

Everything here is pure text generation — no filesystem writes — so the renderer
is unit-testable and the caller (:mod:`store`) owns where the files land.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Parameter types the scaffold maps onto Python annotations for the generated
# tool signatures. Anything else falls back to ``str``.
_TYPE_MAP = {
    "string": "str",
    "str": "str",
    "text": "str",
    "integer": "int",
    "int": "int",
    "number": "float",
    "float": "float",
    "boolean": "bool",
    "bool": "bool",
    "object": "Dict[str, Any]",
    "dict": "Dict[str, Any]",
    "array": "List[Any]",
    "list": "List[Any]",
}


class SpecError(ValueError):
    """Raised when a forge spec is malformed."""


def _is_identifier(name: str) -> bool:
    return bool(name) and name.isidentifier() and not keyword.iskeyword(name)


def to_snake_case(name: str) -> str:
    """Normalize a feature name to a snake_case module name (no ``feature`` suffix)."""
    name = str(name).strip()
    # Strip a trailing "Feature" / "_feature" so callers can pass either form.
    name = re.sub(r"[_\s]*feature$", "", name, flags=re.IGNORECASE)
    # CamelCase -> snake, then non-alphanumerics -> underscore.
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return name


def to_class_name(name: str) -> str:
    """Normalize a feature name to its ``PascalCaseFeature`` class name."""
    snake = to_snake_case(name)
    pascal = "".join(part.capitalize() for part in snake.split("_") if part)
    if not pascal:
        raise SpecError(f"Cannot derive a class name from {name!r}")
    return f"{pascal}Feature"


@dataclass(frozen=True)
class ToolParam:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True

    @property
    def annotation(self) -> str:
        return _TYPE_MAP.get(self.type.lower(), "str")

    def signature_fragment(self) -> str:
        if self.required:
            return f"{self.name}: {self.annotation}"
        # Optional params default to None so the generated tool is callable with
        # only its required args.
        return f"{self.name}: Optional[{self.annotation}] = None"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters: List[ToolParam] = field(default_factory=list)


@dataclass(frozen=True)
class ForgeSpec:
    name: str                       # normalized feature name (as supplied)
    class_name: str
    module_name: str
    purpose: str
    tools: List[ToolSpec]
    permissions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "class_name": self.class_name,
            "module_name": self.module_name,
            "purpose": self.purpose,
            "permissions": list(self.permissions),
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": [
                        {
                            "name": p.name,
                            "type": p.type,
                            "description": p.description,
                            "required": p.required,
                        }
                        for p in t.parameters
                    ],
                }
                for t in self.tools
            ],
        }


def parse_spec(raw: Dict[str, Any]) -> ForgeSpec:
    """Validate and normalize a raw spec dict into a :class:`ForgeSpec`.

    Raises :class:`SpecError` on any structural problem so the caller can surface
    a clear failure instead of scaffolding a broken package.
    """
    if not isinstance(raw, dict):
        raise SpecError("spec must be an object")

    name = raw.get("name")
    if not name or not str(name).strip():
        raise SpecError("spec.name is required")

    class_name = to_class_name(name)
    module_name = to_snake_case(name)
    if not module_name:
        raise SpecError(f"spec.name {name!r} does not yield a valid module name")

    purpose = str(raw.get("purpose") or "").strip()

    raw_tools = raw.get("tools") or []
    if not isinstance(raw_tools, list):
        raise SpecError("spec.tools must be a list")
    if not raw_tools:
        raise SpecError("spec.tools must declare at least one tool")

    tools: List[ToolSpec] = []
    seen_tool_names: set = set()
    for entry in raw_tools:
        if not isinstance(entry, dict):
            raise SpecError("each tool must be an object")
        tname = entry.get("name")
        if not tname or not _is_identifier(str(tname)):
            raise SpecError(f"tool name {tname!r} is not a valid Python identifier")
        tname = str(tname)
        if tname in seen_tool_names:
            raise SpecError(f"duplicate tool name {tname!r}")
        seen_tool_names.add(tname)

        raw_params = entry.get("parameters") or []
        if not isinstance(raw_params, list):
            raise SpecError(f"tool {tname!r} parameters must be a list")
        params: List[ToolParam] = []
        seen_params: set = set()
        for p in raw_params:
            if not isinstance(p, dict):
                raise SpecError(f"tool {tname!r} has a non-object parameter")
            pname = p.get("name")
            if not pname or not _is_identifier(str(pname)):
                raise SpecError(
                    f"tool {tname!r} parameter {pname!r} is not a valid identifier"
                )
            pname = str(pname)
            if pname in seen_params:
                raise SpecError(f"tool {tname!r} has duplicate parameter {pname!r}")
            seen_params.add(pname)
            params.append(
                ToolParam(
                    name=pname,
                    type=str(p.get("type") or "string"),
                    description=str(p.get("description") or "").strip(),
                    required=bool(p.get("required", True)),
                )
            )
        # Required params must precede optional ones in a Python signature.
        params.sort(key=lambda pp: not pp.required)
        tools.append(
            ToolSpec(
                name=tname,
                description=str(entry.get("description") or "").strip(),
                parameters=params,
            )
        )

    raw_perms = raw.get("permissions") or []
    if not isinstance(raw_perms, list):
        raise SpecError("spec.permissions must be a list")
    permissions = [str(p).strip().lower() for p in raw_perms if str(p).strip()]

    return ForgeSpec(
        name=str(name).strip(),
        class_name=class_name,
        module_name=module_name,
        purpose=purpose,
        tools=tools,
        permissions=permissions,
    )


def _py_str(value: str) -> str:
    """Return a safe single-line Python string literal for ``value``."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def _render_tool_method(tool: ToolSpec) -> str:
    params = list(tool.parameters)
    sig_parts = ["self"] + [p.signature_fragment() for p in params]
    signature = ", ".join(sig_parts)

    doc_lines = [tool.description or f"Forged tool {tool.name}."]
    if params:
        doc_lines.append("")
        doc_lines.append("Args:")
        for p in params:
            doc_lines.append(f"    {p.name}: {p.description or p.type}")
    docstring = "\n        ".join(doc_lines)

    # The scaffolded body is a valid, loadable stub: it echoes its inputs and
    # returns a well-formed ToolResult so the package passes registry/contract
    # tests immediately. The forge author fills in real behavior post-approval.
    param_names = [p.name for p in params]
    data_pairs = ", ".join(f'"{n}": {n}' for n in param_names)
    data_literal = "{" + data_pairs + "}" if data_pairs else "{}"

    return f'''    @tool(
        name={_py_str(tool.name)},
        description={_py_str(tool.description or tool.name)},
        category=ToolCategory.UTILITY,
    )
    async def {tool.name}({signature}) -> ToolResult:
        """{docstring}
        """
        # Scaffolded stub — replace with real behavior. Returns a valid
        # ToolResult so the forged package is loadable and testable as-is.
        return ToolResult.ok(
            confirmation="{tool.name} scaffold invoked",
            data={{"tool": "{tool.name}", "args": {data_literal}}},
        )'''


def render_feature_module(spec: ForgeSpec) -> str:
    """Render the ``feature.py`` source for a forged feature package."""
    tool_methods = "\n\n".join(_render_tool_method(t) for t in spec.tools)
    perms_repr = ", ".join(_py_str(p) for p in spec.permissions)
    purpose = spec.purpose or f"Forged feature {spec.class_name}."

    return f'''"""{spec.class_name} — forged feature.

{purpose}

This package was scaffolded by FeatureForgeFeature (issue #2434). It is INERT
until approved by the Sovereign: it lives outside the feature-discovery path and
carries no entry point, so nothing loads, executes, or registers its tools until
the forge pipeline reaches the ``approved``/``loaded`` state.

Declared (narrowed) capabilities: {spec.permissions or "none"}.
"""

from typing import Any, Dict, List, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool

# Capabilities this feature declared to the Iron Rule gate. A narrowing of what
# the platform already grants the agent — never a widening.
FORGED_CAPABILITIES: List[str] = [{perms_repr}]


class {spec.class_name}(Feature):
    """{purpose}"""

    @property
    def tool_description(self) -> str:
        return {_py_str(purpose)}

    async def initialize(self) -> None:
        pass

{tool_methods}
'''


def render_init_module(spec: ForgeSpec) -> str:
    return (
        f'"""Forged feature package: {spec.class_name}."""\n\n'
        f"from .feature import {spec.class_name}\n\n"
        f'__all__ = ["{spec.class_name}"]\n'
    )


def render_test_module(spec: ForgeSpec) -> str:
    """Render a minimal pytest module that instantiates the forged feature and
    asserts the tool contract, mirroring the in-tree feature test pattern."""
    first = spec.tools[0]
    call_args = ", ".join(
        f'{p.name}={"0" if p.annotation in ("int", "float") else ("True" if p.annotation == "bool" else repr("x"))}'
        for p in first.parameters
        if p.required
    )
    return f'''"""Contract test for forged feature {spec.class_name}."""

from unittest.mock import MagicMock

import pytest

from .feature import {spec.class_name}


@pytest.mark.asyncio
async def test_{spec.module_name}_tools_return_toolresult():
    feature = {spec.class_name}(agent=MagicMock())
    await feature.initialize()
    tools = feature.get_tools()
    assert tools, "forged feature exposes no tools"
    result = await feature.{first.name}({call_args})
    assert result.status.value in ("ok", "partial", "error")
'''


def render_package(spec: ForgeSpec) -> Dict[str, str]:
    """Render every source file of the forged package as ``path -> content``."""
    return {
        "__init__.py": render_init_module(spec),
        "feature.py": render_feature_module(spec),
        f"test_{spec.module_name}.py": render_test_module(spec),
    }
