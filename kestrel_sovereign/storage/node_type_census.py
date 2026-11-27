"""Read-only census of graph node types (#4153, step 1 of rv-4135-1).

rv-4135-1 orders "census before enforcement": count each ``node_type`` per
agent, find who writes it, assign owners, and only then (#4135-5) refuse
unregistered types. This module is the census. It changes nothing: it reads
``graph_nodes``, ``graph_node_owners`` and the node-type registry, and parses
source files without importing them.

Two halves:

* **Live rows**, from the database: rows per type, per owning agent, and the
  rows no ``graph_node_owners`` row claims. The database records no history of
  graph writes, so what it can count is the rows that are there now.
* **Write sites**, from the source of core and its in-tree features: every
  ``GraphNode(...)`` built in a function that writes a graph node, every call
  to a graph write method whose node was not built that way, and every SQL
  statement that inserts or updates ``graph_nodes``. A site whose type is a
  literal or a constant is resolved to it. A site whose type is decided at
  runtime is listed as dynamic: a rewrite of a node the function read, a
  write of a node it was handed or got from another call, an import of any
  type. A node counts as built in its function only when it is the
  ``GraphNode(...)`` call itself or a local name every binding of which is
  one; any other binding makes the write dynamic, even on a path that never
  writes it. ``self.CONSTANT`` resolves to the value its class, or a base
  class it names, defines; a subclass elsewhere that overrides it is not
  followed. The generic graph store, which every writer calls, is not a site,
  and nor is a storage facade's write of the node it was handed
  (:data:`GRAPH_WRITE_FORWARDERS`): their callers are.

Each site is attributed to the feature package it lives in, or to ``core``. A
type whose sites name exactly one owner has that owner; any other type with no
registry entry is reported as unowned, not guessed.

Run it against an agent database without changing it::

    python -m kestrel_sovereign.storage.node_type_census --sqlite PATH [--json]
    python -m kestrel_sovereign.storage.node_type_census --postgres-dsn-env NAME [--json]

The SQLite form opens the file read-only and runs no schema initialization.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import node_type_registry
from .node_type_registry import CORE_OWNER, Governance, NodeTypeEntry, derive_governance

#: Graph store methods that write a node, each with the position and keyword
#: of the argument that carries the node. A ``GraphNode`` built in a function
#: that calls one of them (or names one, as ``getattr(store, name)`` does) is a
#: write site, and so is a call whose node was not built in its function.
GRAPH_WRITE_NODE_ARGUMENTS: Mapping[str, Tuple[int, str]] = {
    "add_node": (0, "node"),
    "adopt_shared_content_node": (0, "node"),
    "compare_and_swap_node": (2, "new_node"),
}
GRAPH_WRITE_METHODS = frozenset(GRAPH_WRITE_NODE_ARGUMENTS)

#: The generic graph store. Every write site calls it, and its own SQL takes the
#: type as a parameter, so it is the write path rather than a writer.
GENERIC_GRAPH_STORE = "kestrel_sovereign.storage.async_graph_store"

#: This module's import name. Under ``python -m`` its ``__name__`` is
#: ``__main__``, which names no module the scan indexes.
CENSUS_MODULE = __spec__.name

#: Modules the scan does not read: the generic store, and the census, whose
#: patterns describe writes rather than make them.
UNSCANNED_MODULES = frozenset({GENERIC_GRAPH_STORE, CENSUS_MODULE})

#: Storage facades that hand the node they were given to the generic store,
#: by module and function. Like the generic store they are the write path, not
#: writers: their callers are the sites. Only a write of the function's own
#: node parameter, never rebound, is left out; anything else either function
#: writes is listed.
GRAPH_WRITE_FORWARDERS = frozenset(
    {
        ("kestrel_sovereign.storage.async_storage", "AsyncStorage.add_node"),
        ("kestrel_sovereign.storage.async_storage", "AsyncStorage.compare_and_swap_node"),
        ("kestrel_sovereign.storage.privacy_wrapper", "PrivacyEnforcingStorage.add_node"),
        (
            "kestrel_sovereign.storage.privacy_wrapper",
            "PrivacyEnforcingStorage._governed_compare_and_swap",
        ),
        ("kestrel_sovereign.storage.privacy_wrapper", "_PrivacyGoverningGraphStore.add_node"),
    }
)

#: Kinds of write site.
SITE_GRAPH_NODE = "graph_node"
SITE_SQL = "sql"

_SQL_WRITE = re.compile(r"\b(?:INSERT\s+(?:OR\s+\w+\s+)?INTO|UPDATE)\s+graph_nodes\b", re.I)
_SQL_INSERT = re.compile(
    r"INTO\s+graph_nodes\s*\(([^)]*)\)\s*VALUES\s*\(([^)]*)\)", re.I | re.S
)
_SQL_TYPE_LITERAL = re.compile(r"\bnode_type\s*=\s*'([^']*)'", re.I)
_SQL_LITERAL = re.compile(r"^'([^']*)'$")
_MAX_IMPORT_DEPTH = 6


@dataclass(frozen=True)
class WriteSite:
    """One place in the source that writes graph nodes."""

    module: str
    path: str
    line: int
    function: str
    kind: str
    node_type: Optional[str]
    expression: str
    owning_feature: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "path": self.path,
            "line": self.line,
            "function": self.function,
            "kind": self.kind,
            "node_type": self.node_type,
            "expression": self.expression,
            "owning_feature": self.owning_feature,
        }


@dataclass
class _ModuleInfo:
    name: str
    path: Path
    tree: ast.Module
    constants: Dict[str, str] = field(default_factory=dict)
    class_constants: Dict[str, Dict[str, str]] = field(default_factory=dict)
    class_bases: Dict[str, List[str]] = field(default_factory=dict)
    imports: Dict[str, Tuple[str, Optional[str]]] = field(default_factory=dict)


def _string_assignments(body: Iterable[ast.stmt]) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for statement in body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets = [statement.target]
            value = statement.value
        else:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    found[target.id] = value.value
    return found


def _package_of(name: str, path: Path) -> str:
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


def _absolute_module(info: _ModuleInfo, node: ast.ImportFrom) -> Optional[str]:
    if node.level == 0:
        return node.module
    base = _package_of(info.name, info.path).split(".")
    if node.level - 1 > len(base):
        return None
    base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _index_module(name: str, path: Path) -> _ModuleInfo:
    info = _ModuleInfo(name=name, path=path, tree=ast.parse(path.read_text(encoding="utf-8")))
    info.constants = _string_assignments(info.tree.body)
    for node in ast.walk(info.tree):
        if isinstance(node, ast.ClassDef):
            info.class_constants[node.name] = _string_assignments(node.body)
            info.class_bases[node.name] = [
                base.id for base in node.bases if isinstance(base, ast.Name)
            ]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    info.imports[alias.asname] = (alias.name, None)
        elif isinstance(node, ast.ImportFrom):
            source = _absolute_module(info, node)
            if source is None:
                continue
            for alias in node.names:
                info.imports[alias.asname or alias.name] = (source, alias.name)
    return info


class _Resolver:
    """Static resolution of a ``node_type`` expression to its string value."""

    def __init__(self, modules: Mapping[str, _ModuleInfo]):
        self._modules = modules

    def module_name(self, info: _ModuleInfo, local: str) -> Optional[str]:
        """The module a local name refers to, when it names a module."""
        target = info.imports.get(local)
        if target is None:
            return None
        source, attribute = target
        if attribute is None:
            return source
        candidate = f"{source}.{attribute}"
        return candidate if candidate in self._modules else None

    def name(self, module: str, name: str, depth: int = 0) -> Optional[str]:
        info = self._modules.get(module)
        if info is None or depth > _MAX_IMPORT_DEPTH:
            return None
        if name in info.constants:
            return info.constants[name]
        target = info.imports.get(name)
        if target is not None and target[1] is not None:
            return self.name(target[0], target[1], depth + 1)
        return None

    def class_attribute(
        self, module: str, class_name: str, attribute: str, depth: int = 0
    ) -> Optional[str]:
        info = self._modules.get(module)
        if info is None or depth > _MAX_IMPORT_DEPTH:
            target = None
        elif class_name in info.class_constants:
            if attribute in info.class_constants[class_name]:
                return info.class_constants[class_name][attribute]
            for base in info.class_bases.get(class_name, ()):
                value = self.class_attribute(module, base, attribute, depth + 1)
                if value is not None:
                    return value
            return None
        else:
            target = info.imports.get(class_name)
        if target is not None and target[1] is not None:
            return self.class_attribute(target[0], target[1], attribute, depth + 1)
        return None

    def expression(self, info: _ModuleInfo, node: ast.expr, enclosing_class: Optional[str]) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.name(info.name, node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base = node.value.id
            if base in ("self", "cls"):
                if enclosing_class is None:
                    return None
                return self.class_attribute(info.name, enclosing_class, node.attr)
            module = self.module_name(info, base)
            if module is not None:
                return self.name(module, node.attr)
            return self.class_attribute(info.name, base, node.attr)
        return None


def _feature_classes(info: _ModuleInfo) -> List[str]:
    """The ``Feature`` subclasses a feature module defines.

    Mirrors :func:`kestrel_sovereign.features.find_feature_class`:
    ``ProxyFeature`` is only created programmatically and does not count.
    """
    return sorted(
        class_name
        for class_name, bases in info.class_bases.items()
        if "Feature" in bases and class_name != "ProxyFeature"
    )


def _feature_owners(package_root: Path, modules: Mapping[str, _ModuleInfo]) -> Dict[str, str]:
    """Each in-tree feature's module prefix, mapped to its feature's name.

    A feature's name is its class name, the identity an agent's ``features``
    allowlist and ``Feature.name`` use. Discovery is the same as
    :func:`kestrel_sovereign.features.discover_feature_modules`: a package's
    ``feature.py`` (else its ``__init__.py``), or a single-file module.
    """
    package = package_root.name
    features_dir = package_root / "features"
    owners: Dict[str, str] = {}
    if not features_dir.is_dir():
        return owners
    for item in sorted(features_dir.iterdir()):
        if item.name.startswith(("_", ".")):
            continue
        prefix = f"{package}.features.{item.stem}"
        if item.is_dir() and (item / "feature.py").exists():
            module = f"{prefix}.feature"
        elif item.is_dir() and (item / "__init__.py").exists():
            module = prefix
        elif item.is_file() and item.suffix == ".py" and item.name != "base.py":
            module = prefix
        else:
            continue
        info = modules.get(module)
        classes = [] if info is None else _feature_classes(info)
        if len(classes) > 1:
            raise ValueError(
                f"{module} defines more than one feature ({', '.join(classes)}); "
                "the census cannot say which one owns what it writes"
            )
        if classes:
            owners[prefix] = classes[0]
    return owners


def _owner_of(module: str, feature_owners: Mapping[str, str]) -> str:
    for prefix, feature in feature_owners.items():
        if module == prefix or module.startswith(prefix + "."):
            return feature
    return CORE_OWNER


def _call_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _writes_graph_node(function: ast.AST) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in GRAPH_WRITE_METHODS:
                return True
        if isinstance(node, ast.Constant) and node.value in GRAPH_WRITE_METHODS:
            return True
    return False


def _is_graph_node_call(node: Optional[ast.AST]) -> bool:
    return isinstance(node, ast.Call) and _call_name(node) == "GraphNode"


def _write_method_alias(value: Optional[ast.AST]) -> Optional[str]:
    """The write method ``value`` names: ``getattr(store, "add_node")`` or ``store.add_node``."""
    if isinstance(value, ast.Attribute) and value.attr in GRAPH_WRITE_METHODS:
        return value.attr
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "getattr"
        and len(value.args) >= 2
        and isinstance(value.args[1], ast.Constant)
        and value.args[1].value in GRAPH_WRITE_METHODS
    ):
        return value.args[1].value
    return None


def _node_argument(call: ast.Call, method: str) -> Optional[ast.expr]:
    """The argument carrying the node a write call writes, when it can be found."""
    position, keyword = GRAPH_WRITE_NODE_ARGUMENTS[method]
    for argument in call.keywords:
        if argument.arg == keyword:
            return argument.value
    if any(isinstance(argument, ast.Starred) for argument in call.args[: position + 1]):
        return None
    return call.args[position] if len(call.args) > position else None


_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

#: How a name is bound in a function: as a parameter, by a plain assignment
#: of one value, or any other way (a loop, unpacking, ``with``, an import, an
#: augmented assignment, a ``global`` or ``nonlocal`` declaration, ...).
_PARAMETER = "parameter"
_ASSIGNED = "assigned"
_OTHER = "other"


@dataclass(frozen=True)
class _Binding:
    kind: str
    value: Optional[ast.expr] = None


def _own_namespace(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node whose bindings land in ``scope``'s namespace.

    Not the bodies of nested functions, classes or lambdas, nor a
    comprehension's loop variable, which have namespaces of their own.
    """
    body = scope.body if isinstance(scope.body, list) else [scope.body]
    stack: List[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _NESTED_SCOPES):
            continue
        for child in ast.iter_child_nodes(node):
            if isinstance(node, ast.comprehension) and child is node.target:
                continue
            stack.append(child)


def _sql_text(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) else "{}" for part in node.values
        )
    return None


def _sql_node_type(text: str) -> Tuple[Optional[str], str]:
    """The type a graph_nodes write names, and the text that decides it."""
    insert = _SQL_INSERT.search(text)
    if insert is not None:
        columns = [column.strip() for column in insert.group(1).split(",")]
        values = [value.strip() for value in insert.group(2).split(",")]
        if "node_type" in columns and len(values) == len(columns):
            value = values[columns.index("node_type")]
            literal = _SQL_LITERAL.match(value)
            return (literal.group(1) if literal else None), value
        return None, insert.group(0)
    literal = _SQL_TYPE_LITERAL.search(text)
    if literal is not None:
        return literal.group(1), literal.group(0)
    return None, " ".join(text.split())


class _SiteCollector:
    def __init__(self, info: _ModuleInfo, path: str, owner: str, resolver: _Resolver):
        self._info = info
        self._path = path
        self._owner = owner
        self._resolver = resolver
        self._parents: Dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(info.tree):
            for child in ast.iter_child_nodes(parent):
                self._parents[child] = parent
        # Names some assignment in this module binds to a write method, as
        # ``compare_and_swap = getattr(storage, "compare_and_swap_node")``
        # does. Only a call through one of them can be a write by another name.
        self._write_aliases = {
            target.id
            for node in ast.walk(info.tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
            and _write_method_alias(node.value) is not None
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        self.sites: List[WriteSite] = []

    def _scope(self, node: ast.AST) -> Tuple[Optional[ast.AST], str, Optional[str]]:
        """Enclosing function, its qualified name, and its nearest class."""
        names: List[str] = []
        function: Optional[ast.AST] = None
        enclosing_class: Optional[str] = None
        current = self._parents.get(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function = function or current
                names.append(current.name)
            elif isinstance(current, ast.ClassDef):
                enclosing_class = enclosing_class or current.name
                names.append(current.name)
            current = self._parents.get(current)
        return function, ".".join(reversed(names)) or "<module>", enclosing_class

    def _site(self, node: ast.AST, kind: str, node_type: Optional[str], expression: str, function: str) -> None:
        self.sites.append(
            WriteSite(
                module=self._info.name,
                path=self._path,
                line=node.lineno,
                function=function,
                kind=kind,
                node_type=node_type,
                expression=expression,
                owning_feature=self._owner,
            )
        )

    def _function_scopes(self, node: ast.AST) -> Iterator[ast.AST]:
        """The functions and lambdas enclosing ``node``, innermost first."""
        current = self._parents.get(node)
        while current is not None:
            if isinstance(current, _FUNCTION_SCOPES):
                yield current
            current = self._parents.get(current)

    def _bindings(self, scope: ast.AST, name: str) -> List[_Binding]:
        """Every binding of ``name`` in ``scope``'s own namespace."""
        found: List[_Binding] = []
        arguments = scope.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            arguments.vararg,
            arguments.kwarg,
        ):
            if argument is not None and argument.arg == name:
                found.append(_Binding(_PARAMETER))
        for node in _own_namespace(scope):
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store):
                parent = self._parents.get(node)
                if isinstance(parent, ast.Assign) and node in parent.targets:
                    found.append(_Binding(_ASSIGNED, parent.value))
                elif isinstance(parent, (ast.AnnAssign, ast.NamedExpr)) and parent.target is node:
                    if parent.value is not None:  # a bare annotation binds nothing
                        found.append(_Binding(_ASSIGNED, parent.value))
                else:
                    found.append(_Binding(_OTHER))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == name:
                    found.append(_Binding(_OTHER))
            elif isinstance(node, ast.alias):
                if (node.asname or node.name.partition(".")[0]) == name:
                    found.append(_Binding(_OTHER))
            elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
                if node.name == name:
                    found.append(_Binding(_OTHER))
            elif isinstance(node, ast.MatchMapping):
                if node.rest == name:
                    found.append(_Binding(_OTHER))
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                if name in node.names:
                    found.append(_Binding(_OTHER))
        return found

    def _resolve(self, node: ast.AST, name: str) -> List[_Binding]:
        """The bindings ``name`` refers to at ``node``; empty for a global or builtin."""
        for scope in self._function_scopes(node):
            found = self._bindings(scope, name)
            if found:
                return found
        return []

    def collect(self) -> List[WriteSite]:
        for node in ast.walk(self._info.tree):
            if isinstance(node, ast.Call):
                if _is_graph_node_call(node):
                    self._graph_node(node)
                    continue
                method = self._write_method(node)
                if method is not None:
                    self._write_call(node, method)
            elif isinstance(node, (ast.Constant, ast.JoinedStr)):
                self._sql(node)
        self.sites.sort(key=lambda site: site.line)
        return self.sites

    def _graph_node(self, node: ast.Call) -> None:
        function, qualname, enclosing_class = self._scope(node)
        if function is None or not _writes_graph_node(function):
            return
        expression = next((k.value for k in node.keywords if k.arg == "node_type"), None)
        if expression is None and len(node.args) > 1:
            expression = node.args[1]
        if expression is None:
            # ``GraphNode(**fields)``: the type is whatever ``fields`` holds.
            self._site(node, SITE_GRAPH_NODE, None, ast.unparse(node), qualname)
            return
        node_type = self._resolver.expression(self._info, expression, enclosing_class)
        self._site(node, SITE_GRAPH_NODE, node_type, ast.unparse(expression), qualname)

    def _write_method(self, call: ast.Call) -> Optional[str]:
        """The graph write method ``call`` invokes, by its own name or an alias."""
        if isinstance(call.func, ast.Attribute) and call.func.attr in GRAPH_WRITE_METHODS:
            return call.func.attr
        if isinstance(call.func, ast.Name) and call.func.id in self._write_aliases:
            for binding in self._resolve(call, call.func.id):
                method = _write_method_alias(binding.value)
                if method is not None:
                    return method
        return None

    def _built_here(self, call: ast.Call, argument: ast.expr) -> bool:
        """Whether the node written is a ``GraphNode(...)`` its function builds.

        Such a node is listed where it is built, by :meth:`_graph_node`.
        """
        if _is_graph_node_call(argument):
            return True
        if not isinstance(argument, ast.Name):
            return False
        bindings = self._resolve(call, argument.id)
        return bool(bindings) and all(
            binding.kind == _ASSIGNED and _is_graph_node_call(binding.value)
            for binding in bindings
        )

    def _forwards_its_parameter(self, call: ast.Call, argument: ast.expr, qualname: str) -> bool:
        """Whether ``call`` is a facade in :data:`GRAPH_WRITE_FORWARDERS` passing its node on."""
        if (self._info.name, qualname) not in GRAPH_WRITE_FORWARDERS:
            return False
        if not isinstance(argument, ast.Name):
            return False
        function = next(self._function_scopes(call), None)
        if function is None:
            return False
        return [binding.kind for binding in self._bindings(function, argument.id)] == [_PARAMETER]

    def _write_call(self, call: ast.Call, method: str) -> None:
        _, qualname, _ = self._scope(call)
        argument = _node_argument(call, method)
        if argument is None:
            self._site(call, SITE_GRAPH_NODE, None, ast.unparse(call), qualname)
            return
        if self._built_here(call, argument) or self._forwards_its_parameter(call, argument, qualname):
            return
        self._site(call, SITE_GRAPH_NODE, None, ast.unparse(argument), qualname)

    def _sql(self, node: ast.expr) -> None:
        parent = self._parents.get(node)
        if isinstance(parent, (ast.Expr, ast.JoinedStr)):
            return  # a docstring, or one part of an f-string handled whole
        text = _sql_text(node)
        if text is None or _SQL_WRITE.search(text) is None:
            return
        node_type, expression = _sql_node_type(text)
        _, qualname, _ = self._scope(node)
        self._site(node, SITE_SQL, node_type, expression, qualname)


def default_package_root() -> Path:
    """The installed ``kestrel_sovereign`` package directory."""
    return Path(__file__).resolve().parents[1]


def scan_write_sites(package_root: Optional[Path] = None) -> Tuple[WriteSite, ...]:
    """Every graph-node write site in ``package_root``, without importing it."""
    root = (package_root or default_package_root()).resolve()
    modules: Dict[str, _ModuleInfo] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join(parts)
        modules[name] = _index_module(name, path)
    resolver = _Resolver(modules)
    feature_owners = _feature_owners(root, modules)
    sites: List[WriteSite] = []
    for name, info in sorted(modules.items()):
        if name in UNSCANNED_MODULES:
            continue
        path = info.path.relative_to(root.parent).as_posix()
        sites.extend(
            _SiteCollector(info, path, _owner_of(name, feature_owners), resolver).collect()
        )
    return tuple(sites)


@dataclass(frozen=True)
class NodeTypeCensusRow:
    """Everything the census knows about one node type."""

    node_type: str
    total_rows: int
    rows_by_agent: Mapping[str, int]
    unowned_rows: int
    write_sites: Tuple[WriteSite, ...]
    registered: Optional[NodeTypeEntry]
    governance: Governance

    @property
    def candidate_owners(self) -> Tuple[str, ...]:
        """Every owner the write sites name."""
        return tuple(sorted({site.owning_feature for site in self.write_sites}))

    @property
    def census_owner(self) -> Optional[str]:
        """The owner the write sites identify: exactly one, or none."""
        owners = self.candidate_owners
        return owners[0] if len(owners) == 1 else None

    @property
    def owning_feature(self) -> Optional[str]:
        """The registered owner, else the census owner; ``None`` is unowned."""
        if self.registered is not None:
            return self.registered.owning_feature
        return self.census_owner

    @property
    def foreign_write_sites(self) -> Tuple[WriteSite, ...]:
        """Sites outside the registered owner: evidence for #4135-5."""
        if self.registered is None:
            return ()
        owner = self.registered.owning_feature
        return tuple(site for site in self.write_sites if site.owning_feature != owner)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type,
            "total_rows": self.total_rows,
            "rows_by_agent": dict(self.rows_by_agent),
            "unowned_rows": self.unowned_rows,
            "write_site_count": len(self.write_sites),
            "write_sites": [site.to_dict() for site in self.write_sites],
            "candidate_owners": list(self.candidate_owners),
            "census_owner": self.census_owner,
            "owning_feature": self.owning_feature,
            "registered": None if self.registered is None else self.registered.to_dict(),
            "foreign_write_sites": [site.to_dict() for site in self.foreign_write_sites],
            "governance": self.governance.to_dict(),
        }


def _forwarder_names() -> List[str]:
    return [f"{module}:{function}" for module, function in sorted(GRAPH_WRITE_FORWARDERS)]


@dataclass(frozen=True)
class NodeTypeCensus:
    """The census report."""

    types: Tuple[NodeTypeCensusRow, ...]
    dynamic_sites: Tuple[WriteSite, ...]
    registry_present: bool

    def row(self, node_type: str) -> NodeTypeCensusRow:
        for row in self.types:
            if row.node_type == node_type:
                return row
        raise KeyError(node_type)

    @property
    def unowned(self) -> Tuple[str, ...]:
        """Types with no registry entry and no single owner among their writers."""
        return tuple(row.node_type for row in self.types if row.owning_feature is None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_present": self.registry_present,
            "types": [row.to_dict() for row in self.types],
            "dynamic_write_sites": [site.to_dict() for site in self.dynamic_sites],
            "unowned": list(self.unowned),
            "unscanned_modules": sorted(UNSCANNED_MODULES),
            "write_forwarders": _forwarder_names(),
        }

    def render(self) -> str:
        lines = ["Graph node-type census (read-only)", ""]
        if not self.registry_present:
            lines += ["The database has no node-type registry table.", ""]
        for row in self.types:
            agents = ", ".join(f"{agent}={count}" for agent, count in sorted(row.rows_by_agent.items()))
            entry = row.registered
            if entry is None:
                registered = "unregistered"
            else:
                registered = (
                    f"registered to {entry.owning_feature}, home "
                    f"{entry.canonical_home.value}:{entry.canonical_location}, "
                    f"{'projection' if entry.is_projection else 'original'}"
                )
            lines.append(f"{row.node_type}: {row.total_rows} row(s)")
            lines.append(f"  by agent: {agents or 'none'}; without an owning agent: {row.unowned_rows}")
            lines.append(f"  owner: {row.owning_feature or 'UNOWNED'} ({registered})")
            lines.append(
                f"  governed: {'yes' if row.governance.governed else 'no'} ({row.governance.reason})"
            )
            for site in row.write_sites:
                marker = " [outside the registered owner]" if site in row.foreign_write_sites else ""
                lines.append(
                    f"  writer: {site.path}:{site.line} {site.function} "
                    f"({site.owning_feature}, {site.kind}){marker}"
                )
            lines.append("")
        lines.append("Write sites whose type is decided at runtime:")
        for site in self.dynamic_sites:
            lines.append(
                f"  {site.path}:{site.line} {site.function} ({site.owning_feature}, "
                f"{site.kind}): {site.expression}"
            )
        lines += [
            "",
            f"Not read: {', '.join(sorted(UNSCANNED_MODULES))}",
            "Not listed, as they pass on the node they are handed: "
            + ", ".join(_forwarder_names()),
            "",
            f"Unowned: {', '.join(self.unowned) or 'none'}",
        ]
        return "\n".join(lines) + "\n"


async def _live_rows(db) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]], Dict[str, int]]:
    totals = {
        str(node_type): int(count)
        for node_type, count in await db.fetchall(
            "SELECT node_type, COUNT(*) FROM graph_nodes GROUP BY node_type"
        )
    }
    by_agent: Dict[str, Dict[str, int]] = {}
    for node_type, agent_id, count in await db.fetchall(
        "SELECT graph_nodes.node_type, graph_node_owners.agent_id, COUNT(*) "
        "FROM graph_nodes JOIN graph_node_owners "
        "ON graph_node_owners.node_id = graph_nodes.node_id "
        "GROUP BY graph_nodes.node_type, graph_node_owners.agent_id"
    ):
        by_agent.setdefault(str(node_type), {})[str(agent_id)] = int(count)
    unowned = {
        str(node_type): int(count)
        for node_type, count in await db.fetchall(
            "SELECT node_type, COUNT(*) FROM graph_nodes WHERE NOT EXISTS ("
            "SELECT 1 FROM graph_node_owners "
            "WHERE graph_node_owners.node_id = graph_nodes.node_id"
            ") GROUP BY node_type"
        )
    }
    return totals, by_agent, unowned


async def run_census(db, *, write_sites: Optional[Sequence[WriteSite]] = None) -> NodeTypeCensus:
    """Count ``db``'s node types and join them with their write sites and entries.

    Only reads. ``write_sites`` defaults to :func:`scan_write_sites` over the
    installed package.
    """
    sites = tuple(scan_write_sites() if write_sites is None else write_sites)
    totals, by_agent, unowned = await _live_rows(db)
    registry_present = await db.table_exists(node_type_registry.TABLE)
    registered = (
        {entry.node_type: entry for entry in await node_type_registry.list_node_types(db)}
        if registry_present
        else {}
    )
    typed: Dict[str, List[WriteSite]] = {}
    for site in sites:
        if site.node_type is not None:
            typed.setdefault(site.node_type, []).append(site)
    names = sorted(set(totals) | set(typed) | set(registered))
    rows = tuple(
        NodeTypeCensusRow(
            node_type=name,
            total_rows=totals.get(name, 0),
            rows_by_agent=dict(sorted(by_agent.get(name, {}).items())),
            unowned_rows=unowned.get(name, 0),
            write_sites=tuple(typed.get(name, ())),
            registered=registered.get(name),
            governance=derive_governance(name),
        )
        for name in names
    )
    dynamic = tuple(site for site in sites if site.node_type is None)
    return NodeTypeCensus(types=rows, dynamic_sites=dynamic, registry_present=registry_present)


async def _leave_schema_unchanged(db: Any) -> None:
    """Schema initializer for the census: run no DDL and no migration."""


async def _census_of(args: argparse.Namespace) -> NodeTypeCensus:
    from .async_database import AsyncDatabase, _close_owned_database

    if args.sqlite is not None:
        if not args.sqlite.is_file():
            raise FileNotFoundError(f"no agent database at {args.sqlite}")
        db = await AsyncDatabase.sqlite(
            os.fspath(args.sqlite),
            schema_initializer=_leave_schema_unchanged,
            cold_read=True,
        )
    else:
        dsn = os.environ.get(args.postgres_dsn_env)
        if not dsn:
            raise ValueError(f"{args.postgres_dsn_env} is not exported")
        db = await AsyncDatabase.postgres(dsn, schema_initializer=_leave_schema_unchanged)
    try:
        census = await run_census(db)
        if args.sqlite is not None:
            # An immutable read is blind to a writer that commits during it.
            db.backend.assert_cold_read_still_valid()
        return census
    finally:
        await _close_owned_database(db, sys.exception())
        if args.sqlite is not None:
            db.backend.warn_if_wal_state_was_stranded()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--sqlite", type=Path, help="An existing agent database file")
    target.add_argument("--postgres-dsn-env", help="Name of an exported PostgreSQL DSN variable")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args(argv)
    census = asyncio.run(_census_of(args))
    if args.json:
        sys.stdout.write(json.dumps(census.to_dict(), indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(census.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
