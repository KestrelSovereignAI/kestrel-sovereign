"""Unit tests for the demo-server multi_agent auto-mount guard (#868-1).

A demo run started from the main repo without ``KESTREL_MULTI_AGENT_CONFIG``
set silently picked up the project-root ``multi_agent.toml`` and mounted
Meridian / Claw / Nellie alongside the demo agent.  That's the routing
precondition that wiped Meridian (#867).  When ``KESTREL_DEMO_SERVER=1``
is set, ``server.py``'s startup must refuse the auto-mount unless the
operator has explicitly pointed ``KESTREL_MULTI_AGENT_CONFIG`` at the file.

These tests exercise the real ``server.resolve_multi_agent_path`` helper —
the lifespan handler calls the same function, so the test can't drift
from production behaviour the way an inlined re-implementation could.
"""
from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import textwrap

from server import resolve_multi_agent_path


def test_demo_marker_with_existing_multi_agent_disables_auto_mount(tmp_path, monkeypatch):
    multi_agent = tmp_path / "multi_agent.toml"
    multi_agent.write_text("[host]\nport=8888\n")
    monkeypatch.chdir(tmp_path)  # .exists() check uses cwd-relative path
    result = resolve_multi_agent_path({"KESTREL_DEMO_SERVER": "1"})
    assert not result.exists(), (
        "When KESTREL_DEMO_SERVER=1 and multi_agent.toml exists at the default "
        "path, the guard must redirect to a non-existent path so the lifespan "
        "skips the auto-mount."
    )


def test_explicit_multi_agent_config_honored_even_in_demo_mode(tmp_path):
    multi_agent = tmp_path / "multi_agent.toml"
    multi_agent.write_text("[host]\nport=8888\n")
    result = resolve_multi_agent_path({
        "KESTREL_DEMO_SERVER": "1",
        "KESTREL_MULTI_AGENT_CONFIG": str(multi_agent),
    })
    assert result == multi_agent, (
        "Explicit KESTREL_MULTI_AGENT_CONFIG opts into multi_agent mode regardless "
        "of KESTREL_DEMO_SERVER — the operator made an intentional choice"
    )


def test_no_demo_marker_uses_multi_agent_normally(tmp_path, monkeypatch):
    multi_agent = tmp_path / "multi_agent.toml"
    multi_agent.write_text("[host]\nport=8888\n")
    monkeypatch.chdir(tmp_path)
    result = resolve_multi_agent_path({})
    assert result == Path("multi_agent.toml"), (
        "Without the demo marker, server starts normally (multi_agent loaded "
        "if present) — production behaviour preserved.  The default "
        "relative path is returned; the lifespan's .exists() check "
        "handles missing files."
    )


def test_demo_marker_without_multi_agent_file_no_op(tmp_path, monkeypatch):
    """No multi_agent file → nothing to disable, guard returns the default."""
    monkeypatch.chdir(tmp_path)  # tmp_path has no multi_agent.toml
    result = resolve_multi_agent_path({"KESTREL_DEMO_SERVER": "1"})
    assert result == Path("multi_agent.toml")


def test_demo_marker_truthy_variants(tmp_path, monkeypatch):
    multi_agent = tmp_path / "multi_agent.toml"
    multi_agent.write_text("[host]\nport=8888\n")
    monkeypatch.chdir(tmp_path)
    for value in ["1", "true", "TRUE", "yes", "Yes"]:
        result = resolve_multi_agent_path({"KESTREL_DEMO_SERVER": value})
        assert not result.exists(), (
            f"KESTREL_DEMO_SERVER={value!r} should disable auto-mount"
        )

    # Falsy/unrecognised values fall through to normal behaviour.
    for value in ["0", "false", "no", ""]:
        result = resolve_multi_agent_path({"KESTREL_DEMO_SERVER": value})
        assert result == Path("multi_agent.toml"), (
            f"KESTREL_DEMO_SERVER={value!r} should not disable auto-mount"
        )


def _calls_named_on_outer_scope(function: object, name: str) -> bool:
    """Return whether ``function`` calls ``name`` in its executable outer scope."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    outer_function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )

    class OuterScopeCallFinder(ast.NodeVisitor):
        def __init__(self) -> None:
            self.found = False

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == name:
                self.found = True
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    finder = OuterScopeCallFinder()
    for statement in outer_function.body:
        finder.visit(statement)
    return finder.found


def _enters_async_context_named_on_outer_scope(function: object, name: str) -> bool:
    """Return whether ``function`` enters ``name`` in an outer-scope async with."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    outer_function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )

    class OuterScopeAsyncWithFinder(ast.NodeVisitor):
        def __init__(self) -> None:
            self.found = False

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
            for item in node.items:
                context_expr = item.context_expr
                if (
                    isinstance(context_expr, ast.Call)
                    and isinstance(context_expr.func, ast.Name)
                    and context_expr.func.id == name
                ):
                    self.found = True
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    finder = OuterScopeAsyncWithFinder()
    for statement in outer_function.body:
        finder.visit(statement)
    return finder.found


async def _outer_scope_with_only_nested_startup_context(_lifespan_startup):
    async def nested_lifespan():
        async with _lifespan_startup(None):
            yield

    return nested_lifespan


async def _outer_scope_with_bare_startup_context_factory(_lifespan_startup):
    _lifespan_startup(None)


def _outer_scope_with_only_nested_resolve_calls(resolve_multi_agent_path):
    def nested_function():
        resolve_multi_agent_path({})

    class NestedClass:
        def method(self):
            resolve_multi_agent_path({})

    return nested_function, (lambda: resolve_multi_agent_path({})), NestedClass


def test_async_context_check_ignores_nested_lexical_scopes():
    """A nested context entry cannot satisfy the lifespan wiring guard."""
    assert not _enters_async_context_named_on_outer_scope(
        _outer_scope_with_only_nested_startup_context,
        "_lifespan_startup",
    )


def test_async_context_check_rejects_a_bare_startup_context_factory_call():
    """Constructing a startup context manager is not the same as entering it."""
    assert not _enters_async_context_named_on_outer_scope(
        _outer_scope_with_bare_startup_context_factory,
        "_lifespan_startup",
    )


def test_outer_scope_call_check_ignores_nested_lexical_scopes():
    """A never-invoked nested resolver call cannot satisfy the startup guard."""
    assert not _calls_named_on_outer_scope(
        _outer_scope_with_only_nested_resolve_calls,
        "resolve_multi_agent_path",
    )


def test_lifespan_reaches_resolve_multi_agent_path_through_startup_helper():
    """Belt-and-braces — confirm the real lifespan startup path uses the helper.

    A reviewer rightly flagged the previous version of this file as a
    parallel reimplementation that could pass while the lifespan code
    diverged.  The cancellation-safe lifespan delegates startup, so this test
    follows that production call path rather than requiring an inlined call.
    """
    import server

    assert _enters_async_context_named_on_outer_scope(
        server.lifespan,
        "_lifespan_startup",
    ), (
        "server.lifespan must enter _lifespan_startup through an async with so "
        "multi-agent startup policy remains part of the production lifespan."
    )
    assert _calls_named_on_outer_scope(
        server._lifespan_startup,
        "resolve_multi_agent_path",
    ), (
        "server._lifespan_startup must invoke resolve_multi_agent_path on its "
        "executable outer scope; unit tests exercise that helper and production "
        "must reach it."
    )
