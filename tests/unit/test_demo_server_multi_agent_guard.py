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

import os
from pathlib import Path

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


def test_lifespan_actually_calls_resolve_multi_agent_path():
    """Belt-and-braces — confirm the lifespan handler uses the helper.

    A reviewer rightly flagged the previous version of this file as a
    parallel reimplementation that could pass while the lifespan code
    diverged.  This assertion locks in the contract: the lifespan body
    contains a literal call site for ``resolve_multi_agent_path``.
    """
    import inspect
    import server
    src = inspect.getsource(server.lifespan)
    assert "resolve_multi_agent_path(" in src, (
        "server.lifespan must invoke resolve_multi_agent_path(); the unit tests "
        "exercise that helper and the lifespan needs to reach it."
    )
