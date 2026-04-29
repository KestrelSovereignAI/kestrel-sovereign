"""Unit tests for the demo-server rookery auto-mount guard (#868-1).

A demo run started from the main repo without ``KESTREL_ROOKERY_CONFIG``
set silently picked up the project-root ``rookery.toml`` and mounted
Meridian / Claw / Nellie alongside the demo agent.  That's the routing
precondition that wiped Meridian (#867).  When ``KESTREL_DEMO_SERVER=1``
is set, ``server.py``'s startup must refuse the auto-mount unless the
operator has explicitly pointed ``KESTREL_ROOKERY_CONFIG`` at the file.

These tests exercise the real ``server.resolve_rookery_path`` helper —
the lifespan handler calls the same function, so the test can't drift
from production behaviour the way an inlined re-implementation could.
"""
from __future__ import annotations

import os
from pathlib import Path

from server import resolve_rookery_path


def test_demo_marker_with_existing_rookery_disables_auto_mount(tmp_path, monkeypatch):
    rookery = tmp_path / "rookery.toml"
    rookery.write_text("[host]\nport=8888\n")
    monkeypatch.chdir(tmp_path)  # .exists() check uses cwd-relative path
    result = resolve_rookery_path({"KESTREL_DEMO_SERVER": "1"})
    assert not result.exists(), (
        "When KESTREL_DEMO_SERVER=1 and rookery.toml exists at the default "
        "path, the guard must redirect to a non-existent path so the lifespan "
        "skips the auto-mount."
    )


def test_explicit_rookery_config_honored_even_in_demo_mode(tmp_path):
    rookery = tmp_path / "rookery.toml"
    rookery.write_text("[host]\nport=8888\n")
    result = resolve_rookery_path({
        "KESTREL_DEMO_SERVER": "1",
        "KESTREL_ROOKERY_CONFIG": str(rookery),
    })
    assert result == rookery, (
        "Explicit KESTREL_ROOKERY_CONFIG opts into rookery mode regardless "
        "of KESTREL_DEMO_SERVER — the operator made an intentional choice"
    )


def test_no_demo_marker_uses_rookery_normally(tmp_path, monkeypatch):
    rookery = tmp_path / "rookery.toml"
    rookery.write_text("[host]\nport=8888\n")
    monkeypatch.chdir(tmp_path)
    result = resolve_rookery_path({})
    assert result == Path("rookery.toml"), (
        "Without the demo marker, server starts normally (rookery loaded "
        "if present) — production behaviour preserved.  The default "
        "relative path is returned; the lifespan's .exists() check "
        "handles missing files."
    )


def test_demo_marker_without_rookery_file_no_op(tmp_path, monkeypatch):
    """No rookery file → nothing to disable, guard returns the default."""
    monkeypatch.chdir(tmp_path)  # tmp_path has no rookery.toml
    result = resolve_rookery_path({"KESTREL_DEMO_SERVER": "1"})
    assert result == Path("rookery.toml")


def test_demo_marker_truthy_variants(tmp_path, monkeypatch):
    rookery = tmp_path / "rookery.toml"
    rookery.write_text("[host]\nport=8888\n")
    monkeypatch.chdir(tmp_path)
    for value in ["1", "true", "TRUE", "yes", "Yes"]:
        result = resolve_rookery_path({"KESTREL_DEMO_SERVER": value})
        assert not result.exists(), (
            f"KESTREL_DEMO_SERVER={value!r} should disable auto-mount"
        )

    # Falsy/unrecognised values fall through to normal behaviour.
    for value in ["0", "false", "no", ""]:
        result = resolve_rookery_path({"KESTREL_DEMO_SERVER": value})
        assert result == Path("rookery.toml"), (
            f"KESTREL_DEMO_SERVER={value!r} should not disable auto-mount"
        )


def test_lifespan_actually_calls_resolve_rookery_path():
    """Belt-and-braces — confirm the lifespan handler uses the helper.

    A reviewer rightly flagged the previous version of this file as a
    parallel reimplementation that could pass while the lifespan code
    diverged.  This assertion locks in the contract: the lifespan body
    contains a literal call site for ``resolve_rookery_path``.
    """
    import inspect
    import server
    src = inspect.getsource(server.lifespan)
    assert "resolve_rookery_path(" in src, (
        "server.lifespan must invoke resolve_rookery_path(); the unit tests "
        "exercise that helper and the lifespan needs to reach it."
    )
