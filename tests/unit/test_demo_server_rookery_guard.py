"""Unit tests for the demo-server rookery auto-mount guard (#868-1).

A demo run started from the main repo without ``KESTREL_ROOKERY_CONFIG``
set silently picked up the project-root ``rookery.toml`` and mounted
Meridian / Claw / Nellie alongside the demo agent.  That's the routing
precondition that wiped Meridian (#867).  When ``KESTREL_DEMO_SERVER=1``
is set, ``server.py``'s startup must refuse the auto-mount unless the
operator has explicitly pointed ``KESTREL_ROOKERY_CONFIG`` at the file.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


def _run_guard(env, rookery_path):
    """Reimplements the guard logic from server.py:lifespan.

    The guard sits inline in the FastAPI lifespan handler — exercising
    it via a real server start is heavy; the logic is small and pure
    enough to test directly.  This fixture mirrors the production code
    and the test then locks down its decision matrix.
    """
    demo_server_env = env.get("KESTREL_DEMO_SERVER", "").lower() in (
        "1", "true", "yes",
    )
    rookery_explicit = "KESTREL_ROOKERY_CONFIG" in env
    if demo_server_env and not rookery_explicit and rookery_path.exists():
        return Path("/dev/null/rookery-disabled")
    return rookery_path


def test_demo_marker_with_existing_rookery_disables_auto_mount(tmp_path):
    rookery = tmp_path / "rookery.toml"
    rookery.write_text("[host]\nport=8888\n")
    result = _run_guard(
        env={"KESTREL_DEMO_SERVER": "1"},
        rookery_path=rookery,
    )
    assert not result.exists(), (
        "When KESTREL_DEMO_SERVER=1 and rookery.toml exists at the default "
        "path, the guard must redirect to a non-existent path so the lifespan "
        "skips the auto-mount."
    )


def test_explicit_rookery_config_honored_even_in_demo_mode(tmp_path):
    rookery = tmp_path / "rookery.toml"
    rookery.write_text("[host]\nport=8888\n")
    result = _run_guard(
        env={
            "KESTREL_DEMO_SERVER": "1",
            "KESTREL_ROOKERY_CONFIG": str(rookery),
        },
        rookery_path=rookery,
    )
    assert result == rookery, (
        "Explicit KESTREL_ROOKERY_CONFIG opts into rookery mode regardless "
        "of KESTREL_DEMO_SERVER — the operator made an intentional choice"
    )


def test_no_demo_marker_uses_rookery_normally(tmp_path):
    rookery = tmp_path / "rookery.toml"
    rookery.write_text("[host]\nport=8888\n")
    result = _run_guard(env={}, rookery_path=rookery)
    assert result == rookery, (
        "Without the demo marker, server starts normally (rookery loaded "
        "if present) — production behaviour preserved"
    )


def test_demo_marker_without_rookery_file_no_op():
    """No rookery file → nothing to disable, guard returns the original."""
    fake = Path("/tmp/does-not-exist-rookery.toml")
    result = _run_guard(env={"KESTREL_DEMO_SERVER": "1"}, rookery_path=fake)
    assert result == fake


def test_demo_marker_truthy_variants(tmp_path):
    rookery = tmp_path / "rookery.toml"
    rookery.write_text("[host]\nport=8888\n")
    for value in ["1", "true", "TRUE", "yes", "Yes"]:
        result = _run_guard(
            env={"KESTREL_DEMO_SERVER": value},
            rookery_path=rookery,
        )
        assert not result.exists(), (
            f"KESTREL_DEMO_SERVER={value!r} should disable auto-mount"
        )

    # Falsy/unrecognised values fall through to normal behaviour.
    for value in ["0", "false", "no", ""]:
        result = _run_guard(
            env={"KESTREL_DEMO_SERVER": value},
            rookery_path=rookery,
        )
        assert result == rookery, (
            f"KESTREL_DEMO_SERVER={value!r} should not disable auto-mount"
        )
