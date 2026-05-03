"""Unit tests for the keys step."""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.env_file import read_env, write_env
from kestrel_sovereign.setup.prompts import StubPrompter
from kestrel_sovereign.setup.steps import keys


def _make_ctx(tmp_path: Path, flow: Flow, *, answers=None) -> SetupContext:
    return SetupContext(
        project_dir=tmp_path,
        agent_data_root=tmp_path / "agent_data",
        flow=flow,
        prompter=StubPrompter(answers=list(answers or [])),
    )


def test_keys_generates_data_key_when_absent(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)
    env = read_env(tmp_path / ".env")
    assert "KESTREL_DATA_KEY" in env
    # The generated key must be a valid Fernet key.
    Fernet(env["KESTREL_DATA_KEY"].encode("ascii"))


def test_keys_never_regenerates_existing_data_key(tmp_path):
    """Critical: regenerating KESTREL_DATA_KEY would brick existing DBs."""
    p = tmp_path / ".env"
    original_key = Fernet.generate_key().decode("ascii")
    write_env(p, {"KESTREL_DATA_KEY": original_key})

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)

    env = read_env(p)
    assert env["KESTREL_DATA_KEY"] == original_key


def test_keys_quickstart_generates_api_key_too(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)
    env = read_env(tmp_path / ".env")
    assert env.get("KESTREL_API_KEY")


def test_keys_check_mode_blocks_when_data_key_missing(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.CHECK)
    keys.run(ctx)
    assert any("KESTREL_DATA_KEY missing" in b for b in ctx.blockers)
    # Check mode never writes.
    assert not (tmp_path / ".env").exists()


def test_keys_check_mode_silent_when_data_key_present(tmp_path):
    write_env(
        tmp_path / ".env",
        {"KESTREL_DATA_KEY": Fernet.generate_key().decode("ascii")},
    )
    ctx = _make_ctx(tmp_path, Flow.CHECK)
    keys.run(ctx)
    assert ctx.blockers == []


def test_keys_interactive_asks_about_api_key(tmp_path):
    """Interactive flow prompts a yes/no for API key generation."""
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=[True])
    keys.run(ctx)
    env = read_env(tmp_path / ".env")
    assert env.get("KESTREL_API_KEY")


def test_keys_interactive_skips_api_key_when_declined(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=[False])
    keys.run(ctx)
    env = read_env(tmp_path / ".env")
    assert "KESTREL_API_KEY" not in env


def test_keys_idempotent_when_everything_set(tmp_path):
    """Re-running with all keys set must not produce a new backup."""
    write_env(
        tmp_path / ".env",
        {
            "KESTREL_DATA_KEY": Fernet.generate_key().decode("ascii"),
            "KESTREL_API_KEY": "stable-key",
        },
    )
    backups_before = list(tmp_path.glob(".env.backup-*"))

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)

    backups_after = list(tmp_path.glob(".env.backup-*"))
    assert backups_before == backups_after
