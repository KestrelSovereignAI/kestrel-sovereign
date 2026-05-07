"""CLI tests for ``kestrel deploy secrets sync``.

Sub-PR 1.2 of epic #1050. Exercises argparse wiring and the dispatch
into :mod:`kestrel_sovereign.features.deploy.secrets`. The secrets
module itself is mocked here — its own behaviour is covered in
``tests/unit/test_deploy_secrets.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.cli_deploy import (
    add_deploy_subcommands,
    cmd_deploy,
)
from kestrel_sovereign.features.deploy.secrets import (
    ACTION_CREATED,
    ACTION_DRY_RUN_CREATE,
    ACTION_ERROR,
    ACTION_SKIPPED,
    ACTION_UPDATED,
    SecretSyncResult,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kestrel")
    sub = p.add_subparsers(dest="command")
    add_deploy_subcommands(sub)
    return p


def _make_args(**overrides):
    """Argparse-like namespace with all secrets-relevant defaults."""
    base = {
        "target": None,
        "profile": None,
        "tag": "latest",
        "lines": 100,
        "json": False,
        "secrets_profile": None,
        "env_file": None,
        "dry_run": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def test_argparse_secrets_sync_basic():
    """``kestrel deploy secrets sync`` parses cleanly with all defaults."""
    parser = _build_parser()
    args = parser.parse_args(["deploy", "secrets", "sync"])

    assert args.command == "deploy"
    assert args.target == "secrets"
    assert args.profile == "sync"  # second positional carries the subverb
    assert args.secrets_profile is None
    assert args.env_file is None
    assert args.dry_run is False


def test_argparse_secrets_sync_all_flags():
    """All documented secrets-sync flags parse together."""
    parser = _build_parser()
    args = parser.parse_args([
        "deploy", "secrets", "sync",
        "--profile", "dev",
        "--env-file", "/tmp/test.env",
        "--dry-run",
        "--json",
    ])

    assert args.target == "secrets"
    assert args.profile == "sync"
    assert args.secrets_profile == "dev"
    assert args.env_file == "/tmp/test.env"
    assert args.dry_run is True
    assert args.json is True


# ---------------------------------------------------------------------------
# Dispatcher: ``kestrel deploy secrets`` requires a subverb
# ---------------------------------------------------------------------------

def test_cmd_deploy_secrets_no_subverb_prints_usage(capsys):
    """``kestrel deploy secrets`` (no ``sync``) prints usage and returns 1."""
    rc = cmd_deploy(_make_args(target="secrets"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "Usage: kestrel deploy secrets sync" in captured.err


def test_cmd_deploy_secrets_unknown_subverb_errors(capsys):
    """An unknown subverb (``rotate``, etc.) returns 1 with a clear message."""
    rc = cmd_deploy(_make_args(target="secrets", profile="rotate"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "unknown secrets subverb" in captured.err


# ---------------------------------------------------------------------------
# Dispatcher: configuration prerequisites
# ---------------------------------------------------------------------------

def test_cmd_deploy_secrets_sync_missing_deploy_config(tmp_path, monkeypatch, capsys):
    """No deploy_config.toml → friendly error, return 1."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    rc = cmd_deploy(_make_args(target="secrets", profile="sync"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "deploy_config.toml not found" in captured.err


def test_cmd_deploy_secrets_sync_no_project_id(tmp_path, monkeypatch, capsys):
    """No ``GCP_PROJECT_ID`` env and config has placeholder value → error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    # Write a minimal deploy_config.toml that uses the example placeholder.
    (tmp_path / "deploy_config.toml").write_text(
        '[manager]\n'
        'gcp_project_id = "your-gcp-project-id"\n'
        '\n'
        '[profiles.dev.secrets]\n'
        'OPENAI_API_KEY = "kestrel-openai-key:latest"\n'
    )

    rc = cmd_deploy(_make_args(target="secrets", profile="sync"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "GCP project ID not set" in captured.err


# ---------------------------------------------------------------------------
# Dispatcher: full sync paths (mocked sync_all_secrets)
# ---------------------------------------------------------------------------

def _write_minimal_deploy_config(tmp_path: Path) -> None:
    """Write a deploy_config.toml that satisfies the secrets path's
    prerequisite checks (project_id, [profiles.dev.secrets])."""
    (tmp_path / "deploy_config.toml").write_text(
        '[manager]\n'
        'gcp_project_id = "test-project"\n'
        '\n'
        '[profiles.dev.secrets]\n'
        'OPENAI_API_KEY = "kestrel-openai-key:latest"\n'
        'KESTREL_API_KEY = "kestrel-api-key:latest"\n'
    )


def test_cmd_deploy_secrets_sync_happy_path(tmp_path, monkeypatch, capsys):
    """Happy path: prints per-secret lines + summary, exits 0."""
    monkeypatch.chdir(tmp_path)
    _write_minimal_deploy_config(tmp_path)

    fake_results = [
        SecretSyncResult("kestrel-openai-key", "OPENAI_API_KEY", ACTION_CREATED),
        SecretSyncResult("kestrel-api-key", "KESTREL_API_KEY", ACTION_UPDATED),
    ]
    with patch(
        "kestrel_sovereign.features.deploy.secrets.sync_all_secrets",
        return_value=fake_results,
    ) as mock_sync:
        rc = cmd_deploy(_make_args(
            target="secrets",
            profile="sync",
            secrets_profile="dev",
        ))

    captured = capsys.readouterr()
    assert rc == 0
    # Per-secret lines.
    assert "kestrel-openai-key <- OPENAI_API_KEY" in captured.out
    assert "kestrel-api-key <- KESTREL_API_KEY" in captured.out
    # Summary footer.
    assert "1 created" in captured.out
    assert "1 updated" in captured.out
    assert "0 errors" in captured.out

    # sync_all_secrets called with project_id, profile, dry_run shape.
    call = mock_sync.call_args
    assert call.kwargs.get("profile") == "dev"
    assert call.kwargs.get("dry_run") is False


def test_cmd_deploy_secrets_sync_uses_env_var_project_id(tmp_path, monkeypatch):
    """``GCP_PROJECT_ID`` env var beats the deploy_config.toml value."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GCP_PROJECT_ID", "env-project")
    _write_minimal_deploy_config(tmp_path)  # config says "test-project"

    with patch(
        "kestrel_sovereign.features.deploy.secrets.sync_all_secrets",
        return_value=[],
    ) as mock_sync:
        rc = cmd_deploy(_make_args(target="secrets", profile="sync"))

    assert rc == 0
    # Third positional in the call is project_id.
    call_args = mock_sync.call_args
    project_id_arg = call_args.args[2] if len(call_args.args) >= 3 else call_args.kwargs.get("project_id")
    assert project_id_arg == "env-project"


def test_cmd_deploy_secrets_sync_dry_run_passes_flag(tmp_path, monkeypatch):
    """``--dry-run`` propagates into ``sync_all_secrets(dry_run=True)``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    _write_minimal_deploy_config(tmp_path)

    fake_results = [
        SecretSyncResult(
            "kestrel-openai-key", "OPENAI_API_KEY", ACTION_DRY_RUN_CREATE,
            detail="would create with automatic replication",
        ),
    ]
    with patch(
        "kestrel_sovereign.features.deploy.secrets.sync_all_secrets",
        return_value=fake_results,
    ) as mock_sync:
        rc = cmd_deploy(_make_args(
            target="secrets",
            profile="sync",
            dry_run=True,
        ))

    assert rc == 0
    assert mock_sync.call_args.kwargs.get("dry_run") is True


def test_cmd_deploy_secrets_sync_error_result_returns_1(tmp_path, monkeypatch, capsys):
    """Any ``action == "error"`` result yields exit code 1 even though
    sync_all_secrets returned (didn't raise)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    _write_minimal_deploy_config(tmp_path)

    fake_results = [
        SecretSyncResult("kestrel-openai-key", "OPENAI_API_KEY", ACTION_CREATED),
        SecretSyncResult(
            "kestrel-api-key", "KESTREL_API_KEY", ACTION_ERROR,
            detail="PermissionDenied",
        ),
    ]
    with patch(
        "kestrel_sovereign.features.deploy.secrets.sync_all_secrets",
        return_value=fake_results,
    ):
        rc = cmd_deploy(_make_args(target="secrets", profile="sync"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "1 errors" in captured.out


def test_cmd_deploy_secrets_sync_skipped_does_not_error(tmp_path, monkeypatch, capsys):
    """A 'skipped' result (env var missing) still exits 0 — partial sync
    is the documented behaviour, not a failure."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    _write_minimal_deploy_config(tmp_path)

    fake_results = [
        SecretSyncResult("kestrel-openai-key", "OPENAI_API_KEY", ACTION_CREATED),
        SecretSyncResult(
            "kestrel-api-key", "KESTREL_API_KEY", ACTION_SKIPPED,
            detail="env var KESTREL_API_KEY not set in .env",
        ),
    ]
    with patch(
        "kestrel_sovereign.features.deploy.secrets.sync_all_secrets",
        return_value=fake_results,
    ):
        rc = cmd_deploy(_make_args(target="secrets", profile="sync"))

    captured = capsys.readouterr()
    assert rc == 0
    assert "1 skipped" in captured.out
    # The detail (why it was skipped) should be visible.
    assert "not set in .env" in captured.out


def test_cmd_deploy_secrets_sync_json_output(tmp_path, monkeypatch, capsys):
    """``--json`` emits a structured payload with success + results array."""
    import json as _json

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    _write_minimal_deploy_config(tmp_path)

    fake_results = [
        SecretSyncResult("kestrel-openai-key", "OPENAI_API_KEY", ACTION_CREATED),
    ]
    with patch(
        "kestrel_sovereign.features.deploy.secrets.sync_all_secrets",
        return_value=fake_results,
    ):
        rc = cmd_deploy(_make_args(
            target="secrets",
            profile="sync",
            json=True,
        ))

    captured = capsys.readouterr()
    assert rc == 0
    parsed = _json.loads(captured.out)
    assert parsed["success"] is True
    assert parsed["project_id"] == "test-project"
    assert len(parsed["results"]) == 1
    assert parsed["results"][0]["secret_name"] == "kestrel-openai-key"
    assert parsed["results"][0]["env_var"] == "OPENAI_API_KEY"
    assert parsed["results"][0]["action"] == "created"


def test_cmd_deploy_secrets_sync_missing_env_file(tmp_path, monkeypatch, capsys):
    """``sync_all_secrets`` raises FileNotFoundError → CLI returns 1
    with a friendly message (no traceback)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    _write_minimal_deploy_config(tmp_path)

    with patch(
        "kestrel_sovereign.features.deploy.secrets.sync_all_secrets",
        side_effect=FileNotFoundError(".env file not found: /tmp/nonexistent"),
    ):
        rc = cmd_deploy(_make_args(target="secrets", profile="sync"))

    captured = capsys.readouterr()
    assert rc == 1
    assert ".env file not found" in captured.err


def test_cmd_deploy_secrets_sync_unknown_profile_returns_1(tmp_path, monkeypatch, capsys):
    """``--profile bogus`` (unknown profile name) → KeyError → exit 1."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    _write_minimal_deploy_config(tmp_path)

    with patch(
        "kestrel_sovereign.features.deploy.secrets.sync_all_secrets",
        side_effect=KeyError("profile 'bogus' not found in deploy_config (available: ['dev'])"),
    ):
        rc = cmd_deploy(_make_args(
            target="secrets",
            profile="sync",
            secrets_profile="bogus",
        ))

    captured = capsys.readouterr()
    assert rc == 1
    assert "bogus" in captured.err
