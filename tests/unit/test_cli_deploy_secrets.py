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


@pytest.fixture
def fake_project_root(tmp_path, monkeypatch):
    """Make ``_get_project_dir()`` resolve to ``tmp_path`` so the tests
    can plant ``deploy_config.toml`` / ``.env`` somewhere harmless.

    The CLI helper resolves these files relative to the project root
    (cli._get_project_dir) — caller-CWD lookup was the regression codex
    flagged on PR #1057. Tests must mirror that resolution path.
    """
    monkeypatch.setattr(
        "kestrel_sovereign.cli._get_project_dir", lambda: tmp_path
    )
    return tmp_path


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

def test_cmd_deploy_secrets_sync_missing_deploy_config(fake_project_root, monkeypatch, capsys):
    """No deploy_config.toml → friendly error, return 1."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    rc = cmd_deploy(_make_args(target="secrets", profile="sync"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "deploy_config.toml not found" in captured.err


def test_cmd_deploy_secrets_sync_no_project_id(fake_project_root, monkeypatch, capsys):
    """No ``GCP_PROJECT_ID`` env and config has placeholder value → error."""
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    # Write a minimal deploy_config.toml that uses the example placeholder.
    (fake_project_root / "deploy_config.toml").write_text(
        '[manager]\n'
        'gcp_project_id = "your-gcp-project-id"\n'
        '\n'
        '[profiles.dev.secrets]\n'
        'OPENAI_API_KEY = "kestrel-openai-key:latest"\n'
    )

    rc = cmd_deploy(_make_args(target="secrets", profile="sync"))

    captured = capsys.readouterr()
    assert rc == 1
    # The all-profile scan path emits a per-profile diagnostic when
    # ``[manager].gcp_project_id`` is the placeholder and a profile has
    # no override of its own.
    assert "no GCP project ID" in captured.err
    assert "dev" in captured.err


# ---------------------------------------------------------------------------
# Dispatcher: full sync paths (mocked sync_all_secrets)
# ---------------------------------------------------------------------------

def _write_minimal_deploy_config(project_root: Path) -> None:
    """Write a deploy_config.toml that satisfies the secrets path's
    prerequisite checks (project_id, [profiles.dev.secrets])."""
    (project_root / "deploy_config.toml").write_text(
        '[manager]\n'
        'gcp_project_id = "test-project"\n'
        '\n'
        '[profiles.dev.secrets]\n'
        'OPENAI_API_KEY = "kestrel-openai-key:latest"\n'
        'KESTREL_API_KEY = "kestrel-api-key:latest"\n'
    )


def test_cmd_deploy_secrets_sync_happy_path(fake_project_root, monkeypatch, capsys):
    """Happy path: prints per-secret lines + summary, exits 0."""
    _write_minimal_deploy_config(fake_project_root)

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


def test_cmd_deploy_secrets_sync_uses_env_var_project_id(fake_project_root, monkeypatch):
    """``GCP_PROJECT_ID`` env var beats the deploy_config.toml value."""
    monkeypatch.setenv("GCP_PROJECT_ID", "env-project")
    _write_minimal_deploy_config(fake_project_root)  # config says "test-project"

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


def test_cmd_deploy_secrets_sync_uses_profile_level_project_id(
    fake_project_root, monkeypatch
):
    """Codex review on PR #1057: a profile-specific ``gcp_project_id``
    overrides the manager's value when ``--profile`` is given.
    DeployManagerCore._load_profiles supports this; the CLI must mirror
    it or sync to the wrong project / fail when manager is a placeholder.
    """
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    (fake_project_root / "deploy_config.toml").write_text(
        '[manager]\n'
        'gcp_project_id = "your-gcp-project-id"\n'  # placeholder
        '\n'
        '[profiles.dev]\n'
        'provider = "cloudrun"\n'
        'gcp_project_id = "dev-real-project"\n'
        '\n'
        '[profiles.dev.secrets]\n'
        'OPENAI_API_KEY = "kestrel-openai-key:latest"\n'
    )

    with patch(
        "kestrel_sovereign.features.deploy.secrets.sync_all_secrets",
        return_value=[],
    ) as mock_sync:
        rc = cmd_deploy(_make_args(
            target="secrets", profile="sync", secrets_profile="dev",
        ))

    assert rc == 0
    project_id_arg = mock_sync.call_args.args[2] if len(
        mock_sync.call_args.args
    ) >= 3 else mock_sync.call_args.kwargs.get("project_id")
    assert project_id_arg == "dev-real-project"


def test_cmd_deploy_secrets_sync_one_profile_unconfigured_in_default_scan_errors(
    fake_project_root, monkeypatch, capsys
):
    """Codex review on PR #1057 v7: in the default scan, if any cloudrun
    profile lacks a real ``gcp_project_id`` (and the manager value is
    a placeholder), refuse to pick the configured profile's project as
    a winner — the unconfigured profile's secrets would otherwise be
    silently routed to the wrong project."""
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    (fake_project_root / "deploy_config.toml").write_text(
        '[manager]\n'
        'gcp_project_id = "your-gcp-project-id"\n'  # placeholder
        '\n'
        '[profiles.dev]\n'
        'provider = "cloudrun"\n'
        'gcp_project_id = "dev-real-project"\n'
        '[profiles.dev.secrets]\n'
        'KEY1 = "kestrel-key-a:latest"\n'
        '\n'
        '[profiles.prod]\n'
        'provider = "cloudrun"\n'
        # No gcp_project_id override — would fall back to the placeholder.
        '[profiles.prod.secrets]\n'
        'KEY2 = "kestrel-key-b:latest"\n'
    )

    rc = cmd_deploy(_make_args(target="secrets", profile="sync"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "no GCP project ID" in captured.err
    assert "prod" in captured.err


def test_cmd_deploy_secrets_sync_disagreeing_profile_projects_errors(
    fake_project_root, monkeypatch, capsys
):
    """Default scan with two cloudrun profiles targeting different GCP
    projects refuses to pick a winner — directs the operator at
    ``--profile``."""
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    (fake_project_root / "deploy_config.toml").write_text(
        '[manager]\n'
        'gcp_project_id = "alpha-project"\n'
        '\n'
        '[profiles.dev]\n'
        'provider = "cloudrun"\n'
        'gcp_project_id = "alpha-project"\n'
        '[profiles.dev.secrets]\n'
        'KEY1 = "kestrel-key-a:latest"\n'
        '\n'
        '[profiles.prod]\n'
        'provider = "cloudrun"\n'
        'gcp_project_id = "beta-project"\n'
        '[profiles.prod.secrets]\n'
        'KEY2 = "kestrel-key-b:latest"\n'
    )

    rc = cmd_deploy(_make_args(target="secrets", profile="sync"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "different GCP projects" in captured.err
    assert "--profile" in captured.err


def test_cmd_deploy_secrets_sync_dry_run_passes_flag(fake_project_root, monkeypatch):
    """``--dry-run`` propagates into ``sync_all_secrets(dry_run=True)``."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    _write_minimal_deploy_config(fake_project_root)

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


def test_cmd_deploy_secrets_sync_error_result_returns_1(fake_project_root, monkeypatch, capsys):
    """Any ``action == "error"`` result yields exit code 1 even though
    sync_all_secrets returned (didn't raise)."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    _write_minimal_deploy_config(fake_project_root)

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


def test_cmd_deploy_secrets_sync_skipped_does_not_error(fake_project_root, monkeypatch, capsys):
    """A 'skipped' result (env var missing) still exits 0 — partial sync
    is the documented behaviour, not a failure."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    _write_minimal_deploy_config(fake_project_root)

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


def test_cmd_deploy_secrets_sync_json_output(fake_project_root, monkeypatch, capsys):
    """``--json`` emits a structured payload with success + results array."""
    import json as _json

    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    _write_minimal_deploy_config(fake_project_root)

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


def test_cmd_deploy_secrets_sync_missing_env_file(fake_project_root, monkeypatch, capsys):
    """``sync_all_secrets`` raises FileNotFoundError → CLI returns 1
    with a friendly message (no traceback)."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    _write_minimal_deploy_config(fake_project_root)

    with patch(
        "kestrel_sovereign.features.deploy.secrets.sync_all_secrets",
        side_effect=FileNotFoundError(".env file not found: /tmp/nonexistent"),
    ):
        rc = cmd_deploy(_make_args(target="secrets", profile="sync"))

    captured = capsys.readouterr()
    assert rc == 1
    assert ".env file not found" in captured.err


def test_cmd_deploy_secrets_sync_unknown_profile_returns_1(fake_project_root, monkeypatch, capsys):
    """``--profile bogus`` (unknown profile name) → KeyError → exit 1."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    _write_minimal_deploy_config(fake_project_root)

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
