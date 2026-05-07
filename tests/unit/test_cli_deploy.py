"""
``kestrel deploy`` CLI tests — sub-PR 1.1 of epic #1050 (bash-to-Python port).

Covers the argparse glue and the dispatcher in
:mod:`kestrel_sovereign.cli_deploy`. The manager itself is mocked —
provider behaviour is exercised in ``tests/unit/test_deploy_*.py``.
"""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.cli_deploy import (
    add_deploy_subcommands,
    cmd_deploy,
)
from kestrel_sovereign.features.deploy.models import DeployManagerError


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kestrel")
    sub = p.add_subparsers(dest="command")
    add_deploy_subcommands(sub)
    return p


def test_argparse_help_prints_subcommands():
    """``kestrel deploy`` is registered with the parent parser and
    accepts the documented positional/named arguments without raising
    SystemExit on a known shape."""
    parser = _build_parser()

    # Bare ``kestrel deploy`` — nargs='?' so this should not raise.
    args = parser.parse_args(["deploy"])
    assert args.command == "deploy"
    assert args.target is None
    assert args.tag == "latest"
    assert args.lines == 100

    # ``kestrel deploy dev`` — profile-as-target.
    args = parser.parse_args(["deploy", "dev"])
    assert args.target == "dev"
    assert args.profile is None

    # ``kestrel deploy --tag v1.2.3 dev``
    args = parser.parse_args(["deploy", "--tag", "v1.2.3", "dev"])
    assert args.target == "dev"
    assert args.tag == "v1.2.3"

    # ``kestrel deploy teardown dev``
    args = parser.parse_args(["deploy", "teardown", "dev"])
    assert args.target == "teardown"
    assert args.profile == "dev"

    # ``kestrel deploy logs dev --lines 500``
    args = parser.parse_args(["deploy", "logs", "dev", "--lines", "500"])
    assert args.target == "logs"
    assert args.profile == "dev"
    assert args.lines == 500


# ---------------------------------------------------------------------------
# Build_parser integration — make sure deploy is wired into the real CLI
# ---------------------------------------------------------------------------

def test_kestrel_cli_registers_deploy_subcommand():
    """The full ``kestrel`` parser (built via ``cli.build_parser``)
    accepts ``kestrel deploy <profile>``. Guards against a future
    refactor accidentally dropping the wiring in ``cli.py``."""
    from kestrel_sovereign.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["deploy", "dev"])
    assert args.command == "deploy"
    assert args.target == "dev"


# ---------------------------------------------------------------------------
# Dispatcher: handler routing
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    """Build an argparse.Namespace mirroring add_deploy_subcommands' shape."""
    base = {
        "target": None,
        "profile": None,
        "tag": "latest",
        "lines": 100,
        "json": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cmd_deploy_no_target_prints_usage(capsys):
    """``kestrel deploy`` (no args) prints usage and returns 1."""
    rc = cmd_deploy(_make_args())
    captured = capsys.readouterr()
    assert rc == 1
    assert "Usage: kestrel deploy" in captured.err


def test_cmd_deploy_unknown_profile(capsys):
    """A profile name that doesn't exist (manager raises
    DeployManagerError) prints the error and returns 1, without a
    traceback."""
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.deploy_profile = AsyncMock(
            side_effect=DeployManagerError(
                "Unknown profile 'bogus'. Available: dev, prod"
            )
        )
        mock_mgr_cls.return_value = mock_instance

        rc = cmd_deploy(_make_args(target="bogus"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "Unknown profile" in captured.err


def test_cmd_deploy_init_failure_returns_1(capsys):
    """If DeployManager() construction itself raises (no profiles, no
    GCP_PROJECT_ID, etc.), we print a friendly error and return 1."""
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_mgr_cls.side_effect = DeployManagerError("GCP_PROJECT_ID is required")

        rc = cmd_deploy(_make_args(target="dev"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "GCP_PROJECT_ID is required" in captured.err


def test_cmd_deploy_success(capsys):
    """A successful deploy returns 0 and prints something."""
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.gcp_project_id = "real-project"
        mock_instance.deploy_profile = AsyncMock(
            return_value={
                "success": True,
                "action": "deploy",
                "session": {
                    "service_name": "kestrel-dev",
                    "service_url": "https://kestrel-dev-xyz.run.app",
                    "status": "active",
                },
            }
        )
        mock_mgr_cls.return_value = mock_instance

        rc = cmd_deploy(_make_args(target="dev", tag="v1.2.3"))

    captured = capsys.readouterr()
    assert rc == 0
    # Verify it called deploy_profile with our tag.
    mock_instance.deploy_profile.assert_awaited_once_with("dev", "v1.2.3")
    # And printed something user-visible.
    assert "kestrel-dev" in captured.out


def test_cmd_deploy_success_failed_result_returns_1():
    """A deploy that returns ``success=False`` (e.g. service already
    deployed) yields exit code 1 even though the manager call itself
    succeeded."""
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.gcp_project_id = "real-project"
        mock_instance.deploy_profile = AsyncMock(
            return_value={
                "success": False,
                "error": "Service kestrel-dev already deployed",
            }
        )
        mock_mgr_cls.return_value = mock_instance

        rc = cmd_deploy(_make_args(target="dev"))

    assert rc == 1


def test_cmd_deploy_rejects_placeholder_project_id(capsys):
    """Codex review on the final epic→main PR: the deploy CLI must
    reject ``manager.gcp_project_id == "your-gcp-project-id"`` (the
    deploy_config.toml example placeholder) so deploys fail fast with
    an actionable error rather than building Cloud Run image refs
    against ``gcr.io/your-gcp-project-id/`` and crashing inside the SDK.
    """
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.gcp_project_id = "your-gcp-project-id"
        mock_instance.deploy_profile = AsyncMock()
        mock_mgr_cls.return_value = mock_instance

        rc = cmd_deploy(_make_args(target="dev"))

    assert rc == 1
    err = capsys.readouterr().err
    assert "GCP project ID is not configured" in err
    # And we did NOT attempt the deploy.
    mock_instance.deploy_profile.assert_not_awaited()


def test_cmd_deploy_status(capsys):
    """``kestrel deploy status`` queries providers (not in-memory sessions)
    so a fresh CLI invocation reflects what is actually deployed."""
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.list_all_deployments = AsyncMock(
            return_value={
                "success": True,
                "action": "list",
                "count": 1,
                "deployments": [
                    {"name": "kestrel-dev", "status": "active", "url": "https://kestrel-dev.run.app"},
                ],
            }
        )
        mock_mgr_cls.return_value = mock_instance

        rc = cmd_deploy(_make_args(target="status"))

    captured = capsys.readouterr()
    assert rc == 0
    assert "kestrel-dev" in captured.out
    assert "active_deployments: 1" in captured.out


def test_cmd_deploy_status_no_deployments(capsys):
    """``kestrel deploy status`` with no provider-side deployments prints
    a friendly empty-state message and returns 0."""
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.list_all_deployments = AsyncMock(
            return_value={"success": True, "action": "list", "count": 0, "deployments": []}
        )
        mock_mgr_cls.return_value = mock_instance

        rc = cmd_deploy(_make_args(target="status"))

    captured = capsys.readouterr()
    assert rc == 0
    assert "No active deployments" in captured.out


def test_cmd_deploy_teardown_requires_profile(capsys):
    """``kestrel deploy teardown`` (no profile) errors out."""
    rc = cmd_deploy(_make_args(target="teardown"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "requires a profile name" in captured.err


def test_cmd_deploy_teardown_success():
    """``kestrel deploy teardown dev`` happy path."""
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.teardown_profile = AsyncMock(
            return_value={
                "success": True,
                "action": "teardown",
                "service": "kestrel-dev",
                "result": {},
            }
        )
        mock_mgr_cls.return_value = mock_instance

        rc = cmd_deploy(_make_args(target="teardown", profile="dev"))

    assert rc == 0
    mock_instance.teardown_profile.assert_awaited_once_with("dev")


def test_cmd_deploy_logs_prints_log_payload(capsys):
    """``kestrel deploy logs <profile>`` prints the log string after the
    summary header."""
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.get_profile_logs = AsyncMock(
            return_value={
                "success": True,
                "action": "logs",
                "service": "kestrel-dev",
                "lines": 100,
                "logs": "INFO: started\nINFO: serving on :8080\n",
            }
        )
        mock_mgr_cls.return_value = mock_instance

        rc = cmd_deploy(_make_args(target="logs", profile="dev"))

    captured = capsys.readouterr()
    assert rc == 0
    assert "kestrel-dev" in captured.out
    assert "INFO: serving on :8080" in captured.out


def test_cmd_deploy_list(capsys):
    """``kestrel deploy list`` mocks ``list_all_deployments`` and prints
    the table."""
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.list_all_deployments = AsyncMock(
            return_value={
                "success": True,
                "action": "list",
                "count": 2,
                "deployments": [
                    {
                        "service": "kestrel-dev",
                        "provider": "cloud_run",
                        "status": "active",
                        "url": "https://kestrel-dev-xyz.run.app",
                    },
                    {
                        "service": "kestrel-prod",
                        "provider": "cloud_run",
                        "status": "active",
                        "url": "https://kestrel-prod-xyz.run.app",
                    },
                ],
            }
        )
        mock_mgr_cls.return_value = mock_instance

        rc = cmd_deploy(_make_args(target="list"))

    captured = capsys.readouterr()
    assert rc == 0
    # Should print something — at minimum, the service names.
    assert "kestrel-dev" in captured.out
    assert "kestrel-prod" in captured.out


def test_cmd_deploy_list_empty(capsys):
    """``kestrel deploy list`` with no deployments prints empty-state."""
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.list_all_deployments = AsyncMock(
            return_value={
                "success": True,
                "action": "list",
                "count": 0,
                "deployments": [],
            }
        )
        mock_mgr_cls.return_value = mock_instance

        rc = cmd_deploy(_make_args(target="list"))

    captured = capsys.readouterr()
    assert rc == 0
    assert "No deployments found" in captured.out


def test_cmd_deploy_health_success():
    """``kestrel deploy health <profile>`` happy path."""
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.health_check_profile = AsyncMock(
            return_value={
                "success": True,
                "action": "health",
                "service": "kestrel-dev",
                "url": "https://kestrel-dev-xyz.run.app",
                "health": {"healthy": True},
            }
        )
        mock_mgr_cls.return_value = mock_instance

        rc = cmd_deploy(_make_args(target="health", profile="dev"))

    assert rc == 0
    mock_instance.health_check_profile.assert_awaited_once_with("dev")


def test_cmd_deploy_json_output(capsys):
    """``--json`` prints the raw result dict as JSON."""
    with patch("kestrel_sovereign.cli_deploy.DeployManager") as mock_mgr_cls:
        mock_instance = MagicMock()
        mock_instance.deploy_profile = AsyncMock(
            return_value={
                "success": True,
                "action": "deploy",
                "session": {"service_name": "kestrel-dev"},
            }
        )
        mock_mgr_cls.return_value = mock_instance

        rc = cmd_deploy(_make_args(target="dev", json=True))

    captured = capsys.readouterr()
    assert rc == 0
    # JSON-shaped output starts with {
    assert captured.out.lstrip().startswith("{")
    assert '"success": true' in captured.out
