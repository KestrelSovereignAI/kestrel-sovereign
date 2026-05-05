"""Unit tests for the ``kestrel constitution reanchor`` CLI surface.

These tests mock out the async reanchor helper — they assert on the
*CLI behaviour* (argument parsing, exit codes, messaging, refusal
gates). The real five-location DB rewrite is exercised end-to-end in
``tests/integration/test_constitution_reanchor_e2e.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import toml

from kestrel_sovereign.cli import build_parser, cmd_constitution
from kestrel_sovereign.setup.constitution_reanchor import ReanchorResult


@pytest.fixture
def reanchor_env(tmp_path):
    """Project tree with one multi_agent agent + a stub kestrel_prime.db."""
    agent_dir = tmp_path / "agent_data" / "Test"
    agent_dir.mkdir(parents=True)
    (agent_dir / "kestrel_prime.db").write_bytes(b"stub")

    multi_agent = {
        "host": {"port": 8888, "bind": "0.0.0.0"},
        "agents": {
            "Test": {
                "data_dir": "agent_data/Test",
                "port": 8801,
                "autostart": True,
            }
        },
    }
    (tmp_path / "multi_agent.toml").write_text(toml.dumps(multi_agent))
    return tmp_path


def _parse(argv: list[str]):
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_reanchor_requires_agent_name():
    """argparse must reject a call with no --agent-name."""
    with pytest.raises(SystemExit):
        _parse(["constitution", "reanchor"])


def test_reanchor_parses_force_flag():
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    assert args.force is True
    assert args.agent_name == "Test"


def test_reanchor_force_default_false():
    args = _parse(["constitution", "reanchor", "--agent-name", "Test"])
    assert args.force is False


def test_reanchor_accepts_constitution_path_override():
    args = _parse([
        "constitution", "reanchor", "--agent-name", "Test",
        "--constitution-path", "/tmp/custom.md",
    ])
    assert args.constitution_path == "/tmp/custom.md"


# ---------------------------------------------------------------------------
# Refusal gates (must NOT touch the helper)
# ---------------------------------------------------------------------------

def test_reanchor_rejects_unknown_agent(reanchor_env, capsys):
    args = _parse(["constitution", "reanchor", "--agent-name", "NotInMultiAgent"])
    with patch(
        "kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env,
    ), patch(
        "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution"
    ) as mock_reanchor:
        rc = cmd_constitution(args)
    assert rc == 2
    assert "not in multi_agent" in capsys.readouterr().err.lower()
    mock_reanchor.assert_not_called()


def test_reanchor_refuses_when_agent_appears_running(reanchor_env, capsys):
    """SQLite WAL locking would corrupt the DB if the agent is running.
    The CLI must refuse before invoking the helper."""
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    with patch(
        "kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env,
    ), patch(
        "kestrel_sovereign.cli._agent_appears_running", return_value=True,
    ), patch(
        "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution"
    ) as mock_reanchor:
        rc = cmd_constitution(args)
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "appears to be running" in err
    assert "kestrel stop test" in err
    mock_reanchor.assert_not_called()


# ---------------------------------------------------------------------------
# Result dispatch
# ---------------------------------------------------------------------------

def _stubbed_helper(result: ReanchorResult):
    """Build an async stub that returns the given result."""
    async def _stub(**_kwargs):
        return result

    return _stub


def test_reanchor_unchanged_returns_zero(reanchor_env, capsys):
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="a" * 64,
        new_hash="a" * 64,
        backup_path=None,
        unchanged=True,
    )
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        rc = cmd_constitution(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "already anchored" in out


def test_reanchor_drift_unforced_returns_one(reanchor_env, capsys):
    """Drift detected, --force absent → CLI prints diagnosis and exits 1."""
    args = _parse(["constitution", "reanchor", "--agent-name", "Test"])
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="a" * 64,
        new_hash="b" * 64,
        backup_path=None,
        drift_unforced=True,
    )
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        rc = cmd_constitution(args)
    assert rc == 1
    out = capsys.readouterr().out
    assert "drift detected" in out.lower()
    assert "--force" in out  # Tells the user how to actually do it
    assert "backup" in out.lower()  # And that a backup will happen


def test_reanchor_success_prints_old_new_and_backup(reanchor_env, capsys):
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    backup_path = reanchor_env / "agent_data" / "Test" / "kestrel_prime.db.backup-20260504-120000"
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="aaaaaaaaaaaaaaaa" * 4,
        new_hash="bbbbbbbbbbbbbbbb" * 4,
        backup_path=backup_path,
        reanchored=True,
    )
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        rc = cmd_constitution(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "reanchored" in out.lower()
    assert "aaaaaaaaaaaa" in out  # truncated old hash
    assert "bbbbbbbbbbbb" in out  # truncated new hash
    assert str(backup_path) in out  # full backup path visible


def test_reanchor_helper_error_propagates(reanchor_env, capsys):
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash=None,
        new_hash=None,
        backup_path=None,
        error="Cannot read canonical constitution at /fake/canonical.md: [Errno 2]",
    )
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        rc = cmd_constitution(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "Cannot read canonical constitution" in err


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------

def test_constitution_with_no_subcommand_prints_usage(reanchor_env, capsys):
    args = _parse(["constitution"])
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env):
        rc = cmd_constitution(args)
    assert rc == 1
    assert "Usage" in capsys.readouterr().out
