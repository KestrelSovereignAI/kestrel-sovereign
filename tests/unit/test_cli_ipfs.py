"""``kestrel ipfs`` CLI tests — sub-PR 4 of epic #1050 (bash-to-Python
port of ``scripts/ipfs/{build,deploy,pin_agents}.sh``).

Covers:
- argparse wiring under the local subparser and the real ``kestrel``
  parser
- ``GCP_PROJECT_ID`` validation for ``build`` and ``deploy``
- ``build`` invokes ``docker build`` + ``docker push`` with the right
  GCR refs
- ``deploy {create,update,delete,status,ssh}`` shells out to
  ``gcloud`` with the right flags; firewall rules are idempotent
- ``pin`` reaches the IPFS API, snapshots SQLite DBs, posts each
  snapshot, prints a CID per agent
- Bytes-stable: streamed (no ``capture_output=True``) for the long
  gcloud / docker calls
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign import cli_ipfs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kestrel")
    sub = p.add_subparsers(dest="command")
    cli_ipfs.add_ipfs_subcommand(sub)
    return p


def test_argparse_build_defaults():
    parser = _build_parser()
    args = parser.parse_args(["ipfs", "build"])
    assert args.command == "ipfs"
    assert args.ipfs_command == "build"
    assert args.tag == "latest"


def test_argparse_build_tag_override():
    parser = _build_parser()
    args = parser.parse_args(["ipfs", "build", "--tag", "v1.2.3"])
    assert args.tag == "v1.2.3"


def test_argparse_deploy_action():
    parser = _build_parser()
    args = parser.parse_args(["ipfs", "deploy", "create"])
    assert args.ipfs_command == "deploy"
    assert args.action == "create"
    assert args.zone == "us-central1-a"


def test_argparse_deploy_zone_override():
    parser = _build_parser()
    args = parser.parse_args(
        ["ipfs", "deploy", "delete", "--zone", "europe-west1-b", "--yes"]
    )
    assert args.action == "delete"
    assert args.zone == "europe-west1-b"
    assert args.yes is True


def test_argparse_deploy_invalid_action_rejected():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ipfs", "deploy", "frobnicate"])


def test_argparse_pin_defaults():
    parser = _build_parser()
    args = parser.parse_args(["ipfs", "pin"])
    assert args.ipfs_command == "pin"
    assert args.api_url == "http://localhost:5001"
    assert args.manifest is None


def test_argparse_pin_overrides():
    parser = _build_parser()
    args = parser.parse_args(
        ["ipfs", "pin", "--api-url", "http://10.0.0.1:5001",
         "--manifest", "/tmp/agents"]
    )
    assert args.api_url == "http://10.0.0.1:5001"
    assert args.manifest == "/tmp/agents"


def test_kestrel_cli_registers_ipfs():
    from kestrel_sovereign.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["ipfs", "build"])
    assert args.command == "ipfs"
    assert args.ipfs_command == "build"


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def test_cmd_ipfs_no_subverb_prints_usage(capsys):
    rc = cli_ipfs.cmd_ipfs(_Args(ipfs_command=None))
    assert rc == 1
    err = capsys.readouterr().err
    assert "Usage" in err
    assert "build" in err and "deploy" in err and "pin" in err


# ---------------------------------------------------------------------------
# build subverb
# ---------------------------------------------------------------------------

def test_build_missing_gcp_project_errors(monkeypatch, capsys):
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    rc = cli_ipfs.cmd_ipfs(_Args(ipfs_command="build", tag="latest"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "GCP_PROJECT_ID" in err


def test_build_invokes_docker_build_and_push(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    captured: list = []

    def fake_run(cmd, *, cwd=None, env=None, check=False):
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(cli_ipfs, "run_streaming", fake_run)
    rc = cli_ipfs.cmd_ipfs(_Args(ipfs_command="build", tag="v1"))
    assert rc == 0
    # Three calls: build, push :v1, push :latest
    assert len(captured) == 3
    build = captured[0]
    assert build[0:2] == ["docker", "build"]
    assert "--platform" in build and "linux/amd64" in build
    assert "gcr.io/test-proj/kestrel-ipfs-gcs:v1" in build
    assert "gcr.io/test-proj/kestrel-ipfs-gcs:latest" in build

    push_v1 = captured[1]
    push_latest = captured[2]
    assert push_v1 == [
        "docker", "push", "gcr.io/test-proj/kestrel-ipfs-gcs:v1",
    ]
    assert push_latest == [
        "docker", "push", "gcr.io/test-proj/kestrel-ipfs-gcs:latest",
    ]


def test_build_returns_failure_on_docker_build(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    monkeypatch.setattr(
        cli_ipfs, "run_streaming",
        lambda cmd, **kw: 7,
    )
    rc = cli_ipfs.cmd_ipfs(_Args(ipfs_command="build", tag="latest"))
    assert rc == 7


def test_build_returns_failure_on_push(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    n = {"i": 0}

    def fake_run(cmd, **kw):
        n["i"] += 1
        return 0 if n["i"] == 1 else 5  # first push fails

    monkeypatch.setattr(cli_ipfs, "run_streaming", fake_run)
    rc = cli_ipfs.cmd_ipfs(_Args(ipfs_command="build", tag="latest"))
    assert rc == 5


# ---------------------------------------------------------------------------
# deploy subverb
# ---------------------------------------------------------------------------

def test_deploy_missing_gcp_project_errors(monkeypatch, capsys):
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    rc = cli_ipfs.cmd_ipfs(
        _Args(ipfs_command="deploy", action="create",
              zone="us-central1-a", yes=False)
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "GCP_PROJECT_ID" in err


def test_deploy_no_action_prints_usage(monkeypatch, capsys):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    rc = cli_ipfs.cmd_ipfs(
        _Args(ipfs_command="deploy", action=None,
              zone="us-central1-a", yes=False)
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "Usage" in err


def test_deploy_create_invokes_gcloud(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    captured: list = []

    def fake_streaming(cmd, *, cwd=None, env=None, check=False):
        captured.append(list(cmd))
        return 0

    # subprocess.run is called for firewall existence checks + the
    # external-IP lookup. Always return non-zero exit code for the
    # firewall describe so creation is exercised; return the IP for
    # the describe call.
    fake_subprocess = MagicMock()

    def subprocess_run(cmd, *, capture_output=False, text=False,
                       check=False, **kw):
        # Distinguish describe-firewall (returncode != 0 → create
        # path) from describe-instance (returncode == 0 + IP).
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        if "instances" in cmd and "describe" in cmd:
            result.returncode = 0
            result.stdout = "1.2.3.4"
        return result

    fake_subprocess.run.side_effect = subprocess_run

    monkeypatch.setattr(cli_ipfs, "run_streaming", fake_streaming)
    monkeypatch.setattr(cli_ipfs, "subprocess", fake_subprocess,
                        raising=False)

    # Patch the local `import subprocess` inside the module helpers.
    with patch.object(cli_ipfs, "_resolve_service_account",
                      return_value="sa@proj.iam.gserviceaccount.com"):
        with patch("kestrel_sovereign.cli_ipfs.subprocess.run",
                   side_effect=subprocess_run, create=True):
            rc = cli_ipfs.cmd_ipfs(
                _Args(ipfs_command="deploy", action="create",
                      zone="us-central1-a", yes=False)
            )

    assert rc == 0
    # We should see firewall-rules create * 3 + instances
    # create-with-container.
    fw_creates = [c for c in captured if "firewall-rules" in c
                  and "create" in c]
    inst_creates = [c for c in captured if "instances" in c
                    and "create-with-container" in c]
    assert len(fw_creates) == 3, f"saw {fw_creates}"
    assert len(inst_creates) == 1
    create_argv = inst_creates[0]
    assert "kestrel-ipfs" in create_argv  # instance name
    assert any("gcr.io/test-proj/kestrel-ipfs-gcs:latest" in a
               for a in create_argv)
    assert any(a.startswith("--service-account=") for a in create_argv)


def test_deploy_update_invokes_gcloud(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    captured: list = []
    monkeypatch.setattr(
        cli_ipfs, "run_streaming",
        lambda cmd, **kw: captured.append(list(cmd)) or 0,
    )
    rc = cli_ipfs.cmd_ipfs(
        _Args(ipfs_command="deploy", action="update",
              zone="us-central1-a", yes=False)
    )
    assert rc == 0
    assert len(captured) == 1
    argv = captured[0]
    assert argv[0:5] == [
        "gcloud", "compute", "instances", "update-container",
        "kestrel-ipfs",
    ]
    assert any("gcr.io/test-proj/kestrel-ipfs-gcs:latest" in a
               for a in argv)


def test_deploy_status_when_instance_missing_prints_hint(
    monkeypatch, capsys,
):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    monkeypatch.setattr(
        cli_ipfs, "run_streaming",
        lambda cmd, **kw: 0,
    )
    # subprocess.run.describe returns non-zero → "Instance not found".
    fake = MagicMock()
    fake.returncode = 1
    fake.stdout = ""
    with patch("kestrel_sovereign.cli_ipfs.subprocess.run",
               return_value=fake, create=True):
        rc = cli_ipfs.cmd_ipfs(
            _Args(ipfs_command="deploy", action="status",
                  zone="us-central1-a", yes=False)
        )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Instance not found" in out


def test_deploy_delete_with_yes_skips_prompt(monkeypatch, capsys):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    captured: list = []
    monkeypatch.setattr(
        cli_ipfs, "run_streaming",
        lambda cmd, **kw: captured.append(list(cmd)) or 0,
    )
    # ``input`` is not called when --yes is passed; assert via
    # patch that it would raise if called.
    with patch("builtins.input", side_effect=AssertionError("prompt")):
        rc = cli_ipfs.cmd_ipfs(
            _Args(ipfs_command="deploy", action="delete",
                  zone="us-central1-a", yes=True)
        )
    assert rc == 0
    delete_argv = captured[-1]
    assert delete_argv[0:5] == [
        "gcloud", "compute", "instances", "delete", "kestrel-ipfs",
    ]


def test_deploy_delete_user_says_no_returns_zero(monkeypatch, capsys):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    captured: list = []
    monkeypatch.setattr(
        cli_ipfs, "run_streaming",
        lambda cmd, **kw: captured.append(list(cmd)) or 0,
    )
    with patch("builtins.input", return_value="n"):
        rc = cli_ipfs.cmd_ipfs(
            _Args(ipfs_command="deploy", action="delete",
                  zone="us-central1-a", yes=False)
        )
    assert rc == 0
    # No gcloud delete call.
    assert not any("delete" in c for c in captured)


def test_deploy_ssh_invokes_gcloud_ssh(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    captured: list = []
    monkeypatch.setattr(
        cli_ipfs, "run_streaming",
        lambda cmd, **kw: captured.append(list(cmd)) or 0,
    )
    rc = cli_ipfs.cmd_ipfs(
        _Args(ipfs_command="deploy", action="ssh",
              zone="us-central1-a", yes=False)
    )
    assert rc == 0
    argv = captured[0]
    assert argv[0:3] == ["gcloud", "compute", "ssh"]
    assert "kestrel-ipfs" in argv
    assert "--zone=us-central1-a" in argv


# ---------------------------------------------------------------------------
# pin subverb
# ---------------------------------------------------------------------------

def _make_agent_db(root: Path, name: str) -> Path:
    agent_dir = root / name
    agent_dir.mkdir()
    db_path = agent_dir / "kestrel_prime.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
    conn.execute("INSERT INTO t (val) VALUES ('hello')")
    conn.commit()
    conn.close()
    return db_path


def test_pin_unreachable_api_errors(tmp_path, monkeypatch, capsys):
    _make_agent_db(tmp_path, "alice")

    def fail(*a, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(cli_ipfs, "_ipfs_api_get", fail)
    rc = cli_ipfs.cmd_ipfs(
        _Args(ipfs_command="pin", api_url="http://localhost:5001",
              manifest=str(tmp_path))
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "Cannot reach IPFS API" in err


def test_pin_iterates_agents_and_prints_cid(tmp_path, monkeypatch, capsys):
    _make_agent_db(tmp_path, "alice")
    _make_agent_db(tmp_path, "bob")
    # Plus a non-DB agent dir to exercise the SKIP path.
    (tmp_path / "carol").mkdir()

    monkeypatch.setattr(
        cli_ipfs, "_ipfs_api_get",
        lambda url, path, *, timeout=5.0: {"ID": "QmPeer123"},
    )

    seen: list = []

    def fake_add(api_url, file_path, *, name, timeout=120.0):
        seen.append((api_url, name, Path(file_path).stat().st_size))
        return f"QmCID-{name.split('/')[0]}"

    monkeypatch.setattr(cli_ipfs, "_ipfs_add_file", fake_add)

    rc = cli_ipfs.cmd_ipfs(
        _Args(ipfs_command="pin", api_url="http://localhost:5001",
              manifest=str(tmp_path))
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "QmPeer123" in out
    assert "QmCID-alice" in out
    assert "QmCID-bob" in out
    assert "SKIP carol" in out
    # Both agents posted via _ipfs_add_file with multipart name carrying
    # the agent prefix.
    assert sorted(s[1] for s in seen) == [
        "alice/kestrel_prime.db", "bob/kestrel_prime.db",
    ]


def test_pin_handles_per_agent_failure_without_aborting(
    tmp_path, monkeypatch, capsys,
):
    _make_agent_db(tmp_path, "alice")
    _make_agent_db(tmp_path, "bob")
    monkeypatch.setattr(
        cli_ipfs, "_ipfs_api_get",
        lambda url, path, *, timeout=5.0: {"ID": "QmPeer"},
    )

    n = {"i": 0}

    def flaky_add(api_url, file_path, *, name, timeout=120.0):
        n["i"] += 1
        if n["i"] == 1:
            raise OSError("upload truncated")
        return "QmGOOD"

    monkeypatch.setattr(cli_ipfs, "_ipfs_add_file", flaky_add)
    rc = cli_ipfs.cmd_ipfs(
        _Args(ipfs_command="pin", api_url="http://localhost:5001",
              manifest=str(tmp_path))
    )
    assert rc == 0  # individual failures don't fail the whole run
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "QmGOOD" in out


def test_pin_missing_data_dir_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli_ipfs, "_ipfs_api_get",
        lambda url, path, *, timeout=5.0: {"ID": "QmPeer"},
    )
    rc = cli_ipfs.cmd_ipfs(
        _Args(ipfs_command="pin", api_url="http://localhost:5001",
              manifest=str(tmp_path / "nonexistent"))
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "does not exist" in err


def test_snapshot_sqlite_creates_consistent_copy(tmp_path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    conn = sqlite3.connect(str(src))
    conn.execute("CREATE TABLE x (id INT)")
    conn.execute("INSERT INTO x VALUES (1)")
    conn.execute("INSERT INTO x VALUES (2)")
    conn.commit()
    conn.close()
    cli_ipfs._snapshot_sqlite(src, dst)
    assert dst.is_file()
    out = sqlite3.connect(str(dst))
    rows = out.execute("SELECT id FROM x ORDER BY id").fetchall()
    out.close()
    assert rows == [(1,), (2,)]


def test_format_size_human_readable():
    assert cli_ipfs._format_size(0).endswith("B")
    assert "KB" in cli_ipfs._format_size(2048)
    assert "MB" in cli_ipfs._format_size(2 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Streaming subprocess — no capture
# ---------------------------------------------------------------------------

def test_module_uses_shared_streaming_helper():
    from kestrel_sovereign import _subprocess_helpers as sh
    assert cli_ipfs.run_streaming is sh.run_streaming
