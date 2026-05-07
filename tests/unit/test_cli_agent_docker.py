"""``kestrel agent docker`` CLI tests — sub-PR 3.2 of epic #1050
(bash-to-Python port of ``scripts/sovereign-agent.sh``).

Covers:
- argparse wiring under the local subparser and the real ``kestrel``
  parser
- Missing ``KESTREL_DATA_KEY`` → exit code 1 with the
  ``secrets.token_urlsafe(32)`` hint
- ``~`` expansion + ``mkdir -p`` on the data dir
- ``docker run`` argv shape per subverb (create / chat / retire),
  including ``-it`` only for chat and the volume-mount path
- Image build is invoked when ``docker image inspect`` fails
- ``retire`` prompts for confirmation; ``--yes`` skips it
- Missing ``kestrel_prime.db`` errors cleanly for chat and retire
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign import cli_agent_docker


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kestrel")
    sub = p.add_subparsers(dest="command")
    cli_agent_docker.add_agent_docker_subcommand(sub)
    return p


def test_argparse_create_minimal():
    parser = _build_parser()
    args = parser.parse_args(
        ["agent", "docker", "create", "Emma", "~/emma_data"]
    )
    assert args.command == "agent"
    assert args.agent_command == "docker"
    assert args.agent_docker_command == "create"
    assert args.name == "Emma"
    assert args.data_dir == "~/emma_data"


def test_argparse_chat():
    parser = _build_parser()
    args = parser.parse_args(["agent", "docker", "chat", "~/emma_data"])
    assert args.agent_docker_command == "chat"
    assert args.data_dir == "~/emma_data"


def test_argparse_retire_with_yes():
    parser = _build_parser()
    args = parser.parse_args(
        ["agent", "docker", "retire", "~/emma_data", "--yes"]
    )
    assert args.agent_docker_command == "retire"
    assert args.yes is True


def test_kestrel_cli_registers_agent_docker():
    from kestrel_sovereign.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["agent", "docker", "create", "Emma", "/tmp/x"]
    )
    assert args.command == "agent"
    assert args.agent_command == "docker"
    assert args.agent_docker_command == "create"


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_cmd_agent_no_subverb_prints_usage(capsys):
    rc = cli_agent_docker.cmd_agent(
        _Args(agent_command=None)
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "Usage" in err


def test_cmd_agent_docker_no_subverb_prints_usage(capsys):
    rc = cli_agent_docker.cmd_agent(
        _Args(agent_command="docker", agent_docker_command=None)
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "create" in err and "chat" in err and "retire" in err


# ---------------------------------------------------------------------------
# Missing KESTREL_DATA_KEY
# ---------------------------------------------------------------------------

def test_create_without_data_key_errors_with_hint(monkeypatch, capsys):
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    args = _Args(
        agent_command="docker",
        agent_docker_command="create",
        name="Emma",
        data_dir="/tmp/test_no_key",
    )
    rc = cli_agent_docker.cmd_agent(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "KESTREL_DATA_KEY" in err
    assert "secrets.token_urlsafe(32)" in err


def test_chat_without_data_key_errors(monkeypatch, capsys):
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    args = _Args(
        agent_command="docker",
        agent_docker_command="chat",
        data_dir="/tmp/test_no_key_chat",
    )
    rc = cli_agent_docker.cmd_agent(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "KESTREL_DATA_KEY" in err


def test_retire_without_data_key_errors(monkeypatch, capsys):
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    args = _Args(
        agent_command="docker",
        agent_docker_command="retire",
        data_dir="/tmp/test_no_key_retire",
        yes=True,
    )
    rc = cli_agent_docker.cmd_agent(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "KESTREL_DATA_KEY" in err


# ---------------------------------------------------------------------------
# ~ expansion
# ---------------------------------------------------------------------------

def test_resolve_data_dir_expands_tilde_and_creates_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    out = cli_agent_docker._resolve_data_dir("~/agent")
    assert out == tmp_path / "agent"
    assert out.is_dir()


# ---------------------------------------------------------------------------
# docker run argv shape — create / chat / retire
# ---------------------------------------------------------------------------

def _patch_image_and_runs(monkeypatch):
    """Replace _ensure_image with a no-op success, capture every
    run_streaming call argv into a returned list.
    """
    monkeypatch.setattr(
        cli_agent_docker, "_ensure_image", lambda repo: 0,
    )
    captured: list = []

    def fake_run(cmd, *, cwd=None, env=None, check=False):
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(cli_agent_docker, "run_streaming", fake_run)
    return captured


def test_create_invokes_docker_run_with_inception(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-key")
    captured = _patch_image_and_runs(monkeypatch)

    data_dir = tmp_path / "emma"
    args = _Args(
        agent_command="docker",
        agent_docker_command="create",
        name="Emma",
        data_dir=str(data_dir),
    )
    rc = cli_agent_docker.cmd_agent(args)
    assert rc == 0
    assert len(captured) == 1
    argv = captured[0]
    assert argv[:3] == ["docker", "run", "--rm"]
    # Not interactive on create.
    assert "-it" not in argv
    # KESTREL_DATA_KEY env mount.
    assert "-e" in argv and any(
        a == "KESTREL_DATA_KEY=test-key" for a in argv
    )
    # Volume mount uses POSIX separator.
    assert any(
        a == f"{data_dir.as_posix()}:/data" for a in argv
    )
    # Image + container argv.
    assert "kestrel-sovereign" in argv
    tail = argv[argv.index("kestrel-sovereign") + 1:]
    assert tail == [
        "inception_service.py", "--name", "Emma", "--output", "/data",
    ]


def test_chat_invokes_docker_run_with_it(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-key")
    data_dir = tmp_path / "emma"
    data_dir.mkdir()
    (data_dir / "kestrel_prime.db").write_text("stub")

    captured = _patch_image_and_runs(monkeypatch)
    args = _Args(
        agent_command="docker",
        agent_docker_command="chat",
        data_dir=str(data_dir),
    )
    rc = cli_agent_docker.cmd_agent(args)
    assert rc == 0
    argv = captured[0]
    assert argv[:3] == ["docker", "run", "--rm"]
    # -it ONLY for chat.
    assert "-it" in argv
    tail = argv[argv.index("kestrel-sovereign") + 1:]
    assert tail == ["main.py", "/data/kestrel_prime.db"]


def test_chat_missing_db_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-key")
    data_dir = tmp_path / "no_db"
    monkeypatch.setattr(
        cli_agent_docker, "_ensure_image", lambda repo: 0,
    )
    # data_dir doesn't exist yet — _resolve_data_dir creates it but
    # without a kestrel_prime.db inside.
    args = _Args(
        agent_command="docker",
        agent_docker_command="chat",
        data_dir=str(data_dir),
    )
    rc = cli_agent_docker.cmd_agent(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "no agent found" in err


def test_retire_invokes_docker_run_with_retirement(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-key")
    data_dir = tmp_path / "emma"
    data_dir.mkdir()
    (data_dir / "kestrel_prime.db").write_text("stub")

    captured = _patch_image_and_runs(monkeypatch)
    args = _Args(
        agent_command="docker",
        agent_docker_command="retire",
        data_dir=str(data_dir),
        yes=True,
    )
    rc = cli_agent_docker.cmd_agent(args)
    assert rc == 0
    argv = captured[0]
    # Not interactive on retire.
    assert "-it" not in argv
    tail = argv[argv.index("kestrel-sovereign") + 1:]
    assert tail == ["retirement_service.py", "/data/kestrel_prime.db"]


def test_retire_without_yes_prompts_for_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-key")
    data_dir = tmp_path / "emma"
    data_dir.mkdir()
    (data_dir / "kestrel_prime.db").write_text("stub")

    captured = _patch_image_and_runs(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")

    args = _Args(
        agent_command="docker",
        agent_docker_command="retire",
        data_dir=str(data_dir),
        yes=False,
    )
    rc = cli_agent_docker.cmd_agent(args)
    # User declined — exit code 0, no docker run.
    assert rc == 0
    assert not captured


def test_retire_yes_confirmation_proceeds(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-key")
    data_dir = tmp_path / "emma"
    data_dir.mkdir()
    (data_dir / "kestrel_prime.db").write_text("stub")

    captured = _patch_image_and_runs(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")

    args = _Args(
        agent_command="docker",
        agent_docker_command="retire",
        data_dir=str(data_dir),
        yes=False,
    )
    rc = cli_agent_docker.cmd_agent(args)
    assert rc == 0
    assert captured  # docker run did fire


# ---------------------------------------------------------------------------
# _ensure_image — build is invoked when image absent
# ---------------------------------------------------------------------------

def test_ensure_image_skips_build_when_image_exists(monkeypatch):
    calls: list = []

    def fake_run(cmd, *, cwd=None, env=None, check=False):
        calls.append(list(cmd))
        # First call: docker image inspect → 0 (image exists).
        return 0

    monkeypatch.setattr(cli_agent_docker, "run_streaming", fake_run)
    rc = cli_agent_docker._ensure_image(Path("/repo"))
    assert rc == 0
    # Only the inspect call ran; no build.
    assert len(calls) == 1
    assert calls[0][:3] == ["docker", "image", "inspect"]


def test_ensure_image_builds_when_missing(monkeypatch):
    calls: list = []

    def fake_run(cmd, *, cwd=None, env=None, check=False):
        calls.append(list(cmd))
        # First call (inspect): returns 1 (missing). Second (build): 0.
        if "inspect" in cmd:
            return 1
        return 0

    monkeypatch.setattr(cli_agent_docker, "run_streaming", fake_run)
    rc = cli_agent_docker._ensure_image(Path("/repo"))
    assert rc == 0
    # inspect + build invocations.
    assert len(calls) == 2
    assert calls[1][:2] == ["docker", "build"]
    # Dockerfile path matches the predecessor.
    assert "docker/Dockerfile.sovereign" in " ".join(calls[1])


def test_kestrel_agent_docker_build_subverb_force_rebuilds(monkeypatch):
    """Codex review on PR #1071: ``kestrel agent docker build [--no-cache]``
    must reach ``_ensure_image(force_rebuild=True)``. Without this
    surface, operators with stale images had no way to refresh after
    Dockerfile / kestrel_sovereign/ changes short of running
    ``docker rmi`` first.
    """
    captured: list = []

    def fake_ensure(repo, *, force_rebuild=False):
        captured.append({"repo": repo, "force_rebuild": force_rebuild})
        return 0

    monkeypatch.setattr(cli_agent_docker, "_ensure_image", fake_ensure)
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-key")

    import argparse
    args = argparse.Namespace(
        agent_command="docker",
        agent_docker_command="build",
        no_cache=True,
    )
    rc = cli_agent_docker.cmd_agent(args)
    assert rc == 0
    assert captured == [{"repo": cli_agent_docker._repo_root(), "force_rebuild": True}]


def test_ensure_image_returns_build_failure(monkeypatch):
    def fake_run(cmd, *, cwd=None, env=None, check=False):
        if "inspect" in cmd:
            return 1
        return 7  # build fails

    monkeypatch.setattr(cli_agent_docker, "run_streaming", fake_run)
    rc = cli_agent_docker._ensure_image(Path("/repo"))
    assert rc == 7


# ---------------------------------------------------------------------------
# Streaming subprocess — no capture
# ---------------------------------------------------------------------------

def test_module_uses_shared_streaming_helper():
    """Sanity check: the new module must use the shared streaming
    helper (which omits ``capture_output``), not call subprocess.run
    directly with potentially-buffered output. Codex's Tier 1.3
    lesson — same as cli_verify_install."""
    from kestrel_sovereign import _subprocess_helpers as sh
    assert cli_agent_docker.run_streaming is sh.run_streaming
