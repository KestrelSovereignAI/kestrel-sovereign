"""``kestrel docker remote`` CLI tests — sub-PR 3.3 of epic #1050
(bash-to-Python port of ``scripts/{build,run}_docker_remote.sh``).

Covers:
- argparse wiring under the local subparser and the real ``kestrel``
  parser
- ``build`` invokes ``docker build`` with the right flags
- ``run`` validates ``OPENAI_API_KEY`` is present in .env
- ``run`` errors cleanly when .env is missing
- Host gateway is platform-detected: ``host.docker.internal`` on
  macOS/Windows, ``172.17.0.1`` on Linux
- ``run`` argv shape: container name, port mapping, env-var forwarding
- The new module preserves the property the legacy bash test guarded:
  no hidden ``DEFAULT_LLM_MODEL`` injection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign import cli_docker_remote


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kestrel")
    sub = p.add_subparsers(dest="command")
    cli_docker_remote.add_docker_subcommand(sub)
    return p


def test_argparse_build_defaults():
    parser = _build_parser()
    args = parser.parse_args(["docker", "remote", "build"])
    assert args.command == "docker"
    assert args.docker_command == "remote"
    assert args.docker_remote_command == "build"
    assert args.tag == "latest"
    assert args.platform == "linux/amd64"


def test_argparse_build_overrides():
    parser = _build_parser()
    args = parser.parse_args(
        ["docker", "remote", "build", "--tag", "v1.2.3", "--platform", "linux/arm64"]
    )
    assert args.tag == "v1.2.3"
    assert args.platform == "linux/arm64"


def test_argparse_run_defaults():
    parser = _build_parser()
    args = parser.parse_args(["docker", "remote", "run"])
    assert args.docker_remote_command == "run"
    assert args.port is None
    assert args.env_file is None
    assert args.tag == "latest"


def test_argparse_run_overrides():
    parser = _build_parser()
    args = parser.parse_args(
        ["docker", "remote", "run", "--port", "9000", "--env-file", "/tmp/.env", "--tag", "dev"]
    )
    assert args.port == 9000
    assert args.env_file == "/tmp/.env"
    assert args.tag == "dev"


def test_kestrel_cli_registers_docker_remote():
    from kestrel_sovereign.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["docker", "remote", "build"])
    assert args.command == "docker"
    assert args.docker_command == "remote"
    assert args.docker_remote_command == "build"


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_cmd_docker_no_subverb_prints_usage(capsys):
    rc = cli_docker_remote.cmd_docker(_Args(docker_command=None))
    assert rc == 1
    err = capsys.readouterr().err
    assert "Usage" in err


def test_cmd_docker_remote_no_subverb_prints_usage(capsys):
    rc = cli_docker_remote.cmd_docker(
        _Args(docker_command="remote", docker_remote_command=None)
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "build" in err and "run" in err


# ---------------------------------------------------------------------------
# build subverb
# ---------------------------------------------------------------------------

def test_build_invokes_docker_build_with_defaults(monkeypatch):
    captured: list = []

    def fake_run(cmd, *, cwd=None, env=None, check=False):
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(cli_docker_remote, "run_streaming", fake_run)

    args = _Args(
        docker_command="remote",
        docker_remote_command="build",
        tag="latest",
        platform="linux/amd64",
    )
    rc = cli_docker_remote.cmd_docker(args)
    assert rc == 0
    assert len(captured) == 1
    argv = captured[0]
    assert argv[0:2] == ["docker", "build"]
    assert "-f" in argv and "Dockerfile.agent.remote" in argv
    assert "-t" in argv and "kestrel-remote:latest" in argv
    assert "--platform" in argv and "linux/amd64" in argv
    # Build context is "." with cwd set to repo root.
    assert argv[-1] == "."


def test_build_overrides_propagate(monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        cli_docker_remote, "run_streaming",
        lambda cmd, **kw: captured.append(list(cmd)) or 0,
    )

    args = _Args(
        docker_command="remote",
        docker_remote_command="build",
        tag="v9",
        platform="linux/arm64",
    )
    rc = cli_docker_remote.cmd_docker(args)
    assert rc == 0
    argv = captured[0]
    assert "kestrel-remote:v9" in argv
    assert "linux/arm64" in argv


def test_build_returns_failure(monkeypatch):
    monkeypatch.setattr(
        cli_docker_remote, "run_streaming",
        lambda cmd, **kw: 5,
    )
    args = _Args(
        docker_command="remote",
        docker_remote_command="build",
        tag="latest",
        platform="linux/amd64",
    )
    assert cli_docker_remote.cmd_docker(args) == 5


# ---------------------------------------------------------------------------
# run subverb — .env handling
# ---------------------------------------------------------------------------

def _write_env(tmp_path: Path, **kvs) -> Path:
    env_path = tmp_path / ".env"
    lines = [f"{k}={v}" for k, v in kvs.items()]
    env_path.write_text("\n".join(lines) + "\n")
    return env_path


def test_run_missing_env_file_errors(tmp_path, monkeypatch, capsys):
    args = _Args(
        docker_command="remote",
        docker_remote_command="run",
        port=None,
        env_file=str(tmp_path / "missing.env"),
        tag="latest",
    )
    rc = cli_docker_remote.cmd_docker(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert ".env file not found" in err


def test_run_missing_openai_key_errors(tmp_path, monkeypatch, capsys):
    env_file = _write_env(tmp_path, KESTREL_API_KEY="some-key")
    args = _Args(
        docker_command="remote",
        docker_remote_command="run",
        port=None,
        env_file=str(env_file),
        tag="latest",
    )
    rc = cli_docker_remote.cmd_docker(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err


def test_run_happy_path_argv_shape(tmp_path, monkeypatch):
    env_file = _write_env(
        tmp_path,
        OPENAI_API_KEY="sk-oai",
        ANTHROPIC_API_KEY="sk-ant",
        KESTREL_API_KEY="ks-api",
        KESTREL_DATA_KEY="ks-data",
    )
    captured: list = []

    def fake_run(cmd, *, cwd=None, env=None, check=False):
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(cli_docker_remote, "run_streaming", fake_run)
    monkeypatch.setattr(
        cli_docker_remote, "_detect_ollama_host",
        lambda: "http://host.docker.internal:11434",
    )
    # Skip the post-run /health probe — that's tested separately.
    monkeypatch.setattr(
        cli_docker_remote, "_wait_for_container_health",
        lambda url, *, timeout: True,
    )

    args = _Args(
        docker_command="remote",
        docker_remote_command="run",
        port=9001,
        env_file=str(env_file),
        tag="latest",
    )
    rc = cli_docker_remote.cmd_docker(args)
    assert rc == 0
    # Three calls: stop, rm, run.
    assert [c[:2] for c in captured] == [
        ["docker", "stop"],
        ["docker", "rm"],
        ["docker", "run"],
    ]
    run_argv = captured[2]
    assert "-d" in run_argv
    assert "--name" in run_argv and "kestrel-remote" in run_argv
    # Port mapping uses the user override.
    assert "-p" in run_argv and "9001:8888" in run_argv
    assert "--add-host=host.docker.internal:host-gateway" in run_argv
    flat = " ".join(run_argv)
    assert "OPENAI_API_KEY=sk-oai" in flat
    assert "OLLAMA_HOST=http://host.docker.internal:11434" in flat
    assert "ANTHROPIC_API_KEY=sk-ant" in flat
    assert "KESTREL_API_KEY=ks-api" in flat
    assert "KESTREL_DATA_KEY=ks-data" in flat
    assert run_argv[-1] == "kestrel-remote:latest"


def test_run_default_port_is_8888(tmp_path, monkeypatch):
    env_file = _write_env(tmp_path, OPENAI_API_KEY="sk-oai")
    captured: list = []
    monkeypatch.setattr(
        cli_docker_remote, "run_streaming",
        lambda cmd, **kw: captured.append(list(cmd)) or 0,
    )
    monkeypatch.setattr(
        cli_docker_remote, "_wait_for_container_health",
        lambda url, *, timeout: True,
    )

    args = _Args(
        docker_command="remote",
        docker_remote_command="run",
        port=None,
        env_file=str(env_file),
        tag="latest",
    )
    rc = cli_docker_remote.cmd_docker(args)
    assert rc == 0
    run_argv = captured[2]
    assert "8888:8888" in run_argv


def test_run_unhealthy_container_returns_1_and_dumps_logs(tmp_path, monkeypatch):
    """Codex review on PR #1071: the bash predecessor curled /health
    after ``docker run -d`` and returned non-zero on failure with the
    container logs printed. The Python port must do the same — without
    the probe a crashed container falsely reports success."""
    env_file = _write_env(tmp_path, OPENAI_API_KEY="sk-oai")
    captured: list = []
    monkeypatch.setattr(
        cli_docker_remote, "run_streaming",
        lambda cmd, **kw: captured.append(list(cmd)) or 0,
    )
    monkeypatch.setattr(
        cli_docker_remote, "_wait_for_container_health",
        lambda url, *, timeout: False,  # /health never returns 200
    )

    args = _Args(
        docker_command="remote",
        docker_remote_command="run",
        port=None,
        env_file=str(env_file),
        tag="latest",
    )
    rc = cli_docker_remote.cmd_docker(args)
    assert rc == 1, "must surface failure, not falsely report success"
    # Log dump invocation (``docker logs --tail 50 <container>``) was made.
    assert any(
        c[:3] == ["docker", "logs", "--tail"] for c in captured
    ), f"expected docker logs --tail to be called; saw {captured}"


def test_run_drops_empty_optional_keys(tmp_path, monkeypatch):
    """Optional forwarded env vars with empty values must NOT show up
    in the docker run argv (matches the bash predecessor's
    ``${KEY:-}`` behaviour, but stricter — empties get dropped)."""
    env_file = _write_env(
        tmp_path,
        OPENAI_API_KEY="sk-oai",
        REPLICATE_API_TOKEN="",  # explicit empty
    )
    captured: list = []
    monkeypatch.setattr(
        cli_docker_remote, "run_streaming",
        lambda cmd, **kw: captured.append(list(cmd)) or 0,
    )
    monkeypatch.setattr(
        cli_docker_remote, "_wait_for_container_health",
        lambda url, *, timeout: True,
    )

    args = _Args(
        docker_command="remote",
        docker_remote_command="run",
        port=None,
        env_file=str(env_file),
        tag="latest",
    )
    rc = cli_docker_remote.cmd_docker(args)
    assert rc == 0
    run_argv = captured[2]
    flat = " ".join(run_argv)
    # Empty REPLICATE_API_TOKEN is dropped, not forwarded as
    # ``REPLICATE_API_TOKEN=``.
    assert "REPLICATE_API_TOKEN=" not in flat


# ---------------------------------------------------------------------------
# Cross-platform host detection
# ---------------------------------------------------------------------------

def test_detect_ollama_host_darwin(monkeypatch):
    monkeypatch.setattr(cli_docker_remote.sys, "platform", "darwin")
    assert (
        cli_docker_remote._detect_ollama_host()
        == "http://host.docker.internal:11434"
    )


def test_detect_ollama_host_win32(monkeypatch):
    """Docker Desktop on Windows also resolves
    ``host.docker.internal``; we don't fall back to 172.17.0.1
    there."""
    monkeypatch.setattr(cli_docker_remote.sys, "platform", "win32")
    assert (
        cli_docker_remote._detect_ollama_host()
        == "http://host.docker.internal:11434"
    )


def test_detect_ollama_host_linux(monkeypatch):
    monkeypatch.setattr(cli_docker_remote.sys, "platform", "linux")
    assert (
        cli_docker_remote._detect_ollama_host()
        == "http://172.17.0.1:11434"
    )


# ---------------------------------------------------------------------------
# .env parsing edge cases
# ---------------------------------------------------------------------------

def test_load_env_file_skips_none_values(tmp_path):
    """``FOO=`` (no RHS) → dotenv_values returns ``None`` → we drop it
    so the caller's ``env_vars.get("FOO")`` returns None, not ``""``."""
    env_file = tmp_path / ".env"
    env_file.write_text("KEY1=value1\nKEY2=\n")
    out = cli_docker_remote._load_env_file(env_file)
    assert "KEY1" in out
    # Empty-string values from ``KEY=`` lines come through as empty
    # strings, which our truthy check below treats as missing.
    assert out.get("KEY2", "missing") in ("", "missing")


def test_load_env_file_no_interpolation(tmp_path):
    """``${...}`` placeholders must not be expanded — we want raw
    secret bytes, not their expanded form."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-${SALT}-rest\n")
    out = cli_docker_remote._load_env_file(env_file)
    assert out["OPENAI_API_KEY"] == "sk-${SALT}-rest"


def test_load_env_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli_docker_remote._load_env_file(tmp_path / "no-such.env")


# ---------------------------------------------------------------------------
# Hidden default-model contract — preserves the legacy bash test
# ---------------------------------------------------------------------------

def test_run_does_not_inject_hidden_default_model():
    """The bash predecessor was guarded by
    ``test_run_docker_remote_does_not_inject_hidden_default_model``
    in test_bootstrap_model_defaults.py. The Python port must
    preserve the same property — no hardcoded ``DEFAULT_LLM_MODEL``
    sneaks into the docker run argv path.
    """
    src = Path(cli_docker_remote.__file__).read_text(encoding="utf-8")
    assert "DEFAULT_LLM_MODEL" not in src


# ---------------------------------------------------------------------------
# Streaming subprocess — no capture
# ---------------------------------------------------------------------------

def test_module_uses_shared_streaming_helper():
    from kestrel_sovereign import _subprocess_helpers as sh
    assert cli_docker_remote.run_streaming is sh.run_streaming
