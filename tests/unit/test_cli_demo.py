"""``kestrel demo run`` CLI tests — sub-PR 3.1 of epic #1050
(bash-to-Python port of ``demos/run.sh``).

Covers:
- argparse wires up under both the local subparser and the real
  ``kestrel`` parser
- Unknown demo name → exit code 2 with a list of available demos
- Forbidden DEMO_PORT (8888 — the live server) → exit code 2
- Port already busy → exit code 2
- ``KESTREL_API_KEY`` is stripped from BOTH the demo-server env and
  the playwright env (production key must not auth against demo DB)
- Provider keys (ANTHROPIC_API_KEY, etc.) survive into playwright env
- ``KESTREL_DEMO_SERVER=1`` is set on both server + playwright env
- ``--keep-server`` skips the EXIT-trap teardown
- EXIT-trap teardown runs even when playwright fails

Subprocess + uvicorn are mocked — real demos run in the integration
tier.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign import cli_demo


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kestrel")
    sub = p.add_subparsers(dest="command")
    cli_demo.add_demo_subcommand(sub)
    return p


def test_argparse_demo_run_minimal():
    parser = _build_parser()
    args = parser.parse_args(["demo", "run", "technical"])
    assert args.command == "demo"
    assert args.demo_command == "run"
    assert args.name == "technical"
    assert args.port is None
    assert args.keep_server is False


def test_argparse_demo_run_port_and_keep_server():
    parser = _build_parser()
    args = parser.parse_args(
        ["demo", "run", "spawn", "--port", "9001", "--keep-server"]
    )
    assert args.name == "spawn"
    assert args.port == 9001
    assert args.keep_server is True


def test_kestrel_cli_registers_demo():
    """The full ``kestrel`` parser registers ``demo``. Guards against
    a future cli.py refactor accidentally dropping the wiring."""
    from kestrel_sovereign.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["demo", "run", "technical"])
    assert args.command == "demo"
    assert args.demo_command == "run"


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

class _Args:
    """Minimal argparse-style namespace for direct cmd_ calls."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_cmd_demo_no_subverb_prints_usage(capsys):
    rc = cli_demo.cmd_demo(_Args(demo_command=None))
    assert rc == 1
    captured = capsys.readouterr()
    assert "Usage" in captured.err
    assert "demo run" in captured.err


# ---------------------------------------------------------------------------
# Unknown / missing demo
# ---------------------------------------------------------------------------

def test_cmd_demo_run_unknown_demo_lists_available(monkeypatch, capsys):
    monkeypatch.setattr(cli_demo, "_list_demos", lambda repo: ["a", "b"])

    args = _Args(
        demo_command="run",
        name="zzz",
        port=None,
        keep_server=False,
    )
    rc = cli_demo.cmd_demo(args)
    assert rc == 2
    err = capsys.readouterr().err
    assert "'zzz' not found" in err
    assert "['a', 'b']" in err


# ---------------------------------------------------------------------------
# Port safety
# ---------------------------------------------------------------------------

def test_cmd_demo_run_refuses_port_8888(monkeypatch, capsys):
    monkeypatch.setattr(cli_demo, "_list_demos", lambda repo: ["technical"])
    # Pretend config.cjs exists so we get past that check.
    monkeypatch.setattr(Path, "is_file", lambda self: True)

    args = _Args(
        demo_command="run",
        name="technical",
        port=8888,
        keep_server=False,
    )
    rc = cli_demo.cmd_demo(args)
    assert rc == 2
    err = capsys.readouterr().err
    assert "8888" in err
    assert "live server" in err


def test_cmd_demo_run_refuses_busy_port(monkeypatch, capsys):
    monkeypatch.setattr(cli_demo, "_list_demos", lambda repo: ["technical"])
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(cli_demo, "_port_is_busy", lambda port: True)

    args = _Args(
        demo_command="run",
        name="technical",
        port=8901,
        keep_server=False,
    )
    rc = cli_demo.cmd_demo(args)
    assert rc == 2
    err = capsys.readouterr().err
    assert "already in use" in err


# ---------------------------------------------------------------------------
# Env munging — KESTREL_API_KEY scrub
# ---------------------------------------------------------------------------

def test_build_demo_env_strips_api_key_sets_signal_flags(tmp_path):
    parent = {
        "KESTREL_API_KEY": "production-key-must-not-leak",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "PATH": "/usr/bin",
    }
    demo_db = tmp_path / "demo"
    env = cli_demo._build_demo_env(parent, demo_db)
    assert "KESTREL_API_KEY" not in env
    assert env["KESTREL_DB_PATH"] == str(demo_db)
    assert env["KESTREL_DEMO_SERVER"] == "1"
    assert env["KESTREL_MULTI_AGENT_CONFIG"] == str(
        demo_db / "multi_agent-disabled.toml"
    )
    # Provider key untouched.
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-..."


def test_build_playwright_env_strips_api_key_preserves_provider_keys():
    parent = {
        "KESTREL_API_KEY": "production-key-must-not-leak",
        "ANTHROPIC_API_KEY": "sk-ant-prod",
        "OPENROUTER_API_KEY": "sk-or-prod",
        "OPENAI_API_KEY": "sk-oai-prod",
        "GEMINI_API_KEY": "g-key",
        "XAI_API_KEY": "xai-key",
        "REPLICATE_API_TOKEN": "r8-key",
        "TAVILY_API_KEY": "tvly-key",
        "RUNPOD_API_KEY": "rp-key",
        "OLLAMA_HOST": "http://localhost:11434",
        "PATH": "/usr/bin",
    }
    env = cli_demo._build_playwright_env(parent, "http://127.0.0.1:8900")
    assert "KESTREL_API_KEY" not in env
    assert env["KESTREL_URL"] == "http://127.0.0.1:8900"
    assert env["KESTREL_DEMO_SERVER"] == "1"
    # All provider keys survive.
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-prod"
    assert env["OPENROUTER_API_KEY"] == "sk-or-prod"
    assert env["OPENAI_API_KEY"] == "sk-oai-prod"
    assert env["GEMINI_API_KEY"] == "g-key"
    assert env["XAI_API_KEY"] == "xai-key"
    assert env["REPLICATE_API_TOKEN"] == "r8-key"
    assert env["TAVILY_API_KEY"] == "tvly-key"
    assert env["RUNPOD_API_KEY"] == "rp-key"
    assert env["OLLAMA_HOST"] == "http://localhost:11434"


# ---------------------------------------------------------------------------
# /api/agents sanity check
# ---------------------------------------------------------------------------

def test_verify_only_demo_agents_passes_when_all_demo(monkeypatch):
    body = json.dumps({
        "agents": [
            {"name": "demo-a", "is_demo": True},
            {"name": "demo-b", "is_demo": True},
        ]
    }).encode()

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return body

    monkeypatch.setattr(cli_demo.urllib.request, "urlopen", lambda *a, **kw: _Resp())
    assert cli_demo._verify_only_demo_agents("http://127.0.0.1:8900") is None


def test_verify_only_demo_agents_flags_live_agents(monkeypatch):
    body = json.dumps({
        "agents": [
            {"name": "Meridian", "is_demo": False},
            {"name": "demo-a", "is_demo": True},
        ]
    }).encode()

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return body

    monkeypatch.setattr(cli_demo.urllib.request, "urlopen", lambda *a, **kw: _Resp())
    bad = cli_demo._verify_only_demo_agents("http://127.0.0.1:8900")
    assert bad == "Meridian"


# ---------------------------------------------------------------------------
# Full lifecycle — happy path with mocks
# ---------------------------------------------------------------------------

def _patch_full_lifecycle(monkeypatch, *, playwright_rc: int = 0):
    """Wire up monkeypatches for a full run: list_demos, port checks,
    setup_demo_agent, uvicorn spawn, /health probe, /api/agents
    sanity, and the playwright shell-out.

    Returns a dict the caller can interrogate to confirm what was
    invoked.
    """
    state: dict = {
        "playwright_calls": [],
        "setup_calls": [],
        "started": False,
        "stopped": False,
        "playwright_env": None,
        "server_env": None,
    }

    monkeypatch.setattr(cli_demo, "_list_demos", lambda repo: ["technical"])
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(cli_demo, "_port_is_busy", lambda port: False)
    monkeypatch.setattr(
        cli_demo, "_verify_only_demo_agents", lambda url: None,
    )

    fake_proc = MagicMock()
    fake_proc.pid = 12345
    fake_proc.poll.return_value = None

    def fake_start(cmd, cwd=None, env=None, stdout=None, stderr=None):
        state["started"] = True
        state["server_cmd"] = list(cmd)
        state["server_env"] = dict(env) if env is not None else None
        return fake_proc

    monkeypatch.setattr(cli_demo, "start_background_process", fake_start)
    monkeypatch.setattr(cli_demo, "wait_for_health", lambda port, timeout=60.0, proc=None: True)

    def fake_stop(proc, timeout=10.0):
        state["stopped"] = True

    monkeypatch.setattr(cli_demo, "stop_process", fake_stop)

    def fake_run_streaming(cmd, *, cwd=None, env=None, check=False):
        argv = list(cmd)
        if "setup_demo_agent.py" in " ".join(argv):
            state["setup_calls"].append(argv)
            return 0
        if argv[:2] == ["npx", "playwright"]:
            state["playwright_calls"].append(argv)
            state["playwright_env"] = dict(env) if env is not None else None
            return playwright_rc
        # Anything else — ignore, return 0.
        return 0

    monkeypatch.setattr(cli_demo, "run_streaming", fake_run_streaming)

    return state


def test_cmd_demo_run_happy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("KESTREL_API_KEY", "leak-me-not")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-good")
    state = _patch_full_lifecycle(monkeypatch, playwright_rc=0)

    args = _Args(
        demo_command="run",
        name="technical",
        port=None,
        keep_server=False,
    )
    rc = cli_demo.cmd_demo(args)
    assert rc == 0
    assert state["started"] is True
    assert state["stopped"] is True
    assert state["playwright_calls"], "playwright must have been invoked"
    # KESTREL_API_KEY scrubbed; provider key preserved.
    pw_env = state["playwright_env"]
    assert "KESTREL_API_KEY" not in pw_env
    assert pw_env["ANTHROPIC_API_KEY"] == "sk-ant-good"
    assert pw_env["KESTREL_DEMO_SERVER"] == "1"
    # Server env: KESTREL_DEMO_SERVER + DB path.
    sv_env = state["server_env"]
    assert sv_env["KESTREL_DEMO_SERVER"] == "1"
    assert "KESTREL_API_KEY" not in sv_env


def test_cmd_demo_run_playwright_failure_still_stops_server(monkeypatch):
    state = _patch_full_lifecycle(monkeypatch, playwright_rc=3)

    args = _Args(
        demo_command="run",
        name="technical",
        port=None,
        keep_server=False,
    )
    rc = cli_demo.cmd_demo(args)
    assert rc == 3
    assert state["started"] is True
    # EXIT-trap discipline: server stops even if the demo failed.
    assert state["stopped"] is True


def test_cmd_demo_run_keep_server_skips_teardown(monkeypatch):
    state = _patch_full_lifecycle(monkeypatch, playwright_rc=0)

    args = _Args(
        demo_command="run",
        name="technical",
        port=None,
        keep_server=True,
    )
    rc = cli_demo.cmd_demo(args)
    assert rc == 0
    assert state["started"] is True
    assert state["stopped"] is False  # --keep-server


def test_cmd_demo_run_unhealthy_server_returns_one(monkeypatch):
    monkeypatch.setattr(cli_demo, "_list_demos", lambda repo: ["technical"])
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(cli_demo, "_port_is_busy", lambda port: False)

    fake_proc = MagicMock()
    fake_proc.pid = 12345
    fake_proc.poll.return_value = None
    monkeypatch.setattr(
        cli_demo, "start_background_process",
        lambda *a, **kw: fake_proc,
    )
    monkeypatch.setattr(
        cli_demo, "wait_for_health",
        lambda port, timeout=60.0, proc=None: False,
    )
    stop_calls: list = []
    monkeypatch.setattr(
        cli_demo, "stop_process",
        lambda proc, timeout=10.0: stop_calls.append(proc),
    )
    monkeypatch.setattr(
        cli_demo, "run_streaming",
        lambda cmd, **kw: 0,
    )

    args = _Args(
        demo_command="run",
        name="technical",
        port=None,
        keep_server=False,
    )
    rc = cli_demo.cmd_demo(args)
    assert rc == 1
    assert stop_calls, "server must be stopped on failed health probe"


def test_cmd_demo_run_non_demo_agent_aborts(monkeypatch):
    monkeypatch.setattr(cli_demo, "_list_demos", lambda repo: ["technical"])
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(cli_demo, "_port_is_busy", lambda port: False)
    monkeypatch.setattr(cli_demo, "_verify_only_demo_agents", lambda url: "Meridian")

    fake_proc = MagicMock()
    fake_proc.pid = 12345
    fake_proc.poll.return_value = None
    monkeypatch.setattr(
        cli_demo, "start_background_process",
        lambda *a, **kw: fake_proc,
    )
    monkeypatch.setattr(
        cli_demo, "wait_for_health",
        lambda port, timeout=60.0, proc=None: True,
    )
    stop_calls: list = []
    monkeypatch.setattr(
        cli_demo, "stop_process",
        lambda proc, timeout=10.0: stop_calls.append(proc),
    )

    setup_calls: list = []

    def fake_run_streaming(cmd, *, cwd=None, env=None, check=False):
        argv = list(cmd)
        if "setup_demo_agent.py" in " ".join(argv):
            setup_calls.append(argv)
            return 0
        # Playwright must NOT be invoked in this test.
        if argv[:2] == ["npx", "playwright"]:
            raise AssertionError(
                "playwright must not run when /api/agents reports a "
                "non-demo agent"
            )
        return 0

    monkeypatch.setattr(cli_demo, "run_streaming", fake_run_streaming)

    args = _Args(
        demo_command="run",
        name="technical",
        port=None,
        keep_server=False,
    )
    rc = cli_demo.cmd_demo(args)
    assert rc == 1
    assert stop_calls, "server must be stopped on failed sanity check"


# ---------------------------------------------------------------------------
# Subprocess streaming guarantee
# ---------------------------------------------------------------------------

def test_run_streaming_does_not_capture(monkeypatch):
    """Codex's Tier 1.3 lesson: subprocess output must stream live."""
    from kestrel_sovereign import _subprocess_helpers as sh

    captured_kwargs = {}

    def fake_run(cmd, **kwargs):
        captured_kwargs.update(kwargs)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(sh.subprocess, "run", fake_run)
    rc = sh.run_streaming(["echo", "hi"])
    assert rc == 0
    assert "capture_output" not in captured_kwargs
    assert captured_kwargs.get("check") is False
