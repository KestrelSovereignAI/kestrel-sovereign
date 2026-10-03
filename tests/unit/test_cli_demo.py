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
import signal
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
        "KESTREL_DB_BACKEND": "postgres",
        "KESTREL_DATABASE_URL": "postgresql://primary/live",
        "KESTREL_HOLD_EVIDENCE_DATABASE_URL": "postgresql://evidence/live",
        "KESTREL_HOLD_BACKEND": "postgres",
        "KESTREL_HOST_DB_PATH": "/srv/live/host-features.db",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "PATH": "/usr/bin",
    }
    demo_db = tmp_path / "demo"
    env = cli_demo._build_demo_env(parent, demo_db, tmp_path)
    assert "KESTREL_API_KEY" not in env
    assert env["KESTREL_DB_PATH"] == str(demo_db)
    assert env["KESTREL_DEMO_SERVER"] == "1"
    assert env["KESTREL_MULTI_AGENT_CONFIG"] == str(
        demo_db / "multi_agent-disabled.toml"
    )
    assert env["KESTREL_HOST_DB_PATH"] == str(
        demo_db / "host-data" / "host-features.db"
    )
    assert env["KESTREL_DB_BACKEND"] == "sqlite"
    assert "KESTREL_DATABASE_URL" not in env
    assert "KESTREL_HOLD_EVIDENCE_DATABASE_URL" not in env
    assert "KESTREL_HOLD_BACKEND" not in env
    # Provider key untouched.
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-..."


def test_build_demo_env_loads_dotenv_under_parent(tmp_path):
    """Codex review on PR #1071: ``demos/run.sh`` did ``source $ROOT/.env``;
    the Python port skipped this so demos with provider keys only in
    .env failed silently. Now .env is loaded UNDERNEATH parent_env so
    explicit shell exports still win on collision.
    """
    repo = tmp_path
    (repo / ".env").write_text(
        "ANTHROPIC_API_KEY=from-dotenv\n"
        "OPENAI_API_KEY=from-dotenv\n"
        # Operator's shell exports the same key with a different value;
        # parent_env wins.
        "OVERRIDDEN=stale\n"
    )
    parent = {
        "OVERRIDDEN": "shell-export",
        "KESTREL_API_KEY": "leak",
        "PATH": "/usr/bin",
    }
    demo_db = tmp_path / "demo"
    env = cli_demo._build_demo_env(parent, demo_db, repo)
    assert env["ANTHROPIC_API_KEY"] == "from-dotenv"  # only in .env
    assert env["OPENAI_API_KEY"] == "from-dotenv"     # only in .env
    assert env["OVERRIDDEN"] == "shell-export"        # parent wins
    assert "KESTREL_API_KEY" not in env               # always stripped


def test_build_playwright_env_strips_api_key_preserves_provider_keys(tmp_path):
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
        "KESTREL_DB_BACKEND": "postgres",
        "KESTREL_DATABASE_URL": "postgresql://primary/live",
        "KESTREL_HOLD_EVIDENCE_DATABASE_URL": "postgresql://evidence/live",
        "KESTREL_HOLD_BACKEND": "postgres",
        "KESTREL_HOST_DB_PATH": "/srv/live/host-features.db",
        "PATH": "/usr/bin",
    }
    demo_db = tmp_path / "agent_data" / "demo"
    env = cli_demo._build_playwright_env(parent, "http://127.0.0.1:8900", tmp_path, demo_db)
    assert "KESTREL_API_KEY" not in env
    assert env["KESTREL_URL"] == "http://127.0.0.1:8900"
    assert env["KESTREL_DEMO_SERVER"] == "1"
    # The demo must know its isolated sandbox so resetDemoAgentDatabases can
    # prove which dir is safe to reset (issue #1973).
    assert env["KESTREL_DB_PATH"] == str(demo_db)
    assert env["KESTREL_HOST_DB_PATH"] == str(
        demo_db / "host-data" / "host-features.db"
    )
    assert env["KESTREL_DB_BACKEND"] == "sqlite"
    assert "KESTREL_DATABASE_URL" not in env
    assert "KESTREL_HOLD_EVIDENCE_DATABASE_URL" not in env
    assert "KESTREL_HOLD_BACKEND" not in env
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


# ---------------------------------------------------------------------------
# Demo-agent verification must authenticate (issue: runner-auth)
# ---------------------------------------------------------------------------

def test_verify_only_demo_agents_authenticates_with_minted_key(monkeypatch):
    """/api/agents is authenticated. The verifier must mint the demo key via
    the public /api/auth/key and send it as X-API-Key, else it always 401s and
    `kestrel demo run` can never pass the guard."""
    seen_headers = {}

    def fake_urlopen(req, timeout=5):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/auth/key"):
            return BytesIO(json.dumps({"key": "demo-key-xyz"}).encode())
        # /api/agents — capture headers the verifier attached
        if not isinstance(req, str):
            seen_headers.update(req.headers)
        return BytesIO(json.dumps({"agents": [{"name": "demo", "is_demo": True}]}).encode())

    monkeypatch.setattr(cli_demo.urllib.request, "urlopen", fake_urlopen)
    result = cli_demo._verify_only_demo_agents("http://127.0.0.1:8900")
    assert result is None  # all agents is_demo=true → OK
    # header key is title-cased by urllib (X-api-key)
    assert any(k.lower() == "x-api-key" and v == "demo-key-xyz" for k, v in seen_headers.items())


def test_verify_only_demo_agents_flags_live_agent(monkeypatch):
    def fake_urlopen(req, timeout=5):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/auth/key"):
            return BytesIO(json.dumps({"key": "k"}).encode())
        return BytesIO(json.dumps({"agents": [{"name": "Meridian", "is_demo": False}]}).encode())

    monkeypatch.setattr(cli_demo.urllib.request, "urlopen", fake_urlopen)
    assert cli_demo._verify_only_demo_agents("http://127.0.0.1:8900") == "Meridian"


# ---------------------------------------------------------------------------
# `kestrel demo smoke` — the Sovereign Console Playwright smoke (#2682)
# ---------------------------------------------------------------------------

_SMOKE_DID = "did:web:localhost:kestrel-demo-agent-abc123"


def test_argparse_demo_smoke_defaults_and_options():
    parser = _build_parser()
    args = parser.parse_args(["demo", "smoke"])
    assert args.demo_command == "smoke"
    assert args.home is None
    assert args.port is None
    assert args.keep_server is False

    args = parser.parse_args(
        ["demo", "smoke", "--home", "/tmp/h", "--port", "9010", "--keep-server"]
    )
    assert (args.home, args.port, args.keep_server) == ("/tmp/h", 9010, True)


def test_kestrel_cli_registers_demo_smoke():
    from kestrel_sovereign.cli import build_parser

    args = build_parser().parse_args(["demo", "smoke"])
    assert (args.command, args.demo_command) == ("demo", "smoke")


def test_cmd_demo_usage_names_both_subverbs(capsys):
    assert cli_demo.cmd_demo(_Args(demo_command=None)) == 1
    err = capsys.readouterr().err
    assert "demo run" in err
    assert "demo smoke" in err


def test_build_smoke_env_inherits_no_kestrel_setting_or_credential(tmp_path):
    parent = {
        "PATH": "/usr/bin",
        "HOME": "/home/dev",
        "PYTHONPATH": "/src/checkout",
        "KESTREL_API_KEY": "production-key",
        "KESTREL_DATA_KEY": "production-data-key",
        "KESTREL_HOME": "/srv/live",
        "KESTREL_DATABASE_URL": "postgresql://primary/live",
        "KESTREL_SOVEREIGN_TRUST_ROOT_PATH": "/srv/live/trust-root.json",
        "ANTHROPIC_API_KEY": "sk-ant-prod",
        "OPENROUTER_API_KEY": "sk-or-prod",
        "ANTHROPIC_AUTH_TOKEN": "oauth-prod",
        "CLOUD_ACCESS_KEY": "cloud-prod",
        "STRIPE_WEBHOOK_SECRET": "whsec",
        "DB_PASSWORD": "pw",
        "OLLAMA_HOST": "http://gpu-box:11434",
    }
    home = tmp_path / "home"
    data_dir = home / "agent_data" / "console-smoke"
    env = cli_demo._build_smoke_env(parent, home, data_dir, "ephemeral-key")

    # Ordinary process environment survives.
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/dev"
    assert env["PYTHONPATH"] == "/src/checkout"
    # No production key, credential, or local-LLM redirect crosses over.
    for name in (
        "KESTREL_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
        "ANTHROPIC_AUTH_TOKEN", "CLOUD_ACCESS_KEY", "STRIPE_WEBHOOK_SECRET",
        "DB_PASSWORD", "OLLAMA_HOST", "KESTREL_DATABASE_URL",
        "KESTREL_SOVEREIGN_TRUST_ROOT_PATH",
    ):
        assert name not in env, name
    # Every KESTREL_* setting is the instance's own.
    assert {k: v for k, v in env.items() if k.startswith("KESTREL_")} == {
        "KESTREL_HOME": str(home),
        "KESTREL_SKIP_DOTENV": "1",
        "KESTREL_DATA_KEY": "ephemeral-key",
        "KESTREL_DID_WEB_DOMAIN": "localhost",
        "KESTREL_DB_BACKEND": "sqlite",
        "KESTREL_DB_PATH": str(data_dir),
        "KESTREL_HOST_DB_PATH": str(data_dir / "host-data" / "host-features.db"),
        "KESTREL_MULTI_AGENT_CONFIG": str(home / "multi_agent-disabled.toml"),
        "KESTREL_DEMO_SERVER": "1",
        "KESTREL_SKIP_REACHABILITY_PROBE": "1",
        "KESTREL_PHOENIX_ENABLED": "0",
    }


def test_ephemeral_data_key_is_fresh_each_run():
    first, second = cli_demo._ephemeral_data_key(), cli_demo._ephemeral_data_key()
    assert first != second
    assert len(first) == 44


def test_prepare_smoke_home_creates_a_fresh_private_dir(tmp_path):
    created = cli_demo._prepare_smoke_home(None)
    try:
        assert created.is_dir() and not any(created.iterdir())
    finally:
        created.rmdir()

    target = tmp_path / "nested" / "home"
    assert cli_demo._prepare_smoke_home(str(target)) == target.resolve()
    assert target.is_dir()
    if os.name == "posix":
        assert target.stat().st_mode & 0o777 == 0o700

    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli_demo._prepare_smoke_home(str(empty)) == empty.resolve()


def test_prepare_smoke_home_never_reuses_a_populated_dir(tmp_path):
    populated = tmp_path / "populated"
    populated.mkdir()
    (populated / "kestrel_prime.db").write_text("live")
    with pytest.raises(cli_demo._SmokeSetupError, match="not empty"):
        cli_demo._prepare_smoke_home(str(populated))
    assert (populated / "kestrel_prime.db").read_text() == "live"

    a_file = tmp_path / "file"
    a_file.write_text("x")
    with pytest.raises(cli_demo._SmokeSetupError):
        cli_demo._prepare_smoke_home(str(a_file))


@pytest.mark.parametrize(
    ("port", "busy", "message"),
    [(8888, False, "live server"), (8911, True, "already in use")],
)
def test_cmd_demo_smoke_refuses_unsafe_port(monkeypatch, capsys, tmp_path, port, busy, message):
    monkeypatch.setattr(cli_demo, "_port_is_busy", lambda p: busy)
    home = tmp_path / "home"
    rc = cli_demo.cmd_demo(
        _Args(demo_command="smoke", home=str(home), port=port, keep_server=False)
    )
    assert rc == 2
    assert message in capsys.readouterr().err
    assert not home.exists(), "a refused port must not create the instance"


def test_cmd_demo_smoke_refuses_populated_home(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli_demo, "_port_is_busy", lambda p: False)
    (tmp_path / "leftover").write_text("x")
    monkeypatch.setattr(
        cli_demo, "run_streaming",
        lambda *a, **kw: pytest.fail("no agent may be created in a populated home"),
    )
    rc = cli_demo.cmd_demo(
        _Args(demo_command="smoke", home=str(tmp_path), port=None, keep_server=False)
    )
    assert rc == 2
    assert "not empty" in capsys.readouterr().err


def test_load_inception_manifest_checks_did_and_database_location(tmp_path):
    data_dir = tmp_path / "agent_data" / "console-smoke"
    path = tmp_path / "inception.json"

    path.write_text(json.dumps({
        "agent_did": _SMOKE_DID, "db_path": str(data_dir / "kestrel_prime.db"),
    }))
    assert cli_demo._load_inception_manifest(path, data_dir)["agent_did"] == _SMOKE_DID

    path.write_text(json.dumps({
        "agent_did": _SMOKE_DID, "db_path": "/srv/live/kestrel_prime.db",
    }))
    with pytest.raises(cli_demo._SmokeSetupError, match="outside the smoke data dir"):
        cli_demo._load_inception_manifest(path, data_dir)

    path.write_text(json.dumps({"db_path": str(data_dir / "kestrel_prime.db")}))
    with pytest.raises(cli_demo._SmokeSetupError, match="no agent DID"):
        cli_demo._load_inception_manifest(path, data_dir)

    with pytest.raises(cli_demo._SmokeSetupError, match="cannot read"):
        cli_demo._load_inception_manifest(tmp_path / "missing.json", data_dir)


def test_server_module_origin_resolves_under_the_given_environment(tmp_path):
    """The probe must resolve kestrel_sovereign as a process launched with
    the SAME env and cwd would — here, a PYTHONPATH naming this checkout."""
    package_dir = Path(cli_demo.__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_dir.parent)
    assert cli_demo._server_module_origin(tmp_path, env) == package_dir


def test_server_module_origin_failure_is_a_refusal(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(cli_demo.subprocess, "run", fake_run)
    with pytest.raises(cli_demo._SmokeSetupError, match="boom"):
        cli_demo._server_module_origin(tmp_path, {})


def _agents_urlopen(agents):
    def fake_urlopen(req, timeout=5):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/auth/key"):
            return BytesIO(json.dumps({"key": "k"}).encode())
        return BytesIO(json.dumps({"agents": agents}).encode())
    return fake_urlopen


def test_verify_served_identity_requires_exactly_the_fresh_agent(monkeypatch):
    url = "http://127.0.0.1:8910"
    monkeypatch.setattr(
        cli_demo.urllib.request, "urlopen",
        _agents_urlopen([{"id": _SMOKE_DID, "is_demo": True}]),
    )
    assert cli_demo._verify_served_identity(url, _SMOKE_DID) is None

    monkeypatch.setattr(
        cli_demo.urllib.request, "urlopen",
        _agents_urlopen([{"id": "did:web:localhost:other", "is_demo": True}]),
    )
    assert "did:web:localhost:other" in cli_demo._verify_served_identity(url, _SMOKE_DID)

    monkeypatch.setattr(
        cli_demo.urllib.request, "urlopen",
        _agents_urlopen([
            {"id": _SMOKE_DID, "is_demo": True},
            {"id": "did:web:localhost:other", "is_demo": True},
        ]),
    )
    assert cli_demo._verify_served_identity(url, _SMOKE_DID) is not None


def _patch_smoke_lifecycle(monkeypatch, *, playwright_rc=0, served_did=_SMOKE_DID,
                           module_origin=None, playwright_raises=None):
    """Mock every external effect of `kestrel demo smoke`: the setup script
    (which writes the inception manifest), the module-origin probe, uvicorn,
    /health, /api/agents, and Playwright."""
    state: dict = {
        "setup_calls": [], "setup_env": None, "playwright_calls": [],
        "playwright_env": None, "playwright_cwd": None, "server_cwd": None,
        "server_env": None, "started": False, "stopped": False,
        "pid_file_during_run": None, "manifest_during_run": None,
    }
    monkeypatch.setattr(cli_demo, "_port_is_busy", lambda port: False)
    monkeypatch.setattr(cli_demo, "_verify_only_demo_agents", lambda url: None)
    monkeypatch.setattr(
        cli_demo, "_verify_served_identity",
        lambda url, did: None if did == served_did else f"server reports {[served_did]}",
    )
    package_dir = Path(cli_demo.__file__).resolve().parent
    monkeypatch.setattr(
        cli_demo, "_server_module_origin",
        lambda cwd, env: module_origin or package_dir,
    )

    fake_proc = MagicMock()
    fake_proc.pid = 4242
    fake_proc.poll.return_value = None

    def fake_start(cmd, cwd=None, env=None, stdout=None, stderr=None):
        state.update(started=True, server_cwd=cwd, server_env=dict(env))
        return fake_proc

    monkeypatch.setattr(cli_demo, "start_background_process", fake_start)
    monkeypatch.setattr(
        cli_demo, "wait_for_health", lambda port, timeout=60.0, proc=None: True,
    )
    monkeypatch.setattr(
        cli_demo, "stop_process",
        lambda proc, timeout=10.0: state.update(stopped=True),
    )

    def fake_run_streaming(cmd, *, cwd=None, env=None, check=False):
        argv = list(cmd)
        if "setup_demo_agent.py" in " ".join(argv):
            state["setup_calls"].append(argv)
            state["setup_env"] = dict(env)
            data_dir = Path(argv[argv.index("--data-dir") + 1])
            Path(argv[argv.index("--manifest") + 1]).write_text(json.dumps({
                "agent_name": "Kestrel Demo Agent",
                "agent_did": _SMOKE_DID,
                "db_path": str(data_dir / "kestrel_prime.db"),
                "data_dir": str(data_dir),
            }))
            return 0
        if argv[:3] == ["npx", "playwright", "test"]:
            state["playwright_calls"].append(argv)
            state["playwright_env"] = dict(env)
            state["playwright_cwd"] = cwd
            home = Path(env["KESTREL_HOME"])
            pid_file = home / cli_demo.SMOKE_PID_NAME
            state["pid_file_during_run"] = pid_file.read_text() if pid_file.exists() else None
            manifest_path = Path(env["KESTREL_CONSOLE_SMOKE_MANIFEST"])
            state["manifest_during_run"] = json.loads(manifest_path.read_text())
            if playwright_raises is not None:
                raise playwright_raises
            return playwright_rc
        return 0

    monkeypatch.setattr(cli_demo, "run_streaming", fake_run_streaming)
    return state


def _smoke_args(home: Path, **overrides):
    values = dict(demo_command="smoke", home=str(home), port=None, keep_server=False)
    values.update(overrides)
    return _Args(**values)


def test_cmd_demo_smoke_happy_path_proves_origin_and_tears_down(monkeypatch, tmp_path):
    monkeypatch.setenv("KESTREL_API_KEY", "production-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-prod")
    home = tmp_path / "home"
    state = _patch_smoke_lifecycle(monkeypatch)

    rc = cli_demo.cmd_demo(_smoke_args(home))

    assert rc == 0
    home = home.resolve()
    data_dir = home / "agent_data" / "console-smoke"
    # A fresh local-LLM-only agent is created inside the home.
    (setup,) = state["setup_calls"]
    assert setup[setup.index("--data-dir") + 1] == str(data_dir)
    assert "--local-llm-only" in setup
    assert state["setup_env"]["KESTREL_DB_PATH"] == str(data_dir)
    # The server runs from the fresh home, never the checkout.
    assert state["server_cwd"] == home
    sv = state["server_env"]
    assert sv["KESTREL_HOME"] == str(home)
    assert sv["KESTREL_DEMO_SERVER"] == "1"
    assert "KESTREL_API_KEY" not in sv and "ANTHROPIC_API_KEY" not in sv
    # Setup and server share the throwaway data key, or the server could not
    # decrypt the identity setup created.
    assert sv["KESTREL_DATA_KEY"] == state["setup_env"]["KESTREL_DATA_KEY"]
    # Playwright runs exactly the CI smoke project from the checkout.
    (pw,) = state["playwright_calls"]
    assert pw[pw.index("--project") + 1] == "console-smoke"
    assert pw[pw.index("--config") + 1] == "tests/e2e/playwright.config.cjs"
    assert state["playwright_cwd"] == cli_demo._repo_root()
    pw_env = state["playwright_env"]
    assert pw_env["KESTREL_URL"] == "http://127.0.0.1:8910"
    assert pw_env["KESTREL_DB_PATH"] == str(data_dir)
    assert "KESTREL_DATA_KEY" not in pw_env
    assert "KESTREL_API_KEY" not in pw_env and "ANTHROPIC_API_KEY" not in pw_env
    # The spec receives the proven origin facts while the server runs.
    assert state["manifest_during_run"] == {
        "base_url": "http://127.0.0.1:8910",
        "port": 8910,
        "home": str(home),
        "data_dir": str(data_dir),
        "db_path": str(data_dir / "kestrel_prime.db"),
        "agent_did": _SMOKE_DID,
        "agent_name": "Kestrel Demo Agent",
        "module_origin": str(Path(cli_demo.__file__).resolve().parent),
        "server_pid": 4242,
    }
    assert state["pid_file_during_run"] == "4242\n"
    # Teardown: server stopped and the PID file the CI step reads is gone.
    assert state["stopped"] is True
    assert not (home / cli_demo.SMOKE_PID_NAME).exists()


def test_cmd_demo_smoke_failure_still_tears_down(monkeypatch, tmp_path):
    home = tmp_path / "home"
    state = _patch_smoke_lifecycle(monkeypatch, playwright_rc=1)
    assert cli_demo.cmd_demo(_smoke_args(home)) == 1
    assert state["stopped"] is True
    assert not (home / cli_demo.SMOKE_PID_NAME).exists()


def test_cmd_demo_smoke_cancellation_still_tears_down(monkeypatch, tmp_path):
    """A cancelled run (SIGTERM → SystemExit, Ctrl-C → KeyboardInterrupt)
    must still stop the server and clear the PID file."""
    home = tmp_path / "home"
    state = _patch_smoke_lifecycle(monkeypatch, playwright_raises=SystemExit(143))
    with pytest.raises(SystemExit):
        cli_demo.cmd_demo(_smoke_args(home))
    assert state["stopped"] is True
    assert not (home / cli_demo.SMOKE_PID_NAME).exists()


def test_cmd_demo_smoke_keep_server_leaves_pid_for_the_operator(monkeypatch, tmp_path):
    home = tmp_path / "home"
    state = _patch_smoke_lifecycle(monkeypatch)
    assert cli_demo.cmd_demo(_smoke_args(home, keep_server=True)) == 0
    assert state["stopped"] is False
    assert (home / cli_demo.SMOKE_PID_NAME).read_text() == "4242\n"


def test_cmd_demo_smoke_refuses_a_server_from_another_checkout(monkeypatch, tmp_path, capsys):
    state = _patch_smoke_lifecycle(
        monkeypatch, module_origin=Path("/srv/primary-checkout/kestrel_sovereign"),
    )
    assert cli_demo.cmd_demo(_smoke_args(tmp_path / "home")) == 1
    assert "/srv/primary-checkout/kestrel_sovereign" in capsys.readouterr().err
    assert state["started"] is False
    assert state["playwright_calls"] == []


def test_cmd_demo_smoke_refuses_a_server_serving_another_database(monkeypatch, tmp_path, capsys):
    state = _patch_smoke_lifecycle(monkeypatch, served_did="did:web:localhost:other")
    home = tmp_path / "home"
    assert cli_demo.cmd_demo(_smoke_args(home)) == 1
    assert "did:web:localhost:other" in capsys.readouterr().err
    assert state["playwright_calls"] == []
    assert state["stopped"] is True
    assert not (home / cli_demo.SMOKE_MANIFEST_NAME).exists()


def test_cmd_demo_smoke_setup_failure_starts_nothing(monkeypatch, tmp_path):
    state = _patch_smoke_lifecycle(monkeypatch)
    monkeypatch.setattr(cli_demo, "run_streaming", lambda cmd, **kw: 1)
    assert cli_demo.cmd_demo(_smoke_args(tmp_path / "home")) == 1
    assert state["started"] is False


def test_cmd_demo_smoke_converts_sigterm_while_the_server_runs(monkeypatch, tmp_path):
    """CI cancellation sends SIGTERM. While the server runs, SIGTERM must
    unwind into the teardown instead of killing the runner outright."""
    state = _patch_smoke_lifecycle(monkeypatch)
    inner = cli_demo.run_streaming
    seen = {}

    def recording_run_streaming(cmd, **kwargs):
        if list(cmd)[:2] == ["npx", "playwright"]:
            seen["handler"] = signal.getsignal(signal.SIGTERM)
        return inner(cmd, **kwargs)

    monkeypatch.setattr(cli_demo, "run_streaming", recording_run_streaming)
    before = signal.getsignal(signal.SIGTERM)

    assert cli_demo.cmd_demo(_smoke_args(tmp_path / "home")) == 0

    assert callable(seen["handler"]) and seen["handler"] is not before
    with pytest.raises(SystemExit):
        seen["handler"](signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) is before
    assert state["stopped"] is True


# The smoke server reads no .env file (PR #3453 review). ``server.py`` loads
# its .env files with ``override=False``, which fills in exactly the variables
# that are absent -- so the scrubbing in ``_build_smoke_env`` alone let a legacy
# ``kestrel_sovereign/.env`` put the operator's ``KESTREL_API_KEY`` and
# ``KESTREL_EXPECTED_DID`` back into the "isolated" instance.

_PLANTED_PACKAGE_DOTENV = {
    "KESTREL_API_KEY": "production-from-package",
    "KESTREL_EXPECTED_DID": "did:web:production:agent",
}
_PLANTED_HOME_DOTENV = {
    "ANTHROPIC_API_KEY": "sk-ant-from-home",
    "KESTREL_SOVEREIGN_TRUST_ROOT_PATH": "/srv/live/trust-root.json",
}


def _plant_dotenv(directory: Path, values: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".env").write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )


@pytest.fixture
def server_process_environ(monkeypatch):
    """Give the test process exactly a server child's environment and cwd.

    ``load_dotenv`` writes ``os.environ`` directly, which ``monkeypatch`` does
    not track, so the real environment is restored wholesale afterwards.
    """
    from kestrel_sovereign import paths

    before = dict(os.environ)

    def _become(env: dict, cwd: Path) -> None:
        os.environ.clear()
        os.environ.update(env)
        monkeypatch.chdir(cwd)
        paths.reset_cache()

    yield _become
    os.environ.clear()
    os.environ.update(before)
    paths.reset_cache()


def _smoke_server_launch(monkeypatch, tmp_path):
    """The environment and cwd `kestrel demo smoke` hands its server, with a
    legacy package-level .env and a home/cwd .env planted for it to find."""
    state = _patch_smoke_lifecycle(monkeypatch)
    assert cli_demo.cmd_demo(_smoke_args(tmp_path / "home")) == 0
    package = tmp_path / "package"
    _plant_dotenv(package, _PLANTED_PACKAGE_DOTENV)
    # The server's cwd is the home, which is also KESTREL_HOME.
    _plant_dotenv(state["server_cwd"], _PLANTED_HOME_DOTENV)
    return state["server_env"], state["server_cwd"], package


def test_smoke_server_loads_no_dotenv_not_even_the_package_one(
    monkeypatch, tmp_path, server_process_environ,
):
    from kestrel_sovereign import server

    env, cwd, package = _smoke_server_launch(monkeypatch, tmp_path)
    server_process_environ(env, cwd)

    server.load_server_dotenv(package_dir=package)

    leaked = {
        key: os.environ[key]
        for key in (*_PLANTED_PACKAGE_DOTENV, *_PLANTED_HOME_DOTENV)
        if key in os.environ
    }
    assert leaked == {}, f"the smoke server loaded a .env file: {leaked}"
    assert dict(os.environ) == env


def test_without_the_opt_out_the_planted_dotenvs_do_reach_the_server(
    monkeypatch, tmp_path, server_process_environ,
):
    """Control for the test above: the planted files are ones the server's
    loader really reads, so their absence there is the opt-out's doing."""
    from kestrel_sovereign import server
    from kestrel_sovereign.paths import SKIP_DOTENV_ENV

    env, cwd, package = _smoke_server_launch(monkeypatch, tmp_path)
    del env[SKIP_DOTENV_ENV]
    server_process_environ(env, cwd)

    server.load_server_dotenv(package_dir=package)

    for key, value in {**_PLANTED_PACKAGE_DOTENV, **_PLANTED_HOME_DOTENV}.items():
        assert os.environ.get(key) == value, key


@pytest.mark.parametrize("value", ["true", "yes", "2", " 1"])
def test_server_refuses_an_unrecognised_dotenv_opt_out(
    monkeypatch, tmp_path, server_process_environ, value,
):
    """A misspelled opt-out must not silently mean "load the files"."""
    from kestrel_sovereign import server
    from kestrel_sovereign.paths import SKIP_DOTENV_ENV

    env, cwd, package = _smoke_server_launch(monkeypatch, tmp_path)
    env[SKIP_DOTENV_ENV] = value
    server_process_environ(env, cwd)

    with pytest.raises(ValueError, match=SKIP_DOTENV_ENV):
        server.load_server_dotenv(package_dir=package)
    assert dict(os.environ) == env


def test_server_reads_dotenv_files_only_through_its_opt_out_aware_loader():
    """The opt-out is only as good as its wiring: every ``load_dotenv`` call in
    ``server.py`` must sit inside ``load_server_dotenv``, and the module must
    run that loader at import, which is when uvicorn loads ``server:app``."""
    import ast

    tree = ast.parse(
        (Path(cli_demo.__file__).resolve().parent / "server.py").read_text(
            encoding="utf-8"
        )
    )

    def called_name(node: ast.AST) -> str:
        if not isinstance(node, ast.Call):
            return ""
        func = node.func
        return getattr(func, "id", None) or getattr(func, "attr", "")

    (loader,) = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_server_dotenv"
    ]
    inside_loader = {id(node) for node in ast.walk(loader)}
    dotenv_calls = [
        node for node in ast.walk(tree) if called_name(node) == "load_dotenv"
    ]
    assert dotenv_calls, "server.py no longer loads any .env file"
    assert all(id(call) in inside_loader for call in dotenv_calls)
    assert any(
        isinstance(node, ast.Expr) and called_name(node.value) == "load_server_dotenv"
        for node in tree.body
    ), "server.py no longer runs load_server_dotenv() at import"


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal delivery")
def test_sigterm_raises_exit_so_teardown_runs():
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as excinfo:
        with cli_demo._sigterm_raises_exit():
            os.kill(os.getpid(), signal.SIGTERM)
            # Delivery is asynchronous; give the interpreter a bytecode boundary.
            for _ in range(1000):
                pass
    assert excinfo.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is previous
