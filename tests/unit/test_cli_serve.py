"""``kestrel serve`` CLI tests — local model server launcher/registry.

Covers the testable core without launching real llama-server processes:
- gguf glob resolution (single file, sharded first-shard pick, ambiguous/empty errors)
- registry load + default_port propagation + path resolution order
- build_command (PATH binary vs absolute, kv-cache/reasoning/extra args)
- state file round-trip + stale-pidfile cleanup in running_model()
- memory-pressure refusal in _start (no real launch)
- one-model-at-a-time enforcement in cmd_up
- argparse wiring under the real ``kestrel`` parser
- run() dispatch guards (no subcommand, missing registry, unknown model)

Real serving is exercised manually / in the integration tier.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign import cli_serve


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Redirect state/log dirs to a temp location for every test."""
    state_dir = tmp_path / "state"
    monkeypatch.setattr(cli_serve, "STATE_DIR", state_dir)
    monkeypatch.setattr(cli_serve, "STATE_FILE", state_dir / "serve_state.json")
    monkeypatch.setattr(cli_serve, "LOG_DIR", state_dir / "logs")
    # Default: no registry env leakage
    monkeypatch.delenv("KESTREL_SERVE_REGISTRY", raising=False)
    yield


def _write_registry(tmp_path, body: str) -> Path:
    p = tmp_path / "serve_models.toml"
    p.write_text(body)
    return p


# --------------------------------------------------------------------------- #
# resolve_gguf
# --------------------------------------------------------------------------- #
def test_resolve_gguf_single_file(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_text("x")
    assert cli_serve.resolve_gguf(str(f)) == f


def test_resolve_gguf_picks_first_shard(tmp_path):
    for i in range(1, 5):
        (tmp_path / f"M-Q4-0000{i}-of-00004.gguf").write_text("x")
    got = cli_serve.resolve_gguf(str(tmp_path / "M-Q4-*.gguf"))
    assert got.name == "M-Q4-00001-of-00004.gguf"


def test_resolve_gguf_no_match_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli_serve.resolve_gguf(str(tmp_path / "nope-*.gguf"))


def test_resolve_gguf_ambiguous_raises(tmp_path):
    # Two non-shard files matching -> ambiguous
    (tmp_path / "a.gguf").write_text("x")
    (tmp_path / "b.gguf").write_text("x")
    with pytest.raises(ValueError):
        cli_serve.resolve_gguf(str(tmp_path / "*.gguf"))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_load_registry_applies_default_port(tmp_path):
    p = _write_registry(tmp_path, """
default_port = 9001
[models."a"]
gguf = "/x/a.gguf"
[models."b"]
gguf = "/x/b.gguf"
port = 1234
""")
    models = cli_serve.load_registry(p)
    assert models["a"]["port"] == 9001
    assert models["b"]["port"] == 1234


def test_resolve_registry_path_prefers_env(tmp_path, monkeypatch):
    p = _write_registry(tmp_path, 'default_port = 8001\n')
    monkeypatch.setenv("KESTREL_SERVE_REGISTRY", str(p))
    assert cli_serve.resolve_registry_path() == p


def test_resolve_registry_path_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no serve_models.toml here
    assert cli_serve.resolve_registry_path() is None


# --------------------------------------------------------------------------- #
# build_command
# --------------------------------------------------------------------------- #
def test_build_command_path_binary_and_flags(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_text("x")
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    entry = {
        "gguf": str(gguf), "binary": str(binary), "ctx_size": 4096,
        "kv_cache_type": "q8_0", "reasoning_format": "deepseek",
        "extra_args": ["--flash-attn"], "port": 8001,
    }
    cmd = cli_serve.build_command("m", entry, 8001)
    assert cmd[0] == str(binary)
    assert "--model" in cmd and str(gguf) in cmd
    assert cmd[cmd.index("--ctx-size") + 1] == "4096"
    assert "--cache-type-k" in cmd and "--cache-type-v" in cmd
    assert cmd[cmd.index("--reasoning-format") + 1] == "deepseek"
    assert "--flash-attn" in cmd


def test_build_command_path_binary_must_be_executable(tmp_path):
    gguf = tmp_path / "m.gguf"; gguf.write_text("x")
    binary = tmp_path / "not-exec"; binary.write_text("x")  # no +x
    with pytest.raises(FileNotFoundError):
        cli_serve.build_command("m", {"gguf": str(gguf), "binary": str(binary)}, 8001)


def test_build_command_bare_binary_resolved_on_path(tmp_path, monkeypatch):
    gguf = tmp_path / "m.gguf"; gguf.write_text("x")
    monkeypatch.setattr(cli_serve.shutil, "which", lambda b: "/usr/bin/" + b)
    cmd = cli_serve.build_command("m", {"gguf": str(gguf), "binary": "llama-server"}, 8001)
    assert cmd[0] == "/usr/bin/llama-server"


# --------------------------------------------------------------------------- #
# state / process
# --------------------------------------------------------------------------- #
def test_state_round_trip_and_clear():
    assert cli_serve.read_state() is None
    cli_serve.write_state({"name": "x", "pid": 1, "port": 8001})
    assert cli_serve.read_state()["name"] == "x"
    cli_serve.clear_state()
    assert cli_serve.read_state() is None


def test_running_model_clears_stale_pidfile(monkeypatch):
    cli_serve.write_state({"name": "x", "pid": 999999, "port": 8001})
    monkeypatch.setattr(cli_serve, "pid_alive", lambda pid: False)
    assert cli_serve.running_model() is None
    assert cli_serve.read_state() is None  # cleaned up


def test_running_model_returns_live(monkeypatch):
    cli_serve.write_state({
        "name": "x", "pid": 123, "port": 8001,
        "cmd": ["/opt/homebrew/bin/llama-server", "--model", "/m/x.gguf"],
    })
    monkeypatch.setattr(cli_serve, "pid_alive", lambda pid: True)
    monkeypatch.setattr(cli_serve, "_process_cmdline",
                        lambda pid: "/opt/homebrew/bin/llama-server --model /m/x.gguf --port 8001")
    assert cli_serve.running_model()["name"] == "x"


def test_running_model_rejects_reused_pid(monkeypatch):
    # PID alive, but it's a DIFFERENT process (PID reuse) — must not be treated as ours.
    cli_serve.write_state({
        "name": "x", "pid": 123, "port": 8001,
        "cmd": ["/opt/homebrew/bin/llama-server", "--model", "/m/x.gguf"],
    })
    monkeypatch.setattr(cli_serve, "pid_alive", lambda pid: True)
    monkeypatch.setattr(cli_serve, "_process_cmdline", lambda pid: "/usr/sbin/cupsd")
    assert cli_serve.running_model() is None
    assert cli_serve.read_state() is None  # stale state reaped


def test_state_matches_process_requires_model_path(monkeypatch):
    state = {"cmd": ["/x/llama-server", "--model", "/m/v4.gguf"]}
    monkeypatch.setattr(cli_serve, "_process_cmdline",
                        lambda pid: "/x/llama-server --model /m/OTHER.gguf")
    assert cli_serve._state_matches_process(state, 1) is False
    monkeypatch.setattr(cli_serve, "_process_cmdline",
                        lambda pid: "/x/llama-server --model /m/v4.gguf --port 8001")
    assert cli_serve._state_matches_process(state, 1) is True


# --------------------------------------------------------------------------- #
# _start memory refusal + cmd_up enforcement
# --------------------------------------------------------------------------- #
def test_start_refuses_when_insufficient_ram(monkeypatch):
    monkeypatch.setattr(cli_serve, "memory_status", lambda: (512.0, 50.0, "GREEN"))
    popen = MagicMock()
    monkeypatch.setattr(cli_serve.subprocess, "Popen", popen)
    rc = cli_serve._start("big", {"gguf": "/x", "est_ram_gb": 300}, 8001,
                          timeout=1, wait=False, force=False)
    assert rc == 1
    popen.assert_not_called()  # never launched


def test_start_force_overrides_ram(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_serve, "memory_status", lambda: (512.0, 50.0, "GREEN"))
    monkeypatch.setattr(cli_serve, "build_command", lambda *a: ["/bin/true"])
    proc = SimpleNamespace(pid=4242)
    monkeypatch.setattr(cli_serve.subprocess, "Popen", lambda *a, **k: proc)
    rc = cli_serve._start("big", {"gguf": "/x", "est_ram_gb": 300}, 8001,
                          timeout=1, wait=False, force=True)
    assert rc == 0
    assert cli_serve.read_state()["pid"] == 4242


def test_cmd_up_refuses_when_already_running(monkeypatch):
    monkeypatch.setattr(cli_serve, "running_model",
                        lambda: {"name": "other", "pid": 7, "port": 8001})
    args = SimpleNamespace(name="m", port=None, timeout=1, no_wait=True, force=False)
    rc = cli_serve.cmd_up({"m": {"gguf": "/x"}}, args)
    assert rc == 1


def test_get_entry_unknown_raises():
    with pytest.raises(SystemExit):
        cli_serve._get_entry({"a": {}}, "missing")


# --------------------------------------------------------------------------- #
# argparse wiring + run() dispatch
# --------------------------------------------------------------------------- #
def test_subparser_registers_under_kestrel_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    cli_serve.add_serve_subparser(sub)
    args = parser.parse_args(["serve", "up", "glm-5.2", "--no-wait", "--force"])
    assert args.command == "serve"
    assert args.serve_command == "up"
    assert args.name == "glm-5.2"
    assert args.no_wait is True and args.force is True


def test_run_no_subcommand_returns_1(capsys):
    rc = cli_serve.run(argparse.Namespace(serve_command=None))
    assert rc == 1


def test_run_missing_registry_returns_1(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rc = cli_serve.run(argparse.Namespace(serve_command="list"))
    assert rc == 1


def test_run_status_works_without_registry(monkeypatch, tmp_path, capsys):
    # Registry absent, but status must still work from the state file.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_serve, "memory_status", lambda: (512.0, 400.0, "GREEN"))
    monkeypatch.setattr(cli_serve, "running_model", lambda: None)
    rc = cli_serve.run(argparse.Namespace(serve_command="status"))
    assert rc == 0
    assert "Memory:" in capsys.readouterr().out


def test_run_down_works_without_registry(monkeypatch, tmp_path):
    # Registry absent; down must still be able to stop a managed model.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_serve, "running_model", lambda: None)
    rc = cli_serve.run(argparse.Namespace(serve_command="down", timeout=1))
    assert rc == 0  # nothing running, but did not error on missing registry


def test_switch_validates_target_before_stopping_current(monkeypatch, tmp_path):
    # Target binary is non-executable -> switch must refuse WITHOUT stopping current.
    gguf = tmp_path / "m.gguf"; gguf.write_text("x")
    bad_binary = tmp_path / "not-exec"; bad_binary.write_text("x")  # no +x
    stop = MagicMock()
    monkeypatch.setattr(cli_serve, "_stop", stop)
    args = SimpleNamespace(name="m", port=None, timeout=1, no_wait=True, force=False)
    rc = cli_serve.cmd_switch({"m": {"gguf": str(gguf), "binary": str(bad_binary)}}, args)
    assert rc == 1
    stop.assert_not_called()  # current model left running


def test_run_list_dispatches(monkeypatch, tmp_path, capsys):
    p = _write_registry(tmp_path, 'default_port=8001\n[models."glm-5.2"]\ngguf="/x/g.gguf"\nest_ram_gb=300\n')
    monkeypatch.setenv("KESTREL_SERVE_REGISTRY", str(p))
    monkeypatch.setattr(cli_serve, "running_model", lambda: None)
    rc = cli_serve.run(argparse.Namespace(serve_command="list"))
    assert rc == 0
    assert "glm-5.2" in capsys.readouterr().out
