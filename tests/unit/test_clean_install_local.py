"""The local clean-install rehearsal refuses unsafe state and cleans up fail-closed."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "clean_install_local.py"
_SPEC = importlib.util.spec_from_file_location("clean_install_local_under_test", _SCRIPT)
assert _SPEC and _SPEC.loader
local = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = local
_SPEC.loader.exec_module(local)


# -- refusal -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["KESTREL_HOME", "KESTREL_DATABASE_URL", "DATABASE_URL"])
def test_environment_that_could_reach_a_live_install_is_refused(tmp_path, name):
    with pytest.raises(local.RehearsalRefused, match=f"unset {name}"):
        local.refuse_unsafe(tmp_path, {name: "x"})


def test_the_runtime_selector_is_the_one_accepted_kestrel_variable(tmp_path):
    local.refuse_unsafe(tmp_path, {"KESTREL_PY_VERSION": "3.12", "PATH": "/bin"})


@pytest.mark.parametrize("state", local.STATE_PATHS)
def test_an_existing_install_is_refused(tmp_path, state):
    (tmp_path / state).write_text("", encoding="utf-8")
    with pytest.raises(local.RehearsalRefused, match=state):
        local.refuse_unsafe(tmp_path, {})


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privilege on Windows")
def test_a_dangling_state_symlink_is_refused(tmp_path):
    (tmp_path / ".env").symlink_to(tmp_path / "missing")
    with pytest.raises(local.RehearsalRefused, match=".env"):
        local.refuse_unsafe(tmp_path, {})


# -- child environment -----------------------------------------------------------


def test_child_env_drops_credentials_and_pins_the_rehearsal_settings(tmp_path):
    env = local.child_env(
        {"PATH": "/bin", "ANTHROPIC_API_KEY": "sk", "AGENT_DATA_DIR": "/data",
         "KESTREL_TEST_INSTANCE": "0"},
        tmp_path,
    )
    assert "ANTHROPIC_API_KEY" not in env
    assert "AGENT_DATA_DIR" not in env
    assert env["PATH"] == "/bin"
    assert env["KESTREL_TEST_INSTANCE"] == "1"
    assert env["KESTREL_AUDIT_MODE"] == "skip"
    assert env["KESTREL_HOST_DB_PATH"] == str(tmp_path / "host-features.db")


def test_venv_executable_matches_this_platform(tmp_path):
    python = local.venv_executable(tmp_path, "python")
    if os.name == "nt":
        assert python == tmp_path / "Scripts" / "python.exe"
    else:
        assert python == tmp_path / "bin" / "python"


# -- cleanup ---------------------------------------------------------------------


def _rehearsal(tmp_path: Path) -> "local.Rehearsal":
    harness = tmp_path / "harness"
    harness.mkdir()
    return local.Rehearsal(tmp_path, "sync", "3.13", {}, harness)


def test_harness_is_removed_when_no_port_was_ever_assigned(tmp_path, monkeypatch):
    rehearsal = _rehearsal(tmp_path)
    monkeypatch.setattr(local.subprocess, "run", pytest.fail)
    assert rehearsal.cleanup() is True
    assert not rehearsal.harness_dir.exists()


@pytest.mark.parametrize("probe_rc, removed", [(0, True), (1, False)])
def test_harness_survives_an_inconclusive_port_check(tmp_path, monkeypatch, probe_rc, removed):
    rehearsal = _rehearsal(tmp_path)
    rehearsal.ports_ready = True
    rehearsal.python = ["python"]
    calls = []

    class _Done:
        returncode = probe_rc

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _Done()

    monkeypatch.setattr(local.subprocess, "run", fake_run)
    assert rehearsal.cleanup() is removed
    assert rehearsal.harness_dir.exists() is not removed
    assert calls and "clean_install_local_cleanup.py" in calls[0][1]


def test_a_failed_step_names_the_command(tmp_path, monkeypatch):
    rehearsal = _rehearsal(tmp_path)

    class _Failed:
        returncode = 3

    monkeypatch.setattr(local.subprocess, "run", lambda *a, **k: _Failed())
    with pytest.raises(local.StepFailed, match="uv sync exited 3"):
        rehearsal.run(["uv", "sync"])


def test_main_refuses_before_creating_a_harness(monkeypatch, capsys):
    for name in list(os.environ):
        if name.startswith("KESTREL_") or name == "DATABASE_URL":
            monkeypatch.delenv(name)
    monkeypatch.setenv("KESTREL_HOME", "/somewhere")
    monkeypatch.setattr(local.tempfile, "mkdtemp", pytest.fail)
    assert local.main(["wheel"]) == 2
    assert "unset KESTREL_HOME" in capsys.readouterr().err
