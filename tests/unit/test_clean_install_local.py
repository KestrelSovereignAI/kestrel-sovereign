"""The local clean-install rehearsal refuses unsafe state and cleans up fail-closed."""

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
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
    monkeypatch.setattr(local, "run_child", pytest.fail)
    assert rehearsal.cleanup() is True
    assert not rehearsal.harness_dir.exists()


@pytest.mark.parametrize("probe_rc, removed", [(0, True), (1, False)])
def test_harness_survives_an_inconclusive_port_check(tmp_path, monkeypatch, probe_rc, removed):
    rehearsal = _rehearsal(tmp_path)
    rehearsal.ports_ready = True
    rehearsal.python = ["python"]
    calls = []

    def fake_run_child(argv, **kwargs):
        calls.append(argv)
        return probe_rc

    monkeypatch.setattr(local, "run_child", fake_run_child)
    assert rehearsal.cleanup() is removed
    assert rehearsal.harness_dir.exists() is not removed
    assert calls and "clean_install_local_cleanup.py" in calls[0][1]


def test_an_interrupt_during_the_cleanup_probe_keeps_and_names_the_harness(
    tmp_path, monkeypatch, capsys
):
    rehearsal = _rehearsal(tmp_path)
    rehearsal.ports_ready = True
    rehearsal.python = ["python"]

    def interrupted(argv, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(local, "run_child", interrupted)
    with pytest.raises(KeyboardInterrupt):
        rehearsal.cleanup()
    assert rehearsal.harness_dir.exists()
    assert str(rehearsal.harness_dir) in capsys.readouterr().err


def test_a_failed_step_names_the_command(tmp_path, monkeypatch):
    rehearsal = _rehearsal(tmp_path)
    monkeypatch.setattr(local, "run_child", lambda *a, **k: 3)
    with pytest.raises(local.StepFailed, match="uv sync exited 3"):
        rehearsal.run(["uv", "sync"])


# -- interrupts and termination (review round 1) ----------------------------------


class _InterruptedOnce:
    """A child whose first wait() is interrupted, as Ctrl-C does to the parent."""

    def __init__(self, interrupt: BaseException):
        self.interrupt = interrupt
        self.waits = 0
        self.killed = False

    def wait(self, timeout=None):
        self.waits += 1
        if self.waits == 1:
            raise self.interrupt
        return 0

    def kill(self):
        self.killed = True


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(143)])
def test_the_running_step_finishes_before_an_interrupt_propagates(
    tmp_path, monkeypatch, interrupt
):
    child = _InterruptedOnce(interrupt)
    monkeypatch.setattr(local.subprocess, "Popen", lambda *a, **k: child)
    with pytest.raises(type(interrupt)):
        local.run_child(["verify"], cwd=tmp_path, env={})
    assert child.waits == 2
    assert child.killed is False


def test_a_termination_signal_becomes_system_exit_with_the_shell_status():
    with pytest.raises(SystemExit) as raised:
        local._raise_exit(signal.SIGTERM, None)
    assert raised.value.code == 128 + signal.SIGTERM


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal delivery")
def test_sigterm_mid_install_still_removes_the_harness(tmp_path):
    """End to end: the runner itself, a ``uv`` that hangs, then SIGTERM."""
    root = tmp_path / "checkout"
    (root / "scripts" / "ci").mkdir(parents=True)
    shutil.copy(_SCRIPT, root / "scripts" / "ci" / "clean_install_local.py")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n", encoding="utf-8")
    uv.chmod(0o755)
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("KESTREL_") and k != "DATABASE_URL"
    }
    env.update(PATH=f"{fake_bin}{os.pathsep}{env.get('PATH', '')}", TMPDIR=str(temp_root))
    proc = subprocess.Popen(
        [sys.executable, str(root / "scripts" / "ci" / "clean_install_local.py")],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    def fake_uv_pids() -> list[int]:
        found = subprocess.run(["pgrep", "-f", str(uv)], capture_output=True, text=True,
                               check=False).stdout.split()
        return [int(pid) for pid in found]

    try:
        deadline = time.monotonic() + 20
        while not fake_uv_pids():  # the runner is now waiting on its step
            assert time.monotonic() < deadline, "runner never started its first step"
            time.sleep(0.05)
        proc.send_signal(signal.SIGTERM)
        # The hanging uv is not this runner's to kill (bash would not kill it
        # either); stop it so the runner's wait returns.
        for pid in fake_uv_pids():
            os.kill(pid, signal.SIGTERM)
        assert proc.wait(timeout=20) == 128 + signal.SIGTERM
        assert not any(temp_root.glob("kestrel-clean-install-*"))
    finally:
        for pid in fake_uv_pids():
            os.kill(pid, signal.SIGKILL)
        if proc.poll() is None:
            proc.kill()


def test_a_signal_ignored_at_start_stays_ignored(monkeypatch):
    """Under nohup SIGHUP is ignored; a handler would make every step die of it."""
    installed: dict[int, object] = {}
    ignored = signal.SIGTERM  # present on every platform
    monkeypatch.setattr(local.signal, "getsignal",
                        lambda s: signal.SIG_IGN if s == ignored else signal.SIG_DFL)
    monkeypatch.setattr(local.signal, "signal", lambda s, h: installed.__setitem__(s, h))
    local.install_termination_handlers()
    assert ignored not in installed
    if hasattr(signal, "SIGHUP"):
        assert installed[signal.SIGHUP] is local._raise_exit


@pytest.mark.parametrize("value, expected", [("", "3.13"), ("3.12", "3.12")])
def test_an_empty_python_version_means_the_default(value, expected):
    assert local.python_version_from({"KESTREL_PY_VERSION": value}) == expected
    assert local.python_version_from({}) == "3.13"


def test_main_refuses_before_creating_a_harness(monkeypatch, capsys):
    for name in list(os.environ):
        if name.startswith("KESTREL_") or name == "DATABASE_URL":
            monkeypatch.delenv(name)
    monkeypatch.setenv("KESTREL_HOME", "/somewhere")
    monkeypatch.setattr(local.tempfile, "mkdtemp", pytest.fail)
    assert local.main(["wheel"]) == 2
    assert "unset KESTREL_HOME" in capsys.readouterr().err
