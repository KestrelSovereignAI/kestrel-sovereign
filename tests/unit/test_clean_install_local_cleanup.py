"""The local wheel rehearsal must never kill strangers or orphan its venv."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

from kestrel_sovereign.multi_agent.process_manager import PidRecord, PidStatus

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "ci" / "clean_install_local_cleanup.py"
_SPEC = importlib.util.spec_from_file_location("clean_install_cleanup_under_test", _SCRIPT)
assert _SPEC and _SPEC.loader
cleanup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cleanup)


def _record(status: PidStatus, root: Path, port: int) -> PidRecord:
    return PidRecord(status, 12345, str(root), port, "test", started_at=100.0)


def test_failed_start_terminates_only_fenced_harness_pid(tmp_path, monkeypatch):
    records = iter(
        [_record(PidStatus.LIVE, tmp_path, 62220),
         _record(PidStatus.STALE, tmp_path, 62220),
         _record(PidStatus.STALE, tmp_path, 62220)]
    )
    signals = []
    monkeypatch.setattr(cleanup.ProcessManager, "read_pid_record", lambda _: next(records))
    monkeypatch.setattr(
        cleanup.ProcessManager,
        "kill_process",
        lambda *args, **kwargs: signals.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(cleanup.ProcessManager, "is_port_in_use", lambda *_: False)

    assert cleanup.stop_owned_process(tmp_path / "agent.pid", tmp_path, 62220)
    assert signals == [((12345,), {"started_at": 100.0})]


def test_forced_stop_waits_for_asynchronous_exit(tmp_path, monkeypatch):
    records = iter(
        [_record(PidStatus.LIVE, tmp_path, 62220),
         _record(PidStatus.LIVE, tmp_path, 62220)]
    )
    signals = []
    waits = iter([False, True])
    monkeypatch.setattr(cleanup.ProcessManager, "read_pid_record", lambda _: next(records))
    monkeypatch.setattr(cleanup, "_wait_for_exit", lambda _path: next(waits))
    monkeypatch.setattr(
        cleanup.ProcessManager,
        "kill_process",
        lambda *args, **kwargs: signals.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(cleanup.ProcessManager, "is_port_in_use", lambda *_: False)

    assert cleanup.stop_owned_process(tmp_path / "agent.pid", tmp_path, 62220)
    assert signals == [
        ((12345,), {"started_at": 100.0}),
        ((12345,), {"force": True, "started_at": 100.0}),
    ]


def test_wait_for_exit_polls_live_then_stale(tmp_path, monkeypatch):
    statuses = iter([PidStatus.LIVE, PidStatus.STALE])
    clock = iter([0.0, 0.1, 0.2])
    sleeps = []
    monkeypatch.setattr(cleanup.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(cleanup.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        cleanup.ProcessManager,
        "read_pid_record",
        lambda _: _record(next(statuses), tmp_path, 62220),
    )
    assert cleanup._wait_for_exit(tmp_path / "agent.pid")
    assert sleeps == [0.1]


def test_mismatched_pid_record_is_not_signalled_or_cleaned(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cleanup.ProcessManager,
        "read_pid_record",
        lambda _: _record(PidStatus.LIVE, tmp_path / "another", 62220),
    )
    monkeypatch.setattr(
        cleanup.ProcessManager,
        "kill_process",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("must not kill")),
    )
    assert not cleanup.stop_owned_process(tmp_path / "agent.pid", tmp_path, 62220)


def test_missing_pid_but_busy_port_preserves_harness(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cleanup.ProcessManager,
        "read_pid_record",
        lambda _: _record(PidStatus.ABSENT, tmp_path, 62220),
    )
    monkeypatch.setattr(cleanup.ProcessManager, "is_port_in_use", lambda *_: True)
    assert not cleanup.stop_owned_process(tmp_path / "agent.pid", tmp_path, 62220)


def test_inherited_database_url_is_rejected_before_setup(tmp_path):
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("KESTREL_") and key != "DATABASE_URL"
    }
    env["KESTREL_DATABASE_URL"] = "postgresql://example.invalid/no-connect"
    result = subprocess.run(
        ["bash", str(_ROOT / "scripts" / "ci" / "clean_install_local.sh"), "wheel"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "unset KESTREL_DATABASE_URL" in result.stderr
    assert "Setup wizard" not in result.stdout
