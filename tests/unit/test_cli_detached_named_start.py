"""Named fire-and-exit launches must retain logs after their parent exits."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tempfile

import psutil


def _recorded_probe_process(record):
    """Only validated birth identity may become a cleanup-owned process."""
    try:
        candidate = psutil.Process(record["pid"])
        if abs(candidate.create_time() - record["started_at"]) < 0.01:
            return candidate
    except psutil.NoSuchProcess:
        pass
    return None


def test_reused_probe_pid_does_not_become_cleanup_owned(monkeypatch):
    from unittest.mock import Mock

    replacement = Mock()
    replacement.create_time.return_value = 200.0
    monkeypatch.setattr(psutil, "Process", Mock(return_value=replacement))
    assert _recorded_probe_process({"pid": 123, "started_at": 100.0}) is None
    replacement.send_signal.assert_not_called()


def test_named_cli_retains_real_child_output_after_launcher_exit(tmp_path):
    """Use actual named-start wiring and native fds, without an agent/LLM."""
    project = tmp_path / "project"
    project.mkdir()
    child_root = project / "agent_data" / "probe"
    child_root.mkdir(parents=True)
    (child_root / "kestrel_prime.db").touch()
    log = child_root / "agent.log"
    prior = "prior-restart-log-content\n" * 128
    log.write_text(prior, encoding="utf-8")
    marker = tmp_path / "launcher-has-exited"
    child_program = r'''
import os, sys, time
from pathlib import Path
marker = Path(sys.argv[1])
deadline = time.monotonic() + 20
while not marker.exists():
    if time.monotonic() >= deadline:
        raise SystemExit(2)
    time.sleep(0.05)
os.write(1, b"native-child-after-launcher-exit\n")
'''
    parent_program = r'''
import subprocess, sys
from pathlib import Path
from unittest.mock import patch
from kestrel_sovereign.cli import build_parser, cmd_start
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.multi_agent.process_manager import ProcessManager
project, marker, child_program = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
config = MultiAgentConfig()
config.agents["probe"] = LocalAgentConfig(data_dir=Path("agent_data/probe"), port=18881)
config.save(project / "multi_agent.toml")
real_popen = subprocess.Popen
def spawn_native_child(_command, **kwargs):
    return real_popen([sys.executable, "-c", child_program, marker], **kwargs)
with patch("kestrel_sovereign.cli._get_project_dir", return_value=project), \
     patch("kestrel_sovereign.cli._maybe_first_run_setup", return_value=None), \
     patch.object(ProcessManager, "_load_env", return_value=dict(__import__("os").environ)), \
     patch.object(ProcessManager, "is_port_in_use", return_value=False), \
     patch.object(ProcessManager, "wait_for_health", return_value=True), \
     patch("subprocess.Popen", side_effect=spawn_native_child):
    raise SystemExit(cmd_start(build_parser().parse_args(["start", "probe"])))
'''
    spec = importlib.util.find_spec("kestrel_sovereign")
    assert spec is not None and spec.origin is not None
    # Use the actual imported installation, not this test file's checkout.
    # PYTHONSAFEPATH also prevents -c's cwd from shadowing a wheel with source.
    package_root = Path(spec.origin).resolve().parent.parent
    environment = {
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(package_root),
        "PYTHONSAFEPATH": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHON_DOTENV_DISABLED": "1",
        "KESTREL_SKIP_DOTENV": "1",
        "EMAIL_DRY_RUN": "true",
        "KESTREL_PHOENIX_ENABLED": "0",
        "KESTREL_TRACING_ENABLED": "0",
    }
    for name in ("SystemRoot", "WINDIR", "COMSPEC"):
        if name in os.environ:
            environment[name] = os.environ[name]
    child = None
    try:
        launcher = subprocess.run(
            [sys.executable, "-c", parent_program, str(project), str(marker), child_program],
            env=environment, capture_output=True, text=True, encoding="utf-8",
            timeout=15,
        )
        assert launcher.returncode == 0, launcher.stdout + launcher.stderr
        record = json.loads((child_root / "agent.pid").read_text())
        candidate = _recorded_probe_process(record)
        assert candidate is not None, "probe PID birth identity changed"
        child = candidate
        # Only the test's owning process releases the child, AFTER the launcher
        # has exited. The log therefore cannot pass on pre-exit pump output.
        marker.touch()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if "native-child-after-launcher-exit" in log.read_text(encoding="utf-8"):
                break
            time.sleep(0.05)
        assert log.read_text(encoding="utf-8") == prior + "native-child-after-launcher-exit\n"
    finally:
        # The child has a 20-second self-deadline as well as exact-PID cleanup.
        marker.touch()
        if child is None and (child_root / "agent.pid").exists():
            record = json.loads((child_root / "agent.pid").read_text())
            child = _recorded_probe_process(record)
        if child is not None:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    child.send_signal(signal.SIGTERM)
                child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="kestrel-detached-output-") as directory:
        test_named_cli_retains_real_child_output_after_launcher_exit(Path(directory))
    print("PASS: installed native child retained prior logs after launcher exit")
