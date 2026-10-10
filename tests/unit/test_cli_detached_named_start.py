"""Named fire-and-exit launches must retain logs after their parent exits."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def test_named_cli_retains_real_child_output_after_launcher_exit(tmp_path):
    """Use actual named-start wiring and native fds, without an agent/LLM."""
    project = tmp_path / "project"
    project.mkdir()
    child_root = project / "agent_data" / "probe"
    child_root.mkdir(parents=True)
    (child_root / "kestrel_prime.db").touch()
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
    environment = {
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "PYTHON_DOTENV_DISABLED": "1",
        "KESTREL_SKIP_DOTENV": "1",
        "EMAIL_DRY_RUN": "true",
        "KESTREL_PHOENIX_ENABLED": "0",
        "KESTREL_TRACING_ENABLED": "0",
    }
    child = None
    try:
        launcher = subprocess.run(
            [sys.executable, "-c", parent_program, str(project), str(marker), child_program],
            env=environment, capture_output=True, text=True, timeout=15,
        )
        assert launcher.returncode == 0, launcher.stdout + launcher.stderr
        record = json.loads((child_root / "agent.pid").read_text())
        child = psutil.Process(record["pid"])
        assert abs(child.create_time() - record["started_at"]) < 0.01
        # Only the test's owning process releases the child, AFTER the launcher
        # has exited. The log therefore cannot pass on pre-exit pump output.
        marker.touch()
        log = child_root / "agent.log"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if log.exists() and "native-child-after-launcher-exit" in log.read_text():
                break
            time.sleep(0.05)
        assert "native-child-after-launcher-exit" in log.read_text()
    finally:
        # The child has a 20-second self-deadline as well as exact-PID cleanup.
        marker.touch()
        if child is None and (child_root / "agent.pid").exists():
            record = json.loads((child_root / "agent.pid").read_text())
            try:
                candidate = psutil.Process(record["pid"])
                if abs(candidate.create_time() - record["started_at"]) < 0.01:
                    child = candidate
            except psutil.NoSuchProcess:
                pass
        if child is not None:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    child.send_signal(signal.SIGTERM)
                child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
