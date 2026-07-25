"""Regression coverage for pytest-cov/xdist shutdown with lingering threads."""

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_xdist_worker_can_finalize_coverage_before_lingering_thread_guard(tmp_path):
    """A worker with a live thread must return complete coverage data.

    The repository cleanup hook used to call ``os._exit`` directly from
    ``pytest_sessionfinish``.  That prevented xdist from returning its worker
    output and could leave pytest-cov to combine a half-created SQLite file.
    """
    probe = tmp_path / "test_lingering_thread.py"
    probe.write_text(
        """
import threading
import time

from kestrel_sovereign import api_errors


def test_lingering_non_daemon_thread():
    threading.Thread(target=time.sleep, args=(30.0,), daemon=False).start()
    assert api_errors.__name__ == "kestrel_sovereign.api_errors"
""",
        encoding="utf-8",
    )
    report = tmp_path / "coverage.json"
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(tmp_path / ".coverage")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(probe),
            "-p",
            "tests.conftest",
            "-n",
            "2",
            "--cov=kestrel_sovereign",
            "--cov-report=term",
            f"--cov-report=json:{report}",
            "--cov-fail-under=0",
            "-q",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "Force exiting after plugin-teardown grace period" in output
    assert "no such table: file" not in output
    assert "coverage: failed workers" not in output

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["meta"]["branch_coverage"] is True
    assert "kestrel_sovereign/api_errors.py" in payload["files"]
