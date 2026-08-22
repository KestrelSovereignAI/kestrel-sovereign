"""Regression coverage for pytest-cov/xdist shutdown with lingering threads."""

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# How long the probe's lingering non-daemon thread stays alive, and how long we
# wait for the subprocess. These two MUST NOT be equal, and the gap is the
# whole signal.
#
# What the numbers actually are (measured on a dev box, 2026-07-31):
#
#   nested pytest with the lingering thread     17.2s
#   nested pytest without it                    16.6s
#
# So the guard works — the force-exit costs ~0.6s — and the run is dominated by
# nested-pytest startup: a second interpreter, xdist `-n 2`, and coverage
# instrumenting the whole package. None of that is what the test asserts.
#
# Both values used to be 30, which left a HEALTHY run barely 1.7x its own
# baseline before timing out. Ordinary extra load — a busy CI runner, an outer
# `-n auto` saturating the box — pushes 17s past 30s and fails a guard that is
# working correctly. That flake blocked a talon run AND a PR merge on
# 2026-07-31, on a commit whose sibling run passed three seconds apart.
#
# Raising the bound cannot blunt this test, because the timeout was never what
# detects the regression. Verified by reintroducing the direct `os._exit`: the
# run fails on the `'Force exiting after plugin-teardown grace period'`
# assertion (with `node down: Not properly terminated` in the captured output)
# after ~43s — well inside the bound. The timeout is purely a hang guard for
# the case where the subprocess never returns at all, which is why it is sized
# for liveness (~7x a healthy run) rather than for performance.
#
# LINGERING_THREAD_SECONDS only has to outlast SUBPROCESS_TIMEOUT_SECONDS so a
# genuinely wedged subprocess is still bounded rather than hanging CI.
LINGERING_THREAD_SECONDS = 300.0
SUBPROCESS_TIMEOUT_SECONDS = 120


def test_xdist_worker_can_finalize_coverage_before_lingering_thread_guard(tmp_path):
    """A worker with a live thread must return complete coverage data.

    The repository cleanup hook used to call ``os._exit`` directly from
    ``pytest_sessionfinish``.  That prevented xdist from returning its worker
    output and could leave pytest-cov to combine a half-created SQLite file.
    """
    probe = tmp_path / "test_lingering_thread.py"
    probe.write_text(
        f"""
import threading
import time

from kestrel_sovereign import api_errors


def test_lingering_non_daemon_thread():
    threading.Thread(
        target=time.sleep, args=({LINGERING_THREAD_SECONDS!r},), daemon=False
    ).start()
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
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "Force exiting after plugin-teardown grace period" in output
    assert "no such table: file" not in output
    assert "coverage: failed workers" not in output

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["meta"]["branch_coverage"] is True
    assert "kestrel_sovereign/api_errors.py" in payload["files"]
