"""Run a child pytest that takes nothing from the session that starts it (#3669).

A test of a pytest plugin runs the plugin in a child pytest. The child is
started from inside the parent's test, so it inherits the parent's environment:
``PYTEST_ADDOPTS`` and ``PYTEST_PLUGINS`` change what it loads and runs, and
under xdist ``PYTEST_XDIST_WORKER`` and its siblings describe the parent's
worker, not the child. Its configuration comes from its own ``pytest.ini`` and
command line only, so the parent's ``pyproject.toml`` options, its ``-n`` and
its timeouts never reach it.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def child_pytest_env() -> dict[str, str]:
    """This process's environment without pytest's own variables.

    Every ``PYTEST_`` variable is dropped and plugin autoload is off, so the
    child loads exactly the plugins its command line names with ``-p``. The
    import path is the package root only: forwarding ``sys.path`` lets a
    package directory shadow a stdlib module in the child.
    """
    env = {
        name: value for name, value in os.environ.items()
        if not name.startswith("PYTEST_")
    }
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


def run_child_pytest(
    directory: Path,
    *,
    conftest: str,
    test_file: str,
    source: str,
    args: list[str],
    timeout: float,
) -> subprocess.CompletedProcess:
    """Run ``test_file`` (written from ``source``) in ``directory``.

    The child's rootdir, ini file and cache are its own, and ``args`` name its
    plugins and options. Its stdout and stderr are one stream, as in a CI log.
    ``timeout`` bounds the child's liveness, not its duration.
    """
    ini = directory / "pytest.ini"
    ini.write_text("[pytest]\n", encoding="utf-8")
    (directory / "conftest.py").write_text(conftest, encoding="utf-8")
    (directory / test_file).write_text(source, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "-c", str(ini), "--rootdir", str(directory), "-p", "no:cacheprovider",
            *args,
        ],
        cwd=directory,
        env=child_pytest_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
