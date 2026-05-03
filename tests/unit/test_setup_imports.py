"""Regression tests for setup-package import order.

Originally the setup wizard had a circular import:

    cli.cmd_doctor → doctor → setup.env_file (triggers setup/__init__.py)
      → setup.wizard → setup.steps → setup.steps.verify → doctor

The unit test files happened to bypass it because they always imported
``setup.env_file`` (or another deep submodule) first; ``kestrel doctor``
went straight through ``doctor`` and crashed on collection.

These tests load the user-facing entry points in *fresh subprocesses*
so a cycle reappearing — even one masked by ordinary test imports —
fails loudly. We use subprocesses (rather than mutating ``sys.modules``)
because purging Kestrel modules from the running pytest's cache leaves
already-imported test fixtures bound to stale references.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(script: str, cwd: str = "/tmp") -> subprocess.CompletedProcess:
    """Run a one-shot Python snippet in a fresh subprocess.

    ``cwd`` defaults to /tmp so the project's own ``.env`` / ``kestrel.toml``
    do not satisfy doctor checks accidentally — we want each script to
    drive the import path under controlled state.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _assert_no_circular(result: subprocess.CompletedProcess) -> None:
    combined = (result.stdout + result.stderr).lower()
    assert "circular import" not in combined, result.stderr
    assert "importerror" not in combined, result.stderr


def test_doctor_imports_cold():
    """A cold import of doctor must not deadlock or ImportError."""
    result = _run(
        """
        import kestrel_sovereign.doctor as d
        assert callable(d.diagnose), 'diagnose missing'
        assert callable(d.format_report), 'format_report missing'
        print('OK')
        """
    )
    _assert_no_circular(result)
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_setup_wizard_imports_cold():
    result = _run(
        """
        from kestrel_sovereign.setup.wizard import run_wizard, build_context
        assert callable(run_wizard)
        assert callable(build_context)
        print('OK')
        """
    )
    _assert_no_circular(result)
    assert result.returncode == 0


def test_setup_package_init_is_minimal():
    """Importing the setup package must NOT eagerly load wizard/steps.

    If they're imported eagerly, the cycle returns. We verify the
    side-effects: after ``import kestrel_sovereign.setup``, neither
    submodule should be in sys.modules.
    """
    result = _run(
        """
        import sys
        import kestrel_sovereign.setup  # noqa: F401
        assert 'kestrel_sovereign.setup.wizard' not in sys.modules
        assert 'kestrel_sovereign.setup.steps' not in sys.modules
        print('OK')
        """
    )
    _assert_no_circular(result)
    assert result.returncode == 0, result.stderr


def test_kestrel_doctor_cli_runs_without_circular_import(tmp_path):
    """End-to-end: invoke `kestrel doctor` cold via the module entry point."""
    result = subprocess.run(
        [sys.executable, "-m", "kestrel_sovereign.cli", "doctor"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = (result.stdout + result.stderr).lower()
    assert "circular import" not in combined
    assert "traceback" not in combined
    # Doctor exits non-zero on an empty dir (nothing configured) but it
    # should have produced a real report, not a stack trace.
    assert "kestrel_data_key" in combined or "ready" in combined


def test_kestrel_setup_check_cli_runs_without_circular_import(tmp_path):
    """`kestrel setup --check` is the other entry that hits the cycle path."""
    result = subprocess.run(
        [sys.executable, "-m", "kestrel_sovereign.cli", "setup", "--check"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        env={"KESTREL_NONINTERACTIVE": "1", "PATH": "/usr/bin:/bin"},
    )
    combined = (result.stdout + result.stderr).lower()
    assert "circular import" not in combined
    assert "traceback" not in combined
