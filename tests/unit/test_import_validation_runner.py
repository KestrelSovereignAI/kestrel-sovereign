"""Contracts for import validation timing and project-environment selection."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
from _pytest.terminal import format_session_duration

import run_tests


def _completed_collection(summary: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=f"{summary}\n",
        stderr="",
    )


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ("15198 tests collected in 3.83s", (15198, 3.83)),
        ("7/13570 tests collected (13563 deselected) in 3.14s", (13570, 3.14)),
        ("no tests collected (12 deselected) in 0.04s", (0, 0.04)),
        ("1 test collected in 61.01s (0:01:01)", (1, 61.01)),
        ("===== 12 tests collected in 0.05s =====", (12, 0.05)),
        ("\x1b[32m12 tests collected\x1b[0m\x1b[32m in 0.06s\x1b[0m", (12, 0.06)),
    ],
    ids=["plain", "deselected", "empty", "long-duration", "verbose", "colored"],
)
def test_parse_collection_summary_accepts_pytest_summary_forms(summary, expected):
    assert run_tests.parse_collection_summary(summary) == expected
    assert run_tests.parse_collection_duration(summary) == expected[1]


def test_collection_parser_uses_last_complete_summary_not_node_id_text():
    output = (
        "tests/unit/test_example.py::test_case[15192 tests collected in 3.83s]\n"
        "15198 tests collected in 3.23s\n"
    )

    assert run_tests.parse_collection_summary(output) == (15198, 3.23)


def test_validate_imports_times_direct_pytest_and_pins_summary_environment(
    monkeypatch,
    capsys,
):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _completed_collection("12 tests collected in 5.00s")

    clock = iter((100.0, 107.0))
    monkeypatch.setattr(run_tests, "_run", fake_run)
    monkeypatch.setattr(run_tests, "_now", lambda: next(clock))
    monkeypatch.setenv("PYTEST_ADDOPTS", "-v --color=yes")
    monkeypatch.setenv("PY_COLORS", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    runner = run_tests.SmartTestRunner()

    assert runner.validate_imports([runner.kestrel_tests]) is True

    command = captured["command"]
    assert command[:6] == [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "--color=no",
    ]
    assert command[0] != "uv"
    assert captured["kwargs"]["timeout"] == (
        run_tests.IMPORT_VALIDATION_PROCESS_TIMEOUT_SECONDS
    )
    child_environment = captured["kwargs"]["env"]
    assert child_environment["PYTEST_ADDOPTS"] == ""
    assert child_environment["PY_COLORS"] == "0"
    assert "FORCE_COLOR" not in child_environment
    output = capsys.readouterr().out
    assert "5.00s collection" in output
    assert "7.0s process wall time" in output


def test_validate_imports_enforces_pytest_duration_not_process_wall_time(
    monkeypatch,
):
    monkeypatch.setattr(
        run_tests,
        "_run",
        lambda *args, **kwargs: _completed_collection(
            "12 tests collected in 5.00s"
        ),
    )
    clock = iter((10.0, 110.0))
    monkeypatch.setattr(run_tests, "_now", lambda: next(clock))
    runner = run_tests.SmartTestRunner()

    assert runner.validate_imports([runner.kestrel_tests]) is True


def test_validate_imports_rejects_real_pytest_over_budget_summary(monkeypatch, capsys):
    over_budget = run_tests.IMPORT_VALIDATION_COLLECTION_BUDGET_SECONDS + 0.01
    pytest_duration = format_session_duration(over_budget)
    monkeypatch.setattr(
        run_tests,
        "_run",
        lambda *args, **kwargs: _completed_collection(
            f"1 test collected in {pytest_duration}"
        ),
    )
    clock = iter((10.0, 71.0))
    monkeypatch.setattr(run_tests, "_now", lambda: next(clock))
    runner = run_tests.SmartTestRunner()

    assert runner.validate_imports([runner.kestrel_tests]) is False

    output = capsys.readouterr().out
    assert f">{run_tests.IMPORT_VALIDATION_COLLECTION_BUDGET_SECONDS:g}s budget" in output
    assert "61.0s process wall time" in output


def test_validate_imports_warns_before_collection_budget(monkeypatch, capsys):
    warning_duration = run_tests.IMPORT_VALIDATION_COLLECTION_WARNING_SECONDS + 0.01
    monkeypatch.setattr(
        run_tests,
        "_run",
        lambda *args, **kwargs: _completed_collection(
            f"1 test collected in {warning_duration:.2f}s"
        ),
    )
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(run_tests, "_now", lambda: next(clock))
    runner = run_tests.SmartTestRunner()

    assert runner.validate_imports([runner.kestrel_tests]) is True

    output = capsys.readouterr().out
    assert "warning threshold" in output
    assert "process wall time" in output


def test_successful_unparseable_collection_reports_parser_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        run_tests,
        "_run",
        lambda *args, **kwargs: _completed_collection("unexpected success output"),
    )
    clock = iter((10.0, 11.0))
    monkeypatch.setattr(run_tests, "_now", lambda: next(clock))
    runner = run_tests.SmartTestRunner()

    assert runner.validate_imports([runner.kestrel_tests]) is False

    output = capsys.readouterr().out
    assert "collection summary could not be parsed" in output
    assert "Missing module" not in output


def test_interpreter_project_environment_predicate(monkeypatch, tmp_path):
    assert run_tests.interpreter_uses_project_environment(sys.executable) is True

    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    interpreter_name = "python.exe" if os.name == "nt" else "python"
    outside_interpreter = tmp_path / "outside" / scripts_dir / interpreter_name
    assert run_tests.interpreter_uses_project_environment(outside_interpreter) is False

    shared_environment = tmp_path / "shared-environment"
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(shared_environment))
    assert run_tests.interpreter_uses_project_environment(sys.executable) is False
    assert run_tests.interpreter_uses_project_environment(
        shared_environment / scripts_dir / interpreter_name
    ) is True


def test_is_project_environment_requires_location_and_pytest(monkeypatch):
    monkeypatch.setattr(
        run_tests,
        "interpreter_uses_project_environment",
        lambda interpreter: False,
    )
    monkeypatch.setattr(run_tests, "_find_spec", lambda name: object())
    assert run_tests.is_project_environment() is False

    monkeypatch.setattr(
        run_tests,
        "interpreter_uses_project_environment",
        lambda interpreter: True,
    )
    monkeypatch.setattr(run_tests, "_find_spec", lambda name: None)
    assert run_tests.is_project_environment() is False

    monkeypatch.setattr(run_tests, "_find_spec", lambda name: object())
    assert run_tests.is_project_environment() is True


def test_bare_invocation_reexecutes_once_under_uv(monkeypatch):
    captured = {}

    def fake_execvpe(file, command, environment):
        captured.update(file=file, command=command, environment=environment)

    monkeypatch.setattr(run_tests, "is_project_environment", lambda: False)
    monkeypatch.setattr(run_tests, "_execvpe", fake_execvpe)
    monkeypatch.delenv(run_tests._RUN_TESTS_REEXEC_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["run_tests.py", "--unit", "--skip-check"])

    run_tests.ensure_project_environment()

    assert captured["file"] == "uv"
    assert captured["command"][:5] == [
        "uv",
        "run",
        "--project",
        str(run_tests.PROJECT_ROOT),
        "python",
    ]
    assert captured["command"][-2:] == ["--unit", "--skip-check"]
    assert captured["environment"][run_tests._RUN_TESTS_REEXEC_ENV] == "1"


def test_reexec_guard_reports_missing_project_pytest(monkeypatch, capsys):
    monkeypatch.setattr(run_tests, "is_project_environment", lambda: False)
    monkeypatch.setenv(run_tests._RUN_TESTS_REEXEC_ENV, "1")

    with pytest.raises(SystemExit, match="2"):
        run_tests.ensure_project_environment()

    error = capsys.readouterr().err
    assert "pytest installed" in error
    assert "uv sync --group test" in error


def test_runner_interpreter_can_import_pytest():
    runner = run_tests.SmartTestRunner()

    result = subprocess.run(
        [runner.pytest_interpreter, "-c", "import pytest"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert runner.pytest_interpreter == sys.executable
    assert result.returncode == 0, result.stderr


def test_timeout_message_names_process_guard_and_collection_budget(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(run_tests, "ensure_project_environment", lambda: None)
    monkeypatch.setattr(
        run_tests.SmartTestRunner,
        "validate_imports",
        lambda self, test_dirs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(
                cmd="pytest",
                timeout=run_tests.IMPORT_VALIDATION_PROCESS_TIMEOUT_SECONDS,
            )
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_tests.py", "--kestrel", "--skip-check", "--validate-only"],
    )

    with pytest.raises(SystemExit, match="1"):
        run_tests.main()

    output = capsys.readouterr().out
    assert (
        f"TIMED OUT after {run_tests.IMPORT_VALIDATION_PROCESS_TIMEOUT_SECONDS:g}s"
        in output
    )
    assert (
        f"collection budget is "
        f"{run_tests.IMPORT_VALIDATION_COLLECTION_BUDGET_SECONDS:g}s"
        in output
    )
