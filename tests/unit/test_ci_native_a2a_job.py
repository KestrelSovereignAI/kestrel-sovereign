"""The required native lane must exercise SQLite shutdown and foreign wire use."""

import shlex

import yaml

from tests.utils.ci_budget import CI_WORKFLOW


def test_native_lane_collects_wire_and_original_shutdown_regressions():
    job = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]["a2a-go-interop"]
    step, = [
        step for step in job["steps"]
        if "tests/interop/test_a2a_go.py" in step.get("run", "")
    ]
    command = shlex.split(step["run"])
    modules = {
        "tests/interop/test_a2a_go.py",
        "tests/interop/test_native_drainer_shutdown.py",
    }
    # Require actual collection rather than an inventory comment; filters must
    # not silently select only the public exchange while omitting shutdown.
    assert modules <= set(command), "Native A2A CI omits a maintained interop module"
    assert "-k" not in command and "-m" not in command[command.index("pytest") + 1:]
    assert not step.get("continue-on-error", False)
    assert step.get("if", "") in {"", "success()"}


def test_native_lane_collects_tck_and_retains_raw_results():
    job = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]["a2a-go-interop"]
    step, = [
        step for step in job["steps"]
        if "tests/interop/test_a2a_tck.py" in step.get("run", "")
    ]
    command = shlex.split(step["run"].replace("${{ runner.temp }}", "RUNNER_TEMP"))
    assert command[command.index("-m") + 1] == "pytest"
    assert "pytest_asyncio.plugin" in command and "pytest_timeout" in command
    assert "--basetemp=RUNNER_TEMP/kestrel-a2a-tck" in command
    assert "--timeout=300" in command
    assert "--junitxml=a2a-tck.xml" in command
    assert "-k" not in command and "-m" not in command[command.index("pytest") + 1:]
    assert not step.get("continue-on-error", False)
    assert step.get("if", "") in {"", "success()"}
    retention, = [
        item for item in job["steps"]
        if item.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert retention["if"] == "always()"
    assert retention["with"]["if-no-files-found"] == "error"
    assert {
        "a2a-tck.xml",
        "${{ runner.temp }}/kestrel-a2a-tck/**/tck-*.log",
        "${{ runner.temp }}/kestrel-a2a-tck/**/tck-*-reports/",
    } <= set(retention["with"]["path"].splitlines())
