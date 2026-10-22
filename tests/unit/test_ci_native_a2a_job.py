"""The required native lane must exercise SQLite shutdown and foreign wire use."""

import shlex

import yaml

from tests.utils.ci_budget import CI_WORKFLOW


def test_native_lane_collects_wire_and_original_shutdown_regressions():
    job = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]["a2a-go-interop"]
    step, = [step for step in job["steps"] if "-m pytest" in step.get("run", "")]
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
