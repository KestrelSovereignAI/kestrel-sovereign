"""Semantic contract tests for the scheduled repository-wide coverage gate."""

from copy import deepcopy
from pathlib import Path
import shlex

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "weekly-analysis.yml"
COVERAGE_ARGV = ["uv", "run", "python", "run_tests.py", "--ci", "--feedback"]


def _workflow() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    return steps


def _step_named(job: dict[str, object], name: str) -> dict[str, object]:
    return next(step for step in _steps(job) if step.get("name") == name)


def _is_always(value: object) -> bool:
    expression = str(value or "").replace("${{", "").replace("}}", "")
    return "".join(expression.split()) == "always()"


def _coverage_step(job: dict[str, object]) -> dict[str, object]:
    matches = []
    for step in _steps(job):
        command = step.get("run")
        if isinstance(command, str) and shlex.split(command) == COVERAGE_ARGV:
            matches.append(step)
    assert len(matches) == 1
    return matches[0]


def _simulate_failed_coverage(workflow: dict[str, object]) -> dict[str, object]:
    """Evaluate the failure path relevant to this workflow contract.

    GitHub steps default to ``success()``.  ``always()`` steps run after a
    prior failure; a dependent job needs its own ``always()`` to run after a
    failed prerequisite.  This deliberately models status propagation rather
    than merely comparing condition strings.
    """
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    test_job = jobs["full-test-suite"]
    assert isinstance(test_job, dict)
    coverage_step = _coverage_step(test_job)

    test_status = "success"
    ran: list[str] = []
    for step in _steps(test_job):
        name = str(step.get("name", step.get("uses", "unnamed")))
        should_run = test_status == "success" or _is_always(step.get("if"))
        if not should_run:
            continue
        ran.append(name)

        failed = step is coverage_step
        if name in {"Upload feedback artifact", "Upload coverage diagnostics"}:
            # Simulate the failure path where pytest produced no artifact.
            settings = step.get("with")
            failed = not (
                isinstance(settings, dict)
                and settings.get("if-no-files-found") == "ignore"
            )
        if failed and step.get("continue-on-error") is not True:
            test_status = "failure"

    issue_job = jobs["create-issues"]
    assert isinstance(issue_job, dict)
    needs_failed_job = issue_job.get("needs") == "full-test-suite"
    issue_scheduled = (
        not needs_failed_job
        or test_status == "success"
        or _is_always(issue_job.get("if"))
    )
    issue_steps_ran: list[str] = []
    issue_status = "skipped"
    if issue_scheduled:
        issue_status = "success"
        for step in _steps(issue_job):
            name = str(step.get("name", step.get("uses", "unnamed")))
            if issue_status != "success" and not _is_always(step.get("if")):
                continue
            issue_steps_ran.append(name)
            if name == "Download feedback artifact":
                # Artifact is absent in this simulation.
                if step.get("continue-on-error") is not True:
                    issue_status = "failure"

    workflow_status = (
        "failure"
        if "failure" in {test_status, issue_status}
        else "success"
    )
    return {
        "workflow": workflow_status,
        "test_job": test_status,
        "test_steps": ran,
        "issue_job": issue_status,
        "issue_steps": issue_steps_ran,
    }


def test_failed_coverage_stays_red_while_feedback_path_completes():
    result = _simulate_failed_coverage(_workflow())

    assert result["workflow"] == "failure"
    assert result["test_job"] == "failure"
    assert "Upload feedback artifact" in result["test_steps"]
    assert "Upload coverage diagnostics" in result["test_steps"]
    assert result["issue_job"] == "success"
    assert "Download feedback artifact" in result["issue_steps"]
    assert "Create issues for patterns" in result["issue_steps"]


@pytest.mark.parametrize(
    ("mutate", "broken_assertion"),
    [
        (
            lambda workflow: _coverage_step(
                workflow["jobs"]["full-test-suite"]
            ).__setitem__("continue-on-error", True),
            lambda result: result["workflow"] == "failure",
        ),
        (
            lambda workflow: _step_named(
                workflow["jobs"]["full-test-suite"],
                "Upload feedback artifact",
            ).pop("if"),
            lambda result: "Upload feedback artifact" in result["test_steps"],
        ),
        (
            lambda workflow: workflow["jobs"]["create-issues"].pop("if"),
            lambda result: result["issue_job"] == "success",
        ),
        (
            lambda workflow: _step_named(
                workflow["jobs"]["create-issues"],
                "Download feedback artifact",
            ).pop("continue-on-error"),
            lambda result: "Create issues for patterns" in result["issue_steps"],
        ),
    ],
)
def test_failure_path_simulation_rejects_masking_and_short_circuit_mutations(
    mutate,
    broken_assertion,
):
    workflow = deepcopy(_workflow())
    mutate(workflow)

    assert broken_assertion(_simulate_failed_coverage(workflow)) is False


def test_shell_level_failure_masking_is_not_a_coverage_step():
    workflow = deepcopy(_workflow())
    coverage_step = _coverage_step(workflow["jobs"]["full-test-suite"])
    coverage_step["run"] += " || true"

    with pytest.raises(AssertionError):
        _simulate_failed_coverage(workflow)


def test_missing_feedback_and_coverage_artifacts_are_explicitly_tolerated():
    jobs = _workflow()["jobs"]
    test_job = jobs["full-test-suite"]
    issue_job = jobs["create-issues"]

    for name in ("Upload feedback artifact", "Upload coverage diagnostics"):
        step = _step_named(test_job, name)
        assert _is_always(step["if"])
        assert step["with"]["if-no-files-found"] == "ignore"

    download = _step_named(issue_job, "Download feedback artifact")
    assert download["continue-on-error"] is True
    create = _step_named(issue_job, "Create issues for patterns")
    assert "if [ -f /tmp/test_feedback.db ]; then" in create["run"]
