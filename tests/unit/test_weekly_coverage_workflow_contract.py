"""Contract tests for the scheduled repository-wide coverage gate."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "weekly-analysis.yml"


def _workflow() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    return steps


def _step_named(job: dict[str, object], name: str) -> dict[str, object]:
    return next(step for step in _steps(job) if step.get("name") == name)


def test_weekly_coverage_failure_is_not_masked_before_feedback_runs():
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)

    test_job = jobs["full-test-suite"]
    assert isinstance(test_job, dict)
    coverage_step = _step_named(test_job, "Run full test suite with feedback")
    assert coverage_step["run"] == "uv run python run_tests.py --ci --feedback"
    assert coverage_step.get("continue-on-error", False) is False

    coverage_steps = [
        step
        for job in jobs.values()
        if isinstance(job, dict)
        for step in _steps(job)
        if "--ci" in str(step.get("run", "")) or "--cov" in str(step.get("run", ""))
    ]
    assert coverage_steps == [coverage_step]

    upload_step = _step_named(test_job, "Upload feedback artifact")
    assert upload_step["if"] == "always()"

    issue_job = jobs["create-issues"]
    assert isinstance(issue_job, dict)
    assert issue_job["needs"] == "full-test-suite"
    assert issue_job["if"] == "${{ always() }}"
