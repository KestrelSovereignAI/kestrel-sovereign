"""The supported minimum interpreter must exercise actual native custody."""

import yaml

from tests.utils.ci_budget import CI_WORKFLOW


def minimum_job():
    return yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]["minimum-python-custody"]


def test_minimum_lane_pins_interpreter_and_exact_release_ref():
    job = minimum_job()
    assert job["needs"] == "lint-and-imports"
    assert job["permissions"] == {"contents": "read"}
    checkout = next(step for step in job["steps"]
                    if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["ref"] == "${{ inputs.ref }}"
    assert any(step.get("run") == "uv sync --python 3.11 --locked"
               for step in job["steps"])
    assert not job.get("continue-on-error", False)
    assert int(job["timeout-minutes"]) <= 15


def test_minimum_lane_cannot_silently_skip_native_or_omit_regressions():
    job = minimum_job()
    step, = [step for step in job["steps"] if "-m pytest" in step.get("run", "")]
    run = step["run"]
    assert "uv run --python 3.11 --no-sync python -m pytest" in run
    assert "--session-timeout=600" in run and "--timeout=60" in run
    assert "-p kestrel_sovereign.testing.fixtures" in run
    for name in (
        "tests/integration/test_wait_resident_custody_pg.py",
        "wait_resident_outlives",
        "crash_while_the_late_wake_is_in_flight",
        "spawn_receipt_wait_cannot_outlive",
        "deeply_nested_value_never_raises",
        "public_stream_rename_retains_privacy",
        "native_fanout_physically_joins_stronger_late_sibling",
        "terminate_child_preserves_active_cancellation_group",
    ):
        assert name in run
    env = step["env"]
    assert env["TEST_POSTGRES_URL"].startswith("postgresql://")
    assert env["KESTREL_REQUIRE_POSTGRES_TESTS"] == "1"
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert env["PYTHON_DOTENV_DISABLED"] == env["KESTREL_SKIP_DOTENV"] == "1"
    assert env["OPENROUTER_API_KEY"] == env["OPENROUTER_MANAGEMENT_API_KEY"] == ""
