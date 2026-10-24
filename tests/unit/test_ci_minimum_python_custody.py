"""The supported minimum interpreter must exercise actual native custody."""

import ast
import shlex

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
        "tests/unit/test_a2a_native_question_supervisor.py",
        "crash_after_durable_admission_before_outbox_checkpoint_deduplicates",
        "privacy_elided_question_wake_rehydrates_from_owned_outbox_after_restart",
        "answer_after_clock_deadline_creates_ordered_deadline_and_late_waves",
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


def test_minimum_lane_references_existing_files_and_live_test_selectors():
    """Deleting or renaming a regression must not leave a vacuous CI command."""
    step, = [step for step in minimum_job()["steps"] if "-m pytest" in step.get("run", "")]
    command = shlex.split(step["run"])
    root = CI_WORKFLOW.parents[2]
    modules = [root / token.split("::", 1)[0] for token in command if token.startswith("tests/")]
    assert modules
    missing = [str(path.relative_to(root)) for path in modules if not path.is_file()]
    assert not missing, f"Minimum-Python CI references missing tests: {missing}"
    names = {
        node.name
        for path in modules
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }
    selectors = command[command.index("-k") + 1].split(" or ")
    unmatched = [selector for selector in selectors if not any(selector in name for name in names)]
    assert not unmatched, f"Minimum-Python CI selects no regression for: {unmatched}"
