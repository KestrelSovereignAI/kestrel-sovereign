"""The pull-request Sovereign Console smoke job (#2682).

Each assertion is a property the issue's acceptance criteria rest on and that
an edit to ``ci.yml`` could silently drop while every other test stays green:
the job runs on pull requests, is bounded, references no secret, runs the
isolated ``kestrel demo smoke`` instance, and tears it down on success,
failure, and cancellation.
"""

from __future__ import annotations

import re

import yaml

from tests.utils.ci_budget import CI_WORKFLOW

JOB = "console-smoke"


def _jobs() -> dict:
    return yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]


def _job() -> dict:
    jobs = _jobs()
    assert JOB in jobs, f"ci.yml no longer defines the {JOB} job"
    return jobs[JOB]


def _step(name_fragment: str) -> dict:
    matches = [s for s in _job()["steps"] if name_fragment in s.get("name", "")]
    assert len(matches) == 1, f"expected one {JOB} step named *{name_fragment}*"
    return matches[0]


def test_runs_on_pull_requests_like_the_other_browserless_tiers():
    """Same gate as integration-tests: PRs always run it; main squash-merges
    and duplicate issue-* push runs skip it."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    assert "pull_request" in workflow[True]  # PyYAML reads the `on:` key as True
    assert _job()["if"] == _jobs()["integration-tests"]["if"]
    assert _job()["needs"] == "lint-and-imports"


def test_is_bounded():
    assert int(_job()["timeout-minutes"]) <= 20
    smoke = _step("Run the Console smoke")
    assert int(smoke["timeout-minutes"]) <= 10


def test_references_no_secret_and_reads_only():
    """No production key can reach the smoke: the job names no secret."""
    text = yaml.safe_dump(_job())
    assert "secrets." not in text
    assert _job()["permissions"] == {"contents": "read"}


def test_runs_the_isolated_smoke_on_a_non_live_port_in_a_fresh_home():
    run = _step("Run the Console smoke")["run"]
    assert "kestrel demo smoke" in run
    assert '--home "$RUNNER_TEMP/kestrel-console-smoke"' in run
    port = _job()["env"]["SMOKE_PORT"]
    assert '--port "$SMOKE_PORT"' in run
    assert port != "8888"


def test_installs_one_browser_and_caches_node_dependencies():
    install = _step("Install the Chromium browser")["run"]
    assert re.fullmatch(r"npx playwright install --with-deps chromium", install.strip())
    node = next(
        s for s in _job()["steps"] if str(s.get("uses", "")).startswith("actions/setup-node@")
    )
    assert node["with"]["cache"] == "npm"


def test_teardown_runs_on_success_failure_and_cancellation():
    """`always()` is the only status function that also runs on cancel, and
    the teardown must follow the smoke and stop the recorded server."""
    steps = _job()["steps"]
    teardown = _step("Tear down the smoke instance")
    assert teardown["if"] == "always()"
    assert steps.index(teardown) > steps.index(_step("Run the Console smoke"))
    script = teardown["run"]
    assert "server.pid" in script
    assert 'kill -TERM -- "-$pid"' in script
    assert 'rm -rf "$home"' in script
