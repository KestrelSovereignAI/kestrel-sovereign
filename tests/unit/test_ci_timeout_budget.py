"""Each pytest tier's time budget must outlast a slow runner (#3212, #3330).

Three integration budgets in a row were sized to the worst runner ratio
seen so far — 10, then 20, then 30 — and each was taught its successor
by killing a passing run. The unit tier then repeated the lesson at 20
min (#3330): a slow runner killed a healthy suite mid-session, and the
runner's kill left no traceback for the two tests it named as failed.

So each tier declares the inputs to its argument next to the number, in
a `budget-basis(<tier>)` marker, and this asserts the relation between
them. It is deliberately not an assertion that a budget equals some
literal: a test that pins the literal only says the literal has not
changed, which is the one thing a reader can already see.
"""

from __future__ import annotations

import re

import pytest
import yaml

from tests.utils.ci_budget import CI_WORKFLOW, budget_basis_pattern


# What the step still has to do once pytest returns. Measured on the
# event in #3212: pytest printed its summary at 29:49 and the runner
# killed the step at 30:00 — eleven seconds of teardown and reporting.
_POST_SUITE_MARGIN_MINUTES = 1.0

# (tier named in `budget-basis(<tier>)`, ci.yml job id, suite path)
_TIERS = [
    pytest.param("unit", "unit-tests", "tests/unit/", id="unit"),
    pytest.param(
        "integration", "integration-tests", "tests/integration/", id="integration"
    ),
]


def _basis(tier: str) -> re.Match[str]:
    match = budget_basis_pattern(tier).search(CI_WORKFLOW.read_text())
    assert match is not None, f"budget-basis({tier}) marker missing"
    return match


def _suite_step(job: str, suite: str) -> dict:
    """The step in *job* that runs *suite*."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    steps = workflow["jobs"][job]["steps"]
    running = [
        step for step in steps if f"pytest {suite}" in str(step.get("run", ""))
    ]
    assert len(running) == 1, (
        f"expected exactly one step in {job} running {suite}, "
        f"found {len(running)}"
    )
    return running[0]


def _step_timeout(job: str, suite: str) -> int:
    """The `timeout-minutes` the runner enforces on that step."""
    timeout = _suite_step(job, suite).get("timeout-minutes")
    assert timeout is not None, f"the {suite} step declares no timeout-minutes"
    return int(timeout)


def _pytest_deadlines(tier: str, job: str, suite: str) -> tuple[float, float]:
    """`(session deadline, longest a single test may run)` in minutes."""
    run = str(_suite_step(job, suite)["run"])
    session = re.search(r"--session-timeout=(\d+)", run)
    per_test = re.search(r"--timeout=(\d+)", run)
    assert session is not None, (
        f"the {suite} step has no --session-timeout: without it a slow run "
        f"is killed by the runner, which prints no FAILED line or traceback "
        f"(#3212, #3330)"
    )
    assert per_test is not None, f"the {suite} step has no per-test --timeout"

    # The CLI value is the default ceiling; a marker may raise it, up to
    # the declared `longest-test` that the tier's conftest enforces.
    longest = max(int(per_test[1]), int(_basis(tier)["longest_test"]))
    return int(session[1]) / 60, longest / 60


def test_each_tier_has_its_own_marker():
    """Markers are keyed by tier, so neither tier can read the other's.

    A bare `budget-basis:` search returns whichever tier comes first in
    the file; with two tiers that silently hands one the other's inputs.
    """
    text = CI_WORKFLOW.read_text()
    assert not re.search(r"budget-basis:", text), (
        "an untiered `budget-basis:` marker is ambiguous; "
        "write `budget-basis(<tier>):`"
    )
    for tier in ("unit", "integration"):
        assert len(budget_basis_pattern(tier).findall(text)) == 1, (
            f"expected exactly one budget-basis({tier}) marker"
        )


@pytest.mark.parametrize(("tier", "job", "suite"), _TIERS)
def test_the_budget_basis_is_declared_and_parsable(tier, job, suite):
    """The positive control.

    Without it, a renamed or deleted marker would make the assertions
    below vacuous and the gate would pass by finding nothing.
    """
    match = _basis(tier)
    assert float(match["slowest"]) > 0
    assert float(match["ratio"]) >= 1.0
    assert int(match["samples"]) >= 5, "too few samples to call anything worst"
    assert int(match["longest_test"]) > 0


@pytest.mark.parametrize(("tier", "job", "suite"), _TIERS)
def test_the_budget_outlasts_the_slowest_run_on_the_slowest_runner(
    tier, job, suite
):
    """The invariant the last four budgets each violated in turn.

    A healthy suite's worst credible wall time is the slowest pass ever
    observed, run on the slowest runner ever observed. A budget under
    that number does not bound a hang — it discards passing work, and
    reports the loss as a test failure.
    """
    match = _basis(tier)
    worst_credible = float(match["slowest"]) * float(match["ratio"])

    session_deadline, _ = _pytest_deadlines(tier, job, suite)

    assert session_deadline >= worst_credible, (
        f"the {tier} suite is stopped after {session_deadline:.0f} min, "
        f"but a healthy run on the slowest observed runner takes up to "
        f"{worst_credible:.1f} min ({match['slowest']} x {match['ratio']}). "
        f"Raise --session-timeout, or re-measure and update "
        f"`budget-basis({tier})`."
    )


@pytest.mark.parametrize(("tier", "job", "suite"), _TIERS)
def test_pytest_reaches_its_own_deadline_before_the_runner_kills_the_step(
    tier, job, suite
):
    """Whichever deadline fires first decides what the failure looks like.

    pytest's own says `session-timeout: N sec exceeded` and still prints
    its FAILURES section and short summary. The runner's says
    `##[error]The action ... has timed out` with no traceback anywhere,
    which is why #3330's red named two failing tests and gave nothing to
    diagnose them with. So pytest has to get there first.

    It checks its deadline BETWEEN tests, not during one, so a run that
    crosses it inside a test keeps going until that test ends — up to
    that test's own ceiling, which a `@pytest.mark.timeout` marker may
    raise above the command line's. The runner's budget has to clear
    all of it, plus the teardown that still runs after pytest returns.
    """
    session_deadline, longest_test = _pytest_deadlines(tier, job, suite)
    runner_budget = _step_timeout(job, suite)

    latest_pytest_can_stop = session_deadline + longest_test
    required = latest_pytest_can_stop + _POST_SUITE_MARGIN_MINUTES

    assert runner_budget >= required, (
        f"the runner kills the {tier} step at {runner_budget} min, but "
        f"pytest may not stop until {latest_pytest_can_stop:.0f} min "
        f"({session_deadline:.0f} session + {longest_test:.0f} for a test "
        f"already running, marker overrides included), and the step still "
        f"has ~{_POST_SUITE_MARGIN_MINUTES:.0f} min of teardown after that. "
        f"The runner would win the race and report a timeout with no "
        f"traceback."
    )
