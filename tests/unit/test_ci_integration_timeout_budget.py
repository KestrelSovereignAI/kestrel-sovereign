"""The integration tier's time budget must outlast a slow runner (#3212).

Three budgets in a row were sized to the worst runner ratio seen so far —
10, then 20, then 30 — and each was taught its successor by killing a
passing run. The argument for each lived in a YAML comment, which is
where it rotted: the 30 min version claimed to absorb "~1.9x on the
median", a 1.85x runner turned up, and nothing checked the claim.

So the inputs to the argument are declared next to the number, and this
asserts the relation between them. It is deliberately not an assertion
that the budget equals 45: a test that pins the literal only says the
literal has not changed, which is the one thing a reader can already
see.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"

# The declared inputs: the slowest passing run observed, and the slowest
# runner observed relative to a normal one. Their product is the longest
# a healthy suite can credibly take.
_BUDGET_BASIS = re.compile(
    r"budget-basis:\s*slowest-pass=(?P<slowest>[\d.]+)\s+"
    r"worst-runner-ratio=(?P<ratio>[\d.]+)\s+"
    r"samples=(?P<samples>\d+)"
)


def _integration_step() -> dict:
    """The step that runs the integration suite."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["integration-tests"]["steps"]
    running = [
        step
        for step in steps
        if "pytest tests/integration/" in str(step.get("run", ""))
    ]
    assert len(running) == 1, (
        f"expected exactly one step running the integration suite, "
        f"found {len(running)}"
    )
    return running[0]


def _integration_step_timeout() -> int:
    """The `timeout-minutes` the runner enforces on that step."""
    timeout = _integration_step().get("timeout-minutes")
    assert timeout is not None, "the integration step declares no timeout-minutes"
    return int(timeout)


def _pytest_deadlines() -> tuple[float, float]:
    """`(session deadline, per-test deadline)` in minutes, from the run line."""
    run = str(_integration_step()["run"])
    session = re.search(r"--session-timeout=(\d+)", run)
    per_test = re.search(r"--timeout=(\d+)", run)
    assert session is not None, (
        "the integration step has no --session-timeout: without it a slow "
        "run is killed by the runner, which prints no FAILED line (#3212)"
    )
    assert per_test is not None, "the integration step has no per-test --timeout"
    return int(session[1]) / 60, int(per_test[1]) / 60


def test_the_budget_basis_is_declared_and_parsable():
    """The positive control.

    Without it, a renamed or deleted marker would make the assertion
    below vacuous and the gate would pass by finding nothing.
    """
    match = _BUDGET_BASIS.search(WORKFLOW.read_text())
    assert match is not None, (
        "no `budget-basis:` line in ci.yml — the integration timeout's "
        "justification has to state its inputs where they can be checked"
    )
    assert float(match["slowest"]) > 0
    assert float(match["ratio"]) >= 1.0
    assert int(match["samples"]) >= 5, "too few samples to call anything worst"


def test_the_integration_budget_outlasts_the_slowest_run_on_the_slowest_runner():
    """The invariant the last three budgets each violated in turn.

    A healthy suite's worst credible wall time is the slowest pass ever
    observed, run on the slowest runner ever observed. A budget under
    that number does not bound a hang — it discards passing work, and
    reports the loss as a test failure.
    """
    match = _BUDGET_BASIS.search(WORKFLOW.read_text())
    assert match is not None, "budget-basis marker missing"
    worst_credible = float(match["slowest"]) * float(match["ratio"])

    session_deadline, _ = _pytest_deadlines()

    assert session_deadline >= worst_credible, (
        f"the integration suite is stopped after {session_deadline:.0f} min, "
        f"but a healthy run on the slowest observed runner takes up to "
        f"{worst_credible:.1f} min ({match['slowest']} x {match['ratio']}). "
        f"Raise --session-timeout, or re-measure and update `budget-basis`."
    )


def test_pytest_reaches_its_own_deadline_before_the_runner_kills_the_step():
    """Whichever deadline fires first decides what the failure looks like.

    pytest's own says `session-timeout: N sec exceeded` and exits 1 — a
    greppable failure. The runner's says
    `##[error]The action ... has timed out` with no FAILED line anywhere,
    which is why #3212's red read as a test failure until someone tailed
    the log. So pytest has to get there first.

    It checks its deadline BETWEEN tests, not during one, so a run that
    crosses it inside a test keeps going until that test ends — up to the
    per-test timeout. The runner's budget has to clear both.
    """
    session_deadline, per_test_deadline = _pytest_deadlines()
    runner_budget = _integration_step_timeout()

    latest_pytest_can_stop = session_deadline + per_test_deadline

    assert runner_budget > latest_pytest_can_stop, (
        f"the runner kills the step at {runner_budget} min, but pytest may "
        f"not stop until {latest_pytest_can_stop:.0f} min "
        f"({session_deadline:.0f} session + {per_test_deadline:.0f} for a "
        f"test already running). The runner would win the race and report "
        f"a timeout with no FAILED line."
    )


def test_the_unit_tier_budget_is_also_declared_in_minutes():
    """A cheap guard on the sibling this reasoning was borrowed from.

    Not a ratio check — the unit tier is parallelized and its margin is
    a different question — but a missing timeout there would mean an
    unbounded job, which is the failure the whole family exists to stop.
    """
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["unit-tests"]["steps"]
    running = [
        step for step in steps if "pytest tests/unit/" in str(step.get("run", ""))
    ]
    assert running, "no step runs the unit suite"
    assert all(step.get("timeout-minutes") for step in running), (
        "a unit-test step has no timeout-minutes"
    )
