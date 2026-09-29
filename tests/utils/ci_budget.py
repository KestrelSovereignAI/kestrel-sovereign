"""One reader for the per-tier wall-clock budgets declared in ci.yml.

Each pytest tier in ci.yml states the inputs its time budget was sized
from in a `budget-basis(<tier>):` comment (#3212, #3330). Two consumers
read it: the unit test that checks the arithmetic between those inputs
and the step's deadlines, and each tier's conftest, which refuses to
collect a test whose per-test timeout exceeds the declared ceiling.

The marker is keyed by tier because a bare `budget-basis:` search finds
whichever tier comes first in the file — adding a second tier would
silently have handed the integration tier the unit tier's numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pytest_timeout import _get_item_settings


CI_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"


def budget_basis_pattern(tier: str) -> re.Pattern[str]:
    """The `budget-basis(<tier>):` marker, with its four declared inputs."""
    return re.compile(
        rf"budget-basis\({re.escape(tier)}\):\s*"
        r"slowest-pass=(?P<slowest>[\d.]+)\s+"
        r"worst-runner-ratio=(?P<ratio>[\d.]+)\s+"
        r"samples=(?P<samples>\d+)\s+"
        r"longest-test=(?P<longest_test>\d+)"
    )


def declared_per_test_ceiling(tier: str) -> int | None:
    """The per-test ceiling, in seconds, ci.yml budgeted *tier* for."""
    try:
        text = CI_WORKFLOW.read_text()
    except OSError:
        return None  # Running outside a checkout; nothing to enforce against.
    match = budget_basis_pattern(tier).search(text)
    return int(match["longest_test"]) if match else None


def _unbounded_reason(item: pytest.Item, ceiling: int) -> str | None:
    """Say how *item* escapes the tier's per-test ceiling, or None.

    The settings come from pytest-timeout itself rather than from
    reading the marker here. Four review rounds on #3212 each found
    another rule this file had modelled wrongly — that a marker can be
    spelled three ways, that `timeout(0)` disables rather than shortens,
    that `timeout(None)` inherits instead of disabling, that the value
    is a float — and every one of them was a fact the library already
    knew. Re-deriving someone else's semantics is the same mistake in a
    new place each time; asking closes the class.
    """
    settings = _get_item_settings(item)

    if settings.timeout is None or settings.timeout <= 0:
        # No timer is armed at all: neither a marker nor the command
        # line bounds this test.
        return "no timeout in effect"
    if settings.func_only:
        # The clock covers the test body only, so a hang in a fixture is
        # unbounded — and the session deadline cannot be checked until
        # the whole protocol for this item finishes.
        return f"{settings.timeout:g}s covers the test body only (func_only)"
    if settings.timeout > ceiling:
        return f"{settings.timeout:g}s"
    return None


def refuse_unbudgeted_timeouts(
    config: pytest.Config, items: list[pytest.Item], tier: str
) -> None:
    """Refuse a per-test timeout the tier's wall-clock budget cannot hold.

    pytest stops the session at `--session-timeout`, and the runner's
    `timeout-minutes` is a backstop it must never reach. The deadline is
    checked BETWEEN tests, so a test that claims a larger per-test timeout
    than the budget allows for can push pytest past the backstop — and the
    runner's kill prints no FAILED line. That makes it a collection-time
    error here rather than a mystery red later.

    Only when the run is actually under that budget: with no
    `--session-timeout` there is no race to lose and nothing to enforce,
    and refusing a hand-run tier would make this guard the reason it
    cannot be run by hand.
    """
    if config.getoption("session_timeout", None) is None:
        return

    ceiling = declared_per_test_ceiling(tier)
    if ceiling is None:
        return

    refused = [
        f"{item.nodeid} ({reason})"
        for item in items
        if (reason := _unbounded_reason(item, ceiling)) is not None
    ]

    if refused:
        raise pytest.UsageError(
            f"These tests are not bounded by the {ceiling}s per-test ceiling "
            f"ci.yml budgets the {tier} tier for, so the session deadline "
            f"could overrun the runner's backstop and the failure would print "
            f"no FAILED line (#3212). Raise `longest-test` in the ci.yml "
            f"`budget-basis({tier})` marker (and the timeout-minutes that "
            f"depends on it), or bound the test: " + ", ".join(sorted(refused))
        )
