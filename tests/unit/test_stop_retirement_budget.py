"""A bounded Stop drain cannot accept an unbounded or invalid clock budget."""

import pytest

from kestrel_sovereign.stop.authority import StopCleanupRegistry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget", [float("nan"), float("inf"), -float("inf"), -1, None, True]
)
async def test_original_stop_drain_rejects_nonfinite_or_invalid_budget(budget):
    registry = StopCleanupRegistry()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        await registry.drain(timeout_seconds=budget)


@pytest.mark.asyncio
async def test_empty_original_stop_drain_allows_immediate_zero_budget():
    registry = StopCleanupRegistry()
    await registry.drain(timeout_seconds=0)
    assert not registry._tasks and registry._terminal_error is None
