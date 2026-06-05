"""Tests for ``TaskFeature.wait`` — the generic bounded wait primitive.

#1541: agents were shelling out to ``sleep`` between polls during
autonomous work loops. ``wait`` is the native replacement: a bounded,
audited pause that enforces a conservative maximum duration and reports
the observed elapsed time.
"""

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.tasks.feature import TaskFeature


class TestGenericWait:
    @pytest.mark.asyncio
    async def test_zero_duration_returns_immediately(self):
        feature = TaskFeature(agent=None)
        result = await feature.wait(duration_seconds=0, reason="probe")
        assert result.status is ToolResultStatus.OK
        assert result.data["requested_seconds"] == 0
        assert result.data["reason"] == "probe"
        assert result.data["elapsed_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_short_wait_reports_elapsed(self):
        feature = TaskFeature(agent=None)
        result = await feature.wait(duration_seconds=1)
        assert result.status is ToolResultStatus.OK
        # Observed elapsed should be at least the requested duration.
        assert result.data["elapsed_seconds"] >= 1
        assert result.data["requested_seconds"] == 1

    @pytest.mark.asyncio
    async def test_max_duration_rejected(self):
        feature = TaskFeature(agent=None)
        too_long = TaskFeature._MAX_WAIT_SECONDS + 1
        result = await feature.wait(duration_seconds=too_long)
        assert result.status is ToolResultStatus.ERROR
        assert "exceeds the maximum" in result.error
        assert result.data["requested_seconds"] == too_long
        assert result.data["max_seconds"] == TaskFeature._MAX_WAIT_SECONDS

    @pytest.mark.asyncio
    async def test_negative_duration_rejected(self):
        feature = TaskFeature(agent=None)
        result = await feature.wait(duration_seconds=-5)
        assert result.status is ToolResultStatus.ERROR
        assert "must be >= 0" in result.error

    @pytest.mark.asyncio
    async def test_non_integer_duration_rejected(self):
        feature = TaskFeature(agent=None)
        result = await feature.wait(duration_seconds="soon")
        assert result.status is ToolResultStatus.ERROR
        assert "must be an integer" in result.error
