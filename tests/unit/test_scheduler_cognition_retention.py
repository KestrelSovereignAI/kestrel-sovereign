"""Unit tests for the ``cognition_retention`` built-in scheduler task (#1674).

Covers the inline handler ``SchedulerFeature._run_cognition_retention``:
the opt-in skip when no window is configured, cutoff computation, the
storage-missing skip, and storage-failure tolerance.

The storage primitive itself (``purge_episodes_older_than``, incl. the KG
fan-out) is exercised against real SQLite in
``tests/integration/test_retention_purge_primitive.py``.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sovereign.features.scheduler.feature import SchedulerFeature


def _make_feature(*, purge_return=0, purge_raises=None, has_storage=True):
    feature = SchedulerFeature.__new__(SchedulerFeature)
    if has_storage:
        storage = SimpleNamespace()
        if purge_raises:
            storage.purge_episodes_older_than = AsyncMock(side_effect=purge_raises)
        else:
            storage.purge_episodes_older_than = AsyncMock(return_value=purge_return)
    else:
        storage = SimpleNamespace()  # no purge_episodes_older_than attribute
    agent = SimpleNamespace(did="did:test:agent", storage=storage)
    feature.agent = agent
    feature._agent_id = agent.did
    return feature, storage


@pytest.mark.asyncio
async def test_skips_when_no_window_configured():
    """Opt-in: with no [retention.cognition].episodes_days, keep episodes
    forever — never call the purge primitive."""
    feature, storage = _make_feature()
    with patch(
        "kestrel_sovereign.storage.retention.load_cognition_config",
        return_value={},
    ):
        result = await feature._run_cognition_retention({})

    storage.purge_episodes_older_than.assert_not_awaited()
    payload = json.loads(result)
    assert payload["skipped"] is True
    assert "episodes_days" in payload["reason"]


@pytest.mark.asyncio
async def test_purges_with_configured_window():
    feature, storage = _make_feature(purge_return=3)
    before = datetime.now(timezone.utc) - timedelta(days=180)
    with patch(
        "kestrel_sovereign.storage.retention.load_cognition_config",
        return_value={"episodes_days": 180},
    ):
        result = await feature._run_cognition_retention({})

    storage.purge_episodes_older_than.assert_awaited_once()
    call = storage.purge_episodes_older_than.await_args
    assert call.kwargs["reason"] == "cognition-retention"
    assert call.kwargs["max_rows"] == 10_000

    payload = json.loads(result)
    assert payload["episodes_purged"] == 3
    assert payload["episodes_days"] == 180
    # Cutoff is ~now-180d, tz-aware isoformat (matches the consolidator's
    # created_at format). Allow a small skew for test execution time.
    cutoff = datetime.fromisoformat(payload["cutoff"])
    assert cutoff.tzinfo is not None
    assert abs((cutoff - before).total_seconds()) < 60


@pytest.mark.asyncio
async def test_honors_max_rows_override():
    feature, storage = _make_feature(purge_return=2)
    with patch(
        "kestrel_sovereign.storage.retention.load_cognition_config",
        return_value={"episodes_days": 90},
    ):
        await feature._run_cognition_retention({"max_rows": 2})
    assert storage.purge_episodes_older_than.await_args.kwargs["max_rows"] == 2


@pytest.mark.asyncio
async def test_skips_when_storage_lacks_primitive():
    feature, _ = _make_feature(has_storage=False)
    result = await feature._run_cognition_retention({})
    payload = json.loads(result)
    assert payload["skipped"] is True
    assert "purge_episodes_older_than" in payload["reason"]


@pytest.mark.asyncio
async def test_tolerates_storage_failure():
    feature, _ = _make_feature(purge_raises=RuntimeError("db locked"))
    with patch(
        "kestrel_sovereign.storage.retention.load_cognition_config",
        return_value={"episodes_days": 90},
    ):
        result = await feature._run_cognition_retention({})
    payload = json.loads(result)
    assert "db locked" in payload["error"]
