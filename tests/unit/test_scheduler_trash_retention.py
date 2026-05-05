"""Unit tests for the ``trash_retention`` built-in scheduler task (#764).

The retention sweep runs as a per-agent cron job through the existing
SchedulerFeature — same shape as ``backup_snapshot``. These tests
cover the inline handler ``SchedulerFeature._run_trash_retention``:
config resolution, cutoff computation, the skip-on-bad-config rail,
and storage failure tolerance.

The storage primitive itself (``purge_trash_older_than``) is exercised
against real SQLite in
``tests/integration/test_retention_purge_primitive.py``.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sovereign.features.scheduler.feature import SchedulerFeature


def _make_feature(*, privacy_mode="normal", purge_return=0, purge_raises=None):
    """Build a SchedulerFeature with just enough surface to invoke the
    trash_retention handler.

    Skips ``initialize()`` (which would spawn the background runner) —
    we're testing the inline handler in isolation, not the runner."""
    feature = SchedulerFeature.__new__(SchedulerFeature)

    storage = SimpleNamespace()
    if purge_raises:
        storage.purge_trash_older_than = AsyncMock(side_effect=purge_raises)
    else:
        storage.purge_trash_older_than = AsyncMock(return_value=purge_return)

    agent = SimpleNamespace(
        did="did:test:agent",
        storage=storage,
        _privacy_mode=SimpleNamespace(value=privacy_mode) if privacy_mode else None,
    )
    feature.agent = agent
    feature._agent_id = agent.did
    return feature, storage


@pytest.mark.asyncio
async def test_trash_retention_passes_privacy_aware_cutoff_to_storage():
    """NORMAL agent + 30-day window → cutoff is now-30d. Verifies the
    handler reads the privacy mode, resolves the window through the
    config helper, and hands the cutoff to the storage facade."""
    feature, storage = _make_feature(privacy_mode="normal", purge_return=4)

    with patch(
        "kestrel_sovereign.storage.retention.load_trash_config",
        return_value={"conversation_history_days": 30},
    ):
        result = await feature._run_trash_retention({})

    storage.purge_trash_older_than.assert_awaited_once()
    call = storage.purge_trash_older_than.await_args
    assert call.kwargs["reason"] == "retention-janitor"
    assert call.kwargs["max_rows"] == 10_000  # default cap

    payload = json.loads(result)
    assert payload["rows_purged"] == 4
    assert payload["privacy_mode"] == "normal"
    assert payload["retention_days"] == 30


@pytest.mark.asyncio
async def test_trash_retention_honors_per_privacy_override():
    """ISOLATED override (7 days) wins over the global 30-day default."""
    feature, storage = _make_feature(privacy_mode="isolated")

    with patch(
        "kestrel_sovereign.storage.retention.load_trash_config",
        return_value={
            "conversation_history_days": 30,
            "privacy_overrides": {"isolated": 7},
        },
    ):
        result = await feature._run_trash_retention({})

    payload = json.loads(result)
    assert payload["retention_days"] == 7
    assert payload["privacy_mode"] == "isolated"


@pytest.mark.asyncio
async def test_trash_retention_skips_on_zero_or_negative_window():
    """Refuse-to-run rail — the operator's TOML cannot accidentally
    nuke the table. Handler skips with a "skipped" payload and never
    touches storage."""
    feature, storage = _make_feature(privacy_mode="normal")

    with patch(
        "kestrel_sovereign.storage.retention.load_trash_config",
        return_value={"conversation_history_days": 0},
    ):
        result = await feature._run_trash_retention({})

    storage.purge_trash_older_than.assert_not_awaited()
    payload = json.loads(result)
    assert payload["skipped"] is True
    assert "non-positive" in payload["reason"]


@pytest.mark.asyncio
async def test_trash_retention_accepts_max_rows_override_via_args():
    """Operator can pass per-task overrides through schedule_add args.
    A larger or smaller cap lets the multi_agent tune sweep cost per agent."""
    feature, storage = _make_feature(privacy_mode="normal")

    with patch(
        "kestrel_sovereign.storage.retention.load_trash_config",
        return_value={"conversation_history_days": 30},
    ):
        await feature._run_trash_retention({"max_rows": 250})

    assert storage.purge_trash_older_than.await_args.kwargs["max_rows"] == 250


@pytest.mark.asyncio
async def test_trash_retention_swallows_storage_exception():
    """A DB lock or transient hiccup must not bubble out of the
    scheduler — the next tick should fire normally. Handler returns
    a JSON error payload that lands in the task-execution log."""
    feature, storage = _make_feature(
        privacy_mode="normal",
        purge_raises=RuntimeError("DB locked"),
    )

    with patch(
        "kestrel_sovereign.storage.retention.load_trash_config",
        return_value={"conversation_history_days": 30},
    ):
        result = await feature._run_trash_retention({})

    payload = json.loads(result)
    assert "DB locked" in payload["error"]


@pytest.mark.asyncio
async def test_trash_retention_skips_when_storage_lacks_primitive():
    """Belt-and-braces: an agent whose storage facade is missing
    ``purge_trash_older_than`` (legacy fixture, slim test setup,
    pre-#763 install) is reported as skipped instead of crashing.
    """
    feature = SchedulerFeature.__new__(SchedulerFeature)
    feature.agent = SimpleNamespace(
        did="did:test:agent",
        storage=SimpleNamespace(),  # no purge_trash_older_than
        _privacy_mode=SimpleNamespace(value="normal"),
    )
    feature._agent_id = feature.agent.did

    with patch(
        "kestrel_sovereign.storage.retention.load_trash_config",
        return_value={"conversation_history_days": 30},
    ):
        result = await feature._run_trash_retention({})

    payload = json.loads(result)
    assert payload["skipped"] is True
    assert "purge_trash_older_than" in payload["reason"]


@pytest.mark.asyncio
async def test_trash_retention_falls_back_to_default_when_no_config():
    """No ``[trash]`` section in kestrel.toml → use the compiled-in
    default (30 days). Operators get retention out of the box without
    having to author config first."""
    feature, storage = _make_feature(privacy_mode="normal")

    with patch(
        "kestrel_sovereign.storage.retention.load_trash_config",
        return_value={},
    ):
        result = await feature._run_trash_retention({})

    payload = json.loads(result)
    assert payload["retention_days"] == 30
