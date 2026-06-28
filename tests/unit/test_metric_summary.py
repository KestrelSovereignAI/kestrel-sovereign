"""Unit tests for ObservabilityStore.get_metric_summary (#969 item 2).

The ``assistant_turn_persist_failed`` metric (streaming.py) shipped "dark" —
emitted but with no read surface. ``get_metric_summary`` is the generic,
backend-portable read path that makes any named metric observable.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.a2a.stores.unified.observability_store import (
    ObservabilityStore,
)
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


async def _store(tmp_path, name):
    backend = SQLiteBackend(str(tmp_path / name))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    return store


@pytest.mark.asyncio
async def test_metric_summary_counts_aggregates_and_carries_forensic_metadata(tmp_path):
    store = await _store(tmp_path, "metric-summary.db")

    # Two persist failures on emma, one on meridian...
    await store.log_metric(
        agent_name="did:test:emma",
        metric_name="assistant_turn_persist_failed",
        metric_value=1.0,
        metadata={"session_id": "s-1", "error_type": "TimeoutError", "error_msg": "boom"},
    )
    await store.log_metric(
        agent_name="did:test:emma",
        metric_name="assistant_turn_persist_failed",
        metric_value=1.0,
        metadata={"session_id": "s-2", "error_type": "OperationalError"},
    )
    await store.log_metric(
        agent_name="did:test:meridian",
        metric_name="assistant_turn_persist_failed",
        metric_value=1.0,
        metadata={"session_id": "s-3", "error_type": "TimeoutError"},
    )
    # ...plus an unrelated metric that must NOT be counted.
    await store.log_metric(
        agent_name="did:test:emma",
        metric_name="feature_tools_built_streaming",
        metric_value=7.0,
        metadata={},
    )

    summary = await store.get_metric_summary("assistant_turn_persist_failed")

    assert summary["metric_name"] == "assistant_turn_persist_failed"
    assert summary["count"] == 3
    assert summary["total_value"] == 3.0
    assert summary["by_agent"] == {"did:test:emma": 2, "did:test:meridian": 1}
    assert summary["last_seen"] is not None
    assert summary["first_seen"] is not None
    assert summary["truncated"] is False
    # Samples carry the forensic metadata, minus the injected name/value keys.
    assert len(summary["samples"]) == 3
    sample_meta_keys = set().union(*(s["metadata"].keys() for s in summary["samples"]))
    assert "error_type" in sample_meta_keys
    assert "session_id" in sample_meta_keys
    assert "metric_name" not in sample_meta_keys
    assert "metric_value" not in sample_meta_keys


@pytest.mark.asyncio
async def test_metric_summary_filters_by_agent(tmp_path):
    store = await _store(tmp_path, "metric-summary-agent.db")
    for agent in ("did:test:emma", "did:test:emma", "did:test:meridian"):
        await store.log_metric(
            agent_name=agent,
            metric_name="assistant_turn_persist_failed",
            metric_value=1.0,
            metadata={"session_id": "x"},
        )

    summary = await store.get_metric_summary(
        "assistant_turn_persist_failed", agent_name="did:test:emma"
    )
    assert summary["count"] == 2
    assert summary["by_agent"] == {"did:test:emma": 2}


@pytest.mark.asyncio
async def test_metric_summary_empty_when_metric_never_emitted(tmp_path):
    store = await _store(tmp_path, "metric-summary-empty.db")
    await store.log_metric(
        agent_name="did:test:emma",
        metric_name="some_other_metric",
        metric_value=1.0,
        metadata={},
    )

    summary = await store.get_metric_summary("assistant_turn_persist_failed")
    assert summary["count"] == 0
    assert summary["total_value"] == 0.0
    assert summary["by_agent"] == {}
    assert summary["first_seen"] is None
    assert summary["last_seen"] is None
    assert summary["samples"] == []
    assert summary["truncated"] is False


@pytest.mark.asyncio
async def test_metric_summary_truncated_flag_when_window_exceeds_limit(tmp_path):
    store = await _store(tmp_path, "metric-summary-trunc.db")
    for i in range(5):
        await store.log_metric(
            agent_name="did:test:emma",
            metric_name="assistant_turn_persist_failed",
            metric_value=1.0,
            metadata={"session_id": str(i)},
        )

    # limit smaller than the number of metric events in the window.
    summary = await store.get_metric_summary(
        "assistant_turn_persist_failed", limit=3
    )
    assert summary["truncated"] is True
    # count is a lower bound bounded by the limit window.
    assert summary["count"] <= 3
