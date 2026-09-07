"""The scoped graph purge compares at microsecond precision on both backends (#3227).

Runs on SQLite and, in the integration job, on Postgres through ``db_backend``:
both paths normalise ``properties.created_at`` server-side, and both used to
truncate to whole seconds, so a NORMAL node written earlier in the same
second as the EPHEMERAL transition compared equal to the watermark and was
destroyed. The watermark now carries microseconds and the normalisation
keeps them.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import (
    AsyncGraphStore,
    GraphNode,
    _normalize_purge_watermark,
)


@pytest_asyncio.fixture
async def bound_graph(db_backend):
    """A tenant-bound AsyncGraphStore over the parametrized backend."""
    db = AsyncDatabase(db_backend)
    await db._init_schema()
    db._initialized = True
    agent = f"did:test:purge-{uuid.uuid4().hex}"
    return AsyncGraphStore(db, agent_id=agent), agent


async def _seed(store, agent, stamps: dict[str, str]) -> None:
    for node_id, created_at in stamps.items():
        await store.add_node(GraphNode(
            node_id=f"{agent}:{node_id}", node_type="memory", label=node_id,
            properties={"agent_id": agent, "created_at": created_at},
        ))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_same_second_boundary_on_backend(bound_graph):
    store, agent = bound_graph
    transition = datetime(2026, 9, 7, 12, 0, 5, 500000, tzinfo=timezone.utc)
    await _seed(store, agent, {
        "before": (transition - timedelta(milliseconds=300)).isoformat(),
        "at": transition.isoformat(),
        "after": (transition + timedelta(milliseconds=300)).isoformat(),
        "earlier-second": (transition - timedelta(seconds=1)).isoformat(),
        "whole-second-same": transition.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "whole-second-later": (transition + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S"),
    })
    purged = await store.purge_agent_nodes(
        agent, since_iso=transition.strftime("%Y-%m-%d %H:%M:%S.%f")
    )
    assert purged == 3, purged
    survivors = {
        node_id for node_id in ("before", "at", "after", "earlier-second", "whole-second-same", "whole-second-later")
        if await store.get_node(f"{agent}:{node_id}") is not None
    }
    assert survivors == {"before", "earlier-second", "whole-second-same"}, survivors


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_whole_second_watermark_still_purges_the_whole_second_on_backend(bound_graph):
    """Older callers pass ``YYYY-MM-DD HH:MM:SS``; it reads as ``.000000``,
    so every row in that second and later is purged, as before."""
    store, agent = bound_graph
    transition = datetime(2026, 9, 7, 12, 0, 5, 500000, tzinfo=timezone.utc)
    await _seed(store, agent, {
        "before-in-second": (transition - timedelta(milliseconds=300)).isoformat(),
        "earlier-second": (transition - timedelta(seconds=1)).isoformat(),
    })
    purged = await store.purge_agent_nodes(agent, since_iso="2026-09-07 12:00:05")
    assert purged == 1
    assert await store.get_node(f"{agent}:earlier-second") is not None
    assert await store.get_node(f"{agent}:before-in-second") is None


@pytest.mark.parametrize(
    "given, expected",
    [
        ("2026-09-07 12:00:05", "2026-09-07 12:00:05.000000"),
        ("2026-09-07T12:00:05", "2026-09-07 12:00:05.000000"),
        ("2026-09-07 12:00:05.5", "2026-09-07 12:00:05.500000"),
        ("2026-09-07T12:00:05.123456+00:00", "2026-09-07 12:00:05.123456"),
        ("2026-09-07 12:00:05+00:00", "2026-09-07 12:00:05.000000"),
        ("2026-09-07 12:00:05.1234567", "2026-09-07 12:00:05.123456"),
        (None, None),
        ("", ""),
    ],
)
def test_watermark_normalization(given, expected):
    assert _normalize_purge_watermark(given) == expected


def test_watermark_normalization_refuses_garbage():
    with pytest.raises(ValueError, match="watermark"):
        _normalize_purge_watermark("yesterday")
