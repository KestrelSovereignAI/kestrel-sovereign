"""The scoped EPHEMERAL graph purge dates ``created_at`` with one grammar (#3227).

SQLite used to normalise ``properties.created_at`` by text surgery and
PostgreSQL by a ``::timestamptz`` cast, so the two agreed only shape by shape,
and one value the cast could not read aborted the whole Postgres sweep. Now the
database only hands back each owned row's text and
``_purge_created_at_instant`` decides. One table of shapes is therefore asked
of the grammar itself and of the purge on each backend, and every answer must
be the same: purge, keep, or untimed.

Runs on SQLite and, in the integration job, on PostgreSQL through
``db_backend``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import async_graph_store
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import (
    AsyncGraphStore,
    GraphNode,
    _purge_created_at_instant,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

GRAPH_LOGGER = "kestrel_sovereign.storage.async_graph_store"
TRANSITION = datetime(2026, 9, 7, 12, 0, 5, 500000, tzinfo=timezone.utc)
WATERMARK = "2026-09-07 12:00:05.500000"

PURGE, KEEP, UNTIMED = "purge", "keep", "untimed"
#: A row whose properties carry no ``created_at`` key at all.
ABSENT = object()

#: ``(label, created_at, expected)`` against :data:`WATERMARK`. The one table
#: every consumer below is held to.
SHAPES = [
    # The module contract: isoformat() of an aware UTC datetime.
    ("contract-after", "2026-09-07T12:00:05.600000+00:00", PURGE),
    ("contract-at", "2026-09-07T12:00:05.500000+00:00", PURGE),
    ("contract-just-before", "2026-09-07T12:00:05.499999+00:00", KEEP),
    ("contract-whole-second-same", "2026-09-07T12:00:05+00:00", KEEP),
    ("contract-next-second", "2026-09-07T12:00:06+00:00", PURGE),
    ("contract-midnight", "2026-09-07T00:00:00+00:00", KEEP),
    ("contract-previous-day", "2026-09-06T23:59:59.999999+00:00", KEEP),
    ("contract-next-day", "2026-09-08T00:00:00+00:00", PURGE),
    # Z is the same offset as +00:00.
    ("z-after", "2026-09-07T12:00:05.6Z", PURGE),
    ("z-before", "2026-09-07T12:00:05.4Z", KEEP),
    ("z-whole-second-later", "2026-09-07T12:00:06Z", PURGE),
    # Offset-free values are read as UTC, with T or a space.
    ("naive-after", "2026-09-07T12:00:05.6", PURGE),
    ("naive-before", "2026-09-07T12:00:05.4", KEEP),
    ("space-after", "2026-09-07 12:00:05.6", PURGE),
    ("space-whole-second-later", "2026-09-07 12:00:06", PURGE),
    ("space-whole-second-same", "2026-09-07 12:00:05", KEEP),
    # Fractions: padded when short, truncated past six, refused past nine.
    ("fraction-1-at", "2026-09-07T12:00:05.5+00:00", PURGE),
    ("fraction-7-truncated", "2026-09-07T12:00:05.4999999+00:00", KEEP),
    ("fraction-9-at", "2026-09-07T12:00:05.500000999Z", PURGE),
    ("fraction-10", "2026-09-07T12:00:05.5000000000Z", UNTIMED),
    ("fraction-empty", "2026-09-07T12:00:05.+00:00", UNTIMED),
    # Any other offset, or UTC spelled another way.
    ("offset-plus-0530", "2026-09-07T12:00:05.6+05:30", UNTIMED),
    ("offset-minus-0300", "2026-09-07T12:00:05.6-03:00", UNTIMED),
    ("offset-minus-zero", "2026-09-07T12:00:05.6-00:00", UNTIMED),
    ("offset-no-colon", "2026-09-07T12:00:05.6+0000", UNTIMED),
    ("offset-with-seconds", "2026-09-07T12:00:05.6+00:00:00", UNTIMED),
    ("offset-hours-only", "2026-09-07T12:00:05.6+00", UNTIMED),
    ("lowercase-z", "2026-09-07T12:00:05.6z", UNTIMED),
    # Whitespace.
    ("leading-space", " 2026-09-07T12:00:05.6+00:00", UNTIMED),
    ("trailing-space", "2026-09-07T12:00:05.6+00:00 ", UNTIMED),
    ("trailing-newline", "2026-09-07T12:00:05.6+00:00\n", UNTIMED),
    ("double-space-separator", "2026-09-07  12:00:05.6", UNTIMED),
    ("tab-separator", "2026-09-07\t12:00:05.6", UNTIMED),
    # Trailing garbage.
    ("garbage-after-offset", "2026-09-07T12:00:05.6+00:00extra", UNTIMED),
    ("garbage-after-z", "2026-09-07T12:00:05.6Zextra", UNTIMED),
    ("double-z", "2026-09-07T12:00:05.6ZZ", UNTIMED),
    ("garbage-after-fraction", "2026-09-07T12:00:05.6x", UNTIMED),
    ("garbage-after-seconds", "2026-09-07T12:00:06 UTC", UNTIMED),
    ("lowercase-t", "2026-09-07t12:00:05.6+00:00", UNTIMED),
    # Not a timestamp, or not a whole one.
    ("bare-date", "2026-09-07", UNTIMED),
    ("bare-date-later", "2026-09-08", UNTIMED),
    ("empty", "", UNTIMED),
    ("no-seconds", "2026-09-07T12:00", UNTIMED),
    ("prose", "not a timestamp at all", UNTIMED),
    # Well shaped, but naming no instant.
    ("impossible-date", "2026-02-30T12:00:06+00:00", UNTIMED),
    ("hour-24", "2026-09-07T24:00:00+00:00", UNTIMED),
    ("leap-second", "2026-09-07T12:00:60+00:00", UNTIMED),
    ("year-zero", "0000-09-07T12:00:06+00:00", UNTIMED),
    # Digits that are not ASCII.
    ("fullwidth-digits", "２０２６-09-07T12:00:06+00:00", UNTIMED),
    ("arabic-indic-digit", "2026-09-07T12:00:0٦+00:00", UNTIMED),
    # Not a JSON string.
    ("json-integer", 1757246406, UNTIMED),
    ("json-float", 20260907120006.5, UNTIMED),
    ("json-true", True, UNTIMED),
    ("json-null", None, UNTIMED),
    ("json-object", {"at": "2026-09-07T12:00:06+00:00"}, UNTIMED),
    ("json-array", ["2026-09-07T12:00:06+00:00"], UNTIMED),
    ("absent", ABSENT, UNTIMED),
]
EXPECTED = {label: expected for label, _, expected in SHAPES}


def _grammar_class(value) -> str:
    instant = _purge_created_at_instant(None if value is ABSENT else value)
    if instant is None:
        return UNTIMED
    return PURGE if instant >= _purge_created_at_instant(WATERMARK) else KEEP


def _properties(agent: str, created_at) -> dict:
    properties = {"agent_id": agent}
    if created_at is not ABSENT:
        properties["created_at"] = created_at
    return properties


def _untimed_reported(caplog, agent: str) -> int:
    """The untimed count the purge's WARNING reported for ``agent``."""
    counts = [
        rec.args[0] for rec in caplog.records
        if rec.name == GRAPH_LOGGER
        and "purge_agent_nodes (scoped)" in rec.msg
        and rec.args[1] == agent
    ]
    assert len(counts) <= 1, counts
    return counts[0] if counts else 0


@pytest_asyncio.fixture
async def graph_db(db_backend):
    db = AsyncDatabase(db_backend)
    await db._init_schema()
    db._initialized = True
    return db


@pytest_asyncio.fixture
async def storage(db_backend):
    store = AsyncStorage(backend=db_backend, agent_id=f"did:test:grammar-{uuid.uuid4().hex}")
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


def _pin_transition(monkeypatch, instant: datetime = TRANSITION) -> None:
    """Pin the clock the wrapper reads its watermark from."""
    monkeypatch.setattr(PrivacyEnforcingStorage, "_now_utc", staticmethod(lambda: instant))


# ---------------------------------------------------------------------------
# One table, three readers
# ---------------------------------------------------------------------------


def test_the_grammar_classifies_the_shape_table():
    observed = {label: _grammar_class(value) for label, value, _ in SHAPES}
    assert observed == EXPECTED


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_purge_classifies_the_shape_table_identically_on_backend(graph_db, caplog):
    """Each shape under its own agent, so purge, keep and untimed are read per
    row: purged is the row gone; untimed is the row kept AND counted by the
    purge's WARNING; keep is the row kept and not counted."""
    observed = {}
    with caplog.at_level(logging.WARNING, logger=GRAPH_LOGGER):
        for label, value, _ in SHAPES:
            agent = f"did:test:shape-{uuid.uuid4().hex}"
            store = AsyncGraphStore(graph_db, agent_id=agent)
            await store.add_node(GraphNode(
                node_id=f"{agent}:row", node_type="memory", label=label,
                properties=_properties(agent, value),
            ))
            purged = await store.purge_agent_nodes(agent, since_iso=WATERMARK)
            gone = await store.get_node(f"{agent}:row") is None
            untimed = _untimed_reported(caplog, agent)
            assert purged == int(gone), (label, purged, gone)
            if gone:
                observed[label] = PURGE if untimed == 0 else f"purged-and-untimed({untimed})"
            else:
                observed[label] = {0: KEEP, 1: UNTIMED}.get(untimed, f"untimed({untimed})")
    assert observed == EXPECTED


# ---------------------------------------------------------------------------
# The ticket's boundary, end to end through the wrapper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_same_second_pre_transition_node_survives_on_backend(storage, monkeypatch):
    """A NORMAL node stamped 300 ms BEFORE the transition, in the same second,
    survives the EPHEMERAL exit; nodes stamped at the transition instant and
    300 ms after it are purged. A whole-second comparison treated all three
    alike and destroyed the first (#3227)."""
    agent = storage.agent_id
    await storage.graph.add_node(GraphNode(
        node_id=f"{agent}:before", node_type="memory", label="NORMAL",
        properties={"agent_id": agent,
                    "created_at": (TRANSITION - timedelta(milliseconds=300)).isoformat()},
    ))
    _pin_transition(monkeypatch)
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    assert wrapper._entered_ephemeral_at == WATERMARK

    for name, stamp in (("at", TRANSITION), ("after", TRANSITION + timedelta(milliseconds=300))):
        await storage.graph.add_node(GraphNode(
            node_id=f"{agent}:{name}", node_type="memory", label="leak",
            properties={"agent_id": agent, "created_at": stamp.isoformat()},
        ))

    report = await wrapper.purge_ephemeral_session(reason="test-same-second")

    assert not report.required_sweep_failed, report.failed_stores
    assert report["graph_nodes"] == 2, dict(report)
    assert await storage.graph.get_node(f"{agent}:before") is not None, (
        "a NORMAL node written earlier in the transition second was purged (#3227)"
    )
    assert await storage.graph.get_node(f"{agent}:at") is None
    assert await storage.graph.get_node(f"{agent}:after") is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_row_with_trailing_garbage_does_not_block_ephemeral_exit_on_backend(
    storage, monkeypatch, caplog
):
    """One out-of-grammar ``created_at`` used to fail the Postgres cast and with
    it the whole graph sweep, so the required sweep FAILED and the agent could
    not leave EPHEMERAL. It is untimed now: preserved, counted, and the real
    leak beside it is still purged."""
    agent = storage.agent_id
    _pin_transition(monkeypatch)
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    for name, stamp in (
        ("garbage", "2026-09-07T12:00:06+00:00extra"),
        ("leak", "2026-09-07T12:00:06+00:00"),
    ):
        await storage.graph.add_node(GraphNode(
            node_id=f"{agent}:{name}", node_type="memory", label=name,
            properties={"agent_id": agent, "created_at": stamp},
        ))

    with caplog.at_level(logging.WARNING, logger=GRAPH_LOGGER):
        report = await wrapper.purge_ephemeral_session(reason="test-garbage")

    assert not report.required_sweep_failed, report.failed_stores
    assert report["graph_nodes"] == 1, dict(report)
    assert await storage.graph.get_node(f"{agent}:leak") is None
    assert await storage.graph.get_node(f"{agent}:garbage") is not None
    assert _untimed_reported(caplog, agent) == 1
    # The sweep certified the stint, so the watermark is retired and the mode
    # can change back.
    assert wrapper._entered_ephemeral_at is None
    wrapper.set_privacy_mode(PrivacyMode.NORMAL)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_an_offset_free_value_is_utc_on_the_purges_own_connection_on_backend(graph_db):
    """Replaces a test that could not fail: it set the Postgres session
    TimeZone on a pooled connection that was reset before the purge ran.

    Here the zone is set inside a transaction the purge joins, and the test
    first proves the hazard is live on that connection: a ``timestamptz``
    cast reads an offset-free value four hours late there. The purge reads
    it as UTC anyway, because no SQL interprets the value."""
    agent = f"did:test:zone-{uuid.uuid4().hex}"
    store = AsyncGraphStore(graph_db, agent_id=agent)
    for name, stamp in (("before", "2026-09-07T12:00:05.400000"),
                        ("after", "2026-09-07T12:00:05.600000")):
        await store.add_node(GraphNode(
            node_id=f"{agent}:{name}", node_type="memory", label=name,
            properties={"agent_id": agent, "created_at": stamp},
        ))
    postgres = graph_db.backend_type == "postgres"
    scope = graph_db.transaction() if postgres else contextlib.nullcontext()
    async with scope:
        if postgres:
            await graph_db.execute("SET LOCAL TIME ZONE 'America/New_York'")
            shifted = await graph_db.fetchval(
                "SELECT to_char(('2026-09-07T12:00:05.4'::timestamptz) AT TIME ZONE 'UTC', "
                "'HH24:MI:SS.US')"
            )
            assert shifted == "16:00:05.400000", shifted
        purged = await store.purge_agent_nodes(agent, since_iso=WATERMARK)
        if postgres:
            assert await graph_db.fetchval("SHOW TIME ZONE") == "America/New_York"
    assert purged == 1
    assert await store.get_node(f"{agent}:before") is not None
    assert await store.get_node(f"{agent}:after") is None


# ---------------------------------------------------------------------------
# The scan is paged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "owned, pages",
    [(7, [2, 2, 2, 1]), (6, [2, 2, 2, 0])],
    ids=["partial-last-page", "full-last-page"],
)
async def test_the_scan_reads_owned_nodes_in_bounded_pages_on_backend(
    graph_db, monkeypatch, owned, pages
):
    """The scoped purge dates every node an agent owns on every EPHEMERAL exit,
    so it reads them a page at a time rather than in one read of the whole
    graph. With a page of two, no read returns more than two rows, the leaks
    on every page are found (rows 1, 2, 5 and 6: pages are [0, 1], [2, 3],
    [4, 5], [6], so a leak sits at the end of one page and at the start of
    the next), and another agent's nodes are neither read nor touched."""
    monkeypatch.setattr(async_graph_store, "_PURGE_SCAN_PAGE", 2)
    agent = f"did:test:paged-{uuid.uuid4().hex}"
    other = f"did:test:paged-other-{uuid.uuid4().hex}"
    leak = (TRANSITION + timedelta(seconds=1)).isoformat()
    normal = (TRANSITION - timedelta(seconds=1)).isoformat()
    leaks = [i for i in range(owned) if i % 4 in (1, 2)]
    store = AsyncGraphStore(graph_db, agent_id=agent)
    for i in range(owned):
        await store.add_node(GraphNode(
            node_id=f"{agent}:{i:02d}", node_type="memory", label=str(i),
            properties={"agent_id": agent, "created_at": leak if i in leaks else normal},
        ))
    other_store = AsyncGraphStore(graph_db, agent_id=other)
    for i in range(3):
        await other_store.add_node(GraphNode(
            node_id=f"{other}:{i:02d}", node_type="memory", label=str(i),
            properties={"agent_id": other, "created_at": leak},
        ))

    scan_reads = []
    fetchall = graph_db.fetchall

    async def recording_fetchall(sql, *args, **kwargs):
        rows = await fetchall(sql, *args, **kwargs)
        if "created_at" in sql and "graph_node_owners" in sql:
            scan_reads.append(len(rows))
        return rows

    monkeypatch.setattr(graph_db, "fetchall", recording_fetchall)
    purged = await asyncio.wait_for(
        store.purge_agent_nodes(agent, since_iso=WATERMARK), timeout=60
    )
    monkeypatch.setattr(graph_db, "fetchall", fetchall)

    assert scan_reads == pages
    assert purged == len(leaks)
    survivors = [
        i for i in range(owned) if await store.get_node(f"{agent}:{i:02d}") is not None
    ]
    assert survivors == [i for i in range(owned) if i not in leaks]
    for i in range(3):
        assert await other_store.get_node(f"{other}:{i:02d}") is not None


# ---------------------------------------------------------------------------
# The watermark is read by the same grammar
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "watermark",
    [
        "2026-09-07 12:00:05.500000",
        "2026-09-07T12:00:05.500000+00:00",
        "2026-09-07T12:00:05.5",
        "2026-09-07T12:00:05.5Z",
    ],
    ids=["wrapper-shape", "iso-offset", "short-fraction", "short-fraction-z"],
)
async def test_every_watermark_spelling_scopes_the_same_rows_on_backend(graph_db, watermark):
    agent = f"did:test:watermark-{uuid.uuid4().hex}"
    store = AsyncGraphStore(graph_db, agent_id=agent)
    stamps = {
        "before": (TRANSITION - timedelta(milliseconds=300)).isoformat(),
        "at": TRANSITION.isoformat(),
        "after": (TRANSITION + timedelta(milliseconds=300)).isoformat(),
        "earlier-second": (TRANSITION - timedelta(seconds=1)).isoformat(),
    }
    for name, stamp in stamps.items():
        await store.add_node(GraphNode(
            node_id=f"{agent}:{name}", node_type="memory", label=name,
            properties={"agent_id": agent, "created_at": stamp},
        ))
    assert await store.purge_agent_nodes(agent, since_iso=watermark) == 2
    survivors = {name for name in stamps if await store.get_node(f"{agent}:{name}") is not None}
    assert survivors == {"before", "earlier-second"}


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_whole_second_watermark_reads_as_the_seconds_start_on_backend(graph_db):
    """Older callers pass ``YYYY-MM-DD HH:MM:SS``; it reads as ``.000000``."""
    agent = f"did:test:whole-{uuid.uuid4().hex}"
    store = AsyncGraphStore(graph_db, agent_id=agent)
    for name, stamp in (("in-second", (TRANSITION - timedelta(milliseconds=300)).isoformat()),
                        ("earlier-second", (TRANSITION - timedelta(seconds=1)).isoformat())):
        await store.add_node(GraphNode(
            node_id=f"{agent}:{name}", node_type="memory", label=name,
            properties={"agent_id": agent, "created_at": stamp},
        ))
    assert await store.purge_agent_nodes(agent, since_iso="2026-09-07 12:00:05") == 1
    assert await store.get_node(f"{agent}:earlier-second") is not None
    assert await store.get_node(f"{agent}:in-second") is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "watermark",
    ["2026-09-07", "yesterday", "2026-09-07T12:00:05.5+05:30", "2026-09-07 12:00:05 "],
)
async def test_an_undatable_watermark_is_refused_before_anything_is_destroyed_on_backend(
    graph_db, watermark
):
    agent = f"did:test:refused-{uuid.uuid4().hex}"
    store = AsyncGraphStore(graph_db, agent_id=agent)
    await store.add_node(GraphNode(
        node_id=f"{agent}:row", node_type="memory", label="row",
        properties={"agent_id": agent, "created_at": "2026-09-07T12:00:06+00:00"},
    ))
    with pytest.raises(ValueError, match="purge watermark"):
        await store.purge_agent_nodes(agent, since_iso=watermark)
    assert await store.get_node(f"{agent}:row") is not None
