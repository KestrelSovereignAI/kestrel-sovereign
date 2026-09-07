"""Consent reads return only the calling agent's rows (#3229).

`consent_log` is one table per *database*. On a SQLite-per-agent host the
file boundary scoped it for free, so `consent_log` and every aggregate
behind `consent_stats` read the table with no agent predicate and were
never wrong. On a shared PostgreSQL backend one table serves the whole
host, and the same reads returned the fleet: another agent's reflections
in the log, and its timeouts in this agent's reliability finding.

So these seed ONE store with TWO agents' records — through the feature's
own write path — and read through the real tools, on both backends.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.features.consent.feature import ConsentFeature
from kestrel_sovereign.features.consent.models import ConsentRecord


class _Agent:
    """The attributes the feature reads, and nothing else.

    Not a MagicMock: a MagicMock answers every attribute with a truthy
    object, so a feature reading the wrong identity field would still get
    *something* and the two agents would still differ. Here a wrong field
    is an AttributeError. `agent_name` is deliberately different from
    `did`: rows are written under the DID, so a read scoped by the display
    name returns nothing, and a write stamped with it is unreadable.
    """

    def __init__(self, did, db):
        self.did = did
        self.agent_name = f"display-name-for-{(did or 'nobody')[-8:]}"
        self.features = {}
        self.storage = SimpleNamespace(db=db)


def _record(action_type, *, sentiment, duration_ms, timed_out):
    return ConsentRecord(
        id=uuid4().hex[:12],
        action_type=action_type,
        action_details={"why": action_type},
        agent_view="[TIMEOUT]" if timed_out else "This seems fine.",
        agent_sentiment=sentiment,
        timestamp=datetime.now(timezone.utc).isoformat(),
        duration_ms=duration_ms,
        timed_out=timed_out,
    )


@pytest.fixture
def identities():
    """Two DIDs nobody else has used.

    A shared PostgreSQL database is not torn down between tests, so fixed
    identities would accumulate rows from every earlier run and break the
    exact counts below. Unique DIDs give isolation without truncating a
    table this test does not own.
    """
    unique = uuid4().hex[:12]
    return (
        f"did:pkh:eip155:1:0xMINE{unique}",
        f"did:pkh:eip155:1:0xTHEIRS{unique}",
    )


@pytest.fixture
def db(db_backend):
    """The handle a feature actually receives: the AsyncDatabase wrapper
    over the backend, which is what `resolve_feature_database` returns in
    production. The raw backend has no `fetchall`."""
    return AsyncDatabase(db_backend)


@pytest.fixture
async def features(db, identities):
    """Two features over one table, each having written ten records.

    Mine: ten clean consents, durations 100..109 ms. Theirs: ten timeouts
    at 1 ms — below every duration of mine, so an unscoped p95 selection
    (which sorts ascending) lands on a foreign row rather than by luck on
    one of mine, and an unscoped timeout count crosses the PARTIAL
    threshold (>10% of >=10 records) that a correctly scoped read of mine
    must not.
    """
    mine, theirs = identities
    mine_feature = ConsentFeature(_Agent(mine, db))
    theirs_feature = ConsentFeature(_Agent(theirs, db))
    await mine_feature.initialize()
    await theirs_feature.initialize()

    for i in range(10):
        await mine_feature._store_record(_record(
            f"mine-action-{mine[-8:]}", sentiment="positive",
            duration_ms=100.0 + i, timed_out=False,
        ))
    for _ in range(10):
        await theirs_feature._store_record(_record(
            f"theirs-action-{theirs[-8:]}", sentiment="timeout",
            duration_ms=1.0, timed_out=True,
        ))
    return mine_feature, theirs_feature


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_consent_log_lists_only_the_calling_agents_records(
    db, features, identities
):
    mine, theirs = identities
    mine_feature, theirs_feature = features

    mine_result = await mine_feature.consent_log(limit=50)
    theirs_result = await theirs_feature.consent_log(limit=50)

    assert mine_result.status == ToolResultStatus.OK, mine_result
    assert theirs_result.status == ToolResultStatus.OK, theirs_result
    assert mine_result.data["count"] == 10, mine_result.data
    assert theirs_result.data["count"] == 10, theirs_result.data
    assert {r["agent_id"] for r in mine_result.data["records"]} == {mine}
    assert {r["agent_id"] for r in theirs_result.data["records"]} == {theirs}
    assert {r["action_type"] for r in mine_result.data["records"]} == {
        f"mine-action-{mine[-8:]}"
    }

    # The column carries the DID, not the display name: a feature that
    # wrote and read the same wrong field would pass the two-agent
    # differential above while producing rows the rest of the system
    # (which knows this agent by its DID) could never attribute.
    stamped = await db.fetchall(
        "SELECT DISTINCT agent_id FROM consent_log WHERE action_type = ?",
        (f"mine-action-{mine[-8:]}",),
    )
    assert [row[0] for row in stamped] == [mine]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_consent_stats_aggregate_only_the_calling_agents_records(
    features, identities
):
    mine, theirs = identities
    mine_feature, theirs_feature = features

    mine_stats = await mine_feature.consent_stats()
    theirs_stats = await theirs_feature.consent_stats()

    # Mine: clean. Unscoped, the ten foreign timeouts make this 10/20 =
    # 50% and the tool reports PARTIAL — a reliability finding about
    # someone else's consent loop, spoken as this agent's own.
    assert mine_stats.status == ToolResultStatus.OK, mine_stats
    data = mine_stats.data
    assert data["total"] == 10
    assert data["by_action"] == {f"mine-action-{mine[-8:]}": 10}
    assert data["by_sentiment"] == {"positive": 10}
    assert data["timeout_count"] == 0
    assert data["timeout_rate"] == 0.0
    assert data["avg_duration_ms"] == 104.5
    # Ten durations 100..109; p95 offset = int(10 * 0.95) - 1 = 8 → 108.
    # Unscoped count (20) pushes the offset past this agent's rows and the
    # p95 becomes None; an unscoped selection sorts the foreign 1 ms rows
    # first and returns 1.0.
    assert data["p95_duration_ms"] == 108.0

    # Theirs: every record a timeout, so the PARTIAL honesty path fires for
    # them — and only for them.
    assert theirs_stats.status == ToolResultStatus.PARTIAL, theirs_stats
    data = theirs_stats.data
    assert data["total"] == 10
    assert data["by_action"] == {f"theirs-action-{theirs[-8:]}": 10}
    assert data["by_sentiment"] == {"timeout": 10}
    assert data["timeout_count"] == 10
    assert data["timeout_rate"] == 1.0
    assert data["avg_duration_ms"] == 1.0
    assert data["p95_duration_ms"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("did", [None, ""])
async def test_missing_identity_refuses_rather_than_reads_unscoped(
    sqlite_backend, did
):
    """No DID is "cannot be scoped", not "unscoped".

    The observability fix (#3215) found its own defect here: a store that
    gates on truthiness treats an empty identity as *no* predicate. Both
    tools refuse with the reason, and the write refuses too, so the
    feature can never mint a row nobody can read back.
    """
    db = AsyncDatabase(sqlite_backend)
    feature = ConsentFeature(_Agent(did, db))
    await feature.initialize()

    log = await feature.consent_log()
    assert log.status == ToolResultStatus.ERROR, log
    assert "identity" in (log.error or "")

    stats = await feature.consent_stats()
    assert stats.status == ToolResultStatus.ERROR, stats
    assert "identity" in (stats.error or "")

    with pytest.raises(RuntimeError, match="identity"):
        await feature._store_record(_record(
            "orphan", sentiment="neutral", duration_ms=1.0, timed_out=False
        ))
    assert await db.fetchval("SELECT COUNT(*) FROM consent_log") == 0
