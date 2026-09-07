"""Audit-anchor reads are bound to the calling agent (#3230).

`audit_anchors` lives in the main agent storage: one table per agent on
SQLite, one per host on shared PostgreSQL. The audit *entries* an anchor
hashes stay per-agent either way (the PermissionStore's SQLite file), so
an unscoped anchor read mixes another agent's anchors with this agent's
log. Three things went wrong, each with its own read:

- last-anchor: a foreign newer anchor made this agent skip its own
  unanchored entries ("nothing to anchor");
- count: status reported the fleet's anchors;
- enumeration: verify checked a foreign hash against the local log and
  reported an integrity failure that never happened.

These run two agents over one table, through the real anchor/status/
verify tools, on both backends.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import aiosqlite
import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.features.audit_anchor.feature import AuditAnchorFeature


class SecurityFeature:
    """Named so `_get_permission_store` finds it: it matches on the type name."""

    def __init__(self, db_path):
        self.permission_store = SimpleNamespace(db_path=db_path)


class _Storage:
    def __init__(self, db):
        self.db = db
        self.stored = []

    async def store_file(self, data, filename):
        self.stored.append(filename)
        return f"ref-{len(self.stored)}"

    async def add_node(self, node):
        return None


class _Agent:
    """The attributes the feature reads, and nothing else.

    Not a MagicMock — see test_consent_agent_scope.py. `agent_name` is
    deliberately different from `did`.
    """

    def __init__(self, did, db, audit_db_path):
        self.did = did
        self.agent_name = f"display-name-for-{(did or 'nobody')[-8:]}"
        self.storage = _Storage(db)
        self.features = {"SecurityFeature": SecurityFeature(audit_db_path)}


async def _add_audit_entries(db_path, created_ats):
    """Append rows to this agent's own security_audit_log file.

    Same schema as the PermissionStore creates; this file is per agent,
    which is exactly why a foreign anchor can never verify against it.
    """
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS security_audit_log (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   feature_name TEXT, tool_name TEXT, action TEXT,
                   decision TEXT, user_choice TEXT, args_summary TEXT,
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               )"""
        )
        for created_at in created_ats:
            await conn.execute(
                """INSERT INTO security_audit_log
                   (feature_name, tool_name, action, decision, created_at)
                   VALUES ('WalletAgent', 'get_balance', 'tool_execution', 'allowed', ?)""",
                (created_at,),
            )
        await conn.commit()


@pytest.fixture
def identities():
    """Two DIDs nobody else has used (a shared PostgreSQL accumulates rows)."""
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
async def features(db, identities, tmp_path):
    mine, theirs = identities
    mine_audit = str(tmp_path / "mine_permissions.db")
    theirs_audit = str(tmp_path / "theirs_permissions.db")
    mine_feature = AuditAnchorFeature(_Agent(mine, db, mine_audit))
    theirs_feature = AuditAnchorFeature(_Agent(theirs, db, theirs_audit))
    await mine_feature.initialize()
    await theirs_feature.initialize()
    return mine_feature, theirs_feature, mine_audit, theirs_audit


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_foreign_newer_anchor_does_not_suppress_local_anchoring(
    db, features, identities
):
    mine, theirs = identities
    mine_feature, theirs_feature, mine_audit, theirs_audit = features

    await _add_audit_entries(mine_audit, ["2026-01-15T10:00:00", "2026-01-15T10:01:00"])
    first = await mine_feature.anchor_audit()
    assert first.data["status"] == "anchored", first
    assert first.data["entries_count"] == 2

    # The other agent anchors LATER entries: its anchor is the newest row
    # in the shared table, with a last_entry_at past everything of mine.
    await _add_audit_entries(theirs_audit, ["2026-01-15T12:00:00", "2026-01-15T12:01:00"])
    foreign = await theirs_feature.anchor_audit()
    assert foreign.data["status"] == "anchored", foreign

    # A new local entry, older than the foreign anchor's range but newer
    # than my own last anchor. Unscoped, the last-anchor lookup returns
    # the foreign 12:01 and this entry is "already anchored".
    await _add_audit_entries(mine_audit, ["2026-01-15T11:00:00"])
    second = await mine_feature.anchor_audit()
    assert second.data["status"] == "anchored", second
    assert second.data["entries_count"] == 1

    # Rows are stamped with the DID, not the display name.
    stamped = await db.fetchall(
        "SELECT agent_id FROM audit_anchors WHERE id IN (?, ?)",
        (first.data["anchor_id"], second.data["anchor_id"]),
    )
    assert sorted(row[0] for row in stamped) == [mine, mine]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_status_counts_only_the_calling_agents_anchors(features):
    mine_feature, theirs_feature, mine_audit, theirs_audit = features

    await _add_audit_entries(mine_audit, ["2026-01-15T10:00:00"])
    await mine_feature.anchor_audit()
    await _add_audit_entries(mine_audit, ["2026-01-15T10:30:00"])
    await mine_feature.anchor_audit()
    await _add_audit_entries(theirs_audit, ["2026-01-15T12:00:00"])
    await theirs_feature.anchor_audit()

    mine_status = await mine_feature.anchor_status()
    theirs_status = await theirs_feature.anchor_status()

    assert mine_status.status == ToolResultStatus.OK, mine_status
    assert mine_status.data["total_anchors"] == 2, mine_status.data
    assert mine_status.data["entries_since_last"] == 0
    assert mine_status.data["last_anchor_at"].startswith("2026-01-15T10:30:00")
    assert theirs_status.data["total_anchors"] == 1, theirs_status.data
    assert theirs_status.data["last_anchor_at"].startswith("2026-01-15T12:00:00")


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_verify_checks_only_the_calling_agents_anchors(features):
    """A foreign anchor's range selects nothing from the local log.

    Unscoped, verify enumerates the other agent's anchor, finds no local
    entries in its range, records a warning and reports integrity_failure
    for an audit trail that is intact.
    """
    mine_feature, theirs_feature, mine_audit, theirs_audit = features

    await _add_audit_entries(mine_audit, ["2026-01-15T10:00:00", "2026-01-15T10:01:00"])
    await mine_feature.anchor_audit()
    await _add_audit_entries(theirs_audit, ["2026-01-15T12:00:00"])
    await theirs_feature.anchor_audit()

    mine_verify = await mine_feature.verify_audit()
    theirs_verify = await theirs_feature.verify_audit()

    assert mine_verify.status == ToolResultStatus.OK, mine_verify
    assert mine_verify.data["status"] == "verified"
    assert mine_verify.data["total_anchors"] == 1
    assert mine_verify.data["passed"] == 1
    assert mine_verify.data["failed"] == 0
    assert mine_verify.data["warnings"] == 0
    assert theirs_verify.data["status"] == "verified", theirs_verify
    assert theirs_verify.data["total_anchors"] == 1

    # Tampering is still detected for the agent it belongs to: rewrite one
    # of my entries and my verify fails, theirs is untouched.
    async with aiosqlite.connect(mine_audit) as conn:
        await conn.execute(
            "UPDATE security_audit_log SET decision = 'denied' WHERE created_at = '2026-01-15T10:00:00'"
        )
        await conn.commit()
    tampered = await mine_feature.verify_audit()
    assert tampered.status == ToolResultStatus.ERROR, tampered
    assert tampered.data["status"] == "integrity_failure"
    assert (await theirs_feature.verify_audit()).data["status"] == "verified"


@pytest.mark.asyncio
@pytest.mark.parametrize("did", [None, ""])
async def test_missing_identity_refuses_rather_than_reads_unscoped(
    sqlite_backend, tmp_path, did
):
    """No DID refuses loudly; it is not swallowed into "no anchors yet".

    Each helper catches every exception from the query and answers
    None/0/[] — the shape of a fresh install. If the identity refusal
    landed inside that `except`, an agent with no DID would read "never
    anchored", re-anchor its whole log, and stamp the row with nothing.
    """
    audit = str(tmp_path / "permissions.db")
    await _add_audit_entries(audit, ["2026-01-15T10:00:00"])
    db = AsyncDatabase(sqlite_backend)
    feature = AuditAnchorFeature(_Agent(did, db, audit))
    await feature.initialize()

    with pytest.raises(RuntimeError, match="identity"):
        await feature._get_last_anchor_timestamp()
    with pytest.raises(RuntimeError, match="identity"):
        await feature._count_anchors()
    with pytest.raises(RuntimeError, match="identity"):
        await feature._get_all_anchors()
    with pytest.raises(RuntimeError, match="identity"):
        await feature.anchor_audit()
    with pytest.raises(RuntimeError, match="identity"):
        await feature.anchor_status()
    with pytest.raises(RuntimeError, match="identity"):
        await feature.verify_audit()
    assert await db.fetchval("SELECT COUNT(*) FROM audit_anchors") == 0
    assert feature.agent.storage.stored == []
