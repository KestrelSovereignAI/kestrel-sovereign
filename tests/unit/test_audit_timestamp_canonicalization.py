"""F092: audit timestamps are canonicalized to UTC ISO-8601 so entries from
security_audit_log (CURRENT_TIMESTAMP) and destructive_audit_log (isoformat)
sort/compare correctly for anchor boundaries and verify ranges."""

import aiosqlite
import pytest

from kestrel_sovereign.features.security.permissions import PermissionStore
from kestrel_sovereign.audit_time import (
    normalize_audit_timestamp,
    utc_now_iso,
)


def test_normalize_sqlite_and_iso_agree():
    # Same instant, two source formats → identical canonical string.
    sqlite_form = "2026-07-03 14:00:05"
    iso_form = "2026-07-03T14:00:05+00:00"
    assert normalize_audit_timestamp(sqlite_form) == normalize_audit_timestamp(iso_form)
    assert normalize_audit_timestamp(sqlite_form) == "2026-07-03T14:00:05+00:00"


def test_normalize_is_idempotent_and_handles_z_and_empty():
    canon = normalize_audit_timestamp("2026-07-03 14:00:05")
    assert normalize_audit_timestamp(canon) == canon
    assert normalize_audit_timestamp("2026-07-03T14:00:05Z") == "2026-07-03T14:00:05+00:00"
    assert normalize_audit_timestamp("") == ""
    assert normalize_audit_timestamp(None) == ""


def test_normalize_fixes_cross_format_ordering():
    # The bug: a CURRENT_TIMESTAMP row (space) at a LATER instant string-sorted
    # BEFORE an ISO row at an earlier instant (space 0x20 < 'T' 0x54).
    later_sqlite = "2026-07-03 14:00:09"
    earlier_iso = "2026-07-03T14:00:03+00:00"
    assert later_sqlite < earlier_iso  # raw string comparison is wrong
    assert normalize_audit_timestamp(later_sqlite) > normalize_audit_timestamp(earlier_iso)


def test_utc_now_iso_is_canonical():
    s = utc_now_iso()
    assert "T" in s and s.endswith("+00:00")
    assert normalize_audit_timestamp(s) == s


@pytest.mark.asyncio
async def test_log_decision_writes_canonical_timestamp(tmp_path):
    store = PermissionStore(str(tmp_path / "perm.db"))
    await store.initialize()
    await store.log_decision("F", "t", "tool_execution", "allowed")
    async with aiosqlite.connect(store.db_path) as db:
        cur = await db.execute("SELECT created_at FROM security_audit_log")
        (ts,) = await cur.fetchone()
    assert "T" in ts and ts.endswith("+00:00")  # ISO, not 'YYYY-MM-DD HH:MM:SS'


@pytest.mark.asyncio
async def test_initialize_does_not_mutate_legacy_row_bytes(tmp_path):
    """Legacy rows must NOT be rewritten: AuditHasher hashes raw created_at, so
    mutating it would break every pre-existing cryptographic anchor (codex P1).
    Correctness comes from normalize-on-read, not from migrating bytes."""
    db_path = str(tmp_path / "perm.db")
    await PermissionStore(db_path).initialize()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO security_audit_log (feature_name, tool_name, action, "
            "decision, created_at) VALUES ('F','t','a','allowed','2026-07-03 14:00:05')"
        )
        await db.commit()
    # Re-initialize (would run any migration) — the raw bytes stay identical.
    await PermissionStore(db_path).initialize()
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT created_at FROM security_audit_log WHERE feature_name='F'"
        )
        (ts,) = await cur.fetchone()
    assert ts == "2026-07-03 14:00:05"  # untouched
    # ...but it still normalizes correctly for comparison.
    assert normalize_audit_timestamp(ts) == "2026-07-03T14:00:05+00:00"
