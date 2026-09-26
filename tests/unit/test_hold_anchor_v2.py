"""Whole-history anchor v2: ``authority`` is receipt content (#3166).

The recorded ``hold_receipts.authority`` must be covered by the whole-history
anchor and the boot snapshot, not only by target-local reads. Covering it
changes the anchor digest of existing history, so the change is an explicit
anchor-format version bump performed once, in the schema transaction that adds
and backfills the column, and published through the same staged-candidate
protocol as every other history head.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kestrel_sovereign.hold import HoldAuthority, HoldStore
from kestrel_sovereign.hold.state import (
    _HISTORY_ANCHOR_V2_MIGRATION,
    _HISTORY_ANCHOR_V3_MIGRATION,
    _HOLD_SCHEMA_TABLES,
    _INITIALIZATION_WITNESS_PAYLOAD,
    _POSTGRES_HISTORY_ANCHOR_KEY,
    _POSTGRES_HISTORY_CANDIDATE_KEY,
    _POSTGRES_WITNESS_KEY,
    _WITNESS_BACKFILL,
    HoldCorruptStateError,
    HoldDatabaseSnapshot,
    _HistoryAnchorFormat,
    _sqlite_custody_marker_payload,
    _sqlite_hold_snapshot,
    validate_hold_database_snapshot,
    validate_postgres_hold_readiness_snapshot,
    validate_sqlite_hold_readiness,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.utils.hold_history_v1 import (
    publish_history_head,
    rewind_to_v1_history_anchor,
    v1_history_anchor,
    v1_receipt_rows,
)

TARGET_A = "did:agent:anchor-a"
TARGET_B = "did:agent:anchor-b"
OPERATOR = "did:sovereign:operator"


def _postgres_test_url() -> str | None:
    return (
        os.environ.get("TEST_POSTGRES_URL")
        or os.environ.get("KESTREL_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
    )


def _with_search_path(url: str, schema: str) -> str:
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    query = [
        (key, value) for key, value in parse_qsl(parts.query) if key != "search_path"
    ]
    query.append(("search_path", schema))
    return urlunsplit(parts._replace(query=urlencode(query)))


@pytest.fixture(params=["sqlite", "postgres"])
async def anchor_backend(request, tmp_path):
    """A Hold database on each backend, in an isolated schema for PostgreSQL.

    The upgrade tests drop and re-add ``hold_receipts.authority``, so the
    PostgreSQL leg must never share a schema with another test. SQLite uses
    its full production custody (anchor file and custody marker); PostgreSQL
    uses the portable file-evidence topology of the SQL parity test.
    """

    if request.param == "sqlite":
        path = tmp_path / "anchor-upgrade.db"
        db = await AsyncDatabase.sqlite(str(path))
        try:
            yield SimpleNamespace(
                db=db, open_store=lambda: HoldStore(db), sqlite_path=path
            )
        finally:
            await db.close()
        return

    url = _postgres_test_url()
    if not url:
        pytest.skip("TEST_POSTGRES_URL required for the PostgreSQL leg")
    try:
        from kestrel_sovereign.storage.db.postgres import PostgresBackend
    except ImportError:
        pytest.skip("PostgresBackend not available")
    schema = f"hold_anchor_{uuid4().hex}"
    admin = PostgresBackend(url)
    try:
        await admin.connect()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"PostgreSQL not available: {exc}")
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    backend = PostgresBackend(_with_search_path(url, schema))
    try:
        await backend.connect()
        db = AsyncDatabase(backend)
        witness = tmp_path / "anchor-upgrade.hold-initialized-v1"
        yield SimpleNamespace(
            db=db,
            open_store=lambda: HoldStore(db, initialization_witness_path=witness),
            sqlite_path=None,
        )
    finally:
        await backend.close()
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()


async def _seed_two_targets(store) -> dict:
    """Hold both targets and the host; release and re-hold A so it has history."""

    held_a = await store.set_hold(
        authority=HoldAuthority.SOVEREIGN,
        scope="agent",
        target_id=TARGET_A,
        actor_id=OPERATOR,
        reason="hold a",
        operation_id="anchor-hold-a",
    )
    await store.release_hold(
        authority=HoldAuthority.SOVEREIGN,
        scope="agent",
        target_id=TARGET_A,
        actor_id=OPERATOR,
        reason="release a",
        operation_id="anchor-release-a",
        expected_hold_receipt_id=held_a.receipt.receipt_id,
    )
    reheld_a = await store.set_hold(
        authority=HoldAuthority.SOVEREIGN,
        scope="agent",
        target_id=TARGET_A,
        actor_id=OPERATOR,
        reason="hold a again",
        operation_id="anchor-rehold-a",
    )
    held_b = await store.set_hold(
        authority=HoldAuthority.SOVEREIGN,
        scope="agent",
        target_id=TARGET_B,
        actor_id=OPERATOR,
        reason="hold b",
        operation_id="anchor-hold-b",
    )
    host = await store.set_hold(
        authority=HoldAuthority.SOVEREIGN,
        scope="host",
        actor_id=OPERATOR,
        reason="hold host",
        operation_id="anchor-hold-host",
    )
    return {"a": reheld_a, "b": held_b, "host": host}


def _readiness(path):
    """Doctor's offline SQLite oracle, on a family with production modes."""

    for member in path.parent.glob(f"{path.name}*"):
        member.chmod(0o600)
    return validate_sqlite_hold_readiness(path)


def _active(seeded) -> set:
    return {seeded["a"].current, seeded["b"].current, seeded["host"].current}


async def _assert_migrated_to_v2(db, store) -> bytes:
    """The upgrade committed the column and markers and published a head.

    A v1 host upgrades straight to the current format: the same transaction
    records the v2 authority migration and the v3 mandate-scope migration
    (#3168), and publishes one v3 head.
    """

    assert await db.column_exists("hold_receipts", "authority")
    marker = await db.fetchall(
        "SELECT name FROM hold_schema_migrations WHERE name IN (?, ?) "
        "ORDER BY name",
        (_HISTORY_ANCHOR_V2_MIGRATION, _HISTORY_ANCHOR_V3_MIGRATION),
    )
    assert [tuple(row) for row in marker] == [
        (_HISTORY_ANCHOR_V2_MIGRATION,),
        (_HISTORY_ANCHOR_V3_MIGRATION,),
    ]
    stable = await store._read_history_anchor()
    assert stable is not None
    assert stable.startswith(_HistoryAnchorFormat.V3.header)
    assert stable == await store._current_history_anchor_payload()
    assert await store._read_external_history_candidate() is None
    if store._custody_control_path is not None:
        assert store._read_sqlite_custody_marker() == (
            _sqlite_custody_marker_payload(store._custody_control_path, stable)
        )
    return stable


async def _assert_history_readable(store, seeded) -> None:
    assert set(await store.read_boot_state()) == _active(seeded)
    effective = await store.get_effective(TARGET_A)
    assert effective.agent == seeded["a"].current
    assert effective.host == seeded["host"].current
    receipt = await store.get_receipt("anchor-hold-b")
    assert receipt == seeded["b"].receipt
    assert receipt.authority is HoldAuthority.SOVEREIGN
    page = await store.list_receipts(limit=50)
    assert len(page.entries) == 5
    assert all(
        entry.receipt.authority is HoldAuthority.SOVEREIGN for entry in page.entries
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_v1_anchored_history_boots_after_the_authority_migration(
    anchor_backend,
):
    """(a) A host anchored by a pre-``authority`` release upgrades cleanly.

    The migration adds the column, re-anchors the unchanged history under the
    v2 projection exactly once, and publishes it like any other head. Skipping
    that re-anchor would leave the v1 head beside a v2 database, and boot
    would refuse with the #3287 wedge class this test guards against.
    """

    db = anchor_backend.db
    store = anchor_backend.open_store()
    await store.ensure_schema()
    seeded = await _seed_two_targets(store)
    v1_anchor = await rewind_to_v1_history_anchor(db, store)
    assert await store._read_history_anchor() == v1_anchor
    if anchor_backend.sqlite_path is not None:
        # Doctor predicts that the un-migrated host boots.
        assert set(_readiness(anchor_backend.sqlite_path)) == (
            _active(seeded)
        )

    upgraded = anchor_backend.open_store()
    await upgraded.ensure_schema()

    v2_anchor = await _assert_migrated_to_v2(db, upgraded)
    await _assert_history_readable(upgraded, seeded)
    if anchor_backend.sqlite_path is not None:
        assert set(_readiness(anchor_backend.sqlite_path)) == (
            _active(seeded)
        )

    # Exactly once: a later boot neither re-anchors nor refuses.
    restarted = anchor_backend.open_store()
    await restarted.ensure_schema()
    assert await restarted._read_history_anchor() == v2_anchor
    await _assert_history_readable(restarted, seeded)


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("upgraded", [False, True], ids=["fresh", "upgraded"])
async def test_rewritten_authority_is_caught_by_a_global_read_for_another_target(
    anchor_backend,
    upgraded,
):
    """(b) Target B's reads prove the whole history, including A's authority.

    B's target-local walk never touches A's receipts, so only the anchor and
    snapshot can catch this. A whole-history projection without ``authority``
    would read A's forged value as sovereign and pass.
    """

    db = anchor_backend.db
    store = anchor_backend.open_store()
    await store.ensure_schema()
    seeded = await _seed_two_targets(store)
    if upgraded:
        await rewind_to_v1_history_anchor(db, store)
        store = anchor_backend.open_store()
        await store.ensure_schema()
    # Every read for B is clean before the forgery.
    assert await store.get_hold("agent", TARGET_B) == seeded["b"].current
    await store.list_receipts(target_id=TARGET_B, limit=10)

    # Not a HoldAuthority member. ``mandate`` is one since #3168, and on this
    # sovereign agent latch it is refused by its own authority rule, which
    # ``test_hold_mandate`` covers.
    await db.execute(
        "UPDATE hold_receipts SET authority = ? WHERE operation_id = ?",
        ("delegated", "anchor-hold-a"),
    )

    forged = "hold receipt has invalid typed fields"
    with pytest.raises(HoldCorruptStateError, match=forged):
        await store.get_hold("agent", TARGET_B)
    with pytest.raises(HoldCorruptStateError, match=forged):
        await store.list_receipts(target_id=TARGET_B, limit=10)
    with pytest.raises(HoldCorruptStateError, match=forged):
        await store.get_effective(TARGET_B)
    with pytest.raises(HoldCorruptStateError, match=forged):
        await store.read_boot_state()
    if anchor_backend.sqlite_path is not None:
        with pytest.raises(HoldCorruptStateError, match=forged):
            _readiness(anchor_backend.sqlite_path)


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    ("column", "forged"),
    [
        ("reason", "rewritten reason"),
        ("actor_id", "did:forged:actor"),
        ("occurred_at", "2020-01-01T00:00:00+00:00"),
    ],
)
async def test_v1_fields_stay_covered_after_the_re_anchor(
    anchor_backend,
    column,
    forged,
):
    """(c) Re-anchoring under v2 keeps every v1 field under the anchor."""

    db = anchor_backend.db
    store = anchor_backend.open_store()
    await store.ensure_schema()
    seeded = await _seed_two_targets(store)
    await rewind_to_v1_history_anchor(db, store)
    store = anchor_backend.open_store()
    await store.ensure_schema()
    assert await store.get_hold("agent", TARGET_B) == seeded["b"].current

    await db.execute(
        f"UPDATE hold_receipts SET {column} = ? WHERE operation_id = ?",
        (forged, "anchor-release-a"),
    )

    with pytest.raises(HoldCorruptStateError, match="history anchor"):
        await store.get_hold("agent", TARGET_B)
    with pytest.raises(HoldCorruptStateError):
        await store.read_boot_state()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_migration_refuses_to_re_anchor_a_history_its_v1_anchor_rejects(
    anchor_backend,
):
    """Re-anchoring must not bless receipts the v1 anchor never described."""

    db = anchor_backend.db
    store = anchor_backend.open_store()
    await store.ensure_schema()
    await _seed_two_targets(store)
    v1_anchor = await rewind_to_v1_history_anchor(db, store)
    await db.execute(
        "UPDATE hold_receipts SET reason = ? WHERE operation_id = ?",
        ("rewritten before upgrade", "anchor-hold-b"),
    )

    upgraded = anchor_backend.open_store()
    with pytest.raises(HoldCorruptStateError, match="history anchor"):
        await upgraded.ensure_schema()

    assert await upgraded._read_history_anchor() == v1_anchor
    assert await upgraded._read_external_history_candidate() is None
    assert not await db.fetchall(
        "SELECT name FROM hold_schema_migrations WHERE name = ?",
        (_HISTORY_ANCHOR_V2_MIGRATION,),
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_uncommitted_anchor_format_migration_is_discarded_and_retried(
    anchor_backend,
):
    """A re-anchor staged by a migration that never committed is not a wedge."""

    db = anchor_backend.db
    store = anchor_backend.open_store()
    await store.ensure_schema()
    seeded = await _seed_two_targets(store)
    v2_anchor = await store._read_history_anchor()
    await rewind_to_v1_history_anchor(db, store)
    # The migration staged its re-anchor, then its transaction rolled back.
    await store._stage_external_history_candidate(v2_anchor)
    if anchor_backend.sqlite_path is not None:
        assert set(_readiness(anchor_backend.sqlite_path)) == (
            _active(seeded)
        )

    upgraded = anchor_backend.open_store()
    await upgraded.ensure_schema()

    assert await _assert_migrated_to_v2(db, upgraded) == v2_anchor
    await _assert_history_readable(upgraded, seeded)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_other_candidate_beside_v1_history_still_refuses(anchor_backend):
    """Only the exact re-anchor is discardable; any other candidate is refused."""

    db = anchor_backend.db
    store = anchor_backend.open_store()
    await store.ensure_schema()
    await _seed_two_targets(store)
    await rewind_to_v1_history_anchor(db, store)
    await store._stage_external_history_candidate(
        b"kestrel-hold-history-v2\n6\n" + b"a" * 64 + b"\n"
    )

    upgraded = anchor_backend.open_store()
    with pytest.raises(HoldCorruptStateError, match="ambiguous"):
        await upgraded.ensure_schema()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_committed_anchor_format_migration_is_published_on_restart(
    anchor_backend,
):
    """The migration committed; the process died before promoting its head."""

    db = anchor_backend.db
    store = anchor_backend.open_store()
    await store.ensure_schema()
    seeded = await _seed_two_targets(store)
    v2_anchor = await store._read_history_anchor()
    await publish_history_head(store, v1_history_anchor(await v1_receipt_rows(db)))
    await store._stage_external_history_candidate(v2_anchor)
    if anchor_backend.sqlite_path is not None:
        assert set(_readiness(anchor_backend.sqlite_path)) == (
            _active(seeded)
        )

    restarted = anchor_backend.open_store()
    await restarted.ensure_schema()

    assert await _assert_migrated_to_v2(db, restarted) == v2_anchor
    await _assert_history_readable(restarted, seeded)


@pytest.mark.asyncio
async def test_half_published_anchor_format_migration_finishes_its_custody_marker(
    tmp_path,
):
    """The v2 anchor file landed but the SQLite custody marker is still v1."""

    path = tmp_path / "half-published.db"
    db = await AsyncDatabase.sqlite(str(path))
    try:
        store = HoldStore(db)
        await store.ensure_schema()
        seeded = await _seed_two_targets(store)
        v2_anchor = await store._read_history_anchor()
        store._write_sqlite_custody_marker(
            v1_history_anchor(await v1_receipt_rows(db))
        )
        await store._stage_external_history_candidate(v2_anchor)
        assert set(_readiness(path)) == _active(seeded)

        restarted = HoldStore(db)
        await restarted.ensure_schema()

        assert await _assert_migrated_to_v2(db, restarted) == v2_anchor
        await _assert_history_readable(restarted, seeded)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_postgres_readiness_agrees_with_boot_across_the_migration(tmp_path):
    """Doctor's PostgreSQL evidence oracle predicts the same outcomes as boot."""

    path = tmp_path / "oracle.db"
    db = await AsyncDatabase.sqlite(str(path))
    try:
        store = HoldStore(db)
        await store.ensure_schema()
        seeded = await _seed_two_targets(store)
        v2_anchor = (await store._read_history_anchor()).decode("ascii")
        v1_anchor = v1_history_anchor(await v1_receipt_rows(db)).decode("ascii")

        def snapshot():
            with closing(sqlite3.connect(path)) as connection:
                return _sqlite_hold_snapshot(connection)

        witness = (
            _POSTGRES_WITNESS_KEY,
            _INITIALIZATION_WITNESS_PAYLOAD.decode("ascii"),
        )
        # Committed but unpublished: the v1 head beside the staged re-anchor.
        validate_postgres_hold_readiness_snapshot(
            snapshot=snapshot(),
            evidence_rows=[
                witness,
                (_POSTGRES_HISTORY_ANCHOR_KEY, v1_anchor),
                (_POSTGRES_HISTORY_CANDIDATE_KEY, v2_anchor),
            ],
        )
        # A v1 head beside a migrated database with no candidate is a skipped
        # re-anchor: refused.
        with pytest.raises(HoldCorruptStateError, match="history anchor"):
            validate_postgres_hold_readiness_snapshot(
                snapshot=snapshot(),
                evidence_rows=[witness, (_POSTGRES_HISTORY_ANCHOR_KEY, v1_anchor)],
            )

        await rewind_to_v1_history_anchor(db, store)
        # Un-migrated and v1-anchored: bootable, with or without the staged
        # re-anchor of a migration that never committed.
        for extra in ([], [(_POSTGRES_HISTORY_CANDIDATE_KEY, v2_anchor)]):
            validate_postgres_hold_readiness_snapshot(
                snapshot=snapshot(),
                evidence_rows=[
                    witness,
                    (_POSTGRES_HISTORY_ANCHOR_KEY, v1_anchor),
                    *extra,
                ],
            )
        upgraded = HoldStore(db)
        await upgraded.ensure_schema()
        assert set(await upgraded.read_boot_state()) == _active(seeded)
    finally:
        await db.close()


def test_v2_snapshot_and_anchor_refuse_rows_without_authority():
    """A migrated history must carry the recorded authority of every receipt."""

    receipt = (
        "receipt-one",
        "operation-one",
        "hold",
        "applied",
        "agent",
        "did:agent:kite",
        "pause",
        "did:operator:sovereign",
        "2026-09-04T12:00:00+00:00",
        "",
        "",
        "receipt-one",
    )
    with pytest.raises(HoldCorruptStateError, match="anchor format covers"):
        validate_hold_database_snapshot(
            HoldDatabaseSnapshot(
                existing_tables=frozenset(_HOLD_SCHEMA_TABLES),
                receipt_rows=(receipt,),
                migration_rows=((_HISTORY_ANCHOR_V2_MIGRATION,), (_WITNESS_BACKFILL,)),
            )
        )
    with pytest.raises(HoldCorruptStateError, match="anchor format covers"):
        HoldStore._history_anchor_payload_from_rows(
            (receipt,), anchor_format=_HistoryAnchorFormat.V2
        )
