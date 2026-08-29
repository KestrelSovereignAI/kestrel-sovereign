from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from kestrel_sovereign.hold import (
    HoldDisposition,
    HoldIdempotencyConflict,
    HoldScope,
    HoldStateError,
    HoldStore,
)
from kestrel_sovereign.hold.state import (
    HoldCorruptStateError,
    _receipt_from_row,
)
from kestrel_sovereign.host_features.context import build_host_context
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.fixture
async def hold_db(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "host.db"))
    store = HoldStore(db)
    await store.ensure_schema()
    try:
        yield db, store
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_host_and_agent_holds_compose_and_release_independently(hold_db):
    _db, store = hold_db
    host = await store.set_hold(
        scope=HoldScope.HOST,
        actor_id="did:sovereign:operator",
        reason="fleet investigation",
        operation_id="hold-host-1",
    )
    agent = await store.set_hold(
        scope=HoldScope.AGENT,
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="agent investigation",
        operation_id="hold-agent-1",
    )

    effective = await store.get_effective("did:agent:kite")
    assert effective.held
    assert effective.sources == (HoldScope.HOST, HoldScope.AGENT)
    assert effective.host == host.current
    assert effective.agent == agent.current

    released = await store.release_hold(
        scope=HoldScope.HOST,
        actor_id="did:sovereign:operator",
        reason="fleet cleared",
        operation_id="release-host-1",
        expected_hold_receipt_id=host.receipt.receipt_id,
    )
    assert released.receipt.disposition is HoldDisposition.APPLIED
    assert released.current is None

    effective = await store.get_effective("did:agent:kite")
    assert effective.held
    assert effective.host is None
    assert effective.agent == agent.current


@pytest.mark.asyncio
async def test_state_and_receipts_survive_database_restart(tmp_path):
    path = tmp_path / "host.db"
    first_db = await AsyncDatabase.sqlite(str(path))
    first = HoldStore(first_db)
    await first.ensure_schema()
    mutation = await first.set_hold(
        scope="agent",
        target_id="did:agent:durable",
        actor_id="did:sovereign:operator",
        reason="leave down",
        operation_id="durable-operation",
    )
    await first_db.close()

    second_db = await AsyncDatabase.sqlite(str(path))
    second = HoldStore(second_db)
    await second.ensure_schema()
    try:
        restored = await second.get_effective("did:agent:durable")
        receipt = await second.get_receipt("durable-operation")
        assert restored.agent == mutation.current
        assert receipt == mutation.receipt
        assert datetime.fromisoformat(receipt.occurred_at).tzinfo is not None
    finally:
        await second_db.close()


@pytest.mark.asyncio
async def test_receipt_primary_key_is_explicitly_not_null(hold_db):
    db, _store = hold_db
    columns = await db.fetchall("PRAGMA table_info(hold_receipts)")
    receipt_id = next(row for row in columns if row[1] == "receipt_id")
    assert receipt_id[3] == 1


def test_existing_null_receipt_identity_fails_closed_on_read() -> None:
    with pytest.raises(HoldCorruptStateError, match="receipt identity"):
        _receipt_from_row(
            (
                None,
                "legacy-operation",
                "release",
                "already_in_state",
                "agent",
                "did:agent:kite",
                "legacy import",
                "did:sovereign:operator",
                "2026-08-28T00:00:00+00:00",
                "expected-receipt",
                "",
                "",
            )
        )


@pytest.mark.asyncio
async def test_host_context_reads_hold_store_from_control_database_at_boot(tmp_path):
    """The host boot context, not an agent-local DB, owns the durable latch."""

    path = str(tmp_path / "host-control.db")
    first = await build_host_context(db_path=path)
    try:
        assert first.hold_store is not None
        held = await first.hold_store.set_hold(
            scope="agent",
            target_id="did:agent:boot-held",
            actor_id="did:sovereign:operator",
            reason="remain held through restart",
            operation_id="boot-hold",
        )
    finally:
        if first.session_factory is not None:
            await first.session_factory.close()
        if first.db is not None:
            await first.db.close()

    reopened = await build_host_context(db_path=path)
    try:
        assert reopened.hold_store is not None
        effective = await reopened.hold_store.get_effective(
            "did:agent:boot-held"
        )
        assert effective.agent == held.current
    finally:
        if reopened.session_factory is not None:
            await reopened.session_factory.close()
        if reopened.db is not None:
            await reopened.db.close()


@pytest.mark.asyncio
async def test_operation_replay_is_exact_and_conflicting_reuse_fails(hold_db):
    _db, store = hold_db
    first = await store.set_hold(
        scope="agent",
        target_id="did:agent:one",
        actor_id="did:sovereign:operator",
        reason="inspect",
        operation_id="same-operation",
    )
    replay = await store.set_hold(
        scope="agent",
        target_id="did:agent:one",
        actor_id="did:sovereign:operator",
        reason="inspect",
        operation_id="same-operation",
    )
    assert replay.receipt == first.receipt
    assert replay.current == first.current

    with pytest.raises(HoldIdempotencyConflict):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:two",
            actor_id="did:sovereign:operator",
            reason="inspect",
            operation_id="same-operation",
        )


@pytest.mark.asyncio
async def test_repeated_semantic_hold_is_receipted_without_replacing_latch(hold_db):
    _db, store = hold_db
    first = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="investigate",
        operation_id="hold-once",
    )
    second = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="investigate",
        operation_id="hold-twice",
    )

    assert second.receipt.disposition is HoldDisposition.ALREADY_IN_STATE
    assert second.receipt.receipt_id != first.receipt.receipt_id
    assert second.receipt.resulting_hold_receipt_id == first.receipt.receipt_id
    assert second.current == first.current


@pytest.mark.asyncio
async def test_stale_release_cannot_clear_a_replaced_hold(hold_db):
    _db, store = hold_db
    first = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:one",
        reason="first reason",
        operation_id="hold-first",
    )
    second = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:two",
        reason="replacement reason",
        operation_id="hold-second",
    )

    stale = await store.release_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:one",
        reason="release old observation",
        operation_id="release-stale",
        expected_hold_receipt_id=first.receipt.receipt_id,
    )
    assert stale.receipt.disposition is HoldDisposition.REFUSED_STALE
    assert stale.current == second.current
    assert await store.get_hold("agent", "did:agent:kite") == second.current


@pytest.mark.asyncio
async def test_release_rejects_latch_with_missing_authority_receipt(hold_db):
    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="investigate",
        operation_id="hold-before-receipt-loss",
    )
    await db.execute(
        "DELETE FROM hold_receipts WHERE receipt_id = ?",
        (held.receipt.receipt_id,),
    )

    with pytest.raises(HoldCorruptStateError, match="missing authority receipt"):
        await store.release_hold(
            scope="agent",
            target_id="did:agent:kite",
            actor_id="did:sovereign:operator",
            reason="must remain held",
            operation_id="release-without-authority",
            expected_hold_receipt_id=held.receipt.receipt_id,
        )

    row = await db.fetchone(
        "SELECT active, hold_receipt_id FROM hold_latches "
        "WHERE scope = ? AND target_id = ?",
        ("agent", "did:agent:kite"),
    )
    assert row == (1, held.receipt.receipt_id)
    assert await store.get_receipt("release-without-authority") is None


@pytest.mark.asyncio
async def test_release_rejects_latch_bound_to_another_targets_receipt(hold_db):
    db, store = hold_db
    first = await store.set_hold(
        scope="agent",
        target_id="did:agent:first",
        actor_id="did:sovereign:operator",
        reason="first investigation",
        operation_id="hold-first-target",
    )
    other = await store.set_hold(
        scope="agent",
        target_id="did:agent:other",
        actor_id="did:sovereign:operator",
        reason="other investigation",
        operation_id="hold-other-target",
    )
    await db.execute(
        "UPDATE hold_latches SET hold_receipt_id = ? "
        "WHERE scope = ? AND target_id = ?",
        (other.receipt.receipt_id, "agent", "did:agent:first"),
    )

    with pytest.raises(HoldCorruptStateError, match="does not match"):
        await store.release_hold(
            scope="agent",
            target_id="did:agent:first",
            actor_id="did:sovereign:operator",
            reason="must remain held",
            operation_id="release-mismatched-authority",
            expected_hold_receipt_id=other.receipt.receipt_id,
        )

    row = await db.fetchone(
        "SELECT active, hold_receipt_id FROM hold_latches "
        "WHERE scope = ? AND target_id = ?",
        ("agent", "did:agent:first"),
    )
    assert row == (1, other.receipt.receipt_id)
    assert first.receipt.receipt_id != other.receipt.receipt_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "target_id", "expected_lock"),
    [
        (HoldScope.AGENT, "did:agent:independent", False),
        (HoldScope.HOST, None, True),
    ],
)
async def test_mutation_locks_host_shape_only_for_host_scope(
    hold_db, monkeypatch, scope, target_id, expected_lock,
):
    _db, store = hold_db
    inspect_shape = AsyncMock(wraps=store._assert_host_latch_shape)
    monkeypatch.setattr(store, "_assert_host_latch_shape", inspect_shape)

    await store.set_hold(
        scope=scope,
        target_id=target_id,
        actor_id="did:sovereign:operator",
        reason="scope-specific lock",
        operation_id=f"lock-{scope.value}",
    )

    inspect_shape.assert_awaited_once_with(for_update=expected_lock)


@pytest.mark.asyncio
async def test_receipt_and_latch_roll_back_as_one_unit(hold_db, monkeypatch):
    _db, store = hold_db
    insert = store._insert_receipt

    async def fail_after_receipt(**kwargs):
        await insert(**kwargs)
        raise RuntimeError("injected crash after receipt")

    monkeypatch.setattr(store, "_insert_receipt", fail_after_receipt)
    with pytest.raises(Exception, match="injected crash"):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:kite",
            actor_id="did:sovereign:operator",
            reason="investigate",
            operation_id="rolled-back",
        )

    assert await store.get_hold("agent", "did:agent:kite") is None
    assert await store.get_receipt("rolled-back") is None


@pytest.mark.asyncio
async def test_corrupt_active_latch_fails_closed(hold_db, monkeypatch):
    _db, store = hold_db
    monkeypatch.setattr(
        store,
        "_read_latch_row",
        AsyncMock(
            return_value=(
                "agent",
                "did:agent:kite",
                2,
                "receipt",
                "reason",
                "actor",
                "2026-08-28T00:00:00+00:00",
                1,
            )
        ),
    )
    with pytest.raises(HoldCorruptStateError, match="active flag"):
        await store.get_hold("agent", "did:agent:kite")


@pytest.mark.asyncio
async def test_greenfield_schema_rejects_foreign_host_target(hold_db):
    db, _store = hold_db
    with pytest.raises(Exception, match="CHECK constraint"):
        await db.execute(
            "INSERT INTO hold_latches (scope, target_id) VALUES (?, ?)",
            ("host", "foreign"),
        )


@pytest.mark.asyncio
async def test_upgraded_foreign_host_row_fails_closed_for_reads_and_mutation(
    tmp_path,
):
    """Existing v1 tables lack the new CHECK, so runtime validation is load-bearing."""

    db = await AsyncDatabase.sqlite(str(tmp_path / "legacy-hold.db"))
    await db.execute(
        "CREATE TABLE hold_latches ("
        "scope TEXT NOT NULL, target_id TEXT NOT NULL, "
        "active INTEGER NOT NULL DEFAULT 0, "
        "hold_receipt_id TEXT NOT NULL DEFAULT '', "
        "reason TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', "
        "set_at TEXT NOT NULL DEFAULT '', revision INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY (scope, target_id), "
        "CHECK (scope IN ('host', 'agent')), CHECK (active IN (0, 1)), "
        "CHECK (revision >= 0))"
    )
    await db.execute(
        "INSERT INTO hold_latches "
        "(scope, target_id, active, hold_receipt_id, reason, actor_id, set_at) "
        "VALUES ('host', 'foreign', 1, 'receipt', 'reason', 'actor', 'time')"
    )
    store = HoldStore(db)
    await store.ensure_schema()
    try:
        with pytest.raises(HoldCorruptStateError, match="foreign target"):
            await store.get_hold("host")
        with pytest.raises(HoldCorruptStateError, match="foreign target"):
            await store.get_effective("did:agent:kite")
        with pytest.raises(HoldCorruptStateError, match="foreign target"):
            await store.set_hold(
                scope="agent",
                target_id="did:agent:kite",
                actor_id="did:sovereign:operator",
                reason="must fail closed",
                operation_id="foreign-host-row",
            )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_inactive_latch_with_retained_evidence_fails_closed(
    hold_db,
    monkeypatch,
):
    _db, store = hold_db
    monkeypatch.setattr(
        store,
        "_read_latch_row",
        AsyncMock(
            return_value=(
                "agent",
                "did:agent:kite",
                0,
                "retained-receipt",
                "retained reason",
                "did:sovereign:operator",
                "2026-08-28T00:00:00+00:00",
                2,
            )
        ),
    )

    with pytest.raises(HoldCorruptStateError, match="inactive latch"):
        await store.get_hold("agent", "did:agent:kite")


@pytest.mark.asyncio
async def test_malformed_applied_hold_replay_fails_closed(
    hold_db,
    monkeypatch,
):
    _db, store = hold_db
    monkeypatch.setattr(
        store,
        "_read_receipt_by_operation",
        AsyncMock(
            return_value=(
                "receipt-id",
                "poisoned-replay",
                "hold",
                "applied",
                "agent",
                "did:agent:kite",
                "investigate",
                "did:sovereign:operator",
                "2026-08-28T00:00:00+00:00",
                "",
                "",
                "",
            )
        ),
    )

    with pytest.raises(HoldCorruptStateError, match="receipt invariant"):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:kite",
            actor_id="did:sovereign:operator",
            reason="investigate",
            operation_id="poisoned-replay",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "disposition", "expected", "prior", "resulting"),
    [
        ("hold", "refused_stale", "", "prior", "prior"),
        ("hold", "already_in_state", "", "", ""),
        ("release", "applied", "expected", "other", ""),
        ("release", "already_in_state", "expected", "prior", "prior"),
        ("release", "refused_stale", "expected", "expected", "expected"),
    ],
)
async def test_impossible_receipt_state_combinations_fail_closed(
    hold_db,
    monkeypatch,
    action,
    disposition,
    expected,
    prior,
    resulting,
):
    _db, store = hold_db
    monkeypatch.setattr(
        store,
        "_read_receipt_by_operation",
        AsyncMock(
            return_value=(
                "receipt-id",
                "corrupt-combination",
                action,
                disposition,
                "agent",
                "did:agent:kite",
                "investigate",
                "did:sovereign:operator",
                "2026-08-28T00:00:00+00:00",
                expected,
                prior,
                resulting,
            )
        ),
    )

    with pytest.raises(HoldCorruptStateError, match="receipt invariant"):
        await store.get_receipt("corrupt-combination")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["set", "release"])
async def test_mutations_preserve_typed_corrupt_state_error(
    hold_db,
    monkeypatch,
    operation,
):
    _db, store = hold_db
    monkeypatch.setattr(
        store,
        "_read_latch_row",
        AsyncMock(
            return_value=(
                "agent",
                "did:agent:kite",
                2,
                "receipt",
                "reason",
                "actor",
                "2026-08-28T00:00:00+00:00",
                1,
            )
        ),
    )

    with pytest.raises(HoldCorruptStateError, match="active flag"):
        if operation == "set":
            await store.set_hold(
                scope="agent",
                target_id="did:agent:kite",
                actor_id="did:sovereign:operator",
                reason="replace corrupt latch",
                operation_id="corrupt-set",
            )
        else:
            await store.release_hold(
                scope="agent",
                target_id="did:agent:kite",
                actor_id="did:sovereign:operator",
                reason="release corrupt latch",
                operation_id="corrupt-release",
                expected_hold_receipt_id="receipt",
            )


def test_package_exports_shared_hold_state_error() -> None:
    assert issubclass(HoldCorruptStateError, HoldStateError)
    assert issubclass(HoldIdempotencyConflict, HoldStateError)


@pytest.mark.asyncio
async def test_two_sqlite_workers_serialize_replacement_and_stale_release(tmp_path):
    path = tmp_path / "host.db"
    db_one = await AsyncDatabase.sqlite(str(path))
    db_two = await AsyncDatabase.sqlite(str(path))
    one = HoldStore(db_one)
    two = HoldStore(db_two)
    await asyncio.gather(one.ensure_schema(), two.ensure_schema())
    try:
        results = await asyncio.gather(
            one.set_hold(
                scope="agent",
                target_id="did:agent:kite",
                actor_id="did:sovereign:one",
                reason="one",
                operation_id="race-one",
            ),
            two.set_hold(
                scope="agent",
                target_id="did:agent:kite",
                actor_id="did:sovereign:two",
                reason="two",
                operation_id="race-two",
            ),
        )
        current = await one.get_hold("agent", "did:agent:kite")
        assert current is not None
        winner = next(item for item in results if item.current == current)
        loser = next(item for item in results if item.receipt != winner.receipt)

        stale = await two.release_hold(
            scope="agent",
            target_id="did:agent:kite",
            actor_id="did:sovereign:operator",
            reason="stale concurrent observation",
            operation_id="race-release",
            expected_hold_receipt_id=loser.receipt.receipt_id,
        )
        assert stale.receipt.disposition is HoldDisposition.REFUSED_STALE
        assert stale.current == current
    finally:
        await db_one.close()
        await db_two.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_hold_store_sql_is_backend_portable(db_backend):
    db = AsyncDatabase(db_backend)
    store = HoldStore(db)
    await store.ensure_schema()
    suffix = uuid4().hex
    target = f"did:agent:{suffix}"
    hold_operation = f"hold-{suffix}"
    release_operation = f"release-{suffix}"
    try:
        held = await store.set_hold(
            scope="agent",
            target_id=target,
            actor_id="did:sovereign:operator",
            reason="backend parity",
            operation_id=hold_operation,
        )
        released = await store.release_hold(
            scope="agent",
            target_id=target,
            actor_id="did:sovereign:operator",
            reason="backend parity complete",
            operation_id=release_operation,
            expected_hold_receipt_id=held.receipt.receipt_id,
        )
        assert released.receipt.disposition is HoldDisposition.APPLIED
        assert await store.get_hold("agent", target) is None
    finally:
        await db.execute(
            "DELETE FROM hold_receipts WHERE operation_id IN (?, ?)",
            (hold_operation, release_operation),
        )
        await db.execute(
            "DELETE FROM hold_latches WHERE scope = ? AND target_id = ?",
            (HoldScope.AGENT.value, target),
        )
