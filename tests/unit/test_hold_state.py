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
from kestrel_sovereign.hold.state import HoldCorruptStateError
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
