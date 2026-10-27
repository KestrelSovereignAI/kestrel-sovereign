"""Retain private identities while bounding live manual custody on native DBs."""
import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.computer_use import ActionRequest, EnvironmentLease
from kestrel_sdk.computer_use.human_control import HumanControlRequest
from kestrel_sdk.resources import ResourceConflict
from kestrel_sovereign.visual_computer.human import HumanRecord
from kestrel_sovereign.visual_computer.repository import CustodyIntegrityError, VisualCustodyRepository
from tests.integration.test_visual_custody_backends import custody as custody, record, SCOPE


def channel(store, parent, cid):
    now = datetime.now(UTC)
    return HumanRecord(request=HumanControlRequest(
        control_id=cid, scope=SCOPE, session_id=parent.session_id, lease_id="lease",
        driver_id="driver", consumer_id=store._address("human-consumer", SCOPE.model_dump_json(), parent.session_id, cid),
        control_epoch=1, issued_at=now, expires_at=now+timedelta(seconds=30),
        capabilities=frozenset({"screenshot", "type"}), max_frame_age_seconds=10))


async def put(store, parent, cid):
    return await store.private_record(SCOPE, parent.session_id, "human", cid, HumanRecord,
        accept=channel(store, parent, cid))


async def release(store, parent, *, confirmed=True):
    parent.lease = EnvironmentLease(lease_id="lease", request_id=parent.request.request_id, quote_id="quote",
        scope=SCOPE, provider_id=parent.request.provider_id, location="local", expires_at=parent.request.expires_at,
        state="released" if confirmed else "failed", released_at=datetime.now(UTC) if confirmed else None,
        provider_resource_id="original-worker")
    parent.phase = "released" if confirmed else "fenced"
    await store.save(SCOPE, parent)


def another():
    parent = record()
    parent.session_id = "next-session"
    parent.request = parent.request.model_copy(update={"request_id": "next-request"})
    return parent


@pytest.mark.asyncio
async def test_native_private_populated_migration_retains_exact_history_and_legacy_charge(custody):
    store, reopen = custody
    parent = await store.admit(SCOPE, record(), lambda: None)
    original = await put(store, parent, "old-channel")
    owner = store._owner(SCOPE)
    original_key = store._address("private", parent.session_id, "human", "old-channel")
    original_payload = bytes((await store.db.fetchone("SELECT payload FROM visual_computer_private_records WHERE record_key = ?", (original_key,)))[0])
    # Valid encrypted unknown-format legacy custody remains conservatively charged.
    legacy_key = store._address("private", parent.session_id, "unknown-extension", "legacy")
    await store.db.execute("INSERT INTO visual_computer_private_records (owner, record_key, payload) VALUES (?, ?, ?)",
        (owner, legacy_key, store._seal(owner, legacy_key, HumanRecord())))
    await release(store, parent)
    await store.db.execute("DROP INDEX visual_computer_private_sessions")
    await store.db.execute("ALTER TABLE visual_computer_private_records DROP COLUMN session_key")
    db = await reopen()
    try:
        restarted = VisualCustodyRepository(db, max_actions=2)
        await restarted.ensure_initialized()
        rows = await db.fetchall("SELECT record_key, payload, session_key FROM visual_computer_private_records")
        by_key = {row[0]: row for row in rows}
        assert bytes(by_key[original_key][1]) == original_payload
        assert by_key[original_key][2] == restarted._address("session", parent.session_id)
        assert by_key[legacy_key][2] is None
        recovered = await restarted.private_record(SCOPE, parent.session_id, "human", "old-channel", HumanRecord)
        assert recovered == original
        with pytest.raises(ResourceConflict, match="physically released"):
            await put(restarted, parent, "old-channel")
        active = await restarted.admit(SCOPE, another(), lambda: None)
        await put(restarted, active, "one-last-slot")
        with pytest.raises(ResourceConflict, match="capacity"):
            await put(restarted, active, "beyond-limit")
        assert (await db.fetchone("SELECT COUNT(*) FROM visual_computer_private_records"))[0] == 3
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_native_private_last_slot_serializes_across_repository_connections(custody, monkeypatch):
    store, reopen = custody
    store.max_actions = 1
    parent = await store.admit(SCOPE, record(), lambda: None)
    await put(store, parent, "retained")
    await release(store, parent)
    active = await store.admit(SCOPE, another(), lambda: None)
    db = await reopen()
    try:
        other = VisualCustodyRepository(db, max_actions=1)
        await other.ensure_initialized()
        counted, resume = asyncio.Event(), asyncio.Event()
        original_fetchone = store.db.fetchone
        async def hold_last_slot(sql, *args, **kwargs):
            value = await original_fetchone(sql, *args, **kwargs)
            if sql.startswith("SELECT COUNT(*) FROM visual_computer_private_records"):
                counted.set()
                await resume.wait()
            return value
        monkeypatch.setattr(store.db, "fetchone", hold_last_slot)
        first = asyncio.create_task(put(store, active, "first"))
        second = None
        try:
            await asyncio.wait_for(counted.wait(), 2)
            second = asyncio.create_task(put(other, active, "second"))
            # The first transaction owns the last slot until its commit. A
            # missing owner lock lets PostgreSQL's second connection insert here.
            await asyncio.sleep(.03)
            resume.set()
            results = await asyncio.gather(first, second, return_exceptions=True)
        finally:
            resume.set()
            await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)
        assert sum(isinstance(value, HumanRecord) for value in results) == 1
        assert sum(isinstance(value, ResourceConflict) for value in results) == 1
        assert (await db.fetchone("SELECT COUNT(*) FROM visual_computer_private_records"))[0] == 2
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_native_private_unknown_release_or_input_keeps_capacity_charged(custody):
    store, _ = custody
    store.max_actions = 1
    parent = await store.admit(SCOPE, record(), lambda: None)
    await put(store, parent, "unknown")
    await release(store, parent, confirmed=False)
    with pytest.raises(ResourceConflict, match="capacity"):
        await put(store, parent, "stopped-is-not-destroyed")
    action = ActionRequest(action_id="unknown-manual", session_id=parent.session_id,
        observation_id="human-frame", control_epoch=1, not_after=datetime.now(UTC)+timedelta(seconds=10),
        action={"kind": "type", "text": "private-unknown-manual"})
    await store.action(SCOPE, parent.session_id, action.action_id, accept=action, channel="human", control_id="unknown")
    await release(store, parent)
    active = await store.admit(SCOPE, another(), lambda: None)
    with pytest.raises(ResourceConflict, match="capacity"):
        await put(store, active, "new")
    assert (await store.action(SCOPE, parent.session_id, action.action_id)).receipt is None
    assert await store.private_record(SCOPE, parent.session_id, "human", "unknown", HumanRecord)


@pytest.mark.asyncio
async def test_native_private_migration_refuses_corrupt_encrypted_history(custody):
    store, reopen = custody
    parent = await store.admit(SCOPE, record(), lambda: None)
    await put(store, parent, "old")
    await release(store, parent)
    await store.db.execute("UPDATE visual_computer_private_records SET payload = ?", (b"corrupt-envelope",))
    await store.db.execute("DROP INDEX visual_computer_private_sessions")
    await store.db.execute("ALTER TABLE visual_computer_private_records DROP COLUMN session_key")
    db = await reopen()
    try:
        restarted = VisualCustodyRepository(db)
        with pytest.raises(CustodyIntegrityError):
            await restarted.ensure_initialized()
        assert restarted._key is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_native_private_migration_never_projects_foreign_original_binding(custody):
    store, reopen = custody
    parent = await store.admit(SCOPE, record(), lambda: None)
    value = channel(store, parent, "foreign")
    value.request = value.request.model_copy(update={"scope": SCOPE.model_copy(update={"principal_id": "another-owner"})})
    owner = store._owner(SCOPE)
    identity = store._address("private", parent.session_id, "human", "foreign")
    await store.db.execute("INSERT INTO visual_computer_private_records (owner, record_key, payload) VALUES (?, ?, ?)",
        (owner, identity, store._seal(owner, identity, value)))
    await release(store, parent)
    await store.db.execute("DROP INDEX visual_computer_private_sessions")
    await store.db.execute("ALTER TABLE visual_computer_private_records DROP COLUMN session_key")
    db = await reopen()
    try:
        restarted = VisualCustodyRepository(db, max_actions=1)
        await restarted.ensure_initialized()
        assert (await db.fetchone("SELECT session_key FROM visual_computer_private_records"))[0] is None
        active = await restarted.admit(SCOPE, another(), lambda: None)
        with pytest.raises(ResourceConflict, match="capacity"):
            await put(restarted, active, "new")
    finally:
        await db.close()
