"""Canonical visual receipt/owner/CAS semantics on both native databases."""
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from cryptography.fernet import Fernet
import pytest
import pytest_asyncio
from kestrel_sdk.computer_use import ActionRequest, ActionReceipt, EnvironmentRequest
from kestrel_sdk.resources import ResourceScope, ResourceConflict, ResourceNotFound

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.visual_computer.repository import CustodyRecord, VisualCustodyRepository

SCOPE = ResourceScope(tenant_id="tenant", principal_id="owner", agent_id="agent")


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def custody(tmp_path, monkeypatch, request):
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    async def empty(db):
        pass
    async with AsyncExitStack() as stack:
        if request.param == "postgres":
            from tests.utils.postgres_schema import postgres_test_url, disposable_postgres_schema, with_search_path
            dsn = postgres_test_url()
            if not dsn:
                pytest.skip("PostgreSQL test URL is not configured")
            admin = await AsyncDatabase.postgres(dsn, min_pool_size=1, max_pool_size=2, schema_initializer=empty)
            stack.push_async_callback(admin.close)
            schema = await stack.enter_async_context(disposable_postgres_schema(admin, "visual_custody"))
            async def reopen():
                return await AsyncDatabase.postgres(with_search_path(dsn, schema), min_pool_size=1,
                    max_pool_size=2, schema_initializer=empty)
        else:
            async def reopen():
                return await AsyncDatabase.sqlite(str(tmp_path/"custody.db"), schema_initializer=empty)
        db = await reopen()
        stack.push_async_callback(db.close)
        store = VisualCustodyRepository(db)
        await store.ensure_initialized()
        yield store, reopen


def record():
    now = datetime.now(UTC)
    request = EnvironmentRequest(request_id="original", scope=SCOPE, provider_id="test-provider",
        profile_id="offline", image="worker@sha256:"+"a"*64, surface="browser", location="local",
        required_capabilities=frozenset({"screenshot", "type"}), requested_at=now,
        expires_at=now+timedelta(minutes=10), idle_ttl_seconds=60, ready_timeout_seconds=10,
        max_hourly_cost_usd=Decimal("0"), max_total_cost_usd=Decimal("0"))
    return CustodyRecord(request=request, session_id="original-session")


@pytest.mark.asyncio
async def test_original_admission_cas_and_owner_survive_native_restart(custody):
    store, reopen = custody
    original = await store.admit(SCOPE, record(), lambda: None)
    stale = original.model_copy(deep=True)
    original.phase = "fenced"
    await store.save(SCOPE, original)
    with pytest.raises(ResourceConflict):
        await store.save(SCOPE, stale)
    changed = original.model_copy(deep=True)
    changed.request = changed.request.model_copy(update={"profile_id":"other"})
    with pytest.raises(ResourceConflict):
        await store.admit(SCOPE, changed, lambda: None)
    db = await reopen()
    try:
        restarted = VisualCustodyRepository(db)
        current = await restarted.get(SCOPE, session_id=original.session_id)
        assert current.phase == "fenced" and current.revision == 1
        stranger = SCOPE.model_copy(update={"principal_id":"stranger"})
        with pytest.raises(ResourceNotFound):
            await restarted.get(stranger, session_id=original.session_id)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_native_accepted_input_receipt_is_encrypted_and_never_replaced(custody):
    store, reopen = custody
    original = await store.admit(SCOPE, record(), lambda: None)
    action = ActionRequest(action_id="accepted", session_id=original.session_id, observation_id="frame",
        control_epoch=0, not_after=datetime.now(UTC)+timedelta(seconds=10),
        action={"kind":"type","text":"private-screen-input-c417"})
    accepted = await store.action(SCOPE, original.session_id, action.action_id, accept=action)
    assert accepted.receipt is None and await store.has_pending_actions(SCOPE, original.session_id)
    raw = (await store.db.fetchone("SELECT payload FROM visual_computer_actions"))[0]
    assert b"private-screen-input-c417" not in bytes(raw)
    receipt = ActionReceipt(action_id=action.action_id, session_id=action.session_id,
        observation_id=action.observation_id, control_epoch=0, outcome="completed", recorded_at=datetime.now(UTC))
    await store.action(SCOPE, original.session_id, action.action_id, receipt=receipt)
    db = await reopen()
    try:
        restarted = VisualCustodyRepository(db)
        recovered = await restarted.action(SCOPE, original.session_id, action.action_id)
        assert recovered.request == action and recovered.receipt == receipt
        assert not await restarted.has_pending_actions(SCOPE, original.session_id)
        with pytest.raises(ResourceConflict):
            await restarted.action(SCOPE, original.session_id, action.action_id,
                accept=action.model_copy(update={"observation_id":"changed"}))
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_native_current_guard_rolls_back_admission_after_sql_before_commit(custody, monkeypatch):
    store, _ = custody
    revoked = False
    execute = store.db.execute
    async def revoke_after_insert(sql, *args, **kwargs):
        nonlocal revoked
        value = await execute(sql, *args, **kwargs)
        if sql.startswith("INSERT INTO visual_computer_requests"):
            revoked = True
        return value
    monkeypatch.setattr(store.db, "execute", revoke_after_insert)
    def current():
        if revoked:
            raise ResourceConflict("original current grant retired")
    with pytest.raises(ResourceConflict):
        await store.admit(SCOPE, record(), current)
    assert (await store.db.fetchone("SELECT COUNT(*) FROM visual_computer_requests"))[0] == 0


@pytest.mark.asyncio
async def test_native_preparation_dispatch_cas_and_restart_never_replay(custody):
    store,reopen=custody
    original=await store.admit(SCOPE,record(),lambda:None)
    request=ActionRequest(action_id="prepared",session_id=original.session_id,observation_id="frame",
        control_epoch=0,not_after=datetime.now(UTC)+timedelta(seconds=10),action={"kind":"click","x":1,"y":1})
    await store.action(SCOPE,original.session_id,request.action_id,accept=request,prepare=True)
    db=await reopen()
    try:
        restarted=VisualCustodyRepository(db)
        denied=await restarted.action(SCOPE,original.session_id,request.action_id,abandon_prepared=True)
        assert denied.receipt.outcome=="denied" and not denied.dispatch_started
        with pytest.raises(ResourceConflict):
            await restarted.action(SCOPE,original.session_id,request.action_id,dispatch_revision=original.revision)
        other=request.model_copy(update={"action_id":"dispatch-possible"})
        await restarted.action(SCOPE,original.session_id,other.action_id,accept=other,prepare=True)
        await restarted.action(SCOPE,original.session_id,other.action_id,dispatch_revision=original.revision)
        with pytest.raises(ResourceConflict):
            await store.action(SCOPE,original.session_id,other.action_id,dispatch_revision=original.revision)
        pending=await restarted.action(SCOPE,original.session_id,other.action_id,abandon_prepared=True)
        assert pending.dispatch_started and pending.receipt is None
    finally: await db.close()


@pytest.mark.asyncio
async def test_native_verified_release_capacity_and_unknown_action_retention(custody):
    from kestrel_sdk.computer_use import EnvironmentLease
    store,reopen=custody;store.max_requests=store.max_actions=1
    original=await store.admit(SCOPE,record(),lambda:None)
    request=ActionRequest(action_id="terminal",session_id=original.session_id,observation_id="frame",
        control_epoch=0,not_after=datetime.now(UTC)+timedelta(seconds=10),action={"kind":"type","text":"private-terminal-text"})
    await store.action(SCOPE,original.session_id,request.action_id,accept=request)
    receipt=ActionReceipt(action_id=request.action_id,session_id=request.session_id,
        observation_id=request.observation_id,control_epoch=0,outcome="completed",recorded_at=datetime.now(UTC))
    await store.action(SCOPE,original.session_id,request.action_id,receipt=receipt)
    original.lease=EnvironmentLease(lease_id="lease",request_id=original.request.request_id,quote_id="q",scope=SCOPE,
        provider_id=original.request.provider_id,location="local",expires_at=original.request.expires_at,
        state="released",released_at=datetime.now(UTC),provider_resource_id="original-worker")
    original.phase="released"
    await store.save(SCOPE,original)
    assert (await store.db.fetchone("SELECT COUNT(*) FROM visual_computer_actions"))[0]==0
    other=record();other.session_id="new-session";other.request=other.request.model_copy(update={"request_id":"new-request"})
    await store.admit(SCOPE,other,lambda:None)
    with pytest.raises(ResourceConflict):
        await store.action(SCOPE,original.session_id,request.action_id,accept=request)
    pending=request.model_copy(update={"session_id":other.session_id,"action_id":"unknown"})
    await store.action(SCOPE,other.session_id,pending.action_id,accept=pending)
    # Upgrade populated original tables on each actual native backend.
    await store.db.execute("DROP INDEX visual_computer_retired_requests")
    await store.db.execute("DROP INDEX visual_computer_live_requests")
    await store.db.execute("DROP INDEX visual_computer_pending_actions")
    await store.db.execute("ALTER TABLE visual_computer_requests DROP COLUMN released")
    await store.db.execute("ALTER TABLE visual_computer_actions DROP COLUMN pending")
    db=await reopen()
    try:
        restarted=VisualCustodyRepository(db,max_requests=1,max_actions=1)
        tombstone=await restarted.get(SCOPE,session_id=original.session_id)
        assert tombstone.phase=="released" and tombstone.request==original.request
        assert (await restarted.admit(SCOPE,original,lambda:None)).phase=="released"
        recovered=await restarted.action(SCOPE,other.session_id,pending.action_id)
        assert recovered.request==pending and recovered.dispatch_started and recovered.receipt is None
        other.lease=original.lease.model_copy(update={"request_id":other.request.request_id})
        other.phase="released";await restarted.save(SCOPE,other)
        assert (await restarted.action(SCOPE,other.session_id,pending.action_id)).receipt is None
        assert [r.session_id for r in await restarted.records(SCOPE)]==[other.session_id]
    finally: await db.close()
