"""Original fixed cleanup operations classify real COMMIT/reset acknowledgements."""

import asyncio
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, ProcessRuntimeExecutionFence,
    execution_commit_outcome,
)
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from kestrel_sovereign.stop import DistributedInvocationStore
from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.privacy import PrivacyMode
from tests.integration.test_execution_authority_postgres import (
    GenerationFence, native_pg as native_pg,
)
from tests.integration.test_execution_custody_review13_pg import _seed_owned_deliveries
from tests.unit.test_execution_custody_review43_native import carries

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def lose_native_ack(monkeypatch, *, boundary, original):
    native_commit = asyncpg.transaction.Transaction.__aexit__
    native_release = asyncpg.pool.PoolAcquireContext.__aexit__
    receipts = []
    async def commit(transaction, error_type, error, traceback):
        result = await native_commit(transaction, error_type, error, traceback)
        if boundary == "commit" and error is None and not transaction._nested and not receipts:
            receipts.append(True)
            raise original
        return result
    async def release(checkout, error_type, error, traceback):
        result = await native_release(checkout, error_type, error, traceback)
        if boundary == "release" and error is None and not receipts:
            receipts.append(True)
            raise original
        return result
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", commit)
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", release)
    return receipts


@pytest.mark.parametrize("route", ["stop", "reservations", "handoff", "cognition", "heartbeat"])
@pytest.mark.parametrize("boundary", ["commit", "release"])
@pytest.mark.parametrize("carrier", ["error", "cancel"])
async def test_retired_fixed_cleanup_native_ack_never_reports_success(native_pg, monkeypatch, route, boundary, carrier):
    backend, observer, schema = native_pg
    await _seed_owned_deliveries(backend, observer, schema)
    if route == "stop":
        store = DistributedInvocationStore(AsyncDatabase(backend))
        await store.ensure_schema()
        assert await store.register(generation_id="original-generation", agent_id="original", turn_id="original-turn", owner_id="dispatcher:original", request_generation=1)
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = scope
    scope.revoke("original runtime retired")
    original = OSError("native cleanup ACK") if carrier == "error" else asyncio.CancelledError("native cleanup ACK")
    receipts = lose_native_ack(monkeypatch, boundary=boundary, original=original)
    now = datetime.now(timezone.utc)
    try:
        with pytest.raises(BaseException) as caught:
            if route == "stop":
                await store.settle("original-generation", "dispatcher:original", RequestCompletionDisposition.COMPLETED)
            elif route == "reservations":
                await backend.release_initial_reservations(agent_id="original", owner_id="dispatcher:original", now=now, mark_owner_stopped=True)
            elif route == "handoff":
                await backend.abandon_initial_reservation(agent_id="original", consumer_id=DURABLE_COGNITION_CONSUMER_ID, delivery_id="delivery0", owner_id="dispatcher:original", reservation_token="token0", now=now, reason="original failed handoff")
            elif route == "cognition":
                await backend.fail_cognition_delivery(agent_id="original", consumer_id=DURABLE_COGNITION_CONSUMER_ID, delivery_id="delivery1", owner_id="dispatcher:original", lease_token="token1", error="original terminal control")
            else:
                await backend.retain_cognition_cleanup_owner(agent_id="original", owner_id="dispatcher:original")
        assert receipts == [True]
        assert execution_commit_outcome(caught.value) == ("unknown" if boundary == "commit" else "committed")
        assert carries(caught.value, original)
        if route == "stop":
            assert await observer.fetchval("SELECT COUNT(*) FROM stop_active_invocations") == 0
        elif route in {"reservations", "handoff", "cognition"}:
            selected = "delivery1" if route == "cognition" else "delivery0"
            assert await observer.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id=$1", selected) == ("failed" if route == "cognition" else "retry")
        else:
            assert await observer.fetchval("SELECT heartbeat_at > NOW()-INTERVAL '1 minute' FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:original'")
        assert await observer.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery2'") == "initial_reserved"
        with pytest.raises(ExecutionAuthorityError):
            await backend.execute("INSERT INTO effects VALUES (56, 'forbidden fresh work')")
    finally:
        monkeypatch.undo()
        backend._execution_custody = None


@pytest.mark.parametrize("boundary", ["commit", "release"])
@pytest.mark.parametrize("carrier", ["error", "cancel"])
async def test_retired_privacy_cutoff_published_only_after_physical_ack(native_pg, monkeypatch, boundary, carrier):
    backend, observer, schema = native_pg
    storage = AsyncStorage(backend=backend, agent_id="did:test:privacy56")
    await storage.initialize()
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
    watermark = (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat()
    wrapper._entered_ephemeral_at = watermark
    wrapper._session_conversations = [{"role": "user", "content": "retained private stint"}]
    wrapper._session_files = {"private-content-hash": b"original private file"}
    wrapper._session_file_metadata = {"private-content-hash": {"private": "original metadata"}}
    await storage.conversation.add_conversation("user", "native privacy leak")
    scope = ExecutionCustody(ProcessRuntimeExecutionFence(storage.agent_id))
    backend._execution_custody = scope
    scope.fence.retire()
    original = OSError("native purge checkout ACK") if carrier == "error" else asyncio.CancelledError("native purge checkout ACK")
    receipts = lose_native_ack(monkeypatch, boundary=boundary, original=original)
    try:
        with pytest.raises(BaseException) as caught:
            await backend.purge_retired_ephemeral_session(wrapper, reason="original retired privacy sweep")
        assert receipts == [True] and carries(caught.value, original)
        assert execution_commit_outcome(caught.value) == ("unknown" if boundary == "commit" else "committed")
        assert wrapper._entered_ephemeral_at == watermark
        assert wrapper._session_conversations
        assert wrapper._session_files and wrapper._session_file_metadata
        assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".conversation_history') == 0
        monkeypatch.undo()
        report = await backend.purge_retired_ephemeral_session(wrapper, reason="reconcile original cutoff")
        assert not report.required_sweep_failed
        assert wrapper._entered_ephemeral_at is None and wrapper._session_conversations == []
        assert wrapper._session_files == {} and wrapper._session_file_metadata == {}
    finally:
        monkeypatch.undo()
        backend._execution_custody = None
        await storage.close()
