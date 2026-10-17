"""Real committed callbacks/catalog reads retain every explicit control carrier."""

import asyncio

import asyncpg
import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionCustody, ProcessRuntimeExecutionFence, bind_execution_custody_snapshot,
    execution_commit_outcome, is_execution_control_error,
)
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, _LeaseRenewalState
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("grouped", [False, True])
async def test_committed_callback_preserves_actual_cancellation_carried_local_denial(native_pg, grouped):
    backend, observer, schema = native_pg
    started = asyncio.Event()

    async def original_child():
        started.set()
        await asyncio.Event().wait()

    child = asyncio.create_task(original_child())
    try:
        await asyncio.wait_for(started.wait(), 2)
        child.cancel()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await child
    finally:
        if not child.done():
            child.cancel()
        await asyncio.gather(child, return_exceptions=True)

    state = _LeaseRenewalState()
    scope = ExecutionCustody(state)
    originals = []

    def callback():
        state.lost.set()
        try:
            scope.require_work()
        except BaseException as denial:
            if grouped:
                carrier = BaseExceptionGroup("original cancelled occurrence", [cancelled.value, denial])
                originals.append(carrier)
                raise carrier
            originals.append(cancelled.value)
            raise cancelled.value from denial

    with bind_execution_custody_snapshot((scope,)):
        with pytest.raises(BaseException) as caught:
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (72, 'real committed callback')")
                backend.call_after_commit(callback)
    assert await observer.fetchval(f'SELECT value FROM "{schema}".effects WHERE id=72') == "real committed callback"
    assert execution_commit_outcome(caught.value) == "committed"
    assert caught.value.__cause__ is originals[0]
    notified = []
    runner = SchedulerRunner(AsyncDatabase(backend), "original-host", lambda *args: None,
        on_protocol_failure=notified.append)
    runner._latch_protocol_failure(caught.value)
    assert runner.readiness_failure is caught.value and notified == [caught.value]
    with pytest.raises(BaseException) as retained:
        scope.require_work()
    assert execution_commit_outcome(retained.value) == "committed"


async def test_embedding_catalog_diagnostic_propagates_actual_native_commit_ack_loss(native_pg, monkeypatch):
    backend, _, _ = native_pg
    db = AsyncDatabase(backend)
    # This diagnostic does not use encryption or provider configuration. Keep
    # those unrelated operator surfaces out of the native database proof.
    store = object.__new__(AsyncConversationStore)
    store.db, store.agent_id = db, "native-diagnostic"
    store._embedding_column_dim = store._embedding_column_mismatch = None
    scope = ExecutionCustody(ProcessRuntimeExecutionFence(store.agent_id))
    original_exit = asyncpg.transaction.Transaction.__aexit__
    committed = []

    async def lost_ack(transaction, error_type, error, traceback):
        result = await original_exit(transaction, error_type, error, traceback)
        if error_type is None and not transaction._nested:
            committed.append(transaction)
            raise ConnectionResetError("actual catalog read COMMIT completed before acknowledgement loss")
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", lost_ack)
    try:
        with bind_execution_custody_snapshot((scope,)):
            with pytest.raises(BaseException) as caught:
                await store.check_embedding_column_dimensions()
        assert committed, "the actual guarded catalog transaction never committed"
        assert execution_commit_outcome(caught.value) == "unknown"
        assert is_execution_control_error(caught.value)
        assert store._embedding_column_mismatch is None
    finally:
        monkeypatch.undo()
