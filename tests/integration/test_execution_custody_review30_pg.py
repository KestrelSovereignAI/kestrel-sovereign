"""Real PostgreSQL COMMIT acknowledgment faults reach public owners (#3569)."""

import asyncio
import json
import asyncpg
import pytest

from tests.integration.test_execution_authority_postgres import (
    native_pg as native_pg,
    GenerationFence,
)
from tests.unit.test_execution_custody_review30 import service_owner
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    execution_commit_outcome,
    bind_execution_custody,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.modality_recording import ModalityCall
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
from kestrel_sovereign.storage.async_database import AsyncDatabase
from types import SimpleNamespace

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def assert_native_preference_rows(observer, schema, route, *, mandate=None):
    """Check the whole acknowledged write, including atomic provenance (#2652)."""
    rows = {
        row["key"]: row["value"]
        for row in await observer.fetch(f'SELECT key, value FROM "{schema}".agent_metadata')
    }
    if route == "model":
        assert set(rows) == {
            KestrelAgent.MODEL_PREFERENCE_KEY,
            KestrelAgent.MODEL_PREFERENCE_PROVENANCE_KEY,
        }
        assert json.loads(rows[KestrelAgent.MODEL_PREFERENCE_KEY]) == mandate
        assert json.loads(rows[KestrelAgent.MODEL_PREFERENCE_PROVENANCE_KEY]) == {
            "mandate": mandate, "provenance": None,
        }
    elif route == "embedding":
        assert rows == {KestrelAgent.EMBEDDING_ROUTE_KEY: "null"}
    else:
        assert route == "overrides"
        assert rows == {KestrelAgent.EMBEDDING_MODEL_OVERRIDES_KEY: "{}"}


def lose_ack(monkeypatch, *, cancelled, entered=None, release=None):
    original = asyncpg.transaction.Transaction.__aexit__

    async def native_exit(transaction, error_type, error, traceback):
        result = await original(transaction, error_type, error, traceback)
        if error_type is None:
            if entered is not None:
                entered.set()
                await release.wait()
            if cancelled:
                raise asyncio.CancelledError("native COMMIT acknowledgment cancelled")
            raise OSError("native COMMIT acknowledgment lost")
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_exit)


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("route", ["model", "embedding", "overrides"])
async def test_real_preference_commit_receipt_survives_callback_and_drain(
    native_pg, monkeypatch, cancelled, route
):
    backend, observer, schema = native_pg
    await backend.execute(
        "CREATE TABLE agent_metadata (agent_id TEXT, key TEXT, value TEXT, updated_at TIMESTAMPTZ, PRIMARY KEY(agent_id,key))"
    )
    db = AsyncDatabase(backend)
    service = service_owner()
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.did = "did:test:native-preference"
    agent.llm_service = service
    agent._raw_storage = SimpleNamespace(db=db)
    agent.privacy_agent = SimpleNamespace(
        privacy_config=SimpleNamespace(allows_cloud_llm=lambda: False)
    )
    lose_ack(monkeypatch, cancelled=cancelled)
    completed = asyncio.Event()
    with bind_execution_custody(GenerationFence()):
        if route == "model":
            service.set_preference_persistence_callback(agent._persist_model_preference)
            task = service._schedule_preference_persistence("model", "local", "unit")
        elif route == "embedding":
            service.set_embedding_route_persistence_callback(
                agent._persist_embedding_route
            )
            task = service._schedule_embedding_route_persistence(None)
        else:
            service.set_route_embedding_model_persistence_callback(
                agent._persist_route_embedding_models
            )
            task = service._schedule_route_embedding_model_persistence()
        task.add_done_callback(lambda done: completed.set())
        await asyncio.wait_for(completed.wait(), 5)
    try:
        await asyncio.wait_for(completed.wait(), 5)
        await assert_native_preference_rows(
            observer, schema, route,
            mandate={"model": "model", "vendor": "local", "route": "unit"},
        )
        with pytest.raises(BaseException) as caught:
            await service.drain_preference_persistence()
        assert execution_commit_outcome(caught.value) == "unknown"
        assert not service._preference_persistence_tasks
    finally:
        await await_owned_task(task)


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("boundary", ["drain", "close"])
async def test_real_late_usage_commit_after_public_deadline_reports_uncertainty(
    native_pg, monkeypatch, cancelled, boundary
):
    backend, observer, schema = native_pg
    for table, key in (
        ("model_usage", "model_id TEXT PRIMARY KEY"),
        (
            "model_usage_periods",
            "period_start TIMESTAMP, model_id TEXT, PRIMARY KEY(period_start,model_id,provider)",
        ),
    ):
        await backend.execute(
            f"CREATE TABLE {table} ({key}, provider TEXT, use_count INTEGER, total_tokens INTEGER, cache_creation_input_tokens INTEGER, cache_read_input_tokens INTEGER, cache_creation_input_tokens_report_count INTEGER, cache_read_input_tokens_report_count INTEGER"
            + (
                ", last_used TIMESTAMP, created_at TIMESTAMP)"
                if table == "model_usage"
                else ")"
            )
        )
    service = service_owner()
    service._init_usage_tracking(usage_db=AsyncDatabase(backend))
    entered, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    retired = []

    async def close_client():
        retired.append("client")

    service.providers = [{"client": SimpleNamespace(close=close_client)}]
    lose_ack(monkeypatch, cancelled=cancelled, entered=entered, release=release)
    caller = asyncio.create_task(
        service.record_modality_call(
            ModalityCall(
                modality="embedding",
                provider="native",
                model="unit",
                duration_ms=1,
                success=True,
                input_tokens=7,
                context=LLMInvocationContext(),
            )
        )
    )
    children = ()
    try:
        await asyncio.wait_for(entered.wait(), 5)
        children = tuple(service._pending_modality_records())
        assert len(children) == 1
        children[0].add_done_callback(lambda done: completed.set())
        # The real, unmodified two-second deadline expires AFTER PostgreSQL
        # committed but BEFORE its ACK fault reaches the background owner.
        await asyncio.wait_for(asyncio.shield(caller), 5)
        assert not children[0].done()
        assert (
            await observer.fetchval(f'SELECT total_tokens FROM "{schema}".model_usage')
            == 7
        )
        assert (
            await observer.fetchval(
                f'SELECT total_tokens FROM "{schema}".model_usage_periods'
            )
            == 7
        )
        release.set()
        await asyncio.wait_for(completed.wait(), 5)
        with pytest.raises(BaseException) as caught:
            await (
                service.close()
                if boundary == "close"
                else service.drain_modality_records()
            )
        assert execution_commit_outcome(caught.value) == "unknown"
        if boundary == "close":
            assert retired == ["client"]
    finally:
        release.set()
        await await_owned_task(caller)
        for child in children:
            await await_owned_task(child)


@pytest.mark.parametrize("cancelled", [False, True])
async def test_actual_admission_future_reports_real_postgres_commit_uncertainty(
    native_pg, monkeypatch, cancelled
):
    from kestrel_sovereign.signals import (
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
        OrderedLockManager,
    )
    from kestrel_sovereign.features.channels.models import (
        ChannelMessage,
        MessageDirection,
    )
    from kestrel_sovereign.signals.sources.channels import (
        build_signal_for_channel_message,
    )

    backend, observer, schema = native_pg
    logs = SignalLogStore(backend)
    await logs.initialize()
    owned = []

    def track(coro, *, name):
        task = asyncio.create_task(coro, name=name)
        owned.append(task)
        return task

    dispatcher = SignalDispatcher(
        agent=SimpleNamespace(_track_background_task=track),
        registry=SourceRegistry(),
        lock_manager=OrderedLockManager(),
        store=logs,
    )

    async def native_run(*args, **kwargs):
        async with backend.transaction():
            await backend.execute("INSERT INTO effects VALUES (1, 'committed')")

    monkeypatch.setattr(dispatcher, "_run", native_run)
    lose_ack(monkeypatch, cancelled=cancelled)
    message = ChannelMessage(
        channel_type="telegram",
        direction=MessageDirection.INBOUND,
        sender="555",
        recipient="42",
        content="native admission",
        id="telegram:v2:bot:42:update:101",
    )
    handle = await dispatcher.enqueue_signal(
        build_signal_for_channel_message(message, target_agent="did:test:native")
    )
    try:
        with pytest.raises(BaseException) as caught:
            await handle.wait_for_durable_admission()
        assert execution_commit_outcome(caught.value) == "unknown"
        assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 1
        outcome = await await_owned_task(handle.task)
        assert outcome.error is caught.value
    finally:
        for task in owned:
            await await_owned_task(task)
