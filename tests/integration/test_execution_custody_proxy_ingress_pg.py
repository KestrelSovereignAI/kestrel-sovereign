"""Physical PostgreSQL COMMIT survives every proxy ingress/completion owner."""

import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    bind_execution_custody,
    execution_commit_outcome,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import (
    GenerationFence,
)
from tests.integration.test_execution_authority_postgres import (
    native_pg as native_pg,
)
from tests.integration.test_execution_custody_review30_pg import lose_ack
from tests.unit.test_execution_custody_proxy_ingress import owner
from tests.unit.test_execution_custody_review29 import retire

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("privacy", ["normal", "ephemeral"])
@pytest.mark.parametrize("denied", [False, True])
async def test_native_signal_admission_fence_is_commit_only_not_consumer_custody(native_pg, privacy, denied):
    """Use the original event transaction; release its constraint before action."""
    from uuid import uuid4

    from kestrel_sdk.signals import (
        RedactionPolicy,
        Signal,
        SignalMode,
        SourceRegistration,
        Trust,
    )

    from kestrel_sovereign.execution_custody import current_execution_custody
    from kestrel_sovereign.privacy import get_privacy_preset
    from kestrel_sovereign.signals import (
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )
    from tests.integration.test_durable_signal_backend_parity import (
        _DispatcherAgent,
        _finish_dispatcher,
    )

    backend, control, schema = native_pg
    agent = _DispatcherAgent("did:test:commit-fence:" + uuid4().hex, get_privacy_preset(privacy))
    registry = SourceRegistry()
    calls = []

    async def handler(payload):
        assert not current_execution_custody(), "Finite admission scope leaked to consumer"
        # A real independent generation writer must progress after admission
        # commits. Consumer work is not incorrectly tied to the prior fence.
        await asyncio.wait_for(control.execute(f'UPDATE "{schema}".authority SET generation=2 WHERE id=1'), 2)
        await backend.execute("INSERT INTO effects VALUES (1, 'admitted action')")
        calls.append(payload)
        return {"handled": True}

    registry.register(SourceRegistration(
        name="authority.action", schema=lambda payload: payload,
        default_mode=SignalMode.ACTION, allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler, trust=Trust.TRUSTED,
        log_redaction=RedactionPolicy(summarize=lambda _: "<controlled>"), retention_days=7,
    ))
    log = SignalLogStore(backend)
    await log.initialize()
    dispatcher = SignalDispatcher(agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=log)
    await dispatcher.initialize_durable_delivery()
    if denied:
        await backend.execute("UPDATE authority SET generation=2 WHERE id=1")
    signal = Signal(source="authority.action", kind="controlled", mode=SignalMode.ACTION,
                    payload={"controlled": True}, target_agent=agent.did)
    try:
        if denied:
            with pytest.raises(Exception, match="runtime generation changed"):
                await dispatcher.dispatch_signal(signal, source_event_id=signal.id,
                                                  _durable_admission_fence=GenerationFence())
        else:
            await dispatcher.dispatch_signal(signal, source_event_id=signal.id,
                                             _durable_admission_fence=GenerationFence())
        assert len(calls) == (0 if denied else 1)
        assert await backend.fetch_val("SELECT count(*) FROM effects") == (0 if denied else 1)
        events = dispatcher._durable_store.EVENTS
        assert await backend.fetch_val(f"SELECT count(*) FROM {events} WHERE event_id=?", (signal.id,)) == (0 if denied else 1)
    finally:
        await _finish_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("denied", [False, True])
async def test_legacy_delivery_creation_rechecks_original_admission_commit(native_pg, monkeypatch, denied):
    """A duplicate event is not permission to create a new legacy delivery."""
    from uuid import uuid4

    from kestrel_sovereign.execution_custody import is_execution_control_error
    from kestrel_sovereign.privacy import get_privacy_preset
    from kestrel_sovereign.signals import (
        DurableConsumerRegistration, OrderedLockManager, SignalDispatcher,
        SignalLogStore, SourceRegistry,
    )
    from kestrel_sovereign.signals.sources.channels import (
        DURABLE_COGNITION_CONSUMER_ID, DURABLE_COGNITION_MARKER,
        DURABLE_COGNITION_MARKER_VALUE, build_channel_message_registration,
    )
    from tests.integration.test_durable_signal_backend_parity import (
        _DispatcherAgent, _finish_dispatcher,
    )
    from tests.unit.test_durable_signal_delivery import _channel_signal

    backend, control, schema = native_pg
    agent = _DispatcherAgent("did:test:legacy-fence:" + uuid4().hex, get_privacy_preset("normal"))
    registry = SourceRegistry()
    registry.register(build_channel_message_registration())
    log = SignalLogStore(backend)
    await log.initialize()
    dispatcher = SignalDispatcher(agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=log)
    await dispatcher.initialize_durable_delivery()
    store = dispatcher._durable_store
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID, source="channel.message", agent_id=agent.did,
        correlation_selector=f"payload.{DURABLE_COGNITION_MARKER}={DURABLE_COGNITION_MARKER_VALUE}",
        max_attempts=0,
    )
    legacy = _channel_signal(agent.did, "original-legacy")
    legacy.payload.pop(DURABLE_COGNITION_MARKER)
    observed = []
    try:
        persisted = await store.persist_signal(legacy, agent_id=agent.did, source_event_id="telegram:update:legacy", retention_days=14)
        await dispatcher.register_durable_consumer(consumer)
        assert await dispatcher.list_durable_deliveries() == []
        persist = store.persist_signal

        async def after_original_event_commit(*args, **kwargs):
            result = await persist(*args, **kwargs)
            assert not result.created and result.event_id == persisted.event_id
            # The first original transaction has really committed/released its
            # SHARE row lock. An independent writer can now retire that scope.
            if denied:
                await asyncio.wait_for(control.execute(f'UPDATE "{schema}".authority SET generation=2 WHERE id=1'), 2)
            observed.append(result.event_id)
            return result

        monkeypatch.setattr(store, "persist_signal", after_original_event_commit)
        handle = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "original-legacy"), source_event_id="telegram:update:legacy",
            consumer_id=consumer.consumer_id, admission_fence=GenerationFence(),
        )
        error = None
        try:
            await handle.wait()
        except BaseException as caught:
            error = caught
        deliveries = await dispatcher.list_durable_deliveries()
        assert len(deliveries) == (0 if denied else 1), "Legacy delivery bypassed original admission generation"
        assert observed == [persisted.event_id]
        if denied:
            assert is_execution_control_error(error)
            row = await backend.fetch_one(f"SELECT caller_identity FROM {store.EVENTS} WHERE event_id=?", (persisted.event_id,))
            assert row == (None,), "Denied upgrade partially committed caller identity"
    finally:
        await _finish_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("route", ["deferred", "legacy", "ack", "retry"])
async def test_native_commit_receipt_reaches_terminal_owner_after_worker_removal(
    native_pg, monkeypatch, tmp_path, cancelled, route
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)

    async def narrower_native_effect():
        # Reader callbacks are independent resident inbound events, not
        # continuations of this fixture's outer publication admission. The
        # actual native effect owns its narrower original generation fence.
        with bind_execution_custody(GenerationFence()):
            await db.execute("INSERT INTO effects (id) VALUES (1)")

    agent, _, feature, client, event, _, release = await owner(
        monkeypatch,
        tmp_path,
        None,
        completion_failure=route in {"ack", "retry"},
        native_database=db,
        native_effect=narrower_native_effect,
    )
    commit_entered, commit_release, completed = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    lose_ack(
        monkeypatch, cancelled=cancelled, entered=commit_entered, release=commit_release
    )
    task = None
    try:
        with bind_execution_custody(GenerationFence()):
            if route == "deferred":
                await feature._traffic_gate.close()
                await client.event_handler(event)
                (task,) = tuple(feature._deferred_acknowledged_event_tasks)
                await feature._traffic_gate.reopen()
            elif route == "legacy":
                await client.event_handler(
                    {"type": "channel.inbound", "payload": event["payload"]["message"]}
                )
                (task,) = tuple(feature._event_ingress_tasks)
            else:
                data = event["payload"]
                request = (
                    feature._split_inbound_event_acknowledgement(data)[1]
                    if route == "ack"
                    else feature._inbound_event_retry_completion(data)
                )
                feature._schedule_event_ingress_completion(
                    client,
                    request,
                    kind=route,
                    expected_state="acknowledged" if route == "ack" else "retrying",
                )
                (task,) = tuple(feature._event_ack_tasks)
            task.add_done_callback(lambda done: completed.set())
            await asyncio.wait_for(commit_entered.wait(), 5)
            # Independent physical observer proves the effect before losing
            # its ACK. This is neither a mocked commit nor synthetic error.
            assert (
                await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 1
            )
            commit_release.set()
            await asyncio.wait_for(completed.wait(), 5)
        result = await await_owned_task(task)
        assert result.error is not None, "native COMMIT uncertainty was discarded"
        assert execution_commit_outcome(result.error) == "unknown"
        assert not feature._event_ingress_tasks
        assert not feature._event_ack_tasks
        assert not feature._event_ack_clients
        assert not feature._deferred_acknowledged_event_tasks
        assert client.completions == (
            []
            if route in {"deferred", "legacy"}
            else ["telegram-polling-ack" if route == "ack" else "telegram-polling-nack"]
        )
        with pytest.raises(BaseException) as caught:
            await feature.shutdown()
        assert caught.value is result.error
        assert execution_commit_outcome(caught.value) == "unknown"
        assert client.stopped
        assert feature._client is None
    finally:
        release.set()
        commit_release.set()
        if task is not None:
            await await_owned_task(task)
        await retire(agent)
