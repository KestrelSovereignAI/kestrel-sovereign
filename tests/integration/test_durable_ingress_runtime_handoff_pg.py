"""Real PostgreSQL ingress retirement and channel history (#3768)."""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, bind_execution_custody,
    current_execution_custody, require_execution_work, _explicit_control_chain,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import get_privacy_preset
from kestrel_sovereign.signals import (
    ACKNOWLEDGED, DurableConsumerRegistration, OrderedLockManager,
    SignalDispatcher, SignalLogStore, SourceRegistry,
)
from kestrel_sovereign.signals.sources.channels import (
    DURABLE_COGNITION_CONSUMER_ID, DURABLE_COGNITION_MARKER,
    DURABLE_COGNITION_MARKER_VALUE, build_channel_message_registration,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import (
    GenerationFence, native_pg as native_pg,
)
from tests.unit.test_channel_timestamp_contract import assert_channel_timestamp_roundtrip
from tests.unit.test_durable_signal_delivery import _channel_signal
from tests.utils.runtime_task_factory import refusing_runtime_completion_factory

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("offset", [0, -5, 9])
@pytest.mark.parametrize("nullable", [False, True])
async def test_real_pg_sleep_episode_timestamp_binding_and_native_reader(native_pg, offset, nullable):
    """Exercise real persistence only: no sleep/reflection/provider invocation."""
    import re
    from datetime import datetime, timedelta, timezone
    from kestrel_sovereign.storage.async_database import core_schema_sql
    from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator
    from kestrel_sovereign.storage.memory_models import MemoryEpisode

    backend, _, _ = native_pg
    db = AsyncDatabase(backend)
    statements = re.findall(r"CREATE TABLE IF NOT EXISTS memory_episodes \([\s\S]*?\n\);", core_schema_sql("postgres"))
    assert len(statements) == 1
    await db.execute(statements[0])
    zone = timezone(timedelta(hours=offset))
    start = datetime(2026, 10, 20, 17, 15, 44, 971730, tzinfo=zone)
    end = start + timedelta(minutes=3)
    episode = MemoryEpisode(
        id="native-sleep-episode", agent_id="did:test:sleep-episode",
        title="Provider-free native persistence", summary="Typed timestamp boundary",
        timespan_start=None if nullable else start, timespan_end=None if nullable else end,
        key_message_ids=["one", "two"],
    )
    consolidator = MemoryConsolidator(db, episode.agent_id)
    await consolidator._save_episode(episode)
    row = await db.fetchone("SELECT timespan_start,timespan_end,created_at FROM memory_episodes")
    expected_start = None if nullable else start.astimezone(timezone.utc).replace(tzinfo=None)
    expected_end = None if nullable else end.astimezone(timezone.utc).replace(tzinfo=None)
    assert row[0] == expected_start
    assert row[1] == expected_end
    assert isinstance(row[2], datetime)
    episodes = await consolidator.get_episodes()
    assert len(episodes) == 1
    assert episodes[0].timespan_start == expected_start
    assert episodes[0].timespan_end == expected_end
    assert episodes[0].created_at == row[2]
    assert episodes[0].key_message_ids == ["one", "two"]


@pytest.mark.parametrize("privacy", ["normal", "ephemeral", "isolated", "anonymous"])
async def test_real_pg_host_ack_targets_one_run_and_receipt_binds_session(native_pg, privacy):
    """Real proxy, producer, persistence and session selectors; no Workflows substitute claim."""
    from dataclasses import replace
    from kestrel_sdk.signals import Trust, Visibility
    from kestrel_sovereign.features.channels.feature import ChannelFeature, InboundAdmissionDisposition
    from kestrel_sovereign.features.channels.models import ChannelConfig, ChannelMessage, MessageDirection
    from kestrel_sovereign.features.isolated_runtime import ProxyFeature
    from kestrel_sovereign.signals.sources.channels import (
        DURABLE_TERMINAL_CONSUMER_ID, DURABLE_TERMINAL_MARKER,
        DURABLE_TERMINAL_MARKER_VALUE, build_signal_for_channel_message,
    )
    from tests.unit.test_channels_feature import StubAdapter
    from tests.unit.test_durable_signal_delivery import _Agent

    backend, control, schema = native_pg
    agent = _Agent("did:test:pg-run-bound-ack")
    agent.storage = SimpleNamespace(db=AsyncDatabase(backend), agent_id=agent.did)
    agent.privacy_config = get_privacy_preset(privacy)
    logs = SignalLogStore(backend)
    await logs.initialize()
    registry = SourceRegistry()
    registry.register(build_channel_message_registration())
    dispatcher = SignalDispatcher(agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=logs)
    agent.dispatcher = dispatcher
    agent.signal_registry = registry
    channel = ChannelFeature(agent)
    source_id = "telegram:v2:bot:42:update:19"
    message = ChannelMessage(
        id=source_id, channel_type="telegram", direction=MessageDirection.INBOUND,
        sender="123", recipient="123", content=" OK ", agent_id=agent.did,
        metadata={"dedupe_key": source_id, "chat_id": "123", "reply_to_message_id": "777",
                  "session_id": "forged-run"},
    )
    proxy = object.__new__(ProxyFeature)
    proxy.agent = agent
    proxy._require_event_ingress_work = lambda: None
    proxy._channel_feature = lambda: channel
    proxy._authoritative_inbound_channel_type = lambda: "telegram"
    try:
        await channel.initialize()
        channel.registry.register(StubAdapter(
            channel="telegram", config=ChannelConfig(channel_type="telegram", allowed_senders=["123"]),
        ))
        for run in ("run-a", "run-b"):
            await dispatcher.register_durable_consumer(DurableConsumerRegistration(
                consumer_id=f"workflow-{run}", source="channel.message", agent_id=agent.did,
                correlation_selector=f"session_id={run}",
            ))
        admission = await proxy._route_validated_inbound(
            message.to_dict(), cursor_owned_protocol=True,
            telegram_terminal_disposition="care_acknowledgement",
            acknowledgement_session_id="run-a",
        )
        assert admission.disposition == InboundAdmissionDisposition.DURABLY_ADMITTED
        expected = build_signal_for_channel_message(message, target_agent=agent.did, session_id="run-a")
        expected.payload[DURABLE_TERMINAL_MARKER] = DURABLE_TERMINAL_MARKER_VALUE
        registration = registry.get("channel.message")
        expected.payload = registration.schema(registration.sanitizer(expected.payload))
        expected.origin_trust = registration.trust
        args = dict(consumer_id=DURABLE_TERMINAL_CONSUMER_ID, source="channel.message", source_event_id=source_id)
        async with asyncio.timeout(5):
            while True:
                receipt = await dispatcher.get_durable_delivery_for_source_event(**args)
                if receipt is not None and receipt.status == "terminal_ackable":
                    break
                await asyncio.sleep(0.01)
        receipt = await dispatcher.get_durable_delivery_for_source_event(**args, expected_channel_signal=expected)
        assert receipt is not None
        assert receipt.event.session_id == "run-a"
        assert receipt.event.visibility == Visibility.USER_VISIBLE.value
        assert registry.get("channel.message").trust == Trust.UNTRUSTED
        for session in (None, "run-b"):
            assert await dispatcher.get_durable_delivery_for_source_event(
                **args, expected_channel_signal=replace(expected, session_id=session),
            ) is None
        deliveries = await dispatcher.list_durable_deliveries()
        assert {delivery.consumer_id for delivery in deliveries} == {DURABLE_TERMINAL_CONSUMER_ID, "workflow-run-a"}
        assert agent.action_calls == 0
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 0
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".durable_signal_events') == 1
    finally:
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


async def retire_handoff_test(agent, dispatcher):
    """A failing dispatcher close cannot skip its original native task joins."""
    try:
        await dispatcher.shutdown_durable_delivery()
    finally:
        tasks = tuple(agent._background_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("offset", [0, -5, 9])
async def test_channel_history_real_postgres_utc_roundtrip(native_pg, offset):
    backend, _, _ = native_pg
    await assert_channel_timestamp_roundtrip(backend, offset)


@pytest.mark.parametrize("privacy", ["normal", "ephemeral", "isolated", "anonymous"])
async def test_real_pg_terminal_receipt_proves_exact_private_envelope_without_work(native_pg, privacy):
    from kestrel_sovereign.signals.sources.channels import (
        DURABLE_TERMINAL_CONSUMER_ID, DURABLE_TERMINAL_MARKER,
        DURABLE_TERMINAL_MARKER_VALUE,
    )
    from tests.unit.test_channel_terminal_receipt_verification import _terminal_signal
    from tests.unit.test_durable_signal_delivery import _Agent

    backend, control, schema = native_pg
    agent = _Agent("did:test:pg-private-receipt")
    agent.privacy_config = get_privacy_preset(privacy)
    logs = SignalLogStore(backend)
    await logs.initialize()
    registry = SourceRegistry()
    registry.register(build_channel_message_registration())
    dispatcher = SignalDispatcher(agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=logs)
    source_id = "telegram:v2:bot:42:update:17"
    try:
        await dispatcher.register_durable_consumer(DurableConsumerRegistration(
            consumer_id=DURABLE_TERMINAL_CONSUMER_ID, source="channel.message", agent_id=agent.did,
            correlation_selector=f"payload.{DURABLE_TERMINAL_MARKER}={DURABLE_TERMINAL_MARKER_VALUE}", max_attempts=0,
        ))
        handle = await dispatcher.enqueue_durable_terminal(
            _terminal_signal(agent.did), source_event_id=source_id, consumer_id=DURABLE_TERMINAL_CONSUMER_ID,
        )
        assert (await handle.wait_for_durable_admission()).acknowledged
        await handle.wait()
        args = {"consumer_id": DURABLE_TERMINAL_CONSUMER_ID, "source": "channel.message", "source_event_id": source_id}
        before = await dispatcher.get_durable_delivery_for_source_event(**args)
        assert before is not None
        if privacy in {"ephemeral", "isolated"}:
            assert "content" not in before.event.payload
        assert await dispatcher.get_durable_delivery_for_source_event(
            **args, expected_channel_signal=_terminal_signal(agent.did),
        ) == before
        changed = _terminal_signal(agent.did)
        changed.payload["metadata"]["reply_to_message_id"] = "778"
        assert await dispatcher.get_durable_delivery_for_source_event(
            **args, expected_channel_signal=changed,
        ) is None
        assert await dispatcher.get_durable_delivery_for_source_event(**args) == before
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".durable_signal_events') == 1
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 0
    finally:
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


@pytest.mark.parametrize("ready_at_admission", [False, True])
@pytest.mark.parametrize("publication_refusal", [False, True])
async def test_real_pg_cognition_and_audit_survive_http_admission_retirement(
    native_pg, tmp_path, ready_at_admission, publication_refusal,
):
    backend, control, schema = native_pg
    proceed, entered = asyncio.Event(), asyncio.Event()

    class ControlledAgent(KestrelAgent):
        async def process_input(self, prompt):
            require_execution_work(self)
            assert current_execution_custody() == (self._execution_custody,)
            entered.set()
            await proceed.wait()
            await self.storage.db.execute("INSERT INTO effects (id, value) VALUES (2, 'resident')")
            return "controlled, provider-free response"

    async def close_test_service():
        pass

    agent = ControlledAgent(
        did="did:test:pg-ingress", storage_path=str(tmp_path / "unused.db"),
        llm_service=SimpleNamespace(close=close_test_service),
    )
    agent.privacy_agent = SimpleNamespace(privacy_config=get_privacy_preset("normal"))
    agent._execution_custody = ExecutionCustody(GenerationFence())
    backend._execution_custody = agent._execution_custody
    storage = AsyncStorage(backend=backend)
    storage.db = AsyncDatabase(backend)
    agent.storage = agent._raw_storage = storage
    if ready_at_admission:
        agent._runtime_publication_ready.set()
    logs = SignalLogStore(backend)
    await logs.initialize()
    registry = SourceRegistry()
    registry.register(build_channel_message_registration())
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=logs,
    )
    agent.dispatcher = dispatcher
    consumer = DURABLE_COGNITION_CONSUMER_ID
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    publication_error = OSError("native completion callback refused")
    try:
        await dispatcher.initialize_durable_delivery()
        await dispatcher.register_durable_consumer(DurableConsumerRegistration(
            consumer_id=consumer, source="channel.message", agent_id=agent.did,
            correlation_selector=f"payload.{DURABLE_COGNITION_MARKER}={DURABLE_COGNITION_MARKER_VALUE}",
            max_attempts=0,
        ))
        if publication_refusal:
            loop.set_task_factory(refusing_runtime_completion_factory(
                publication_error,
            ))
        with bind_execution_custody(GenerationFence()) as ingress:
            handle = await dispatcher.enqueue_durable_cognition(
                _channel_signal(agent.did, "native"),
                source_event_id="telegram:update:native", consumer_id=consumer,
            )
            receipt = await asyncio.wait_for(handle.wait_for_durable_admission(), 5)
            if publication_refusal:
                loop.set_task_factory(previous_factory)
                assert not receipt.acknowledged
                try:
                    await asyncio.wait_for(handle.wait(), 5)
                except BaseException as error:
                    # Cancelled accepted children retain the opener in the
                    # native finite error rather than leaking a live Task.
                    assert publication_error in tuple(_explicit_control_chain(error))
                assert handle.task.done()
                assert not entered.is_set()
                delivery = await dispatcher.get_durable_delivery_for_event(
                    consumer_id=consumer, event_id=receipt.signal_id,
                )
                assert delivery.status == "retry"
                assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=2') == 0
                assert not any(not task.done() for task in agent._background_tasks)
                return
            assert receipt.acknowledged and receipt.resident_owned
            if not ready_at_admission:
                assert not entered.is_set(), "effects ran before real READY"
        with pytest.raises(ExecutionAuthorityError, match="admission retired"):
            ingress.require_work()
        agent._runtime_publication_ready.set()
        await asyncio.wait_for(entered.wait(), 5)
        proceed.set()
        await asyncio.wait_for(handle.wait(), 10)
        delivery = await dispatcher.get_durable_delivery_for_event(
            consumer_id=consumer, event_id=receipt.signal_id,
        )
        assert delivery.status == ACKNOWLEDGED
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=2') == 1
        assert dispatcher._outcome_log_control_error is None
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".signal_log') >= 1
    finally:
        loop.set_task_factory(previous_factory)
        proceed.set()
        agent._runtime_publication_ready.set()
        await retire_handoff_test(agent, dispatcher)
