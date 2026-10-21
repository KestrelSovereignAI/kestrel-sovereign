"""Native read-only receipt checks share ingress's privacy equality rule."""
from dataclasses import replace

import pytest

from kestrel_sovereign.privacy import get_privacy_preset
from kestrel_sovereign.signals.durable import DurableConsumerRegistration
from kestrel_sovereign.signals.sources.channels import (
    DURABLE_COGNITION_MARKER,
    DURABLE_TERMINAL_CONSUMER_ID,
    DURABLE_TERMINAL_MARKER,
    DURABLE_TERMINAL_MARKER_VALUE,
    build_channel_message_registration,
)
from tests.unit.test_durable_signal_delivery import (
    _channel_dispatcher,
    _channel_signal,
    _close,
)


def _terminal_signal(did):
    signal = _channel_signal(did, "telegram:v2:bot:42:update:17")
    signal.payload.pop(DURABLE_COGNITION_MARKER)
    signal.payload[DURABLE_TERMINAL_MARKER] = DURABLE_TERMINAL_MARKER_VALUE
    signal.payload["metadata"] = {"chat_id": "123", "reply_to_message_id": "777"}
    registration = build_channel_message_registration()
    signal.payload = registration.schema(registration.sanitizer(signal.payload))
    signal.origin_trust = registration.trust
    return signal


@pytest.mark.asyncio
@pytest.mark.parametrize("privacy", ["normal", "ephemeral", "isolated", "anonymous"])
async def test_terminal_read_verifies_native_envelope_without_re_admission(tmp_path, privacy):
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / "terminal.db", "did:agent:receipt")
    agent.privacy_config = get_privacy_preset(privacy)
    source_id = "telegram:v2:bot:42:update:17"
    try:
        await dispatcher.register_durable_consumer(DurableConsumerRegistration(
            consumer_id=DURABLE_TERMINAL_CONSUMER_ID, source="channel.message", agent_id=agent.did,
            correlation_selector=f"payload.{DURABLE_TERMINAL_MARKER}={DURABLE_TERMINAL_MARKER_VALUE}", max_attempts=0,
        ))
        handle = await dispatcher.enqueue_durable_terminal(
            _terminal_signal(agent.did), source_event_id=source_id,
            consumer_id=DURABLE_TERMINAL_CONSUMER_ID,
        )
        assert (await handle.wait_for_durable_admission()).acknowledged
        await handle.wait()
        kwargs = {"consumer_id": DURABLE_TERMINAL_CONSUMER_ID, "source": "channel.message", "source_event_id": source_id}
        before = await dispatcher.get_durable_delivery_for_source_event(**kwargs)
        assert before is not None
        if privacy in {"ephemeral", "isolated"}:
            assert "content" not in before.event.payload
        assert await dispatcher.get_durable_delivery_for_source_event(
            **kwargs, expected_channel_signal=_terminal_signal(agent.did),
        ) == before
        for field in ("content", "sender", "recipient"):
            changed = _terminal_signal(agent.did)
            changed.payload[field] = "foreign"
            assert await dispatcher.get_durable_delivery_for_source_event(
                **kwargs, expected_channel_signal=changed,
            ) is None, field
        changed = _terminal_signal(agent.did)
        changed.payload["metadata"]["reply_to_message_id"] = "778"
        assert await dispatcher.get_durable_delivery_for_source_event(
            **kwargs, expected_channel_signal=changed,
        ) is None
        for change in ({"target_agent": "did:foreign"}, {"caller": "foreign"}, {"dedupe_key": "foreign"}):
            assert await dispatcher.get_durable_delivery_for_source_event(
                **kwargs, expected_channel_signal=replace(_terminal_signal(agent.did), **change),
            ) is None
        assert await dispatcher.get_durable_delivery_for_source_event(**kwargs) == before
        assert (await backend.fetch_all("SELECT COUNT(*) FROM durable_signal_events"))[0][0] == 1
        assert len(await dispatcher.list_durable_deliveries()) == 1
        # Reopen the actual database under a fresh dispatcher. Original
        # privacy, not the retry's now-NORMAL policy, owns envelope equality.
        did = agent.did
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)
        backend, agent, dispatcher = await _channel_dispatcher(tmp_path / "terminal.db", did)
        agent.privacy_config = get_privacy_preset("normal")
        assert await dispatcher.get_durable_delivery_for_source_event(
            **kwargs, expected_channel_signal=_terminal_signal(agent.did),
        ) == before
        if privacy in {"ephemeral", "isolated"}:
            await backend.execute("DELETE FROM durable_signal_event_integrity")
            assert await dispatcher.get_durable_delivery_for_source_event(
                **kwargs, expected_channel_signal=_terminal_signal(agent.did),
            ) is None
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)
