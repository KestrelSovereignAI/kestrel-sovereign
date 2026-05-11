"""Source registration tests for inbound channel messages."""

from __future__ import annotations

import pytest

from kestrel_sdk.channels import ChannelMessage, MessageDirection
from kestrel_sdk.signals import SignalMode, Trust, Visibility
from kestrel_sovereign.signals import SourceRegistry
from kestrel_sovereign.signals.sources.channels import (
    SOURCE_NAME,
    _channel_sanitize,
    build_channel_message_registration,
    build_signal_for_channel_message,
)


def test_registration_is_untrusted_cognition_source():
    reg = build_channel_message_registration()
    assert reg.name == SOURCE_NAME
    assert reg.allowed_modes == frozenset({SignalMode.COGNITION})
    assert reg.trust is Trust.UNTRUSTED
    assert reg.sanitizer is not None
    assert reg.prompt_template.exists()

    registry = SourceRegistry()
    registry.register(reg)
    assert registry.get(SOURCE_NAME) is reg


def test_sanitizer_caps_content_and_metadata():
    payload = _channel_sanitize(
        {
            "message_id": "m1",
            "channel_type": "telegram",
            "sender": "alice",
            "recipient": "bot",
            "content": "x" * 5000,
            "metadata": {"thread": "y" * 1000, 42: "ignored"},
        }
    )
    assert payload["content"].endswith("...(truncated)")
    assert payload["metadata"]["thread"].endswith("...(truncated)")
    assert 42 not in payload["metadata"]


def test_sanitizer_preserves_invalid_required_fields_for_schema_rejection():
    payload = _channel_sanitize(
        {
            "message_id": None,
            "channel_type": "telegram",
            "sender": 42,
            "recipient": "bot",
            "content": ["not", "text"],
            "metadata": {},
        }
    )
    reg = build_channel_message_registration()

    assert payload["message_id"] is None
    assert payload["sender"] == 42
    assert payload["content"] == ["not", "text"]
    with pytest.raises(ValueError, match="must be a string"):
        reg.schema(payload)


def test_build_signal_for_channel_message():
    message = ChannelMessage(
        id="msg-1",
        channel_type="telegram",
        direction=MessageDirection.INBOUND,
        sender="alice",
        recipient="bot",
        content="hello",
        metadata={"thread": "abc"},
    )

    signal = build_signal_for_channel_message(message, "did:test:agent")

    assert signal.source == SOURCE_NAME
    assert signal.kind == "inbound"
    assert signal.mode is SignalMode.COGNITION
    assert signal.visibility is Visibility.USER_VISIBLE
    assert signal.target_agent == "did:test:agent"
    assert signal.payload["message_id"] == "msg-1"
    assert signal.dedupe_key == "telegram:msg-1"


def test_build_signal_rejects_outbound_messages():
    message = ChannelMessage(
        channel_type="telegram",
        direction=MessageDirection.OUTBOUND,
        sender="bot",
        recipient="alice",
        content="hello",
    )
    with pytest.raises(ValueError, match="inbound"):
        build_signal_for_channel_message(message, "did:test:agent")
