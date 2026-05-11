"""Signal source for inbound channel messages."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from kestrel_sdk.channels import ChannelMessage, MessageDirection
from kestrel_sdk.signals import (
    AttentionPolicy,
    RateLimit,
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Trust,
    Urgency,
    Visibility,
)


SOURCE_NAME = "channel.message"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "signals"
    / "channel_message.md"
)


def _cap_text(value: str, limit: int = 2000) -> str:
    text = "".join(ch for ch in value if ch.isprintable() or ch in "\n\t")
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return text


def _cap_metadata_value(value: Any, limit: int = 500) -> str:
    return _cap_text("" if value is None else str(value), limit)


def _channel_schema(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"channel payload must be a dict, got {type(payload).__name__}")
    for key in ("channel_type", "sender", "recipient", "content", "message_id"):
        if key not in payload:
            raise ValueError(f"channel payload missing required key: {key}")
        if not isinstance(payload[key], str):
            raise ValueError(f"channel payload {key} must be a string")
    metadata = payload.get("metadata", {})
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("channel payload metadata must be a dict")
    return payload


def _channel_sanitize(payload: dict) -> dict:
    payload = dict(payload)
    for key, limit in (
        ("message_id", 128),
        ("channel_type", 80),
        ("sender", 200),
        ("recipient", 200),
        ("content", 4000),
    ):
        if isinstance(payload.get(key), str):
            payload[key] = _cap_text(payload[key], limit)
    metadata = payload.get("metadata") or {}
    payload["metadata"] = {
        _cap_text(key, 80): _cap_metadata_value(value)
        for key, value in metadata.items()
        if isinstance(key, str)
    }
    return payload


def _channel_redact(payload: dict) -> str:
    channel = payload.get("channel_type", "<missing>")
    sender = payload.get("sender", "<missing>")
    content = payload.get("content", "") or ""
    if len(content) > 120:
        content = content[:120] + "...(truncated)"
    return f"channel={channel} sender={sender} content={content!r}"


def build_channel_message_registration() -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_channel_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        trust=Trust.UNTRUSTED,
        sanitizer=_channel_sanitize,
        rate_limit=RateLimit(per_minute=30, per_hour=300),
        coalescing_window=timedelta(seconds=2),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_channel_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        retention_days=14,
    )


def build_signal_for_channel_message(
    message: ChannelMessage,
    target_agent: str,
) -> Signal:
    if message.direction != MessageDirection.INBOUND:
        raise ValueError("channel message signals require an inbound message")
    return Signal(
        source=SOURCE_NAME,
        kind="inbound",
        mode=SignalMode.COGNITION,
        payload={
            "message_id": message.id,
            "channel_type": message.channel_type,
            "sender": message.sender,
            "recipient": message.recipient,
            "content": message.content,
            "metadata": message.metadata,
        },
        target_agent=target_agent,
        visibility=Visibility.USER_VISIBLE,
        caller=message.sender,
        urgency=Urgency.NORMAL,
        dedupe_key=f"{message.channel_type}:{message.id}",
    )


__all__ = [
    "PROMPT_TEMPLATE",
    "SOURCE_NAME",
    "build_channel_message_registration",
    "build_signal_for_channel_message",
]
