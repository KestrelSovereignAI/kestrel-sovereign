"""Actual channel-history writes preserve UTC on both SQL drivers (#3768)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign.features.channels.feature import ChannelFeature
from kestrel_sovereign.features.channels.models import ChannelMessage, MessageDirection
from kestrel_sovereign.signals import SourceRegistry
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.async_database import AsyncDatabase


async def assert_channel_timestamp_roundtrip(backend, offset):
    agent = SimpleNamespace(
        did="did:test:channel-time",
        storage=SimpleNamespace(db=AsyncDatabase(backend), agent_id="did:test:channel-time"),
        signal_registry=SourceRegistry(), dispatcher=None,
    )
    channel = ChannelFeature(agent)
    await channel.initialize()
    expected = datetime(2026, 10, 20, 22, 25, 42, tzinfo=timezone.utc)
    message = ChannelMessage(
        id="timestamp", channel_type="slack", direction=MessageDirection.INBOUND,
        sender="sender", recipient="bot", content="native history",
        timestamp=expected.astimezone(timezone(timedelta(hours=offset))),
    )
    await channel._log_message(message)
    row = await backend.fetch_one(
        "SELECT created_at, content FROM channel_messages WHERE id = ?", (message.id,),
    )
    assert row is not None, "channel history insert was lost"
    actual = row[0] if isinstance(row[0], datetime) else datetime.fromisoformat(row[0])
    if backend.backend_type == "postgres":
        assert actual.tzinfo is None
        actual = actual.replace(tzinfo=timezone.utc)
    assert actual.utcoffset() == timedelta(0), "history must be normalized to UTC"
    assert actual == expected
    history = await channel.channels_history(limit=1)
    assert history.data["messages"][0]["content"] == "native history"


@pytest.mark.asyncio
@pytest.mark.parametrize("offset", [0, -5, 9])
async def test_channel_history_real_sqlite_utc_roundtrip(tmp_path, offset):
    backend = SQLiteBackend(str(tmp_path / "history.db"))
    await backend.connect()
    try:
        await assert_channel_timestamp_roundtrip(backend, offset)
    finally:
        await backend.close()
