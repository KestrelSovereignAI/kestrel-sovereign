"""
Unit Tests for the Channels Feature.

Covers:
- ChannelMessage / DeliveryReceipt / ChannelConfig data models
- ChannelAdapter abstract interface contract
- ChannelRegistry register/unregister/lookup/routing
- ChannelFeature tools: channels_list, channels_send, channels_history
- Allowed-sender filtering
- Inbound message handling and logging
- Graceful degradation when DB is unavailable
- Tool discovery via get_tools()
"""

import json
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.channels.models import (
    ChannelConfig,
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
    MessageDirection,
)
from kestrel_sovereign.features.channels.adapter import ChannelAdapter
from kestrel_sovereign.features.channels.registry import ChannelRegistry
from kestrel_sovereign.features.channels.feature import ChannelFeature
from kestrel_sdk.channels import ChannelMessage as SDKChannelMessage


# ============================================================================
# Helpers
# ============================================================================


class StubAdapter(ChannelAdapter):
    """Minimal concrete adapter for testing."""

    def __init__(
        self,
        channel: str = "test",
        connected: bool = True,
        config: Optional[ChannelConfig] = None,
    ):
        super().__init__(config=config)
        self._channel = channel
        self._connected = connected
        self._callbacks = []
        self._sent: List[Dict[str, Any]] = []

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def send_message(self, to: str, content: str, **kwargs) -> DeliveryReceipt:
        self._sent.append({"to": to, "content": content, **kwargs})
        return DeliveryReceipt(
            message_id=f"msg-{len(self._sent)}",
            status=DeliveryStatus.SUCCESS,
            channel_type=self._channel,
        )

    async def on_message(self, callback) -> None:
        self._callbacks.append(callback)

    @property
    def channel_type(self) -> str:
        return self._channel

    @property
    def is_connected(self) -> bool:
        return self._connected


class FailingAdapter(StubAdapter):
    """Adapter whose send_message always raises."""

    async def send_message(self, to: str, content: str, **kwargs) -> DeliveryReceipt:
        raise ConnectionError("channel down")


def _make_db(fetchall_data=None, fetchone_data=None):
    """Create a mock AsyncDatabase."""
    db = AsyncMock()
    db.fetchall = AsyncMock(return_value=fetchall_data or [])
    db.fetchone = AsyncMock(return_value=fetchone_data)
    db.execute = AsyncMock(return_value=0)
    db.table_exists = AsyncMock(return_value=True)
    return db


def _make_agent(db=None, agent_id="test-agent"):
    """Create a mock KestrelAgent."""
    agent = MagicMock()
    agent.agent_id = agent_id

    storage = MagicMock()
    storage.db = db
    storage.agent_id = agent_id
    agent.storage = storage
    agent._raw_storage = None

    return agent


# ============================================================================
# Model Tests
# ============================================================================


class TestChannelMessage:
    def test_channel_message_uses_sdk_contract(self):
        assert ChannelMessage is SDKChannelMessage

    def test_defaults(self):
        msg = ChannelMessage(
            channel_type="telegram",
            direction=MessageDirection.INBOUND,
            sender="alice",
            recipient="bot",
            content="hello",
        )
        assert msg.channel_type == "telegram"
        assert msg.direction == MessageDirection.INBOUND
        assert msg.sender == "alice"
        assert msg.content == "hello"
        assert msg.id  # auto-generated uuid
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}

    def test_to_dict(self):
        msg = ChannelMessage(
            channel_type="discord",
            direction=MessageDirection.OUTBOUND,
            sender="bot",
            recipient="bob",
            content="hi bob",
            id="fixed-id",
        )
        d = msg.to_dict()
        assert d["id"] == "fixed-id"
        assert d["direction"] == "outbound"
        assert d["channel_type"] == "discord"


class TestDeliveryReceipt:
    def test_success(self):
        r = DeliveryReceipt(
            message_id="m1",
            status=DeliveryStatus.SUCCESS,
            channel_type="slack",
        )
        assert r.error is None
        d = r.to_dict()
        assert d["status"] == "success"

    def test_failure(self):
        r = DeliveryReceipt(
            message_id="m2",
            status=DeliveryStatus.FAILURE,
            channel_type="telegram",
            error="rate limited",
        )
        assert r.error == "rate limited"
        assert r.to_dict()["status"] == "failure"


class TestChannelConfig:
    def test_is_sender_allowed_empty_list(self):
        cfg = ChannelConfig(channel_type="telegram")
        assert cfg.is_sender_allowed("anyone") is True

    def test_is_sender_allowed_with_allowlist(self):
        cfg = ChannelConfig(
            channel_type="telegram",
            allowed_senders=["alice", "bob"],
        )
        assert cfg.is_sender_allowed("alice") is True
        assert cfg.is_sender_allowed("eve") is False

    def test_to_dict_hides_api_key(self):
        cfg = ChannelConfig(
            channel_type="telegram",
            api_key="super-secret",
        )
        d = cfg.to_dict()
        assert "api_key" not in d
        assert d["has_api_key"] is True


# ============================================================================
# Adapter Tests
# ============================================================================


class TestChannelAdapterContract:
    """Verify that a concrete adapter satisfies the abstract interface."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        adapter = StubAdapter(connected=False)
        assert adapter.is_connected is False
        await adapter.connect()
        assert adapter.is_connected is True
        await adapter.disconnect()
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_send_message(self):
        adapter = StubAdapter()
        receipt = await adapter.send_message(to="user1", content="ping")
        assert receipt.status == DeliveryStatus.SUCCESS
        assert receipt.channel_type == "test"
        assert len(adapter._sent) == 1
        assert adapter._sent[0]["to"] == "user1"

    @pytest.mark.asyncio
    async def test_on_message_registers_callback(self):
        adapter = StubAdapter()
        cb = AsyncMock()
        await adapter.on_message(cb)
        assert len(adapter._callbacks) == 1

    def test_channel_type(self):
        adapter = StubAdapter(channel="telegram")
        assert adapter.channel_type == "telegram"

    def test_config_property(self):
        cfg = ChannelConfig(channel_type="slack")
        adapter = StubAdapter(config=cfg)
        assert adapter.config is cfg

    def test_config_default_none(self):
        adapter = StubAdapter()
        assert adapter.config is None


# ============================================================================
# Registry Tests
# ============================================================================


class TestChannelRegistry:
    def test_register_and_get(self):
        reg = ChannelRegistry()
        adapter = StubAdapter(channel="telegram")
        reg.register(adapter)
        assert reg.get("telegram") is adapter
        assert reg.adapter_count == 1

    def test_unregister(self):
        reg = ChannelRegistry()
        adapter = StubAdapter(channel="discord")
        reg.register(adapter)
        removed = reg.unregister("discord")
        assert removed is adapter
        assert reg.get("discord") is None
        assert reg.adapter_count == 0

    def test_unregister_missing(self):
        reg = ChannelRegistry()
        removed = reg.unregister("nonexistent")
        assert removed is None

    def test_contains(self):
        reg = ChannelRegistry()
        reg.register(StubAdapter(channel="slack"))
        assert "slack" in reg
        assert "telegram" not in reg

    def test_list_channels(self):
        reg = ChannelRegistry()
        reg.register(StubAdapter(channel="telegram", connected=True))
        reg.register(StubAdapter(channel="discord", connected=False))
        channels = reg.list_channels()
        assert len(channels) == 2
        types = {ch["channel_type"] for ch in channels}
        assert types == {"telegram", "discord"}

    def test_list_channels_with_config(self):
        reg = ChannelRegistry()
        cfg = ChannelConfig(channel_type="slack", enabled=False)
        reg.register(StubAdapter(channel="slack", config=cfg))
        channels = reg.list_channels()
        assert channels[0]["enabled"] is False

    def test_replace_adapter_warning(self):
        reg = ChannelRegistry()
        reg.register(StubAdapter(channel="telegram"))
        new_adapter = StubAdapter(channel="telegram")
        reg.register(new_adapter)
        assert reg.get("telegram") is new_adapter
        assert reg.adapter_count == 1

    @pytest.mark.asyncio
    async def test_route_message_calls_router(self):
        reg = ChannelRegistry()
        handler = AsyncMock()
        reg.set_inbound_router(handler)

        msg = ChannelMessage(
            channel_type="telegram",
            direction=MessageDirection.INBOUND,
            sender="alice",
            recipient="bot",
            content="hello",
        )
        await reg.route_message(msg)
        handler.assert_awaited_once_with(msg)

    @pytest.mark.asyncio
    async def test_route_message_no_router(self):
        """Messages are dropped (not raised) when no router is set."""
        reg = ChannelRegistry()
        msg = ChannelMessage(
            channel_type="telegram",
            direction=MessageDirection.INBOUND,
            sender="alice",
            recipient="bot",
            content="hello",
        )
        # Should not raise
        await reg.route_message(msg)

    @pytest.mark.asyncio
    async def test_route_message_ignores_outbound(self):
        reg = ChannelRegistry()
        handler = AsyncMock()
        reg.set_inbound_router(handler)

        msg = ChannelMessage(
            channel_type="telegram",
            direction=MessageDirection.OUTBOUND,
            sender="bot",
            recipient="alice",
            content="reply",
        )
        await reg.route_message(msg)
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_route_message_exception_handled(self):
        """Router exceptions are caught, not propagated."""
        reg = ChannelRegistry()
        handler = AsyncMock(side_effect=RuntimeError("handler crash"))
        reg.set_inbound_router(handler)

        msg = ChannelMessage(
            channel_type="telegram",
            direction=MessageDirection.INBOUND,
            sender="alice",
            recipient="bot",
            content="boom",
        )
        # Should not raise
        await reg.route_message(msg)


# ============================================================================
# ChannelFeature Tests
# ============================================================================


class TestChannelFeature:
    @pytest_asyncio.fixture
    async def feature(self):
        """Create an initialized ChannelFeature with mock agent."""
        db = _make_db()
        agent = _make_agent(db=db)
        feat = ChannelFeature(agent)
        await feat.initialize()
        return feat

    @pytest_asyncio.fixture
    async def feature_no_db(self):
        """Create an initialized ChannelFeature without a database."""
        agent = _make_agent(db=None)
        agent.storage.db = None
        feat = ChannelFeature(agent)
        await feat.initialize()
        return feat

    # ----------------------------------------------------------------
    # Initialize / Shutdown
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, feature):
        """Verify initialize executes CREATE TABLE statements."""
        assert feature._db is not None
        # Multiple execute calls for CREATE TABLE and CREATE INDEX
        assert feature._db.execute.call_count >= 2

    @pytest.mark.asyncio
    async def test_initialize_no_db(self, feature_no_db):
        """Initialize should succeed even without a database."""
        assert feature_no_db._db is None

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_adapters(self, feature):
        adapter = StubAdapter(channel="telegram", connected=True)
        feature.registry.register(adapter)
        await feature.shutdown()
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_shutdown_handles_disconnect_error(self, feature):
        """Shutdown should not raise even if disconnect fails."""
        adapter = StubAdapter(channel="telegram", connected=True)
        adapter.disconnect = AsyncMock(side_effect=RuntimeError("oops"))
        feature.registry.register(adapter)
        # Should not raise
        await feature.shutdown()

    # ----------------------------------------------------------------
    # channels_list
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_channels_list_empty(self, feature):
        envelope = await feature.channels_list()
        assert envelope.data["count"] == 0
        assert envelope.data["channels"] == []

    @pytest.mark.asyncio
    async def test_channels_list_with_adapters(self, feature):
        feature.registry.register(StubAdapter(channel="telegram"))
        feature.registry.register(StubAdapter(channel="discord", connected=False))
        envelope = await feature.channels_list()
        assert envelope.data["count"] == 2
        types = {ch["channel_type"] for ch in envelope.data["channels"]}
        assert types == {"telegram", "discord"}

    # ----------------------------------------------------------------
    # channels_send
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_send_success(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature.registry.register(StubAdapter(channel="telegram"))
        envelope = await feature.channels_send(
            channel="telegram", to="user123", message="hello"
        )
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["receipt"]["status"] == "success"
        assert envelope.data["receipt"]["channel_type"] == "telegram"

    @pytest.mark.asyncio
    async def test_send_unknown_channel(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.channels_send(
            channel="unknown", to="user", message="hi"
        )
        assert envelope.status is ToolResultStatus.ERROR
        assert "No adapter registered" in envelope.error

    @pytest.mark.asyncio
    async def test_send_disconnected_channel(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature.registry.register(
            StubAdapter(channel="telegram", connected=False)
        )
        envelope = await feature.channels_send(
            channel="telegram", to="user", message="hi"
        )
        assert envelope.status is ToolResultStatus.ERROR
        assert "not connected" in envelope.error

    @pytest.mark.asyncio
    async def test_send_disabled_channel(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        cfg = ChannelConfig(channel_type="telegram", enabled=False)
        feature.registry.register(
            StubAdapter(channel="telegram", connected=True, config=cfg)
        )
        envelope = await feature.channels_send(
            channel="telegram", to="user", message="hi"
        )
        assert envelope.status is ToolResultStatus.ERROR
        assert "disabled" in envelope.error

    @pytest.mark.asyncio
    async def test_send_adapter_raises(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature.registry.register(FailingAdapter(channel="telegram"))
        envelope = await feature.channels_send(
            channel="telegram", to="user", message="hi"
        )
        assert envelope.status is ToolResultStatus.ERROR
        assert envelope.data["receipt"]["status"] == "failure"
        assert "channel down" in envelope.data["receipt"]["error"]

    @pytest.mark.asyncio
    async def test_send_logs_outbound(self, feature):
        feature.registry.register(StubAdapter(channel="telegram"))
        await feature.channels_send(
            channel="telegram", to="user123", message="hello"
        )
        # Verify an INSERT into channel_messages was executed
        insert_calls = [
            call
            for call in feature._db.execute.call_args_list
            if "INSERT INTO channel_messages" in str(call)
        ]
        assert len(insert_calls) >= 1

    # ----------------------------------------------------------------
    # channels_history
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_history_returns_messages(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature._db.fetchall = AsyncMock(
            return_value=[
                ("id-1", "telegram", "inbound", "alice", "bot",
                 "hello", "received", "2026-03-01T10:00:00"),
                ("id-2", "telegram", "outbound", "bot", "alice",
                 "hi alice", "success", "2026-03-01T10:01:00"),
            ]
        )
        envelope = await feature.channels_history(limit=10)
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["count"] == 2
        assert envelope.data["messages"][0]["id"] == "id-1"
        assert envelope.data["messages"][1]["direction"] == "outbound"

    @pytest.mark.asyncio
    async def test_history_with_channel_filter(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature._db.fetchall = AsyncMock(return_value=[])
        envelope = await feature.channels_history(limit=5, channel="discord")
        assert envelope.status is ToolResultStatus.OK
        # Verify the query included channel_type filter
        call_args = feature._db.fetchall.call_args
        sql = call_args[0][0]
        assert "channel_type = ?" in sql

    @pytest.mark.asyncio
    async def test_history_no_db(self, feature_no_db):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature_no_db.channels_history()
        assert envelope.status is ToolResultStatus.ERROR
        assert "Database not available" in envelope.error

    @pytest.mark.asyncio
    async def test_history_db_error(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature._db.fetchall = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )
        envelope = await feature.channels_history()
        assert envelope.status is ToolResultStatus.ERROR
        assert "connection lost" in envelope.error

    # ----------------------------------------------------------------
    # handle_inbound
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_handle_inbound_logs_and_routes(self, feature):
        router = AsyncMock()
        feature.registry.set_inbound_router(router)
        feature.registry.register(StubAdapter(channel="telegram"))

        msg = ChannelMessage(
            channel_type="telegram",
            direction=MessageDirection.INBOUND,
            sender="alice",
            recipient="bot",
            content="hi there",
        )
        await feature.handle_inbound(msg)

        # Message should be logged
        insert_calls = [
            call
            for call in feature._db.execute.call_args_list
            if "INSERT INTO channel_messages" in str(call)
        ]
        assert len(insert_calls) >= 1

        # Message should be routed
        router.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_inbound_enqueues_signal_when_dispatcher_available(self):
        db = _make_db()
        agent = _make_agent(db=db)
        agent.did = "did:test:channels"
        agent.dispatcher = MagicMock()
        agent.dispatcher.enqueue_signal = AsyncMock()

        feat = ChannelFeature(agent)
        await feat.initialize()
        router = AsyncMock()
        feat.registry.set_inbound_router(router)
        feat.registry.register(StubAdapter(channel="telegram"))

        msg = ChannelMessage(
            channel_type="telegram",
            direction=MessageDirection.INBOUND,
            sender="alice",
            recipient="bot",
            content="hi there",
        )
        await feat.handle_inbound(msg)

        agent.dispatcher.enqueue_signal.assert_awaited_once()
        signal = agent.dispatcher.enqueue_signal.await_args.args[0]
        assert signal.source == "channel.message"
        assert signal.payload["content"] == "hi there"
        router.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handle_inbound_blocked_sender(self, feature):
        router = AsyncMock()
        feature.registry.set_inbound_router(router)

        cfg = ChannelConfig(
            channel_type="telegram",
            allowed_senders=["bob"],
        )
        feature.registry.register(
            StubAdapter(channel="telegram", config=cfg)
        )

        msg = ChannelMessage(
            channel_type="telegram",
            direction=MessageDirection.INBOUND,
            sender="eve",
            recipient="bot",
            content="sneaky",
        )
        await feature.handle_inbound(msg)

        # Message should NOT be routed
        router.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handle_inbound_allowed_sender(self, feature):
        router = AsyncMock()
        feature.registry.set_inbound_router(router)

        cfg = ChannelConfig(
            channel_type="telegram",
            allowed_senders=["alice"],
        )
        feature.registry.register(
            StubAdapter(channel="telegram", config=cfg)
        )

        msg = ChannelMessage(
            channel_type="telegram",
            direction=MessageDirection.INBOUND,
            sender="alice",
            recipient="bot",
            content="allowed",
        )
        await feature.handle_inbound(msg)
        router.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_inbound_sets_agent_id(self, feature):
        feature.registry.register(StubAdapter(channel="telegram"))

        msg = ChannelMessage(
            channel_type="telegram",
            direction=MessageDirection.INBOUND,
            sender="alice",
            recipient="bot",
            content="hi",
            agent_id="",
        )
        await feature.handle_inbound(msg)
        assert msg.agent_id == "test-agent"

    # ----------------------------------------------------------------
    # Tool discovery
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_tool_discovery(self, feature):
        tools = feature.get_tools()
        tool_names = {t.name for t in tools}
        assert "channels_list" in tool_names
        assert "channels_send" in tool_names
        assert "channels_history" in tool_names

    @pytest.mark.asyncio
    async def test_tool_schemas_valid(self, feature):
        """Each tool should produce a valid OpenAI function calling schema."""
        tools = feature.get_tools()
        for t in tools:
            schema = t.schema
            oai = schema.to_openai_format()
            assert oai["type"] == "function"
            assert "name" in oai["function"]
            assert "parameters" in oai["function"]

    # ----------------------------------------------------------------
    # Logging without DB
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_log_message_no_db(self, feature_no_db):
        """_log_message should silently skip when DB is None."""
        msg = ChannelMessage(
            channel_type="telegram",
            direction=MessageDirection.OUTBOUND,
            sender="bot",
            recipient="alice",
            content="hi",
        )
        # Should not raise
        await feature_no_db._log_message(msg)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_send_with_no_config(self):
        """Adapter with no config should still work for send."""
        db = _make_db()
        agent = _make_agent(db=db)
        feat = ChannelFeature(agent)
        await feat.initialize()

        # Register adapter without config
        from kestrel_sdk.tools.result import ToolResultStatus
        feat.registry.register(StubAdapter(channel="telegram", config=None))
        envelope = await feat.channels_send(
            channel="telegram", to="user", message="hi"
        )
        assert envelope.status is ToolResultStatus.OK

    @pytest.mark.asyncio
    async def test_multiple_adapters_isolated(self):
        """Messages sent to one channel don't affect another."""
        db = _make_db()
        agent = _make_agent(db=db)
        feat = ChannelFeature(agent)
        await feat.initialize()

        tg = StubAdapter(channel="telegram")
        dc = StubAdapter(channel="discord")
        feat.registry.register(tg)
        feat.registry.register(dc)

        await feat.channels_send(channel="telegram", to="user", message="tg msg")
        await feat.channels_send(channel="discord", to="user", message="dc msg")

        assert len(tg._sent) == 1
        assert tg._sent[0]["content"] == "tg msg"
        assert len(dc._sent) == 1
        assert dc._sent[0]["content"] == "dc msg"

    @pytest.mark.asyncio
    async def test_message_direction_enum_values(self):
        assert MessageDirection.INBOUND.value == "inbound"
        assert MessageDirection.OUTBOUND.value == "outbound"

    @pytest.mark.asyncio
    async def test_delivery_status_enum_values(self):
        assert DeliveryStatus.SUCCESS.value == "success"
        assert DeliveryStatus.FAILURE.value == "failure"
        assert DeliveryStatus.PENDING.value == "pending"
