"""
Channels Feature -- pluggable messaging channel management.

Provides tools for listing connected channels, sending messages through
a named channel, and viewing recent inbound/outbound message history.
No concrete channel implementations are included; those are added by
registering ChannelAdapter subclasses at runtime.

DB tables (created on initialize):
  channel_messages  -- log of all inbound/outbound messages
  channel_config    -- per-agent per-channel configuration
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

from .models import (
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
    MessageDirection,
)
from .registry import ChannelRegistry

logger = logging.getLogger(__name__)

# SQL for the two tables managed by this feature.
CHANNEL_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS channel_messages (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    channel_type TEXT NOT NULL,
    direction TEXT NOT NULL,
    sender TEXT NOT NULL,
    recipient TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'success',
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_channel_messages_agent
    ON channel_messages(agent_id);
CREATE INDEX IF NOT EXISTS idx_channel_messages_channel
    ON channel_messages(agent_id, channel_type);
CREATE INDEX IF NOT EXISTS idx_channel_messages_created
    ON channel_messages(agent_id, created_at);

CREATE TABLE IF NOT EXISTS channel_config (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    channel_type TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_channel_config_agent
    ON channel_config(agent_id);
CREATE INDEX IF NOT EXISTS idx_channel_config_unique
    ON channel_config(agent_id, channel_type);
"""


class ChannelFeature(Feature):
    """
    Messaging channel management for the Kestrel agent.

    Exposes three tools:
    - ``!channels list``    -- show connected channels and status
    - ``!channels send``    -- send a message through a named channel
    - ``!channels history`` -- recent inbound/outbound message log
    """

    @property
    def tool_description(self) -> str:
        return (
            "Manage external messaging channels (Telegram, Discord, Slack, etc.) -- "
            "list connected channels, send messages, and view message history"
        )

    async def initialize(self):
        """Set up DB tables and the channel registry."""
        # Database handle (may be None in tests or ephemeral mode)
        self._db = getattr(self.agent.storage, "db", None)
        if self._db is None:
            raw = getattr(self.agent, "_raw_storage", None)
            if raw:
                self._db = getattr(raw, "db", None)

        # Resolve agent_id from storage hierarchy
        storage = self.agent.storage
        self._agent_id = (
            getattr(storage, "agent_id", "")
            or getattr(getattr(storage, "_storage", None), "agent_id", "")
        )

        # Create the channel registry
        self.registry = ChannelRegistry()

        # Create tables if DB is available
        if self._db:
            try:
                for statement in CHANNEL_TABLES_SQL.strip().split(";"):
                    statement = statement.strip()
                    if statement:
                        await self._db.execute(statement)
            except Exception as exc:
                logger.warning("Could not create channel tables: %s", exc)

        logger.info(
            "ChannelFeature initialized for agent: %s",
            (self._agent_id[:30] + "...") if len(self._agent_id) > 30 else self._agent_id,
        )

    async def shutdown(self):
        """Disconnect all registered adapters."""
        for info in self.registry.list_channels():
            adapter = self.registry.get(info["channel_type"])
            if adapter and adapter.is_connected:
                try:
                    await adapter.disconnect()
                except Exception as exc:
                    logger.warning(
                        "Error disconnecting channel %s: %s",
                        info["channel_type"],
                        exc,
                    )

    # ------------------------------------------------------------------
    # Message logging helpers
    # ------------------------------------------------------------------

    async def _log_message(
        self,
        message: ChannelMessage,
        status: str = "success",
    ) -> None:
        """Persist a channel message to the database."""
        if not self._db:
            return
        try:
            await self._db.execute(
                """INSERT INTO channel_messages
                   (id, agent_id, channel_type, direction, sender,
                    recipient, content, status, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    message.id,
                    message.agent_id or self._agent_id,
                    message.channel_type,
                    message.direction.value,
                    message.sender,
                    message.recipient,
                    message.content,
                    status,
                    json.dumps(message.metadata) if message.metadata else None,
                    message.timestamp.isoformat(),
                ),
            )
        except Exception as exc:
            logger.error("Failed to log channel message: %s", exc)

    async def _log_outbound(
        self,
        channel_type: str,
        to: str,
        content: str,
        receipt: DeliveryReceipt,
    ) -> None:
        """Log an outbound message with its delivery receipt."""
        msg = ChannelMessage(
            id=receipt.message_id,
            channel_type=channel_type,
            direction=MessageDirection.OUTBOUND,
            sender=self._agent_id,
            recipient=to,
            content=content,
            agent_id=self._agent_id,
        )
        await self._log_message(msg, status=receipt.status.value)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool(
        name="channels_list",
        description="List all connected messaging channels and their current status.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!channels list",
    )
    async def channels_list(self) -> Dict[str, Any]:
        """
        Show connected channels and their status.

        Returns a list of registered channels with connection state and
        whether each channel is enabled.
        """
        channels = self.registry.list_channels()
        return {
            "channels": channels,
            "count": len(channels),
        }

    @tool(
        name="channels_send",
        description="Send a message to a recipient via a specific messaging channel.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!channels send",
    )
    async def channels_send(
        self,
        channel: str,
        to: str,
        message: str,
    ) -> Dict[str, Any]:
        """
        Send a message through a named channel.

        Args:
            channel: Channel type to send through (e.g. "telegram")
            to: Recipient identifier (channel-specific)
            message: Text content to send
        """
        adapter = self.registry.get(channel)
        if adapter is None:
            return {
                "success": False,
                "error": f"No adapter registered for channel '{channel}'",
                "available_channels": [
                    ch["channel_type"]
                    for ch in self.registry.list_channels()
                ],
            }

        if not adapter.is_connected:
            return {
                "success": False,
                "error": f"Channel '{channel}' is registered but not connected",
            }

        # Check allowed-sender filtering on the adapter config
        config = adapter.config
        if config and not config.enabled:
            return {
                "success": False,
                "error": f"Channel '{channel}' is disabled",
            }

        try:
            receipt = await adapter.send_message(to=to, content=message)
        except Exception as exc:
            logger.error("Failed to send via %s: %s", channel, exc)
            receipt = DeliveryReceipt(
                message_id=str(uuid.uuid4()),
                status=DeliveryStatus.FAILURE,
                channel_type=channel,
                error=str(exc),
            )

        # Log the outbound message
        await self._log_outbound(channel, to, message, receipt)

        return {
            "success": receipt.status == DeliveryStatus.SUCCESS,
            "receipt": receipt.to_dict(),
        }

    @tool(
        name="channels_history",
        description="View recent inbound and outbound channel messages.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!channels history",
    )
    async def channels_history(
        self,
        limit: int = 20,
        channel: str = "",
    ) -> Dict[str, Any]:
        """
        Get recent channel message history.

        Args:
            limit: Maximum number of messages to return (default 20)
            channel: Optional channel type filter (empty = all channels)
        """
        if not self._db:
            return {
                "success": False,
                "error": "Database not available",
            }

        try:
            if channel:
                rows = await self._db.fetchall(
                    """SELECT id, channel_type, direction, sender, recipient,
                              content, status, created_at
                       FROM channel_messages
                       WHERE agent_id = ? AND channel_type = ?
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (self._agent_id, channel, limit),
                )
            else:
                rows = await self._db.fetchall(
                    """SELECT id, channel_type, direction, sender, recipient,
                              content, status, created_at
                       FROM channel_messages
                       WHERE agent_id = ?
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (self._agent_id, limit),
                )

            messages = []
            for row in rows:
                messages.append({
                    "id": row[0],
                    "channel_type": row[1],
                    "direction": row[2],
                    "sender": row[3],
                    "recipient": row[4],
                    "content": row[5],
                    "status": row[6],
                    "created_at": row[7],
                })

            return {
                "success": True,
                "messages": messages,
                "count": len(messages),
            }
        except Exception as exc:
            logger.error("channels_history failed: %s", exc)
            return {
                "success": False,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Inbound message handling (called by adapters)
    # ------------------------------------------------------------------

    async def handle_inbound(self, message: ChannelMessage) -> None:
        """
        Process an inbound message from a channel adapter.

        Checks allowed-sender filtering, logs the message, and routes
        it through the registry.

        Args:
            message: The inbound ChannelMessage.
        """
        # Check sender filtering
        adapter = self.registry.get(message.channel_type)
        if adapter and adapter.config:
            if not adapter.config.is_sender_allowed(message.sender):
                logger.info(
                    "Blocked message from disallowed sender '%s' on channel '%s'",
                    message.sender,
                    message.channel_type,
                )
                return

        # Set agent_id if not already set
        if not message.agent_id:
            message.agent_id = self._agent_id

        # Log inbound message
        await self._log_message(message, status="received")

        # Route through registry
        await self.registry.route_message(message)
