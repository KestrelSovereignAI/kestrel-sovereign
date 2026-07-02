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
from kestrel_sovereign.features.storage_access import (
    hides_persisted_user_content,
    resolve_agent_privacy_config,
    resolve_feature_database,
)
from kestrel_sovereign.security.encryption import (
    DecryptionError,
    decrypt_string_fernet,
    encrypt_string_fernet,
    get_agent_fernet,
    get_fernet,
)
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from .models import (
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
    MessageDirection,
)
from .registry import ChannelRegistry
from kestrel_sovereign.signals.sources.channels import (
    build_channel_message_registration,
    build_signal_for_channel_message,
)

logger = logging.getLogger(__name__)

# Key version stamped into channel_messages metadata when a row is
# encrypted at rest. Mirrors
# ``async_conversation_store.CURRENT_KEY_VERSION`` (per-agent HKDF key)
# so channel content matches the conversation_history encryption
# guarantee (#2096 / F112).
CHANNEL_KEY_VERSION = 1

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
        self._db = resolve_feature_database(self.agent)

        # Resolve agent_id from storage hierarchy
        storage = self.agent.storage
        self._agent_id = (
            getattr(storage, "agent_id", "")
            or getattr(getattr(storage, "_storage", None), "agent_id", "")
        )

        # Encryption-at-rest keys for channel_messages content. Same key
        # hierarchy the conversation store uses (per-agent HKDF key with a
        # global-key fallback); both are ``None`` when KESTREL_DATA_KEY is
        # unset, in which case content is persisted in plaintext exactly as
        # conversation_history would be (#2096 / F112).
        self._agent_fernet = (
            get_agent_fernet(self._agent_id) if self._agent_id else None
        )
        self._global_fernet = get_fernet()

        # Create the channel registry
        self.registry = ChannelRegistry()
        self._register_channel_signal_source()

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

    def _register_channel_signal_source(self) -> None:
        signal_registry = getattr(self.agent, "signal_registry", None)
        if signal_registry is None:
            return
        try:
            if signal_registry.get("channel.message") is None:
                signal_registry.register(build_channel_message_registration())
        except Exception as exc:
            logger.warning("Could not register channel.message signal source: %s", exc)

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

    def _persistent_content_hidden(self) -> bool:
        """True when the active privacy mode forbids persisting user content.

        EPHEMERAL/ISOLATED promise "leave no trace" / session-only storage,
        so raw inbound/outbound channel text must never reach the
        persistent ``channel_messages`` table (#2096 / F112).
        """
        return hides_persisted_user_content(self.agent)

    def _requires_anonymization(self) -> bool:
        """True when the active privacy config mandates PII redaction."""
        config = resolve_agent_privacy_config(self.agent)
        if config is None:
            return False
        requires = getattr(config, "requires_anonymization", None)
        return bool(callable(requires) and requires())

    def _anonymize_channel_text(self, value: str) -> str:
        """Redact PII from channel content using the shared detector path."""
        from kestrel_sovereign.features.privacy.pii_detector import anonymize_text

        return anonymize_text(value)

    def _decrypt_content(self, content: str, meta: Optional[Dict]) -> str:
        """Decrypt persisted channel content when it was encrypted at rest.

        Rows written without a key (KESTREL_DATA_KEY unset) carry no ``enc``
        flag and pass straight through. Mirrors the conversation store's
        per-agent-first, global-key-fallback decryption.
        """
        if not meta or not meta.get("enc"):
            return content
        for fernet in (self._agent_fernet, self._global_fernet):
            if fernet is None:
                continue
            try:
                return decrypt_string_fernet(content, meta, fernet)
            except DecryptionError:
                continue
        logger.error(
            "Failed to decrypt channel_messages content for agent %s",
            self._agent_id,
        )
        return content

    async def _log_message(
        self,
        message: ChannelMessage,
        status: str = "success",
    ) -> None:
        """Persist a channel message to the database.

        Privacy gating (#2096 / F112) — the same contract enforced for
        conversation_history is applied here because this feature writes
        user content via the raw DB:

        - EPHEMERAL/ISOLATED: skip the persistent write entirely so nothing
          survives the session (``channels_history`` can't surface it).
        - ANONYMOUS: run the content through the PII anonymizer first.
        - Always: encrypt content at rest with the same key hierarchy the
          conversation store uses when a data key is configured.
        """
        if not self._db:
            return

        # EPHEMERAL/ISOLATED: never persist raw channel content.
        if self._persistent_content_hidden():
            logger.debug(
                "Skipping channel_messages write for agent %s: privacy mode "
                "hides persisted user content",
                self._agent_id,
            )
            return

        content = message.content
        if self._requires_anonymization():
            content = self._anonymize_channel_text(content)

        # Encrypt at rest so channel_messages matches conversation_history.
        fernet = self._agent_fernet or self._global_fernet
        stored_content, was_encrypted = encrypt_string_fernet(content, fernet)

        meta = dict(message.metadata) if message.metadata else {}
        if was_encrypted:
            meta["enc"] = True
            meta["key_version"] = CHANNEL_KEY_VERSION

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
                    stored_content,
                    status,
                    json.dumps(meta) if meta else None,
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
    async def channels_list(self) -> ToolResult:
        """
        Show connected channels and their status.

        Returns a list of registered channels with connection state and
        whether each channel is enabled.
        """
        channels = self.registry.list_channels()
        if not channels:
            return ToolResult.ok(
                "No messaging channels registered.",
                data={"channels": [], "count": 0},
            )
        names = ", ".join(ch["channel_type"] for ch in channels)
        return ToolResult.ok(
            f"{len(channels)} messaging channel(s) registered: {names}.",
            data={"channels": channels, "count": len(channels)},
        )

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
    ) -> ToolResult:
        """
        Send a message through a named channel.

        Args:
            channel: Channel type to send through (e.g. "telegram")
            to: Recipient identifier (channel-specific)
            message: Text content to send

        Returns:
            ToolResult.ok on a SUCCESS receipt; PARTIAL on a PENDING
            receipt (the channel queued the message but has not yet
            confirmed delivery, so the LLM should NOT promise the
            sovereign that the message was received); ERROR for
            adapter-not-found, disconnected, disabled, or send-failure.
        """
        adapter = self.registry.get(channel)
        if adapter is None:
            available = [ch["channel_type"] for ch in self.registry.list_channels()]
            return ToolResult.failed(
                error=(
                    f"No adapter registered for channel '{channel}' "
                    f"(available: {', '.join(available) if available else 'none'})"
                ),
                data={"available_channels": available},
            )

        if not adapter.is_connected:
            return ToolResult.failed(
                error=f"Channel '{channel}' is registered but not connected"
            )

        # Check allowed-sender filtering on the adapter config
        config = adapter.config
        if config and not config.enabled:
            return ToolResult.failed(error=f"Channel '{channel}' is disabled")

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

        receipt_dict = receipt.to_dict()
        if receipt.status == DeliveryStatus.SUCCESS:
            return ToolResult.ok(
                f"Message sent via {channel} to {to} (id={receipt.message_id}).",
                data={"receipt": receipt_dict},
            )
        if receipt.status == DeliveryStatus.PENDING:
            # Honesty: PENDING means the channel accepted the request
            # but has not yet confirmed delivery. The LLM should not
            # tell the sovereign "your message was sent" — it was
            # queued, and may still fail.
            return ToolResult.partial(
                f"Message queued via {channel} to {to} (id={receipt.message_id}).",
                (
                    f"channel '{channel}' returned PENDING — delivery is not "
                    "yet confirmed; check !channels history for the final "
                    "status."
                ),
                data={"receipt": receipt_dict},
            )
        # FAILURE
        err = receipt.error or f"send failed via {channel}"
        return ToolResult.failed(
            error=f"Failed to send via {channel}: {err}",
            data={"receipt": receipt_dict},
        )

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
    ) -> ToolResult:
        """
        Get recent channel message history.

        Args:
            limit: Maximum number of messages to return (default 20)
            channel: Optional channel type filter (empty = all channels)
        """
        # Privacy gating (#2096 / F112): EPHEMERAL/ISOLATED promise the
        # session leaves no persisted trace. The write path already skips
        # persisting channel content in these modes, but a row could still
        # linger from a prior NORMAL stint or a privacy-layer leak — so the
        # read path must refuse to surface persisted content too, mirroring
        # the conversation store's ephemeral read guards. Return an empty
        # success without touching the DB.
        if self._persistent_content_hidden():
            logger.debug(
                "channels_history suppressed for agent %s: privacy mode hides "
                "persisted user content",
                self._agent_id,
            )
            scope = f" for channel '{channel}'" if channel else ""
            return ToolResult.ok(
                f"Returned 0 channel message(s){scope}.",
                data={"messages": [], "count": 0, "channel": channel or None},
            )

        if not self._db:
            return ToolResult.failed(error="Database not available")

        try:
            if channel:
                rows = await self._db.fetchall(
                    """SELECT id, channel_type, direction, sender, recipient,
                              content, status, created_at, metadata
                       FROM channel_messages
                       WHERE agent_id = ? AND channel_type = ?
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (self._agent_id, channel, limit),
                )
            else:
                rows = await self._db.fetchall(
                    """SELECT id, channel_type, direction, sender, recipient,
                              content, status, created_at, metadata
                       FROM channel_messages
                       WHERE agent_id = ?
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (self._agent_id, limit),
                )

            messages = []
            for row in rows:
                # metadata (row[8]) carries the at-rest ``enc`` flag; it's
                # only used to decrypt content and is not surfaced to the
                # caller. Defensive ``len(row) > 8`` tolerates legacy 8-col
                # rows.
                raw_meta = row[8] if len(row) > 8 else None
                meta = json.loads(raw_meta) if raw_meta else None
                messages.append({
                    "id": row[0],
                    "channel_type": row[1],
                    "direction": row[2],
                    "sender": row[3],
                    "recipient": row[4],
                    "content": self._decrypt_content(row[5], meta),
                    "status": row[6],
                    "created_at": row[7],
                })

            scope = f" for channel '{channel}'" if channel else ""
            return ToolResult.ok(
                f"Returned {len(messages)} channel message(s){scope}.",
                data={
                    "messages": messages,
                    "count": len(messages),
                    "channel": channel or None,
                },
            )
        except Exception as exc:
            logger.error("channels_history failed: %s", exc)
            return ToolResult.failed(error=str(exc))

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

        dispatched_signal = False
        dispatcher = getattr(self.agent, "dispatcher", None)
        if dispatcher is not None:
            try:
                signal = build_signal_for_channel_message(
                    message,
                    target_agent=getattr(self.agent, "did", self._agent_id),
                )
                await dispatcher.enqueue_signal(signal)
                dispatched_signal = True
            except Exception:
                logger.exception(
                    "Failed to enqueue channel.message signal for message id=%s",
                    message.id,
                )

        if not dispatched_signal:
            # Legacy adapter/router path remains as fallback when the
            # dispatcher is unavailable or rejected the signal.
            await self.registry.route_message(message)
