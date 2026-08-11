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
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

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
from kestrel_sovereign.signals.sources.channels import (
    DURABLE_COGNITION_CONSUMER_ID,
    DURABLE_COGNITION_MARKER,
    DURABLE_COGNITION_MARKER_VALUE,
    build_channel_message_registration,
    build_signal_for_channel_message,
)

from .models import (
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
    MessageDirection,
)
from .registry import ChannelRegistry

logger = logging.getLogger(__name__)

# Key version stamped into channel_messages metadata when a row is
# encrypted at rest. Mirrors
# ``async_conversation_store.CURRENT_KEY_VERSION`` (per-agent HKDF key)
# so channel content matches the conversation_history encryption
# guarantee (#2096 / F112).
CHANNEL_KEY_VERSION = 1


def canonical_telegram_user_id(value: object) -> str | None:
    """Return one immutable Telegram user ID, rejecting display identities.

    Telegram usernames are mutable presentation data.  The host must apply the
    same numeric-only authorization boundary as the isolated Telegram service,
    because a child notification is not trusted to preserve that distinction.
    """

    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdecimal()
        or value.startswith("0")
    ):
        return None
    try:
        return value if int(value) > 0 else None
    except ValueError:
        return None


def canonical_telegram_allowed_senders(value: object) -> list[str]:
    """Normalize host-side Telegram authorization to canonical numeric IDs."""

    if type(value) is not list:
        return []
    return [sender for item in value if (sender := canonical_telegram_user_id(item))]


class InboundAdmissionDisposition(str, Enum):
    """What the channel feature proved about one inbound message."""

    DURABLY_ADMITTED = "durably_admitted"
    RETRYABLE = "retryable"
    LEGACY_ROUTED = "legacy_routed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class InboundAdmission:
    """Explicit ingress result consumed by ACK-bearing isolated channels.

    Legacy routing intentionally has a different disposition: it may preserve
    compatibility for a non-dispatcher host, but it is not a durable receipt
    and therefore can never advance an external provider cursor.
    """

    disposition: InboundAdmissionDisposition

    @property
    def durably_admitted(self) -> bool:
        return self.disposition is InboundAdmissionDisposition.DURABLY_ADMITTED

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
        agent_did = getattr(self.agent, "did", None)
        self._authoritative_agent_id = (
            agent_did
            if isinstance(agent_did, str) and agent_did.strip()
            else self._agent_id
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
        self._durable_cognition_ready = False
        self._durable_cognition_registration_failed = False
        self._register_channel_signal_source()
        await self._register_durable_cognition_consumer()

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
        register = getattr(signal_registry, "register_with_policy", None)
        get_registration = getattr(signal_registry, "get", None)
        if not callable(register) or not callable(get_registration):
            self._durable_cognition_registration_failed = True
            logger.error(
                "Channel signal registry cannot verify the required channel.message "
                "contract; ACK-bearing ingress will remain retryable"
            )
            return
        from kestrel_sovereign.signals import (
            RegistrationOutcome,
            RegistrationPolicy,
            SourceRegistry,
        )

        # OPTIONAL policy (#2522): idempotent on re-init, but an existing
        # channel.message source with a DIFFERENT contract is reported rather
        # than silently accepted by a precheck-by-name skip. Never raises.
        required = build_channel_message_registration()
        try:
            outcome = register(required, RegistrationPolicy.OPTIONAL)
            # A host which merely claims registration succeeded is insufficient
            # for cursor-owned ingress. Verify the installed source itself so
            # an older/embedder registry cannot ACK an unknown contract.
            actual = get_registration(required.name)
            verified = (
                isinstance(outcome, RegistrationOutcome)
                and outcome.ok
                and SourceRegistry.contract_equivalent(actual, required)
            )
        except Exception:
            self._durable_cognition_registration_failed = True
            logger.exception(
                "Could not register and verify the channel.message signal source; "
                "ACK-bearing ingress will remain retryable"
            )
            return
        if not verified:
            # ``OPTIONAL`` keeps a pre-existing source alive on a mismatch or
            # validation failure.  That is tolerable for an ordinary optional
            # feature, but not for cursor-owning channel ingress: its durable
            # consumer would otherwise bind to a different (or absent)
            # ``channel.message`` contract and let a provider ACK work that
            # Core cannot safely process.  Preserve the provider cursor until
            # this feature can start against its intended contract.
            self._durable_cognition_registration_failed = True
            logger.error(
                "Channel signal source registration is not verifiably usable for "
                "durable cognition (state=%s): %s; ACK-bearing ingress will "
                "remain retryable",
                getattr(getattr(outcome, "state", None), "value", "unknown"),
                getattr(outcome, "detail", "required contract missing or mismatched"),
            )
            return
        # Own the source we newly registered so shutdown / boot rollback
        # unregisters it (#2522 P2). This is deliberately after verifying the
        # actual registration rather than trusting an embedder's return value.
        self._own_signal_sources(outcome)

    async def _register_durable_cognition_consumer(self) -> None:
        """Register the restart-safe delivery behind ACK-bearing channels.

        This registration is intentionally nonfatal for older/embedder
        dispatchers.  In that compatibility case ingress retains its cursor
        rather than claiming delivery was made durable.
        """
        if self._durable_cognition_registration_failed:
            return
        dispatcher = getattr(self.agent, "dispatcher", None)
        register = getattr(dispatcher, "register_durable_consumer", None)
        agent_did = getattr(self.agent, "did", None)
        if not callable(register) or not isinstance(agent_did, str) or not agent_did:
            return
        try:
            from kestrel_sovereign.signals.durable import DurableConsumerRegistration

            await register(
                DurableConsumerRegistration(
                    consumer_id=DURABLE_COGNITION_CONSUMER_ID,
                    source="channel.message",
                    agent_id=agent_did,
                    correlation_selector=(
                        f"payload.{DURABLE_COGNITION_MARKER}="
                        f"{DURABLE_COGNITION_MARKER_VALUE}"
                    ),
                    # Provider cursors are retryable until cognition is
                    # acknowledged; bounded retries would turn outages into
                    # irreversible loss.
                    max_attempts=0,
                )
            )
        except Exception:
            self._durable_cognition_registration_failed = True
            logger.exception(
                "Could not register durable channel cognition consumer; "
                "ACK-bearing ingress will remain retryable"
            )
            return
        self._durable_cognition_ready = True

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
        # Unregister channel.message (base #2522 P2 teardown).
        await super().shutdown()

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

    async def handle_inbound(self, message: ChannelMessage) -> InboundAdmission:
        """
        Process an inbound message from a channel adapter.

        Checks allowed-sender filtering, logs the message, and routes
        it through the registry.

        Args:
            message: The inbound ChannelMessage.
        """
        # The child transport is not an authorization boundary. Enforce the
        # host adapter's enabled state and sender policy before logging,
        # dispatching, or issuing a cursor-advancing receipt.
        cursor_owning = message.channel_type == "telegram"
        canonical_sender = (
            canonical_telegram_user_id(message.sender) if cursor_owning else None
        )
        if cursor_owning and canonical_sender is None:
            logger.info("Blocked noncanonical Telegram sender identity %r", message.sender)
            return InboundAdmission(InboundAdmissionDisposition.REJECTED)
        adapter = self.registry.get(message.channel_type)
        if adapter and adapter.config:
            config = adapter.config
            telegram_default_deny = (
                cursor_owning
                and not getattr(config, "allowed_senders", None)
            )
            if (
                not config.enabled
                or telegram_default_deny
                or not config.is_sender_allowed(message.sender)
            ):
                logger.info(
                    "Blocked inbound message from sender '%s' on channel '%s'",
                    message.sender,
                    message.channel_type,
                )
                return InboundAdmission(InboundAdmissionDisposition.REJECTED)

        # A child transport is never an authority for the message tenant. Bind
        # every inbound row and signal to this host feature's resolved agent,
        # including a non-empty child-supplied value.
        message.agent_id = self._authoritative_agent_id

        # Log inbound message
        await self._log_message(message, status="received")

        durable_admission = False
        durable_cognition_attempted = False
        dispatcher = getattr(self.agent, "dispatcher", None)
        if self._durable_cognition_registration_failed and cursor_owning:
            # A dispatcher that rejected the durable consumer contract is not
            # an older compatibility host. Falling back to ordinary enqueue
            # here would let a provider cursor advance on event persistence
            # alone, before cursor-owning cognition has a durable consumer.
            return InboundAdmission(InboundAdmissionDisposition.RETRYABLE)
        if dispatcher is not None:
            try:
                signal = build_signal_for_channel_message(
                    message,
                    target_agent=getattr(self.agent, "did", self._agent_id),
                )
                # Provider retries reuse the channel message ID.  ACK-bearing
                # transports use the durable cognition lease, so the callback
                # becomes ACKable only after cognition itself is durably
                # acknowledged rather than when its event envelope is written.
                enqueue_durable_cognition = getattr(
                    dispatcher, "enqueue_durable_cognition", None
                )
                if self._durable_cognition_ready and callable(enqueue_durable_cognition):
                    durable_cognition_attempted = True
                    handle = await enqueue_durable_cognition(
                        signal,
                        source_event_id=message.id,
                        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
                    )
                elif cursor_owning:
                    # Telegram polling owns a provider cursor. It has an
                    # explicit negotiated durable path and must never fall
                    # through to ordinary queue persistence or legacy routing.
                    logger.error(
                        "Telegram inbound lacks a verified durable cognition path; "
                        "retaining provider cursor for message id=%s",
                        message.id,
                    )
                    return InboundAdmission(InboundAdmissionDisposition.RETRYABLE)
                else:
                    handle = await dispatcher.enqueue_signal(
                        signal, source_event_id=message.id
                    )
                wait_for_durable_admission = getattr(
                    handle, "wait_for_durable_admission", None
                )
                if not callable(wait_for_durable_admission):
                    logger.error(
                        "Channel dispatcher returned no durable-admission receipt "
                        "for message id=%s",
                        message.id,
                    )
                else:
                    receipt = await wait_for_durable_admission()
                    durable_admission = (
                        getattr(receipt, "acknowledged", False) is True
                    )
            except Exception:
                logger.exception(
                    "Failed to durably admit channel.message signal for message id=%s",
                    message.id,
                )

        if durable_admission:
            return InboundAdmission(InboundAdmissionDisposition.DURABLY_ADMITTED)

        if durable_cognition_attempted or cursor_owning:
            # Do not send this message through the legacy in-memory router
            # after its durable cognition path was rate-limited or failed.
            # The external producer must retain its cursor and redeliver.
            return InboundAdmission(InboundAdmissionDisposition.RETRYABLE)

        # A legacy adapter/router path remains available for hosts without the
        # dispatcher contract, but is deliberately not reported as a durable
        # admission. ACK-bearing providers must retain their cursor and retry.
        if not durable_admission:
            await self.registry.route_message(message)
            return InboundAdmission(InboundAdmissionDisposition.LEGACY_ROUTED)
