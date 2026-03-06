"""
Data models for the Channel Adapter system.

Defines the core message, receipt, configuration, and callback types
used across all channel implementations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional
import uuid


class MessageDirection(Enum):
    """Direction of a channel message."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class DeliveryStatus(Enum):
    """Status of a message delivery attempt."""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"


@dataclass
class ChannelMessage:
    """
    Represents an inbound or outbound message on a channel.

    Attributes:
        id: Unique message identifier.
        channel_type: Channel this message belongs to (e.g. "telegram").
        direction: Whether the message is inbound or outbound.
        sender: Identifier of the message sender.
        recipient: Identifier of the message recipient.
        content: Text content of the message.
        timestamp: When the message was created.
        metadata: Optional extra data (e.g. attachments, reply-to).
        agent_id: The agent that owns this message.
    """
    channel_type: str
    direction: MessageDirection
    sender: str
    recipient: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "id": self.id,
            "channel_type": self.channel_type,
            "direction": self.direction.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "agent_id": self.agent_id,
        }


@dataclass
class DeliveryReceipt:
    """
    Result of a send_message operation.

    Attributes:
        message_id: Identifier assigned by the channel or internally.
        status: Whether delivery succeeded, failed, or is pending.
        channel_type: Which channel the message was sent through.
        error: Error description if status is FAILURE.
        timestamp: When the delivery attempt occurred.
    """
    message_id: str
    status: DeliveryStatus
    channel_type: str
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "message_id": self.message_id,
            "status": self.status.value,
            "channel_type": self.channel_type,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ChannelConfig:
    """
    Per-channel configuration for an agent.

    Attributes:
        channel_type: Which channel this config applies to.
        agent_id: The agent this config belongs to.
        enabled: Whether this channel is currently active.
        api_key: Optional API key / bot token for the channel.
        allowed_senders: If non-empty, only these senders are permitted.
            An empty list means all senders are allowed.
        extra: Arbitrary channel-specific settings.
    """
    channel_type: str
    agent_id: str = ""
    enabled: bool = True
    api_key: Optional[str] = None
    allowed_senders: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def is_sender_allowed(self, sender: str) -> bool:
        """Check whether a sender is permitted by this config.

        Returns True if:
        - allowed_senders is empty (no filtering), OR
        - sender is in the allowed_senders list.
        """
        if not self.allowed_senders:
            return True
        return sender in self.allowed_senders

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary (excluding api_key for safety)."""
        return {
            "channel_type": self.channel_type,
            "agent_id": self.agent_id,
            "enabled": self.enabled,
            "has_api_key": self.api_key is not None,
            "allowed_senders": self.allowed_senders,
            "extra": self.extra,
        }


# Type alias for the async callback invoked when a message arrives.
MessageCallback = Callable[[ChannelMessage], Awaitable[None]]
