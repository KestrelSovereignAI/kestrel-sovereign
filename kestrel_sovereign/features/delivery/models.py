"""
Data models for the Delivery Queue feature.

Defines the queue entry, status enum, and delivery result types used
by the queue, feature, and background worker.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class DeliveryStatus(Enum):
    """Lifecycle states for a delivery queue entry."""
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class QueueEntry:
    """Represents a single message in the delivery queue."""
    id: str
    agent_id: str
    channel_type: str
    recipient: str
    content_json: str  # JSON-encoded message content
    status: DeliveryStatus
    attempts: int
    max_retries: int
    next_retry_at: Optional[str]  # ISO timestamp or None
    last_error: Optional[str]
    created_at: str  # ISO timestamp
    delivered_at: Optional[str]  # ISO timestamp or None
    content_hash: Optional[str] = None

    @property
    def content(self) -> Dict[str, Any]:
        """Parse content_json into a dict."""
        if not self.content_json:
            return {}
        try:
            return json.loads(self.content_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for API responses."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "channel_type": self.channel_type,
            "recipient": self.recipient,
            "content": self.content,
            "status": self.status.value,
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "next_retry_at": self.next_retry_at,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
        }

    @staticmethod
    def compute_content_hash(recipient: str, content_json: str) -> str:
        """Compute a deduplication hash from recipient + content."""
        raw = f"{recipient}:{content_json}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class DeliveryResult:
    """Result of attempting to deliver a single message."""
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"success": self.success}
        if self.error:
            d["error"] = self.error
        return d
