"""
Data models for the generic webhook receiver.

Defines configuration, event, and authentication models used
throughout the webhook feature.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class WebhookAuthType(Enum):
    """Supported webhook authentication methods."""

    BEARER_TOKEN = "bearer_token"
    HMAC_SHA256 = "hmac_sha256"
    IP_ALLOWLIST = "ip_allowlist"
    NONE = "none"


@dataclass
class WebhookConfig:
    """Configuration for a registered webhook endpoint.

    Attributes:
        name: Unique identifier for the webhook (used in URL path).
        auth_type: Authentication method required for this webhook.
        auth_config: Authentication-specific configuration (e.g. token, secret, IPs).
        event_type: Optional event type label for categorisation.
        enabled: Whether the webhook is currently accepting requests.
        rate_limit: Maximum requests per minute (0 = no limit).
        id: Database primary key (auto-generated UUID).
        agent_id: Owning agent identifier.
        created_at: ISO-8601 timestamp of creation.
    """

    name: str
    auth_type: WebhookAuthType
    auth_config: Dict[str, Any] = field(default_factory=dict)
    event_type: str = ""
    enabled: bool = True
    rate_limit: int = 60
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary (excludes auth secrets)."""
        return {
            "id": self.id,
            "name": self.name,
            "auth_type": self.auth_type.value,
            "event_type": self.event_type,
            "enabled": self.enabled,
            "rate_limit": self.rate_limit,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
        }


@dataclass
class WebhookEvent:
    """A single received webhook event (used for logging/audit).

    Attributes:
        id: Unique event identifier.
        webhook_name: Name of the webhook that received the event.
        payload: Raw payload body (string or dict).
        headers: Subset of request headers relevant to the webhook.
        source_ip: IP address of the sender.
        authenticated: Whether the request passed authentication.
        status_code: HTTP status code returned to the caller.
        payload_hash: SHA-256 hash of the payload for integrity.
        timestamp: ISO-8601 timestamp of receipt.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    webhook_name: str = ""
    payload: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    source_ip: str = ""
    authenticated: bool = False
    status_code: int = 200
    payload_hash: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "webhook_name": self.webhook_name,
            "source_ip": self.source_ip,
            "authenticated": self.authenticated,
            "status_code": self.status_code,
            "payload_hash": self.payload_hash,
            "timestamp": self.timestamp,
        }
