"""
Bridge Protocol Models.

Defines the wire format for communication between external gateways
(KestrelClaw, Discord bots, etc.) and the Kestrel Sovereign brain.

All models use Pydantic for validation and serialization.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChannelType(str, Enum):
    """Supported gateway channel types."""
    BROWSER_EXTENSION = "browser_extension"
    DISCORD = "discord"
    SLACK = "slack"
    TELEGRAM = "telegram"
    API = "api"
    CUSTOM = "custom"


class BridgeRequest(BaseModel):
    """
    Inbound request from a gateway to the sovereign brain.

    The gateway sends this when a user interacts with the agent through
    an external channel (browser extension, Discord, etc.).
    """
    session_id: Optional[str] = Field(
        None,
        description="Gateway-side session ID. The bridge maps this to an agent session.",
    )
    message: str = Field(
        ...,
        description="The user's message to the agent.",
        min_length=1,
        max_length=32_000,
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Optional context from the gateway (e.g., current page URL, "
            "selected text, channel metadata)."
        ),
    )
    channel_type: ChannelType = Field(
        ChannelType.API,
        description="The type of channel the request originated from.",
    )
    sender_id: Optional[str] = Field(
        None,
        description="Identifier for the sender within the gateway (user ID, etc.).",
    )
    model_override: Optional[str] = Field(
        None,
        description="Optional model override (e.g., 'anthropic/claude-3-opus').",
    )
    did: Optional[str] = Field(
        None,
        description="Optional DID for identity verification of the caller.",
    )


class BridgeResponse(BaseModel):
    """
    Response from the sovereign brain back to the gateway.
    """
    message: str = Field(
        ...,
        description="The agent's text response.",
    )
    session_id: str = Field(
        ...,
        description="The bridge session ID (gateway can use this for continuity).",
    )
    tool_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Results from any tools the agent invoked during processing.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tokens used, model, duration, etc.).",
    )


class BridgeSession(BaseModel):
    """
    Maps a gateway session to an internal agent session.

    The bridge maintains this mapping so that a gateway can resume
    conversations using its own session identifiers.
    """
    id: str = Field(
        ...,
        description="Internal bridge session ID (UUID).",
    )
    agent_id: str = Field(
        ...,
        description="The agent's DID or identifier.",
    )
    gateway_session_id: Optional[str] = Field(
        None,
        description="The gateway's own session ID (opaque string).",
    )
    channel_type: ChannelType = Field(
        ChannelType.API,
        description="Channel type for this session.",
    )
    sender_id: Optional[str] = Field(
        None,
        description="Sender identifier within the gateway.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    last_activity_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def touch(self) -> None:
        """Update last_activity_at to now."""
        self.last_activity_at = datetime.now(timezone.utc)


class BridgeCapability(BaseModel):
    """A single capability exposed to the gateway for discovery."""
    name: str
    description: str
    category: str
    command_prefix: Optional[str] = None
    parameters: List[Dict[str, Any]] = Field(default_factory=list)


class BridgeCapabilitiesResponse(BaseModel):
    """Response for the capabilities discovery endpoint."""
    agent_id: str
    features: List[str] = Field(default_factory=list)
    capabilities: List[BridgeCapability] = Field(default_factory=list)
    channel_types: List[str] = Field(
        default_factory=lambda: [ct.value for ct in ChannelType],
    )
