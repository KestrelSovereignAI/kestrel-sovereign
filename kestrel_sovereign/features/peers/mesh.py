"""
Agent Mesh Protocol — structured message types for inter-agent communication.

Defines the message schema used by Falconer agents (Claws, Talon, Eye, Flight)
to communicate through the rookery. Messages flow through PeersFeature's
ask_agent() transport but carry structured payloads.

Message types:
    assign          — Claws assigns an issue to Talon for implementation
    review_needed   — Talon requests Eye review of a PR/screenshot
    complete        — Agent reports a task finished
    reject          — Agent declines assignment (capacity, scope, blocked)
    status_update   — Progress update on an in-flight task

Reference: signal-and-ship #3 (Agent Mesh Protocol)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class MeshMessageType(str, Enum):
    """Types of inter-agent mesh messages."""
    ASSIGN = "assign"
    REVIEW_NEEDED = "review_needed"
    COMPLETE = "complete"
    REJECT = "reject"
    STATUS_UPDATE = "status_update"


class MeshPriority(str, Enum):
    """Priority levels for mesh assignments."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class MeshMessage:
    """
    Structured message exchanged between Falconer agents.

    Attributes:
        id: Unique message identifier.
        type: The message type (assign, complete, etc.).
        sender: Name of the sending agent (e.g. "claws").
        recipient: Name of the target agent (e.g. "talon").
        priority: Priority level for the task.
        payload: Type-specific data (issue number, PR URL, etc.).
        timestamp: When the message was created (UTC).
        correlation_id: Links related messages (e.g. assign → complete).
        repo: GitHub repo in "owner/name" format (if applicable).
    """
    type: MeshMessageType
    sender: str
    recipient: str
    payload: Dict[str, Any]
    priority: MeshPriority = MeshPriority.NORMAL
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: Optional[str] = None
    repo: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        d = asdict(self)
        d["type"] = self.type.value
        d["priority"] = self.priority.value
        return d

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MeshMessage:
        """Deserialize from a dict."""
        data = dict(data)  # shallow copy
        data["type"] = MeshMessageType(data["type"])
        data["priority"] = MeshPriority(data.get("priority", "normal"))
        return cls(**data)

    @classmethod
    def from_json(cls, raw: str) -> MeshMessage:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(raw))


def make_assign_message(
    sender: str,
    recipient: str,
    repo: str,
    issue_number: int,
    issue_title: str,
    priority: MeshPriority = MeshPriority.NORMAL,
    context: str = "",
) -> MeshMessage:
    """Create an assignment message (Claws → Talon)."""
    return MeshMessage(
        type=MeshMessageType.ASSIGN,
        sender=sender,
        recipient=recipient,
        repo=repo,
        priority=priority,
        payload={
            "issue_number": issue_number,
            "issue_title": issue_title,
            "context": context,
        },
    )


def make_complete_message(
    sender: str,
    recipient: str,
    correlation_id: str,
    repo: str,
    issue_number: int,
    pr_number: Optional[int] = None,
    summary: str = "",
) -> MeshMessage:
    """Create a completion message (Talon → Claws)."""
    return MeshMessage(
        type=MeshMessageType.COMPLETE,
        sender=sender,
        recipient=recipient,
        repo=repo,
        correlation_id=correlation_id,
        payload={
            "issue_number": issue_number,
            "pr_number": pr_number,
            "summary": summary,
        },
    )


def make_review_message(
    sender: str,
    recipient: str,
    repo: str,
    pr_number: int,
    pr_title: str,
    correlation_id: Optional[str] = None,
) -> MeshMessage:
    """Create a review-needed message (Talon → Eye)."""
    return MeshMessage(
        type=MeshMessageType.REVIEW_NEEDED,
        sender=sender,
        recipient=recipient,
        repo=repo,
        correlation_id=correlation_id,
        payload={
            "pr_number": pr_number,
            "pr_title": pr_title,
        },
    )


def make_reject_message(
    sender: str,
    recipient: str,
    correlation_id: str,
    reason: str,
) -> MeshMessage:
    """Create a rejection message (agent declines assignment)."""
    return MeshMessage(
        type=MeshMessageType.REJECT,
        sender=sender,
        recipient=recipient,
        correlation_id=correlation_id,
        payload={"reason": reason},
    )
