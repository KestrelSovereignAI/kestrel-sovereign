"""
Agent Mesh Protocol — structured message types for inter-agent communication.

Defines the message schema used by Falconer agents (Claws, Talon, Eye, Flight)
to communicate through the multi_agent. Messages flow through PeersFeature's
ask_agent() transport but carry structured payloads.

Message types:
    assign          — Claws assigns an issue to Talon for implementation
    review_needed   — Talon requests Eye review of a PR/screenshot
    red_action      — Claws dispatches adversarial code review before merge
    complete        — Agent reports a task finished
    reject          — Agent declines assignment (capacity, scope, blocked)
    status_update   — Progress update on an in-flight task

Reference: signal-and-ship #3 (Agent Mesh Protocol), #447 (Red-Action)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class MeshMessageType(str, Enum):
    """Types of inter-agent mesh messages."""
    ASSIGN = "assign"
    REVIEW_NEEDED = "review_needed"
    RED_ACTION = "red_action"
    COMPLETE = "complete"
    REJECT = "reject"
    STATUS_UPDATE = "status_update"


class MeshPriority(str, Enum):
    """Priority levels for mesh assignments."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class RedActionRisk(str, Enum):
    """Risk level for red-action review dispatch."""
    AUTO = "auto"      # Claws decides based on critical path tags
    MANUAL = "manual"  # Falconer explicitly requests red-action
    SKIP = "skip"      # Low-risk, no red-action needed


# Critical paths that auto-trigger red-action review.
# PRs touching files in these directories always get red-action.
CRITICAL_PATH_PATTERNS: List[str] = [
    "kestrel_sovereign/agent/context",       # Context assembly — core pipeline
    "kestrel_sovereign/hooks/",              # Hook system — security enforcement
    "kestrel_sovereign/agent/constitution",  # Constitutional governance
    "kestrel_sovereign/llm/",               # LLM routing — model selection
    "kestrel_sovereign/features/privacy",    # Privacy modes
    "kestrel_sovereign/storage/encryption",  # Key management / encryption
    "kestrel_sovereign/auth",               # Authentication
    "endpoints/auth",                       # Auth endpoints
    "kestrel_sovereign/features/wallet",     # Wallet / sovereignty
]


def is_critical_path(changed_files: List[str]) -> bool:
    """Check if any changed files touch a critical path that requires red-action."""
    for filepath in changed_files:
        normalised = filepath.replace("\\", "/")
        for pattern in CRITICAL_PATH_PATTERNS:
            if pattern in normalised:
                return True
    return False


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


def make_red_action_message(
    sender: str,
    recipient: str,
    repo: str,
    pr_number: int,
    pr_title: str,
    changed_files: List[str],
    risk: RedActionRisk = RedActionRisk.AUTO,
    issue_number: Optional[int] = None,
    correlation_id: Optional[str] = None,
    context: str = "",
) -> MeshMessage:
    """
    Create a red-action review request (Claws → Red-Action reviewer).

    Dispatched after Eye (visual QA) but before merge. The reviewer
    performs adversarial code review on the actual code paths.

    Args:
        sender: Requesting agent (usually "claws").
        recipient: Agent performing review (could be self or dedicated reviewer).
        repo: GitHub repo in "owner/name" format.
        pr_number: Pull request number to review.
        pr_title: PR title for context.
        changed_files: List of files changed in the PR.
        risk: How the review was triggered (auto, manual, skip).
        issue_number: Related issue number (if any).
        correlation_id: Links to the original assign message chain.
        context: Additional context for the reviewer.
    """
    critical = is_critical_path(changed_files)
    priority = MeshPriority.HIGH if critical else MeshPriority.NORMAL

    return MeshMessage(
        type=MeshMessageType.RED_ACTION,
        sender=sender,
        recipient=recipient,
        repo=repo,
        priority=priority,
        correlation_id=correlation_id,
        payload={
            "pr_number": pr_number,
            "pr_title": pr_title,
            "issue_number": issue_number,
            "changed_files": changed_files,
            "critical_path": critical,
            "risk_level": risk.value,
            "context": context,
        },
    )
