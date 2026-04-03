"""
Data models for the Reflection feature.

Defines the core types for agent self-reflection:
- Layered reflection: Arms → Memory → Mind → Action
- HealthCheck: Result of a single check
- LayerResult: Result of running all checks in a layer
- ActionItem: A prioritized action from reflection
- Insight: A single observation from interaction analysis
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import json


# =============================================================================
# Layer 4 Reflection Enums
# =============================================================================

class ReflectionLayer(Enum):
    """The four layers of reflection."""
    ARMS = "arms"       # Physical/functional checks
    MEMORY = "memory"   # Knowledge/context checks
    MIND = "mind"       # Cognitive checks
    ACTION = "action"   # Prioritized fixes


class CheckStatus(Enum):
    """Status of a health check."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class Severity(Enum):
    """Severity of issues found."""
    CRITICAL = "critical"  # P0 - System unusable
    HIGH = "high"          # P1 - Major capability broken
    MEDIUM = "medium"      # P2 - Degraded performance
    LOW = "low"            # P3 - Minor improvement


# =============================================================================
# Layer 4 Reflection Models
# =============================================================================

@dataclass
class HealthCheck:
    """Result of a single health check."""
    id: str
    layer: ReflectionLayer
    name: str
    description: str
    status: CheckStatus
    severity: Optional[Severity] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer.value,
            "name": self.name,
            "status": self.status.value,
            "severity": self.severity.value if self.severity else None,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "file_path": self.file_path,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class LayerResult:
    """Result of running all checks in a layer."""
    layer: ReflectionLayer
    checks: List[HealthCheck] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def has_critical(self) -> bool:
        return any(
            c.severity == Severity.CRITICAL
            for c in self.checks
            if c.status == CheckStatus.FAIL
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer.value,
            "passed": self.passed,
            "failed": self.failed,
            "has_critical": self.has_critical,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class ActionItem:
    """A prioritized action from reflection."""
    id: str
    priority: Severity
    title: str
    description: str
    source_check: Optional[str] = None
    source_insight: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    fix_description: str = ""
    effort_estimate: str = ""
    actionable: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "fix_description": self.fix_description,
            "effort_estimate": self.effort_estimate,
        }


@dataclass
class ReflectionResult:
    """Complete result of layered reflection."""
    id: str
    trigger: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    arms: Optional[LayerResult] = None
    memory: Optional[LayerResult] = None
    mind: Optional[LayerResult] = None
    actions: List[ActionItem] = field(default_factory=list)
    layers_completed: int = 0
    stopped_at_layer: Optional[ReflectionLayer] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "trigger": self.trigger,
            "layers_completed": self.layers_completed,
            "stopped_at_layer": self.stopped_at_layer.value if self.stopped_at_layer else None,
            "arms": self.arms.to_dict() if self.arms else None,
            "memory": self.memory.to_dict() if self.memory else None,
            "mind": self.mind.to_dict() if self.mind else None,
            "actions": [a.to_dict() for a in self.actions],
            "error": self.error,
        }


# =============================================================================
# Original Insight/Proposal Models (for Mind layer)
# =============================================================================

class InsightType(Enum):
    """Types of insights that can be generated during reflection."""
    PATTERN = "pattern"           # Recurring behavior noticed
    IMPROVEMENT = "improvement"   # Something to fix
    SUCCESS = "success"           # What worked well
    FAILURE = "failure"           # What went wrong
    ANOMALY = "anomaly"          # Unusual occurrence


class ChangeType(Enum):
    """Types of self-modifications an agent can propose."""
    PROMPT = "prompt"             # Modify system prompt additions
    BEHAVIOR = "behavior"         # Change response to certain situations
    TOOL_USAGE = "tool_usage"     # Adjust when/how tools are used
    RESPONSE_STYLE = "response_style"  # Modify tone, length, format


@dataclass
class Insight:
    """
    A single insight from reflection.

    Represents an observation, pattern, or learning that emerged
    from analyzing past interactions.
    """
    id: str
    type: InsightType
    title: str
    description: str
    evidence: List[str] = field(default_factory=list)  # Message/episode IDs
    confidence: float = 0.5  # 0.0 - 1.0
    actionable: bool = False
    suggested_action: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "actionable": self.actionable,
            "suggested_action": self.suggested_action,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Insight":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=InsightType(data["type"]),
            title=data["title"],
            description=data["description"],
            evidence=data.get("evidence", []),
            confidence=data.get("confidence", 0.5),
            actionable=data.get("actionable", False),
            suggested_action=data.get("suggested_action"),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow()),
        )

    def __str__(self) -> str:
        actionable_marker = " [ACTIONABLE]" if self.actionable else ""
        return f"[{self.type.value.upper()}] {self.title}{actionable_marker} (confidence: {self.confidence:.0%})"


@dataclass
class ReflectionSession:
    """
    A complete reflection session.

    Tracks a single run of reflection including what was analyzed,
    what insights emerged, and what improvements were proposed.
    """
    id: str
    trigger: str  # 'sleep', 'on_demand', 'error', 'threshold'
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # What was analyzed
    interactions_analyzed: int = 0
    episodes_analyzed: int = 0
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None

    # What was produced
    insights: List[Insight] = field(default_factory=list)
    improvements_proposed: int = 0
    improvements_approved: int = 0

    # Error handling
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[int]:
        """Duration of the session in milliseconds."""
        if self.completed_at and self.started_at:
            return int((self.completed_at - self.started_at).total_seconds() * 1000)
        return None

    @property
    def insights_count(self) -> int:
        """Number of insights generated."""
        return len(self.insights)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "id": self.id,
            "trigger": self.trigger,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "interactions_analyzed": self.interactions_analyzed,
            "episodes_analyzed": self.episodes_analyzed,
            "time_range_start": self.time_range_start.isoformat() if self.time_range_start else None,
            "time_range_end": self.time_range_end.isoformat() if self.time_range_end else None,
            "insights_count": self.insights_count,
            "insights": [i.to_dict() for i in self.insights],
            "improvements_proposed": self.improvements_proposed,
            "improvements_approved": self.improvements_approved,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }

    def __str__(self) -> str:
        status = "completed" if self.completed_at else "in_progress"
        if self.error:
            status = f"failed: {self.error}"
        return (
            f"ReflectionSession({self.trigger}, {status})\n"
            f"  Analyzed: {self.interactions_analyzed} interactions, {self.episodes_analyzed} episodes\n"
            f"  Insights: {self.insights_count}\n"
            f"  Improvements: {self.improvements_proposed} proposed, {self.improvements_approved} approved"
        )


@dataclass
class ImprovementProposal:
    """
    A proposed self-modification.

    When the agent identifies something it could do better, it creates
    a proposal that must be approved before being applied.
    """
    id: str
    insight_id: Optional[str]  # Source insight (if any)

    # What is being proposed
    title: str
    description: str
    change_type: ChangeType
    proposed_change: str  # The actual modification

    # Approval status
    requires_approval: bool = True
    approved: bool = False
    rejection_reason: Optional[str] = None
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None  # User ID or "auto"

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    applied_at: Optional[datetime] = None

    @property
    def is_pending(self) -> bool:
        """Check if proposal is waiting for approval."""
        return self.requires_approval and not self.approved and not self.rejection_reason

    @property
    def is_rejected(self) -> bool:
        """Check if proposal was rejected."""
        return self.rejection_reason is not None

    @property
    def is_applied(self) -> bool:
        """Check if proposal was applied."""
        return self.applied_at is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "id": self.id,
            "insight_id": self.insight_id,
            "title": self.title,
            "description": self.description,
            "change_type": self.change_type.value,
            "proposed_change": self.proposed_change,
            "requires_approval": self.requires_approval,
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "approved_by": self.approved_by,
            "created_at": self.created_at.isoformat(),
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImprovementProposal":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            insight_id=data.get("insight_id"),
            title=data["title"],
            description=data["description"],
            change_type=ChangeType(data["change_type"]),
            proposed_change=data["proposed_change"],
            requires_approval=data.get("requires_approval", True),
            approved=data.get("approved", False),
            rejection_reason=data.get("rejection_reason"),
            approved_at=datetime.fromisoformat(data["approved_at"]) if data.get("approved_at") else None,
            approved_by=data.get("approved_by"),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow()),
            applied_at=datetime.fromisoformat(data["applied_at"]) if data.get("applied_at") else None,
        )

    def __str__(self) -> str:
        status = "pending" if self.is_pending else ("rejected" if self.is_rejected else ("applied" if self.is_applied else "approved"))
        return f"ImprovementProposal({self.change_type.value}, {status}): {self.title}"


@dataclass
class BehaviorRule:
    """
    A behavioral rule that modifies agent behavior.

    Applied improvements are stored as rules that the agent
    checks during response generation.
    """
    id: str
    proposal_id: str  # Source proposal

    # Rule definition
    trigger_condition: str  # When to apply this rule
    action: str  # What to do
    change_type: ChangeType

    # Status
    active: bool = True
    priority: int = 0  # Higher = more important

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    deactivated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "proposal_id": self.proposal_id,
            "trigger_condition": self.trigger_condition,
            "action": self.action,
            "change_type": self.change_type.value,
            "active": self.active,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "deactivated_at": self.deactivated_at.isoformat() if self.deactivated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorRule":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            proposal_id=data["proposal_id"],
            trigger_condition=data["trigger_condition"],
            action=data["action"],
            change_type=ChangeType(data["change_type"]),
            active=data.get("active", True),
            priority=data.get("priority", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow()),
            deactivated_at=datetime.fromisoformat(data["deactivated_at"]) if data.get("deactivated_at") else None,
        )


@dataclass
class SelfModel:
    """
    Agent's self-model stored in decentralized storage (Filecoin/IPFS).

    The self-model captures the agent's learned personality, communication
    style, preferences, and behavior patterns. It evolves over time through
    reflection and is stored encrypted in Lighthouse/Filecoin for sovereignty.
    """
    agent_did: str

    # Version tracking
    version: int = 1

    # Personality traits (0.0 - 1.0 scale)
    # e.g., {"helpfulness": 0.9, "formality": 0.6, "verbosity": 0.4}
    personality_traits: Dict[str, float] = field(default_factory=dict)

    # Communication style preferences
    # e.g., {"tone": "friendly", "detail_level": "moderate", "use_examples": True}
    communication_style: Dict[str, Any] = field(default_factory=dict)

    # Learned preferences from interactions
    # e.g., ["user prefers code examples", "user likes concise explanations"]
    learned_preferences: List[str] = field(default_factory=list)

    # Behavior patterns to follow or avoid
    # e.g., ["[Success] Always explain reasoning", "[Avoid] Long responses without breaks"]
    behavior_patterns: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "agent_did": self.agent_did,
            "version": self.version,
            "personality_traits": self.personality_traits,
            "communication_style": self.communication_style,
            "learned_preferences": self.learned_preferences,
            "behavior_patterns": self.behavior_patterns,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_bytes(self) -> bytes:
        """Convert to bytes for storage."""
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelfModel":
        """Create from dictionary."""
        return cls(
            agent_did=data["agent_did"],
            version=data.get("version", 1),
            personality_traits=data.get("personality_traits", {}),
            communication_style=data.get("communication_style", {}),
            learned_preferences=data.get("learned_preferences", []),
            behavior_patterns=data.get("behavior_patterns", []),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow()),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else data.get("updated_at", datetime.utcnow()),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "SelfModel":
        """Create from bytes."""
        return cls.from_dict(json.loads(data.decode("utf-8")))

    @classmethod
    def default(cls, agent_did: str) -> "SelfModel":
        """Create a default self-model for a new agent."""
        return cls(
            agent_did=agent_did,
            version=1,
            personality_traits={
                "helpfulness": 0.8,
                "formality": 0.5,
                "verbosity": 0.5,
                "creativity": 0.5,
            },
            communication_style={
                "tone": "friendly",
                "detail_level": "moderate",
                "use_examples": True,
            },
            learned_preferences=[],
            behavior_patterns=[],
        )

    def __str__(self) -> str:
        return (
            f"SelfModel(v{self.version}, {self.agent_did[:20]}...)\n"
            f"  Traits: {len(self.personality_traits)}\n"
            f"  Preferences: {len(self.learned_preferences)}\n"
            f"  Patterns: {len(self.behavior_patterns)}"
        )
