"""
Memory Models for Human-Like Memory System.

Data classes for emotional tagging, temporal patterns, and narrative episodes.
These extend the existing conversation metadata without schema changes.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class EmotionalCategory(Enum):
    """Categories of emotions that can be detected in messages."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NOSTALGIA = "nostalgia"
    LOVE = "love"
    ANXIETY = "anxiety"
    HOPE = "hope"
    GRATITUDE = "gratitude"
    FRUSTRATION = "frustration"


@dataclass
class MemoryMetadata:
    """
    Extended metadata for conversation messages.

    Stored in the existing conversation_history.metadata JSON column.
    All fields are optional with sensible defaults for backward compatibility.
    """
    # Emotional layer
    emotional_valence: float = 0.0      # -1.0 (negative) to +1.0 (positive)
    emotional_intensity: float = 0.0    # 0.0 (neutral) to 1.0 (intense)
    emotional_categories: List[str] = field(default_factory=list)

    # Importance layer
    importance: float = 0.5             # 0.0 to 1.0
    importance_reasons: List[str] = field(default_factory=list)

    # Temporal layer
    time_of_day: str = ""               # morning/afternoon/evening/late_night
    day_of_week: str = ""               # monday-sunday

    # Decay layer
    access_count: int = 0                # Times retrieved (loaded into context)
    last_accessed: Optional[str] = None  # ISO format timestamp
    # `applied_count` is distinct from `access_count`: a retrieval scored
    # the memory into the context window, but "applied" means the memory
    # was demonstrably load-bearing — it changed what the agent did or
    # said next.  A memory can be accessed every session and never
    # applied; treating those identically rewards decoration on the
    # decay side.  Reflection / pre-sleep hooks populate this via
    # ``MemorySystem.mark_applied``; auto-detection is deliberately out
    # of scope for the primitive itself.  See #1326.
    applied_count: int = 0
    last_applied: Optional[str] = None   # ISO format timestamp
    decay_protected: bool = False

    # Context management layer (agent-controlled pruning)
    context_priority: Optional[str] = None  # "protected" | "droppable" | None
    excluded_from_context: bool = False
    excluded_at: Optional[str] = None       # ISO format timestamp
    excluded_reason: Optional[str] = None
    summarized: bool = False
    summarized_into: Optional[str] = None   # ID of summary message that replaced this

    # Stash layer (temporary context parking)
    stashed: bool = False
    stash_id: Optional[str] = None          # Groups messages in same stash
    stash_name: Optional[str] = None        # Human-readable name for the stash
    stashed_at: Optional[str] = None        # ISO format timestamp

    # Epistemic status layer
    claim_certainty: Optional[float] = None    # 0.0 (uncertain) to 1.0 (certain)
    claim_source: Optional[str] = None         # "direct" | "inferred" | "hearsay" | "observed"
    temporal_validity: Optional[str] = None    # "durable" | "ephemeral" | "moment"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON storage in metadata column."""
        return {
            "emotional_valence": self.emotional_valence,
            "emotional_intensity": self.emotional_intensity,
            "emotional_categories": self.emotional_categories,
            "importance": self.importance,
            "importance_reasons": self.importance_reasons,
            "time_of_day": self.time_of_day,
            "day_of_week": self.day_of_week,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "applied_count": self.applied_count,
            "last_applied": self.last_applied,
            "decay_protected": self.decay_protected,
            # Context management fields
            "context_priority": self.context_priority,
            "excluded_from_context": self.excluded_from_context,
            "excluded_at": self.excluded_at,
            "excluded_reason": self.excluded_reason,
            "summarized": self.summarized,
            "summarized_into": self.summarized_into,
            # Stash fields
            "stashed": self.stashed,
            "stash_id": self.stash_id,
            "stash_name": self.stash_name,
            "stashed_at": self.stashed_at,
            # Epistemic status fields
            "claim_certainty": self.claim_certainty,
            "claim_source": self.claim_source,
            "temporal_validity": self.temporal_validity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMetadata":
        """Create from metadata dict. Handles missing fields gracefully."""
        if not data:
            return cls()
        return cls(
            emotional_valence=data.get("emotional_valence", 0.0),
            emotional_intensity=data.get("emotional_intensity", 0.0),
            emotional_categories=data.get("emotional_categories", []),
            importance=data.get("importance", 0.5),
            importance_reasons=data.get("importance_reasons", []),
            time_of_day=data.get("time_of_day", ""),
            day_of_week=data.get("day_of_week", ""),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
            applied_count=data.get("applied_count", 0),
            last_applied=data.get("last_applied"),
            decay_protected=data.get("decay_protected", False),
            # Context management fields
            context_priority=data.get("context_priority"),
            excluded_from_context=data.get("excluded_from_context", False),
            excluded_at=data.get("excluded_at"),
            excluded_reason=data.get("excluded_reason"),
            summarized=data.get("summarized", False),
            summarized_into=data.get("summarized_into"),
            # Stash fields
            stashed=data.get("stashed", False),
            stash_id=data.get("stash_id"),
            stash_name=data.get("stash_name"),
            stashed_at=data.get("stashed_at"),
            # Epistemic status fields
            claim_certainty=data.get("claim_certainty"),
            claim_source=data.get("claim_source"),
            temporal_validity=data.get("temporal_validity"),
        )

    def merge_with(self, existing_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge memory metadata with existing metadata dict.

        Preserves existing fields (enc, audit_failure, etc.) while adding
        memory enrichment fields.
        """
        result = dict(existing_meta) if existing_meta else {}
        result.update(self.to_dict())
        return result


@dataclass
class TemporalPattern:
    """
    A detected pattern in user behavior.

    Stored in the temporal_patterns table.
    """
    id: str
    agent_id: str
    pattern_type: str           # weekly_rhythm, time_preference, emotional_cycle
    description: str            # "User reflects deeply late at night"
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0     # 0.0 to 1.0
    observations: int = 0       # Number of observations supporting this pattern
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for database storage."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "trigger_conditions": self.trigger_conditions,
            "confidence": self.confidence,
            "observations": self.observations,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "TemporalPattern":
        """Create from database row."""
        import json
        return cls(
            id=row[0],
            agent_id=row[1],
            pattern_type=row[2],
            description=row[3],
            trigger_conditions=json.loads(row[4]) if row[4] else {},
            confidence=row[5] or 0.0,
            observations=row[6] or 0,
            created_at=datetime.fromisoformat(row[7]) if row[7] else None,
            updated_at=datetime.fromisoformat(row[8]) if row[8] else None,
        )


@dataclass
class MemoryEpisode:
    """
    A consolidated narrative from related messages.

    Stored in the memory_episodes table.
    Episodes are created during nightly consolidation to summarize
    related conversations into coherent narratives.
    """
    id: str
    agent_id: str
    title: str                  # "The debugging session breakthrough"
    summary: str                # LLM-generated narrative
    timespan_start: Optional[datetime] = None
    timespan_end: Optional[datetime] = None
    key_message_ids: List[str] = field(default_factory=list)
    emotional_arc: str = ""     # "frustration → breakthrough → celebration"
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for database storage."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "title": self.title,
            "summary": self.summary,
            "timespan_start": self.timespan_start.isoformat() if self.timespan_start else None,
            "timespan_end": self.timespan_end.isoformat() if self.timespan_end else None,
            "key_message_ids": self.key_message_ids,
            "emotional_arc": self.emotional_arc,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "MemoryEpisode":
        """Create from database row."""
        import json
        return cls(
            id=row[0],
            agent_id=row[1],
            title=row[2],
            summary=row[3],
            timespan_start=datetime.fromisoformat(row[4]) if row[4] else None,
            timespan_end=datetime.fromisoformat(row[5]) if row[5] else None,
            key_message_ids=json.loads(row[6]) if row[6] else [],
            emotional_arc=row[7] or "",
            created_at=datetime.fromisoformat(row[8]) if row[8] else None,
        )
