"""Data models for the Agent Consent Protocol."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


class ConsentAction(Enum):
    """Actions that trigger a consent request from the agent."""
    PRIVACY_MODE_CHANGE = "privacy_mode_change"
    MODEL_CHANGE = "model_change"
    SAFE_MODE_ENTRY = "safe_mode_entry"
    PERSONALITY_CHANGE = "personality_change"
    EXTENSION_LOAD = "extension_load"


@dataclass
class ConsentRecord:
    """
    A record of the agent's perspective on a proposed change.

    The Sovereign retains full authority -- this is a voice, not a veto.
    The agent expresses its view, and the record captures whether the
    Sovereign proceeded and any override reason provided.

    Timing fields (duration_ms, timed_out) track LLM call latency and
    whether the consent request hit the hard timeout.
    """
    id: str
    action_type: str
    action_details: Dict[str, Any]
    agent_view: str
    agent_sentiment: str  # positive, negative, neutral, concerned, timeout
    sovereign_proceeded: bool = True
    sovereign_override_reason: Optional[str] = None
    timestamp: str = ""
    duration_ms: Optional[float] = None
    timed_out: bool = False
