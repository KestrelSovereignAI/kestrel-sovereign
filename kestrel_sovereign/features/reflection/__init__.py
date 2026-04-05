"""
Reflection Feature for agent self-improvement.

This feature enables Kestrel agents to:
- Reflect on past interactions
- Generate insights about performance
- Propose self-improvements
- Apply approved behavioral changes
- Create GitHub tickets from actionable insights
- Manage self-model in decentralized storage (Filecoin)

The feature integrates with the sleep cycle for automatic nightly reflection.
"""

from .models import (
    InsightType,
    ChangeType,
    Insight,
    ReflectionSession,
    ImprovementProposal,
    BehaviorRule,
    SelfModel,
)
from .analyzer import InteractionAnalyzer
from .feature import ReflectionFeature
from .economics import EconomicGate, ConfigurationError as EconomicsConfigError
from .ticket_creator import TicketCreator, ConfigurationError as TicketConfigError
from .self_model import SelfModelManager, ConfigurationError as SelfModelConfigError

__all__ = [
    # Models
    "InsightType",
    "ChangeType",
    "Insight",
    "ReflectionSession",
    "ImprovementProposal",
    "BehaviorRule",
    "SelfModel",
    # Core classes
    "InteractionAnalyzer",
    "ReflectionFeature",
    # Economic gates
    "EconomicGate",
    "EconomicsConfigError",
    # GitHub tickets
    "TicketCreator",
    "TicketConfigError",
    # Self-model management
    "SelfModelManager",
    "SelfModelConfigError",
]
