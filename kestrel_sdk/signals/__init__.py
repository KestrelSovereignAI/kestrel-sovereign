"""Kestrel SDK — Signal Dispatcher interfaces.

Types and contracts that any signal source must develop against. The
runtime (`SignalDispatcher`, `SourceRegistry`, lock manager, signal_log
store) lives in `kestrel_sovereign.signals` and should never be imported
from feature packages — depend on these types instead.

Design: docs/architecture/SIGNAL_DISPATCHER.md (in the sovereign repo).
"""

from kestrel_sdk.signals.models import (
    ActionHandler,
    ArtifactHandler,
    AttentionPolicy,
    CausationFrame,
    PayloadSchema,
    RateLimit,
    RedactionPolicy,
    ResourceLock,
    Signal,
    SignalHandle,
    SignalMode,
    SignalResult,
    SourceRegistration,
    Status,
    Trust,
    Urgency,
    Visibility,
)

__all__ = [
    "ActionHandler",
    "ArtifactHandler",
    "AttentionPolicy",
    "CausationFrame",
    "PayloadSchema",
    "RateLimit",
    "RedactionPolicy",
    "ResourceLock",
    "Signal",
    "SignalHandle",
    "SignalMode",
    "SignalResult",
    "SourceRegistration",
    "Status",
    "Trust",
    "Urgency",
    "Visibility",
]
