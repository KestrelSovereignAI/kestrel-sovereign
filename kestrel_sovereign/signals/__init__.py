"""Signal Dispatcher — runtime.

Public contract types live in `kestrel_sdk.signals`. This package contains
the runtime pieces (registry, dispatcher, lock manager, signal_log store)
that nothing outside sovereign should import.

Design: docs/architecture/SIGNAL_DISPATCHER.md.

Signals originate work; hooks intercept work in flight. A signal can produce
a turn that fires hooks; a hook never produces a signal. One-way arrow.

Approval is NOT a signal — it is a gate release on a paused turn and lives
in the hook system.
"""

# Re-export contract types from the SDK so internal sovereign callers have
# a single import path. The SDK module is the source of truth.
from kestrel_sdk.signals import (
    ActionHandler,
    ArtifactHandler,
    AttentionPolicy,
    CausationFrame,
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

# Runtime — sovereign-internal.
from kestrel_sovereign.signals.dispatcher import (
    DEFAULT_COALESCING_WINDOW,
    DEFAULT_TTL,
    DispatcherAgent,
    SignalDispatcher,
)
from kestrel_sovereign.signals.handlers import template_artifact_handler
from kestrel_sovereign.signals.lock_manager import OrderedLockManager
from kestrel_sovereign.signals.prompt_overrides import (
    SignalWithPromptTemplateOverride,
    SourceRegistrationWithPromptOverride,
)
from kestrel_sovereign.signals.registry import RegistrationError, SourceRegistry
from kestrel_sovereign.signals.store import SignalLogStore

__all__ = [
    # Contract (re-exported from SDK)
    "ActionHandler",
    "ArtifactHandler",
    "AttentionPolicy",
    "CausationFrame",
    "RateLimit",
    "RedactionPolicy",
    "ResourceLock",
    "Signal",
    "SignalHandle",
    "SignalMode",
    "SignalResult",
    "SignalWithPromptTemplateOverride",
    "SourceRegistration",
    "SourceRegistrationWithPromptOverride",
    "Status",
    "Trust",
    "Urgency",
    "Visibility",
    # Runtime
    "DEFAULT_COALESCING_WINDOW",
    "DEFAULT_TTL",
    "DispatcherAgent",
    "OrderedLockManager",
    "RegistrationError",
    "SignalDispatcher",
    "SignalLogStore",
    "SourceRegistry",
    "template_artifact_handler",
]
