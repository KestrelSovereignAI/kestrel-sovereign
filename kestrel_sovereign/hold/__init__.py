"""Durable lifecycle Hold state.

Hold is a restart-surviving latch.  It is deliberately separate from the
momentary cooperative Stop authority in :mod:`kestrel_sovereign.stop`.
"""

from .state import (
    HOST_HOLD_TARGET,
    EffectiveHoldState,
    HoldAction,
    HoldCorruptStateError,
    HoldDisposition,
    HoldIdempotencyConflict,
    HoldMutation,
    HoldReceipt,
    HoldScope,
    HoldState,
    HoldStateError,
    HoldStore,
)
from .enforcement import (
    HoldEnforcementUnavailableError,
    HoldTurnRefusal,
    build_bound_host_context,
    close_bound_host_context,
    initialize_with_bound_hold_context,
    require_context_hold_store,
    require_turn_start_allowed,
)

__all__ = [
    "HOST_HOLD_TARGET",
    "EffectiveHoldState",
    "HoldAction",
    "HoldCorruptStateError",
    "HoldDisposition",
    "HoldIdempotencyConflict",
    "HoldMutation",
    "HoldReceipt",
    "HoldScope",
    "HoldState",
    "HoldStateError",
    "HoldStore",
    "HoldEnforcementUnavailableError",
    "HoldTurnRefusal",
    "build_bound_host_context",
    "close_bound_host_context",
    "initialize_with_bound_hold_context",
    "require_context_hold_store",
    "require_turn_start_allowed",
]
