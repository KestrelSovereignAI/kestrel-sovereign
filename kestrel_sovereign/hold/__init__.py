"""Durable lifecycle Hold state.

Hold is a restart-surviving latch.  It is deliberately separate from the
momentary cooperative Stop authority in :mod:`kestrel_sovereign.stop`.
"""

from .enforcement import (
    HeldWorkDisposition,
    HoldEnforcementUnavailableError,
    HoldTurnRefusal,
    build_bound_host_context,
    close_bound_host_context,
    get_effective_hold_state,
    initialize_with_bound_hold_context,
    require_context_hold_store,
    require_turn_start_allowed,
    source_owns_hold_disposition,
)
from .state import (
    HOST_HOLD_TARGET,
    EffectiveHoldState,
    HoldAction,
    HoldCorruptStateError,
    HoldDisposition,
    HoldIdempotencyConflict,
    HoldMutation,
    HoldReceipt,
    HoldReceiptPage,
    HoldScope,
    HoldState,
    HoldStateError,
    HoldStore,
    hold_latch_payload,
    hold_receipt_payload,
)

__all__ = [
    "HOST_HOLD_TARGET",
    "EffectiveHoldState",
    "HeldWorkDisposition",
    "HoldAction",
    "HoldCorruptStateError",
    "HoldDisposition",
    "HoldEnforcementUnavailableError",
    "HoldIdempotencyConflict",
    "HoldMutation",
    "HoldReceipt",
    "HoldReceiptPage",
    "HoldScope",
    "HoldState",
    "HoldStateError",
    "HoldStore",
    "HoldTurnRefusal",
    "build_bound_host_context",
    "close_bound_host_context",
    "get_effective_hold_state",
    "hold_latch_payload",
    "hold_receipt_payload",
    "initialize_with_bound_hold_context",
    "require_context_hold_store",
    "require_turn_start_allowed",
    "source_owns_hold_disposition",
]
