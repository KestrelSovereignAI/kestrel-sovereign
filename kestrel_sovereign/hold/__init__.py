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
    HoldStore,
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
    "HoldStore",
]
