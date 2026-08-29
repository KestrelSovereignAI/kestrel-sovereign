"""Cooperative work-cancellation domain.

Process and runtime termination deliberately live outside this package.
"""

from .authority import (
    CancellationAuthority,
    CooperativeStopTarget,
    StopCleanupRegistry,
)
from .receipt import (
    StopOperationClaim,
    StopReceipt,
    StopReceiptConflict,
    StopReceiptCorruptError,
    StopReceiptError,
    StopReceiptStore,
    UnavailableStopReceiptStore,
)
from .types import StopDisposition, StopOutcome, StopRequest, StopScope

__all__ = [
    "CancellationAuthority",
    "CooperativeStopTarget",
    "StopDisposition",
    "StopCleanupRegistry",
    "StopOutcome",
    "StopOperationClaim",
    "StopReceipt",
    "StopReceiptConflict",
    "StopReceiptCorruptError",
    "StopReceiptError",
    "StopReceiptStore",
    "StopRequest",
    "StopScope",
    "UnavailableStopReceiptStore",
]
