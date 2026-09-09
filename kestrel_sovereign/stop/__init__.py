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
from .invocation import (
    DistributedInvocationRegistry,
    DistributedInvocationStore,
    DistributedStopTicket,
)
from .types import (
    AuthoritativeStopDescendant,
    MAX_STOP_CORRELATION_ID_BYTES,
    StopDisposition,
    StopOutcome,
    StopRequest,
    StopScope,
)

__all__ = [
    "AuthoritativeStopDescendant",
    "CancellationAuthority",
    "CooperativeStopTarget",
    "DistributedInvocationRegistry",
    "DistributedInvocationStore",
    "DistributedStopTicket",
    "MAX_STOP_CORRELATION_ID_BYTES",
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
