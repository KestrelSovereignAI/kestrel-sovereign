"""Cooperative work-cancellation domain.

Process and runtime termination deliberately live outside this package.
"""

from .authority import (
    CancellationAuthority,
    CooperativeStopTarget,
    StopCleanupRegistry,
)
from .fleet import execute_fleet_stop, fleet_in_flight_count
from .invocation import (
    DistributedInvocationRegistry,
    DistributedInvocationStore,
    DistributedStopTicket,
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
from .types import (
    MAX_STOP_CORRELATION_ID_BYTES,
    StopDisposition,
    StopOutcome,
    StopRequest,
    StopScope,
)

__all__ = [
    "MAX_STOP_CORRELATION_ID_BYTES",
    "CancellationAuthority",
    "CooperativeStopTarget",
    "DistributedInvocationRegistry",
    "DistributedInvocationStore",
    "DistributedStopTicket",
    "StopCleanupRegistry",
    "StopDisposition",
    "StopOperationClaim",
    "StopOutcome",
    "StopReceipt",
    "StopReceiptConflict",
    "StopReceiptCorruptError",
    "StopReceiptError",
    "StopReceiptStore",
    "StopRequest",
    "StopScope",
    "UnavailableStopReceiptStore",
    "execute_fleet_stop",
    "fleet_in_flight_count",
]
