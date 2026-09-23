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
    StopLegacyRegistrationsError,
)
from .receipt import (
    StopOperationClaim,
    StopReceipt,
    StopReceiptConflict,
    StopReceiptCorruptError,
    StopReceiptError,
    StopReceiptOutcomeRecord,
    StopReceiptPage,
    StopReceiptRecord,
    StopReceiptStore,
    UnavailableStopReceiptStore,
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
    "MAX_STOP_CORRELATION_ID_BYTES",
    "AuthoritativeStopDescendant",
    "CancellationAuthority",
    "CooperativeStopTarget",
    "DistributedInvocationRegistry",
    "DistributedInvocationStore",
    "DistributedStopTicket",
    "StopCleanupRegistry",
    "StopDisposition",
    "StopLegacyRegistrationsError",
    "StopOperationClaim",
    "StopOutcome",
    "StopReceipt",
    "StopReceiptConflict",
    "StopReceiptCorruptError",
    "StopReceiptError",
    "StopReceiptOutcomeRecord",
    "StopReceiptPage",
    "StopReceiptRecord",
    "StopReceiptStore",
    "StopRequest",
    "StopScope",
    "UnavailableStopReceiptStore",
    "execute_fleet_stop",
    "fleet_in_flight_count",
]
