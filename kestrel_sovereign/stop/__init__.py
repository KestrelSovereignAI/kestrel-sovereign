"""Cooperative work-cancellation domain.

Process and runtime termination deliberately live outside this package.
"""

from .authority import (
    CancellationAuthority,
    CooperativeStopTarget,
    StopCleanupRegistry,
)
from .types import StopDisposition, StopOutcome, StopRequest, StopScope

__all__ = [
    "CancellationAuthority",
    "CooperativeStopTarget",
    "StopDisposition",
    "StopCleanupRegistry",
    "StopOutcome",
    "StopRequest",
    "StopScope",
]
