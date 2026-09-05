"""Cooperative work-cancellation domain.

Process and runtime termination deliberately live outside this package.
"""

from .authority import (
    CancellationAuthority,
    CooperativeStopTarget,
    StopCleanupRegistry,
)
from .agent_target import (
    agent_stop_identity,
    build_agent_cancellation_authority,
    build_agent_stop_target,
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
    "agent_stop_identity",
    "build_agent_cancellation_authority",
    "build_agent_stop_target",
]
