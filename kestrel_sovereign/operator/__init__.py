"""Generic runtime support for SDK operator contracts.

Feature packages declare operator-facing contracts through :mod:`kestrel_sdk`.
Sovereign owns their active in-process registrations through the registry
exported here; no concrete feature or execution engine is imported.
"""

from .runtime import (
    ExecutionTargetRegistration,
    ExecutionTargetUnavailableError,
    OperatorRegistrationConflictError,
    OperatorRegistrationError,
    OperatorRegistrationIdentityError,
    OperatorRegistrationSet,
    OperatorRuntimeRegistry,
)

__all__ = [
    "ExecutionTargetRegistration",
    "ExecutionTargetUnavailableError",
    "OperatorRegistrationConflictError",
    "OperatorRegistrationError",
    "OperatorRegistrationIdentityError",
    "OperatorRegistrationSet",
    "OperatorRuntimeRegistry",
]
