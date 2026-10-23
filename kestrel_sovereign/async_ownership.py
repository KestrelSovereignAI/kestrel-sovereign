"""Public, cancellation-safe ownership of already-owned asynchronous tails.

These are identity aliases of the canonical implementation, not replacement
owners. They grant no execution authority and do not prove descendant-process
drain. See ``docs/development/ASYNC_OWNERSHIP_PUBLIC_API.md`` for the contract.
"""

from ._async_ownership import (
    OwnedTaskOutcome,
    await_owned_operation,
    await_owned_task,
    raise_owned_outcome,
    run_blocking_operation,
)

__all__ = [
    "OwnedTaskOutcome",
    "await_owned_operation",
    "await_owned_task",
    "raise_owned_outcome",
    "run_blocking_operation",
]
