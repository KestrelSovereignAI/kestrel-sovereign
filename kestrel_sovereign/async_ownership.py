"""Public, cancellation-safe ownership of already-owned asynchronous tails.

These are identity aliases of the canonical implementation, not replacement
owners. They grant no execution authority and do not prove descendant-process
drain. See ``docs/development/ASYNC_OWNERSHIP_PUBLIC_API.md`` for the contract.
"""

from ._async_ownership import (
    OwnedTaskAdmission,
    OwnedTaskOutcome,
    await_owned_operation,
    await_owned_task,
    create_owned_task,
    owned_task_error,
    pending_owned_retirement_error,
    raise_owned_outcome,
    run_blocking_operation,
    wait_owned_tasks_until,
)

__all__ = [
    "OwnedTaskAdmission",
    "OwnedTaskOutcome",
    "await_owned_operation",
    "await_owned_task",
    "create_owned_task",
    "owned_task_error",
    "pending_owned_retirement_error",
    "raise_owned_outcome",
    "run_blocking_operation",
    "wait_owned_tasks_until",
]
