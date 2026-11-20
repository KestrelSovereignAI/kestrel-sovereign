"""Public, cancellation-safe ownership of already-owned asynchronous tails.

These are identity aliases of the canonical implementation, not replacement
owners. They do not independently grant execution authority or prove
descendant-process drain. See ``docs/development/ASYNC_OWNERSHIP_PUBLIC_API.md``.
"""

from ._async_ownership import (
    OwnedOperation,
    OwnedTaskAdmission,
    OwnedTaskLender,
    OwnedTaskLenderCapture,
    OwnedTaskOutcome,
    OwnedTaskRetirement,
    await_owned_operation,
    await_owned_task,
    bind_owned_task_lender,
    capture_owned_task_lender,
    create_owned_task,
    owned_task_error,
    pending_owned_retirement_error,
    raise_owned_outcome,
    reconcile_owned_task_admission,
    run_blocking_operation,
    wait_owned_tasks_until,
)

__all__ = [
    "OwnedOperation",
    "OwnedTaskAdmission",
    "OwnedTaskLender",
    "OwnedTaskLenderCapture",
    "OwnedTaskOutcome",
    "OwnedTaskRetirement",
    "await_owned_operation",
    "await_owned_task",
    "bind_owned_task_lender",
    "capture_owned_task_lender",
    "create_owned_task",
    "owned_task_error",
    "pending_owned_retirement_error",
    "raise_owned_outcome",
    "reconcile_owned_task_admission",
    "run_blocking_operation",
    "wait_owned_tasks_until",
]
