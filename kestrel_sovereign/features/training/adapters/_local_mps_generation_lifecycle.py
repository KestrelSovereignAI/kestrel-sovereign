"""Facade for Local MPS generation artifacts and subprocess ownership.

The filesystem capability implementation lives in
``_local_mps_generation_workspace``; bounded process-group ownership lives in
``_local_mps_generation_process``.  This module composes them in the only
cross-cutting operation: terminalize the process tree before workspace cleanup.
"""

from __future__ import annotations

import asyncio

from ._local_mps_generation_process import (
    GenerationProcessLease,
    start_generation_process,
    terminate_generation_process,
)
from ._local_mps_generation_workspace import (
    GENERATION_WORKSPACE_PREFIX,
    GenerationArtifactLease,
    GenerationWorkspaceIdentityError,
    GenerationWorkspaceLease,
    _cleanup_generation_workspace,
    _close_generation_workspace,
    _create_generation_artifact as _create_generation_artifact,
    _create_generation_workspace as _create_generation_workspace,
    _read_generation_artifact as _read_generation_artifact,
    create_generation_artifact,
    create_generation_workspace,
    read_generation_artifact,
    validate_generation_workspace,
)
from ._owned_async_task import (
    await_owned_task,
    raise_owned_outcome,
    run_blocking_operation,
)


# Kept as a narrow compatibility alias for callers that already offload a
# private artifact operation through this lifecycle facade.
run_blocking_artifact_operation = run_blocking_operation


async def finalize_generation_resources(
    *,
    process: GenerationProcessLease | None,
    process_communicated: bool,
    workspace: GenerationWorkspaceLease | None,
    pending_cancellation: asyncio.CancelledError | None = None,
) -> None:
    """Prove group extinction, then conditionally remove the workspace.

    Caller cancellation has explicit precedence.  The owned finalizer still
    reaches a terminal state; any teardown error is retrieved, logged, and
    chained beneath the original ``CancelledError``.
    """

    async def finalize() -> None:
        process_safe = process is None
        try:
            if process is not None:
                await terminate_generation_process(
                    process,
                    direct_child_reaped=process_communicated,
                )
                process_safe = True
        finally:
            if workspace is not None:
                if process_safe:
                    await asyncio.to_thread(_cleanup_generation_workspace, workspace)
                else:
                    await asyncio.to_thread(_close_generation_workspace, workspace)

    finalization_task = asyncio.create_task(finalize())
    outcome = await await_owned_task(finalization_task, pending_cancellation)
    raise_owned_outcome(outcome, operation="Generation resource finalization")


__all__ = [
    "GENERATION_WORKSPACE_PREFIX",
    "GenerationArtifactLease",
    "GenerationProcessLease",
    "GenerationWorkspaceIdentityError",
    "GenerationWorkspaceLease",
    "create_generation_artifact",
    "create_generation_workspace",
    "finalize_generation_resources",
    "read_generation_artifact",
    "run_blocking_artifact_operation",
    "start_generation_process",
    "validate_generation_workspace",
]
