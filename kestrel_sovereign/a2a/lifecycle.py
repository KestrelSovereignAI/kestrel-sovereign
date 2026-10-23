"""Native A2A lifecycle policy shared by persistence and task execution."""

from __future__ import annotations

from kestrel_sovereign._vendor.a2a.types import TaskState

TERMINAL_STATES = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }
)

# An interrupted task can resume, but its current streaming response ends.
STREAM_STOP_STATES = TERMINAL_STATES | frozenset(
    {TaskState.TASK_STATE_INPUT_REQUIRED, TaskState.TASK_STATE_AUTH_REQUIRED}
)


def task_state_label(state: int) -> str:
    """Use official enum names for internal storage and projection labels."""
    if state == TaskState.TASK_STATE_UNSPECIFIED:
        raise ValueError("A persisted A2A task requires a concrete lifecycle state")
    return TaskState.Name(state).removeprefix("TASK_STATE_").lower().replace("_", "-")


def task_state_from_label(state: str) -> int:
    """Reject unknown stored labels instead of inventing a native state."""
    result = TaskState.Value("TASK_STATE_" + state.upper().replace("-", "_"))
    task_state_label(result)
    return result
