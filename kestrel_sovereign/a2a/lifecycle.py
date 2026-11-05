"""Native A2A lifecycle policy shared by persistence and task execution."""

from __future__ import annotations

from datetime import datetime

from google.protobuf.timestamp_pb2 import Timestamp
from kestrel_sdk.native_a2a.types import Message, Task, TaskState, TaskStatus

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

_OUTCOME_UNKNOWN = "outcome_unknown"


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


def new_task_status(
    state: int, *, at: datetime | Timestamp, message: Message | None = None
) -> TaskStatus:
    """Record an actual lifecycle transition using its owner's explicit clock.

    Native timestamps retain nanoseconds. Validate before persistence and never
    synthesize a new transition time when reading an existing status.
    """
    task_state_label(state)
    stamp = Timestamp()
    if isinstance(at, datetime):
        if at.utcoffset() is None:
            raise ValueError("A task status timestamp requires a timezone")
        stamp.FromDatetime(at)
    elif isinstance(at, Timestamp):
        stamp.CopyFrom(at)
    else:
        raise TypeError("A task status timestamp requires datetime or Timestamp")
    stamp.ToJsonString()
    return TaskStatus(state=state, timestamp=stamp, message=message)


def mark_outcome_unknown(task: Task) -> None:
    """Record that a failed task's effect may already have happened.

    The task stays FAILED with its own message. Its caller must reconcile
    before running the work again; it must never retry it automatically.
    """
    if task.status.state != TaskState.TASK_STATE_FAILED:
        raise ValueError("Only a failed task can carry an unknown outcome")
    task.metadata.update({_OUTCOME_UNKNOWN: True})


def task_outcome_unknown(task: Task) -> bool:
    """Whether the task failed with an effect that may already have happened."""
    value = task.metadata.fields.get(_OUTCOME_UNKNOWN)
    return (
        value is not None
        and value.WhichOneof("kind") == "bool_value"
        and value.bool_value
    )
