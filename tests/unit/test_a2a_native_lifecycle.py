"""Lifecycle labels come from the public SDK, including both added states."""

from __future__ import annotations

import pytest
from kestrel_sdk.native_a2a.types import TaskState

from kestrel_sovereign.a2a.lifecycle import (
    TERMINAL_STATES,
    task_state_from_label,
    task_state_label,
)


@pytest.mark.parametrize(
    "label,number",
    [
        ("submitted", TaskState.TASK_STATE_SUBMITTED),
        ("working", TaskState.TASK_STATE_WORKING),
        ("completed", TaskState.TASK_STATE_COMPLETED),
        ("failed", TaskState.TASK_STATE_FAILED),
        ("canceled", TaskState.TASK_STATE_CANCELED),
        ("input-required", TaskState.TASK_STATE_INPUT_REQUIRED),
        ("rejected", TaskState.TASK_STATE_REJECTED),
        ("auth-required", TaskState.TASK_STATE_AUTH_REQUIRED),
    ],
)
def test_native_state_labels_round_trip(label, number):
    assert task_state_label(number) == label
    assert task_state_from_label(label) == number


@pytest.mark.parametrize("state", [TaskState.TASK_STATE_UNSPECIFIED, -1, 99])
def test_unknown_or_unspecified_states_cannot_be_persisted(state):
    with pytest.raises(ValueError):
        task_state_label(state)


def test_rejected_is_terminal_and_auth_required_is_live():
    assert TaskState.TASK_STATE_REJECTED in TERMINAL_STATES
    assert TaskState.TASK_STATE_AUTH_REQUIRED not in TERMINAL_STATES
    assert TaskState.TASK_STATE_INPUT_REQUIRED not in TERMINAL_STATES


def test_private_unknown_state_is_not_a_native_state():
    with pytest.raises(ValueError):
        task_state_from_label("unknown")


def test_unspecified_filter_cannot_silently_list_every_state():
    with pytest.raises(ValueError, match="concrete lifecycle"):
        task_state_from_label("unspecified")


@pytest.mark.parametrize('native', [False, True])
def test_transition_timestamp_uses_explicit_instant_and_preserves_nanoseconds(native):
    from datetime import datetime, timedelta, timezone

    from google.protobuf.json_format import MessageToDict
    from google.protobuf.timestamp_pb2 import Timestamp
    from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message

    from kestrel_sovereign.a2a.lifecycle import new_task_status

    at = (Timestamp(seconds=0, nanos=123456789) if native else
          datetime(1970, 1, 1, 1, 0, 0, 123456, tzinfo=timezone(timedelta(hours=1))))
    status = new_task_status(TaskState.TASK_STATE_WORKING, at=at, message=new_text_message('working'))
    assert MessageToDict(status)['timestamp'] == ('1970-01-01T00:00:00.123456789Z' if native else '1970-01-01T00:00:00.123456Z')
    assert status.message.parts[0].text == 'working'
    if native:
        at.nanos = 0
        assert status.timestamp.nanos == 123456789


@pytest.mark.parametrize('invalid', ['naive', 'overflow', 'wrong-type', 'unknown-state'])
def test_invalid_transition_timestamp_or_state_is_refused(invalid):
    from datetime import datetime

    from google.protobuf.timestamp_pb2 import Timestamp

    from kestrel_sovereign.a2a.lifecycle import new_task_status

    naive = datetime(2026, 1, 1, tzinfo=None)  # noqa: DTZ001 - intentionally invalid status input
    at = (naive if invalid == 'naive' else
          Timestamp(seconds=253402300800) if invalid == 'overflow' else
          '2026-01-01' if invalid == 'wrong-type' else Timestamp())
    with pytest.raises((ValueError, TypeError)):
        new_task_status(-1 if invalid == 'unknown-state' else TaskState.TASK_STATE_WORKING, at=at)
