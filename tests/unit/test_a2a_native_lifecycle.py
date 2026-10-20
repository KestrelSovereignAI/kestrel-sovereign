"""Lifecycle labels come from the public SDK, including both added states."""

from __future__ import annotations

import pytest
from a2a.types import TaskState

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
