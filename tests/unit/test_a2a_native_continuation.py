"""Standard same-task continuation through both official SDK HTTP bindings."""

import pytest
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import Role, SendMessageRequest, TaskState

from tests.unit.test_a2a_native_http import submit
from tests.unit.test_a2a_native_http import wire as wire


@pytest.mark.parametrize(
    "state",
    [
        TaskState.TASK_STATE_SUBMITTED,
        TaskState.TASK_STATE_WORKING,
        TaskState.TASK_STATE_INPUT_REQUIRED,
        TaskState.TASK_STATE_AUTH_REQUIRED,
    ],
)
async def test_continuation_infers_context_and_preserves_the_same_task_and_history(
    wire, state
):
    initial = await submit(wire)
    if state != initial.status.state:
        await wire.manager.update_status(
            initial.id,
            state,
            recipient_agent_id=wire.admission.scope.recipient_id,
        )
    request = SendMessageRequest(
        tenant=wire.admission.scope.tenant,
        message=new_text_message(
            "A new refinement", role=Role.ROLE_USER, task_id=initial.id
        ),
        configuration={"return_immediately": True},
    )
    before = request.SerializeToString()
    [event] = [event async for event in wire.client().send_message(request)]
    assert event.HasField("task")
    continued = event.task
    assert continued.id == initial.id and continued.context_id == initial.context_id
    assert continued.history[0] == initial.history[0]
    assert continued.history[-1].parts[0].text == "A new refinement"
    assert request.SerializeToString() == before
    snapshot = await wire.manager.get_task_snapshot(
        initial.id,
        recipient_agent_id=wire.admission.scope.recipient_id,
    )
    assert snapshot.input_revision == 1
    assert wire.submissions[-1].input_revision == 1
    assert len(wire.submissions) == 2


@pytest.mark.parametrize("fault", ["missing-task", "context", "terminal", "agent-role"])
async def test_continuation_returns_standard_errors_without_changing_the_task(
    wire, fault
):
    from a2a.utils.errors import (
        InvalidParamsError,
        TaskNotFoundError,
        UnsupportedOperationError,
    )

    initial = await submit(wire)
    if fault == "terminal":
        await wire.manager.update_status(
            initial.id,
            TaskState.TASK_STATE_COMPLETED,
            recipient_agent_id=wire.admission.scope.recipient_id,
        )
    before = await wire.manager.get_task_snapshot(
        initial.id, recipient_agent_id=wire.admission.scope.recipient_id
    )
    request = SendMessageRequest(
        tenant=wire.admission.scope.tenant,
        message=new_text_message(
            "Follow-up",
            role=Role.ROLE_AGENT if fault == "agent-role" else Role.ROLE_USER,
            task_id="missing" if fault == "missing-task" else initial.id,
            context_id="wrong-context" if fault == "context" else "",
        ),
        configuration={"return_immediately": True},
    )
    expected = (
        TaskNotFoundError
        if fault == "missing-task"
        else UnsupportedOperationError
        if fault == "terminal"
        else InvalidParamsError
    )
    with pytest.raises(expected):
        [event async for event in wire.client().send_message(request)]
    assert (
        await wire.manager.get_task_snapshot(
            initial.id, recipient_agent_id=wire.admission.scope.recipient_id
        )
        == before
    )
    assert len(wire.submissions) == 1
