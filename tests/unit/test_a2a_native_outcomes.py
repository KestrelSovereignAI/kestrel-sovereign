"""Native outcome deltas preserve committed progress and refuse conflicts."""

import pytest
from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_task, new_text_artifact, new_text_message
from kestrel_sovereign._vendor.a2a.types import Role, Task, TaskState
from google.protobuf.json_format import MessageToDict

from kestrel_sovereign.a2a.outcomes import TaskOutcomeConflictError, merge_task_outcome


def snapshots():
    baseline = new_task(
        "task",
        "context",
        TaskState.TASK_STATE_SUBMITTED,
        history=[new_text_message("request", role=Role.ROLE_USER)],
    )
    baseline.artifacts.append(
        new_text_artifact("original", "first", artifact_id="original")
    )
    baseline.metadata.update({"value": {"nested": [None, {}, []]}, "removed": "old"})
    return (
        baseline,
        Task.FromString(baseline.SerializeToString()),
        Task.FromString(baseline.SerializeToString()),
    )


def test_outcome_preserves_concurrent_progress_and_applies_only_its_changes():
    baseline, current, outcome = snapshots()
    current.artifacts[0].parts.add(text="concurrent chunk")
    current.artifacts.append(
        new_text_artifact("concurrent", "keep", artifact_id="concurrent")
    )
    progress = new_text_message(
        "progress", role=Role.ROLE_AGENT, task_id="task", context_id="context"
    )
    current.history.append(progress)
    current.metadata.update({"concurrent": False})
    outcome.status.state = TaskState.TASK_STATE_COMPLETED
    outcome.artifacts.append(new_text_artifact("result", "done", artifact_id="result"))
    reply = new_text_message(
        "reply", role=Role.ROLE_AGENT, task_id="task", context_id="context"
    )
    outcome.history.append(reply)
    outcome.metadata.update({"result": [None, {}, [], False]})
    del outcome.metadata.fields["removed"]
    inputs = [
        value.SerializeToString(deterministic=True)
        for value in (baseline, current, outcome)
    ]
    merged = merge_task_outcome(current, baseline, outcome)
    assert [artifact.artifact_id for artifact in merged.artifacts] == [
        "original",
        "concurrent",
        "result",
    ]
    assert merged.artifacts[0] == current.artifacts[0]
    assert list(merged.history) == [baseline.history[0], progress, reply]
    assert MessageToDict(merged.metadata) == {
        "value": {"nested": [None, {}, []]},
        "concurrent": False,
        "result": [None, {}, [], False],
    }
    assert merged.status.state == TaskState.TASK_STATE_COMPLETED
    assert [
        value.SerializeToString(deterministic=True)
        for value in (baseline, current, outcome)
    ] == inputs


@pytest.mark.parametrize(
    "field",
    ["artifact", "history", "metadata", "identity", "removed-artifact", "duplicate-id"],
)
def test_conflicting_or_destructive_outcomes_fail_explicitly(field):
    baseline, current, outcome = snapshots()
    if field == "artifact":
        current.artifacts[0].parts[0].text = "concurrent replacement"
        outcome.artifacts[0].parts[0].text = "handler replacement"
    elif field == "history":
        outcome.history[0].parts[0].text = "rewritten request"
    elif field == "metadata":
        # Distinguish JSON false from numeric zero; Python dict equality does not.
        current.metadata.update({"value": False})
        outcome.metadata.update({"value": 0})
    elif field == "identity":
        outcome.context_id = "other"
    elif field == "removed-artifact":
        del outcome.artifacts[:]
    else:
        outcome.artifacts.append(outcome.artifacts[0])
    before = current.SerializeToString(deterministic=True)
    with pytest.raises(TaskOutcomeConflictError):
        merge_task_outcome(current, baseline, outcome)
    assert current.SerializeToString(deterministic=True) == before


def test_matching_concurrent_changes_are_idempotent_data_updates():
    baseline, current, outcome = snapshots()
    artifact = new_text_artifact("same", "same", artifact_id="same")
    current.artifacts.append(artifact)
    outcome.artifacts.append(artifact)
    message = new_text_message(
        "same", role=Role.ROLE_AGENT, task_id="task", context_id="context"
    )
    current.history.append(message)
    outcome.history.append(message)
    current.metadata.update({"same": None})
    outcome.metadata.update({"same": None})
    merged = merge_task_outcome(current, baseline, outcome)
    assert list(merged.artifacts) == list(current.artifacts)
    assert list(merged.history) == list(current.history)


@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize("latest", [False, True])
@pytest.mark.parametrize("wanted", [False, True])
def test_empty_metadata_presence_is_a_native_delta(before, latest, wanted):
    baseline, current, outcome = snapshots()
    for value, present in ((baseline, before), (current, latest), (outcome, wanted)):
        value.ClearField("metadata")
        if present:
            value.metadata.SetInParent()
    merged = merge_task_outcome(current, baseline, outcome)
    assert merged.HasField("metadata") is (wanted if wanted != before else latest)


def test_concurrent_metadata_keys_survive_a_handler_clearing_its_metadata():
    baseline, current, outcome = snapshots()
    outcome.ClearField("metadata")
    current.metadata.update({"concurrent": [None, False, {}]})
    merged = merge_task_outcome(current, baseline, outcome)
    assert MessageToDict(merged.metadata) == {"concurrent": [None, False, {}]}
