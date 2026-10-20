"""Merge an owned handler's changes without losing concurrent durable work."""

from collections.abc import Sequence
from typing import TypeVar

from a2a.types import Artifact, Message, Task
from google.protobuf.message import Message as ProtoMessage

_Item = TypeVar("_Item", Artifact, Message)


class TaskOutcomeConflictError(ValueError):
    """An execution changed data that another writer changed after admission."""


def _index(items: Sequence[_Item], attribute: str) -> dict[str, _Item]:
    indexed = {}
    for item in items:
        identity = getattr(item, attribute)
        if not identity or identity in indexed:
            raise TaskOutcomeConflictError(
                f"Outcome has missing or duplicate {attribute}"
            )
        indexed[identity] = item
    return indexed


def _same(left: ProtoMessage | None, right: ProtoMessage | None) -> bool:
    return left == right


def merge_task_outcome(current: Task, baseline: Task, outcome: Task) -> Task:
    """Apply a handler delta to the latest row, preserving independent changes.

    The caller must hold the row lock and enforce its recipient, tenant and
    live-state predicate. This function supplies no authentication authority.
    History and artifact removal are forbidden; published data cannot silently
    disappear. Conflicting replacements fail before any database mutation.
    """
    if not all(isinstance(value, Task) for value in (current, baseline, outcome)):
        raise TypeError("Outcome merging requires official A2A Task values")
    identity = (baseline.id, baseline.context_id)
    if any((value.id, value.context_id) != identity for value in (current, outcome)):
        raise TaskOutcomeConflictError(
            "Handler outcome changed its task or context identity"
        )
    merged = Task.FromString(current.SerializeToString())
    merged.status.CopyFrom(outcome.status)

    baseline_artifacts = _index(baseline.artifacts, "artifact_id")
    current_artifacts = _index(current.artifacts, "artifact_id")
    outcome_artifacts = _index(outcome.artifacts, "artifact_id")
    if baseline_artifacts.keys() - outcome_artifacts.keys():
        raise TaskOutcomeConflictError("Handler outcome removed a committed artifact")
    positions = {
        artifact.artifact_id: index for index, artifact in enumerate(merged.artifacts)
    }
    for identity, wanted in outcome_artifacts.items():
        before = baseline_artifacts.get(identity)
        if _same(before, wanted):
            continue
        latest = current_artifacts.get(identity)
        if not _same(latest, before) and not _same(latest, wanted):
            raise TaskOutcomeConflictError(
                f"Artifact changed while handler ran: {identity}"
            )
        if identity in positions:
            merged.artifacts[positions[identity]].CopyFrom(wanted)
        else:
            merged.artifacts.append(wanted)

    prefix = list(baseline.history)
    if (
        list(current.history[: len(prefix)]) != prefix
        or list(outcome.history[: len(prefix)]) != prefix
    ):
        raise TaskOutcomeConflictError(
            "Handler outcome changed committed message history"
        )
    messages = _index(current.history, "message_id")
    _index(outcome.history, "message_id")
    for message in outcome.history[len(prefix) :]:
        latest = messages.get(message.message_id)
        if latest is not None and latest != message:
            raise TaskOutcomeConflictError(
                f"Message changed while handler ran: {message.message_id}"
            )
        if latest is None:
            merged.history.append(message)
            messages[message.message_id] = message

    before_fields = baseline.metadata.fields
    wanted_fields = outcome.metadata.fields
    current_fields = current.metadata.fields
    for key in before_fields.keys() | wanted_fields.keys():
        before, wanted = before_fields.get(key), wanted_fields.get(key)
        if _same(before, wanted):
            continue
        latest = current_fields.get(key)
        if not _same(latest, before) and not _same(latest, wanted):
            raise TaskOutcomeConflictError(f"Metadata changed while handler ran: {key}")
        if wanted is None:
            merged.metadata.fields.pop(key, None)
        else:
            merged.metadata.fields[key].CopyFrom(wanted)
    if not merged.metadata.fields and baseline.HasField("metadata") != outcome.HasField(
        "metadata"
    ):
        if outcome.HasField("metadata"):
            merged.metadata.SetInParent()
        else:
            merged.ClearField("metadata")
    return merged
