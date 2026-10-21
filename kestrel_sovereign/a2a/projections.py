"""Read-only presentation of official native protocol values."""

from a2a.types import Message


def message_text(message: Message) -> str | None:
    """Keep text parts in order without treating other oneof values as text."""
    if not isinstance(message, Message):
        raise TypeError("Text projection requires an official A2A Message")
    texts = [part.text for part in message.parts if part.HasField("text")]
    return "\n".join(texts) if texts else None


def task_presentation(candidate, *, include_artifacts: bool = True) -> dict:
    """Present native task content and durable ownership through one contract."""
    from google.protobuf.json_format import MessageToDict
    from kestrel_sovereign.a2a.extensions import parse_skill_hint
    from kestrel_sovereign.a2a.lifecycle import task_state_label
    from kestrel_sovereign.a2a.stores.unified.task_store import NativeTaskSnapshot

    if not isinstance(candidate, NativeTaskSnapshot):
        raise TypeError("Task presentation requires a native ownership snapshot")
    candidate.require_recipient(candidate.ownership.recipient_agent_id)
    task = candidate.task
    metadata = MessageToDict(task.metadata)
    result = {
        "task_id": task.id,
        "status": task_state_label(task.status.state),
        "message": message_text(task.status.message)
        if task.status.HasField("message")
        else None,
        "request_content": message_text(task.history[0]) if task.history else None,
        "request": MessageToDict(task.history[0]) if task.history else None,
        "context_id": task.context_id,
        "skill_hint": parse_skill_hint(metadata),
        "sender": candidate.ownership.creator_agent_id,
        "recipient_id": candidate.ownership.recipient_agent_id,
        "tenant": candidate.ownership.tenant,
        "task_type": metadata.get("task_type"),
        "artifacts_count": len(task.artifacts),
        "status_timestamp": task.status.timestamp.ToJsonString()
        if task.status.HasField("timestamp")
        else None,
    }
    if include_artifacts:
        result["artifacts"] = [MessageToDict(artifact) for artifact in task.artifacts]
    return result
