"""One-shot offline conversion of stored draft A2A data.

No runtime handler or client imports this module. Durable ownership comes from
an operator-reviewed migration plan and existing authority columns, never from
sender claims in the legacy metadata. Live work must drain before conversion.
The migration retains the original rows separately for audit.
"""

import base64
import binascii
import json
from collections.abc import Mapping
from uuid import NAMESPACE_URL, uuid5

from kestrel_sdk.native_a2a.types import Artifact, Message, Part, Role, Task, TaskState, TaskStatus
from kestrel_sdk.native_a2a.utils.proto_utils import validate_proto_required_fields
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct

from kestrel_sovereign.a2a.extensions import (
    NATIVE_EXTENSION_URIS,
    QUESTION_CORRELATION_URI,
    causation_metadata,
    correlated_answer_metadata,
    delivery_intent_metadata,
    question_correlation_metadata,
    skill_hint_metadata,
)
from kestrel_sovereign.a2a.lifecycle import TERMINAL_STATES, task_state_from_label
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.context_store import ContextOwnership


class LegacyTaskMigrationError(ValueError):
    """Stored data is malformed, still live, or lacks reviewed provenance."""


def _json(value):
    """Clone finite JSON without normalizing corrupt or non-JSON values."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise LegacyTaskMigrationError(
            "Legacy task contains non-finite or non-JSON content"
        ) from error


def _object(value, description):
    if not isinstance(value, Mapping):
        raise LegacyTaskMigrationError(f"Legacy {description} must be an object")
    return _json(dict(value))


def _text(value, description):
    if not isinstance(value, str) or not value.strip():
        raise LegacyTaskMigrationError(
            f"Legacy {description} must be a concrete string"
        )
    return value


def _optional_text(value, description):
    if value is not None and not isinstance(value, str):
        raise LegacyTaskMigrationError(f"Legacy {description} must be text or null")
    return value


def _parts(values):
    if not isinstance(values, list) or not values:
        raise LegacyTaskMigrationError(
            "Legacy message/artifact requires nonempty parts"
        )
    result = []
    for raw in values:
        entry = _object(raw, "part")
        kind = entry.get("type")
        body = (
            {"metadata": _object(entry["metadata"], "part metadata")}
            if entry.get("metadata") is not None
            else {}
        )
        if kind == "text" and set(entry) <= {"type", "text", "metadata"}:
            if not isinstance(entry.get("text"), str):
                raise LegacyTaskMigrationError("Legacy text part must contain text")
            body["text"] = entry["text"]
        elif kind == "data" and set(entry) <= {"type", "data", "metadata"}:
            body["data"] = _object(entry.get("data"), "data part")
        elif kind == "file" and set(entry) <= {"type", "file", "metadata"}:
            file = _object(entry.get("file"), "file content")
            if set(file) - {"name", "mimeType", "bytes", "uri"}:
                raise LegacyTaskMigrationError("Legacy file has unknown fields")
            raw_bytes, uri = file.get("bytes"), file.get("uri")
            if bool(raw_bytes) == bool(uri):
                raise LegacyTaskMigrationError(
                    "Legacy file requires exactly one of bytes or uri"
                )
            if raw_bytes:
                try:
                    base64.b64decode(
                        _text(raw_bytes, "base64 file bytes"), validate=True
                    )
                except (binascii.Error, ValueError) as error:
                    raise LegacyTaskMigrationError(
                        "Legacy file bytes are not valid base64"
                    ) from error
                body["raw"] = raw_bytes
            else:
                body["url"] = _text(uri, "file URI")
            for old, new in (("name", "filename"), ("mimeType", "mediaType")):
                value = _optional_text(file.get(old), f"file {old}")
                if value is not None:
                    body[new] = value
        else:
            raise LegacyTaskMigrationError("Legacy part has an unknown type or fields")
        result.append(ParseDict(body, Part()))
    return result


def _message(raw, *, task_id, context_id, position):
    entry = _object(raw, "message")
    if set(entry) - {"role", "parts", "metadata"} or entry.get("role") not in {
        "user",
        "agent",
    }:
        raise LegacyTaskMigrationError("Legacy message has an invalid role or fields")
    identity = json.dumps(
        [task_id, context_id, position, entry], sort_keys=True, allow_nan=False
    )
    return Message(
        message_id=str(uuid5(NAMESPACE_URL, identity)),
        task_id=task_id,
        context_id=context_id,
        role=Role.ROLE_USER if entry["role"] == "user" else Role.ROLE_AGENT,
        parts=_parts(entry.get("parts")),
        metadata=_object(entry["metadata"], "message metadata")
        if entry.get("metadata") is not None
        else {},
    )


def _metadata(raw, task_id):
    metadata = {} if raw is None else _object(raw, "task metadata")
    if set(metadata) & set(NATIVE_EXTENSION_URIS):
        raise LegacyTaskMigrationError(
            "Mixed native/draft task metadata requires a separate reviewed plan"
        )
    # These draft authentication fields are retained in the source archive;
    # they are not assertions on the newly encoded native Task.
    result = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "sender",
            "sender_verified",
            "sender_did",
            "sender_agent_id",
            "signature",
            "signatures",
            "signature_mode",
            "classical_signature",
            "pq_signature",
            "nonce",
            "timestamp",
            "a2a_verb",
            "reply_expected",
            "question_id",
            "causation_chain",
            "skill",
            "cancellation_receipt",
        }
    }
    skill = metadata.get("skill")
    if skill is not None:
        result.update(skill_hint_metadata(_text(skill, "skill hint")))
    verb = metadata.get("a2a_verb")
    if verb == "question":
        result.update(question_correlation_metadata("question", task_id))
    elif verb in {"message", "task"}:
        result.update(
            delivery_intent_metadata(
                "notification" if verb == "message" else "assignment"
            )
        )
    elif verb is not None:
        raise LegacyTaskMigrationError("Legacy task has an unknown delivery verb")
    if metadata.get("causation_chain") is not None:
        result.update(causation_metadata(metadata["causation_chain"]))
    return result


def convert_legacy_task(row: Mapping, *, ownership: ContextOwnership) -> Task:
    """Convert one drained snapshot with explicit, non-metadata provenance.

    Context IDs in the reviewed plan must already separate creator/recipient
    ownership. New Message/Artifact IDs are deterministic migration-assigned IDs;
    they do not claim an ID or signature that the draft sender never supplied.
    No timestamp is invented from the moment this migration runs.
    """
    if not isinstance(ownership, ContextOwnership):
        raise TypeError("Offline conversion requires reviewed ContextOwnership")
    scope = NativeRecipientScope(ownership.recipient_agent_id, ownership.tenant)
    _text(ownership.context_id, "reviewed context ID")
    _text(ownership.creator_agent_id, "reviewed creator identity")
    if not isinstance(row, Mapping):
        raise LegacyTaskMigrationError("Legacy task row must be an object")
    row = dict(row)
    task_id = _text(row.get("id"), "task ID")
    for key, expected in (
        ("creator_agent_id", ownership.creator_agent_id),
        ("recipient_agent_id", scope.recipient_id),
        ("tenant", scope.tenant),
    ):
        current = row.get(key)
        if current is not None and current != expected:
            raise LegacyTaskMigrationError(
                f"Reviewed provenance conflicts with durable {key}"
            )
    try:
        state = task_state_from_label(_text(row.get("status"), "task status"))
    except ValueError as error:
        raise LegacyTaskMigrationError("Legacy task has an unknown state") from error
    if state not in TERMINAL_STATES:
        raise LegacyTaskMigrationError(
            "Live legacy work must drain before the native cutover"
        )
    status = TaskStatus(state=state)
    if row.get("message") is not None:
        status.message.CopyFrom(
            _message(
                row["message"],
                task_id=task_id,
                context_id=ownership.context_id,
                position="status",
            )
        )
    history = row.get("history")
    artifacts = row.get("artifacts")
    if history is not None and not isinstance(history, list):
        raise LegacyTaskMigrationError("Legacy history must be an array or null")
    if artifacts is not None and not isinstance(artifacts, list):
        raise LegacyTaskMigrationError("Legacy artifacts must be an array or null")
    result = Task(
        id=task_id,
        context_id=ownership.context_id,
        status=status,
        metadata=_metadata(row.get("metadata"), task_id),
    )
    for index, message in enumerate(history or []):
        result.history.append(
            _message(
                message,
                task_id=task_id,
                context_id=ownership.context_id,
                position=f"history:{index}",
            )
        )
    artifact_positions = {}
    closed_artifacts = set()
    for index, raw in enumerate(artifacts or []):
        artifact = _object(raw, "artifact")
        if set(artifact) - {
            "name",
            "description",
            "parts",
            "metadata",
            "index",
            "append",
            "lastChunk",
        }:
            raise LegacyTaskMigrationError("Legacy artifact has unknown fields")
        for flag in ("append", "lastChunk"):
            if artifact.get(flag) is not None and not isinstance(artifact[flag], bool):
                raise LegacyTaskMigrationError(
                    f"Legacy artifact {flag} must be boolean or null"
                )
        ordinal = artifact.get("index", 0)
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise LegacyTaskMigrationError(
                "Legacy artifact index must be a nonnegative integer"
            )
        if artifact.get("append") is True:
            positions = artifact_positions.get(ordinal, [])
            if len(positions) != 1 or positions[0] in closed_artifacts:
                raise LegacyTaskMigrationError(
                    "Legacy artifact append target is missing, ambiguous, or closed"
                )
            position = positions[0]
            previous = result.artifacts[position]
            for field in ("name", "description"):
                value = _optional_text(artifact.get(field), f"artifact {field}")
                if value is not None and value != getattr(previous, field):
                    raise LegacyTaskMigrationError(
                        "Legacy artifact append changes its target identity"
                    )
            if (
                artifact.get("metadata") is not None
                and ParseDict(
                    _object(artifact["metadata"], "artifact metadata"), Struct()
                )
                != previous.metadata
            ):
                raise LegacyTaskMigrationError(
                    "Legacy artifact append changes target metadata"
                )
            previous.parts.extend(_parts(artifact.get("parts")))
            if artifact.get("lastChunk") is True:
                closed_artifacts.add(position)
            continue
        position = len(result.artifacts)
        artifact_positions.setdefault(ordinal, []).append(position)
        if artifact.get("lastChunk") is True:
            closed_artifacts.add(position)
        result.artifacts.append(
            Artifact(
                artifact_id=str(
                    uuid5(
                        NAMESPACE_URL,
                        json.dumps([task_id, ownership.context_id, "artifact", index]),
                    )
                ),
                name=_optional_text(artifact.get("name"), "artifact name") or "",
                description=_optional_text(
                    artifact.get("description"), "artifact description"
                )
                or "",
                parts=_parts(artifact.get("parts")),
                metadata=_object(artifact["metadata"], "artifact metadata")
                if artifact.get("metadata") is not None
                else {},
            )
        )
    if (
        state != TaskState.TASK_STATE_CANCELED
        and QUESTION_CORRELATION_URI in MessageToDict(result.metadata)
    ):
        if (
            not result.status.HasField("message")
            or result.status.message.role != Role.ROLE_AGENT
        ):
            raise LegacyTaskMigrationError(
                "Historical question has no attributable agent answer message"
            )
        result.status.message.metadata.update(
            correlated_answer_metadata(
                MessageToDict(result.metadata),
                MessageToDict(result.status.message.metadata),
            )
        )
    validate_proto_required_fields(result)
    return result
