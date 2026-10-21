"""Versioned native A2A metadata extensions; causation never grants authority."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kestrel_sdk.signals import CausationFrame

EXTENSION_DOCUMENT = (
    "https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/"
    "docs/architecture/NATIVE_MCP_A2A.md"
)
SIGNED_SENDER_URI = EXTENSION_DOCUMENT + "#signed-sender-v1"
RECIPIENT_POLICY_URI = EXTENSION_DOCUMENT + "#recipient-policy-v1"
QUESTION_CORRELATION_URI = EXTENSION_DOCUMENT + "#question-correlation-v1"
CAUSATION_URI = EXTENSION_DOCUMENT + "#causation-v1"
SKILL_HINT_URI = EXTENSION_DOCUMENT + "#skill-hint-v1"
DELIVERY_INTENT_URI = EXTENSION_DOCUMENT + "#delivery-intent-v1"
NATIVE_EXTENSION_URIS = (
    SIGNED_SENDER_URI,
    RECIPIENT_POLICY_URI,
    QUESTION_CORRELATION_URI,
    CAUSATION_URI,
    SKILL_HINT_URI,
    DELIVERY_INTENT_URI,
)


def _payload(metadata: Mapping[str, Any], uri: str, keys: set[str]) -> dict | None:
    if uri not in metadata:
        return None
    payload = metadata[uri]
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError(f"Invalid native A2A extension payload: {uri}")
    # Protobuf Struct represents JSON numbers as doubles. 1 and 1.0 are the
    # same supported version; booleans and string coercion are not versions.
    version = payload["version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, (int, float))
        or version != 1
    ):
        raise ValueError(f"Unsupported native A2A extension version: {uri}")
    return payload


@dataclass(frozen=True)
class QuestionCorrelation:
    """Correlation data, subject to the sender/recipient's durable ledger."""

    kind: str
    question_id: str


def question_correlation_metadata(kind: str, question_id: str) -> dict[str, Any]:
    metadata = {
        QUESTION_CORRELATION_URI: {
            "version": 1,
            "kind": kind,
            "question_id": question_id,
        }
    }
    parse_question_correlation(metadata)
    return metadata


def correlated_answer_metadata(
    original: Mapping[str, Any], reply: Mapping[str, Any]
) -> dict[str, Any]:
    question = parse_question_correlation(original)
    answer = parse_question_correlation(reply)
    if question is None:
        if answer is not None:
            raise ValueError(
                "An answer cannot introduce correlation absent from its task"
            )
        return dict(reply)
    if question.kind != "question" or (
        answer is not None
        and answer != QuestionCorrelation("answer", question.question_id)
    ):
        raise ValueError("Answer correlation disagrees with the original question")
    return dict(reply) | question_correlation_metadata("answer", question.question_id)


def require_metadata_extensions(
    metadata: Mapping[str, Any], activated: frozenset[str]
) -> None:
    """URI metadata cannot activate its own optional wire extension."""
    if not isinstance(activated, frozenset):
        raise TypeError("Activated native extensions must be immutable")
    if any(
        uri in metadata and uri not in activated
        for uri in (
            QUESTION_CORRELATION_URI,
            CAUSATION_URI,
            SKILL_HINT_URI,
            DELIVERY_INTENT_URI,
        )
    ):
        raise ValueError("Native metadata uses an extension that was not activated")


def parse_skill_hint(metadata: Mapping[str, Any]) -> str | None:
    """Read an advisory published skill; it grants no tool-execution authority."""
    payload = _payload(metadata, SKILL_HINT_URI, {"version", "skill_id"})
    if payload is None:
        return None
    skill_id = payload["skill_id"]
    if not isinstance(skill_id, str) or not skill_id.strip():
        raise ValueError("Native A2A skill hint requires a concrete skill_id")
    return skill_id


def skill_hint_metadata(skill_id: str) -> dict[str, Any]:
    metadata = {SKILL_HINT_URI: {"version": 1, "skill_id": skill_id}}
    parse_skill_hint(metadata)
    return metadata


def parse_delivery_intent(metadata: Mapping[str, Any]) -> str | None:
    payload = _payload(metadata, DELIVERY_INTENT_URI, {"version", "intent"})
    if payload is None:
        return None
    if payload["intent"] not in ("notification", "assignment"):
        raise ValueError("Native delivery intent must be notification or assignment")
    if QUESTION_CORRELATION_URI in metadata:
        raise ValueError("Question correlation already defines its own reply intent")
    return payload["intent"]


def delivery_intent_metadata(intent: str) -> dict[str, Any]:
    metadata = {DELIVERY_INTENT_URI: {"version": 1, "intent": intent}}
    parse_delivery_intent(metadata)
    return metadata


def parse_question_correlation(
    metadata: Mapping[str, Any],
) -> QuestionCorrelation | None:
    """Read a namespaced correlation claim; this does not authorize a reply."""
    payload = _payload(
        metadata, QUESTION_CORRELATION_URI, {"version", "kind", "question_id"}
    )
    if payload is None:
        return None
    if not isinstance(payload["kind"], str) or payload["kind"] not in {
        "question",
        "answer",
    }:
        raise ValueError(
            "Native A2A question correlation requires question or answer kind"
        )
    question_id = payload["question_id"]
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError("Native A2A question correlation requires a question_id")
    return QuestionCorrelation(payload["kind"], question_id)


def serialize_chain_for_metadata(
    chain: Sequence[CausationFrame],
) -> list[dict[str, Any]]:
    """Serialize the canonical frames used by local causation and persistence."""
    frames = [
        {
            "agent_id": frame.agent_id,
            "source": frame.source,
            "signal_id": frame.signal_id,
            "turn_id": frame.turn_id,
            "depth": frame.depth,
            "emitted_at": frame.emitted_at.isoformat(),
        }
        for frame in chain
    ]
    causation_metadata(frames)
    return frames


def causation_metadata(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Place serialized causation under its native versioned extension URI."""
    metadata = {CAUSATION_URI: {"version": 1, "chain": frames}}
    deserialize_causation_chain(metadata)
    return metadata


def deserialize_causation_chain(metadata: Mapping[str, Any]) -> list[CausationFrame]:
    """Reject malformed declared causation instead of silently resetting it."""
    payload = _payload(metadata, CAUSATION_URI, {"version", "chain"})
    if payload is None:
        return []
    if not isinstance(payload["chain"], list):
        raise ValueError("Native A2A causation chain must be an array")
    frames = []
    required = {"agent_id", "source", "signal_id", "depth", "emitted_at"}
    for entry in payload["chain"]:
        if (
            not isinstance(entry, dict)
            or not required <= set(entry)
            or set(entry) - required - {"turn_id"}
        ):
            raise ValueError("Native A2A causation frame has invalid fields")
        for key in ("agent_id", "source", "signal_id", "emitted_at"):
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise ValueError(f"Native A2A causation frame requires {key}")
        turn_id = entry.get("turn_id")
        if turn_id is not None and (
            not isinstance(turn_id, str) or not turn_id.strip()
        ):
            raise ValueError(
                "Native A2A causation turn_id must be a non-empty string or null"
            )
        depth = entry["depth"]
        if (
            isinstance(depth, bool)
            or not isinstance(depth, (int, float))
            or not math.isfinite(depth)
            or depth < 0
            or int(depth) != depth
        ):
            raise ValueError(
                "Native A2A causation depth must be a non-negative integer"
            )
        emitted_at = datetime.fromisoformat(entry["emitted_at"])
        if emitted_at.utcoffset() is None:
            raise ValueError("Native A2A causation emitted_at requires a timezone")
        frames.append(
            CausationFrame(
                agent_id=entry["agent_id"],
                source=entry["source"],
                signal_id=entry["signal_id"],
                turn_id=turn_id,
                depth=int(depth),
                emitted_at=emitted_at,
            )
        )
    return frames


@dataclass(frozen=True)
class NativeMessageMetadata:
    correlation: QuestionCorrelation | None
    skill_hint: str | None
    delivery_intent: str | None
    causation_chain: tuple[CausationFrame, ...]


def parse_message_metadata(metadata: Mapping[str, Any]) -> NativeMessageMetadata:
    """One extension contract for new native messages, before any task admission."""
    correlation = parse_question_correlation(metadata)
    if correlation is not None and correlation.kind != "question":
        raise ValueError("Question answers must use their original task response")
    return NativeMessageMetadata(
        correlation,
        parse_skill_hint(metadata),
        parse_delivery_intent(metadata),
        tuple(deserialize_causation_chain(metadata)),
    )
