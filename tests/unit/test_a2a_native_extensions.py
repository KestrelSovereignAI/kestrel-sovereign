"""Public native ProtoJSON carries versioned Kestrel extension payloads."""

from datetime import UTC, datetime

import pytest
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import SendMessageRequest
from google.protobuf.json_format import MessageToDict
from kestrel_sdk.signals import CausationFrame

from kestrel_sovereign.a2a.extensions import (
    CAUSATION_URI,
    QUESTION_CORRELATION_URI,
    causation_metadata,
    deserialize_causation_chain,
    parse_question_correlation,
    serialize_chain_for_metadata,
)


def frame():
    return CausationFrame(
        agent_id="did:kestrel:sender",
        source="cron.followup",
        signal_id="signal-one",
        turn_id=None,
        depth=2,
        emitted_at=datetime(2026, 10, 20, tzinfo=UTC),
    )


def native_metadata(metadata):
    return MessageToDict(
        SendMessageRequest(
            message=new_text_message("request"), metadata=metadata
        ).metadata
    )


def test_causation_round_trip_accepts_integral_struct_numbers_and_preserves_frame_identity():
    original = frame()
    serialized = serialize_chain_for_metadata([original])
    wire = native_metadata(causation_metadata(serialized))
    assert wire[CAUSATION_URI]["chain"][0]["depth"] == 2.0
    assert deserialize_causation_chain(wire) == [original]
    assert serialized[0]["turn_id"] is None


@pytest.mark.parametrize("kind", ["question", "answer"])
def test_question_correlation_is_namespaced_and_survives_native_struct_round_trip(kind):
    wire = native_metadata(
        {
            QUESTION_CORRELATION_URI: {
                "version": 1,
                "kind": kind,
                "question_id": "question-one",
            }
        }
    )
    parsed = parse_question_correlation(wire)
    assert parsed.kind == kind and parsed.question_id == "question-one"


@pytest.mark.parametrize("replacement", [True, "1", 2, None])
def test_declared_extension_versions_cannot_be_coerced_or_ignored(replacement):
    metadata = causation_metadata([])
    metadata[CAUSATION_URI]["version"] = replacement
    with pytest.raises(ValueError, match="version"):
        deserialize_causation_chain(native_metadata(metadata))


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", []),
        ("kind", {}),
        ("kind", "task"),
        ("question_id", ""),
        ("question_id", " "),
        ("question_id", False),
    ],
)
def test_invalid_question_claims_fail_explicitly(field, value):
    payload = {
        "version": 1,
        "kind": "question",
        "question_id": "question-one",
        field: value,
    }
    with pytest.raises(ValueError):
        parse_question_correlation(native_metadata({QUESTION_CORRELATION_URI: payload}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("depth", True),
        ("depth", -1),
        ("depth", 1.5),
        ("depth", "2"),
        ("agent_id", ""),
        ("source", None),
        ("turn_id", 2),
        ("emitted_at", "2026-10-20T00:00:00"),
        ("emitted_at", "not-a-time"),
    ],
)
def test_malformed_declared_causation_is_not_dropped_into_a_fresh_chain(field, value):
    metadata = causation_metadata(serialize_chain_for_metadata([frame()]))
    metadata[CAUSATION_URI]["chain"][0][field] = value
    with pytest.raises(ValueError):
        deserialize_causation_chain(native_metadata(metadata))


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"version": 1},
        {"version": 1, "chain": "wrong"},
        {"version": 1, "chain": [], "unknown": 2},
    ],
)
def test_declared_causation_has_one_explicit_payload_shape(payload):
    with pytest.raises(ValueError):
        deserialize_causation_chain(native_metadata({CAUSATION_URI: payload}))


@pytest.mark.parametrize("skill", ["ask", "workflow.assign"])
def test_native_skill_hint_is_advisory_versioned_metadata(skill):
    from kestrel_sovereign.a2a.extensions import (
        SKILL_HINT_URI,
        parse_skill_hint,
        skill_hint_metadata,
    )

    metadata = skill_hint_metadata(skill)
    assert metadata == {SKILL_HINT_URI: {"version": 1, "skill_id": skill}}
    assert parse_skill_hint(metadata) == skill
    assert parse_skill_hint({"skill": skill}) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"version": True, "skill_id": "ask"},
        {"version": 2, "skill_id": "ask"},
        {"version": 1, "skill_id": ""},
        {"version": 1, "skill_id": False},
        {"version": 1, "skill_id": "ask", "authority": "operator"},
    ],
)
def test_invalid_native_skill_hint_cannot_claim_behavior_or_authority(payload):
    from kestrel_sovereign.a2a.extensions import SKILL_HINT_URI, parse_skill_hint

    with pytest.raises(ValueError):
        parse_skill_hint({SKILL_HINT_URI: payload})
