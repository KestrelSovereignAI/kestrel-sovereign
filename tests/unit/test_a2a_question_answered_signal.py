"""``a2a.question_answered`` signal source (#1444 step 4).

Pins the registration shape, schema validation (including the default
injection that keeps the prompt template safe from KeyError per #1438
codex round 2), the 8 KiB reply-text overflow handling, and the dedupe
key shape that prevents subscription/startup-replay races from waking
two cognition turns for the same answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel_sdk.signals import SignalMode, Trust, Visibility
from kestrel_sovereign.signals.sources.a2a_question_answered import (
    PROMPT_TEMPLATE,
    REPLY_TEXT_INLINE_CAP_BYTES,
    SOURCE_NAME,
    build_a2a_question_answered_registration,
    build_signal_for_question_answered,
)


class TestRegistration:
    def test_registration_shape(self):
        reg = build_a2a_question_answered_registration()
        assert reg.name == SOURCE_NAME == "a2a.question_answered"
        assert reg.default_mode == SignalMode.COGNITION
        assert reg.trust == Trust.TRUSTED
        assert reg.allow_self_loops is False, (
            "Sender firing the signal to themselves only happens if they "
            "asked themselves a question; redundant guard against that."
        )
        # Sovereign decision on #1444: 90-day retention.
        assert reg.retention_days == 90, (
            f"retention_days must be 90 per the Sovereign decision on #1444. "
            f"Got {reg.retention_days}."
        )

    def test_registration_supplies_result_summary(self):
        """#1522: USER_VISIBLE alone makes ``signal_completed`` a
        metadata-only toast — the UI side channel has nothing to render.
        The source must set ``result_summary`` so the agent's response
        text (the COGNITION dispatch artifact) becomes the event body
        the chat pane appends live."""
        reg = build_a2a_question_answered_registration()
        assert reg.result_summary is not None, (
            "Source must opt into result_summary so the signal_completed "
            "event carries the resumed turn's response for the chat UI."
        )
        # The COGNITION artifact is the agent's own response string —
        # surface it verbatim.
        assert reg.result_summary("the answer is 4") == "the answer is 4"
        # Defensive: None / non-str bodies don't blow up the callback.
        assert reg.result_summary(None) == ""
        assert reg.result_summary(42) == "42"

    def test_prompt_template_exists_in_package(self):
        assert PROMPT_TEMPLATE.exists(), (
            f"Prompt template missing at {PROMPT_TEMPLATE}. Without it, "
            f"the dispatcher would FileNotFoundError on every cognition "
            f"wake — same class of bug as #1416."
        )
        body = PROMPT_TEMPLATE.read_text()
        # Must reference the placeholders we inject — without these the
        # template will KeyError at render time.
        for required in (
            "{payload[recipient_id]}",
            "{payload[dispatch_id]}",
            "{payload[original_question]}",
            "{payload[reply_text]}",
            "{payload[state]}",
        ):
            assert required in body, (
                f"Prompt template missing required placeholder: {required}"
            )


def native_payload(**changes):
    from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_text_message
    from kestrel_sovereign._vendor.a2a.types import Role
    from google.protobuf.json_format import MessageToDict

    return {
        "question_id": "question-1", "dispatch_id": "dispatch-1",
        "recipient_id": "did:test:peer", "original_question": "What is 2+2?",
        "reply_text": "4", "state": "completed",
        "request": MessageToDict(new_text_message("What is 2+2?", role=Role.ROLE_USER)),
        **changes,
    }


def render(payload):
    return PROMPT_TEMPLATE.read_text().format(
        source=SOURCE_NAME, target_agent="did:test:sender",
        arrived_at="2026-10-21T00:00:00Z", urgency="normal", payload=payload,
    )


@pytest.mark.parametrize("key", [
    "question_id", "dispatch_id", "recipient_id", "original_question",
    "reply_text", "state", "request",
])
def test_required_native_correlation_and_input(key):
    payload = native_payload()
    payload.pop(key)
    with pytest.raises(ValueError, match=key if key != "request" else "native Message"):
        build_a2a_question_answered_registration().schema(payload)


@pytest.mark.parametrize("key", ["question_id", "dispatch_id", "recipient_id"])
def test_identity_fields_cannot_be_empty(key):
    with pytest.raises(ValueError, match="concrete"):
        build_a2a_question_answered_registration().schema(native_payload(**{key: " "}))


@pytest.mark.parametrize("state", ["completed", "failed", "canceled", "rejected", "expired"])
def test_native_terminal_states_and_local_deadline(state):
    payload = build_a2a_question_answered_registration().schema(native_payload(state=state))
    assert payload["state"] == state
    assert payload["origin_session_id"] == ""
    assert not any(payload[key] for key in ("truncated", "late", "withdrawn", "requires_full_response"))
    assert "dispatch-1" in render(payload)


def test_nonterminal_state_is_not_an_answer():
    with pytest.raises(ValueError, match="terminal state"):
        build_a2a_question_answered_registration().schema(native_payload(state="working"))


def test_builder_uses_immutable_outbox_identity_and_sender_dispatch():
    payload = native_payload(origin_session_id="session-1")
    sig = build_signal_for_question_answered(
        event_id="outbox-event-1", payload=payload, target_agent="did:test:sender",
    )
    assert sig.id == sig.dedupe_key == "outbox-event-1"
    assert sig.source == SOURCE_NAME and sig.target_agent == "did:test:sender"
    assert sig.visibility == Visibility.USER_VISIBLE and sig.session_id == "session-1"
    assert sig.caller == "did:test:peer"
    assert sig.payload == build_a2a_question_answered_registration().schema(dict(payload))
    assert sig.payload is not payload and "truncated" not in payload


@pytest.mark.parametrize("character", ["A", "😀"])
def test_oversized_reply_is_bounded_utf8_and_recoverable_by_dispatch(character):
    sig = build_signal_for_question_answered(
        event_id="large-answer", payload=native_payload(reply_text=character * REPLY_TEXT_INLINE_CAP_BYTES),
        target_agent="did:test:sender",
    )
    # ASCII must exceed the cap too; the first exact-cap payload is unchanged.
    if character == "A":
        assert not sig.payload["truncated"]
        sig = build_signal_for_question_answered(
            event_id="large-answer", payload=native_payload(reply_text=character * (REPLY_TEXT_INLINE_CAP_BYTES + 1)),
            target_agent="did:test:sender",
        )
    assert sig.payload["truncated"]
    assert len(sig.payload["reply_text"].encode("utf-8")) <= REPLY_TEXT_INLINE_CAP_BYTES
    assert 'get_peer_task_result("dispatch-1")' in sig.payload["reply_text"]


def test_late_answer_and_deadline_use_distinct_durable_event_identities():
    expired = build_signal_for_question_answered(
        event_id="deadline-event", payload=native_payload(state="expired", reply_text=""),
        target_agent="did:test:sender",
    )
    late = build_signal_for_question_answered(
        event_id="late-answer-event", payload=native_payload(late=True), target_agent="did:test:sender",
    )
    assert expired.dedupe_key != late.dedupe_key
    assert expired.payload["dispatch_id"] == late.payload["dispatch_id"]
    assert late.payload["late"] and not expired.payload["late"]
    assert "Late answer: `True`" in render(late.payload)


@pytest.mark.parametrize("changes,reason", [
    ({"state": "expired", "late": True}, "late"),
    ({"late": "yes"}, "bool"),
    ({"withdrawn": True}, "withdrawn"),
    ({"state": "expired", "withdrawn": "yes"}, "bool"),
    ({"truncated": "yes"}, "bool"),
    ({"requires_full_response": "yes"}, "boolean"),
    ({"origin_session_id": 42}, "string"),
])
def test_optional_flags_reject_invalid_evidence(changes, reason):
    with pytest.raises(ValueError, match=reason):
        build_a2a_question_answered_registration().schema(native_payload(**changes))


def test_expiry_prompt_preserves_withdrawal_uncertainty_and_structured_results():
    for withdrawn in (False, True):
        sig = build_signal_for_question_answered(
            event_id="deadline", payload=native_payload(state="expired", withdrawn=withdrawn, requires_full_response=True),
            target_agent="did:test:sender",
        )
        body = render(sig.payload)
        assert f"Confirmed withdrawal: `{withdrawn}`" in body
        assert "Only `withdrawn=True` confirms cancellation" in body
        assert "Do not resend an uncertain dispatch" in body
        assert "structured data, files, or artifacts: fetch it" in body


def test_causation_uses_the_declared_native_extension():
    frame = {"agent_id": "did:test:sender", "source": "user", "signal_id": "signal-1",
             "depth": 0, "emitted_at": "2026-10-21T00:00:00Z"}
    sig = build_signal_for_question_answered(
        event_id="answer", payload=native_payload(causation={"version": 1, "chain": [frame]}),
        target_agent="did:test:sender",
    )
    assert len(sig.causation_chain) == 1
    assert sig.causation_chain[0].agent_id == frame["agent_id"]
    assert "causation" not in sig.payload


def test_template_path_inside_package():
    package_root = Path(__import__("kestrel_sovereign").__file__).resolve().parent
    assert PROMPT_TEMPLATE.is_relative_to(package_root)
