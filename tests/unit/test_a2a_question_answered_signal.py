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
            "{payload[recipient]}",
            "{payload[task_id]}",
            "{payload[original_question]}",
            "{payload[reply_text]}",
            "{payload[state]}",
        ):
            assert required in body, (
                f"Prompt template missing required placeholder: {required}"
            )


class TestSchema:
    def _reg(self):
        return build_a2a_question_answered_registration()

    def test_required_keys_enforced(self):
        reg = self._reg()
        with pytest.raises(ValueError, match="missing required key"):
            reg.schema({})

    def test_state_must_be_enum(self):
        reg = self._reg()
        with pytest.raises(ValueError, match="state must be one of"):
            reg.schema({
                "task_id": "t", "recipient": "r",
                "original_question": "q", "reply_text": "r",
                "state": "in_progress",  # not terminal
            })

    def test_state_accepts_each_valid_value(self):
        reg = self._reg()
        for state in ("completed", "failed", "canceled", "expired"):
            out = reg.schema({
                "task_id": "t", "recipient": "r",
                "original_question": "q", "reply_text": "r",
                "state": state,
            })
            assert out["state"] == state

    def test_schema_injects_defaults_for_optional_fields(self):
        """Per #1438 codex round 2 — legacy callers building a payload
        with only required keys must still render through the prompt
        template without KeyError. Schema injects defaults at validation
        time."""
        reg = self._reg()
        out = reg.schema({
            "task_id": "t", "recipient": "r",
            "original_question": "q", "reply_text": "r",
            "state": "completed",
        })
        assert out["origin_session_id"] == ""
        assert out["truncated"] is False
        # Smoke-test the actual prompt format against the schema's
        # enriched payload — if anything we inject is missing the
        # template needs, this raises.
        template = PROMPT_TEMPLATE.read_text()
        template.format(
            source=SOURCE_NAME,
            target_agent="did:test:agent",
            arrived_at="2026-05-29T16:00:00Z",
            urgency="normal",
            payload=out,
        )


class TestSignalBuilder:
    def test_basic_signal_shape(self):
        sig = build_signal_for_question_answered(
            task_id="t-1", recipient="Meridian",
            original_question="What is 2+2?",
            reply_text="4", state="completed",
            target_agent="did:test:sender",
        )
        assert sig.source == SOURCE_NAME
        assert sig.target_agent == "did:test:sender"
        assert sig.payload["task_id"] == "t-1"
        assert sig.payload["recipient"] == "Meridian"
        assert sig.payload["reply_text"] == "4"
        assert sig.payload["state"] == "completed"
        assert sig.payload["truncated"] is False
        # Dedupe key prevents subscription + startup-replay racing for
        # the same answer from waking two cognition turns.
        assert sig.dedupe_key == "t-1:answered"

    def test_oversized_reply_is_truncated_with_hint(self):
        """Reply > 8 KiB inline cap must clip with a clear hint pointing
        at get_a2a_task(task_id) for the full body. Sovereign decision
        on #1444 question 2."""
        big_reply = "A" * (REPLY_TEXT_INLINE_CAP_BYTES + 1024)
        sig = build_signal_for_question_answered(
            task_id="t-big", recipient="Meridian",
            original_question="long?",
            reply_text=big_reply, state="completed",
            target_agent="did:test:sender",
        )
        assert sig.payload["truncated"] is True, (
            "Oversized reply must set truncated=True so the prompt "
            "template knows to render the overflow branch."
        )
        rendered = sig.payload["reply_text"]
        assert len(rendered.encode("utf-8")) <= REPLY_TEXT_INLINE_CAP_BYTES, (
            f"Truncated reply must fit under {REPLY_TEXT_INLINE_CAP_BYTES} "
            f"bytes; got {len(rendered.encode('utf-8'))}."
        )
        assert "get_peer_task_result(\"Meridian\", \"t-big\")" in rendered, (
            "Overflow hint must cite get_peer_task_result with BOTH "
            "recipient and task_id so the resumed turn can fetch the "
            "full body through the host proxy. Citing only task_id (or "
            "a nonexistent get_a2a_task) makes the truncated reply "
            "unrecoverable — codex round 2 P2b on PR #1453."
        )

    def test_signal_is_user_visible_for_live_chat_render(self):
        """#1522: the wake must be USER_VISIBLE (not INTERNAL) so the
        dispatcher emits a ``signal_completed`` SSE event after the
        resumed turn logs. INTERNAL signals never reach the UI side
        channel (see SignalDispatcher._log_safe), which is exactly the
        bug — the A2A wake response persisted but the open chat tab
        never rendered it live."""
        sig = build_signal_for_question_answered(
            task_id="t-1", recipient="Meridian",
            original_question="What is 2+2?",
            reply_text="4", state="completed",
            target_agent="did:test:sender",
        )
        assert sig.visibility == Visibility.USER_VISIBLE, (
            "A reply-capable A2A wake must be USER_VISIBLE so the open "
            "chat renders the resumed turn live (#1522). INTERNAL would "
            "log silently and the UI would only see it on a manual "
            "refresh."
        )

    def test_causation_chain_threaded_through(self):
        """A→B→A→B chains must hit the dispatcher's depth-2 cycle cap.
        The supervisor rehydrates the chain from the terminal task's
        metadata and threads it here; we just verify it survives."""
        chain = [{"agent": "A", "source": "user", "frame_id": "f1"}]
        sig = build_signal_for_question_answered(
            task_id="t-c", recipient="B",
            original_question="?", reply_text=".", state="completed",
            target_agent="did:test:A",
            causation_chain=chain,
        )
        assert sig.causation_chain == chain


class TestPackagedTemplate:
    """Pin that the prompt template ships inside the package wheel
    (same lesson as #1416). If this regresses, deployed agents
    FileNotFoundError on every a2a.question_answered wake."""

    def test_template_path_inside_package(self):
        package_root = Path(
            __import__("kestrel_sovereign").__file__
        ).resolve().parent
        assert str(PROMPT_TEMPLATE).startswith(str(package_root)), (
            f"Prompt template {PROMPT_TEMPLATE} must live inside the "
            f"kestrel_sovereign package so it ships with the wheel. "
            f"Package root: {package_root}."
        )
