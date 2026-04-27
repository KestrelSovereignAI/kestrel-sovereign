"""Codex adapter continuation protocol tests (#808 / #806).

Two layers:
- Pure-function tests for ``_compute_request_signature`` and ``_plan_continuation``.
- End-to-end tests with a fake ``httpx.AsyncClient`` that drive two synthetic
  turns through ``CodexAdapter.get_response`` and assert:
    * Turn 1 sends full input, no ``previous_response_id``, writes a cursor.
    * Turn 2 (same session_id, same tools/instructions) sends only the
      delta input + ``previous_response_id``, then refreshes the cursor.
    * Tool/instruction drift between turns drops continuation.
    * No session_id ⇒ behavior identical to pre-#808 (no body keys added).
"""

import base64
import json
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from kestrel_sovereign.llm.codex_adapter import (
    CodexAdapter,
    _compute_request_signature,
    _plan_continuation,
)
from kestrel_sovereign.llm.continuation_store import (
    ContinuationCursor,
    InMemoryContinuationStore,
)


def _fake_token() -> str:
    """Build a minimal JWT carrying a chatgpt_account_id claim."""
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}}
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(b"sig").rstrip(b"=").decode()
    return f"{header}.{payload}.{sig}"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestComputeRequestSignature:
    def test_stable_for_identical_inputs(self):
        a = _compute_request_signature("hello", [{"name": "tool_a"}])
        b = _compute_request_signature("hello", [{"name": "tool_a"}])
        assert a == b

    def test_changes_when_instructions_change(self):
        a = _compute_request_signature("v1", [{"name": "t"}])
        b = _compute_request_signature("v2", [{"name": "t"}])
        assert a != b

    def test_changes_when_tools_change(self):
        a = _compute_request_signature("x", [{"name": "tool_a"}])
        b = _compute_request_signature("x", [{"name": "tool_b"}])
        assert a != b

    def test_none_inputs_produce_stable_hash(self):
        a = _compute_request_signature(None, None)
        b = _compute_request_signature(None, None)
        assert a == b

    def test_short_hash_length(self):
        # Truncated to 16 hex chars — long enough to collision-resist within a
        # single conversation, short enough not to bloat logs.
        sig = _compute_request_signature("x", None)
        assert len(sig) == 16


class TestPlanContinuation:
    def test_no_cursor_means_full_input(self):
        prev, slice_start = _plan_continuation(
            cursor=None, messages_count=5, signature="sig",
        )
        assert prev is None
        assert slice_start is None

    def test_signature_match_emits_delta(self):
        cursor = ContinuationCursor("resp_1", last_message_count=3, last_request_signature="sig")
        prev, slice_start = _plan_continuation(
            cursor=cursor, messages_count=5, signature="sig",
        )
        assert prev == "resp_1"
        assert slice_start == 3

    def test_signature_mismatch_drops_continuation(self):
        cursor = ContinuationCursor("resp_1", last_message_count=3, last_request_signature="old")
        prev, slice_start = _plan_continuation(
            cursor=cursor, messages_count=5, signature="new",
        )
        assert prev is None
        assert slice_start is None

    def test_no_new_messages_falls_back_to_full(self):
        # Defensive: empty input slice would mean either a duplicate turn or a
        # bug; resubmit the full context rather than send an empty input array.
        cursor = ContinuationCursor("resp_1", last_message_count=5, last_request_signature="sig")
        prev, slice_start = _plan_continuation(
            cursor=cursor, messages_count=5, signature="sig",
        )
        assert prev is None
        assert slice_start is None


# ---------------------------------------------------------------------------
# End-to-end tests with a fake httpx.AsyncClient
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    """Minimal stand-in for httpx.Response on a streaming POST."""

    def __init__(self, status_code: int, sse_lines: List[str]):
        self.status_code = status_code
        self._sse_lines = sse_lines
        self.text = ""

    async def aiter_lines(self):
        for line in self._sse_lines:
            yield line

    async def aread(self) -> bytes:
        return b""


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse, captured_body: List[Dict[str, Any]]):
        self._response = response
        self._captured = captured_body

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeAsyncClient:
    """Captures the JSON body of the POST and returns canned SSE events."""

    def __init__(self, sse_lines: List[str], captured_bodies: List[Dict[str, Any]]):
        self._sse_lines = sse_lines
        self._captured_bodies = captured_bodies

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def stream(self, method: str, url: str, *, headers, json):
        # Capture the request body for later assertions.
        self._captured_bodies.append(json)
        return _FakeStreamContext(
            _FakeStreamResponse(200, self._sse_lines), self._captured_bodies,
        )


def _sse(events: List[Dict[str, Any]]) -> List[str]:
    """Format a list of events as SSE ``data:`` lines."""
    return [f"data: {json.dumps(e)}" for e in events]


def _completed_event(response_id: str) -> Dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        },
    }


def _patch_httpx_with(captured: List[Dict[str, Any]], sse_lines: List[str]):
    fake = _FakeAsyncClient(sse_lines, captured)

    class _Factory:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return fake

        async def __aexit__(self, *a):
            return None

    return patch(
        "kestrel_sovereign.llm.codex_adapter.httpx.AsyncClient", _Factory,
    )


@pytest.mark.asyncio
class TestCodexContinuationE2E:
    async def test_no_session_id_means_no_continuation_keys(self):
        captured: List[Dict[str, Any]] = []
        sse = _sse(
            [
                {"type": "response.output_text.delta", "delta": "ok"},
                _completed_event("resp_1"),
            ]
        )
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        with _patch_httpx_with(captured, sse):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
            )

        assert "previous_response_id" not in captured[0]
        # Store stays empty — without session_id we don't track anything.
        assert len(adapter._continuation_store) == 0

    async def test_first_turn_writes_cursor(self):
        captured: List[Dict[str, Any]] = []
        sse = _sse(
            [
                {"type": "response.output_text.delta", "delta": "ok"},
                _completed_event("resp_1"),
            ]
        )
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

        with _patch_httpx_with(captured, sse):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.4",
                messages=messages,
                session_id="conv-A",
            )

        # First turn: full input, no previous_response_id.
        assert "previous_response_id" not in captured[0]
        assert len(captured[0]["input"]) == 1  # the user message

        cursor = adapter._continuation_store.get("openai_plan", "conv-A")
        assert cursor is not None
        assert cursor.last_response_id == "resp_1"
        # Watermark is the *full input message count* (system was extracted to
        # instructions, so the user-only count is 1).
        assert cursor.last_message_count == 1

    async def test_second_turn_sends_delta_with_previous_response_id(self):
        captured: List[Dict[str, Any]] = []
        sse = _sse(
            [
                {"type": "response.output_text.delta", "delta": "ack"},
                _completed_event("resp_1"),
            ]
        )
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        # Turn 1
        with _patch_httpx_with(captured, sse):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
                session_id="conv-B",
            )

        # Turn 2: two new user messages; same session_id, same tools (None).
        # (Tool-role conversion is exercised separately in
        # test_codex_responses_format.py — here we isolate continuation slicing.)
        captured.clear()
        sse2 = _sse(
            [
                {"type": "response.output_text.delta", "delta": "done"},
                _completed_event("resp_2"),
            ]
        )
        with _patch_httpx_with(captured, sse2):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                    {"role": "user", "content": "more context"},
                    {"role": "user", "content": "follow-up"},
                ],
                session_id="conv-B",
            )

        body = captured[0]
        assert body["previous_response_id"] == "resp_1"
        # Delta = messages[1:] from the user-input slice (system was stripped).
        # Watermark on the cursor was 1 (user-only count after turn 1); turn 2
        # presents 3 user-side messages, so the delta is the last 2.
        assert len(body["input"]) == 2
        assert body["input"][0]["content"] == "more context"
        assert body["input"][1]["content"] == "follow-up"

        # Cursor refreshed.
        cursor = adapter._continuation_store.get("openai_plan", "conv-B")
        assert cursor.last_response_id == "resp_2"
        assert cursor.last_message_count == 3

    async def test_signature_drift_drops_continuation(self):
        captured: List[Dict[str, Any]] = []
        sse = _sse(
            [
                {"type": "response.output_text.delta", "delta": "x"},
                _completed_event("resp_1"),
            ]
        )
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        # Turn 1 with one toolset.
        with _patch_httpx_with(captured, sse):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "tool_a", "description": "", "parameters": {},
                    },
                }],
                session_id="conv-C",
            )

        # Turn 2 with a *different* toolset — signature must mismatch and the
        # adapter must drop continuation, sending full input + no previous_response_id.
        captured.clear()
        sse2 = _sse(
            [
                {"type": "response.output_text.delta", "delta": "y"},
                _completed_event("resp_2"),
            ]
        )
        with _patch_httpx_with(captured, sse2):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                    {"role": "user", "content": "again"},
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "tool_b", "description": "", "parameters": {},
                    },
                }],
                session_id="conv-C",
            )

        body = captured[0]
        assert "previous_response_id" not in body
        assert len(body["input"]) == 2  # full input restored
