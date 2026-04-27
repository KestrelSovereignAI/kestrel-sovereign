"""Codex adapter cursor-tracking tests.

#841 reduced this file's scope. The original #808 design sent
``previous_response_id`` plus delta input to preserve encrypted reasoning
across turns — but the live ChatGPT-backend Responses API rejects
``previous_response_id`` (caught by the integration tests in
``tests/integration/test_codex_real.py``). The wire-side continuation
mechanism was removed; the cursor is still written after each successful
response so that any future per-session diagnostics or alternative
continuation mechanism (e.g. reasoning-item resubmission) has the data.

What's tested here:
- ``_compute_request_signature`` is stable and discriminating.
- End-to-end: cursor is recorded with ``last_response_id`` from the live
  ``response.completed`` event when ``session_id`` is provided; nothing is
  recorded when ``session_id`` is omitted.
- The request body never carries ``previous_response_id`` regardless of
  whether ``session_id`` is provided.
"""

import base64
import json
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from kestrel_sovereign.llm.codex_adapter import (
    CodexAdapter,
    _compute_request_signature,
)
from kestrel_sovereign.llm.continuation_store import InMemoryContinuationStore


def _fake_token() -> str:
    """Build a minimal JWT carrying a chatgpt_account_id claim."""
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}}
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(b"fake-sig").rstrip(b"=").decode()
    return f"{header}.{payload}.{sig}"


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
        sig = _compute_request_signature("x", None)
        assert len(sig) == 16


# ---------------------------------------------------------------------------
# End-to-end tests with a fake httpx.AsyncClient
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
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
    def __init__(self, response: _FakeStreamResponse):
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeAsyncClient:
    def __init__(self, sse_lines: List[str], captured_bodies: List[Dict[str, Any]]):
        self._sse_lines = sse_lines
        self._captured = captured_bodies

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def stream(self, method: str, url: str, *, headers, json):
        self._captured.append(json)
        return _FakeStreamContext(_FakeStreamResponse(200, self._sse_lines))


def _sse(events: List[Dict[str, Any]]) -> List[str]:
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
class TestCodexCursorRecordingE2E:
    async def test_no_session_id_no_cursor_no_previous_response_id(self):
        captured: List[Dict[str, Any]] = []
        sse = _sse([
            {"type": "response.output_text.delta", "delta": "ok"},
            _completed_event("resp_1"),
        ])
        store = InMemoryContinuationStore()
        adapter = CodexAdapter(continuation_store=store)

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
        # Without session_id, no cursor is written.
        assert len(store) == 0

    async def test_session_id_writes_cursor_but_no_previous_response_id(self):
        captured: List[Dict[str, Any]] = []
        sse = _sse([
            {"type": "response.output_text.delta", "delta": "ok"},
            _completed_event("resp_1"),
        ])
        store = InMemoryContinuationStore()
        adapter = CodexAdapter(continuation_store=store)

        with _patch_httpx_with(captured, sse):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
                session_id="conv-A",
            )

        # Cursor is written so future tooling has access to the response_id.
        cursor = store.get("openai_plan", "conv-A")
        assert cursor is not None
        assert cursor.last_response_id == "resp_1"
        assert cursor.last_message_count == 1

        # But the wire never carried previous_response_id — the ChatGPT
        # backend rejects it (#841).
        assert "previous_response_id" not in captured[0]

    async def test_constructor_uses_supplied_store_not_internal_one(self):
        # Regression for the #841 truthiness bug: an *empty* caller-supplied
        # store was silently replaced with a fresh internal one because
        # ``InMemoryContinuationStore.__len__`` returned 0 → falsy under the
        # old ``or`` default.
        captured: List[Dict[str, Any]] = []
        sse = _sse([_completed_event("resp_1")])
        external_store = InMemoryContinuationStore()
        adapter = CodexAdapter(continuation_store=external_store)
        assert adapter._continuation_store is external_store

        with _patch_httpx_with(captured, sse):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
                session_id="conv-X",
            )

        # The cursor must land in the *external* store, not a hidden internal
        # one. Pre-fix, this assertion failed: external_store stayed empty
        # while the adapter wrote to its own store.
        assert external_store.get("openai_plan", "conv-X") is not None

    async def test_second_turn_sends_full_input_no_previous_response_id(self):
        # Continuation is disabled at the wire — even with a matching cursor,
        # turn 2 sends the full input list and no previous_response_id.
        captured: List[Dict[str, Any]] = []
        sse1 = _sse([_completed_event("resp_1")])
        sse2 = _sse([_completed_event("resp_2")])
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        with _patch_httpx_with(captured, sse1):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
                session_id="conv-B",
            )

        captured.clear()
        with _patch_httpx_with(captured, sse2):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                    {"role": "user", "content": "follow-up"},
                ],
                session_id="conv-B",
            )

        body = captured[0]
        assert "previous_response_id" not in body
        # Full input is sent, not a slice.
        assert len(body["input"]) == 2

        # Cursor refreshed with the new response_id.
        cursor = adapter._continuation_store.get("openai_plan", "conv-B")
        assert cursor.last_response_id == "resp_2"
        assert cursor.last_message_count == 2
