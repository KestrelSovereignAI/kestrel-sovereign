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

    async def test_adapter_extracts_call_id_not_item_id(self):
        # #857 regression: the Responses API ships function_call items with
        # both ``id`` (output-item id, ``fc_...``) and ``call_id`` (tool-call
        # id, ``call_...``). The latter is what function_call_output uses to
        # match. The adapter must expose ``call_id`` as ``ToolCall.id`` so
        # the orchestrator's downstream tool_call_id round-trips correctly.
        captured: List[Dict[str, Any]] = []
        sse = _sse([
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_OUTPUTITEM",
                    "call_id": "call_TOOLID",
                    "name": "some_tool",
                },
                "output_index": 0,
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "arguments": "{}",
            },
            _completed_event("resp_T1"),
        ])
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())
        with _patch_httpx_with(captured, sse):
            resp = await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert resp.tool_calls is not None and len(resp.tool_calls) == 1
        # Pre-#857: ``fc_OUTPUTITEM``. Post-fix: ``call_TOOLID``.
        assert resp.tool_calls[0].id == "call_TOOLID"

    async def test_reasoning_items_captured_and_replayed_on_next_turn(self):
        # #842 + #857: full realistic round trip. T1 emits a function_call
        # with distinct ``id`` and ``call_id`` fields; T2 builds the
        # orchestrator-style follow-up using the EXACT id the adapter
        # exposed (no synthesis). The body sent on T2 must have matching
        # call_ids on the replayed function_call and the synthesized
        # function_call_output — that's the invariant the live wire enforces
        # (mismatch ⇒ 400 ""No tool output found"").
        captured: List[Dict[str, Any]] = []
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        # T1: backend emits a function_call output item.
        sse1 = _sse([
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_OUTPUT",
                    "call_id": "call_TOOL",
                    "name": "model_agent",
                },
                "output_index": 0,
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "arguments": "{}",
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_T1",
                    "encrypted_content": "ENC_T1",
                    "summary": [],
                },
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_OUTPUT",
                    "call_id": "call_TOOL",
                    "name": "model_agent",
                    "arguments": "{}",
                },
            },
            _completed_event("resp_T1"),
        ])
        with _patch_httpx_with(captured, sse1):
            resp_t1 = await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
                session_id="conv-replay",
            )

        # The id we expose to the orchestrator is the one used to build the
        # NEXT turn's tool_call_id. It must be the API's ``call_id``.
        assert resp_t1.tool_calls[0].id == "call_TOOL"

        cursor = adapter._continuation_store.get("openai_plan", "conv-replay")
        cached = json.loads(cursor.turn_outputs[0])
        assert [item["type"] for item in cached] == ["reasoning", "function_call"]
        assert cached[1]["call_id"] == "call_TOOL"

        # T2: orchestrator-style follow-up using the EXACT id from resp_t1
        # (no hardcoded synthesis — this is the production round trip).
        captured.clear()
        sse2 = _sse([
            {"type": "response.output_text.delta", "delta": "ack"},
            {
                "type": "response.output_item.done",
                "item": {"type": "message", "id": "msg_T2", "role": "assistant"},
            },
            _completed_event("resp_T2"),
        ])
        tc_id = resp_t1.tool_calls[0].id  # what the orchestrator would use
        with _patch_httpx_with(captured, sse2):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": "model_agent", "arguments": "{}"},
                        }],
                    },
                    {"role": "tool", "tool_call_id": tc_id, "content": "gpt-5.5"},
                ],
                session_id="conv-replay",
            )

        body = captured[0]
        types = [item.get("type") or item.get("role") for item in body["input"]]
        assert types == ["user", "reasoning", "function_call", "function_call_output"]
        # Replayed reasoning rides along verbatim.
        assert body["input"][1]["encrypted_content"] == "ENC_T1"
        # The strict invariant: function_call.call_id MUST match
        # function_call_output.call_id. This is what the wire enforces.
        # Pre-#857 this assertion failed because the orchestrator's
        # tool_call_id was the wrong field.
        fc_call_id = body["input"][2]["call_id"]
        fco_call_id = body["input"][3]["call_id"]
        assert fc_call_id == fco_call_id == "call_TOOL", (
            f"function_call.call_id={fc_call_id!r} must match "
            f"function_call_output.call_id={fco_call_id!r}"
        )

    async def test_stale_cache_call_ids_do_not_leak_into_new_loop(self):
        # #875 regression. Two real adapter calls: the first plays the role
        # of ""prior agent loop"" and writes a tool turn into the cache; the
        # second plays the role of ""new agent loop"" with a different
        # call_id. The converter must emit the synthesized fc(call_NEW)
        # from the new message and skip the cached fc(call_STALE).
        # Pre-fix: positional replay injected fc(call_STALE) regardless,
        # producing fc(STALE)+fco(NEW) orphan pair → 400 on the wire.
        captured: List[Dict[str, Any]] = []
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())
        tools = [{
            "type": "function",
            "function": {"name": "tool_a", "description": "", "parameters": {}},
        }]

        # --- Loop 1: tool turn populates cache via the real adapter path ---
        sse_loop1 = _sse([
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_STALE_OUTPUT",
                    "call_id": "call_STALE",
                    "name": "tool_a",
                },
                "output_index": 0,
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "arguments": "{}",
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_STALE_OUTPUT",
                    "call_id": "call_STALE",
                    "name": "tool_a",
                    "arguments": "{}",
                },
            },
            _completed_event("resp_loop1"),
        ])
        with _patch_httpx_with(captured, sse_loop1):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "loop 1"},
                ],
                tools=tools,
                session_id="conv-stale",
            )

        # Sanity: cache holds the loop-1 tool turn.
        cursor = adapter._continuation_store.get("openai_plan", "conv-stale")
        assert "call_STALE" in cursor.turn_outputs[0]

        # --- Loop 2: a new tool exchange with a DIFFERENT call_id ---
        captured.clear()
        sse_loop2 = _sse([_completed_event("resp_loop2")])
        with _patch_httpx_with(captured, sse_loop2):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "loop 1"},
                    {
                        "role": "assistant", "content": "",
                        "tool_calls": [{
                            "id": "call_STALE",  # loop 1's tool call
                            "type": "function",
                            "function": {"name": "tool_a", "arguments": "{}"},
                        }],
                    },
                    {"role": "tool", "tool_call_id": "call_STALE", "content": "result1"},
                    {"role": "user", "content": "loop 2"},
                    {
                        "role": "assistant", "content": "",
                        "tool_calls": [{
                            "id": "call_NEW",  # ← different from cached call_STALE
                            "type": "function",
                            "function": {"name": "tool_a", "arguments": "{}"},
                        }],
                    },
                    {"role": "tool", "tool_call_id": "call_NEW", "content": "result2"},
                ],
                tools=tools,
                session_id="conv-stale",
            )

        body = captured[0]
        fcs = [it for it in body["input"] if it.get("type") == "function_call"]
        # Two function_calls: loop-1's (replayed from cache by id-match on
        # call_STALE) and loop-2's (synthesized from message because no
        # cache entry has call_NEW). Both must be present so their
        # respective function_call_outputs match.
        call_ids = sorted(it["call_id"] for it in fcs)
        assert call_ids == ["call_NEW", "call_STALE"], (
            f"Expected fc for both call_STALE (replayed) and call_NEW "
            f"(synthesized), got {call_ids}"
        )
        # Each function_call has a matching function_call_output.
        fcos = [it for it in body["input"] if it.get("type") == "function_call_output"]
        fco_call_ids = sorted(it["call_id"] for it in fcos)
        assert fco_call_ids == ["call_NEW", "call_STALE"]

    async def test_text_only_reasoning_replays_positionally(self):
        # #842 invariant preserved alongside #875 fix. T1 is a text-only
        # response; the cache captures its reasoning + message. T2's input
        # references the prior assistant text — the cached items should
        # replay positionally so encrypted reasoning rides along.
        # Two real adapter calls so the signature flow matches production.
        captured: List[Dict[str, Any]] = []
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        # T1: text-only response with reasoning + message output items.
        sse_t1 = _sse([
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_T1",
                    "encrypted_content": "ENC_T1",
                    "summary": [],
                },
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_T1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "prior answer"}],
                },
            },
            _completed_event("resp_T1"),
        ])
        with _patch_httpx_with(captured, sse_t1):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "first question"},
                ],
                session_id="conv-text",
            )

        # T2: text-only follow-up that includes T1's assistant text.
        captured.clear()
        sse_t2 = _sse([_completed_event("resp_T2")])
        with _patch_httpx_with(captured, sse_t2):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "prior answer"},
                    {"role": "user", "content": "follow up"},
                ],
                session_id="conv-text",
            )

        body = captured[0]
        types = [it.get("type") or it.get("role") for it in body["input"]]
        # The text-only assistant message at position 1 is replaced by the
        # cached turn's reasoning + message items, in that order.
        assert types == ["user", "reasoning", "message", "user"], types
        assert body["input"][1]["encrypted_content"] == "ENC_T1"
        assert body["input"][1]["id"] == "rs_T1"
        assert body["input"][2]["id"] == "msg_T1"

    async def test_stale_tool_turn_does_not_leak_into_text_replay(self):
        # Cache holds a tool turn AND a text turn. New input has only a
        # text-only assistant message. Tool turn must NOT leak (id-match on
        # an absent call_id); text turn DOES replay positionally.
        # Three adapter calls: the first two populate the cache faithfully
        # (loop with a tool exchange + a text follow-up), the third reads.
        captured: List[Dict[str, Any]] = []
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())
        tools = [{
            "type": "function",
            "function": {"name": "tool_a", "description": "", "parameters": {}},
        }]

        # Call 1: tool turn (cached as turn_outputs[0])
        sse_call1 = _sse([
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_TOOL_OUT",
                    "call_id": "call_STALE",
                    "name": "tool_a",
                },
                "output_index": 0,
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "arguments": "{}",
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_TOOL_OUT",
                    "call_id": "call_STALE",
                    "name": "tool_a",
                    "arguments": "{}",
                },
            },
            _completed_event("resp_C1"),
        ])
        with _patch_httpx_with(captured, sse_call1):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                ],
                tools=tools,
                session_id="conv-mixed",
            )

        # Call 2: text follow-up — populates a TEXT turn (turn_outputs[1])
        captured.clear()
        sse_call2 = _sse([
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_TEXT",
                    "encrypted_content": "ENC_TEXT",
                    "summary": [],
                },
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_TEXT",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            },
            _completed_event("resp_C2"),
        ])
        with _patch_httpx_with(captured, sse_call2):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                    {
                        "role": "assistant", "content": "",
                        "tool_calls": [{
                            "id": "call_STALE", "type": "function",
                            "function": {"name": "tool_a", "arguments": "{}"},
                        }],
                    },
                    {"role": "tool", "tool_call_id": "call_STALE", "content": "result"},
                ],
                tools=tools,
                session_id="conv-mixed",
            )

        # Call 3: NEW user message, text-only follow-up.
        # The orchestrator's storage keeps only final assistant text from
        # the prior conversation; tool/tool_call messages are transient.
        # Tools are still passed (the agent's palette is stable across user
        # messages) so the request signature stays in sync with prior calls
        # and the cache survives the session boundary.
        captured.clear()
        sse_call3 = _sse([_completed_event("resp_C3")])
        with _patch_httpx_with(captured, sse_call3):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "follow"},
                ],
                tools=tools,
                session_id="conv-mixed",
            )

        body = captured[0]
        # No stale tool-turn items in the body.
        assert all(it.get("call_id") != "call_STALE" for it in body["input"])
        assert all(it.get("id") != "fc_TOOL_OUT" for it in body["input"])
        # Text-turn reasoning + message replayed at the assistant position.
        ids = [it.get("id") for it in body["input"]]
        assert "rs_TEXT" in ids
        assert "msg_TEXT" in ids

    async def test_signature_drift_drops_cached_reasoning(self):
        # If instructions or tools change mid-conversation, cached reasoning
        # was conditioned on a different prompt and must NOT be replayed —
        # would either confuse the model or be rejected by the server.
        captured: List[Dict[str, Any]] = []
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        # Turn 1 with one toolset.
        sse1 = _sse([
            {
                "type": "response.output_item.done",
                "item": {"type": "reasoning", "id": "rs_T1", "encrypted_content": "ENC", "summary": []},
            },
            _completed_event("resp_T1"),
        ])
        with _patch_httpx_with(captured, sse1):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys-A"},
                    {"role": "user", "content": "hi"},
                ],
                tools=[{"type": "function", "function": {"name": "tool_a", "description": "", "parameters": {}}}],
                session_id="conv-drift",
            )

        # Turn 2 with DIFFERENT instructions → signature mismatch → no replay.
        captured.clear()
        sse2 = _sse([_completed_event("resp_T2")])
        with _patch_httpx_with(captured, sse2):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys-B"},  # changed
                    {"role": "user", "content": "hi"},
                    {"role": "user", "content": "follow-up"},
                ],
                tools=[{"type": "function", "function": {"name": "tool_a", "description": "", "parameters": {}}}],
                session_id="conv-drift",
            )

        body = captured[0]
        types = [item.get("type") or item.get("role") for item in body["input"]]
        # No reasoning item — drift dropped the replay.
        assert "reasoning" not in types

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
