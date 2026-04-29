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
        # #875 regression. Cache holds fc(call_STALE) from a prior agent
        # loop. The new loop's orchestrator emits assistant.tool_calls with
        # call_id=call_NEW. The converter must emit fc(call_NEW) (synthesized
        # from message — id-match cache miss path), NOT fc(call_STALE).
        # Pre-fix: positional replay injected fc(call_STALE) regardless,
        # producing fc(STALE)+fco(NEW) orphan pair → 400 on the wire.
        from kestrel_sovereign.llm.codex_adapter import (
            _compute_request_signature, _convert_tools_to_responses_format,
        )
        from kestrel_sovereign.llm.continuation_store import ContinuationCursor

        captured: List[Dict[str, Any]] = []
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        # Pre-populate cache as if a prior loop ended.
        tools = [{
            "type": "function",
            "function": {"name": "tool_a", "description": "", "parameters": {}},
        }]
        sig = _compute_request_signature("sys", _convert_tools_to_responses_format(tools))
        adapter._continuation_store.put(
            "openai_plan", "conv-stale",
            ContinuationCursor(
                last_response_id="resp_prior",
                last_message_count=1,
                last_request_signature=sig,
                turn_outputs=(json.dumps([{
                    "type": "function_call",
                    "id": "fc_STALE",
                    "call_id": "call_STALE",
                    "name": "tool_a",
                    "arguments": "{}",
                }]),),
            ),
        )

        sse = _sse([_completed_event("resp_new")])
        with _patch_httpx_with(captured, sse):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant", "content": "",
                        "tool_calls": [{
                            "id": "call_NEW",
                            "type": "function",
                            "function": {"name": "tool_a", "arguments": "{}"},
                        }],
                    },
                    {"role": "tool", "tool_call_id": "call_NEW", "content": "result"},
                ],
                tools=tools,
                session_id="conv-stale",
            )

        body = captured[0]
        fcs = [it for it in body["input"] if it.get("type") == "function_call"]
        assert len(fcs) == 1, f"Expected exactly one function_call, got {fcs}"
        assert fcs[0]["call_id"] == "call_NEW"
        # No stale call_id leaked from the cache.
        assert all(it.get("call_id") != "call_STALE" for it in body["input"]), (
            f"Stale cached call_STALE leaked into the new loop: {body['input']}"
        )

    async def test_fresh_agent_loop_clears_stale_cache(self):
        # #875 secondary fix. When the input has no assistant.tool_calls
        # (= start of a new agent loop after a prior conversation), the
        # cache is cleared so it doesn't grow unboundedly and stale entries
        # can't leak via id-match either.
        from kestrel_sovereign.llm.codex_adapter import _compute_request_signature
        from kestrel_sovereign.llm.continuation_store import ContinuationCursor

        captured: List[Dict[str, Any]] = []
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        sig = _compute_request_signature("sys", None)
        adapter._continuation_store.put(
            "openai_plan", "conv-fresh",
            ContinuationCursor(
                last_response_id="resp_prior",
                last_message_count=1,
                last_request_signature=sig,
                turn_outputs=(json.dumps([
                    {"type": "reasoning", "id": "rs_OLD", "encrypted_content": "OLD", "summary": []},
                    {"type": "function_call", "id": "fc_OLD", "call_id": "call_OLD",
                     "name": "x", "arguments": "{}"},
                ]),),
            ),
        )

        sse = _sse([
            {"type": "response.output_text.delta", "delta": "ok"},
            {
                "type": "response.output_item.done",
                "item": {"type": "message", "id": "msg_NEW", "role": "assistant"},
            },
            _completed_event("resp_new"),
        ])
        with _patch_httpx_with(captured, sse):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Fresh user message."},
                ],
                session_id="conv-fresh",
            )

        body = captured[0]
        # No stale reasoning/function_call leaked into the body.
        assert all(
            it.get("type") not in ("reasoning", "function_call") for it in body["input"]
        ), f"Stale cache leaked: {body['input']}"
        # Cache was cleared, then turn 1 of the new loop was recorded fresh.
        cursor = adapter._continuation_store.get("openai_plan", "conv-fresh")
        assert len(cursor.turn_outputs) == 1
        new_outputs = json.loads(cursor.turn_outputs[0])
        # OLD entries are not in the new turn's outputs.
        assert all(it.get("id") not in {"rs_OLD", "fc_OLD"} for it in new_outputs)
        # The fresh turn captured the new message item.
        assert any(it.get("id") == "msg_NEW" for it in new_outputs)

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
