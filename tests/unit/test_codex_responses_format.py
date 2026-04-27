"""Tests for Chat-Completions → Responses-API message conversion (#828).

The orchestrator builds multi-turn message lists in Chat-Completions format
(``tool_calls`` field on assistant, ``role=tool`` for results). The Codex
adapter must translate those into Responses-API ``function_call`` /
``function_call_output`` items before sending — the Responses API rejects
the Chat-Completions shape with ``Unknown parameter: 'input[*].tool_calls'``.
"""

import base64
import json
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from kestrel_sovereign.llm.codex_adapter import (
    CodexAdapter,
    _content_to_text,
    _convert_messages_to_responses_format,
)
from kestrel_sovereign.llm.continuation_store import InMemoryContinuationStore


def _fake_token() -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct"}}
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(b"s").rstrip(b"=").decode()
    return f"{header}.{payload}.{sig}"


class _FakeStreamResponse:
    def __init__(self, status_code, sse_lines):
        self.status_code = status_code
        self._sse_lines = sse_lines
        self.text = ""

    async def aiter_lines(self):
        for line in self._sse_lines:
            yield line

    async def aread(self):
        return b""


class _FakeStreamCtx:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *a):
        return None


class _FakeAsyncClient:
    def __init__(self, sse_lines, captured):
        self._sse_lines = sse_lines
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    def stream(self, method, url, *, headers, json):
        self._captured.append(json)
        return _FakeStreamCtx(_FakeStreamResponse(200, self._sse_lines))


def _sse(events):
    return [f"data: {json.dumps(e)}" for e in events]


def _completed(response_id):
    return {
        "type": "response.completed",
        "response": {"id": response_id, "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
    }


def _patch_httpx(captured, sse_lines):
    fake = _FakeAsyncClient(sse_lines, captured)

    class _Factory:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return fake

        async def __aexit__(self, *a):
            return None

    return patch("kestrel_sovereign.llm.codex_adapter.httpx.AsyncClient", _Factory)


class TestContentToText:
    def test_string_passes_through(self):
        assert _content_to_text("hello") == "hello"

    def test_none_becomes_empty(self):
        assert _content_to_text(None) == ""

    def test_list_of_text_parts_joined(self):
        parts = [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
        assert _content_to_text(parts) == "hello world"

    def test_non_text_parts_dropped(self):
        # Image parts are dropped — vision support deferred (#828 non-goal).
        parts = [
            {"type": "text", "text": "describe this:"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]
        assert _content_to_text(parts) == "describe this:"

    def test_empty_list_becomes_empty(self):
        assert _content_to_text([]) == ""


class TestConvertMessagesToResponsesFormat:
    def test_user_text_passes_through(self):
        out = _convert_messages_to_responses_format([
            {"role": "user", "content": "hi"},
        ])
        assert out == [{"role": "user", "content": "hi"}]

    def test_assistant_text_passes_through(self):
        out = _convert_messages_to_responses_format([
            {"role": "assistant", "content": "hello back"},
        ])
        assert out == [{"role": "assistant", "content": "hello back"}]

    def test_assistant_tool_calls_become_function_call_items(self):
        out = _convert_messages_to_responses_format([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "model_agent",
                            "arguments": '{"action":"get_current_model"}',
                        },
                    },
                ],
            },
        ])
        assert out == [{
            "type": "function_call",
            "call_id": "call_123",
            "name": "model_agent",
            "arguments": '{"action":"get_current_model"}',
        }]

    def test_assistant_with_text_and_tool_calls_emits_both(self):
        out = _convert_messages_to_responses_format([
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                ],
            },
        ])
        assert out == [
            {"role": "assistant", "content": "Let me check."},
            {
                "type": "function_call",
                "call_id": "call_a",
                "name": "f",
                "arguments": "{}",
            },
        ]

    def test_multiple_tool_calls_emit_multiple_items(self):
        out = _convert_messages_to_responses_format([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
        ])
        assert [item["call_id"] for item in out] == ["c1", "c2"]
        assert all(item["type"] == "function_call" for item in out)

    def test_dict_arguments_serialized_to_json_string(self):
        # Defensive: orchestrator uses json.dumps already, but if a caller
        # leaks a dict through, we serialize rather than send a non-string.
        out = _convert_messages_to_responses_format([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": {"x": 1}},
                    },
                ],
            },
        ])
        assert out[0]["arguments"] == json.dumps({"x": 1})

    def test_tool_role_becomes_function_call_output(self):
        out = _convert_messages_to_responses_format([
            {"role": "tool", "tool_call_id": "call_123", "content": "result text"},
        ])
        assert out == [{
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "result text",
        }]

    def test_full_round_trip_sequence(self):
        # User → assistant tool call → tool result → next user → next assistant.
        out = _convert_messages_to_responses_format([
            {"role": "user", "content": "what model are you?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "model_agent", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "gpt-5.5"},
            {"role": "assistant", "content": "I'm gpt-5.5."},
        ])
        assert out == [
            {"role": "user", "content": "what model are you?"},
            {"type": "function_call", "call_id": "c1", "name": "model_agent", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "gpt-5.5"},
            {"role": "assistant", "content": "I'm gpt-5.5."},
        ]
        # No ``tool_calls`` field should survive in any output item — that's
        # the exact field the Responses API rejects.
        assert all("tool_calls" not in item for item in out)

    def test_empty_assistant_message_dropped(self):
        # Bare ``{"role": "assistant", "content": ""}`` with no tool_calls is
        # not a valid Responses-API item; drop it rather than ship junk.
        out = _convert_messages_to_responses_format([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ])
        assert out == [{"role": "user", "content": "hi"}]

    def test_user_with_list_content_coerced_to_text(self):
        out = _convert_messages_to_responses_format([
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ])
        assert out == [{"role": "user", "content": "hi"}]

    def test_unknown_role_passes_through(self):
        # Forward-compat: don't silently swallow shapes we haven't accounted for.
        out = _convert_messages_to_responses_format([
            {"role": "developer", "content": "stay terse"},
        ])
        assert out == [{"role": "developer", "content": "stay terse"}]

    def test_does_not_mutate_input(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            ]},
        ]
        original = json.dumps(msgs)
        _convert_messages_to_responses_format(msgs)
        assert json.dumps(msgs) == original


@pytest.mark.asyncio
class TestCodexAdapterAppliesConverter:
    """End-to-end: the live request body never carries Chat-Completions tool fields.

    Drives ``CodexAdapter.get_response`` through the same fake-httpx pattern
    used in ``test_codex_continuation.py``. Each test asserts the captured
    request body has the right shape — specifically that no payload ever
    contains ``input[*].tool_calls`` (the field that triggered the live
    400 reported in #828).
    """

    async def test_no_tool_calls_field_on_first_turn(self):
        captured: List[Dict[str, Any]] = []
        sse = _sse([
            {"type": "response.output_text.delta", "delta": "ack"},
            _completed("resp_1"),
        ])
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())
        with _patch_httpx(captured, sse):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
            )
        body = captured[0]
        assert all("tool_calls" not in item for item in body["input"])
        # Sanity: this is a regular text turn — only a user item.
        assert body["input"] == [{"role": "user", "content": "hi"}]

    async def test_assistant_tool_call_message_becomes_function_call_item(self):
        # Repro of the #828 live failure. Orchestrator turn 2: messages carry
        # a Chat-Completions ``role=assistant`` with ``tool_calls`` and a
        # ``role=tool`` result. Pre-fix this generated 400 ""Unknown parameter:
        # 'input[1].tool_calls'"". After the fix, the body must contain
        # ``function_call`` + ``function_call_output`` items and no ``tool_calls``
        # field anywhere.
        captured: List[Dict[str, Any]] = []
        sse = _sse([
            {"type": "response.output_text.delta", "delta": "I'm gpt-5.5."},
            _completed("resp_2"),
        ])
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())
        with _patch_httpx(captured, sse):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "what model are you?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_xyz",
                            "type": "function",
                            "function": {
                                "name": "model_agent",
                                "arguments": '{"action":"get_current_model"}',
                            },
                        }],
                    },
                    {"role": "tool", "tool_call_id": "call_xyz", "content": "gpt-5.5"},
                ],
            )

        body = captured[0]
        # The exact field that triggered the 400 must never appear on input items.
        for item in body["input"]:
            assert "tool_calls" not in item
        # Spot-check structure.
        types = [item.get("type") or item.get("role") for item in body["input"]]
        assert types == ["user", "function_call", "function_call_output"]
        function_call = body["input"][1]
        assert function_call["call_id"] == "call_xyz"
        assert function_call["name"] == "model_agent"
        function_output = body["input"][2]
        assert function_output["call_id"] == "call_xyz"
        assert function_output["output"] == "gpt-5.5"

    async def test_continuation_watermark_unaffected_by_conversion(self):
        # The cursor's ``last_message_count`` tracks the *Chat Completions*
        # message count we sliced on, not the post-conversion item count.
        # Verify on a two-turn run.
        captured: List[Dict[str, Any]] = []
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        # Turn 1: full input goes through, cursor records count=2 (user only,
        # since system extracted to instructions).
        sse1 = _sse([_completed("r1")])
        with _patch_httpx(captured, sse1):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
                    },
                ],
                session_id="sess-1",
            )

        cursor = adapter._continuation_store.get("openai_plan", "sess-1")
        # Watermark = original Chat Completions count after system extraction:
        # (user + assistant_with_tool_calls) = 2 — even though the converter
        # rewrote those into 2 Responses-API items (user + function_call).
        assert cursor.last_message_count == 2

        # Turn 2: append a tool result. Cursor matches → delta path. The
        # delta input must be the converted form of just the new message
        # (function_call_output for the tool result).
        captured.clear()
        sse2 = _sse([_completed("r2")])
        with _patch_httpx(captured, sse2):
            await adapter.get_response(
                client=_fake_token(),
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "result"},
                ],
                session_id="sess-1",
            )
        body = captured[0]
        assert body["previous_response_id"] == "r1"
        assert body["input"] == [
            {"type": "function_call_output", "call_id": "c1", "output": "result"},
        ]
