"""A tool-bearing gpt-6.2-sol turn through openai:api uses the Responses API (#4198).

Native OpenAI refuses function tools on ``/v1/chat/completions`` while
gpt-6.2-sol reasons, and the model does not accept ``reasoning_effort='none'``,
so the gpt-5.6 normalization (#2501) cannot apply: every tool-bearing turn
failed with HTTP 400. ``OpenAIAdapter`` now sends those requests to the
Responses API, through the same ``get_response`` /
``get_streaming_response_with_tools`` entry points the agent's tool loops call.

These tests drive the real ``openai`` SDK over ``httpx.MockTransport``: no
network, no key. They pin the request the provider receives, the reasoning a
tool call is sent back with, the provider's own errors, early close, and the
provider-served evidence each response carries.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import httpx
import openai
import pytest
from kestrel_sdk.llm import LLMProtocolError, RequestOptions, ToolCallStarted

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.llm.adapter import ThinkingDelta, ToolCallWithExtraContent
from kestrel_sovereign.llm.completion_protocol import GenerationProtocolError
from kestrel_sovereign.llm.generation_gate import judge_generation
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter, ResponsesAPIFailure
from kestrel_sovereign.llm.output_ceiling import (
    OutputCapReachedError,
    reported_usage,
    response_stop_reason,
)
from kestrel_sovereign.llm.provider_served import response_provider_served

MODEL = "gpt-6.2-sol"
SERVED = "gpt-6.2-sol-2026-11-01"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup_marker",
        "description": "Return the marker.",
        "parameters": {"type": "object", "properties": {}},
    },
}]
FIRST_TURN = [
    {"role": "system", "content": "You are Kestrel."},
    {"role": "user", "content": "Look up the marker."},
]
MARKER = "marker-6f1c"
ANSWER = f"The marker is {MARKER}."
ENCRYPTED = "gAAAAB-encrypted-reasoning"

CALL_OUTPUT = [
    {"type": "reasoning", "id": "rs_1", "summary": [
        {"type": "summary_text", "text": "Need the marker."},
    ], "encrypted_content": ENCRYPTED, "status": "completed"},
    {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
     "phase": "commentary",
     "content": [{"type": "output_text", "text": "Checking.", "annotations": []}]},
    {"type": "function_call", "id": "fc_1", "call_id": "call_1",
     "name": "lookup_marker", "arguments": "{}", "status": "completed"},
]
ANSWER_OUTPUT = [
    {"type": "message", "id": "msg_2", "role": "assistant", "status": "completed",
     "phase": "final_answer",
     "content": [{"type": "output_text", "text": ANSWER, "annotations": []}]},
]
USAGE = {
    "input_tokens": 120, "input_tokens_details": {"cached_tokens": 20},
    "output_tokens": 30, "output_tokens_details": {"reasoning_tokens": 12},
    "total_tokens": 150,
}


def _response(output, *, response_id="resp_1", status="completed", **extra):
    return {
        "id": response_id, "object": "response", "created_at": 1,
        "model": SERVED, "status": status, "output": output, "usage": USAGE,
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        **extra,
    }


#: ``_stream_events(text=OWN_TEXT)`` streams each message's own text.
OWN_TEXT = object()


def _stream_events(response: Dict[str, Any], *, text=None, thinking=None,
                   terminal="response.completed") -> List[Dict[str, Any]]:
    """The SSE events the Responses API streams for ``response``: ``text``
    is the pieces each message streams, or :data:`OWN_TEXT` for its own."""
    events: List[Dict[str, Any]] = [{
        "type": "response.created",
        "response": {**response, "status": "in_progress", "output": []},
    }]
    for index, item in enumerate(response["output"]):
        events.append({"type": "response.output_item.added", "output_index": index,
                       "item": {**item, "status": "in_progress"}})
        if item["type"] == "reasoning" and thinking:
            events.append({"type": "response.reasoning_summary_text.delta",
                           "item_id": item["id"], "output_index": index,
                           "summary_index": 0, "delta": thinking})
        if item["type"] == "message" and text:
            pieces = (
                [part["text"] for part in item["content"]] if text is OWN_TEXT else text
            )
            for piece in pieces:
                events.append({"type": "response.output_text.delta",
                               "item_id": item["id"], "output_index": index,
                               "content_index": 0, "delta": piece,
                               "logprobs": []})
        if item["type"] == "function_call":
            events.append({"type": "response.function_call_arguments.delta",
                           "item_id": item["id"], "output_index": index,
                           "delta": item["arguments"]})
        events.append({"type": "response.output_item.done", "output_index": index,
                       "item": item})
    if terminal is not None:
        events.append({"type": terminal, "response": response})
    for number, event in enumerate(events):
        event["sequence_number"] = number
    return events


def _sse(events: List[Dict[str, Any]]) -> bytes:
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        for event in events
    )


class _OpenAIWire:
    """A mock OpenAI API: each ``/v1/responses`` or ``/v1/chat/completions``
    call takes the next scripted reply for that path. Records every request."""

    def __init__(self, replies: List[Any]) -> None:
        self.replies = list(replies)
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, httpx.Response):
            return reply
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200, content=_sse(reply), headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=reply)

    def bodies(self) -> List[Dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def paths(self) -> List[str]:
        return [request.url.path for request in self.requests]


def _client(wire) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key="offline-transport-only",
        base_url="https://openai.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(wire)),
        max_retries=0,
    )


async def _call(path: str, wire, messages, *, model=MODEL, tools=TOOLS,
                native_openai=True, **kwargs):
    """One adapter call the way the agent's tool loops make it: the
    terminal ``LLMResponse`` and every item streamed before it."""
    adapter = OpenAIAdapter(native_openai=native_openai)
    async with _client(wire) as client:
        if path == "generate":
            return await adapter.get_response(
                client, model, messages, tools=tools, **kwargs,
            ), []
        items = [
            item async for item in adapter.get_streaming_response_with_tools(
                client, model, messages, tools=tools, **kwargs,
            )
        ]
    return items[-1], items[:-1]


def _reply(path: str, output, **response_extra):
    response = _response(output, **response_extra)
    if path == "generate":
        return response
    return _stream_events(response, text=["Checking."] if output is CALL_OUTPUT
                          else [ANSWER[:10], ANSWER[10:]], thinking="Need it.")


def _history_message(response) -> Dict[str, Any]:
    """The assistant message the agent's tool loop replays ``response`` as."""
    return OrchestratorEngineMixin._build_assistant_tool_history_msg(
        OrchestratorEngineMixin(), response,
    )


def _tool_round(response) -> List[Dict[str, Any]]:
    return [
        *FIRST_TURN,
        _history_message(response),
        {"role": "tool", "tool_call_id": response.tool_calls[0].id, "content": MARKER},
    ]


PATHS = ["generate", "stream"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("effort", [None, "high"])
async def test_a_tool_bearing_sol_request_goes_to_the_responses_api(path, effort):
    wire = _OpenAIWire([_reply(path, CALL_OUTPUT)])
    kwargs = {} if effort is None else {
        "request_options": RequestOptions(reasoning_effort=effort),
    }

    response, _ = await _call(path, wire, FIRST_TURN, **kwargs)

    assert wire.paths() == ["/v1/responses"]
    [body] = wire.bodies()
    assert body["model"] == MODEL
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["tools"] == [{
        "type": "function", "name": "lookup_marker",
        "parameters": {"type": "object", "properties": {}},
        "strict": False, "description": "Return the marker.",
    }]
    assert body["tool_choice"] == "auto"
    assert body["input"][1] == {"role": "user", "content": "Look up the marker."}
    # Reasoning is never disabled: absent unless configured, then forwarded.
    assert body.get("reasoning") == (None if effort is None else {"effort": effort})
    for chat_only in ("messages", "reasoning_effort", "stream_options",
                      "max_completion_tokens", "previous_response_id"):
        assert chat_only not in body
    assert response_stop_reason(response) == "completed"
    assert judge_generation(response) is None
    [call] = response.tool_calls
    assert (call.id, call.name, call.arguments) == ("call_1", "lookup_marker", {})
    assert (response.input_tokens, response.cache_read_input_tokens,
            response.output_tokens, response.total_tokens) == (100, 20, 30, 130)
    assert response_provider_served(response) == {
        "api": "responses", "model": SERVED, "response_id": "resp_1",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_the_tool_loop_sends_reasoning_back_with_the_call_and_its_output(path):
    """The reasoning that produced a call, the call and the tool's output all
    reach the follow-up request, and the follow-up's answer is the turn's."""
    wire = _OpenAIWire([
        _reply(path, CALL_OUTPUT),
        _reply(path, ANSWER_OUTPUT, response_id="resp_2"),
    ])
    first, _ = await _call(path, wire, FIRST_TURN)
    [call] = first.tool_calls
    assert isinstance(call, ToolCallWithExtraContent)
    # The history survives a JSON round trip, as a journal checkpoint does.
    messages = json.loads(json.dumps(_tool_round(first)))

    final, streamed = await _call(path, wire, messages)

    follow_up = wire.bodies()[1]
    assert follow_up["input"] == [
        {"role": "system", "content": "You are Kestrel."},
        {"role": "user", "content": "Look up the marker."},
        {"type": "reasoning", "summary": [
            {"type": "summary_text", "text": "Need the marker."},
        ], "encrypted_content": ENCRYPTED},
        {"type": "message", "role": "assistant", "content": "Checking.",
         "phase": "commentary"},
        {"type": "function_call", "call_id": "call_1", "name": "lookup_marker",
         "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": MARKER},
    ]
    assert final.content == ANSWER
    assert final.tool_calls is None
    assert response_stop_reason(final) == "completed"
    assert response_provider_served(final)["response_id"] == "resp_2"
    if path == "stream":
        assert "".join(item for item in streamed if isinstance(item, str)) == ANSWER


@pytest.mark.asyncio
@pytest.mark.parametrize("native_openai, model, tools, api", [
    (True, "gpt-6.2-sol", TOOLS, "/v1/responses"),
    (True, "gpt-6.2-sol-2026-11-01", TOOLS, "/v1/responses"),
    (True, "openai:api/gpt-6.2-sol", TOOLS, "/v1/responses"),
    (True, "ft:gpt-6.2-sol:org:custom:id1", TOOLS, "/v1/responses"),
    (True, "gpt-6.2-sol", None, "/v1/chat/completions"),
    (True, "gpt-6.2-solar", TOOLS, "/v1/chat/completions"),
    (True, "gpt-6.2-sol-mini", TOOLS, "/v1/chat/completions"),
    (True, "gpt-6.1-sol", TOOLS, "/v1/chat/completions"),
    (True, "gpt-5.5", TOOLS, "/v1/chat/completions"),
    (False, "gpt-6.2-sol", TOOLS, "/v1/chat/completions"),
], ids=["sol", "snapshot", "qualified", "fine-tune", "no-tools", "other-model",
        "sibling-model", "other-version", "gpt-5.5", "compatible-route"])
async def test_only_native_tool_bearing_sol_requests_change_api(native_openai, model, tools, api):
    chat = {"id": "chatcmpl_1", "object": "chat.completion", "created": 1,
            "model": model, "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    wire = _OpenAIWire([_response(ANSWER_OUTPUT) if api == "/v1/responses" else chat])

    await _call("generate", wire, FIRST_TURN, model=model, tools=tools,
                native_openai=native_openai)

    assert wire.paths() == [api]


@pytest.mark.asyncio
async def test_a_stream_yields_reasoning_tool_starts_and_one_terminal_response():
    wire = _OpenAIWire([_reply("stream", CALL_OUTPUT)])

    terminal, items = await _call("stream", wire, FIRST_TURN)

    assert [item for item in items if isinstance(item, ThinkingDelta)] == [
        ThinkingDelta("Need it.", provider="openai"),
    ]
    [started] = [item for item in items if isinstance(item, ToolCallStarted)]
    assert (started.index, started.id, started.name) == (0, "call_1", "lookup_marker")
    assert [item for item in items if isinstance(item, str)] == ["Checking."]
    assert terminal.content == "Checking."
    assert [call.id for call in terminal.tool_calls] == ["call_1"]
    assert response_provider_served(terminal) == {
        "api": "responses", "model": SERVED, "response_id": "resp_1",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_the_providers_http_rejection_reaches_the_caller(path):
    rejection = httpx.Response(400, json={"error": {
        "message": "Invalid schema for function 'lookup_marker'.",
        "type": "invalid_request_error", "param": "tools[0]", "code": "invalid_function_parameters",
    }})
    wire = _OpenAIWire([rejection])

    with pytest.raises(openai.BadRequestError, match="Invalid schema for function"):
        await _call(path, wire, FIRST_TURN)
    assert wire.paths() == ["/v1/responses"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path, failure", [
    ("generate", "response-failed"),
    ("stream", "response-failed"),
    ("stream", "error-event"),
])
async def test_a_failed_response_raises_the_providers_own_error(path, failure):
    error = {"code": "server_error", "message": "The model failed to respond."}
    failed = _response([], status="failed", error=error)
    if path == "generate":
        reply = failed
    elif failure == "response-failed":
        reply = _stream_events(failed, terminal="response.failed")
    else:
        reply = _stream_events(failed, terminal=None) + [
            {"type": "error", "sequence_number": 9, "param": None, **error},
        ]
    wire = _OpenAIWire([reply])

    with pytest.raises(ResponsesAPIFailure) as raised:
        await _call(path, wire, FIRST_TURN)
    assert raised.value.code == "server_error"
    assert raised.value.provider_message == "The model failed to respond."
    assert "server_error" in str(raised.value)
    assert "The model failed to respond." in str(raised.value)
    assert raised.value.response_id == "resp_1"
    # The attempt is recorded with what its provider reported: the model and
    # response that answered, and the tokens a failed response still counts.
    # An ``error`` event names neither; the stream's ``response.created`` did.
    evidence = reported_usage(raised.value)
    assert evidence is raised.value.usage
    assert response_provider_served(evidence) == {
        "api": "responses", "model": SERVED, "response_id": "resp_1",
    }
    assert evidence.content is None and evidence.tool_calls is None
    if failure == "error-event":
        assert (evidence.input_tokens, evidence.output_tokens) == (None, None)
        assert response_stop_reason(evidence) is None
    else:
        assert (evidence.input_tokens, evidence.cache_read_input_tokens,
                evidence.output_tokens, evidence.total_tokens) == (100, 20, 30, 130)
        assert response_stop_reason(evidence) == "failed"


@pytest.mark.asyncio
async def test_a_stream_error_before_the_response_began_carries_no_evidence():
    wire = _OpenAIWire([[{"type": "error", "sequence_number": 0, "param": None,
                          "code": "server_error", "message": "Overloaded."}]])

    with pytest.raises(ResponsesAPIFailure) as raised:
        await _call("stream", wire, FIRST_TURN)
    assert raised.value.provider_message == "Overloaded."
    assert raised.value.response_id is None
    assert reported_usage(raised.value) is None


@pytest.mark.asyncio
async def test_a_stream_that_ends_without_a_terminal_event_is_no_answer():
    events = _stream_events(_response(CALL_OUTPUT), text=["Checking."], terminal=None)
    wire = _OpenAIWire([events])

    with pytest.raises(GenerationProtocolError) as raised:
        await _call("stream", wire, FIRST_TURN)
    # What served the unfinished response is still recorded with the attempt.
    assert response_provider_served(reported_usage(raised.value)) == {
        "api": "responses", "model": SERVED, "response_id": "resp_1",
    }


@pytest.mark.asyncio
async def test_streamed_text_the_final_response_does_not_hold_is_no_answer():
    """The streamed text is the turn's text, and the replay record describes
    the final response: a final response holding other text is refused."""
    events = _stream_events(_response(CALL_OUTPUT), text=["Something else."])
    wire = _OpenAIWire([events])

    with pytest.raises(GenerationProtocolError) as raised:
        await _call("stream", wire, FIRST_TURN)
    evidence = reported_usage(raised.value)
    assert (evidence.input_tokens, evidence.output_tokens) == (100, 30)
    assert response_provider_served(evidence)["response_id"] == "resp_1"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_an_incomplete_response_is_cut_at_its_output_limit(path):
    response = _response(ANSWER_OUTPUT, status="incomplete",
                         incomplete_details={"reason": "max_output_tokens"})
    reply = response if path == "generate" else _stream_events(
        response, text=[ANSWER], terminal="response.incomplete",
    )
    wire = _OpenAIWire([reply])

    terminal, _ = await _call(path, wire, FIRST_TURN)

    assert response_stop_reason(terminal) == "max_output_tokens"
    assert "[Output limit reached" in terminal.content
    assert isinstance(judge_generation(terminal), OutputCapReachedError)


class _HeldBody(httpx.AsyncByteStream):
    """An SSE body that sends ``data`` and then stays open, like a response
    still being generated; records whether it was closed."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.closed = asyncio.Event()

    async def __aiter__(self):
        yield self.data
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed.set()


@pytest.mark.asyncio
async def test_closing_the_stream_early_retires_the_provider_response():
    """Stop closes the turn's stream mid-response; the provider's HTTP
    response is closed with it, and nothing is sent again."""
    events = _stream_events(_response(CALL_OUTPUT), text=["Checking."], terminal=None)
    first_delta = next(
        index for index, event in enumerate(events)
        if event["type"] == "response.output_text.delta"
    )
    body = _HeldBody(_sse(events[:first_delta + 1]))
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})

    adapter = OpenAIAdapter(native_openai=True)
    async with _client(respond) as client:
        stream = adapter.get_streaming_response_with_tools(
            client, MODEL, FIRST_TURN, tools=TOOLS,
        )
        first_text = None
        async for item in stream:
            if isinstance(item, str):
                first_text = item
                break
        await stream.aclose()

    assert first_text == "Checking."
    await asyncio.wait_for(body.closed.wait(), 2)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_a_chat_field_the_responses_api_lacks_is_refused_not_dropped():
    wire = _OpenAIWire([])

    with pytest.raises(ValueError, match="frequency_penalty"):
        await _call("generate", wire, FIRST_TURN, frequency_penalty=0.5)
    assert wire.requests == []


@pytest.mark.asyncio
async def test_extra_body_wins_over_a_top_level_field_as_it_does_on_chat():
    """The OpenAI SDK merges ``extra_body`` over the typed body, so on chat
    completions a nested ``reasoning_effort`` or ``prompt_cache_key`` wins.
    The Responses request keeps the same caller intent."""
    wire = _OpenAIWire([_response(ANSWER_OUTPUT)])

    await _call("generate", wire, FIRST_TURN, request_options=RequestOptions(
        reasoning_effort="medium",
        raw={
            "prompt_cache_key": "top-level-key",
            "verbosity": "low",
            "extra_body": {"reasoning_effort": "high", "prompt_cache_key": "nested-key"},
        },
    ))

    [body] = wire.bodies()
    assert body["reasoning"] == {"effort": "high"}
    assert body["prompt_cache_key"] == "nested-key"
    assert body["text"] == {"verbosity": "low"}
    for consumed in ("reasoning_effort", "verbosity", "extra_body"):
        assert consumed not in body


@pytest.mark.asyncio
async def test_extra_body_fields_are_translated_like_top_level_ones():
    """Nothing in ``extra_body`` reaches the Responses API untranslated: a
    chat-named field is translated, a Responses object merges with what the
    request already carries, and an unknown field is refused."""
    wire = _OpenAIWire([_response(ANSWER_OUTPUT)])

    await _call("generate", wire, FIRST_TURN, request_options=RequestOptions(
        reasoning_effort="high",
        raw={"extra_body": {
            "max_completion_tokens": 900,
            "verbosity": "low",
            "reasoning": {"summary": "auto"},
            "text": {"format": {"type": "json_object"}},
        }},
    ))

    [body] = wire.bodies()
    assert body["max_output_tokens"] == 900
    assert body["reasoning"] == {"summary": "auto", "effort": "high"}
    assert body["text"] == {"format": {"type": "json_object"}, "verbosity": "low"}
    for chat_only in ("max_completion_tokens", "verbosity", "reasoning_effort"):
        assert chat_only not in body

    refused = _OpenAIWire([])
    with pytest.raises(ValueError, match="logit_bias"):
        await _call("generate", refused, FIRST_TURN, request_options=RequestOptions(
            raw={"extra_body": {"logit_bias": {"50256": -100}}},
        ))
    assert refused.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("raw, field", [
    ({"extra_body": {"reasoning": {"effort": "low"}}}, "reasoning.effort"),
    ({"text": {"format": {"type": "json_object"}}, "response_format": {"type": "text"}},
     "text.format"),
    ({"max_tokens": 10, "max_completion_tokens": 20}, "max_output_tokens"),
], ids=["reasoning-effort", "text-format", "output-budget"])
async def test_two_spellings_of_one_field_are_refused_not_resolved(raw, field):
    wire = _OpenAIWire([])

    with pytest.raises(ValueError, match=field):
        await _call("generate", wire, FIRST_TURN, request_options=RequestOptions(
            reasoning_effort="medium", raw=raw,
        ))
    assert wire.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_content, kept", [
    ({"openai": {"responses": {"output": [
        {"type": "message", "phase": "commentary", "text_length": 0},
        {"type": "function_call", "call_id": "call_1"},
    ]}}}, None),
    ({"openai": {"responses": {"output": []}}, "google": {"thought_signature": "c2ln"}},
     {"google": {"thought_signature": "c2ln"}}),
], ids=["openai-only", "openai-and-google"])
async def test_chat_completions_never_receive_responses_replay_data(extra_content, kept):
    """A Responses tool call replayed on chat completions (a call of the turn
    made without tools) loses only the Responses replay namespace."""
    chat = {"id": "chatcmpl_1", "object": "chat.completion", "created": 1,
            "model": MODEL, "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    wire = _OpenAIWire([chat])
    messages = [*FIRST_TURN, {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "lookup_marker", "arguments": {}},
        "extra_content": extra_content,
    }]}, {"role": "tool", "tool_call_id": "call_1", "content": MARKER}]

    await _call("generate", wire, messages, tools=None)

    [body] = wire.bodies()
    assert wire.paths() == ["/v1/chat/completions"]
    [call] = body["messages"][2]["tool_calls"]
    assert call.get("extra_content") == kept
    assert messages[2]["tool_calls"][0]["extra_content"] is extra_content


@pytest.mark.asyncio
async def test_structured_output_becomes_the_responses_text_format():
    from pydantic import BaseModel

    class Marker(BaseModel):
        value: str

    wire = _OpenAIWire([_response(ANSWER_OUTPUT)])

    await _call("generate", wire, FIRST_TURN, response_format=Marker)

    [body] = wire.bodies()
    text_format = body["text"]["format"]
    assert (text_format["type"], text_format["name"], text_format["strict"]) == (
        "json_schema", "Marker", True,
    )
    assert text_format["schema"]["properties"] == {"value": {"title": "Value", "type": "string"}}
    assert "response_format" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("replay, refusal", [
    ({"output": [{"type": "reasoning"}, {"type": "function_call", "call_id": "call_1"}]},
     "encrypted reasoning items"),
    ({"output": [{"type": "function_call", "call_id": "call_1"}], "reasoning": []},
     "only the response's output order"),
    ({"output": [{"type": "message", "text_length": "0"},
                 {"type": "function_call", "call_id": "call_1"}]},
     "text length"),
    ({"output": [{"type": "web_search_call"}]}, "not one Kestrel wrote"),
    ({"output": [{"type": "function_call", "call_id": "call_1"}]}, "JSON object text"),
    ({"output": [{"type": "function_call", "call_id": "call_1", "arguments": "[]"}]},
     "JSON object text"),
    ({"output": [{"type": "function_call", "call_id": "call_1", "arguments": "{"}]},
     "must be JSON text"),
    ({"output": [{"type": "function_call", "call_id": "call_1",
                  "arguments": '{"a":' * 200_000 + "1" + "}" * 200_000}]},
     "must be JSON text"),
], ids=["unencrypted-reasoning", "unknown-key", "bad-length", "unknown-entry",
        "no-argument-text", "non-object-arguments", "invalid-argument-text",
        "too-deeply-nested-argument-text"])
async def test_malformed_replay_data_is_refused_before_any_request(replay, refusal):
    wire = _OpenAIWire([])
    messages = [*FIRST_TURN, {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "lookup_marker", "arguments": "{}"},
        "extra_content": {"openai": {"responses": replay}},
    }]}, {"role": "tool", "tool_call_id": "call_1", "content": MARKER}]

    with pytest.raises(ValueError, match=refusal):
        await _call("generate", wire, messages)
    assert wire.requests == []


@pytest.mark.asyncio
async def test_a_call_from_another_provider_is_replayed_without_reasoning():
    """A tool call carrying another provider's ``extra_content`` (Gemini's
    signature) replays as a plain call: no foreign data reaches OpenAI."""
    wire = _OpenAIWire([_response(ANSWER_OUTPUT)])
    messages = [*FIRST_TURN, {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "lookup_marker", "arguments": {}},
        "extra_content": {"google": {"thought_signature": "c2ln"}},
    }]}, {"role": "tool", "tool_call_id": "call_1", "content": MARKER}]

    await _call("generate", wire, messages)

    assert wire.bodies()[0]["input"][2:] == [
        {"type": "function_call", "call_id": "call_1", "name": "lookup_marker",
         "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": MARKER},
    ]


INTERLEAVED_OUTPUT = [
    {"type": "reasoning", "id": "rs_1", "summary": [
        {"type": "summary_text", "text": "First look."},
    ], "encrypted_content": "enc-first", "status": "completed"},
    {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
     "phase": "commentary",
     "content": [{"type": "output_text", "text": "Looking it up.", "annotations": []}]},
    {"type": "function_call", "id": "fc_1", "call_id": "call_1",
     "name": "lookup_marker", "arguments": "{}", "status": "completed"},
    {"type": "reasoning", "id": "rs_2", "summary": [],
     "encrypted_content": "enc-second", "status": "completed"},
    {"type": "message", "id": "msg_2", "role": "assistant", "status": "completed",
     "content": [{"type": "output_text", "text": " Checking twice.", "annotations": []}]},
    {"type": "function_call", "id": "fc_2", "call_id": "call_2",
     "name": "lookup_marker", "arguments": "{\"again\": true}", "status": "completed"},
]
INTERLEAVED_REPLAY = [
    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "First look."}],
     "encrypted_content": "enc-first"},
    {"type": "message", "role": "assistant", "content": "Looking it up.",
     "phase": "commentary"},
    {"type": "function_call", "call_id": "call_1", "name": "lookup_marker",
     "arguments": "{}"},
    {"type": "reasoning", "summary": [], "encrypted_content": "enc-second"},
    {"type": "message", "role": "assistant", "content": " Checking twice."},
    {"type": "function_call", "call_id": "call_2", "name": "lookup_marker",
     "arguments": "{\"again\": true}"},
]


def _interleaved_reply(path: str, **response_extra):
    response = _response(INTERLEAVED_OUTPUT, **response_extra)
    if path == "generate":
        return response
    return _stream_events(response, text=OWN_TEXT, thinking="Thinking.")


def _two_call_round(response) -> List[Dict[str, Any]]:
    return [
        *FIRST_TURN,
        _history_message(response),
        *({"role": "tool", "tool_call_id": call.id, "content": MARKER}
          for call in response.tool_calls),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_interleaved_output_is_sent_back_in_its_own_order(path):
    """Reasoning, messages and calls the response interleaved go back exactly
    where it produced them: no reasoning moved ahead, no messages merged, and
    each message keeps its own phase."""
    wire = _OpenAIWire([
        _interleaved_reply(path),
        _reply(path, ANSWER_OUTPUT, response_id="resp_2"),
    ])
    first, _ = await _call(path, wire, FIRST_TURN)
    assert first.content == "Looking it up. Checking twice."
    assert [call.id for call in first.tool_calls] == ["call_1", "call_2"]
    assert first.tool_calls[1].arguments == {"again": True}
    messages = json.loads(json.dumps(_two_call_round(first)))

    final, _ = await _call(path, wire, messages)

    assert wire.bodies()[1]["input"] == [
        {"role": "system", "content": "You are Kestrel."},
        {"role": "user", "content": "Look up the marker."},
        *INTERLEAVED_REPLAY,
        {"type": "function_call_output", "call_id": "call_1", "output": MARKER},
        {"type": "function_call_output", "call_id": "call_2", "output": MARKER},
    ]
    assert final.content == ANSWER


def _interleaved_history(path: str):
    """The interleaved response's history message, built from a real call."""
    async def build():
        first, _ = await _call(path, _OpenAIWire([_interleaved_reply(path)]), FIRST_TURN)
        return _two_call_round(first)
    return build()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("change, refusal", [
    ("text", "no longer matches the Responses output"),
    ("drop-call", "does not carry"),
    ("add-call", "does not place"),
    ("other-arguments", "arguments no longer match"),
    ("same-truth-as-number", "arguments no longer match"),
], ids=["edited-text", "dropped-call", "added-call", "edited-arguments",
        "bool-as-int-arguments"])
async def test_a_session_message_that_no_longer_fits_its_replay_is_refused(
    path, change, refusal,
):
    messages = await _interleaved_history(path)
    assistant = messages[2]
    if change == "text":
        assistant["content"] += " (edited)"
    elif change == "other-arguments":
        assistant["tool_calls"][1]["function"]["arguments"] = {"again": False}
    elif change == "same-truth-as-number":
        # ``1 == True`` in Python; as JSON they are different arguments.
        assistant["tool_calls"][1]["function"]["arguments"] = {"again": 1}
    elif change == "drop-call":
        assistant["tool_calls"].pop()
        messages.pop()
    else:
        assistant["tool_calls"].append({
            "id": "call_3", "type": "function",
            "function": {"name": "lookup_marker", "arguments": {}},
        })
        messages.append({"role": "tool", "tool_call_id": "call_3", "content": MARKER})
    wire = _OpenAIWire([])

    with pytest.raises(ValueError, match=refusal):
        await _call(path, wire, messages)
    assert wire.requests == []


#: Argument text a provider may write that ``json.dumps`` of its decoded value
#: does not reproduce: compact separators, a raw non-ASCII character, a JSON
#: escape, exponent notation and a negative zero.
NATIVE_ARGUMENTS = [
    '{"x":1}',
    '{"city":"Z\u00fcrich","note":"\\u2603"}',
    '{"n":1e3,"z":-0.0}',
]
NATIVE_ARGUMENTS_OUTPUT = [
    {"type": "reasoning", "id": "rs_1", "summary": [],
     "encrypted_content": "enc-before", "status": "completed"},
    *({"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}",
       "name": "lookup_marker", "arguments": arguments, "status": "completed"}
      for index, arguments in enumerate(NATIVE_ARGUMENTS)),
    {"type": "reasoning", "id": "rs_2", "summary": [],
     "encrypted_content": "enc-after", "status": "completed"},
    {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
     "content": [{"type": "output_text", "text": "One more.", "annotations": []}]},
    {"type": "function_call", "id": "fc_last", "call_id": "call_last",
     "name": "lookup_marker", "arguments": '{ "spaced" : true }', "status": "completed"},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_each_call_goes_back_with_the_argument_text_the_provider_wrote(path):
    """A call's arguments are sent back as the provider wrote them, byte for
    byte. Re-encoding the decoded arguments (``{"x":1}`` as ``{"x": 1}``)
    changes the history the later reasoning item was produced after."""
    response = _response(NATIVE_ARGUMENTS_OUTPUT)
    reply = response if path == "generate" else _stream_events(response, text=OWN_TEXT)
    wire = _OpenAIWire([reply, _reply(path, ANSWER_OUTPUT, response_id="resp_2")])
    first, _ = await _call(path, wire, FIRST_TURN)
    assert [call.arguments for call in first.tool_calls] == [
        {"x": 1}, {"city": "Z\u00fcrich", "note": "\u2603"}, {"n": 1000.0, "z": -0.0},
        {"spaced": True},
    ]
    messages = json.loads(json.dumps(_two_call_round(first)))

    await _call(path, wire, messages)

    sent = [item for item in wire.bodies()[1]["input"] if item.get("type") in (
        "reasoning", "message", "function_call")]
    assert [item.get("encrypted_content") or item.get("arguments") or item["content"]
            for item in sent] == [
        "enc-before", *NATIVE_ARGUMENTS, "enc-after", "One more.", '{ "spaced" : true }',
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_a_response_of_calls_alone_keeps_their_argument_text(path):
    """With no reasoning or message to place, the calls' own argument text is
    still the provider's, not a re-encoding of it."""
    output = [{"type": "function_call", "id": "fc_1", "call_id": "call_1",
               "name": "lookup_marker", "arguments": '{"x":1}', "status": "completed"}]
    response = _response(output)
    reply = response if path == "generate" else _stream_events(response)
    wire = _OpenAIWire([reply, _reply(path, ANSWER_OUTPUT, response_id="resp_2")])
    first, _ = await _call(path, wire, FIRST_TURN)
    messages = json.loads(json.dumps(_tool_round(first)))

    await _call(path, wire, messages)

    assert wire.bodies()[1]["input"][2:] == [
        {"type": "function_call", "call_id": "call_1", "name": "lookup_marker",
         "arguments": '{"x":1}'},
        {"type": "function_call_output", "call_id": "call_1", "output": MARKER},
    ]


class _DroppedBody(httpx.AsyncByteStream):
    """An SSE body that sends ``data`` and then loses its connection."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    async def __aiter__(self):
        yield self.data
        raise httpx.ReadError("connection lost mid-response")

    async def aclose(self) -> None:
        return None


def _dropped(events: List[Dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, stream=_DroppedBody(_sse(events)),
                          headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
async def test_a_stream_that_drops_raises_the_sdks_error_with_what_served_it():
    """A connection lost after ``response.created`` is the SDK's own error,
    raised as the SDK raised it, and the attempt keeps the model and response
    id the stream had already named."""
    events = _stream_events(_response(CALL_OUTPUT), text=["Checking."], terminal=None)
    wire = _OpenAIWire([_dropped(events)])

    with pytest.raises(openai.APIConnectionError) as raised:
        await _call("stream", wire, FIRST_TURN)

    assert type(raised.value) is openai.APIConnectionError
    assert isinstance(raised.value.__cause__, httpx.ReadError)
    evidence = reported_usage(raised.value)
    assert response_provider_served(evidence) == {
        "api": "responses", "model": SERVED, "response_id": "resp_1",
    }
    assert evidence.content is None and evidence.tool_calls is None
    assert (evidence.input_tokens, evidence.output_tokens) == (None, None)
    assert wire.paths() == ["/v1/responses"]


@pytest.mark.asyncio
async def test_a_stream_that_drops_after_its_terminal_event_keeps_its_counts():
    events = _stream_events(_response(CALL_OUTPUT), text=["Checking."])
    wire = _OpenAIWire([_dropped(events)])

    with pytest.raises(openai.APIConnectionError) as raised:
        await _call("stream", wire, FIRST_TURN)

    evidence = reported_usage(raised.value)
    assert response_provider_served(evidence)["response_id"] == "resp_1"
    assert (evidence.input_tokens, evidence.cache_read_input_tokens,
            evidence.output_tokens, evidence.total_tokens) == (100, 20, 30, 130)
    assert response_stop_reason(evidence) == "completed"


@pytest.mark.asyncio
async def test_a_stream_that_drops_before_it_began_carries_nothing():
    wire = _OpenAIWire([_dropped([])])

    with pytest.raises(openai.APIConnectionError) as raised:
        await _call("stream", wire, FIRST_TURN)

    assert reported_usage(raised.value) is None


@pytest.mark.asyncio
async def test_a_non_streamed_transport_failure_names_nothing_it_never_saw(monkeypatch):
    """A non-streamed response arrives whole, so a transport failure comes
    before anything served it is known: the SDK's error is raised with no
    evidence invented for it (the call is retried, as any dropped connection
    is, first)."""
    real_sleep = asyncio.sleep

    async def no_backoff(delay, *args, **kwargs):
        await real_sleep(0)

    monkeypatch.setattr("kestrel_sovereign.llm.retry.asyncio.sleep", no_backoff)

    def drop(request):
        raise httpx.ReadError("connection lost", request=request)

    with pytest.raises(openai.APIConnectionError) as raised:
        await _call("generate", drop, FIRST_TURN)

    assert reported_usage(raised.value) is None


@pytest.mark.asyncio
async def test_a_non_streamed_response_that_cannot_be_read_keeps_what_served_it():
    """A failure after a non-streamed response arrived is raised as it was,
    and the attempt keeps what the response reported."""
    malformed = [{"type": "message", "id": "msg_1", "role": "assistant",
                  "status": "completed", "content": 5}]
    wire = _OpenAIWire([_response(malformed)])

    with pytest.raises(TypeError) as raised:
        await _call("generate", wire, FIRST_TURN)

    evidence = reported_usage(raised.value)
    assert response_provider_served(evidence) == {
        "api": "responses", "model": SERVED, "response_id": "resp_1",
    }
    assert (evidence.input_tokens, evidence.output_tokens) == (100, 30)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_function_tools_in_extra_body_select_the_responses_api(path):
    """The SDK sends ``extra_body`` merged over the typed fields, so tools
    named only there are function tools of the request: the endpoint is
    chosen from the same payload that is translated."""
    wire = _OpenAIWire([_reply(path, CALL_OUTPUT)])

    terminal, _ = await _call(path, wire, FIRST_TURN, tools=None, request_options=RequestOptions(
        raw={"extra_body": {"tools": TOOLS}},
    ))

    assert wire.paths() == ["/v1/responses"]
    [body] = wire.bodies()
    assert [tool["name"] for tool in body["tools"]] == ["lookup_marker"]
    assert "extra_body" not in body
    assert [call.id for call in terminal.tool_calls] == ["call_1"]


@pytest.mark.asyncio
async def test_gpt56_function_tools_in_extra_body_disable_chat_reasoning():
    """The gpt-5.6 chat-completions rule (#2501) reads the same merged payload."""
    chat = {"id": "chatcmpl_1", "object": "chat.completion", "created": 1,
            "model": "gpt-5.6-luna", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    wire = _OpenAIWire([chat])

    await _call("generate", wire, FIRST_TURN, model="gpt-5.6-luna", tools=None,
                request_options=RequestOptions(
                    reasoning_effort="high", raw={"extra_body": {"tools": TOOLS}},
                ))

    assert wire.paths() == ["/v1/chat/completions"]
    [body] = wire.bodies()
    assert body["reasoning_effort"] == "none"
    assert [tool["function"]["name"] for tool in body["tools"]] == ["lookup_marker"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_chat_completions_responses_carry_provider_served_evidence(path):
    """The restored default's turns are on chat completions; their provider's
    own model and response id are recorded too."""
    if path == "generate":
        reply = {"id": "chatcmpl_9", "object": "chat.completion", "created": 1,
                 "model": "gpt-5.5-2026-01-01", "choices": [{"index": 0,
                 "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        wire = _OpenAIWire([reply])
    else:
        chunk = {"id": "chatcmpl_9", "object": "chat.completion.chunk", "created": 1,
                 "model": "gpt-5.5-2026-01-01"}
        sse = (
            f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {'content': 'ok'}}]})}\n\n"
            f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            "data: [DONE]\n\n"
        ).encode()
        wire = _OpenAIWire([httpx.Response(
            200, content=sse, headers={"content-type": "text/event-stream"},
        )])

    terminal, _ = await _call(path, wire, FIRST_TURN, model="gpt-5.5")

    assert wire.paths() == ["/v1/chat/completions"]
    assert response_provider_served(terminal) == {
        "api": "chat.completions", "model": "gpt-5.5-2026-01-01",
        "response_id": "chatcmpl_9",
    }


def test_a_protocol_error_is_never_completion_evidence():
    from kestrel_sovereign.llm.openai_adapter import _parse_responses_output

    with pytest.raises(LLMProtocolError):
        _parse_responses_output({"output": [{"type": "web_search_call", "id": "ws_1"}]})
    with pytest.raises(LLMProtocolError):
        _parse_responses_output({"output": [
            {"type": "function_call", "call_id": "", "name": "lookup_marker", "arguments": "{}"},
        ]})
    # The Responses API writes arguments as JSON text, and that text is what
    # the call is sent back as: decoded arguments have no text to send.
    with pytest.raises(LLMProtocolError, match="JSON text"):
        _parse_responses_output({"output": [
            {"type": "function_call", "call_id": "call_1", "name": "lookup_marker",
             "arguments": {}},
        ]})
