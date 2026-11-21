"""#4198: ``OpenAIAdapter``'s Responses API path, at the adapter boundary.

The real ``openai`` SDK runs over :class:`tests.shared.openai_responses_wire.ResponsesWire`;
assertions read the requests as the SDK serialized them. Booted-agent turns
are covered in ``test_openai_responses_continuation_turn.py``.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
from typing import Any, Dict, List

import pytest

from kestrel_sdk.llm import ToolCallStarted
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.completion_protocol import GenerationProtocolError
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.openai_responses import (
    ResponsesAPIFailure,
    requires_responses_api,
)
from kestrel_sovereign.llm.output_ceiling import reported_usage
from kestrel_sovereign.llm.provider_served import provider_served
from kestrel_sovereign.llm.responses_continuation import (
    MAX_SNAPSHOTS_PER_TURN,
    ResponsesContinuations,
    current_responses_continuations,
    responses_continuation_scope,
)
from kestrel_sovereign.turn_scope import capture_turn_scope
from tests.shared.openai_responses_wire import (
    MODEL,
    SERVED,
    Reply,
    ResponsesWire,
    StallingWire,
    call,
    input_types,
    message,
    reasoning,
    wire_items,
)

pytestmark = pytest.mark.asyncio

TOOLS = [{"type": "function", "function": {
    "name": "lookup", "description": "Look a value up.",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
}}]
ARGUMENTS = '{"key":"\\u00e9","n":1e2}'
SYSTEM = {"role": "system", "content": "You are Kestrel."}
USER = {"role": "user", "content": "Look it up."}
OPERATOR = {"role": "system", "content": "Operator context: auto-mode is on."}


def _history(call_id: str = "call_1") -> List[Dict[str, Any]]:
    """The tool loop's rebuilt continuation: no operator notice, re-encoded
    arguments, exactly as ``_seed_orchestrator_messages`` builds it."""
    return [
        SYSTEM, USER,
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "lookup", "arguments": {"key": "é", "n": 100.0}},
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": '{"value": 42}'},
    ]


def _first_then_answer(body, index):
    if index == 0:
        return Reply([reasoning("rs_1", "ENC-1"), call("fc_1", "call_1", "lookup", ARGUMENTS)], "resp_1")
    return Reply([reasoning("rs_2", "ENC-2"), message("msg_2", "It is 42.")], "resp_2")


async def _call(adapter, client, messages, *, stream: bool, tools=TOOLS, model=MODEL, **kwargs):
    if stream:
        items = [
            item async for item in adapter.get_streaming_response_with_tools(
                client=client, model=model, messages=messages, tools=tools, **kwargs,
            )
        ]
        assert isinstance(items[-1], LLMResponse)
        return items[-1], items
    return await adapter.get_response(client=client, model=model, messages=messages, tools=tools, **kwargs), None


def _adapter() -> OpenAIAdapter:
    return OpenAIAdapter("openai", native_openai=True)


STREAM = pytest.mark.parametrize("stream", [False, True], ids=["nonstreaming", "streaming"])


@pytest.mark.parametrize("model, native, request_body, expected", [
    (MODEL, True, {"tools": TOOLS}, True),
    ("openai/gpt-6.2-sol-2026-11-01", True, {"tools": TOOLS}, True),
    ("ft:gpt-6.2-sol:org::abc", True, {"tools": TOOLS}, True),
    # A hyphenated variant of the family is matched on purpose (see the
    # pattern's comment): it shares the chat-completions refusal.
    ("gpt-6.2-sol-mini", True, {"tools": TOOLS}, True),
    (MODEL, True, {}, False),
    (MODEL, False, {"tools": TOOLS}, False),
    (MODEL, False, {"extra_body": {"tools": TOOLS}}, False),
    ("gpt-6.2-solar", True, {"tools": TOOLS}, False),
    ("gpt-5.5", True, {"tools": TOOLS}, False),
    ("gpt-5.5", True, {"extra_body": {"tools": TOOLS}}, False),
])
async def test_the_endpoint_is_chosen_from_the_request_as_sent(model, native, request_body, expected):
    from kestrel_sovereign.llm.openai_responses import ResponsesRoute

    assert requires_responses_api(
        native_openai=native, model=model, request=request_body,
        messages=[SYSTEM, USER], route=ResponsesRoute(label="openai|", client=object()),
    ) is expected


# ---------------------------------------------------------------------------
# One tool source: the typed parameter (#4198 item o)
# ---------------------------------------------------------------------------

CONFLICTING_TOOLS = [{"type": "function", "function": {
    "name": "lookup", "description": "A different lookup.",
    "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}},
}}]


def _raw_tools_kwargs(channel: str, raw_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    from kestrel_sdk.llm import RequestOptions

    if channel == "extra_body":
        return {"extra_body": {"tools": raw_tools}}
    return {"request_options": RequestOptions(raw={"extra_body": {"tools": raw_tools}})}


@STREAM
@pytest.mark.parametrize("channel", ["extra_body", "request_options_raw"])
@pytest.mark.parametrize("typed, raw_tools", [
    (None, TOOLS),
    (TOOLS, TOOLS),
    (TOOLS, CONFLICTING_TOOLS),
    (None, CONFLICTING_TOOLS),
], ids=["raw_only", "typed_plus_identical_raw", "typed_plus_conflicting_raw", "raw_only_conflicting"])
async def test_tools_supplied_through_extra_body_are_refused_before_transmission(
    stream, channel, typed, raw_tools,
):
    """Rewritten from ``test_tools_supplied_through_extra_body_reach_the_responses_api``:
    the native Sol route has one tool source, the typed ``tools`` parameter.
    A raw declaration is refused by name before any request is sent, even
    one identical to the typed declaration; nothing is merged or
    de-duplicated."""
    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as client:
        with pytest.raises(ValueError, match=r"extra_body\['tools'\].*typed `tools` parameter"):
            await _call(
                _adapter(), client, [SYSTEM, USER], stream=stream, tools=typed,
                **_raw_tools_kwargs(channel, raw_tools),
            )
    assert len(wire.requests) == 0


@STREAM
async def test_provider_cache_body_still_reaches_the_sol_request_unchanged(stream):
    """The service's own ``extra_body`` channel is untouched by the refusal:
    whatever ``provider_cache_body`` returns reaches the Responses request."""
    from kestrel_sovereign.llm.provider_registry import provider_cache_body

    assert provider_cache_body({"vendor": "openai"}) is None
    cache_body = provider_cache_body({"vendor": "llama_cpp"})
    assert cache_body == {"cache_prompt": True}
    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as client:
        await _call(_adapter(), client, [SYSTEM, USER], stream=stream, extra_body=dict(cache_body))
    [body] = wire.bodies()
    assert body["cache_prompt"] is True
    assert [tool["name"] for tool in body["tools"]] == ["lookup"]


class _ChatWire:
    """A chat-completions endpoint that records what the SDK sent."""

    def __init__(self) -> None:
        self.requests: List[Any] = []

    def __call__(self, request):
        import httpx

        self.requests.append(request)
        body = json.loads(request.content)
        message = {"role": "assistant", "content": "ok"}
        usage = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        if not body.get("stream"):
            return httpx.Response(200, json={
                "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
                "model": body["model"], "usage": usage,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            })
        chunks = [
            {"choices": [{"index": 0, "delta": message, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage},
        ]
        data = "".join(
            "data: " + json.dumps({
                "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0,
                "model": body["model"], **chunk,
            }) + "\n\n"
            for chunk in chunks
        ) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=data.encode())

    def client(self):
        import httpx
        import openai

        return openai.AsyncOpenAI(
            api_key="test-key", base_url="https://api.openai.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self)), max_retries=0,
        )


@STREAM
async def test_a_raw_tools_option_passes_through_unchanged_on_a_non_sol_route(stream):
    """The refusal is the native Sol Responses route's alone: elsewhere a raw
    option, a ``tools`` key included, reaches the provider as given."""
    from kestrel_sdk.llm import RequestOptions

    raw = {"extra_body": {"tools": CONFLICTING_TOOLS, "custom_flag": {"a": 1}}}
    wire = _ChatWire()
    async with wire.client() as client:
        await _call(
            _adapter(), client, [SYSTEM, USER], stream=stream, tools=None, model="gpt-5.5",
            request_options=RequestOptions(raw=raw),
        )
    [request] = wire.requests
    assert request.url.path == "/v1/chat/completions"
    body = json.loads(request.content)
    assert body["tools"] == CONFLICTING_TOOLS
    assert body["custom_flag"] == {"a": 1}


@STREAM
async def test_a_continuation_appends_to_the_captured_prefix(stream):
    """The rebuilt list drops the notice and re-encodes the arguments; the
    request replays what was sent and emitted instead."""
    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            first, _ = await _call(adapter, client, [SYSTEM, USER, OPERATOR], stream=stream)
            second, items = await _call(adapter, client, _history(), stream=stream)
    sent, continued = wire.raw()
    assert wire_items(continued)[: len(wire_items(sent))] == wire_items(sent)
    gained = json.loads(continued)["input"][len(wire_items(sent)):]
    assert [item["type"] for item in gained] == ["reasoning", "function_call", "function_call_output"]
    assert gained[1]["arguments"] == ARGUMENTS
    assert second.content == "It is 42."
    assert provider_served(second).continuation == {
        "status": "continued", "previous_response_id": "resp_1", "appended_items": 1,
    }
    if stream:
        assert [item for item in items if isinstance(item, str)] == ["It is 42."]


@STREAM
async def test_a_continuation_outside_a_turn_replays_no_tool_history(stream):
    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as client:
        response, _ = await _call(_adapter(), client, _history(), stream=stream)
    [raw] = wire.raw()
    assert input_types(json.loads(raw)) == ["message", "message"]
    assert b"function_call" not in raw and b"ENC-" not in raw
    assert provider_served(response).continuation == {
        "status": "boundary", "reason": "outside_turn", "omitted_tool_items": 2,
    }


@STREAM
@pytest.mark.parametrize("mismatch", ["model", "route", "call_id"])
async def test_a_continuation_without_its_snapshot_replays_no_tool_history(stream, mismatch):
    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            history = _history("call_other" if mismatch == "call_id" else "call_1")
            if mismatch == "model":
                response, _ = await _call(adapter, client, history, stream=stream, model="gpt-6.2-sol-2026-11-01")
            elif mismatch == "route":
                response, _ = await _call(OpenAIAdapter("openai-eu", native_openai=True), client, history, stream=stream)
            else:
                response, _ = await _call(adapter, client, history, stream=stream)
    raw = wire.raw()[1]
    assert set(input_types(json.loads(raw))) == {"message"}
    assert b"ENC-" not in raw and b"function_call" not in raw
    assert provider_served(response).continuation["reason"] == "no_snapshot"


@STREAM
async def test_a_tool_call_the_provider_never_made_ends_the_chain(stream):
    """An assistant tool call after the anchored reply is not the provider's:
    replaying it would present it as native history."""
    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            history = _history() + [
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_injected", "type": "function",
                    "function": {"name": "lookup", "arguments": {}},
                }]},
                {"role": "tool", "tool_call_id": "call_injected", "content": "{}"},
            ]
            response, _ = await _call(adapter, client, history, stream=stream)
            assert len(registry.snapshots()) == 1  # only the boundary's own reply
    raw = wire.raw()[1]
    assert b"ENC-" not in raw and b"function_call" not in raw
    assert provider_served(response).continuation["reason"] == "unanchored_assistant_message"


async def test_a_parameter_responses_cannot_express_is_refused_before_sending():
    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as client:
        with pytest.raises(ValueError, match="frequency_penalty"):
            await _adapter().get_response(
                client=client, model=MODEL, messages=[USER], tools=TOOLS, frequency_penalty=0.5,
            )
    assert wire.requests == []


@STREAM
async def test_a_failed_response_raises_with_its_usage_and_identity(stream):
    wire = ResponsesWire(lambda body, index: Reply(
        [], "resp_failed", status="failed",
        error={"code": "server_error", "message": "The model could not finish."},
    ))
    async with wire.client() as client:
        with pytest.raises(ResponsesAPIFailure) as raised:
            await _call(_adapter(), client, [USER], stream=stream)
    error = raised.value
    assert "server_error" in str(error) and "The model could not finish." in str(error)
    usage = reported_usage(error)
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens) == (100, 30, 20)
    for source in (error, usage):
        served = provider_served(source)
        assert (served.model, served.response_id) == (SERVED, "resp_failed")


@STREAM
async def test_a_stream_error_event_raises_with_the_identity_already_named(stream):
    wire = ResponsesWire(lambda body, index: Reply(
        [], "resp_err", error_event={"code": "rate_limit_exceeded", "message": "Slow down."},
    ))
    async with wire.client() as client:
        with pytest.raises(ResponsesAPIFailure, match="rate_limit_exceeded: Slow down."):
            await _call(_adapter(), client, [USER], stream=stream)


@STREAM
async def test_a_disconnect_after_created_keeps_the_identity(stream):
    import openai

    wire = ResponsesWire(lambda body, index: Reply([], "resp_cut", disconnect_after_created=True))
    async with wire.client() as client:
        with pytest.raises(openai.APIConnectionError) as raised:
            await _call(_adapter(), client, [USER], stream=stream)
    served = provider_served(raised.value)
    assert (served.model, served.response_id) == (SERVED, "resp_cut")


@STREAM
async def test_a_response_cut_at_its_output_limit_stops_at_length(stream):
    wire = ResponsesWire(lambda body, index: Reply([message("msg", "Partial")], "resp_cut", status="incomplete"))
    async with wire.client() as client:
        response, _ = await _call(_adapter(), client, [USER], stream=stream)
    assert response.stop_reason == "length"
    assert response.content.startswith("Partial\n\n")


@STREAM
async def test_a_prefix_mismatch_the_provider_reports_is_recorded(stream):
    wire = ResponsesWire(lambda body, index: Reply(
        [message("msg", "ok")], "resp_pm",
        extra={"input_transformations": [{"type": "removed", "reason": "prefix_mismatch", "item_id": "rs_1"}]},
    ))
    async with wire.client() as client:
        response, _ = await _call(_adapter(), client, [USER], stream=stream)
    assert provider_served(response).prefix_mismatch == ("rs_1",)


async def test_a_streamed_tool_call_is_announced_as_it_starts():
    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as client:
        _, items = await _call(_adapter(), client, [USER], stream=True)
    [started] = [item for item in items if isinstance(item, ToolCallStarted)]
    assert (started.id, started.name) == ("call_1", "lookup")


async def test_a_malformed_reply_is_a_protocol_failure_with_its_identity():
    wire = ResponsesWire(lambda body, index: Reply(
        [call("fc", "call_x", "lookup", "not json")], "resp_bad",
    ))
    async with wire.client() as client:
        with pytest.raises(GenerationProtocolError) as raised:
            await _adapter().get_response(client=client, model=MODEL, messages=[USER], tools=TOOLS)
    assert provider_served(raised.value).response_id == "resp_bad"
    usage = reported_usage(raised.value)
    assert usage.output_tokens == 30
    assert provider_served(usage).response_id == "resp_bad"


@STREAM
async def test_a_changed_reasoning_setting_ends_the_chain(stream):
    """Only appends may change after capture; a request setting that is part
    of the prefix may not."""
    from kestrel_sdk.llm import RequestOptions

    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            response, _ = await _call(
                adapter, client, _history(), stream=stream,
                request_options=RequestOptions(reasoning_effort="high"),
            )
    raw = wire.raw()[1]
    assert b"ENC-" not in raw and b"function_call" not in raw
    assert provider_served(response).continuation["reason"] == "request_settings_changed"


async def test_the_registry_expires_with_its_scope_and_keeps_a_bounded_number():
    from kestrel_sovereign.llm.responses_continuation import ResponsesSnapshot

    def snapshot(n):
        return ResponsesSnapshot(
            route="r", client=None, requested_model=MODEL, served_model=SERVED,
            response_id=f"resp_{n}", settings="{}", items=(), declared_tools=(),
            anchors=(), turn_request=None,
        )

    with responses_continuation_scope() as registry:
        for n in range(MAX_SNAPSHOTS_PER_TURN + 3):
            assert registry.record(snapshot(n))
        kept = registry.snapshots()
        assert len(kept) == MAX_SNAPSHOTS_PER_TURN
        assert kept[0].response_id == f"resp_{MAX_SNAPSHOTS_PER_TURN + 2}"
        assert current_responses_continuations() is registry
    assert registry.expired and registry.snapshots() == ()
    assert not registry.record(snapshot(99))
    assert current_responses_continuations() is None


async def test_a_task_that_predates_the_turn_sees_its_registry_through_the_turn_scope():
    seen: Dict[str, Any] = {}
    with responses_continuation_scope() as registry:
        snapshot = capture_turn_scope(object())

        async def foreign():
            seen["bare"] = current_responses_continuations()
            with snapshot.bind():
                seen["bound"] = current_responses_continuations()

        await asyncio.create_task(foreign(), context=contextvars.Context())
    assert seen == {"bare": None, "bound": registry}
    assert isinstance(registry, ResponsesContinuations)


# ---------------------------------------------------------------------------
# extra_body is part of the request, not merged over it afterwards
# ---------------------------------------------------------------------------


def _raw(extra_body: Dict[str, Any]):
    from kestrel_sdk.llm import RequestOptions

    return RequestOptions(raw={"extra_body": extra_body})


@STREAM
@pytest.mark.parametrize("before, after", [
    ({"text": {"format": {"type": "text"}}}, {"text": {"format": {"type": "json_object"}}}),
    ({"reasoning": {"effort": "medium"}}, {"reasoning": {"effort": "high"}}),
    ({"truncation": "disabled"}, {"truncation": "auto"}),
], ids=["text_format", "reasoning_effort", "unknown_key"])
async def test_a_changed_extra_body_prefix_setting_ends_the_chain(stream, before, after):
    """``extra_body`` settings are captured like the request's own: changing
    one between requests is a changed prefix, never a continuation."""
    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER], stream=stream, request_options=_raw(before))
            response, _ = await _call(
                adapter, client, _history(), stream=stream, request_options=_raw(after),
            )
    first, raw = (json.loads(body) for body in wire.raw())
    [(key, value)] = before.items()
    assert first[key] == value
    assert b"ENC-" not in wire.raw()[1] and b"function_call" not in wire.raw()[1]
    assert provider_served(response).continuation["reason"] == "request_settings_changed"
    assert raw[key] == after[key]


@STREAM
async def test_unchanged_extra_body_settings_keep_the_chain(stream):
    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    options = {"text": {"format": {"type": "text"}, "verbosity": "low"}, "truncation": "disabled"}
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER], stream=stream, request_options=_raw(options))
            response, _ = await _call(adapter, client, _history(), stream=stream, request_options=_raw(options))
    sent, continued = wire.raw()
    assert wire_items(continued)[: len(wire_items(sent))] == wire_items(sent)
    assert provider_served(response).continuation["status"] == "continued"
    for key in ("text", "truncation"):
        assert json.loads(continued)[key] == json.loads(sent)[key] == options[key]


@pytest.mark.parametrize("request_options, kwargs, named", [
    ({"reasoning": {"effort": "high"}}, {"reasoning_effort": "low"}, "reasoning"),
    ({"text": {"format": {"type": "text"}}}, {"format": "json"}, "text"),
    ({"tool_choice": "required"}, {}, "tool_choice"),
    ({"max_output_tokens": 10}, {"max_tokens": 20}, "max_output_tokens"),
    ({"temperature": 1.0}, {"temperature": 0.5}, "temperature"),
], ids=["reasoning", "text", "tool_choice", "budget", "temperature"])
async def test_a_value_given_twice_with_different_values_is_refused_before_sending(
    request_options, kwargs, named,
):
    """Whichever of two values the SDK merge left would be sent; neither is
    guessed."""
    from kestrel_sdk.llm import RequestOptions

    reasoning = kwargs.pop("reasoning_effort", None)
    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as client:
        with pytest.raises(ValueError, match=named):
            await _adapter().get_response(
                client=client, model=MODEL, messages=[USER], tools=TOOLS,
                request_options=RequestOptions(
                    reasoning_effort=reasoning, raw={"extra_body": request_options},
                ),
                **kwargs,
            )
    assert wire.requests == []


# ---------------------------------------------------------------------------
# A snapshot belongs to its configured route, not to a vendor and URL
# ---------------------------------------------------------------------------


def _registry_routes():
    """Two configured routes as the provider registry builds them:
    ``openai:primary`` and ``openai:other`` share the adapter name and the
    official base URL; only their route names, adapters and clients (and so
    their accounts) differ."""
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    providers = ProviderRegistry({})
    return [
        providers._build_route(
            "openai", name, {"is_cloud": True},
            {"adapter": "OpenAIAdapter", "api_key": f"key-{name}", "model": MODEL},
        )
        for name in ("primary", "other")
    ]


@STREAM
@pytest.mark.parametrize("clients", ["own_clients", "one_shared_client"])
async def test_real_registry_routes_cannot_share_a_snapshot(stream, clients):
    """Two distinct configured routes never continue each other's chain, even
    sent through one client: a snapshot is bound to its route's name and its
    own adapter, not to a vendor and URL."""
    primary, other = _registry_routes()
    try:
        assert primary.name == "openai:primary" and other.name == "openai:other"
        assert primary.adapter.name == other.adapter.name == "openai"
        assert primary.adapter is not other.adapter
        assert str(primary.client.base_url) == str(other.client.base_url)
        assert primary.client is not other.client
    finally:
        await primary.client.close()
        await other.client.close()

    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as one, wire.client() as two:
        other_client = one if clients == "one_shared_client" else two
        with responses_continuation_scope():
            await _call(primary.adapter, one, [SYSTEM, USER], stream=stream)
            crossed, _ = await _call(other.adapter, other_client, _history(), stream=stream)
            own, _ = await _call(primary.adapter, one, _history(), stream=stream)
    first, other_request, own_request = wire.raw()
    assert b"ENC-" not in other_request and b"function_call" not in other_request
    assert provider_served(crossed).continuation["status"] == "boundary"
    assert provider_served(crossed).continuation["reason"] == "no_snapshot"
    # The chain is intact on its own route.
    assert wire_items(own_request)[: len(wire_items(first))] == wire_items(first)
    assert provider_served(own).continuation["status"] == "continued"


@STREAM
async def test_two_adapters_with_one_label_and_client_never_share_a_snapshot(stream):
    """Two adapters built outside the registry carry the same label (vendor
    and base URL) and here even one client; each is still its own route."""
    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(_adapter(), client, [SYSTEM, USER], stream=stream)
            crossed, _ = await _call(_adapter(), client, _history(), stream=stream)
    assert b"ENC-" not in wire.raw()[1] and b"function_call" not in wire.raw()[1]
    assert provider_served(crossed).continuation["reason"] == "no_snapshot"


async def test_a_registry_built_openai_adapter_knows_its_route_name():
    """The snapshot binding's route name is the configured route's own."""
    primary, other = _registry_routes()
    try:
        assert (primary.adapter.route, other.adapter.route) == ("openai:primary", "openai:other")
        assert primary.adapter._responses_route(primary.client).label == "openai:primary"
    finally:
        await primary.client.close()
        await other.client.close()
    assert _adapter()._responses_route(object()).label.startswith("openai|")


# ---------------------------------------------------------------------------
# Endpoint selection follows the continuation, not only the tools offered
# ---------------------------------------------------------------------------


@STREAM
async def test_continuation_with_no_current_tools_keeps_responses(stream):
    """``continue_with_tool_results`` defaults to no tools, and a streamed
    continuation may offer none: the chain stays on the Responses API and
    continues, its declarations are kept, and the model is told to call none
    of them."""
    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    results = [{"tool_call_id": "call_1", "content": '{"value": 42}'}]
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER, OPERATOR], stream=stream)
            if stream:
                response, _ = await _call(adapter, client, _history(), stream=True, tools=None)
            else:
                response = await adapter.continue_with_tool_results(
                    client, MODEL, _history()[:-1], results,
                )
    assert not wire.unexpected
    assert [request.url.path for request in wire.requests] == ["/v1/responses", "/v1/responses"]
    sent, continued = wire.raw()
    assert wire_items(continued)[: len(wire_items(sent))] == wire_items(sent)
    assert wire_items(continued, "tools") == wire_items(sent, "tools")
    assert json.loads(continued)["tool_choice"] == "none"
    assert provider_served(response).continuation["status"] == "continued"
    assert response.content == "It is 42."


@STREAM
async def test_a_snapshot_turn_never_leaves_the_responses_api(stream):
    """A later request on the route that no snapshot accounts for (its
    earlier reply's text was rewritten, so nothing locates it) and that
    offers no tools stays on the Responses API, as an ordinary conversation
    that replays nothing, never on chat completions."""
    wire = ResponsesWire(lambda body, index: Reply(
        [reasoning(f"rs_{index}", f"ENC-{index}"), message(f"msg_{index}", f"Answer {index}.")],
        f"resp_{index}",
    ))
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            response, _ = await _call(
                adapter, client,
                [SYSTEM, USER, {"role": "assistant", "content": "A rewritten answer."},
                 {"role": "user", "content": "Go on."}],
                stream=stream, tools=None,
            )
    assert not wire.unexpected
    assert [request.url.path for request in wire.requests] == ["/v1/responses", "/v1/responses"]
    assert b"ENC-" not in wire.raw()[1] and b'"encrypted_content":' not in wire.raw()[1]
    assert input_types(json.loads(wire.raw()[1])) == ["message"] * 4
    assert provider_served(response).continuation == {"status": "fresh"}


@STREAM
async def test_tool_history_offered_no_tools_meets_the_boundary_not_chat_completions(stream):
    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as client:
        response, _ = await _call(_adapter(), client, _history(), stream=stream, tools=None)
    assert not wire.unexpected
    [raw] = wire.raw()
    assert input_types(json.loads(raw)) == ["message", "message"]
    assert b"function_call" not in raw and b"ENC-" not in raw
    assert provider_served(response).continuation == {
        "status": "boundary", "reason": "outside_turn", "omitted_tool_items": 2,
    }


@STREAM
async def test_a_continuation_offering_fewer_tools_restricts_them_with_allowed_tools(stream):
    note = {"type": "function", "function": {
        "name": "note", "description": "Take a note.",
        "parameters": {"type": "object", "properties": {}},
    }}
    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER], stream=stream, tools=TOOLS + [note])
            response, _ = await _call(adapter, client, _history(), stream=stream, tools=TOOLS)
    sent, continued = wire.raw()
    assert wire_items(continued, "tools") == wire_items(sent, "tools")
    assert json.loads(continued)["tool_choice"] == {
        "type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "lookup"}],
    }
    assert "tool_choice" not in json.loads(sent) or json.loads(sent)["tool_choice"] == "auto"
    assert provider_served(response).continuation["status"] == "continued"


# ---------------------------------------------------------------------------
# A continuation is measured, as sent, against the turn's context budget
# ---------------------------------------------------------------------------

LONG = "An earlier answer that the tool loop later pruned. " * 60
EARLIER = [{"role": "user", "content": "Tell me something."}, {"role": "assistant", "content": LONG}]


def _pruned_history() -> List[Dict[str, Any]]:
    """The continuation after the loop shed the replayed earlier exchange."""
    return _history()


@STREAM
@pytest.mark.parametrize("fits", [False, True], ids=["over_budget", "within_budget"])
async def test_a_continuation_over_the_turn_context_budget_ends_the_chain(stream, fits):
    """Nothing excluded: the continuation still carries the whole prefix, and
    only its size against the turn's budget decides."""
    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            await _call(adapter, client, [SYSTEM, *EARLIER, USER], stream=stream)
            first = wire.raw()[0]
            budget = len(LONG) * 2 if fits else len(LONG) // 2
            registry.limit_context(budget)
            history = _history()
            response, _ = await _call(
                adapter, client, [history[0], *EARLIER, *history[1:]], stream=stream,
            )
            # The ended chain is retired; the boundary reply starts a new one.
            kept = [snapshot.response_id for snapshot in registry.snapshots()]
    continued = wire.raw()[1]
    served = provider_served(response).continuation
    if fits:
        assert served["status"] == "continued"
        assert wire_items(continued)[: len(wire_items(first))] == wire_items(first)
        return
    assert b"ENC-" not in continued and b"function_call" not in continued
    assert served["reason"] == "budget_exceeded"
    assert served["budget_chars"] == budget < served["request_chars"]
    assert kept == ["resp_2"]


@STREAM
async def test_pruned_budget_is_not_silently_reinflated(stream):
    """The real prune sheds replayed history to fit; the replay never puts it
    back. The chain ends at ``prefix_pruned`` instead."""
    from types import SimpleNamespace

    from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin

    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    prior = [{"role": "user", "content": "OLD-" * 750}, {"role": "assistant", "content": "ack"}]
    initial = [SYSTEM, *prior, USER]
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, initial, stream=stream)
            continuation = [
                *initial, _history()[2],
                {"role": "tool", "tool_call_id": "call_1", "content": "NEW-" * 1000},
            ]
            pruned = OrchestratorEngineMixin._prune_orchestrator_messages(
                SimpleNamespace(), continuation, TOOLS,
                context_limit=2000, protected_prefix=4, history_len=2,
            )
            assert prior[0] not in pruned
            response, _ = await _call(adapter, client, pruned, stream=stream)
    sent = wire.raw()[1]
    assert b"OLD-" not in sent, "the real prune shed history to fit; the replay must not restore it"
    assert b"ENC-" not in sent and b"function_call" not in sent
    boundary = provider_served(response).continuation
    assert (boundary["status"], boundary["reason"]) == ("boundary", "prefix_pruned")


def _three_steps(body, index):
    if index == 0:
        return Reply([reasoning("rs_1", "ENC-1"), call("fc_1", "call_1", "lookup", ARGUMENTS)], "resp_1")
    if index == 1:
        return Reply([reasoning("rs_2", "ENC-2"), call("fc_2", "call_2", "lookup", "{}")], "resp_2")
    return Reply([message(f"msg_{index}", "Done.")], f"resp_{index + 1}")


@STREAM
@pytest.mark.parametrize("excluded", ["history_message", "tool_output"])
async def test_excluded_history_within_budget_is_not_replayed(stream, excluded):
    """One item the chain sent is no longer in the caller's list (an
    ordinary history message, or a tool output stubbed by the prune), and the
    replay would still fit the budget: no excluded input and no opaque
    replay is transmitted, and the chain ends at ``prefix_pruned``."""
    from kestrel_sovereign.agent.orchestrator_engine import _pruned_tool_result_stub

    big_output = "TOOL-OUTPUT-" * 50
    wire = ResponsesWire(_three_steps)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            registry.limit_context(len(LONG) * 10)
            history = _history()
            await _call(adapter, client, [history[0], *EARLIER, history[1]], stream=stream)
            assert LONG.encode() in wire.raw()[0]
            if excluded == "history_message":
                kept = history
                secret = LONG
            else:
                first_turn = [history[0], *EARLIER, history[1], history[2],
                              {"role": "tool", "tool_call_id": "call_1", "content": big_output}]
                await _call(adapter, client, first_turn, stream=stream)
                assert big_output.encode() in wire.raw()[1]
                second_call = {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_2", "type": "function", "function": {"name": "lookup", "arguments": {}},
                }]}
                kept = [
                    *first_turn[:-1],
                    {"role": "tool", "tool_call_id": "call_1", "content": _pruned_tool_result_stub(big_output)},
                    second_call, {"role": "tool", "tool_call_id": "call_2", "content": "{}"},
                ]
                secret = big_output
            response, _ = await _call(adapter, client, kept, stream=stream)
    sent = wire.raw()[-1]
    assert secret.encode() not in sent
    assert b"ENC-" not in sent and b'"encrypted_content":' not in sent
    assert b"function_call" not in sent
    boundary = provider_served(response).continuation
    assert (boundary["status"], boundary["reason"]) == ("boundary", "prefix_pruned")
    assert registry.expired


async def test_a_context_budget_is_a_positive_number_of_characters():
    registry = ResponsesContinuations()
    for bad in (0, -1, 1.5, True, None):
        with pytest.raises(ValueError):
            registry.limit_context(bad)
    registry.limit_context(10)
    assert registry.context_budget == 10


@STREAM
async def test_a_text_reply_continued_without_tools_keeps_its_chain(stream):
    """A repair prompt after a text-only reply carries no tools and no tool
    history; it still continues the snapshot the reply belongs to."""
    wire = ResponsesWire(lambda body, index: Reply(
        [reasoning(f"rs_{index}", f"ENC-{index}"), message(f"msg_{index}", f"Answer {index}.")],
        f"resp_{index}",
    ))
    adapter = _adapter()
    repair = {"role": "user", "content": "Please finish the answer."}
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            response, _ = await _call(
                adapter, client,
                [SYSTEM, USER, {"role": "assistant", "content": "Answer 0."}, repair],
                stream=stream, tools=None,
            )
    assert not wire.unexpected
    sent, continued = wire.raw()
    assert wire_items(continued)[: len(wire_items(sent))] == wire_items(sent)
    assert json.loads(continued)["input"][-1] == {"type": "message", "role": "user", "content": repair["content"]}
    assert json.loads(continued)["tool_choice"] == "none"
    assert provider_served(response).continuation["status"] == "continued"


async def test_extra_body_can_never_overwrite_a_field_the_body_sets():
    """The SDK merges ``extra_body`` over the body. Translation keeps every
    key the body sets out of it; the serializer refuses one regardless."""
    from kestrel_sovereign.llm.openai_responses import ResponsesRoute, _finish_request

    with pytest.raises(ValueError, match="overwrite"):
        _finish_request(
            settings={"tools": [], "include": [], "store": False, "extra_body": {"store": True}},
            params={}, items=(), route=ResponsesRoute(label="r", client=None), model=MODEL,
            request_digest="", declared=(), tool_definitions=(), anchors=(), carried=(), notices_from=0,
            turn_request=None, continuation={}, parent=None, registry=None,
        )


# ---------------------------------------------------------------------------
# The binding digest covers the merged provider-visible body (#4198 item i)
# ---------------------------------------------------------------------------


@STREAM
@pytest.mark.parametrize("before, after", [
    ({"extra_body": {"text": {"format": {"type": "text"}}}},
     {"extra_body": {"text": {"format": {"type": "json_object"}}}}),
    ({"extra_body": {"reasoning": {"effort": "medium", "summary": "auto"}}},
     {"extra_body": {"reasoning": {"effort": "medium", "summary": "detailed"}}}),
    ({"temperature": 0.2}, {"temperature": 0.7}),
    ({"extra_body": {"metadata": {"run": "a"}}}, {"extra_body": {"metadata": {"run": "b"}}}),
], ids=["extra_body_text_format", "extra_body_reasoning_summary", "temperature", "extra_body_metadata"])
async def test_nested_prefix_setting_changes_end_the_chain(stream, before, after):
    """A provider-visible setting that changes between two requests of a
    chain, whether nested in ``extra_body`` or a top-level parameter, ends the
    chain: nothing encrypted is replayed under settings the reasoning was not
    produced with."""
    from kestrel_sdk.llm import RequestOptions

    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER], stream=stream,
                        request_options=RequestOptions(raw=before))
            response, _ = await _call(adapter, client, _history(), stream=stream,
                                      request_options=RequestOptions(raw=after))
    first, second = wire.bodies()
    assert first != second
    continuation = provider_served(response).continuation
    assert continuation["status"] == "boundary", (continuation, second)
    assert continuation["reason"] == "request_settings_changed"
    assert b"ENC-1" not in wire.raw()[1] and b"function_call" not in wire.raw()[1]


async def test_the_binding_digest_covers_the_merged_body_but_not_the_input_or_tool_offer():
    from kestrel_sovereign.llm.openai_responses import request_digest

    base = {"tools": TOOLS, "tool_choice": "auto", "extra_body": {"text": {"verbosity": "low"}}}
    digest = request_digest(model=MODEL, request=base)
    # The per-request tool offer may change without changing the binding.
    assert request_digest(model=MODEL, request={**base, "tools": None, "tool_choice": None}) == digest
    # Every other provider-visible field is bound, wherever it is given.
    for changed in (
        {**base, "extra_body": {"text": {"verbosity": "high"}}},
        {**base, "extra_body": {"text": {"verbosity": "low"}, "truncation": "auto"}},
        {**base, "temperature": 0.5},
        {**base, "max_completion_tokens": 64},
        {**base, "reasoning_effort": "high"},
    ):
        assert request_digest(model=MODEL, request=changed) != digest, changed
    assert request_digest(model="gpt-6.2-sol-2026-11-01", request=base) != digest


# ---------------------------------------------------------------------------
# Cancellation and disconnects keep the evidence already observed (m, n)
# ---------------------------------------------------------------------------


@STREAM
async def test_a_cancelled_stream_is_re_raised_with_the_identity_already_named(stream):
    """The inactivity watchdog stops a silent stream by cancelling it: the
    cancellation is re-raised, never swallowed, and carries the model and
    response id ``response.created`` named, so the timeout that wraps it
    reaches the failed attempt's row with them."""
    wire = StallingWire()
    async with wire.client() as client:
        with pytest.raises(TimeoutError) as raised:
            async with asyncio.timeout(0.2):
                await _call(_adapter(), client, [USER], stream=stream)
    cancelled = raised.value.__cause__
    assert isinstance(cancelled, asyncio.CancelledError)
    served = provider_served(raised.value)
    assert (served.model, served.response_id) == (SERVED, "resp_stall")


@STREAM
@pytest.mark.parametrize("usage", ["reported", "none_reported"])
async def test_a_disconnect_after_the_terminal_event_keeps_its_usage(stream, usage):
    """The terminal response arrived and the connection dropped before the
    stream's clean end: the attempt is incomplete, its error keeps the usage
    the terminal response reported (or reports none, never zero) and the
    provider's identification, and is raised from the transport error."""
    import openai

    from kestrel_sovereign.llm.output_ceiling import incomplete_generation

    wire = ResponsesWire(lambda body, index: Reply(
        [message("msg", "Done")], "resp_terminal", disconnect_after_terminal=True,
        **({} if usage == "reported" else {"usage": None}),
    ))
    async with wire.client() as client:
        with pytest.raises(Exception) as raised:
            await _call(_adapter(), client, [USER], stream=stream)
    error = raised.value
    # The attempt is marked incomplete, raised from the transport failure.
    assert incomplete_generation(error) is error
    assert isinstance(error.__cause__, openai.APIConnectionError)
    served = provider_served(error)
    assert (served.model, served.response_id) == (SERVED, "resp_terminal")
    counts = reported_usage(error)
    if usage == "none_reported":
        assert counts is None
    else:
        assert (counts.input_tokens, counts.output_tokens, counts.cache_read_input_tokens) == (100, 30, 20)
        assert provider_served(counts).response_id == "resp_terminal"
    from kestrel_sovereign.llm.openai_responses import ResponsesStreamInterrupted

    assert isinstance(error, ResponsesStreamInterrupted)


@STREAM
async def test_a_close_failure_after_a_clean_end_is_not_a_lost_stream(stream, monkeypatch):
    """Only a transport failure before the stream's clean end loses the
    stream; a failure closing it afterwards is raised as itself, with the
    identity already named."""
    from kestrel_sovereign.llm import openai_adapter

    real_close = openai_adapter.close_native_execution_stream

    async def failing_close(stream_handle, owner):
        await real_close(stream_handle, owner)
        raise RuntimeError("close failed")

    monkeypatch.setattr(openai_adapter, "close_native_execution_stream", failing_close)
    wire = ResponsesWire(lambda body, index: Reply([message("msg", "Done")], "resp_closed"))
    async with wire.client() as client:
        with pytest.raises(RuntimeError, match="close failed") as raised:
            await _call(_adapter(), client, [USER], stream=stream)
    assert provider_served(raised.value).response_id == "resp_closed"


# ---------------------------------------------------------------------------
# A reported prefix mismatch ends the chain (p)
# ---------------------------------------------------------------------------

_MISMATCH = {"input_transformations": [{"type": "removed", "reason": "prefix_mismatch", "item_id": "rs_1"}]}


def _degraded(body, index):
    if index == 0:
        return Reply([reasoning("rs_1", "ENC-1"), call("fc_1", "call_1", "lookup", ARGUMENTS)], "resp_1")
    if index == 1:
        return Reply([reasoning("rs_2", "ENC-2"), call("fc_2", "call_2", "lookup", "{}")], "resp_2",
                     extra=_MISMATCH)
    return Reply([message(f"msg_{index}", "Done.")], f"resp_{index + 1}")


def _second_call(history):
    return history + [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_2", "type": "function", "function": {"name": "lookup", "arguments": {}},
        }]},
        {"role": "tool", "tool_call_id": "call_2", "content": "{}"},
    ]


@STREAM
async def test_a_reported_prefix_mismatch_ends_the_chain(stream):
    """The provider reported dropping replayed input: the reply is used, but
    the degraded chain is never replayed again."""
    wire = ResponsesWire(_degraded)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            degraded, _ = await _call(adapter, client, _history(), stream=stream)
            after, _ = await _call(adapter, client, _second_call(_history()), stream=stream)
            live = [snapshot.response_id for snapshot in registry.snapshots()]
    assert [call.id for call in degraded.tool_calls] == ["call_2"]
    served = provider_served(degraded)
    assert served.prefix_mismatch == ("rs_1",)
    assert served.continuation["status"] == "continued"
    assert served.continuation["chain_ended"] == "prefix_mismatch"
    assert served.snapshot == {"stored": False, "reason": "prefix_mismatch"}
    sent = wire.raw()[2]
    assert b"ENC-" not in sent and b"function_call" not in sent
    boundary = provider_served(after).continuation
    assert (boundary["status"], boundary["reason"], boundary["retired_reason"]) == (
        "boundary", "retired", "prefix_mismatch",
    )
    assert live == ["resp_3"]


# ---------------------------------------------------------------------------
# Evidence: the request sent, and the snapshot actually kept (q)
# ---------------------------------------------------------------------------


@STREAM
async def test_evidence_separates_the_request_sent_from_the_snapshot_kept(stream):
    """Two continuations of one snapshot in flight at once: both were sent as
    continuations, only one reply's snapshot is kept, and the other's
    evidence says why. A later request that continues the superseded chain
    is told it was retired, never that there was no snapshot."""
    wire = ResponsesWire(lambda body, index: (
        Reply([reasoning("rs_1", "ENC-1"), call("fc_1", "call_1", "lookup", ARGUMENTS)], "resp_1")
        if index == 0 else
        Reply([reasoning(f"rs_{index}", f"ENC-{index}"), call(f"fc_{index}", f"call_{index + 1}", "lookup", "{}")],
              f"resp_{index + 1}")
    ))
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            first, _ = await _call(adapter, client, [SYSTEM, USER], stream=stream)
            racing = await asyncio.gather(
                _call(adapter, client, _history(), stream=stream),
                _call(adapter, client, _history(), stream=stream),
            )
            stale, _ = await _call(adapter, client, _history(), stream=stream)
            live = [snapshot.response_id for snapshot in registry.snapshots()]
    boundary = provider_served(stale).continuation
    assert boundary["reason"] == "retired" and boundary["retired_reason"] == "superseded"
    assert b"ENC-" not in wire.raw()[3]
    assert provider_served(first).snapshot == {"stored": True}
    outcomes = [provider_served(response) for response, _ in racing]
    assert [served.continuation["status"] for served in outcomes] == ["continued", "continued"]
    assert sorted(served.snapshot["stored"] for served in outcomes) == [False, True]
    [lost] = [served for served in outcomes if not served.snapshot["stored"]]
    assert lost.snapshot == {"stored": False, "reason": "parent_retired", "parent_reason": "superseded"}
    [kept] = [served for served in outcomes if served.snapshot["stored"]]
    # The kept reply's chain, and the chain the stale request's boundary
    # reply started; never the reply whose snapshot was not kept.
    assert live == ["resp_4", kept.response_id]
    assert lost.response_id not in live


@STREAM
async def test_outside_a_turn_the_evidence_says_no_snapshot_was_kept(stream):
    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as client:
        response, _ = await _call(_adapter(), client, [SYSTEM, USER], stream=stream)
    served = provider_served(response)
    assert served.continuation == {"status": "fresh"}
    assert served.snapshot == {"stored": False, "reason": "outside_turn"}
    assert served.as_metadata()["snapshot"] == {"stored": False, "reason": "outside_turn"}


async def test_the_registry_records_why_each_snapshot_was_retired():
    from kestrel_sovereign.llm.responses_continuation import (
        MAX_RETIRED_PER_TURN,
        ResponsesAnchor,
        ResponsesSnapshot,
    )

    def snapshot(n):
        return ResponsesSnapshot(
            route="r", client=None, requested_model=MODEL, served_model=SERVED,
            response_id=f"resp_{n}", settings="{}", items=(), declared_tools=(),
            anchors=(ResponsesAnchor(content="", call_ids=(f"call_{n}",)),), turn_request=None,
        )

    with responses_continuation_scope() as registry:
        snapshots = [snapshot(n) for n in range(MAX_SNAPSHOTS_PER_TURN + 1)]
        for kept in snapshots:
            assert registry.record(kept)
        assert registry.retirement_of(snapshots[0]) == "evicted"
        extended = snapshot(100)
        assert registry.record(extended, replaces=snapshots[1])
        assert registry.retirement_of(snapshots[1]) == "superseded"
        assert not registry.record(snapshot(101), replaces=snapshots[1])
        registry.discard(snapshots[2], "prefix_pruned")
        assert registry.retirement_of(snapshots[2]) == "prefix_pruned"
        with pytest.raises(ValueError):
            registry.discard(snapshots[3], "not_a_reason")
        for n in range(MAX_RETIRED_PER_TURN + 5):
            registry.retire(snapshot(200 + n), "served_model_mismatch")
        assert len(registry.retired()) == MAX_RETIRED_PER_TURN
        assert registry.retirement_of(snapshots[0]) is None  # forgotten: bounded
    assert registry.retired() == ()



# ---------------------------------------------------------------------------
# Full-diff review findings, each pinned (#4198)
# ---------------------------------------------------------------------------


async def test_a_snapshot_turn_stays_on_the_responses_api_after_its_records_are_evicted():
    """However many snapshots and retirements a turn holds on other routes,
    a route it once held a snapshot on stays a snapshot turn."""
    from kestrel_sovereign.llm.openai_responses import ResponsesRoute
    from kestrel_sovereign.llm.responses_continuation import (
        MAX_RETIRED_PER_TURN,
        ResponsesAnchor,
        ResponsesSnapshot,
    )

    sol = ResponsesRoute(label="openai:sol", client=object(), adapter=object())
    other = ResponsesRoute(label="openai:other", client=object(), adapter=object())

    def snapshot(route, n):
        return ResponsesSnapshot(
            route=route.label, client=route.client, adapter=route.adapter,
            requested_model=MODEL, served_model=SERVED, response_id=f"resp_{route.label}_{n}",
            settings="{}", items=(), declared_tools=(),
            anchors=(ResponsesAnchor(content="", call_ids=(f"call_{n}",)),), turn_request=None,
        )

    def stays(route):
        return requires_responses_api(
            native_openai=True, model=MODEL, request={}, messages=[SYSTEM, USER], route=route,
        )

    with responses_continuation_scope():
        # Control: a tool-free request on a route the turn never used.
        assert stays(sol) is False
    with responses_continuation_scope() as registry:
        registry.record(snapshot(sol, 0))
        for n in range(MAX_SNAPSHOTS_PER_TURN + MAX_RETIRED_PER_TURN + 5):
            registry.record(snapshot(other, n))
        assert not any(sol.serves(kept) for kept in registry.snapshots())
        assert not any(sol.serves(record) for record in registry.retired())
        assert stays(sol) is True
        assert stays(ResponsesRoute(label="openai:unused", client=object(), adapter=object())) is False


@STREAM
async def test_a_parsing_bug_after_the_terminal_event_is_not_a_lost_stream(stream, monkeypatch):
    """Only a transport failure loses a finished stream; a bug handling an
    event after the terminal one is raised as itself."""
    from kestrel_sovereign.llm.openai_responses import ResponsesEventState

    original = ResponsesEventState.observe

    def observe(self, event):
        if getattr(event, "type", None) == "response.trailing":
            raise KeyError("trailing event bug")
        return original(self, event)

    monkeypatch.setattr(ResponsesEventState, "observe", observe)
    wire = ResponsesWire(lambda body, index: Reply(
        [message("msg", "Done")], "resp_trailing", after_terminal=[{"type": "response.trailing"}],
    ))
    async with wire.client() as client:
        with pytest.raises(KeyError, match="trailing event bug") as raised:
            await _call(_adapter(), client, [USER], stream=stream)
    assert provider_served(raised.value).response_id == "resp_trailing"


async def test_typed_tools_given_as_a_tuple_are_declared():
    from kestrel_sovereign.llm.openai_responses import ResponsesRoute

    route = ResponsesRoute(label="openai|", client=object())
    assert requires_responses_api(
        native_openai=True, model=MODEL, request={"tools": tuple(TOOLS)}, messages=[USER], route=route,
    ) is True
    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as client:
        await _call(_adapter(), client, [SYSTEM, USER], stream=False, tools=tuple(TOOLS))
    assert [tool["name"] for tool in wire.bodies()[0]["tools"]] == ["lookup"]
    with pytest.raises(ValueError, match="tools must be a list"):
        requires_responses_api(
            native_openai=True, model=MODEL, request={"tools": {"lookup": TOOLS[0]}},
            messages=[USER], route=route,
        )


def _round_two(history):
    return history + [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_2", "type": "function", "function": {"name": "lookup", "arguments": {}},
        }]},
        {"role": "tool", "tool_call_id": "call_2", "content": "{}"},
    ]


@STREAM
async def test_a_dropped_earlier_round_ends_the_chain_as_pruned(stream):
    """The caller kept the chain's latest reply but dropped an earlier round:
    the prefix was pruned, the chain ends, nothing is replayed."""
    wire = ResponsesWire(_three_steps)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            await _call(adapter, client, _history(), stream=stream)
            dropped = [SYSTEM, USER] + _round_two([])
            response, _ = await _call(adapter, client, dropped, stream=stream)
            live = [snapshot.response_id for snapshot in registry.snapshots()]
    sent = wire.raw()[2]
    assert b"ENC-" not in sent and b"function_call" not in sent
    boundary = provider_served(response).continuation
    assert (boundary["status"], boundary["reason"]) == ("boundary", "prefix_pruned")
    assert "resp_2" not in live


@STREAM
async def test_an_insertion_inside_the_prefix_ends_the_chain_as_changed(stream):
    wire = ResponsesWire(_three_steps)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            await _call(adapter, client, _history(), stream=stream)
            inserted = _round_two(_history() + [{"role": "assistant", "content": "An aside."}])
            response, _ = await _call(adapter, client, inserted, stream=stream)
    assert b"ENC-" not in wire.raw()[2]
    boundary = provider_served(response).continuation
    assert (boundary["status"], boundary["reason"]) == ("boundary", "prefix_changed")


@STREAM
async def test_only_notices_directly_after_the_user_turn_are_exempt(stream):
    """A trailing developer message after an assistant reply is not the
    turn's notice: a continuation that drops it pruned the prefix."""
    note = {"role": "developer", "content": "NOTE-X: answer briefly."}
    earlier = {"role": "assistant", "content": "An earlier answer."}
    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER, earlier, note], stream=stream)
            history = _history()
            response, _ = await _call(adapter, client, [*history[:2], earlier, *history[2:]], stream=stream)
    sent = wire.raw()[1]
    assert b"NOTE-X" not in sent and b"ENC-" not in sent
    assert provider_served(response).continuation["reason"] == "prefix_pruned"


@STREAM
async def test_a_tool_redefined_under_its_declared_name_ends_the_chain(stream):
    redefined = [{"type": "function", "function": {
        "name": "lookup", "description": "Look a value up, differently.",
        "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
    }}]
    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            response, _ = await _call(adapter, client, _history(), stream=stream, tools=redefined)
    sent = json.loads(wire.raw()[1])
    assert b"ENC-" not in wire.raw()[1]
    assert sent["tools"][0]["description"] == "Look a value up, differently."
    assert provider_served(response).continuation["reason"] == "request_settings_changed"


@STREAM
async def test_an_unidentified_continuation_reply_is_not_called_a_model_mismatch(stream):
    wire = ResponsesWire(lambda body, index: (
        _first_then_answer(body, index) if index == 0
        else Reply([reasoning("rs_2", "ENC-2"), message("msg_2", "It is 42.")], "resp_2", model=None)
    ))
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            response, _ = await _call(adapter, client, _history(), stream=stream)
            live = registry.snapshots()
    served = provider_served(response)
    assert served.continuation["chain_ended"] == "unidentified_response"
    assert "expected_model" not in served.continuation
    assert served.snapshot == {"stored": False, "reason": "unidentified_response"}
    assert live == ()


@STREAM
async def test_a_failed_reply_reporting_a_prefix_mismatch_still_ends_the_chain(stream):
    """The provider reported dropping replayed input on a reply that then
    failed: a retry of the same continuation replays nothing."""
    wire = ResponsesWire(lambda body, index: (
        _first_then_answer(body, index) if index == 0 else
        Reply([], f"resp_{index + 1}", status="failed",
              error={"code": "server_error", "message": "failed"}, extra=_MISMATCH)
        if index == 1 else _first_then_answer(body, index)
    ))
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            with pytest.raises(ResponsesAPIFailure):
                await _call(adapter, client, _history(), stream=stream)
            retried, _ = await _call(adapter, client, _history(), stream=stream)
    assert b"ENC-1" in wire.raw()[1]
    assert b"ENC-" not in wire.raw()[2]
    boundary = provider_served(retried).continuation
    assert (boundary["reason"], boundary["retired_reason"]) == ("retired", "prefix_mismatch")


@STREAM
async def test_a_disconnect_after_a_reported_prefix_mismatch_still_ends_the_chain(stream):
    """The terminal response reported a prefix mismatch and the connection
    then dropped before the stream's clean end. The chain ends when the
    terminal response arrives, not at the clean end: the interrupted
    attempt's error keeps the mismatch, and a retry of the same continuation
    replays nothing from the degraded chain."""
    import openai

    from kestrel_sovereign.llm.openai_responses import ResponsesStreamInterrupted

    wire = ResponsesWire(lambda body, index: (
        Reply([reasoning("rs_2", "ENC-2"), call("fc_2", "call_2", "lookup", "{}")], "resp_2",
              extra=_MISMATCH, disconnect_after_terminal=True)
        if index == 1 else _first_then_answer(body, index)
    ))
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            with pytest.raises(ResponsesStreamInterrupted) as raised:
                await _call(adapter, client, _history(), stream=stream)
            live_after_drop = [snapshot.response_id for snapshot in registry.snapshots()]
            retried, _ = await _call(adapter, client, _history(), stream=stream)
    error = raised.value
    assert isinstance(error.__cause__, openai.APIConnectionError)
    served = provider_served(error)
    assert served.response_id == "resp_2"
    assert served.prefix_mismatch == ("rs_1",)
    assert served.continuation["chain_ended"] == "prefix_mismatch"
    assert provider_served(reported_usage(error)).prefix_mismatch == ("rs_1",)
    assert live_after_drop == []
    assert b"ENC-1" in wire.raw()[1]
    assert b"ENC-" not in wire.raw()[2]
    boundary = provider_served(retried).continuation
    assert (boundary["status"], boundary["reason"], boundary["retired_reason"]) == (
        "boundary", "retired", "prefix_mismatch",
    )


async def test_the_turn_registry_expires_even_when_ownership_retirement_fails(monkeypatch):
    from types import SimpleNamespace

    from kestrel_sovereign.agent import turn_lifecycle

    class FailingOwnership:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            raise RuntimeError("ownership retirement failed")

    real_scope = turn_lifecycle.responses_continuation_scope
    exits: List[Any] = []

    class ObservedScope:
        """The real scope, recording that the lifecycle exited it itself
        (garbage collection of an abandoned scope would not call this)."""

        def __init__(self):
            self.inner = real_scope()

        def __enter__(self):
            return self.inner.__enter__()

        def __exit__(self, *exc):
            exits.append(exc)
            return self.inner.__exit__(*exc)

    monkeypatch.setattr(turn_lifecycle, "publish_turn_ownership", lambda agent, turn_id: FailingOwnership())
    monkeypatch.setattr(turn_lifecycle, "responses_continuation_scope", ObservedScope)
    host = SimpleNamespace(_turn_trace_index=lambda: {})
    seen: Dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="ownership retirement failed"):
        async with turn_lifecycle.TurnLifecycleMixin._active_turn_scope(host, "turn-4198", "test", 0.0):
            seen["registry"] = current_responses_continuations()
    assert len(exits) == 1
    assert seen["registry"] is not None and seen["registry"].expired


def _route_snapshot(route_label, n, *, client=None, adapter=None):
    from kestrel_sovereign.llm.responses_continuation import ResponsesAnchor, ResponsesSnapshot

    return ResponsesSnapshot(
        route=route_label, client=client, adapter=adapter, requested_model=MODEL,
        served_model=SERVED, response_id=f"resp_{route_label}_{n}", settings="{}", items=(),
        declared_tools=(), anchors=(ResponsesAnchor(content="", call_ids=(f"call_{route_label}_{n}",)),),
        turn_request=None,
    )


async def test_a_route_with_a_live_snapshot_is_never_forgotten():
    from kestrel_sovereign.llm.responses_continuation import MAX_ROUTES_PER_TURN

    with responses_continuation_scope() as registry:
        live = _route_snapshot("openai:live", 0)
        registry.record(live)
        for n in range(MAX_ROUTES_PER_TURN + 10):
            registry.retire(_route_snapshot(f"openai:other-{n}", n), "served_model_mismatch")
        routes = registry.routes()
        assert len(routes) == MAX_ROUTES_PER_TURN
        assert any(key.route == "openai:live" for key in routes)


async def test_routine_retirements_never_push_out_a_chain_ending_one():
    from kestrel_sovereign.llm.responses_continuation import MAX_RETIRED_PER_TURN

    with responses_continuation_scope() as registry:
        ended = _route_snapshot("openai:a", 0)
        registry.record(ended)
        registry.discard(ended, "prefix_pruned")
        step = _route_snapshot("openai:b", 0)
        registry.record(step)
        for n in range(1, MAX_RETIRED_PER_TURN + 10):
            following = _route_snapshot("openai:b", n)
            assert registry.record(following, replaces=step)
            step = following
        assert len(registry.retired()) == MAX_RETIRED_PER_TURN
        assert registry.retirement_of(ended) == "prefix_pruned"


@STREAM
async def test_a_prefix_mismatch_and_a_model_change_report_one_reason(stream):
    """A reply that reports a prefix mismatch and came from another model:
    the evidence and the parent's retirement name the same reason."""
    wire = ResponsesWire(lambda body, index: (
        _first_then_answer(body, index) if index == 0 else
        Reply([reasoning("rs_2", "ENC-2"), message("msg_2", "It is 42.")], "resp_2",
              model="gpt-6.2-sol-2026-12-01", extra=_MISMATCH)
    ))
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            [parent] = registry.snapshots()
            response, _ = await _call(adapter, client, _history(), stream=stream)
            parent_reason = registry.retirement_of(parent)
    served = provider_served(response)
    assert served.continuation["chain_ended"] == parent_reason == "prefix_mismatch"
    assert served.continuation["expected_model"] == SERVED
    assert served.snapshot == {"stored": False, "reason": "prefix_mismatch"}
