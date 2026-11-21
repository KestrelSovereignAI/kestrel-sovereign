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


# ---------------------------------------------------------------------------
# A reply found by its call ids must still be carried whole (#4204)
# ---------------------------------------------------------------------------

ANCHORED_TEXT = "EXCLUDED-ASSISTANT-CONTENT"


def _commentary_then(*later_calls: str):
    """A reply with reasoning, commentary text and ``call_1``; then one reply
    per ``later_calls`` id carrying only that call; then a text answer."""
    def script(body, index):
        if index == 0:
            return Reply([
                reasoning("rs_0", "ENC-0"), message("msg_0", ANCHORED_TEXT, phase="commentary"),
                call("fc_0", "call_1", "lookup", '{"key":"ARG-ORIGINAL"}'),
            ], "resp_0")
        if index <= len(later_calls):
            call_id = later_calls[index - 1]
            return Reply([reasoning(f"rs_{index}", f"ENC-{index}"),
                          call(f"fc_{index}", call_id, "lookup", "{}")], f"resp_{index}")
        return Reply([message(f"msg_{index}", "Done.")], f"resp_{index}")
    return script


def _tool_history_msg(response: LLMResponse) -> Dict[str, Any]:
    """The assistant message the real tool loop replays ``response`` as."""
    from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin

    return OrchestratorEngineMixin()._build_assistant_tool_history_msg(response)


def _output(call_id: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": "{}"}


@pytest.mark.parametrize("stream", [False, True])
async def test_excluded_anchored_assistant_text_is_not_replayed(stream):
    """The caller kept a reply's tool call but excluded the text the reply
    carried beside it. The call id still finds the message; replaying the
    snapshot would restore the text. So the chain ends at ``prefix_pruned``
    and the request is the ordinary boundary: no opaque replay."""
    wire = ResponsesWire(_commentary_then())
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            first, _ = await _call(adapter, client, [SYSTEM, USER], stream=stream)
            [parent] = registry.snapshots()
            assistant = _tool_history_msg(first)
            assert assistant["content"] == ANCHORED_TEXT
            assistant["content"] = ""
            response, _ = await _call(
                adapter, client, [SYSTEM, USER, assistant, _output("call_1")], stream=stream,
            )
            retired = registry.retirement_of(parent)
            live = [snapshot.response_id for snapshot in registry.snapshots()]
    sent = wire.raw()[1]
    assert ANCHORED_TEXT.encode() not in sent
    assert b"ENC-0" not in sent and b"function_call" not in sent
    assert json.loads(sent)["input"] == [
        {"type": "message", "role": "system", "content": SYSTEM["content"]},
        {"type": "message", "role": "user", "content": USER["content"]},
    ]
    served = provider_served(response)
    assert served.continuation == {
        "status": "boundary", "reason": "prefix_pruned", "omitted_tool_items": 2,
    }
    assert served.snapshot == {"stored": True}
    assert retired == "prefix_pruned"
    assert live == ["resp_1"]


@STREAM
async def test_an_anchored_reply_the_caller_keeps_whole_continues_its_chain(stream):
    """The control: the same reply, replayed by the real history builder
    unchanged, continues its chain, and the replay carries the reply's text
    and call exactly as the provider emitted them."""
    wire = ResponsesWire(_commentary_then())
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            first, _ = await _call(adapter, client, [SYSTEM, USER], stream=stream)
            response, _ = await _call(
                adapter, client, [SYSTEM, USER, _tool_history_msg(first), _output("call_1")],
                stream=stream,
            )
    sent, continued = wire.raw()
    assert wire_items(continued)[: len(wire_items(sent))] == wire_items(sent)
    gained = json.loads(continued)["input"][len(wire_items(sent)):]
    assert [item["type"] for item in gained] == [
        "reasoning", "message", "function_call", "function_call_output",
    ]
    assert gained[1]["content"][0]["text"] == ANCHORED_TEXT
    assert gained[2]["arguments"] == '{"key":"ARG-ORIGINAL"}'
    assert provider_served(response).continuation == {
        "status": "continued", "previous_response_id": "resp_0", "appended_items": 1,
    }


@STREAM
async def test_excluded_text_of_an_earlier_anchored_reply_is_not_replayed(stream):
    """Every reply the chain absorbed is checked, not only its latest: the
    caller kept round one's call but excluded its text, then continued round
    two whole."""
    wire = ResponsesWire(_commentary_then("call_2"))
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            first, _ = await _call(adapter, client, [SYSTEM, USER], stream=stream)
            round_one = [SYSTEM, USER, _tool_history_msg(first), _output("call_1")]
            second, _ = await _call(adapter, client, round_one, stream=stream)
            assert provider_served(second).continuation["status"] == "continued"
            round_one[2] = {**round_one[2], "content": ""}
            response, _ = await _call(
                adapter, client, [*round_one, _tool_history_msg(second), _output("call_2")],
                stream=stream,
            )
    assert ANCHORED_TEXT.encode() in wire.raw()[1]
    sent = wire.raw()[2]
    assert ANCHORED_TEXT.encode() not in sent
    assert b"ENC-" not in sent and b"function_call" not in sent
    boundary = provider_served(response).continuation
    assert (boundary["status"], boundary["reason"]) == ("boundary", "prefix_pruned")


@STREAM
@pytest.mark.parametrize("also", ["inserted_into_prefix", "over_budget"])
async def test_excluded_anchored_text_is_reported_as_pruned_first(stream, also):
    """Pruning wins, as it does for any excluded item: a continuation that
    excluded an anchored reply's text cannot replay at any budget, so its
    reason is ``prefix_pruned`` even when it also inserted into the prefix
    or would exceed the turn's context budget."""
    wire = ResponsesWire(_commentary_then())
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            first, _ = await _call(adapter, client, [SYSTEM, USER], stream=stream)
            assistant = {**_tool_history_msg(first), "content": ""}
            kept = [SYSTEM, USER, assistant, _output("call_1")]
            if also == "inserted_into_prefix":
                kept.insert(2, {"role": "user", "content": "An inserted aside."})
            else:
                registry.limit_context(10)
            response, _ = await _call(adapter, client, kept, stream=stream)
    sent = wire.raw()[1]
    assert ANCHORED_TEXT.encode() not in sent and b"ENC-0" not in sent
    boundary = provider_served(response).continuation
    assert (boundary["status"], boundary["reason"]) == ("boundary", "prefix_pruned")


@STREAM
@pytest.mark.parametrize("arguments, continues", [
    ({"key": "[redacted]"}, False),
    ('["ARG-REPLACED"]', False),
    ('{\n "key": "\\u0041RG-ORIGINAL"\n}', True),
], ids=["rewritten", "not_an_object", "re_encoded"])
async def test_a_kept_call_is_compared_by_its_argument_values(stream, arguments, continues):
    """A call the caller kept with a rewritten argument is not the call the
    reply holds: replaying it would restore the original argument text. One
    whose arguments encode no JSON object is not either, and ends the chain
    rather than failing the request. The same arguments in another encoding
    are the same call."""
    wire = ResponsesWire(_commentary_then())
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            first, _ = await _call(adapter, client, [SYSTEM, USER], stream=stream)
            assistant = _tool_history_msg(first)
            assistant["tool_calls"][0]["function"]["arguments"] = arguments
            response, _ = await _call(
                adapter, client, [SYSTEM, USER, assistant, _output("call_1")], stream=stream,
            )
    sent = wire.raw()[1]
    served = provider_served(response).continuation
    if continues:
        assert served["status"] == "continued"
        assert b"ARG-ORIGINAL" in sent
        return
    assert b"ARG-ORIGINAL" not in sent
    assert b"ENC-0" not in sent and b"function_call" not in sent
    assert (served["status"], served["reason"]) == ("boundary", "prefix_pruned")


_DELETED = object()


def _empty_arguments_call(body, index):
    """A reply with reasoning and one call that takes no arguments; then an answer."""
    if index == 0:
        return Reply([reasoning("rs_0", "ENC-0"), call("fc_0", "call_1", "lookup", "{}")], "resp_0")
    return Reply([message(f"msg_{index}", "Done.")], f"resp_{index}")


@STREAM
@pytest.mark.parametrize("arguments, continues", [
    ({}, True),
    ("{}", True),
    (_DELETED, False),
    (None, False),
    ("", False),
    (False, False),
    ([], False),
    (0, False),
], ids=["dict", "json_text", "missing", "none", "empty_text", "false", "empty_list", "zero"])
async def test_a_kept_call_without_arguments_is_not_read_as_empty(stream, arguments, continues):
    """A reply whose call took ``{}`` is replayed by the real history builder
    with ``{}``; that, or the same object as JSON text, continues the chain.
    A call whose arguments the caller removed or replaced with a falsy
    non-object is not that call: it is not read as ``{}``, so the snapshot is
    retired at ``prefix_pruned`` and the request is the ordinary boundary."""
    wire = ResponsesWire(_empty_arguments_call)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            first, _ = await _call(adapter, client, [SYSTEM, USER], stream=stream)
            [parent] = registry.snapshots()
            assistant = _tool_history_msg(first)
            function = assistant["tool_calls"][0]["function"]
            assert function["arguments"] == {}
            if arguments is _DELETED:
                del function["arguments"]
            else:
                function["arguments"] = arguments
            response, _ = await _call(
                adapter, client, [SYSTEM, USER, assistant, _output("call_1")], stream=stream,
            )
            retired = registry.retirement_of(parent)
    sent = wire.raw()[1]
    served = provider_served(response).continuation
    if continues:
        assert served == {
            "status": "continued", "previous_response_id": "resp_0", "appended_items": 1,
        }
        assert b"ENC-0" in sent
        assert retired == "superseded"
        return
    assert b"ENC-0" not in sent and b"function_call" not in sent
    assert json.loads(sent)["input"] == [
        {"type": "message", "role": "system", "content": SYSTEM["content"]},
        {"type": "message", "role": "user", "content": USER["content"]},
    ]
    assert served == {"status": "boundary", "reason": "prefix_pruned", "omitted_tool_items": 2}
    assert retired == "prefix_pruned"


@STREAM
@pytest.mark.parametrize("carried", ["replaced", "one_of_two_dropped"])
async def test_a_turn_notice_the_caller_rewrote_is_not_replayed(stream, carried):
    """Turn notices may be omitted from a continuation, because the tool loop
    never carries them back, and they are then replayed in place. A notice
    the caller still carries, but changed (retracted and replaced, or one of
    two dropped), is not the one sent: the chain ends and the notice it
    withdrew is not transmitted.

    A notice omitted altogether cannot be told apart from the tool loop's
    rebuild, so it is replayed (``test_a_continuation_appends_to_the_captured_prefix``):
    the exemption rests on role and position only. The core turn's inline
    notice is never stored as a conversation row, so no exclusion can name it."""
    withdrawn = {"role": "system", "content": "NOTICE-WITHDRAWN: auto-mode is on."}
    if carried == "replaced":
        sent_first = [SYSTEM, USER, withdrawn]
        now = [{"role": "system", "content": "NOTICE-CURRENT: auto-mode is off."}]
    else:
        kept_notice = {"role": "developer", "content": "NOTICE-KEPT: answer briefly."}
        sent_first = [SYSTEM, USER, kept_notice, withdrawn]
        now = [kept_notice]
    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope():
            await _call(adapter, client, sent_first, stream=stream)
            assert b"NOTICE-WITHDRAWN" in wire.raw()[0]
            history = _history()
            response, _ = await _call(adapter, client, [*history[:2], *now, *history[2:]], stream=stream)
    sent = wire.raw()[1]
    assert b"NOTICE-WITHDRAWN" not in sent
    assert b"ENC-" not in sent and b"function_call" not in sent
    boundary = provider_served(response).continuation
    assert (boundary["status"], boundary["reason"]) == ("boundary", "prefix_changed")


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
            anchors=(ResponsesAnchor(content="", call_ids=(f"call_{n}",), calls_digest=""),),
            turn_request=None,
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
            anchors=(ResponsesAnchor(content="", call_ids=(f"call_{n}",), calls_digest=""),),
            turn_request=None,
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
        declared_tools=(),
        anchors=(ResponsesAnchor(content="", call_ids=(f"call_{route_label}_{n}",), calls_digest=""),),
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


# ---------------------------------------------------------------------------
# #4205: every exit keeps what the provider reported, and nothing more
# ---------------------------------------------------------------------------

#: How the continuation's call ends, and whether its terminal response had
#: arrived first. A consumer can close only a stream.
_EXITS = {
    "completed": True,
    "transport_before_terminal": False,
    "transport_after_terminal": True,
    "watchdog_before_terminal": False,
    "watchdog_after_terminal": True,
    "caller_cancel_before_terminal": False,
    "caller_cancel_after_terminal": True,
    "other_error_before_terminal": False,
    "other_error_after_terminal": True,
    "consumer_close": False,
}
_EXIT_CASES = [
    pytest.param(stream, exit_, id=f"{'stream' if stream else 'invoke'}-{exit_}")
    for stream in (False, True)
    for exit_ in _EXITS
    if stream or exit_ != "consumer_close"
]

#: The wire's default usage, as the service counts it (cached input apart).
_REPORTED = (100, 30, 20)


def _exit_reply(exit_: str, mismatch: bool) -> Reply:
    ending = {
        "transport_before_terminal": {"disconnect_after_created": True},
        "transport_after_terminal": {"disconnect_after_terminal": True},
        "watchdog_before_terminal": {"stall_after_created": True},
        "watchdog_after_terminal": {"stall_after_terminal": True},
        "caller_cancel_before_terminal": {"stall_after_created": True},
        "caller_cancel_after_terminal": {"stall_after_terminal": True},
        "other_error_after_terminal": {"after_terminal": [{"type": "response.trailing"}]},
        "consumer_close": {"stall_after_terminal": True},
    }.get(exit_, {})
    return Reply(
        [message("msg_2", "It is 42.")], "resp_2",
        extra=dict(_MISMATCH) if mismatch else {}, **ending,
    )


async def _until_silent(wire: ResponsesWire) -> None:
    """Return once the reply was sent in full and the stream went silent."""
    async def silent():
        while not wire.silent:
            await asyncio.sleep(0)
        await wire.silent[-1].stalled.wait()

    await asyncio.wait_for(silent(), timeout=10)


async def _run_to_exit(
    adapter, client, wire, messages, *, stream: bool, exit_: str, model=MODEL, tools=TOOLS,
):
    """Run the call to ``exit_``; return the response, or the error it raised.

    ``exit_`` is ``completed``, ``consumer_close``, or starts with
    ``watchdog`` or ``caller_cancel`` (which wait for the wire's reply to go
    silent); anything else is a call expected to raise an ``Exception``.
    """
    if exit_ == "consumer_close":
        events = adapter.get_streaming_response_with_tools(
            client=client, model=model, messages=messages, tools=tools,
        )
        async for item in events:
            if isinstance(item, str):
                break
        await events.aclose()
        with pytest.raises(StopAsyncIteration):
            await events.__anext__()
        return None
    if exit_ == "completed":
        response, _ = await _call(adapter, client, messages, stream=stream, model=model, tools=tools)
        return response
    if exit_.startswith("watchdog"):
        # The inactivity watchdog's own mechanism: a deadline it reschedules,
        # here expired the moment the stream goes silent.
        with pytest.raises(TimeoutError) as raised:
            async with asyncio.timeout(None) as deadline:
                async def expire_when_silent():
                    await _until_silent(wire)
                    deadline.reschedule(asyncio.get_running_loop().time())

                expiry = asyncio.create_task(expire_when_silent())
                try:
                    await _call(adapter, client, messages, stream=stream, model=model, tools=tools)
                finally:
                    expiry.cancel()
        assert isinstance(raised.value.__cause__, asyncio.CancelledError)
        return raised.value
    if exit_.startswith("caller_cancel"):
        caught: List[BaseException] = []

        async def run():
            try:
                await _call(adapter, client, messages, stream=stream, model=model, tools=tools)
            except asyncio.CancelledError as cancelled:
                caught.append(cancelled)
                raise

        task = asyncio.create_task(run())
        await _until_silent(wire)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Re-raised, never swallowed.
        assert task.cancelled() and len(caught) == 1
        return caught[0]
    with pytest.raises(Exception) as raised:
        await _call(adapter, client, messages, stream=stream, model=model, tools=tools)
    return raised.value


@pytest.mark.parametrize("mismatch", [False, True], ids=["no_mismatch", "prefix_mismatch"])
@pytest.mark.parametrize("stream, exit_", _EXIT_CASES)
async def test_every_exit_keeps_what_the_provider_reported(stream, exit_, mismatch, monkeypatch):
    """A continuation's call ends each way it can, on both entry points.

    Whatever it ends with keeps the provider's identification, which
    ``response.created`` already gave, and the usage once the terminal
    response reported it. Before that the usage is unknown, never zero. A
    ``prefix_mismatch`` the terminal response reported ends the chain
    whatever follows; one never reported ends nothing. Only a reply that
    was returned keeps a snapshot, the stream is released on every exit, and
    a cancellation is re-raised as itself.
    """
    import openai

    from kestrel_sovereign.llm.openai_responses import (
        ResponsesEventState,
        ResponsesStreamInterrupted,
    )

    failing_event = {
        "other_error_before_terminal": "response.output_text.delta",
        "other_error_after_terminal": "response.trailing",
    }.get(exit_)
    if failing_event is not None:
        original = ResponsesEventState.observe

        def observe(self, event):
            if getattr(event, "type", None) == failing_event:
                raise KeyError("a bug handling an event")
            return original(self, event)

        monkeypatch.setattr(ResponsesEventState, "observe", observe)

    wire = ResponsesWire(lambda body, index: (
        _first_then_answer(body, index) if index == 0 else _exit_reply(exit_, mismatch)
    ))
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            await _call(adapter, client, [SYSTEM, USER], stream=stream)
            [parent] = registry.snapshots()
            outcome = await _run_to_exit(adapter, client, wire, _history(), stream=stream, exit_=exit_)
            live = [snapshot.response_id for snapshot in registry.snapshots()]
            parent_reason = registry.retirement_of(parent)

    terminal = _EXITS[exit_]
    if wire.silent:
        assert wire.silent[-1].closed, "the stream was not released"
    expected_error = {
        "transport_before_terminal": openai.APIConnectionError,
        "transport_after_terminal": ResponsesStreamInterrupted,
        "other_error_before_terminal": KeyError,
        "other_error_after_terminal": KeyError,
    }.get(exit_)
    if expected_error is not None:
        assert isinstance(outcome, expected_error), repr(outcome)

    if outcome is not None:
        served = provider_served(outcome)
        assert (served.model, served.response_id) == (SERVED, "resp_2")
        usage = outcome if isinstance(outcome, LLMResponse) else reported_usage(outcome)
        if terminal:
            assert usage is not None, "the usage the terminal response reported was lost"
            counts = (usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens)
            assert counts == _REPORTED
        else:
            assert usage is None, "usage the provider never reported was recorded"
        assert served.prefix_mismatch == (("rs_1",) if terminal and mismatch else None)

    if terminal and mismatch:
        assert live == [] and parent_reason == "prefix_mismatch"
    elif exit_ == "completed":
        assert live == ["resp_2"] and parent_reason == "superseded"
    else:
        # The reply never reached the caller: the chain it would have
        # extended is kept, untouched, for a retry.
        assert live == ["resp_1"] and parent_reason is None


async def test_a_watchdog_timeout_reports_the_usage_of_the_cancellation_it_wraps():
    """The service reads a failed attempt's usage through the watchdog's
    usage-less timeout to the cancellation the adapter recorded it on."""
    from kestrel_sovereign.llm.call_progress import AttemptInactivityTimeout
    from kestrel_sovereign.llm.output_ceiling import attach_reported_usage

    counts = LLMResponse(input_tokens=100, output_tokens=30, cache_read_input_tokens=20)
    cancelled = attach_reported_usage(asyncio.CancelledError(), counts)
    expired = TimeoutError()
    expired.__cause__ = cancelled
    timeout = AttemptInactivityTimeout(provider="openai:api", model=MODEL, phase="inter-chunk", timeout=1.0)
    timeout.__cause__ = expired
    assert reported_usage(timeout) is counts
    # Nothing reported is nothing recorded: never a zero.
    assert reported_usage(attach_reported_usage(asyncio.CancelledError(), None)) is None


# ---------------------------------------------------------------------------
# #4205: the ordinary chat-completions route names what served it
# ---------------------------------------------------------------------------

#: A chat-completions request names the alias; its reply the dated model.
_CHAT_ID = "chatcmpl-synthetic-4198"


def _chat_wire(**reply) -> ResponsesWire:
    from tests.shared.openai_responses_wire import ChatReply

    return ResponsesWire(
        lambda body, index: Reply([], "unused"),
        chat_script=lambda body, index: ChatReply("example", **{"response_id": _CHAT_ID, **reply}),
    )


async def test_default_turn_reports_chat_model_and_id_from_upstream():
    """The ordinary default route's reply says which model served it and the
    id it gave the response; Kestrel's evidence carries both, never the
    requested alias in place of the served model."""
    from tests.shared.openai_responses_wire import CHAT_MODEL, CHAT_SERVED

    wire = _chat_wire()
    async with wire.client() as client:
        result = await _adapter().get_response(
            client=client, model=CHAT_MODEL, messages=[USER], tools=None,
        )
    assert [request.url.path for request in wire.requests] == ["/v1/chat/completions"]
    assert result.content == "example"
    served = provider_served(result)
    assert served is not None, "the chat-completions reply has no provider identity"
    assert (served.api, served.model, served.response_id) == ("chat_completions", CHAT_SERVED, _CHAT_ID)
    assert served.model != CHAT_MODEL
    assert served.as_metadata() == {
        "api": "chat_completions", "model": CHAT_SERVED, "response_id": _CHAT_ID,
        "continuation": {"status": "fresh"},
    }


async def test_default_streamed_turn_reports_chat_model_and_id_from_upstream():
    from tests.shared.openai_responses_wire import CHAT_MODEL, CHAT_SERVED

    wire = _chat_wire()
    async with wire.client() as client:
        result, items = await _call(_adapter(), client, [USER], stream=True, tools=TOOLS, model=CHAT_MODEL)
    assert [request.url.path for request in wire.requests] == ["/v1/chat/completions"]
    assert items[:-1] == ["example"]
    served = provider_served(result)
    assert (served.api, served.model, served.response_id) == ("chat_completions", CHAT_SERVED, _CHAT_ID)


@STREAM
async def test_a_chat_reply_that_names_no_model_or_id_leaves_them_unknown(stream):
    """What the reply does not name stays unknown: never the requested model."""
    from tests.shared.openai_responses_wire import CHAT_MODEL

    wire = _chat_wire(model=None, response_id=None)
    async with wire.client() as client:
        result, _ = await _call(_adapter(), client, [USER], stream=stream, tools=None, model=CHAT_MODEL)
    served = provider_served(result)
    assert (served.api, served.model, served.response_id) == ("chat_completions", None, None)
    assert served.as_metadata() == {"api": "chat_completions", "continuation": {"status": "fresh"}}


@STREAM
async def test_a_compatible_chat_route_claims_no_provider_identity(stream):
    """Off the native OpenAI route a server's reply fields are not the
    provider's own identification, so none is recorded."""
    from tests.shared.openai_responses_wire import CHAT_MODEL

    wire = _chat_wire()
    async with wire.client() as client:
        result, _ = await _call(
            OpenAIAdapter("openai", native_openai=False), client, [USER],
            stream=stream, tools=None, model=CHAT_MODEL,
        )
    assert result.content == "example"
    assert provider_served(result) is None


# ---------------------------------------------------------------------------
# #4205: every chat-completions exit keeps what the provider reported
# ---------------------------------------------------------------------------

#: Each way a native chat-completions call ends, and what the provider had
#: reported by then: whether it had named its reply (the model and the id),
#: and whether it had reported the usage. A stream names its reply on its
#: first chunk and reports its usage on its last; an ordinary completion is
#: one body, so it reports both at once or neither. A consumer can close only
#: a stream.
_CHAT_EXITS = {
    True: {
        "completed": (True, True),
        "transport_before_usage": (True, False),
        "transport_after_usage": (True, True),
        "watchdog_before_usage": (True, False),
        "watchdog_after_usage": (True, True),
        "caller_cancel_before_usage": (True, False),
        "caller_cancel_after_usage": (True, True),
        "protocol_error": (True, True),
        "other_error_before_usage": (True, False),
        "other_error_after_usage": (True, True),
        "consumer_close": (True, False),
    },
    False: {
        "completed": (True, True),
        "transport_before_usage": (False, False),
        "watchdog_before_usage": (False, False),
        "caller_cancel_before_usage": (False, False),
        "protocol_error": (True, True),
        "other_error_after_usage": (True, True),
    },
}
_CHAT_EXIT_CASES = [
    pytest.param(stream, exit_, id=f"{'stream' if stream else 'invoke'}-{exit_}")
    for stream, exits in _CHAT_EXITS.items()
    for exit_ in exits
]

#: A chat usage report, and the counts the service records from it: the
#: cached prompt tokens are counted apart from the other input.
_CHAT_USAGE = {
    "prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
    "prompt_tokens_details": {"cached_tokens": 20},
}


def _chat_exit_reply(exit_: str) -> Dict[str, Any]:
    return {
        "transport_before_usage": {"disconnect_before_usage": True},
        "transport_after_usage": {"disconnect_after_usage": True},
        "watchdog_before_usage": {"stall_before_usage": True},
        "watchdog_after_usage": {"stall_after_usage": True},
        "caller_cancel_before_usage": {"stall_before_usage": True},
        "caller_cancel_after_usage": {"stall_after_usage": True},
        # Tool arguments that are not JSON: the reply breaks the protocol.
        "protocol_error": {"tool_arguments": '{"key": '},
        "consumer_close": {"stall_after_usage": True},
    }.get(exit_, {})


@pytest.mark.parametrize("stream, exit_", _CHAT_EXIT_CASES)
async def test_every_chat_exit_keeps_what_the_provider_reported(stream, exit_, monkeypatch):
    """A native chat-completions call ends each way it can, on both entry
    points, over the real SDK and a synthetic transport (#4205).

    Whatever it ends with names the model and the id the reply gave, once
    the reply gave them, and keeps the usage once the provider reported it:
    a transport failure, a watchdog or caller cancellation, a broken
    protocol and any other error included. Before the provider reported
    them they are unknown, never the requested model and never zero. The
    stream is released on every exit, and a cancellation is re-raised as
    itself.
    """
    import openai

    from kestrel_sovereign.llm import openai_adapter
    from tests.shared.openai_responses_wire import CHAT_MODEL, CHAT_SERVED

    failing = {
        "other_error_before_usage": "feed",
        "other_error_after_usage": "flush" if stream else None,
    }.get(exit_)
    if failing is not None:
        class FailingSplitter(openai_adapter._ThinkingContentSplitter):
            pass

        def fail(*_args, **_kwargs):
            raise KeyError("a bug handling the reply")

        setattr(FailingSplitter, failing, fail)
        monkeypatch.setattr(openai_adapter, "_ThinkingContentSplitter", FailingSplitter)
    elif exit_ == "other_error_after_usage":
        def fail(*_args, **_kwargs):
            raise KeyError("a bug handling the reply")

        monkeypatch.setattr(openai_adapter, "_split_thinking_from_content", fail)

    if exit_.startswith("transport") and not stream:
        # An ordinary completion that lost its body is retried; skip the
        # backoff, keeping each wait a real yield to the loop.
        real_sleep = asyncio.sleep
        monkeypatch.setattr(asyncio, "sleep", lambda _delay, *args, **kwargs: real_sleep(0))

    wire = _chat_wire(usage=_CHAT_USAGE, **_chat_exit_reply(exit_))
    async with wire.client() as client:
        outcome = await _run_to_exit(
            _adapter(), client, wire, [USER], stream=stream, exit_=exit_,
            model=CHAT_MODEL, tools=TOOLS,
        )

    assert {request.url.path for request in wire.requests} == {"/v1/chat/completions"}
    if wire.silent:
        assert wire.silent[-1].closed, "the stream was not released"
    expected_error = {
        "transport_before_usage": openai.APIConnectionError,
        "transport_after_usage": openai.APIConnectionError,
        "protocol_error": GenerationProtocolError,
        "other_error_before_usage": KeyError,
        "other_error_after_usage": KeyError,
    }.get(exit_)
    if expected_error is not None:
        assert isinstance(outcome, expected_error), repr(outcome)
    if outcome is None:
        return

    identified, usage_reported = _CHAT_EXITS[stream][exit_]
    served = provider_served(outcome)
    if identified:
        assert served is not None, "the provider's identification of its reply was lost"
        assert (served.api, served.model, served.response_id) == (
            "chat_completions", CHAT_SERVED, _CHAT_ID,
        )
        assert served.continuation == {"status": "fresh"}
    else:
        assert served is None, "an identity the provider never gave was recorded"
    usage = outcome if isinstance(outcome, LLMResponse) else reported_usage(outcome)
    if usage_reported:
        assert usage is not None, "the usage the provider reported was lost"
        counts = (usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens)
        assert counts == (100, 30, 20)
        assert provider_served(usage) == served
    else:
        assert usage is None, "usage the provider never reported was recorded"


@STREAM
async def test_a_chat_failure_keeps_an_observed_zero(stream):
    """A provider that reported zero tokens reported something: the failed
    attempt keeps the zero, distinct from the unknown of no report (#4205)."""
    from tests.shared.openai_responses_wire import CHAT_MODEL

    zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    wire = _chat_wire(
        usage=zero,
        **({"disconnect_after_usage": True} if stream else {"tool_arguments": "[]"}),
    )
    async with wire.client() as client:
        with pytest.raises(Exception) as raised:
            await _call(_adapter(), client, [USER], stream=stream, tools=TOOLS, model=CHAT_MODEL)
    usage = reported_usage(raised.value)
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (0, 0, 0)
    assert usage.cache_read_input_tokens is None


# ---------------------------------------------------------------------------
# #4207: one receipt per provider attempt, kept on whatever error ends it
# ---------------------------------------------------------------------------

#: The three entry points whose provider facts arrive as stream events or
#: chunks: Responses invoke reduces an internal stream (#4207).
_STREAMED_ENTRIES = ["responses-invoke", "responses-stream", "chat-stream"]
#: The chat usage the wire reports in this section, and the counts the
#: service records from it: the cached prompt tokens are counted apart.
_REPORTED_CHAT_USAGE = {
    "prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
    "prompt_tokens_details": {"cached_tokens": 20},
}


def _entry(entry: str):
    """``(stream, chat, model, served model, response id, tools)`` of an entry point."""
    from tests.shared.openai_responses_wire import CHAT_MODEL, CHAT_SERVED

    chat = entry.startswith("chat")
    return (
        entry.endswith("stream"), chat,
        CHAT_MODEL if chat else MODEL, CHAT_SERVED if chat else SERVED,
        "chatcmpl-test-end" if chat else "resp_test_end",
        None if chat else TOOLS,
    )


def _evidence(error: BaseException):
    served = provider_served(error)
    usage = reported_usage(error)
    counts = None if usage is None else (
        usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens,
    )
    return (served.model if served else None, served.response_id if served else None, counts)


@pytest.mark.parametrize("entry", _STREAMED_ENTRIES)
@pytest.mark.parametrize("close_fails", [False, True], ids=["closing-succeeds", "closing-fails"])
async def test_provider_evidence_survives_transport_and_native_cleanup(entry, close_fails, monkeypatch):
    """The transport breaks after the provider's terminal event or usage
    chunk, and then, with ``closing-fails``, closing the stream fails too.

    The provider's model, id and usage were already reported, so whatever
    error escapes carries them: the ordinary transport failure, or the
    execution-control exception group closing raised. That group is raised
    as itself, the very object the close raised, with its members, its
    native cleanup debt and the stream it still owes (#4207); it is never
    wrapped or replaced. No snapshot is kept for a reply never returned.
    """
    from kestrel_sovereign.execution_custody import (
        execution_cleanup_debts,
        is_execution_control_error,
    )
    from kestrel_sovereign.llm import openai_adapter
    from tests.shared import openai_responses_wire as wiretypes

    closes: List[Any] = []
    if close_fails:
        async def fail_close(body):
            raise OSError("synthetic close failure after provider terminal event")

        monkeypatch.setattr(wiretypes._DisconnectingBody, "aclose", fail_close)
        real_close = openai_adapter.close_native_execution_stream

        async def observed_close(stream, owner=None, **kwargs):
            try:
                await real_close(stream, owner, **kwargs)
            except BaseException as raised:
                closes.append((stream, raised, tuple(getattr(raised, "exceptions", ()))))
                raise

        monkeypatch.setattr(openai_adapter, "close_native_execution_stream", observed_close)
    stream, chat, model, serving, response_id, tools = _entry(entry)
    wire = ResponsesWire(
        lambda body, index: Reply([message("msg_end", "Example")], response_id,
                                  disconnect_after_terminal=True),
        chat_script=lambda body, index: wiretypes.ChatReply(
            "Example", response_id, usage=_REPORTED_CHAT_USAGE, disconnect_after_usage=True,
        ),
    )
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            with pytest.raises(BaseException) as caught:
                await _call(_adapter(), client, [USER], stream=stream, tools=tools, model=model)
            assert not registry.snapshots(), "a reply never returned kept a snapshot"

    error = caught.value
    assert is_execution_control_error(error) == close_fails
    assert _evidence(error) == (serving, response_id, (100, 30, 20))
    if close_fails:
        [(closed, raised, members)] = closes
        assert error is raised, "the close's exception group was wrapped or replaced"
        assert isinstance(error, BaseExceptionGroup)
        assert len(error.exceptions) == len(members)
        assert all(now is then for now, then in zip(error.exceptions, members))
        debts = execution_cleanup_debts(error)
        assert any(resource is closed for resource, _owner in debts), (
            "the stream still owed its native retirement is no longer on the error"
        )
        assert reported_usage(error) is not None and provider_served(error) is not None
    else:
        assert not closes


async def _receipts_opened(monkeypatch) -> List[Any]:
    """Every :class:`ProviderAttemptReceipt` opened from now on, in order."""
    from kestrel_sovereign.llm.provider_served import ProviderAttemptReceipt

    opened: List[Any] = []
    original = ProviderAttemptReceipt.__init__

    def record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        opened.append(self)

    monkeypatch.setattr(ProviderAttemptReceipt, "__init__", record)
    return opened


def _receipt_evidence(receipt):
    served, usage = receipt.served, receipt.usage
    counts = None if usage is None else (
        usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens,
    )
    return (served.model if served else None, served.response_id if served else None, counts)


@pytest.mark.parametrize("entry", _STREAMED_ENTRIES)
async def test_the_receipt_holds_each_fact_before_the_attempt_ends(entry, monkeypatch):
    """The provider's stream reports its identity, its usage and, on the
    Responses API, a ``prefix_mismatch``, then goes silent before it ends.

    While the iterator is still open, before any exit runs, the attempt's
    receipt already holds the model, the id and the usage, and the reported
    mismatch has already retired the parent snapshot: each fact is recorded
    as it arrives, not when the attempt ends (#4207). The caller then
    cancels, and the cancellation carries the same receipt.
    """
    from tests.shared.openai_responses_wire import ChatReply

    stream, chat, model, serving, response_id, tools = _entry(entry)
    script = (
        (lambda body, index: Reply([message("msg_end", "Example")], response_id, stall_after_terminal=True))
        if chat else
        (lambda body, index: _first_then_answer(body, index) if index == 0 else Reply(
            [message("msg_2", "It is 42.")], response_id, extra=dict(_MISMATCH), stall_after_terminal=True,
        ))
    )
    wire = ResponsesWire(script, chat_script=lambda body, index: ChatReply(
        "Example", response_id, usage=_REPORTED_CHAT_USAGE, stall_after_usage=True,
    ))
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            parent = None
            if not chat:
                await _call(adapter, client, [SYSTEM, USER], stream=stream)
                [parent] = registry.snapshots()
            opened = await _receipts_opened(monkeypatch)
            messages = [USER] if chat else _history()
            attempt = asyncio.create_task(
                _call(adapter, client, messages, stream=stream, tools=tools, model=model)
            )
            await _until_silent(wire)

            # Paused: the iterator has not exited.
            assert not attempt.done()
            [receipt] = opened
            assert _receipt_evidence(receipt) == (serving, response_id, (100, 30, 20))
            if not chat:
                assert receipt.served.prefix_mismatch == ("rs_1",)
                assert registry.retirement_of(parent) == "prefix_mismatch"
                assert not registry.snapshots()

            attempt.cancel()
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await attempt
    assert wire.silent[-1].closed
    assert _evidence(cancelled.value) == (serving, response_id, (100, 30, 20))


async def test_a_chat_reply_is_recorded_before_its_tool_arguments_are_parsed(monkeypatch):
    """An ordinary chat completion arrives whole: its model, id and usage are
    on the attempt's receipt before the adapter parses any of it (#4207)."""
    from kestrel_sovereign.llm import openai_adapter
    from tests.shared.openai_responses_wire import CHAT_MODEL, CHAT_SERVED, ChatReply

    seen: List[Any] = []
    real_decode = openai_adapter.decode_tool_arguments

    def decode(arguments):
        seen.append(_receipt_evidence(opened[-1]))
        return real_decode(arguments)

    monkeypatch.setattr(openai_adapter, "decode_tool_arguments", decode)
    wire = ResponsesWire(lambda body, index: Reply([], "unused"), chat_script=lambda body, index: ChatReply(
        "Example", "chatcmpl-parsed", usage=_REPORTED_CHAT_USAGE, tool_arguments='{"key": "x"}',
    ))
    async with wire.client() as client:
        opened = await _receipts_opened(monkeypatch)
        result = await _adapter().get_response(client=client, model=CHAT_MODEL, messages=[USER], tools=TOOLS)
    assert result.tool_calls and len(opened) == 1
    assert seen == [(CHAT_SERVED, "chatcmpl-parsed", (100, 30, 20))]


#: Two calls that overlap: each its own identity and usage.
_OVERLAP = {
    "A": {"response": "resp_overlap_a", "chat": "chatcmpl-overlap-a",
          "usage": (100, 30, 20)},
    "B": {"response": "resp_overlap_b", "chat": "chatcmpl-overlap-b",
          "usage": (7, 3, 1)},
}


def _overlap_key(body: Dict[str, Any]) -> str:
    return "A" if "call-A" in json.dumps(body) else "B"


def _overlap_usage(key: str, chat: bool) -> Dict[str, Any]:
    inputs, outputs, cached = _OVERLAP[key]["usage"]
    if chat:
        return {"prompt_tokens": inputs + cached, "completion_tokens": outputs,
                "total_tokens": inputs + cached + outputs,
                "prompt_tokens_details": {"cached_tokens": cached}}
    return {"input_tokens": inputs + cached, "output_tokens": outputs,
            "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": inputs + cached + outputs}


@pytest.mark.parametrize("entry", [*_STREAMED_ENTRIES, "chat-invoke"])
@pytest.mark.parametrize("close_fails", [False, True], ids=["closing-succeeds", "closing-fails"])
async def test_overlapping_attempts_never_share_a_receipt(entry, close_fails, monkeypatch):
    """Two public adapter calls run at once on one client, each served its
    own identity and usage, and both fail while both are open: after the
    terminal event or usage chunk (closing the stream failing too, with
    ``closing-fails``), or, as an ordinary chat completion, with tool
    arguments that are not JSON. Each error carries its own attempt's
    receipt and never the other's (#4207).

    An ordinary chat completion has no stream to close: ``closing-fails``
    makes its body's close fail as the SDK reads it, before it decodes, and
    then neither call supplied anything.
    """
    from tests.shared import openai_responses_wire as wiretypes

    if close_fails:
        async def fail_close(body):
            raise OSError("synthetic close failure")

        target = wiretypes._HeldBody if entry == "chat-invoke" else wiretypes._DisconnectingBody
        monkeypatch.setattr(target, "aclose", fail_close, raising=False)
    stream, chat, model, serving, _rid, tools = _entry(entry)
    release = asyncio.Event()
    wire = ResponsesWire(
        lambda body, index: Reply(
            [message("msg", "Example")], _OVERLAP[_overlap_key(body)]["response"],
            usage=_overlap_usage(_overlap_key(body), chat=False), disconnect_after_terminal=True,
        ),
        chat_script=lambda body, index: wiretypes.ChatReply(
            "Example", _OVERLAP[_overlap_key(body)]["chat"],
            usage=_overlap_usage(_overlap_key(body), chat=True),
            **({"tool_arguments": '{"key": '} if entry == "chat-invoke" else {"disconnect_after_usage": True}),
        ),
        release=release,
    )
    adapter = _adapter()

    async def attempt(key: str) -> BaseException:
        messages = [{"role": "user", "content": f"Look it up for call-{key}."}]
        try:
            await _call(adapter, client, messages, stream=stream, tools=TOOLS if entry == "chat-invoke" else tools,
                        model=model)
        except BaseException as error:  # noqa: BLE001 - the error is the subject
            return error
        raise AssertionError(f"call {key} did not fail")

    async with wire.client() as client:
        with responses_continuation_scope():
            calls = {key: asyncio.create_task(attempt(key)) for key in _OVERLAP}

            async def both_open():
                while len(wire.held) < 2 or not all(body.waiting.is_set() for body in wire.held):
                    await asyncio.sleep(0)

            await asyncio.wait_for(both_open(), timeout=10)
            assert not any(task.done() for task in calls.values())
            release.set()
            errors = {key: await task for key, task in calls.items()}

    undecoded = entry == "chat-invoke" and close_fails
    for key, error in errors.items():
        identity = _OVERLAP[key]["chat" if chat else "response"]
        expected = (None, None, None) if undecoded else (serving, identity, _OVERLAP[key]["usage"])
        assert _evidence(error) == expected, (key, error)
    assert errors["A"] is not errors["B"]


@pytest.mark.parametrize("entry", _STREAMED_ENTRIES)
@pytest.mark.parametrize("control", [False, True], ids=["ordinary", "control-member"])
async def test_an_explicit_nested_group_keeps_its_members_and_classification(entry, control, monkeypatch):
    """Handling the reply raises an explicit, nested exception group after
    the provider reported everything. The same group object escapes with its
    members, carrying the attempt's receipt; a group with an execution-control
    member stays control, and one without stays an ordinary failure (#4207)."""
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError,
        is_execution_control_error,
    )
    from kestrel_sovereign.llm import openai_adapter
    from kestrel_sovereign.llm.openai_responses import ResponsesEventState
    from tests.shared.openai_responses_wire import ChatReply

    inner = ExceptionGroup("inner", [ValueError("b")])
    member = ExecutionAuthorityError("synthetic denial") if control else KeyError("a")
    raised = BaseExceptionGroup("outer", [member, inner])
    stream, chat, model, serving, response_id, tools = _entry(entry)
    if chat:
        class FailingSplitter(openai_adapter._ThinkingContentSplitter):
            def flush(self):
                raise raised

        monkeypatch.setattr(openai_adapter, "_ThinkingContentSplitter", FailingSplitter)
    else:
        original = ResponsesEventState.observe

        def observe(self, event):
            if getattr(event, "type", None) == "response.trailing":
                raise raised
            return original(self, event)

        monkeypatch.setattr(ResponsesEventState, "observe", observe)
    wire = ResponsesWire(
        lambda body, index: Reply([message("msg_end", "Example")], response_id,
                                  after_terminal=[{"type": "response.trailing"}]),
        chat_script=lambda body, index: ChatReply("Example", response_id, usage=_REPORTED_CHAT_USAGE),
    )
    async with wire.client() as client:
        with pytest.raises(BaseException) as caught:
            await _call(_adapter(), client, [USER], stream=stream, tools=tools, model=model)
    error = caught.value
    assert error is raised
    assert error.exceptions[0] is member and error.exceptions[1] is inner
    assert inner.exceptions[0].__class__ is ValueError
    assert is_execution_control_error(error) is control
    assert _evidence(error) == (serving, response_id, (100, 30, 20))


@pytest.mark.parametrize("ending", ["truncated", "close-fails"])
async def test_a_chat_body_that_never_decoded_supplied_nothing(ending, monkeypatch):
    """An ordinary chat completion's body arrives but never decodes: cut off
    mid-JSON, or its close failing as the SDK reads it. The bytes held the
    reply's model, id and usage, but the adapter never had a reply, so the
    error carries none of them: unknown, never inferred (#4207)."""
    import httpx
    import openai

    from kestrel_sovereign.execution_custody import is_execution_control_error
    from tests.shared.openai_responses_wire import CHAT_MODEL, ChatReply

    reply = json.dumps(ChatReply("Example", "chatcmpl-undecoded", usage=_REPORTED_CHAT_USAGE).completion())

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (reply[: len(reply) // 2] if ending == "truncated" else reply).encode()

        async def aclose(self):
            if ending == "close-fails":
                raise OSError("synthetic close failure")

    def handle(request):
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=Body())

    client = openai.AsyncOpenAI(
        api_key="test-key", base_url="https://api.openai.com/v1", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    if ending == "truncated":
        real_sleep = asyncio.sleep
        monkeypatch.setattr(asyncio, "sleep", lambda _delay, *args, **kwargs: real_sleep(0))
    async with client:
        opened = await _receipts_opened(monkeypatch)
        with pytest.raises(BaseException) as caught:
            await _adapter().get_response(client=client, model=CHAT_MODEL, messages=[USER], tools=TOOLS)
    assert is_execution_control_error(caught.value) is (ending == "close-fails")
    assert _evidence(caught.value) == (None, None, None)
    assert [_receipt_evidence(receipt) for receipt in opened] == [(None, None, None)]


@STREAM
async def test_a_chat_failure_with_no_usage_report_keeps_it_unknown(stream):
    """The reply named itself but reported no usage before the attempt
    failed: the identity is kept and the usage stays unknown, never zero."""
    from tests.shared.openai_responses_wire import CHAT_MODEL, CHAT_SERVED

    wire = _chat_wire(usage=None, tool_arguments='{"key": ')
    async with wire.client() as client:
        with pytest.raises(GenerationProtocolError) as raised:
            await _call(_adapter(), client, [USER], stream=stream, tools=TOOLS, model=CHAT_MODEL)
    assert _evidence(raised.value) == (CHAT_SERVED, _CHAT_ID, None)
