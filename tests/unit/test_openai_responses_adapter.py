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
    ResponsesRoute,
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
    (MODEL, True, {"extra_body": {"tools": TOOLS}}, True),
    (MODEL, True, {}, False),
    (MODEL, False, {"tools": TOOLS}, False),
    ("gpt-6.2-solar", True, {"tools": TOOLS}, False),
    ("gpt-5.5", True, {"tools": TOOLS}, False),
])
async def test_the_endpoint_is_chosen_from_the_request_as_sent(model, native, request_body, expected):
    assert requires_responses_api(
        native_openai=native, model=model, request=request_body,
        messages=[SYSTEM, USER], route=ResponsesRoute(label="openai|", client=object()),
    ) is expected


@STREAM
async def test_tools_supplied_through_extra_body_reach_the_responses_api(stream):
    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as client:
        response, _ = await _call(
            _adapter(), client, [SYSTEM, USER], stream=stream, tools=None,
            extra_body={"tools": TOOLS},
        )
    [body] = wire.bodies()
    assert [tool["name"] for tool in body["tools"]] == ["lookup"]
    assert body["tools"][0]["strict"] is False
    assert body["store"] is False and body["include"] == ["reasoning.encrypted_content"]
    assert "extra_body" not in body
    assert response.tool_calls[0].arguments == {"key": "é", "n": 100.0}
    assert not wire.unexpected


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


@STREAM
async def test_two_configured_routes_on_the_official_url_never_share_a_chain(stream):
    """``openai:primary`` and ``openai:other`` share the adapter name and the
    official base URL; only their clients (and so their accounts) differ."""
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    providers = ProviderRegistry({})
    primary, other = (
        providers._build_route(
            "openai", name, {"is_cloud": True},
            {"adapter": "OpenAIAdapter", "api_key": f"key-{name}", "model": MODEL},
        )
        for name in ("primary", "other")
    )
    try:
        assert primary.adapter.name == other.adapter.name == "openai"
        assert str(primary.client.base_url) == str(other.client.base_url)
        assert primary.client is not other.client
    finally:
        await primary.client.close()
        await other.client.close()

    wire = ResponsesWire(_first_then_answer)
    async with wire.client() as one, wire.client() as two:
        with responses_continuation_scope():
            await _call(primary.adapter, one, [SYSTEM, USER], stream=stream)
            crossed, _ = await _call(other.adapter, two, _history(), stream=stream)
            own, _ = await _call(primary.adapter, one, _history(), stream=stream)
    first, other_request, own_request = wire.raw()
    assert b"ENC-" not in other_request and b"function_call" not in other_request
    assert provider_served(crossed).continuation["reason"] == "no_snapshot"
    # The chain is intact on its own route.
    assert wire_items(own_request)[: len(wire_items(first))] == wire_items(first)
    assert provider_served(own).continuation["status"] == "continued"


# ---------------------------------------------------------------------------
# Endpoint selection follows the continuation, not only the tools offered
# ---------------------------------------------------------------------------


@STREAM
async def test_a_tool_free_continuation_keeps_its_chain_on_the_responses_api(stream):
    """``continue_with_tool_results`` defaults to no tools, and a streamed
    continuation may offer none: the chain continues, its declarations are
    kept, and the model is told to call none of them."""
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
    assert {request.url.path for request in wire.requests} == {"/v1/responses"}
    sent, continued = wire.raw()
    assert wire_items(continued)[: len(wire_items(sent))] == wire_items(sent)
    assert wire_items(continued, "tools") == wire_items(sent, "tools")
    assert json.loads(continued)["tool_choice"] == "none"
    assert provider_served(response).continuation["status"] == "continued"
    assert response.content == "It is 42."


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


def _pruned_history() -> List[Dict[str, Any]]:
    """The continuation after the loop shed the replayed earlier exchange:
    the list itself fits the budget, the captured prefix does not."""
    return _history()


@STREAM
@pytest.mark.parametrize("fits", [False, True], ids=["over_budget", "within_budget"])
async def test_a_continuation_over_the_turn_context_budget_ends_the_chain(stream, fits):
    earlier = [{"role": "user", "content": "Tell me something."}, {"role": "assistant", "content": LONG}]
    wire = ResponsesWire(_first_then_answer)
    adapter = _adapter()
    async with wire.client() as client:
        with responses_continuation_scope() as registry:
            await _call(adapter, client, [SYSTEM, *earlier, USER], stream=stream)
            first = wire.raw()[0]
            # The pruned list fits; the captured prefix still holds LONG.
            budget = len(LONG) * 2 if fits else len(LONG) // 2
            registry.limit_context(budget)
            response, _ = await _call(adapter, client, _pruned_history(), stream=stream)
            # The ended chain is retired; the boundary reply starts a new one.
            kept = [snapshot.response_id for snapshot in registry.snapshots()]
    continued = wire.raw()[1]
    assert LONG.encode() in first
    served = provider_served(response).continuation
    if fits:
        assert served["status"] == "continued"
        assert wire_items(continued)[: len(wire_items(first))] == wire_items(first)
        return
    assert LONG.encode() not in continued
    assert b"ENC-" not in continued and b"function_call" not in continued
    assert served["reason"] == "context_budget"
    assert served["budget_chars"] == budget < served["request_chars"]
    assert kept == ["resp_2"]


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
    from kestrel_sovereign.llm.openai_responses import _finish_request

    with pytest.raises(ValueError, match="overwrite"):
        _finish_request(
            settings={"tools": [], "include": [], "store": False, "extra_body": {"store": True}},
            params={}, items=(), route=ResponsesRoute(label="r", client=None), model=MODEL,
            declared=(), anchors=(), turn_request=None, continuation={}, parent=None, registry=None,
        )
