"""#3619: a Gemini 3 call's ``thought_signature`` goes back with its replay.

Gemini 3 signs the ``function_call`` parts it returns with an opaque
``thought_signature``, and rejects with a 400 a request that replays a call of
the current turn without it. Both adapters built ``ToolCall(id, name,
arguments)`` and dropped the signature, the agent replayed the call as an
OpenAI-format ``tool_calls`` entry with no field for it, and
``to_gemini_contents`` (#3608) had nothing to restore. So every tool-loop
follow-up on a Gemini 3 model would be refused.

Verified against Google's documented contract only: no live Gemini 3 request.

These tests run the real google-genai client over an ``httpx.MockTransport``
(the #3608 harness), so the SDK's own parsing and serialization run. For both
adapters they pin:

* a signed call's follow-up request carries the same signature bytes on that
  call's part, on the plain and the streamed request;
* of parallel calls Gemini signs only the first, and only the first gets one
  back;
* an unsigned call's follow-up is the request #3608 sent;
* the signature survives the agent's own entry point
  (``get_streaming_response_with_tools``) and Vertex's streamed parse;
* the agent's history entry carrying it survives a JSON round trip, and a real
  agent turn sends it on its follow-up, while the next turn, rebuilt from the
  stored history, replays no call that would need one.
"""

from __future__ import annotations

import base64
import copy
import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.llm.adapter import ToolCall, ToolCallWithExtraContent
from kestrel_sovereign.llm.gemini_contents import (
    to_gemini_contents,
    tool_call_from_part,
    tool_result_messages,
)
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter
from tests.unit.test_gemini_contents import (
    FIRST_TURN,
    MODEL,
    PEERS_RESULT,
    ROUTE_IDS,
    ROUTES,
    TOOLS,
    WEATHER_RESULT,
    WIRE_QUESTION,
    _vertex_client,
)

#: Opaque signature bytes: not UTF-8, and their base64 holds ``+`` and ``/``,
#: which URL-safe base64 writes differently, so an alphabet slip shows.
SIGNATURE = bytes([0xFB, 0xEF, 0xFF, 0x00, 0x3E, 0x3F, 0x80]) + b"sig-A"
OTHER_SIGNATURE = bytes([0xFF, 0xFE, 0xFD]) + b"sig-B"
WIRE_SIGNATURE = base64.b64encode(SIGNATURE).decode("ascii")
ANSWER = "Sunny, 21C."
RESULTS = {"get_weather": WEATHER_RESULT, "list_peers": PEERS_RESULT}

WEATHER_CALL = {"functionCall": {"name": "get_weather", "args": {"location": "Tokyo"}}}
PEERS_CALL = {"functionCall": {"name": "list_peers", "args": {}}}


def _signed(part: Dict[str, Any], signature: bytes = SIGNATURE) -> Dict[str, Any]:
    """``part`` as Gemini 3 sends it: the signature beside the call."""
    return {**part, "thoughtSignature": base64.b64encode(signature).decode("ascii")}


def _wire_bytes(value: str) -> bytes:
    """A wire signature's bytes: the SDK sends standard base64 in API-key
    mode and URL-safe base64 in Vertex mode; both are valid proto JSON."""
    return base64.urlsafe_b64decode(value)


class _Gemini:
    """A mock Gemini endpoint: the first generation answers ``call_parts``,
    every later one answers text. Records every request."""

    def __init__(self, call_parts: List[Dict[str, Any]]) -> None:
        self.call_parts = call_parts
        self.requests: List[Any] = []

    def __call__(self, request):
        import httpx

        self.requests.append(request)
        if request.method == "GET":
            # GoogleAdapter looks up the model's output ceiling (#3355).
            return httpx.Response(200, json={
                "name": f"models/{MODEL}", "outputTokenLimit": 65_536,
            })
        if not request.url.path.endswith("enerateContent"):
            # Retrieval embeds the turn on the same route; this model is
            # chat-only.
            return httpx.Response(404, json={"error": {
                "code": 404, "message": "not found", "status": "NOT_FOUND",
            }})
        first = len(self.generations()) == 1
        parts = self.call_parts if first else [{"text": ANSWER}]
        candidate = {
            "content": {"role": "model", "parts": parts}, "finishReason": "STOP",
        }
        if request.url.path.endswith(":streamGenerateContent"):
            sse = f"data: {json.dumps({'candidates': [candidate]})}\r\n\r\n"
            return httpx.Response(
                200, content=sse.encode(), headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"candidates": [candidate]})

    def generations(self) -> List[Dict[str, Any]]:
        return [
            json.loads(r.content) for r in self.requests
            if r.method == "POST" and r.url.path.endswith("enerateContent")
        ]

    def follow_up(self) -> Dict[str, Any]:
        """The request that replays the first generation's calls."""
        bodies = self.generations()
        assert len(bodies) == 2, [str(r.url) for r in self.requests]
        return bodies[1]


def _model_turn(body: Dict[str, Any]) -> Dict[str, Any]:
    [turn] = [content for content in body["contents"] if content["role"] == "model"]
    return turn


def _history_message(response) -> Dict[str, Any]:
    """The assistant message the agent's tool loop replays ``response`` as."""
    return OrchestratorEngineMixin._build_assistant_tool_history_msg(
        OrchestratorEngineMixin(), response,
    )


def _tool_round(response) -> List[Dict[str, Any]]:
    """The agent's messages after it ran ``response``'s calls."""
    return [
        *FIRST_TURN,
        _history_message(response),
        *tool_result_messages([
            {"tool_call_id": call.id, "content": RESULTS[call.name]}
            for call in response.tool_calls
        ]),
    ]


async def _follow_up(route, wire: _Gemini, messages, path: str) -> Any:
    adapter = route.adapter()
    client = route.client(wire)
    if path == "generate":
        return await adapter.get_response(
            client=client, model=MODEL, messages=messages, tools=TOOLS,
        )
    return [item async for item in adapter.get_streaming_response(
        client=client, model=MODEL, messages=messages, tools=TOOLS,
    )]


# ---------------------------------------------------------------------------
# The real google-genai SDK, both adapters, call -> history -> follow-up
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["generate", "stream"])
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_a_signed_calls_follow_up_sends_its_signature_back(route, path):
    wire = _Gemini([{"text": "Let me check."}, _signed(WEATHER_CALL)])

    response = await route.adapter().get_response(
        client=route.client(wire), model=MODEL, messages=FIRST_TURN, tools=TOOLS,
    )
    [call] = response.tool_calls
    assert isinstance(call, ToolCallWithExtraContent)
    assert (call.name, call.arguments) == ("get_weather", {"location": "Tokyo"})

    await _follow_up(route, wire, _tool_round(response), path)

    model_turn = _model_turn(wire.follow_up())
    text, replayed = model_turn["parts"]
    assert text == {"text": "Let me check."}
    signature = replayed.pop("thoughtSignature")
    assert _wire_bytes(signature) == SIGNATURE
    assert replayed == WEATHER_CALL


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_of_parallel_calls_only_the_signed_first_gets_its_signature(route):
    wire = _Gemini([_signed(WEATHER_CALL), PEERS_CALL])

    response = await route.adapter().get_response(
        client=route.client(wire), model=MODEL, messages=FIRST_TURN, tools=TOOLS,
    )
    first, second = response.tool_calls
    assert isinstance(first, ToolCallWithExtraContent)
    assert type(second) is ToolCall

    await _follow_up(route, wire, _tool_round(response), "generate")

    replayed_first, replayed_second = _model_turn(wire.follow_up())["parts"]
    assert _wire_bytes(replayed_first.pop("thoughtSignature")) == SIGNATURE
    assert replayed_first == WEATHER_CALL
    assert replayed_second == PEERS_CALL


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_sequential_signed_calls_each_get_their_own_signature(route):
    """Multi-step: every round's call is signed, and every signature of the
    turn goes back with its own call."""
    wire = _Gemini([_signed(WEATHER_CALL)])
    adapter, client = route.adapter(), route.client(wire)
    first = await adapter.get_response(
        client=client, model=MODEL, messages=FIRST_TURN, tools=TOOLS,
    )
    wire.call_parts = [_signed(PEERS_CALL, OTHER_SIGNATURE)]
    messages = _tool_round(first)
    # The second round's calls come back on the next generation.
    wire.requests.clear()
    second = await adapter.get_response(
        client=client, model=MODEL, messages=messages, tools=TOOLS,
    )
    messages += [
        _history_message(second),
        *tool_result_messages([
            {"tool_call_id": second.tool_calls[0].id, "content": PEERS_RESULT},
        ]),
    ]

    await adapter.get_response(client=client, model=MODEL, messages=messages, tools=TOOLS)

    model_turns = [
        content for content in wire.follow_up()["contents"] if content["role"] == "model"
    ]
    assert [
        _wire_bytes(turn["parts"][0]["thoughtSignature"]) for turn in model_turns
    ] == [SIGNATURE, OTHER_SIGNATURE]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["generate", "stream"])
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_an_unsigned_calls_follow_up_is_the_request_sent_before(route, path):
    """A model that does not sign (Gemini 2.x) gets exactly the #3608 request."""
    wire = _Gemini([{"text": "Let me check."}, WEATHER_CALL, PEERS_CALL])

    response = await route.adapter().get_response(
        client=route.client(wire), model=MODEL, messages=FIRST_TURN, tools=TOOLS,
    )
    assert all(type(call) is ToolCall for call in response.tool_calls)
    messages = _tool_round(response)
    assert all("extra_content" not in call for call in messages[2]["tool_calls"])

    await _follow_up(route, wire, messages, path)

    assert wire.follow_up()["contents"] == [
        WIRE_QUESTION,
        {"role": "model", "parts": [{"text": "Let me check."}, WEATHER_CALL, PEERS_CALL]},
        {"role": "user", "parts": [
            {"functionResponse": {
                "name": "get_weather", "response": {"result": WEATHER_RESULT},
            }},
            {"functionResponse": {
                "name": "list_peers", "response": {"result": PEERS_RESULT},
            }},
        ]},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_the_agents_streaming_entry_point_keeps_the_signature(route):
    """``get_streaming_response_with_tools`` is how the agent's turns call the
    adapter; the signed call it yields, replayed through it, is sent signed."""
    wire = _Gemini([_signed(WEATHER_CALL)])
    adapter, client = route.adapter(), route.client(wire)

    items = [item async for item in adapter.get_streaming_response_with_tools(
        client=client, model=MODEL, messages=FIRST_TURN, tools=TOOLS,
    )]
    [response] = items
    items = [item async for item in adapter.get_streaming_response_with_tools(
        client=client, model=MODEL, messages=_tool_round(response), tools=TOOLS,
    )]

    assert items[0] == ANSWER
    [replayed] = _model_turn(wire.follow_up())["parts"]
    assert _wire_bytes(replayed.pop("thoughtSignature")) == SIGNATURE
    assert replayed == WEATHER_CALL


@pytest.mark.asyncio
async def test_a_call_streamed_by_vertex_keeps_its_signature():
    """Vertex reads calls off its stream onto the terminal response (#2811)."""
    wire = _Gemini([_signed(WEATHER_CALL), PEERS_CALL])
    adapter, client = VertexAIAdapter(), _vertex_client(wire)

    items = [item async for item in adapter._stream_with_usage(
        client=client, model=MODEL, messages=FIRST_TURN, tools=TOOLS,
    )]
    response = items[-1]
    first, second = response.tool_calls
    assert first.extra_content == {"google": {"thought_signature": WIRE_SIGNATURE}}
    assert type(second) is ToolCall

    [item async for item in adapter.get_streaming_response(
        client=client, model=MODEL, messages=_tool_round(response), tools=TOOLS,
    )]

    replayed_first, replayed_second = _model_turn(wire.follow_up())["parts"]
    assert _wire_bytes(replayed_first.pop("thoughtSignature")) == SIGNATURE
    assert replayed_first == WEATHER_CALL
    assert replayed_second == PEERS_CALL


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_the_history_entry_survives_a_json_round_trip(route):
    """The signature rides the history entry as base64 text, so a history
    that is serialized and reloaded still sends the same bytes."""
    wire = _Gemini([_signed(WEATHER_CALL)])
    response = await route.adapter().get_response(
        client=route.client(wire), model=MODEL, messages=FIRST_TURN, tools=TOOLS,
    )

    reloaded = json.loads(json.dumps(_tool_round(response)))
    await _follow_up(route, wire, reloaded, "generate")

    [replayed] = _model_turn(wire.follow_up())["parts"]
    assert _wire_bytes(replayed["thoughtSignature"]) == SIGNATURE


# ---------------------------------------------------------------------------
# The parse, the history entry, and the conversion, directly
# ---------------------------------------------------------------------------


def _part(signature: Any = None, **call) -> SimpleNamespace:
    return SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name=call.get("name", "get_weather"),
                                      args=call.get("args", {"location": "Tokyo"})),
        thought_signature=signature,
    )


def test_a_signed_part_keeps_its_signature_as_base64_extra_content():
    call = tool_call_from_part(_part(SIGNATURE), "vertex_call_0")

    assert call == ToolCallWithExtraContent(
        id="vertex_call_0", name="get_weather", arguments={"location": "Tokyo"},
        extra_content={"google": {"thought_signature": WIRE_SIGNATURE}},
    )


def test_an_unsigned_part_is_a_plain_tool_call():
    call = tool_call_from_part(_part(None), "gemini_call_0")

    assert call == ToolCall(
        id="gemini_call_0", name="get_weather", arguments={"location": "Tokyo"},
    )
    assert type(call) is ToolCall


def test_a_part_with_no_call_is_no_tool_call():
    part = SimpleNamespace(text="", function_call=None, thought_signature=SIGNATURE)

    assert tool_call_from_part(part, "gemini_call_0") is None


def test_a_signature_that_is_not_bytes_is_refused():
    with pytest.raises(TypeError, match="get_weather"):
        tool_call_from_part(_part(WIRE_SIGNATURE), "gemini_call_0")


def _signed_response():
    return SimpleNamespace(
        content="Let me check.",
        raw=None,
        tool_calls=[
            tool_call_from_part(_part(SIGNATURE), "gemini_call_0"),
            tool_call_from_part(_part(None, name="list_peers", args={}), "gemini_call_1"),
        ],
    )


EXPECTED_HISTORY_CALLS = [
    {
        "id": "gemini_call_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"location": "Tokyo"}},
        "extra_content": {"google": {"thought_signature": WIRE_SIGNATURE}},
    },
    {
        "id": "gemini_call_1",
        "type": "function",
        "function": {"name": "list_peers", "arguments": {}},
    },
]


@pytest.mark.parametrize("arguments, replayed", [
    ({"location": "Tokyo"}, {"location": "Tokyo"}),
    ('{"location": "Tokyo"}', {"location": "Tokyo"}),
    ("", {}),
    (None, {}),
], ids=["dict", "json-string", "empty-string", "none"])
def test_a_plain_calls_history_entry_is_the_entry_built_before(arguments, replayed):
    """Both tool loops' builders now share one entry; a call carrying no
    provider data gets exactly the entry they built before #3619."""
    entry = {
        "id": "call_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": replayed},
    }
    response = SimpleNamespace(content="", raw=None, tool_calls=[
        ToolCall(id="call_0", name="get_weather", arguments=arguments),
    ])
    feature = SimpleNamespace(
        _extract_response_reasoning_content=Feature._extract_response_reasoning_content,
    )

    assert _history_message(response)["tool_calls"] == [entry]
    assert Feature._build_subagent_assistant_tool_history_msg(
        feature, response,
    )["tool_calls"] == [entry]


def test_the_orchestrators_history_entry_carries_extra_content():
    assert _history_message(_signed_response())["tool_calls"] == EXPECTED_HISTORY_CALLS


def test_a_feature_subagents_history_entry_carries_extra_content():
    """Feature subagent loops replay calls through their own builder."""
    feature = SimpleNamespace(
        _extract_response_reasoning_content=Feature._extract_response_reasoning_content,
    )

    message = Feature._build_subagent_assistant_tool_history_msg(
        feature, _signed_response(),
    )

    assert message["tool_calls"] == EXPECTED_HISTORY_CALLS


def test_conversion_restores_the_signature_onto_its_own_part_only():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": copy.deepcopy(
            EXPECTED_HISTORY_CALLS,
        )},
    ]
    sent = copy.deepcopy(messages)

    _, contents = to_gemini_contents(messages)

    assert messages == sent
    assert contents[-1] == {"role": "model", "parts": [
        {"function_call": {"name": "get_weather", "args": {"location": "Tokyo"}},
         "thought_signature": SIGNATURE},
        {"function_call": {"name": "list_peers", "args": {}}},
    ]}


@pytest.mark.parametrize("extra_content", [
    None, {}, {"openai": {"anything": 1}}, {"google": {}},
], ids=["none", "empty", "other-provider", "google-without-signature"])
def test_a_call_without_a_gemini_signature_converts_as_before(extra_content):
    call = {"id": "c0", "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"}}
    if extra_content is not None:
        call["extra_content"] = extra_content

    _, contents = to_gemini_contents([{"role": "assistant", "tool_calls": [call]}])

    assert contents == [{"role": "model", "parts": [
        {"function_call": {"name": "get_weather", "args": {}}},
    ]}]


@pytest.mark.parametrize("extra_content, match", [
    ("text", "extra_content that is not an object"),
    ({"google": "sig"}, "Gemini extra_content that is not an object"),
    ({"google": {"thought_signature": 3}}, "not a string"),
    ({"google": {"thought_signature": "not base64!"}}, "not base64"),
], ids=["extra-not-object", "google-not-object", "signature-not-string",
        "signature-not-base64"])
def test_a_malformed_signature_is_refused_before_any_request(extra_content, match):
    with pytest.raises(ValueError, match=match):
        to_gemini_contents([{"role": "assistant", "tool_calls": [{
            "id": "c0", "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"},
            "extra_content": extra_content,
        }]}])


# ---------------------------------------------------------------------------
# A real agent turn on a Gemini route
# ---------------------------------------------------------------------------


def _gemini_route(agent, route, wire: _Gemini) -> dict:
    """A real Gemini route, built as the registry builds one, whose client
    is the real SDK over ``wire``."""
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    adapter_name = "VertexAIAdapter" if route.name == "vertex" else "GoogleAdapter"
    info = ProviderRegistry({})._build_route(
        "google", route.name, {"is_cloud": True},
        {"adapter": adapter_name, "api_key": "test-key", "model": MODEL},
    )
    [provider] = agent.llm_service._convert_providers_format([info])
    provider["client"] = route.client(wire)
    return provider


async def _turn(agent, path: str, prompt: str, session_id: str) -> str:
    if path == "stream":
        chunks = [
            chunk async for chunk in agent.process_input_streaming(
                prompt, session_id=session_id, request_id=f"{session_id}-{prompt}",
            )
        ]
        return "".join(chunk for chunk in chunks if isinstance(chunk, str))
    return await agent.process_input(prompt, session_id=session_id)


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize("path", ["invoke", "stream"])
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_a_real_agent_turn_sends_the_signature_and_the_next_turn_needs_none(
    tmp_path, monkeypatch, route, path,
):
    """The agent's own tool loop replays the signed call with its signature.

    Gemini validates the calls of the current turn only: those after the
    latest user text. The next turn is rebuilt from the stored history, which
    keeps a tool turn as its text, so it replays no call to sign.
    """
    from tests.shared.booted_agent import booted_agent

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    wire = _Gemini([_signed({"functionCall": {"name": "state_of_mind", "args": {}}})])
    async with booted_agent(
        tmp_path, name="Signed", provenance="test:gemini_thought_signatures",
        features={"StateOfMindFeature"},
    ) as agent:
        service = agent.llm_service
        provider = _gemini_route(agent, route, wire)
        monkeypatch.setattr(service, "providers", [provider])
        monkeypatch.setattr(service, "config", {
            **(service.config or {}), "route_priority": [provider["name"]],
        })
        monkeypatch.setattr(service, "_mandate_preference", {})
        session_id = f"signed-{route.name}-{path}"

        first = await _turn(agent, path, "How are you feeling?", session_id)
        second = await _turn(agent, path, "And now?", session_id)

    assert ANSWER in first and ANSWER in second
    turns = wire.generations()
    assert len(turns) == 3, [str(r.url) for r in wire.requests]
    calls, follow_up, next_turn = turns
    signed = [
        part for content in follow_up["contents"] for part in content["parts"]
        if "functionCall" in part
    ]
    assert [part["functionCall"]["name"] for part in signed] == ["state_of_mind"]
    assert _wire_bytes(signed[0]["thoughtSignature"]) == SIGNATURE
    assert not any(
        "functionCall" in part or "thoughtSignature" in part
        for content in next_turn["contents"] for part in content["parts"]
    )
