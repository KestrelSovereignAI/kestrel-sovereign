"""#3608: the Google and Vertex adapters send the agent's messages as Gemini contents.

The agent hands every adapter OpenAI-format messages (``content``, an
``assistant`` turn with ``tool_calls``, ``role: "tool"`` results), and both
adapters passed them to google-genai as ``contents=`` unchanged. The SDK
validates ``contents`` as Gemini ``Content`` and refused them with a pydantic
``ValidationError`` before any request was sent, so an agent turn on a
``google:vertex`` (or ``google``) route failed on its first message, and the
tool loop's follow-up failed too. The existing tests drove a fake client that
accepts any ``contents``, so they could not see it.

These tests run the real google-genai client over an ``httpx.MockTransport``,
so the SDK's own validation and serialization run, and pin for both adapters:

* an OpenAI-format first turn and a full tool-loop follow-up reach the wire as
  Gemini contents (roles, ``functionCall``, ``functionResponse`` named after
  its call), on the plain and the streamed request;
* a message list already in Gemini shape is sent unchanged;
* ``continue_with_tool_results`` answers the calls of the turn before it;

and pin :func:`to_gemini_contents` itself directly.
"""

from __future__ import annotations

import base64
import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import pytest

from kestrel_sovereign.llm.gemini_contents import (
    to_gemini_contents,
    tool_result_messages,
)
from kestrel_sovereign.llm.google_adapter import GoogleAdapter
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter

MODEL = "gemini-2.5-flash"
SYSTEM = "You are Kestrel. Be brief."
QUESTION = "What's the weather in Tokyo?"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
    {"type": "function", "function": {"name": "list_peers", "description": "List peers"}},
]
WEATHER_RESULT = json.dumps({"success": True, "temperature_c": 21})
PEERS_RESULT = json.dumps({"success": True, "peers": []})

# What the agent builds (streaming.py, orchestrator_engine.py): a system turn,
# the user's turn, then per tool round an assistant turn carrying tool_calls
# and one tool turn per call. The orchestrator keeps arguments as a dict; the
# inline path sends a JSON string. Both appear here.
FIRST_TURN = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": QUESTION},
]
TOOL_LOOP = FIRST_TURN + [
    {
        "role": "assistant",
        "content": "Let me check.",
        "tool_calls": [
            {
                "id": "vertex_call_0",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"location": "Tokyo"}),
                },
            },
            {
                "id": "vertex_call_1",
                "type": "function",
                "function": {"name": "list_peers", "arguments": {}},
            },
        ],
    },
    {"role": "tool", "tool_call_id": "vertex_call_0", "content": WEATHER_RESULT},
    {"role": "tool", "tool_call_id": "vertex_call_1", "content": PEERS_RESULT},
]
WIRE_QUESTION = {"role": "user", "parts": [{"text": QUESTION}]}
WIRE_TOOL_LOOP = [
    WIRE_QUESTION,
    {"role": "model", "parts": [
        {"text": "Let me check."},
        {"functionCall": {"name": "get_weather", "args": {"location": "Tokyo"}}},
        {"functionCall": {"name": "list_peers", "args": {}}},
    ]},
    {"role": "user", "parts": [
        {"functionResponse": {"name": "get_weather", "response": {"result": WEATHER_RESULT}}},
        {"functionResponse": {"name": "list_peers", "response": {"result": PEERS_RESULT}}},
    ]},
]


# ---------------------------------------------------------------------------
# The real google-genai SDK, over a mock HTTP transport, for both adapters
# ---------------------------------------------------------------------------


def _http_options(handler):
    httpx = pytest.importorskip("httpx")
    genai_types = pytest.importorskip("google.genai.types")
    if "httpx_async_client" not in genai_types.HttpOptions.model_fields:
        pytest.skip("installed google-genai cannot take an httpx client")
    return genai_types.HttpOptions(
        httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _vertex_client(handler):
    genai = pytest.importorskip("google.genai")
    oauth2 = pytest.importorskip("google.oauth2.credentials")
    return genai.Client(
        vertexai=True,
        project="test-project",
        location="us-central1",
        # A static token: nothing refreshes it, so nothing leaves the process.
        credentials=oauth2.Credentials(token="test-token"),
        http_options=_http_options(handler),
    )


def _google_client(handler):
    genai = pytest.importorskip("google.genai")
    return genai.Client(api_key="test-key", http_options=_http_options(handler))


@dataclass(frozen=True)
class _Route:
    name: str
    adapter: Callable[[], Any]
    client: Callable[[Any], Any]


ROUTES = [
    _Route("vertex", VertexAIAdapter, _vertex_client),
    _Route("google", GoogleAdapter, _google_client),
]
ROUTE_IDS = [route.name for route in ROUTES]


class _Wire:
    """A mock Gemini endpoint that records every request it receives."""

    def __init__(self) -> None:
        self.requests: List[Any] = []

    def __call__(self, request):
        import httpx

        self.requests.append(request)
        if request.method == "GET":
            # GoogleAdapter looks up the model's output ceiling (#3355).
            return httpx.Response(200, json={
                "name": f"models/{MODEL}", "outputTokenLimit": 65_536,
            })
        candidate = {
            "content": {"role": "model", "parts": [{"text": "Sunny, 21C."}]},
            "finishReason": "STOP",
        }
        if request.url.path.endswith(":streamGenerateContent"):
            sse = f"data: {json.dumps({'candidates': [candidate]})}\r\n\r\n"
            return httpx.Response(
                200, content=sse.encode(), headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"candidates": [candidate]})

    def generation_body(self) -> Dict[str, Any]:
        posts = [r for r in self.requests if r.method == "POST"]
        assert len(posts) == 1, [str(r.url) for r in self.requests]
        return json.loads(posts[0].content)

    def generation_path(self) -> str:
        return next(r.url.path for r in self.requests if r.method == "POST")


async def _request(route: _Route, path: str, wire: _Wire, messages, **kwargs) -> Any:
    """Send ``messages`` through ``route`` on the plain or the streamed request."""
    adapter = route.adapter()
    client = route.client(wire)
    if path == "generate":
        return await adapter.get_response(
            client=client, model=MODEL, messages=messages, **kwargs,
        )
    return [item async for item in adapter.get_streaming_response(
        client=client, model=MODEL, messages=messages, **kwargs,
    )]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["generate", "stream"])
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_an_openai_first_turn_reaches_the_wire_as_gemini_contents(route, path):
    wire = _Wire()

    await _request(route, path, wire, FIRST_TURN, tools=TOOLS)

    expected_method = ":generateContent" if path == "generate" else ":streamGenerateContent"
    assert wire.generation_path().endswith(f"/models/{MODEL}{expected_method}")
    body = wire.generation_body()
    assert body["contents"] == [WIRE_QUESTION]
    assert body["systemInstruction"]["parts"] == [{"text": SYSTEM}]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["generate", "stream"])
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_a_tool_loop_follow_up_reaches_the_wire_as_gemini_contents(route, path):
    wire = _Wire()

    await _request(route, path, wire, TOOL_LOOP, tools=TOOLS)

    body = wire.generation_body()
    assert body["contents"] == WIRE_TOOL_LOOP
    assert body["systemInstruction"]["parts"] == [{"text": SYSTEM}]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_the_agents_tool_detection_turn_is_sent_and_answered(route):
    """The agent's own entry point: ``get_streaming_response_with_tools``."""
    wire = _Wire()

    items = [item async for item in route.adapter().get_streaming_response_with_tools(
        client=route.client(wire), model=MODEL, messages=TOOL_LOOP, tools=TOOLS,
    )]

    assert wire.generation_body()["contents"] == WIRE_TOOL_LOOP
    assert items[0] == "Sunny, 21C."
    assert items[-1].content == "Sunny, 21C."


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["generate", "stream"])
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_a_gemini_shaped_message_list_is_sent_unchanged(route, path):
    messages = [
        {"role": "user", "parts": [{"text": QUESTION}]},
        {"role": "model", "parts": [
            {"function_call": {"name": "get_weather", "args": {"location": "Tokyo"}}},
        ]},
        {"role": "user", "parts": [
            {"function_response": {"name": "get_weather", "response": {"result": "Sunny"}}},
        ]},
    ]
    sent = copy.deepcopy(messages)
    wire = _Wire()

    await _request(route, path, wire, messages, tools=TOOLS)

    assert messages == sent
    body = wire.generation_body()
    assert body["contents"] == [
        WIRE_QUESTION,
        {"role": "model", "parts": [
            {"functionCall": {"name": "get_weather", "args": {"location": "Tokyo"}}},
        ]},
        {"role": "user", "parts": [
            {"functionResponse": {"name": "get_weather", "response": {"result": "Sunny"}}},
        ]},
    ]
    assert "systemInstruction" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_continue_with_tool_results_answers_the_calls_of_the_turn_before_it(route):
    """The results become ``function_response`` parts named after the calls
    of the assistant turn they follow, sent after that turn, never as a
    ``role: "function"`` turn with an ``"unknown"`` name."""
    wire = _Wire()

    response = await route.adapter().continue_with_tool_results(
        client=route.client(wire),
        model=MODEL,
        messages=TOOL_LOOP[:3],
        tool_results=[
            {"tool_call_id": "vertex_call_0", "content": WEATHER_RESULT},
            {"tool_call_id": "vertex_call_1", "content": PEERS_RESULT},
        ],
        tools=TOOLS,
    )

    assert wire.generation_body()["contents"] == WIRE_TOOL_LOOP
    assert response.content == "Sunny, 21C."


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_a_tool_result_without_its_call_is_refused_before_any_request(route):
    wire = _Wire()

    with pytest.raises(ValueError, match="answers no tool call"):
        await route.adapter().continue_with_tool_results(
            client=route.client(wire),
            model=MODEL,
            messages=FIRST_TURN,
            tool_results=[{"tool_call_id": "vertex_call_0", "content": WEATHER_RESULT}],
            tools=TOOLS,
        )

    assert wire.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_an_openai_image_part_reaches_the_wire_as_inline_data(route):
    image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    wire = _Wire()

    await route.adapter().get_response(
        client=route.client(wire),
        model=MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        ]}],
    )

    [turn] = wire.generation_body()["contents"]
    assert turn["role"] == "user"
    text, image_part = turn["parts"]
    assert text == {"text": "What is this?"}
    # The SDK re-encodes the bytes it decoded: URL-safe base64 on Vertex,
    # and Vertex keeps the Blob's field name as written. Both are valid
    # proto JSON; what matters is that the image itself arrives.
    inline = dict(image_part["inlineData"])
    assert base64.urlsafe_b64decode(inline.pop("data")) == base64.b64decode(image)
    assert inline in ({"mimeType": "image/png"}, {"mime_type": "image/png"})


# ---------------------------------------------------------------------------
# to_gemini_contents
# ---------------------------------------------------------------------------


def test_system_turns_become_one_system_instruction_in_order():
    """``VertexAIAdapter``'s ``_system`` marker and every ``system`` turn,
    including one mid-history (a compaction marker), are lifted out."""
    system, contents = to_gemini_contents([
        {"role": "_system", "parts": [{"text": "Marker."}]},
        {"role": "system", "content": "Leading."},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": [{"type": "text", "text": "Compacted."}]},
        {"role": "assistant", "content": "hello"},
    ])

    assert system == "Marker.\n\nLeading.\n\nCompacted."
    assert contents == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
    ]


def test_no_system_turn_means_no_system_instruction():
    assert to_gemini_contents([{"role": "user", "content": "hi"}])[0] is None


def test_gemini_shaped_messages_pass_through_as_the_same_objects():
    messages = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
        {"role": "function", "parts": [
            {"function_response": {"name": "f", "response": {"result": "x"}}},
        ]},
    ]

    system, contents = to_gemini_contents(messages)

    assert system is None
    assert len(contents) == len(messages)
    assert all(sent is given for sent, given in zip(contents, messages))


def test_the_messages_given_are_not_changed():
    messages = copy.deepcopy(TOOL_LOOP)

    to_gemini_contents(messages)

    assert messages == TOOL_LOOP


@pytest.mark.parametrize("arguments, expected", [
    (json.dumps({"location": "Tokyo"}), {"location": "Tokyo"}),
    ({"location": "Tokyo"}, {"location": "Tokyo"}),
    ("", {}),
    (None, {}),
    ("null", {}),
], ids=["json-string", "dict", "empty-string", "none", "json-null"])
def test_tool_call_arguments_are_json_decoded(arguments, expected):
    _, contents = to_gemini_contents([{"role": "assistant", "tool_calls": [
        {"id": "c0", "type": "function",
         "function": {"name": "get_weather", "arguments": arguments}},
    ]}])

    assert contents == [{"role": "model", "parts": [
        {"function_call": {"name": "get_weather", "args": expected}},
    ]}]


@pytest.mark.parametrize("arguments", ["not json", "[1, 2]", "3"])
def test_tool_call_arguments_that_are_not_a_json_object_are_refused(arguments):
    with pytest.raises(ValueError, match="get_weather"):
        to_gemini_contents([{"role": "assistant", "tool_calls": [
            {"id": "c0", "function": {"name": "get_weather", "arguments": arguments}},
        ]}])


def test_a_tool_call_with_no_function_name_is_refused():
    with pytest.raises(ValueError, match="no function name"):
        to_gemini_contents([{"role": "assistant", "tool_calls": [
            {"id": "c0", "function": {"arguments": "{}"}},
        ]}])


def _calls(*calls) -> Dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
        for call_id, name in calls
    ]}


def _result(call_id: str, content: str = "ok") -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _response_names(contents) -> List[List[str]]:
    return [
        [part["function_response"]["name"] for part in turn["parts"]]
        for turn in contents if "function_response" in turn["parts"][0]
    ]


def test_a_tool_result_is_named_from_the_model_turn_it_follows():
    """Adapters number calls per response, so ``vertex_call_0`` names a
    different call in every round of the tool loop."""
    _, contents = to_gemini_contents([
        {"role": "user", "content": "go"},
        _calls(("vertex_call_0", "get_weather")),
        _result("vertex_call_0"),
        _calls(("vertex_call_0", "list_peers"), ("vertex_call_1", "get_weather")),
        _result("vertex_call_1"),
        _result("vertex_call_0"),
    ])

    assert _response_names(contents) == [["get_weather"], ["get_weather", "list_peers"]]
    assert [turn["role"] for turn in contents] == ["user", "model", "user", "model", "user"]


def test_a_system_turn_between_tool_results_keeps_them_in_one_turn():
    system, contents = to_gemini_contents([
        _calls(("c0", "get_weather"), ("c1", "list_peers")),
        _result("c0"),
        {"role": "system", "content": "Operator notice."},
        _result("c1"),
    ])

    assert system == "Operator notice."
    assert _response_names(contents) == [["get_weather", "list_peers"]]


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "go"}, _result("c0")],
    [_calls(("c0", "get_weather")), {"role": "user", "content": "go"}, _result("c0")],
    [_calls(("c0", "get_weather")), _result("c1")],
    [_calls(("c0", "get_weather")), {"role": "tool", "content": "ok"}],
], ids=["no-call", "call-before-user-turn", "other-id", "no-id"])
def test_a_tool_result_that_answers_no_call_of_the_turn_before_it_is_refused(messages):
    with pytest.raises(ValueError, match="answers no tool call"):
        to_gemini_contents(messages)


def test_tool_result_content_carries_its_text():
    _, contents = to_gemini_contents([
        _calls(("c0", "get_weather")),
        {"role": "tool", "tool_call_id": "c0", "content": [
            {"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"},
        ]},
    ])

    assert contents[-1]["parts"][0]["function_response"]["response"] == {
        "result": "line 1\nline 2",
    }


def test_empty_text_is_dropped_and_an_empty_turn_is_not_sent():
    """Gemini refuses an empty text part and a turn without parts."""
    _, contents = to_gemini_contents([
        {"role": "user", "content": [
            {"type": "text", "text": ""}, "", {"type": "text", "text": "hi"},
        ]},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "again"},
    ])

    assert contents == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "user", "parts": [{"text": "again"}]},
    ]


@pytest.mark.parametrize("url, part", [
    (
        "data:image/jpeg;base64,/9j/4AAQ",
        {"inline_data": {"mime_type": "image/jpeg", "data": "/9j/4AAQ"}},
    ),
    (
        # RFC 2397: the scheme and the base64 token are case-insensitive.
        "DATA:image/png;name=a.png;BASE64,iVBORw0K",
        {"inline_data": {"mime_type": "image/png", "data": "iVBORw0K"}},
    ),
    (
        "https://example.com/images/cat.png?signature=abc",
        {"file_data": {"file_uri": "https://example.com/images/cat.png?signature=abc",
                       "mime_type": "image/png"}},
    ),
    (
        "gs://bucket/photo.jpeg",
        {"file_data": {"file_uri": "gs://bucket/photo.jpeg", "mime_type": "image/jpeg"}},
    ),
], ids=["data-url", "data-url-uppercase-with-parameter", "https", "gcs"])
def test_image_url_parts_become_inline_or_file_data(url, part):
    for image_url in ({"url": url, "detail": "high"}, url):
        _, contents = to_gemini_contents([{"role": "user", "content": [
            {"type": "image_url", "image_url": image_url},
        ]}])
        assert contents == [{"role": "user", "parts": [part]}]


@pytest.mark.parametrize("messages, match", [
    ([{"role": "function", "content": "x"}], "role 'function'"),
    ([{"content": "x"}], "role None"),
    ([{"role": "user", "content": 3}], "of type int"),
    ([{"role": "user", "content": [{"type": "input_audio"}]}], "'input_audio'"),
    ([{"role": "user", "content": [{"type": "text"}]}], "carries no text"),
    ([{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
     "carries no url"),
    ([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png,rawbytes"}},
    ]}], "must be base64"),
    ([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image"}},
    ]}], "MIME type"),
    ([{"role": "system", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]}], "can only carry text"),
], ids=[
    "unknown-role", "no-role", "content-not-text-or-list", "unknown-part",
    "text-part-without-text", "image-without-url", "data-url-not-base64",
    "url-without-mime-type", "image-in-system-turn",
])
def test_a_message_gemini_cannot_be_sent_is_refused(messages, match):
    with pytest.raises(ValueError, match=match):
        to_gemini_contents(messages)


def test_tool_result_messages_are_openai_tool_turns():
    assert tool_result_messages([
        {"tool_call_id": "c0", "content": "ok", "name": "get_weather"},
    ]) == [{"role": "tool", "tool_call_id": "c0", "content": "ok"}]
