"""#2811: the Vertex adapter sends its tools, and reads the calls that come back.

``VertexAIAdapter`` advertised ``supports_tools=True`` but built its function
declarations into a local that no request ever carried, on both
``get_response`` and the streamed path. Vertex was never told a tool existed,
so it answered in prose and the adapter's tool-call branch was unreachable.
Its response parser also branched on ``hasattr(part, "function_call")``,
which is true of every google-genai ``Part``, so a part with no text and no
call (a thought signature) crashed on ``None.name``.

These tests pin:

* a request with tools carries them on its config as ``function_declarations``
  (``GenerateContentConfig.tools``), on the plain and the streamed path, and a
  request without tools carries no ``tools`` key at all;
* a ``function_call`` part becomes a ``ToolCall``, and a part whose
  ``function_call`` is ``None`` never does;
* ``get_streaming_response_with_tools`` surfaces the call end to end;
* the real google-genai SDK, in Vertex mode over a mock transport, puts the
  declarations on the wire and reads the call back.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter

MODEL = "gemini-2.5-flash"
MESSAGES = [{"role": "user", "parts": [{"text": "What's the weather in Tokyo?"}]}]
WEATHER_PARAMETERS = {
    "type": "object",
    "properties": {"location": {"type": "string", "description": "City name"}},
    "required": ["location"],
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": WEATHER_PARAMETERS,
        },
    },
    {
        "type": "function",
        "function": {"name": "list_peers", "description": "List known peers"},
    },
]
EXPECTED_DECLARATIONS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": WEATHER_PARAMETERS,
    },
    {
        "name": "list_peers",
        "description": "List known peers",
        "parameters": {"type": "object", "properties": {}},
    },
]


def _text(text: Any) -> SimpleNamespace:
    """A google-genai ``Part`` shape: ``function_call`` is present, and None."""
    return SimpleNamespace(text=text, function_call=None)


def _call(name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        text=None, function_call=SimpleNamespace(name=name, args=args),
    )


def _response(parts: List[Any], *, text: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[SimpleNamespace(
            content=SimpleNamespace(parts=parts), finish_reason="STOP",
        )],
        text=text,
        usage_metadata=None,
    )


def _chunk(parts: List[Any], *, text: Any = None, finish_reason: Any = None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(
            content=SimpleNamespace(parts=parts), finish_reason=finish_reason,
        )],
        usage_metadata=None,
    )


def _client(response: Any = None, *, chunks: List[Any] = ()) -> SimpleNamespace:
    async def _stream(**_kwargs):
        async def _iterate():
            for chunk in chunks:
                yield chunk
        return _iterate()

    return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content=AsyncMock(return_value=response),
        generate_content_stream=AsyncMock(side_effect=_stream),
    )))


def _sent_config(call: AsyncMock) -> Any:
    call.assert_awaited_once()
    return call.await_args.kwargs["config"]


async def _collect(stream) -> List[Any]:
    return [item async for item in stream]


# ---------------------------------------------------------------------------
# The request: tools travel on the config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_response_sends_the_tools_as_function_declarations():
    client = _client(_response([_text("Sunny")]))

    await VertexAIAdapter().get_response(
        client=client, model=MODEL, messages=MESSAGES, tools=TOOLS,
    )

    config = _sent_config(client.aio.models.generate_content)
    assert config["tools"] == [{"function_declarations": EXPECTED_DECLARATIONS}]


@pytest.mark.asyncio
async def test_streaming_response_sends_the_tools_as_function_declarations():
    client = _client(chunks=[_chunk([_text("Sunny")], text="Sunny")])

    streamed = await _collect(VertexAIAdapter().get_streaming_response(
        client=client, model=MODEL, messages=MESSAGES, tools=TOOLS,
    ))

    assert streamed == ["Sunny"]
    config = _sent_config(client.aio.models.generate_content_stream)
    assert config["tools"] == [{"function_declarations": EXPECTED_DECLARATIONS}]


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [
    None,
    [],
    # Only "function" tools convert to declarations; nothing else is sent.
    [{"type": "web_search"}],
], ids=["none", "empty", "no-function-tools"])
async def test_a_request_without_tools_carries_no_tools_key(tools):
    adapter = VertexAIAdapter()
    client = _client(
        _response([_text("Sunny")]), chunks=[_chunk([_text("Sunny")], text="Sunny")],
    )

    await adapter.get_response(
        client=client, model=MODEL, messages=MESSAGES, tools=tools, max_tokens=64,
    )
    await _collect(adapter.get_streaming_response(
        client=client, model=MODEL, messages=MESSAGES, tools=tools, max_tokens=64,
    ))

    for call in (client.aio.models.generate_content,
                 client.aio.models.generate_content_stream):
        config = _sent_config(call)
        assert "tools" not in config
        assert config["max_output_tokens"] == 64


@pytest.mark.asyncio
async def test_a_request_with_nothing_to_configure_still_sends_no_config():
    """Unchanged: with no tools and no settings the config is omitted."""
    client = _client(_response([_text("Sunny")]))

    await VertexAIAdapter().get_response(client=client, model=MODEL, messages=MESSAGES)

    assert _sent_config(client.aio.models.generate_content) is None


class _Forecast(BaseModel):
    summary: str


@pytest.mark.asyncio
async def test_tools_travel_alongside_the_other_request_settings():
    adapter = VertexAIAdapter()
    messages = adapter.create_messages(
        user_prompt="Weather?", system_prompt="Be brief.",
    )
    client = _client(_response([_text("{}")]))

    await adapter.get_response(
        client=client, model=MODEL, messages=messages, tools=TOOLS,
        response_format=_Forecast, max_tokens=128, temperature=0.2,
    )

    config = _sent_config(client.aio.models.generate_content)
    assert config == {
        "max_output_tokens": 128,
        "temperature": 0.2,
        "system_instruction": "Be brief.",
        "response_mime_type": "application/json",
        "response_schema": _Forecast.model_json_schema(),
        "tools": [{"function_declarations": EXPECTED_DECLARATIONS}],
    }


# ---------------------------------------------------------------------------
# The response: a function_call part is a tool call, nothing else is
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_function_call_part_becomes_a_tool_call():
    client = _client(_response([
        _call("get_weather", {"location": "Tokyo"}),
        _call("list_peers", {}),
    ]))

    response = await VertexAIAdapter().get_response(
        client=client, model=MODEL, messages=MESSAGES, tools=TOOLS,
    )

    assert response.has_tool_calls
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [
        ("vertex_call_0", "get_weather", {"location": "Tokyo"}),
        ("vertex_call_1", "list_peers", {}),
    ]
    assert response.content is None


@pytest.mark.asyncio
async def test_a_part_whose_function_call_is_none_never_becomes_a_tool_call():
    """``hasattr(part, "function_call")`` is true of every google-genai Part;
    a part with neither text nor a call (a thought signature) must not be
    read as a call with a ``None`` name."""
    client = _client(_response([_text("Sunny in Tokyo"), _text(None), _text("")]))

    response = await VertexAIAdapter().get_response(
        client=client, model=MODEL, messages=MESSAGES, tools=TOOLS,
    )

    assert response.tool_calls is None
    assert not response.has_tool_calls
    assert response.content == "Sunny in Tokyo"


@pytest.mark.asyncio
async def test_streaming_with_tools_surfaces_the_call_end_to_end():
    client = _client(_response([_call("get_weather", {"location": "Tokyo"})]))

    items = await _collect(VertexAIAdapter().get_streaming_response_with_tools(
        client=client, model=MODEL, messages=MESSAGES, tools=TOOLS,
    ))

    assert len(items) == 1
    response = items[0]
    assert isinstance(response, LLMResponse)
    assert response.has_tool_calls
    assert [(c.name, c.arguments) for c in response.tool_calls] == [
        ("get_weather", {"location": "Tokyo"}),
    ]
    assert getattr(response, "stop_reason", None) == "STOP"
    config = _sent_config(client.aio.models.generate_content)
    assert config["tools"] == [{"function_declarations": EXPECTED_DECLARATIONS}]


@pytest.mark.asyncio
async def test_a_streamed_function_call_rides_the_terminal_response():
    """With tools on the streamed request, a call that arrives on the stream
    is carried on the terminal response rather than dropped."""
    client = _client(chunks=[
        _chunk([_text("Checking.")], text="Checking."),
        _chunk([_text(None), _call("get_weather", {"location": "Tokyo"})],
               finish_reason="STOP"),
    ])

    items = await _collect(VertexAIAdapter()._stream_with_usage(
        client=client, model=MODEL, messages=MESSAGES, tools=TOOLS,
    ))

    assert [i for i in items if isinstance(i, str)] == ["Checking."]
    terminal = items[-1]
    assert isinstance(terminal, LLMResponse)
    assert terminal.content == "Checking."
    assert [(c.name, c.arguments) for c in terminal.tool_calls] == [
        ("get_weather", {"location": "Tokyo"}),
    ]


@pytest.mark.asyncio
async def test_the_text_only_stream_logs_a_call_it_cannot_carry(caplog):
    client = _client(chunks=[
        _chunk([_call("get_weather", {"location": "Tokyo"})], finish_reason="STOP"),
    ])

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.vertex_adapter"):
        streamed = await _collect(VertexAIAdapter().get_streaming_response(
            client=client, model=MODEL, messages=MESSAGES, tools=TOOLS,
        ))

    assert streamed == []
    dropped = [r for r in caplog.records if "text-only stream" in r.getMessage()]
    assert len(dropped) == 1
    assert "get_weather" in dropped[0].getMessage()


# ---------------------------------------------------------------------------
# The real google-genai SDK, in Vertex mode, over a mock HTTP transport
# ---------------------------------------------------------------------------


def _real_vertex_client(handler):
    genai = pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    genai_types = pytest.importorskip("google.genai.types")
    oauth2 = pytest.importorskip("google.oauth2.credentials")
    if "httpx_async_client" not in genai_types.HttpOptions.model_fields:
        pytest.skip("installed google-genai cannot take an httpx client")
    return genai.Client(
        vertexai=True,
        project="test-project",
        location="us-central1",
        # A static token: nothing refreshes it, so nothing leaves the process.
        credentials=oauth2.Credentials(token="test-token"),
        http_options=genai_types.HttpOptions(
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
    )


_WIRE_DECLARATIONS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "location": {"type": "STRING", "description": "City name"},
            },
            "required": ["location"],
        },
    },
    {
        "name": "list_peers",
        "description": "List known peers",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
]
_CALL_CANDIDATE = {
    "content": {"role": "model", "parts": [
        # Gemini 2.5 sends thought signatures as parts with no text or call.
        {"thoughtSignature": "c2lnbmF0dXJl"},
        {"functionCall": {"name": "get_weather", "args": {"location": "Tokyo"}}},
    ]},
    "finishReason": "STOP",
}


@pytest.mark.asyncio
async def test_real_sdk_puts_the_declarations_on_the_vertex_wire_and_reads_the_call():
    import httpx

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"candidates": [_CALL_CANDIDATE]})

    items = await _collect(VertexAIAdapter().get_streaming_response_with_tools(
        client=_real_vertex_client(handler), model=MODEL, messages=MESSAGES,
        tools=TOOLS,
    ))

    assert len(sent) == 1
    assert sent[0].url.host == "us-central1-aiplatform.googleapis.com"
    assert sent[0].url.path.endswith(f"/publishers/google/models/{MODEL}:generateContent")
    body = json.loads(sent[0].content)
    assert body["tools"] == [{"functionDeclarations": _WIRE_DECLARATIONS}]

    assert len(items) == 1
    response = items[0]
    assert response.has_tool_calls
    assert [(c.name, c.arguments) for c in response.tool_calls] == [
        ("get_weather", {"location": "Tokyo"}),
    ]
    assert getattr(response, "stop_reason", None) == "STOP"


@pytest.mark.asyncio
async def test_real_sdk_stream_puts_the_declarations_on_the_wire_and_carries_the_call():
    import httpx

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        events = [
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "Checking."}]}}]},
            {"candidates": [_CALL_CANDIDATE]},
        ]
        sse = "".join(f"data: {json.dumps(event)}\r\n\r\n" for event in events)
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"},
        )

    items = await _collect(VertexAIAdapter()._stream_with_usage(
        client=_real_vertex_client(handler), model=MODEL, messages=MESSAGES,
        tools=TOOLS,
    ))

    assert sent[0].url.path.endswith(
        f"/publishers/google/models/{MODEL}:streamGenerateContent",
    )
    body = json.loads(sent[0].content)
    assert body["tools"] == [{"functionDeclarations": _WIRE_DECLARATIONS}]
    assert [i for i in items if isinstance(i, str)] == ["Checking."]
    assert [(c.name, c.arguments) for c in items[-1].tool_calls] == [
        ("get_weather", {"location": "Tokyo"}),
    ]


@pytest.mark.asyncio
async def test_real_sdk_request_without_tools_has_no_tools_on_the_wire():
    import httpx

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"candidates": [{
            "content": {"role": "model", "parts": [{"text": "Sunny"}]},
            "finishReason": "STOP",
        }]})

    response = await VertexAIAdapter().get_response(
        client=_real_vertex_client(handler), model=MODEL, messages=MESSAGES,
    )

    assert "tools" not in json.loads(sent[0].content)
    assert response.content == "Sunny"
    assert response.tool_calls is None
