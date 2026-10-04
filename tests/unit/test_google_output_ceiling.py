"""#3355: a Gemini turn's output ceiling is the model's own, and a response cut
at it is never presented as a finished one.

``GoogleAdapter`` used to send ``max_output_tokens: 8192`` whenever a caller
gave no budget (no generation caller does), and nothing read
``finish_reason``, so every Gemini turn was capped at 8,192 tokens and the cut
was invisible. The shape of #3300, on another adapter. These tests pin:

* the ceiling is the model's ``outputTokenLimit`` (from discovery or one
  ``models.get``), an explicit ``max_tokens`` still wins, and an unknown
  ceiling is a named error, never a guess;
* ``finish_reason`` is carried on the response on every path, and a
  ``MAX_TOKENS`` cut at the model's ceiling is marked in the text, while a
  cut function call is dropped rather than executed.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.google_adapter import GoogleAdapter
from kestrel_sovereign.llm.output_ceiling import (
    OutputCeilingUnknownError,
    output_ceiling_notice,
    response_stop_reason,
)

MESSAGES = [{"role": "user", "parts": [{"text": "Answer the four design questions."}]}]
TOOLS = [{"type": "function", "function": {"name": "shell", "parameters": {}}}]
CEILING = 65_536
NOTICE = output_ceiling_notice(ceiling=CEILING, stop_reason="MAX_TOKENS")


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, function_call=None)


def _call(name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(text=None, function_call=SimpleNamespace(name=name, args=args))


def _response(parts: List[Any], *, finish_reason: Any) -> SimpleNamespace:
    return SimpleNamespace(candidates=[SimpleNamespace(
        content=SimpleNamespace(parts=parts), finish_reason=finish_reason,
    )])


def _client(response: Any = None, *, output_token_limit: Any = CEILING, chunks=None):
    async def _stream(**_kwargs):
        async def _iterate():
            for chunk in chunks or []:
                yield chunk
        return _iterate()

    models = SimpleNamespace(
        generate_content=AsyncMock(return_value=response),
        generate_content_stream=AsyncMock(side_effect=_stream),
        get=AsyncMock(return_value=SimpleNamespace(
            name="models/gemini-2.5-pro", output_token_limit=output_token_limit,
        )),
    )
    return SimpleNamespace(aio=SimpleNamespace(models=models))


def _sent_config(client) -> dict:
    return client.aio.models.generate_content.call_args.kwargs["config"]


async def _collect(stream) -> List[Any]:
    return [item async for item in stream]


# ---------------------------------------------------------------------------
# The ceiling: sourced from the model, never a literal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_ceiling_is_the_models_output_token_limit_not_8192():
    """The issue's pin: a request without a caller budget sends what Gemini
    reports for the model, not 8192."""
    client = _client(_response([_text("ok")], finish_reason="STOP"))
    await GoogleAdapter().get_response(
        client=client, model="gemini-2.5-pro", messages=MESSAGES,
    )
    assert _sent_config(client)["max_output_tokens"] == CEILING
    assert _sent_config(client)["max_output_tokens"] != 8192
    client.aio.models.get.assert_awaited_once_with(model="gemini-2.5-pro")


@pytest.mark.asyncio
async def test_each_model_gets_its_own_ceiling_and_is_looked_up_once():
    ceilings = {"gemini-2.5-pro": 65_536, "gemini-2.0-flash": 8_192}
    client = _client(_response([_text("ok")], finish_reason="STOP"))
    client.aio.models.get = AsyncMock(side_effect=lambda *, model: SimpleNamespace(
        output_token_limit=ceilings[model],
    ))
    adapter = GoogleAdapter()

    sent = []
    for model in ("gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.5-pro"):
        await adapter.get_response(client=client, model=model, messages=MESSAGES)
        sent.append(_sent_config(client)["max_output_tokens"])

    assert sent == [65_536, 8_192, 65_536]
    assert [c.kwargs["model"] for c in client.aio.models.get.await_args_list] == [
        "gemini-2.5-pro", "gemini-2.0-flash",
    ]


@pytest.mark.asyncio
async def test_discovered_ceiling_is_used_without_another_lookup(monkeypatch):
    """Discovery already carries each model's ``output_token_limit``; it lands
    on ``ModelInfo.output_limit`` and a request uses it rather than asking."""
    listed = [
        SimpleNamespace(
            name="models/gemini-2.5-pro", display_name="Gemini 2.5 Pro",
            description=None, input_token_limit=1_048_576, output_token_limit=65_536,
        ),
        SimpleNamespace(
            name="models/gemini-2.0-flash", display_name="Gemini 2.0 Flash",
            description=None, input_token_limit=1_048_576, output_token_limit=8_192,
        ),
    ]
    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = lambda **_: None
    fake_genai.list_models = lambda: iter(listed)
    # ``import google.generativeai as genai`` reads the package attribute
    # before sys.modules, so replace both.
    import google

    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)
    monkeypatch.setattr(google, "generativeai", fake_genai, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    adapter = GoogleAdapter()
    models = await adapter.list_models()
    client = _client(_response([_text("ok")], finish_reason="STOP"))
    await adapter.get_response(client=client, model="gemini-2.0-flash", messages=MESSAGES)

    assert {m.id: (m.context_limit, m.output_limit) for m in models} == {
        "gemini-2.5-pro": (1_048_576, 65_536),
        "gemini-2.0-flash": (1_048_576, 8_192),
    }
    assert _sent_config(client)["max_output_tokens"] == 8_192
    client.aio.models.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_max_tokens_wins_and_skips_the_lookup():
    client = _client(_response([_text("ok")], finish_reason="STOP"))
    await GoogleAdapter().get_response(
        client=client, model="gemini-2.5-pro", messages=MESSAGES, max_tokens=300,
    )
    assert _sent_config(client)["max_output_tokens"] == 300
    client.aio.models.get.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reported", [None, 0])
async def test_unknown_ceiling_is_a_named_error_not_a_guess(reported):
    client = _client(
        _response([_text("ok")], finish_reason="STOP"), output_token_limit=reported,
    )
    with pytest.raises(OutputCeilingUnknownError) as excinfo:
        await GoogleAdapter().get_response(
            client=client, model="gemini-2.5-pro", messages=MESSAGES,
        )
    assert excinfo.value.model == "gemini-2.5-pro"
    assert "gemini-2.5-pro" in str(excinfo.value)
    client.aio.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_ceiling_fails_the_streaming_paths_too():
    client = _client(output_token_limit=None)
    adapter = GoogleAdapter()
    with pytest.raises(OutputCeilingUnknownError):
        await _collect(adapter.get_streaming_response(
            client=client, model="gemini-2.5-pro", messages=MESSAGES,
        ))
    with pytest.raises(OutputCeilingUnknownError):
        await _collect(adapter.get_streaming_response_with_tools(
            client=client, model="gemini-2.5-pro", messages=MESSAGES,
        ))
    client.aio.models.generate_content_stream.assert_not_called()


# ---------------------------------------------------------------------------
# The cut: read, recorded and visible
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_tokens_cut_at_the_models_ceiling_is_marked_incomplete():
    client = _client(_response([_text("Partial answer")], finish_reason="MAX_TOKENS"))
    response = await GoogleAdapter().get_response(
        client=client, model="gemini-2.5-pro", messages=MESSAGES,
    )
    assert response.content == f"Partial answer\n\n{NOTICE}"
    assert response_stop_reason(response) == "MAX_TOKENS"


@pytest.mark.asyncio
async def test_cut_with_no_text_is_still_visible():
    """A MAX_TOKENS candidate with no usable parts (#2129) is not an empty
    finished answer: the notice is the content."""
    client = _client(SimpleNamespace(candidates=[SimpleNamespace(
        content=SimpleNamespace(parts=None), finish_reason="MAX_TOKENS",
    )]))
    response = await GoogleAdapter().get_response(
        client=client, model="gemini-2.5-pro", messages=MESSAGES,
    )
    assert response.content == NOTICE


@pytest.mark.asyncio
async def test_cut_at_the_callers_budget_keeps_its_text():
    client = _client(_response([_text("Short")], finish_reason="MAX_TOKENS"))
    response = await GoogleAdapter().get_response(
        client=client, model="gemini-2.5-pro", messages=MESSAGES, max_tokens=50,
    )
    assert response.content == "Short"
    assert response_stop_reason(response) == "MAX_TOKENS"


@pytest.mark.asyncio
async def test_finished_response_is_untouched_and_records_its_reason():
    client = _client(_response([_text("Done.")], finish_reason="STOP"))
    response = await GoogleAdapter().get_response(
        client=client, model="gemini-2.5-pro", messages=MESSAGES,
    )
    assert response.content == "Done."
    assert response_stop_reason(response) == "STOP"


@pytest.mark.asyncio
async def test_cut_trailing_function_call_is_dropped_earlier_calls_kept():
    """A MAX_TOKENS stop falls in the last part; a function call there was cut
    mid-generation and must not be executed. A call before it is whole."""
    client = _client(_response(
        [_text("Checking."), _call("read", {"path": "a.txt"}),
         _call("shell", {"command": "rm -"})],
        finish_reason="MAX_TOKENS",
    ))
    response = await GoogleAdapter().get_response(
        client=client, model="gemini-2.5-pro", messages=MESSAGES, tools=TOOLS,
    )
    assert [tc.name for tc in response.tool_calls] == ["read"]
    assert response.content == f"Checking.\n\n{NOTICE}"


@pytest.mark.asyncio
async def test_trailing_function_call_of_a_finished_response_is_kept():
    client = _client(_response([_call("shell", {"command": "ls"})], finish_reason="STOP"))
    response = await GoogleAdapter().get_response(
        client=client, model="gemini-2.5-pro", messages=MESSAGES, tools=TOOLS,
    )
    assert [tc.name for tc in response.tool_calls] == ["shell"]


def _chunk(text: str, finish_reason: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        usage_metadata=None,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
    )


@pytest.mark.asyncio
async def test_streamed_cut_ends_with_the_notice_and_terminal_mirrors_it():
    client = _client(chunks=[_chunk("Partial "), _chunk("answer", "MAX_TOKENS")])
    items = await _collect(GoogleAdapter().get_streaming_response_with_tools(
        client=client, model="gemini-2.5-pro", messages=MESSAGES,
    ))
    streamed = "".join(i for i in items if isinstance(i, str))
    assert streamed == f"Partial answer\n\n{NOTICE}"
    terminal = items[-1]
    assert isinstance(terminal, LLMResponse)
    assert terminal.content == streamed
    assert response_stop_reason(terminal) == "MAX_TOKENS"
    stream_config = client.aio.models.generate_content_stream.call_args.kwargs["config"]
    assert stream_config["max_output_tokens"] == CEILING


@pytest.mark.asyncio
async def test_text_only_stream_carries_the_notice_too():
    client = _client(chunks=[_chunk("Partial answer", "MAX_TOKENS")])
    items = await _collect(GoogleAdapter().get_streaming_response(
        client=client, model="gemini-2.5-pro", messages=MESSAGES,
    ))
    assert all(isinstance(i, str) for i in items)
    assert "".join(items) == f"Partial answer\n\n{NOTICE}"


@pytest.mark.asyncio
async def test_streamed_finished_response_has_no_notice():
    client = _client(chunks=[_chunk("Done."), _chunk("", "STOP")])
    items = await _collect(GoogleAdapter().get_streaming_response_with_tools(
        client=client, model="gemini-2.5-pro", messages=MESSAGES,
    ))
    assert "".join(i for i in items if isinstance(i, str)) == "Done."
    assert response_stop_reason(items[-1]) == "STOP"


@pytest.mark.asyncio
async def test_tool_fallback_stream_surfaces_the_cut():
    """With tools, Google streams through the non-streaming probe; its marked
    content and stop reason reach the stream."""
    client = _client(_response([_text("Partial answer")], finish_reason="MAX_TOKENS"))
    items = await _collect(GoogleAdapter().get_streaming_response_with_tools(
        client=client, model="gemini-2.5-pro", messages=MESSAGES, tools=TOOLS,
    ))
    assert "".join(i for i in items if isinstance(i, str)) == f"Partial answer\n\n{NOTICE}"
    assert response_stop_reason(items[-1]) == "MAX_TOKENS"


# ---------------------------------------------------------------------------
# The real google-genai SDK, over a mock HTTP transport
# ---------------------------------------------------------------------------


def _real_sdk_client(handler):
    genai = pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    genai_types = pytest.importorskip("google.genai.types")
    if "httpx_async_client" not in genai_types.HttpOptions.model_fields:
        pytest.skip("installed google-genai cannot take an httpx client")
    return genai.Client(
        api_key="test-key",
        http_options=genai_types.HttpOptions(
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
    )


_MODEL_RECORD = {
    "name": "models/gemini-2.5-pro",
    "inputTokenLimit": 1_048_576,
    "outputTokenLimit": CEILING,
}
_CUT_CANDIDATE = {
    "content": {"role": "model", "parts": [{"text": "Partial answer"}]},
    "finishReason": "MAX_TOKENS",
}


@pytest.mark.asyncio
async def test_real_sdk_sends_the_models_ceiling_and_reads_the_cut():
    """End to end through google-genai: the wire carries
    ``maxOutputTokens`` = the model's ``outputTokenLimit``, and the SDK's
    ``FinishReason.MAX_TOKENS`` is read as the cut."""
    import httpx

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=_MODEL_RECORD)
        return httpx.Response(200, json={"candidates": [_CUT_CANDIDATE]})

    response = await GoogleAdapter().get_response(
        client=_real_sdk_client(handler), model="gemini-2.5-pro", messages=MESSAGES,
    )

    assert [r.url.path for r in sent] == [
        "/v1beta/models/gemini-2.5-pro",
        "/v1beta/models/gemini-2.5-pro:generateContent",
    ]
    body = json.loads(sent[1].content)
    assert body["generationConfig"]["maxOutputTokens"] == CEILING
    assert response.content == f"Partial answer\n\n{NOTICE}"
    assert response_stop_reason(response) == "MAX_TOKENS"


@pytest.mark.asyncio
async def test_real_sdk_stream_reads_the_cut_from_the_last_chunk():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_MODEL_RECORD)
        events = [
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "Partial "}]}}]},
            {"candidates": [{**_CUT_CANDIDATE, "content": {
                "role": "model", "parts": [{"text": "answer"}],
            }}]},
        ]
        sse = "".join(f"data: {json.dumps(event)}\r\n\r\n" for event in events)
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"},
        )

    items = await _collect(GoogleAdapter().get_streaming_response_with_tools(
        client=_real_sdk_client(handler), model="gemini-2.5-pro", messages=MESSAGES,
    ))
    streamed = "".join(i for i in items if isinstance(i, str))
    assert streamed == f"Partial answer\n\n{NOTICE}"
    assert response_stop_reason(items[-1]) == "MAX_TOKENS"
