"""Native vendor SDK parsers at Core's public plain/tool stream boundary."""

import json
from contextlib import asynccontextmanager

from google import genai
from google.genai import types
import httpx2
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import pytest

from kestrel_sdk.llm import LLMProtocolError, LLMResponse
from kestrel_sdk.testing import drain_streaming_with_tools
from kestrel_sovereign.llm.genai_custody import close_genai_client
from kestrel_sovereign.llm.google_adapter import GoogleAdapter
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter


@asynccontextmanager
async def native_openai_client(payload, *, streaming=True):
    """One owned native SDK/transport peer for the OpenAI parser cases."""
    responses = []

    def respond(request):
        if streaming:
            response = httpx2.Response(200, headers={"content-type": "text/event-stream"},
                content=("".join("data: " + json.dumps(frame) + "\n\n" for frame in payload) + "data: [DONE]\n\n").encode())
        else:
            response = httpx2.Response(200, json=payload)
        responses.append(response)
        return response

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
        async with AsyncOpenAI(api_key="owned-native-key", http_client=http, max_retries=0) as client:
            yield client, responses


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", [OpenAIAdapter, OpenRouterAdapter])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("arguments", ["{", "[]", "null", '\"text\"'])
async def test_core_native_malformed_tool_arguments_never_complete(adapter_type, streaming, arguments):
    call = {"id": "owned-call", "type": "function", "function": {"name": "write", "arguments": arguments}}
    if streaming:
        payload = [{"id": "owned", "object": "chat.completion.chunk", "created": 0, "model": "owned",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {"tool_calls": [{"index": 0, **call}]}}]}]
    else:
        payload = {"id": "owned", "object": "chat.completion", "created": 0, "model": "owned",
            "choices": [{"index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}
    async with native_openai_client(payload, streaming=streaming) as (client, responses):
        observed = []
        with pytest.raises(LLMProtocolError):
            if streaming:
                async for item in adapter_type().get_streaming_response_with_tools(client, "owned", [{"role": "user", "content": "owned"}]):
                    observed.append(item)
            else:
                await adapter_type().get_response(client, "owned", [{"role": "user", "content": "owned"}])
        assert not any(isinstance(item, LLMResponse) for item in observed)
        assert responses and all(response.is_closed for response in responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("arguments", [None, [], 17])
async def test_core_native_anthropic_nonobject_tool_input_is_refused(streaming, arguments):
    message = {"id": "owned", "type": "message", "role": "assistant", "model": "claude-owned",
        "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 2, "output_tokens": 0}}
    frames = [{"type": "message_start", "message": message},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "owned-call", "name": "write", "input": arguments}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 3}},
        {"type": "message_stop"}]
    responses = []

    def respond(request):
        response = httpx2.Response(200, headers={"content-type": "text/event-stream"},
            content="".join("event: " + frame["type"] + "\ndata: " + json.dumps(frame) + "\n\n" for frame in frames).encode())
        responses.append(response)
        return response

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
        async with AsyncAnthropic(api_key="owned-native-key", http_client=http, max_retries=0) as client:
            with pytest.raises(LLMProtocolError):
                if streaming:
                    _ = [item async for item in AnthropicAdapter().get_streaming_response_with_tools(
                        client, "claude-owned", [{"role": "user", "content": "owned"}], max_tokens=16)]
                else:
                    await AnthropicAdapter().get_response(client, "claude-owned", [{"role": "user", "content": "owned"}], max_tokens=16)
            assert responses and all(response.is_closed for response in responses)


def google_tool_response():
    return {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"id": "owned-call", "name": "write", "args": {"value": "owned"}}}]},
        "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5}}


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["google", "vertex"])
@pytest.mark.parametrize("plain", [False, True])
async def test_native_google_function_calls_are_preserved_or_refused(backend, plain):
    requests, responses = [], []

    def respond(request):
        requests.append(request)
        native = google_tool_response()
        if ":streamGenerateContent" in request.url.path:
            response = httpx2.Response(200, headers={"content-type": "text/event-stream"},
                content=("data: " + json.dumps(native) + "\n\n").encode())
        elif request.method == "GET":
            response = httpx2.Response(200, json={"name": "models/owned", "outputTokenLimit": 32})
        else:
            response = httpx2.Response(200, json=native)
        responses.append(response)
        return response

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
        client = genai.Client(api_key="owned-native-key", http_options=types.HttpOptions(httpx_async_client=http))
        adapter = GoogleAdapter() if backend == "google" else VertexAIAdapter(project_id="owned")
        try:
            messages = [{"role": "user", "content": "owned"}]
            if plain:
                observed = []
                with pytest.raises(LLMProtocolError):
                    async for item in adapter.get_streaming_response(client, "owned", messages, max_tokens=16):
                        observed.append(item)
                assert not any(isinstance(item, LLMResponse) for item in observed)
            else:
                result = await drain_streaming_with_tools(adapter.get_streaming_response_with_tools(
                    client, "owned", messages, max_tokens=16,
                    tools=[{"type": "function", "function": {"name": "write", "description": "Owned write",
                        "parameters": {"type": "object", "properties": {"value": {"type": "string"}}}}}]))
                assert result.final_response.stop_reason == "STOP"
                assert [marker.id for marker in result.tool_starts] == ["owned-call"]
                assert [call.id for call in result.final_response.tool_calls] == ["owned-call"]
            assert responses and all(response.is_closed for response in responses)
            assert sum(request.method == "POST" for request in requests) == 1
        finally:
            await close_genai_client(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", [OpenAIAdapter, OpenRouterAdapter])
async def test_native_openai_sparse_tool_calls_keep_marker_order(adapter_type):
    frames = [{"id": "owned", "object": "chat.completion.chunk", "created": 0, "model": "owned",
        "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [{"index": index,
            "id": str(index), "type": "function", "function": {"name": "write", "arguments": "{}"}}]}}]}
        for index in [2, 0]]
    frames.append({"id": "owned", "object": "chat.completion.chunk", "created": 0, "model": "owned",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}]})

    async with native_openai_client(frames) as (client, responses):
        result = await drain_streaming_with_tools(adapter_type().get_streaming_response_with_tools(
            client, "owned", [{"role": "user", "content": "owned"}]))
        assert [marker.index for marker in result.tool_starts] == [2, 0]
        assert [call.id for call in result.final_response.tool_calls] == ["2", "0"]
        assert responses and all(response.is_closed for response in responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", [OpenAIAdapter, OpenRouterAdapter])
@pytest.mark.parametrize("second_id,second_name", [("owned", "write"), ("other", "write"), ("owned", "other")])
async def test_core_native_repeated_tool_identity_is_preserved_or_refused(adapter_type, second_id, second_name):
    frames = [{"id": "owned", "object": "chat.completion.chunk", "created": 0, "model": "owned",
        "choices": [{"index": 0, "finish_reason": reason, "delta": {"tool_calls": [{"index": 0,
            "id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}]}}]}
        for call_id, name, arguments, reason in [("owned", "write", '{"value":', None),
            (second_id, second_name, '\"owned\"}', "tool_calls")]]
    async with native_openai_client(frames) as (client, responses):
        stream = adapter_type().get_streaming_response_with_tools(client, "owned", [{"role": "user", "content": "owned"}])
        if (second_id, second_name) == ("owned", "write"):
            result = await drain_streaming_with_tools(stream)
            call, = result.final_response.tool_calls
            assert (call.id, call.name, call.arguments) == ("owned", "write", {"value": "owned"})
        else:
            with pytest.raises(LLMProtocolError):
                _ = [item async for item in stream]
        assert responses and all(response.is_closed for response in responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", [OpenAIAdapter, OpenRouterAdapter])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("indices", [[1], [0, 1]])
async def test_core_native_unselected_choices_are_refused(adapter_type, streaming, indices):
    choices = [{"index": index, "finish_reason": "stop",
        ("delta" if streaming else "message"): {"role": "assistant", "content": "unselected"}} for index in indices]
    payload = {"id": "owned", "object": "chat.completion.chunk" if streaming else "chat.completion",
        "created": 0, "model": "owned", "choices": choices}
    async with native_openai_client([payload] if streaming else payload, streaming=streaming) as (client, responses):
        with pytest.raises(LLMProtocolError):
            if streaming:
                _ = [item async for item in adapter_type().get_streaming_response_with_tools(client, "owned", [{"role": "user", "content": "owned"}])]
            else:
                await adapter_type().get_response(client, "owned", [{"role": "user", "content": "owned"}])
        assert responses and all(response.is_closed for response in responses)
