"""Native vendor SDK parsers at Core's public plain/tool stream boundary."""

import json

from google import genai
from google.genai import types
import httpx2
from openai import AsyncOpenAI
import pytest

from kestrel_sdk.llm import LLMProtocolError, LLMResponse
from kestrel_sdk.testing import drain_streaming_with_tools
from kestrel_sovereign.llm.genai_custody import close_genai_client
from kestrel_sovereign.llm.google_adapter import GoogleAdapter
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter


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
    responses = []
    frames = [{"id": "owned", "object": "chat.completion.chunk", "created": 0, "model": "owned",
        "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [{"index": index,
            "id": str(index), "type": "function", "function": {"name": "write", "arguments": "{}"}}]}}]}
        for index in [2, 0]]
    frames.append({"id": "owned", "object": "chat.completion.chunk", "created": 0, "model": "owned",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}]})

    def respond(request):
        response = httpx2.Response(200, headers={"content-type": "text/event-stream"},
            content=("".join("data: " + json.dumps(frame) + "\n\n" for frame in frames) + "data: [DONE]\n\n").encode())
        responses.append(response)
        return response

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
        async with AsyncOpenAI(api_key="owned-native-key", http_client=http, max_retries=0) as client:
            result = await drain_streaming_with_tools(adapter_type().get_streaming_response_with_tools(
                client, "owned", [{"role": "user", "content": "owned"}]))
            assert [marker.index for marker in result.tool_starts] == [2, 0]
            assert [call.id for call in result.final_response.tool_calls] == ["2", "0"]
            assert responses and all(response.is_closed for response in responses)
