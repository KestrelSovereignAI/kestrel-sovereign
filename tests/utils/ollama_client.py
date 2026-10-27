"""Actual Ollama SDK with a test-owned bounded NDJSON response and close."""

from contextlib import asynccontextmanager
import json

import httpx
import ollama

from kestrel_sovereign.execution_custody import owned_native_http_client


@asynccontextmanager
async def ollama_client(response, *, requests):
    async def handle(request):
        assert request.method == "POST" and request.url.path == "/api/chat"
        body = json.loads(request.content)
        requests.append(body)
        assert body["stream"] is True
        return httpx.Response(
            200, content=(json.dumps(response) + "\n").encode(),
            headers={"Content-Type": "application/x-ndjson"},
        )

    client = ollama.AsyncClient(
        host="http://ollama.test", transport=httpx.MockTransport(handle),
        trust_env=False,
    )
    async with owned_native_http_client(client._client):
        yield client
    assert client._client.is_closed
