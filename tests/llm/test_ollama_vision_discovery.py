"""Concrete vision support comes from the daemon, never a model-name guess."""
import json

import httpx
import ollama
import pytest

from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter


@pytest.mark.asyncio
async def test_real_ollama_client_uses_show_capabilities_for_vision():
    capabilities={"gemma4:31b":["completion","vision"],"qwen3-vl:235b":["completion","vision"],
                  "llama3.2:1b":["completion"],"legacy-vision-name":None}
    async def daemon(request):
        if request.url.path=="/api/tags":
            return httpx.Response(200,json={"models":[{"name":name,"model":name,"size":1,
                "modified_at":"2026-10-26T00:00:00Z","digest":"a"*64} for name in capabilities]})
        assert request.url.path=="/api/show"
        name=json.loads(request.content)["model"]
        return httpx.Response(200,json={"model_info":{},**({"capabilities":capabilities[name]} if capabilities[name] is not None else {})})
    client=ollama.AsyncClient(host="http://synthetic.test",transport=httpx.MockTransport(daemon))
    adapter=OllamaAdapter()
    try:
        models=await adapter.list_models(client)
        assert {model.id:model.supports_vision for model in models} == {
            "gemma4:31b":True,"qwen3-vl:235b":True,"llama3.2:1b":False,"legacy-vision-name":False}
    finally:
        await client._client.aclose()
        await adapter.aclose()
