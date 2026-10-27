"""Provider rejection text can echo private prompts; diagnostics keep its class."""
import json

import httpx
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_sdk_rejection_never_logs_private_content(provider, streaming, caplog):
    private = "private-model-screen-value-25e1"
    posts = []

    def body(request):
        posts.append(json.loads(request.content))
        assert private in request.content.decode()
        if provider == "google":
            return {"error":{"code":400,"message":private,"status":"INVALID_ARGUMENT"}}
        return {"type":"error","error":{"type":"invalid_request_error","message":private}}

    if provider == "anthropic":
        import anthropic
        import httpx2
        from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
        async def daemon(request):
            return httpx2.Response(400, json=body(request))
        client = anthropic.AsyncAnthropic(api_key="synthetic", base_url="http://synthetic.test",
            max_retries=0, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(daemon)))
        adapter, model = AnthropicAdapter(), "claude-sonnet-4-6"
        async def close():
            await client.close()
    else:
        async def daemon(request):
            return httpx.Response(400, json=body(request))
        transport = httpx.MockTransport(daemon)
        if provider == "openai":
            import openai
            from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
            client = openai.AsyncOpenAI(api_key="synthetic", base_url="http://synthetic.test/v1",
                max_retries=0, http_client=httpx.AsyncClient(transport=transport))
            adapter, model = OpenAIAdapter(), "gpt-4o"
            async def close():
                await client.close()
        else:
            from google import genai
            from kestrel_sovereign.llm.google_adapter import GoogleAdapter
            wire = httpx.AsyncClient(transport=transport)
            client = genai.Client(api_key="synthetic", http_options={"base_url":"http://synthetic.test",
                "httpx_async_client":wire})
            adapter, model = GoogleAdapter(), "gemini-2.5-flash"
            async def close():
                await client.aio.aclose()
                client.close()
                await wire.aclose()
    caplog.set_level("ERROR", logger="kestrel_sovereign.llm")
    try:
        with pytest.raises(Exception) as failure:
            if streaming:
                async for _ in adapter.get_streaming_response(client, model,
                        [{"role":"user","content":private}], max_tokens=32):
                    pass
            else:
                await adapter.get_response(client, model,
                    [{"role":"user","content":private}], max_tokens=32)
        assert len(posts) == 1
        assert private in str(failure.value), "the original SDK error is preserved for the authorized caller"
        assert private not in caplog.text
        records = [r for r in caplog.records if r.name.startswith("kestrel_sovereign.llm")]
        assert records and all(r.exc_info is None for r in records)
    finally:
        await close()
