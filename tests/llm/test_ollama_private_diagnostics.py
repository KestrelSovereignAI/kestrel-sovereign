"""Real daemon error/validation paths never log private model content."""
import json

import httpx
import ollama
import pytest

from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.visual_computer.inference import VisualProposal


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["validation", "daemon", "transport"])
async def test_private_content_is_absent_from_adapter_diagnostics(caplog, streaming, failure):
    private = "private-screen-value-4c8b"

    async def daemon(request):
        body = json.loads(request.content)
        assert body["messages"][-1]["content"] == private
        if failure == "daemon":
            return httpx.Response(400, json={"error": private})
        if failure == "transport":
            raise RuntimeError(private)
        response = {"model": "synthetic-vision", "created_at": "2026-10-26T00:00:00Z",
                    "message": {"role": "assistant", "content": json.dumps({"action": {"key": private}})},
                    "done": True, "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1}
        return httpx.Response(200, content=(json.dumps(response)+"\n").encode())

    client = ollama.AsyncClient(host="http://synthetic.test", transport=httpx.MockTransport(daemon))
    adapter = OllamaAdapter()
    caplog.set_level("WARNING", logger="kestrel_sovereign.llm.ollama_adapter")
    try:
        async def invoke():
            if streaming:
                return [item async for item in adapter.get_streaming_response(client, "synthetic-vision",
                    [{"role": "user", "content": private}], response_format=VisualProposal)]
            return await adapter.get_response(client, "synthetic-vision",
                [{"role": "user", "content": private}], response_format=VisualProposal)
        if failure == "validation":
            await invoke()
        else:
            with pytest.raises(ollama.ResponseError if failure == "daemon" else RuntimeError):
                await invoke()
        records = [record for record in caplog.records if record.name == "kestrel_sovereign.llm.ollama_adapter"]
        assert records, "the content-free failure classification remains observable"
        assert private not in caplog.text
        assert all(record.exc_info is None for record in records)
    finally:
        await client._client.aclose()
        await adapter.aclose()
