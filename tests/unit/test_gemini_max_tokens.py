"""#3621: a caller's ``max_tokens`` on a Gemini model that thinks.

Gemini 2.5 thinks by default and counts its thinking against
``maxOutputTokens``. A small budget can go entirely to thinking: the candidate
then has no ``parts``, only ``finishReason: MAX_TOKENS``. Whether any text
survives varies from call to call. The live
``tests/llm/test_vertex_adapter.py::TestGeneration::test_generation_with_max_tokens``
asserted text and failed whenever none survived. CI skips that test, as it
needs credentials.

The bodies below are what a live ``gemini-2.5-flash`` call returned for
"Tell me a very long story about a dragon." at ``max_tokens=50`` (2026-10-13),
trimmed to the fields the adapters read. Over the real google-genai SDK and a
mock transport, for both Gemini adapters, on the plain and the streamed
request, these pin:

* the caller's budget reaches the wire unchanged, as
  ``generationConfig.maxOutputTokens``;
* a response whose budget went to thinking has no content (``None``, not an
  empty or placeholder string) and the stop reason ``MAX_TOKENS``;
* a response cut after some text keeps that text and the same stop reason;
* either way the generation gate refuses it as an answer
  (``OutputCapReachedError``), so it is reported as cut, never as finished.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.generation_gate import judge_generation
from kestrel_sovereign.llm.output_ceiling import (
    OutputCapReachedError,
    response_stop_reason,
)
from tests.unit.test_gemini_contents import MODEL, ROUTE_IDS, ROUTES

BUDGET = 50
QUESTION = [{"role": "user", "content": "Tell me a very long story about a dragon."}]

#: Thinking took the whole budget: no parts at all.
THINKING_TOOK_THE_BUDGET = {
    "candidates": [{
        "content": {"role": "model"}, "finishReason": "MAX_TOKENS", "index": 0,
    }],
    "usageMetadata": {
        "promptTokenCount": 11, "totalTokenCount": 58, "thoughtsTokenCount": 47,
    },
    "modelVersion": "gemini-2.5-flash",
}
#: Thinking took most of the budget; two tokens of text survived.
CUT_AFTER_TEXT = {
    "candidates": [{
        "content": {"role": "model", "parts": [{"text": "The sun"}]},
        "finishReason": "MAX_TOKENS",
        "index": 0,
    }],
    "usageMetadata": {
        "promptTokenCount": 11, "candidatesTokenCount": 2,
        "totalTokenCount": 57, "thoughtsTokenCount": 44,
    },
    "modelVersion": "gemini-2.5-flash",
}


class _Capped:
    """A mock Gemini endpoint that answers every generation with ``body``.
    Records every request."""

    def __init__(self, body: Dict[str, Any]) -> None:
        self.body = body
        self.requests: List[Any] = []

    def __call__(self, request):
        import httpx

        self.requests.append(request)
        if request.url.path.endswith(":streamGenerateContent"):
            sse = f"data: {json.dumps(self.body)}\r\n\r\n"
            return httpx.Response(
                200, content=sse.encode(), headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=self.body)

    def generation_config(self) -> Dict[str, Any]:
        assert len(self.requests) == 1, [str(r.url) for r in self.requests]
        [request] = self.requests
        assert request.method == "POST"
        return json.loads(request.content)["generationConfig"]


async def _respond(route, wire: _Capped, path: str) -> tuple[LLMResponse, Optional[str]]:
    """The response ``route`` makes of ``wire``'s answer, and the text it
    streamed (``None`` on the plain request, which streams nothing)."""
    adapter = route.adapter()
    client = route.client(wire)
    if path == "generate":
        response = await adapter.get_response(
            client=client, model=MODEL, messages=QUESTION, max_tokens=BUDGET,
        )
        return response, None
    # The agent's streamed entry point; with no tools it streams.
    items = [item async for item in adapter.get_streaming_response_with_tools(
        client=client, model=MODEL, messages=QUESTION, max_tokens=BUDGET,
    )]
    *chunks, terminal = items
    assert isinstance(terminal, LLMResponse)
    assert all(isinstance(chunk, str) for chunk in chunks)
    return terminal, "".join(chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["generate", "stream"])
@pytest.mark.parametrize("body, text", [
    (THINKING_TOOK_THE_BUDGET, None),
    (CUT_AFTER_TEXT, "The sun"),
], ids=["thinking-took-the-budget", "cut-after-text"])
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_a_budget_spent_on_thinking_is_reported_as_cut(route, body, text, path):
    wire = _Capped(body)

    response, streamed = await _respond(route, wire, path)

    assert wire.generation_config()["maxOutputTokens"] == BUDGET
    assert response.content == text
    if streamed is not None:
        assert streamed == (text or "")
    assert response.tool_calls is None
    assert response_stop_reason(response) == "MAX_TOKENS"
    verdict = judge_generation(response, provider=route.name, model=MODEL)
    assert isinstance(verdict, OutputCapReachedError)
