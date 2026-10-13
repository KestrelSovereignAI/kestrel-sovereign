"""#3623: Gemini's thinking tokens are output tokens.

Gemini 2.5 thinks by default, counts its thinking against
``maxOutputTokens`` and bills it as output. Gemini reports it apart from the
text, as ``thoughtsTokenCount``, and ``output_tokens`` used to be
``candidatesTokenCount`` alone. A ``max_tokens=1000`` story billed 996 output
tokens and reported 40; one whose budget went entirely to thinking reported
none. ``total_tokens`` then exceeded ``input_tokens + output_tokens``, and the
metering callback and the Prometheus output series under-counted every Gemini
call by its thinking.

The bodies below are live ``gemini-2.5-flash`` answers (2026-10-13) unless
marked, trimmed to the fields the adapters read, run through the real
google-genai SDK over a mock transport for both Gemini adapters. They pin:

* ``output_tokens`` is the candidates count plus the thoughts count, also when
  Gemini reports no candidates count at all;
* ``total_tokens == input_tokens + output_tokens`` whenever all three are
  reported;
* a model that reports no thoughts count keeps its candidates count;
* a stream reports the same, from its last chunk: Gemini repeats the
  cumulative counts on every chunk, so no chunk's count is added to another's;
* the metering callback bills the thinking as ``completion_tokens``, and the
  Prometheus output series counts it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.google_adapter import _normalized_google_genai_usage
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
from kestrel_sovereign.llm.output_ceiling import response_stop_reason
from tests.unit.test_gemini_contents import MODEL, ROUTE_IDS, ROUTES
from tests.unit.test_gemini_max_tokens import CUT_AFTER_TEXT, THINKING_TOOK_THE_BUDGET

STORY = [{"role": "user", "content": "Tell me a very long story about a dragon."}]
HELLO = [{"role": "user", "content": "Say hello in one short sentence."}]
COUNTING = [{
    "role": "user",
    "content": "Write the numbers from 1 to 120 as words, separated by commas.",
}]


def _body(
    parts: Optional[List[Dict[str, Any]]],
    finish_reason: Optional[str],
    usage: Dict[str, int],
) -> Dict[str, Any]:
    """A ``generateContent`` answer, or one event of a stream. A stream's
    chunks before its last have no finish reason."""
    content: Dict[str, Any] = {"role": "model"}
    if parts is not None:
        content["parts"] = parts
    candidate: Dict[str, Any] = {"content": content, "index": 0}
    if finish_reason is not None:
        candidate["finishReason"] = finish_reason
    return {
        "candidates": [candidate],
        "usageMetadata": usage,
        "modelVersion": MODEL,
    }


#: ``max_tokens=50``, measured in #3623: thinking took the whole budget, so
#: there is no text and no candidates count.
BUDGET_50_ALL_THINKING = _body(None, "MAX_TOKENS", {
    "promptTokenCount": 11, "totalTokenCount": 57, "thoughtsTokenCount": 46,
})
#: ``max_tokens=200``, streamed: six tokens of text survived 190 of thinking.
BUDGET_200 = _body([{"text": "The Scales of Aerion,"}], "MAX_TOKENS", {
    "promptTokenCount": 11, "candidatesTokenCount": 6,
    "totalTokenCount": 207, "thoughtsTokenCount": 190,
})
#: A finished answer: 21 tokens of thinking for 2 of text.
SAID_HELLO = _body([{"text": "Hello!"}], "STOP", {
    "promptTokenCount": 8, "candidatesTokenCount": 2,
    "totalTokenCount": 31, "thoughtsTokenCount": 21,
})
#: The same answer as a model that does not think reports it: no thoughts
#: count. Not recorded; ``SAID_HELLO`` without its thinking.
SAID_HELLO_WITHOUT_THINKING = _body([{"text": "Hello!"}], "STOP", {
    "promptTokenCount": 8, "candidatesTokenCount": 2, "totalTokenCount": 10,
})

#: Each body, the question and ``max_tokens`` it answered, and what the
#: adapters must report: content, stop reason, then input, output and total
#: tokens.
CASES = {
    "budget-50-all-thinking": (
        BUDGET_50_ALL_THINKING, STORY, 50, (None, "MAX_TOKENS", 11, 46, 57),
    ),
    "budget-200": (
        BUDGET_200, STORY, 200,
        ("The Scales of Aerion,", "MAX_TOKENS", 11, 196, 207),
    ),
    "said-hello": (SAID_HELLO, HELLO, None, ("Hello!", "STOP", 8, 23, 31)),
    "no-thinking": (
        SAID_HELLO_WITHOUT_THINKING, HELLO, None, ("Hello!", "STOP", 8, 2, 10),
    ),
    # #3621's recorded ``max_tokens=50`` bodies.
    "3621-thinking-took-the-budget": (
        THINKING_TOOK_THE_BUDGET, STORY, 50, (None, "MAX_TOKENS", 11, 47, 58),
    ),
    "3621-cut-after-text": (
        CUT_AFTER_TEXT, STORY, 50, ("The sun", "MAX_TOKENS", 11, 46, 57),
    ),
}

#: A streamed answer to ``COUNTING``: three of its nine chunks (the first two and the last). Every
#: chunk carries the counts so far; the thoughts count is the same on each.
COUNTING_STREAM = [
    _body(
        [{"text": "one, two, three, four, five, six, seven, eight, nine, ten, "
                  "eleven, twelve, thirteen, fourteen, fifteen, sixteen, "
                  "seventeen, eighteen, nineteen, twenty, twenty-one, twenty"}],
        None,
        {"promptTokenCount": 19, "candidatesTokenCount": 45,
         "totalTokenCount": 1706, "thoughtsTokenCount": 1642},
    ),
    _body(
        [{"text": "-two, twenty-three, twenty-four, twenty-five, twenty-six, "
                  "twenty-seven, twenty-eight, twenty-nine, thirty, thirty-one, "
                  "thirty-two, thirty-three, thirty-four"}],
        None,
        {"promptTokenCount": 19, "candidatesTokenCount": 93,
         "totalTokenCount": 1754, "thoughtsTokenCount": 1642},
    ),
    _body(
        [{"text": " ten, one hundred eleven, one hundred twelve, one hundred "
                  "thirteen, one hundred fourteen, one hundred fifteen, one "
                  "hundred sixteen, one hundred seventeen, one hundred "
                  "eighteen, one hundred nineteen, one hundred twenty"}],
        "STOP",
        {"promptTokenCount": 19, "candidatesTokenCount": 424,
         "totalTokenCount": 2085, "thoughtsTokenCount": 1642},
    ),
]


class _Gemini:
    """A mock Gemini endpoint. Every generation answers ``bodies``: the one
    body, or each in turn as one event of a stream. Records every request."""

    def __init__(self, *bodies: Dict[str, Any]) -> None:
        self.bodies = bodies
        self.requests: List[Any] = []

    def __call__(self, request):
        import httpx

        self.requests.append(request)
        if request.method == "GET":
            # GoogleAdapter looks up the model's output ceiling (#3355).
            return httpx.Response(200, json={
                "name": f"models/{MODEL}", "outputTokenLimit": 65_536,
            })
        if request.url.path.endswith(":streamGenerateContent"):
            sse = "".join(f"data: {json.dumps(body)}\r\n\r\n" for body in self.bodies)
            return httpx.Response(
                200, content=sse.encode(), headers={"content-type": "text/event-stream"},
            )
        [body] = self.bodies
        return httpx.Response(200, json=body)

    def generations(self) -> List[str]:
        """The generation endpoints called, by their method suffix."""
        return [
            r.url.path.rsplit(":", 1)[1] for r in self.requests
            if r.method == "POST"
        ]

    def max_output_tokens(self) -> Optional[int]:
        [body] = [json.loads(r.content) for r in self.requests if r.method == "POST"]
        return body["generationConfig"].get("maxOutputTokens")


async def _respond(
    route, wire: _Gemini, messages, path: str, max_tokens: Optional[int] = None,
) -> LLMResponse:
    """What ``route`` makes of ``wire``'s answer. A recorded cut is replayed
    at the budget it was cut at: a caller's budget, not the model's ceiling,
    which the Google route would mark as such."""
    adapter = route.adapter()
    client = route.client(wire)
    budget = {} if max_tokens is None else {"max_tokens": max_tokens}
    if path == "generate":
        response = await adapter.get_response(
            client=client, model=MODEL, messages=messages, **budget,
        )
        assert wire.generations() == ["generateContent"]
    else:
        # The agent's streamed entry point; with no tools it streams.
        items = [item async for item in adapter.get_streaming_response_with_tools(
            client=client, model=MODEL, messages=messages, **budget,
        )]
        *chunks, response = items
        assert isinstance(response, LLMResponse)
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert wire.generations() == ["streamGenerateContent"]
    if max_tokens is not None:
        assert wire.max_output_tokens() == max_tokens
    return response


def _usage(response: LLMResponse) -> tuple:
    return response.input_tokens, response.output_tokens, response.total_tokens


# ---------------------------------------------------------------------------
# The adapters, over the real SDK
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["generate", "stream"])
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_thinking_is_counted_as_output(route, case, path):
    body, messages, max_tokens, expected = CASES[case]
    content, stop_reason, input_tokens, output_tokens, total_tokens = expected

    response = await _respond(route, _Gemini(body), messages, path, max_tokens)

    assert response.content == content
    assert response_stop_reason(response) == stop_reason
    assert _usage(response) == (input_tokens, output_tokens, total_tokens)
    assert response.total_tokens == response.input_tokens + response.output_tokens
    assert response.cache_read_input_tokens is None


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_a_stream_reports_the_thinking_of_its_last_chunk_once(route):
    wire = _Gemini(*COUNTING_STREAM)

    response = await _respond(route, wire, COUNTING, "stream")

    assert response.content == "".join(
        chunk["candidates"][0]["content"]["parts"][0]["text"]
        for chunk in COUNTING_STREAM
    )
    assert response_stop_reason(response) == "STOP"
    # 424 text + 1642 thinking: the last chunk's counts, not the sum of every
    # chunk's thinking.
    assert _usage(response) == (19, 2066, 2085)


# ---------------------------------------------------------------------------
# The normalization, on the counts #3623 measured
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("usage, expected", [
    # ``max_tokens`` 50, 200 and 1000 on the dragon story.
    ({"prompt_token_count": 11, "thoughts_token_count": 46,
      "total_token_count": 57}, (11, 46, 57, None)),
    ({"prompt_token_count": 11, "candidates_token_count": 8,
      "thoughts_token_count": 188, "total_token_count": 207}, (11, 196, 207, None)),
    ({"prompt_token_count": 11, "candidates_token_count": 40,
      "thoughts_token_count": 956, "total_token_count": 1007}, (11, 996, 1007, None)),
    # Neither output count: output stays unreported.
    ({"prompt_token_count": 11, "total_token_count": 11}, (11, None, 11, None)),
    # No total reported: it is input plus output, thinking included.
    ({"prompt_token_count": 11, "candidates_token_count": 8,
      "thoughts_token_count": 188}, (11, 196, 207, None)),
], ids=["budget-50", "budget-200", "budget-1000", "no-output", "no-total"])
def test_measured_counts_normalize_with_thinking_as_output(usage, expected):
    assert _normalized_google_genai_usage(usage) == expected
    assert _normalized_google_genai_usage(SimpleNamespace(**usage)) == expected


def test_cached_prompt_tokens_leave_thinking_in_output():
    """Cached prompt tokens leave input and total; output keeps the thinking,
    and the total is still input plus output."""
    assert _normalized_google_genai_usage(SimpleNamespace(
        prompt_token_count=11,
        cached_content_token_count=7,
        candidates_token_count=4,
        thoughts_token_count=10,
        total_token_count=25,
    )) == (4, 14, 18, 7)


def test_tool_use_prompt_tokens_stay_in_the_total_only():
    """Gemini's total also counts the results of its own built-in tools
    (``tool_use_prompt_token_count``), which neither input nor output carries.
    Kestrel declares none of those tools; when Gemini reports one, the total
    exceeds input plus output by that count rather than being forced."""
    assert _normalized_google_genai_usage(SimpleNamespace(
        prompt_token_count=11,
        tool_use_prompt_token_count=5,
        candidates_token_count=4,
        thoughts_token_count=10,
        total_token_count=30,
    )) == (11, 14, 30, None)


# ---------------------------------------------------------------------------
# Billing: the metering callback and the Prometheus output series
# ---------------------------------------------------------------------------

#: A billable caller: the metering callback bills only an attributed call.
CONTEXT = LLMInvocationContext(companion_id="companion-1", user_id="user-1")


class _Metric:
    """Records what a Prometheus series was given."""

    def __init__(self, sink: List[tuple]) -> None:
        self._sink = sink

    def labels(self, **labels: str):
        sink = self._sink

        class _Child:
            def inc(self, value: float = 1) -> None:
                sink.append((labels, value))

            def observe(self, value: float) -> None:
                sink.append((labels, value))

        return _Child()


@pytest.fixture
def prometheus_tokens(monkeypatch) -> List[tuple]:
    tokens: List[tuple] = []
    monkeypatch.setattr("kestrel_sdk.metrics.PROMETHEUS_AVAILABLE", True)
    monkeypatch.setattr("kestrel_sdk.metrics.LLM_CALLS", _Metric([]))
    monkeypatch.setattr("kestrel_sdk.metrics.LLM_DURATION", _Metric([]))
    monkeypatch.setattr("kestrel_sdk.metrics.LLM_TOKENS", _Metric(tokens))
    return tokens


def _metered_service(monkeypatch, route, wire: _Gemini):
    """A real ``LLMService`` whose only route is a real Gemini route, built as
    the registry builds one, with the real SDK over ``wire``; its metering
    callback, with the original signature, records what it is given."""
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry
    from kestrel_sovereign.llm.service import LLMService

    llm = LLMService()
    adapter_name = "VertexAIAdapter" if route.name == "vertex" else "GoogleAdapter"
    info = ProviderRegistry({})._build_route(
        "google", route.name, {"is_cloud": True},
        {"adapter": adapter_name, "api_key": "test-key", "model": MODEL},
    )
    [provider] = llm._convert_providers_format([info])
    provider["client"] = route.client(wire)
    monkeypatch.setattr(llm, "providers", [provider])
    monkeypatch.setattr(llm, "config", {"route_priority": [provider["name"]]})
    monkeypatch.setattr(llm, "_mandate_preference", {})
    monkeypatch.setattr(llm, "_ensure_models_discovered", AsyncMock())
    monkeypatch.setattr(llm, "_record_model_usage", AsyncMock())
    llm.metered = []

    async def meter(
        *, companion_id, user_id, provider, model, prompt_tokens,
        completion_tokens,
    ) -> None:
        llm.metered.append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    llm.set_metering_callback(meter)
    return llm


async def _call(llm, entry: str, messages) -> None:
    if entry == "generate_with_messages":
        await llm.generate_with_messages(
            messages=messages, invocation_context=CONTEXT,
        )
        return
    async for _ in llm.stream_with_tool_detection(
        messages=messages, invocation_context=CONTEXT,
    ):
        pass


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize("entry", ["generate_with_messages", "stream_with_tool_detection"])
@pytest.mark.parametrize("case", ["said-hello", "budget-200"])
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_the_thinking_is_billed_as_completion_tokens(
    monkeypatch, prometheus_tokens, route, case, entry,
):
    """A finished answer, and one cut at its budget: the generation gate
    refuses the cut one, and its provider still billed it (#3576)."""
    body, messages, _, expected = CASES[case]
    _, stop_reason, input_tokens, output_tokens, total_tokens = expected
    wire = _Gemini(body)
    llm = _metered_service(monkeypatch, route, wire)

    if stop_reason == "STOP":
        await _call(llm, entry, messages)
    else:
        with pytest.raises(Exception):
            await _call(llm, entry, messages)

    assert wire.generations() == [
        "generateContent" if entry == "generate_with_messages"
        else "streamGenerateContent"
    ]
    assert llm.metered == [{
        "prompt_tokens": input_tokens, "completion_tokens": output_tokens,
    }]
    assert prometheus_tokens == [
        ({"model": MODEL, "direction": "input"}, input_tokens),
        ({"model": MODEL, "direction": "output"}, output_tokens),
    ]
    llm._record_model_usage.assert_awaited_once()
    assert llm._record_model_usage.await_args.kwargs["tokens"] == total_tokens
