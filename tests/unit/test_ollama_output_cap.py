"""#3552: an Ollama chat call always carries an output cap, and a generation
that stops on it is a failed attempt, never an answer.

A local-only privacy mode moved a test agent onto ``llama3.2:1b``, which fell
into a repetition loop and generated 129,966 tokens over 19 minutes while the
turn held the conversation lock: the chat call set no ``num_predict``, and
Ollama applies no output limit of its own.

These tests drive the real ``ollama.AsyncClient`` over an in-process HTTP
transport standing in for the Ollama daemon, so what is asserted is the request
Ollama would receive and what the adapter does with its reply.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List

import httpx
import pytest

ollama = pytest.importorskip("ollama")

from kestrel_sovereign.llm.adapter import LLMResponse  # noqa: E402
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter  # noqa: E402
from kestrel_sovereign.llm.output_ceiling import (  # noqa: E402
    IncompleteGenerationError,
    OutputCapReachedError,
    incomplete_generation,
    response_stop_reason,
)

MODEL = "llama3.2:1b"
MESSAGES = [{"role": "user", "content": "Where do I work?"}]
TOOLS = [{
    "type": "function",
    "function": {"name": "lookup", "description": "", "parameters": {}},
}]


def _line(payload: Dict[str, Any]) -> bytes:
    return (json.dumps(payload) + "\n").encode()


def _chat_reply(content: str, *, done_reason: str = "stop", eval_count: int = 3):
    return {
        "model": MODEL,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 11,
        "eval_count": eval_count,
    }


class _Daemon:
    """A stand-in Ollama daemon that records every ``/api/chat`` request."""

    def __init__(self, respond: Callable[[Dict[str, Any]], httpx.Response]) -> None:
        self._respond = respond
        self.requests: List[Dict[str, Any]] = []

    def client(self) -> "ollama.AsyncClient":
        return ollama.AsyncClient(
            host="http://ollama.test:11434",
            transport=httpx.MockTransport(self._handle),
        )

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        return self._respond(body)

    @property
    def num_predict(self) -> Any:
        return self.requests[-1]["options"]["num_predict"]


def _replying(content: str, *, done_reason: str = "stop") -> _Daemon:
    def respond(body: Dict[str, Any]) -> httpx.Response:
        reply = _chat_reply(content, done_reason=done_reason)
        if body.get("stream"):
            return httpx.Response(200, content=_line(reply))
        return httpx.Response(200, json=reply)

    return _Daemon(respond)


async def _get_response(adapter: OllamaAdapter, daemon: _Daemon, **kwargs) -> LLMResponse:
    return await adapter.get_response(
        client=daemon.client(), model=MODEL, messages=MESSAGES, tools=TOOLS, **kwargs,
    )


async def _stream(adapter: OllamaAdapter, daemon: _Daemon, **kwargs) -> List[Any]:
    return [
        item async for item in adapter.get_streaming_response_with_tools(
            client=daemon.client(), model=MODEL, messages=MESSAGES, **kwargs,
        )
    ]


# --------------------------------------------------------------------------
# The cap is on every chat call
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_tool_bearing_chat_call_sends_the_default_cap():
    """The path the incident took: a turn always advertises tools, so the
    Ollama adapter answers it with one non-streaming ``/api/chat`` call."""
    daemon = _replying("You work at Acme.")

    response = await _get_response(OllamaAdapter(), daemon)

    assert daemon.num_predict == OllamaAdapter.DEFAULT_MAX_OUTPUT_TOKENS == 4096
    assert response.content == "You work at Acme."
    assert response_stop_reason(response) == "stop"


@pytest.mark.asyncio
async def test_a_streamed_chat_call_sends_the_default_cap():
    daemon = _replying("You work at Acme.")

    items = await _stream(OllamaAdapter(), daemon)

    assert daemon.requests[-1]["stream"] is True
    assert daemon.num_predict == 4096
    assert "".join(i for i in items if isinstance(i, str)) == "You work at Acme."
    [terminal] = [i for i in items if isinstance(i, LLMResponse)]
    assert response_stop_reason(terminal) == "stop"


@pytest.mark.asyncio
async def test_the_route_configures_its_own_cap():
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    info = ProviderRegistry({})._build_route(
        "ollama", "local", {"is_cloud": False},
        {"adapter": "OllamaAdapter", "host": "http://ollama.test:11434",
         "max_output_tokens": 256},
    )
    daemon = _replying("ok")

    await _get_response(info.adapter, daemon)

    assert daemon.num_predict == 256


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "8192"])
def test_a_cap_that_is_not_a_positive_integer_fails_the_route(value):
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    with pytest.raises(ValueError, match="max_output_tokens"):
        ProviderRegistry({})._build_route(
            "ollama", "local", {"is_cloud": False},
            {"adapter": "OllamaAdapter", "max_output_tokens": value},
        )


@pytest.mark.asyncio
async def test_a_caller_asking_for_fewer_tokens_wins():
    daemon = _replying("short")

    await _get_response(OllamaAdapter(max_output_tokens=512), daemon, max_tokens=64)

    assert daemon.num_predict == 64


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", [512, 100_000, 0, -1, None])
async def test_a_caller_cannot_lift_or_remove_the_cap(requested):
    daemon = _replying("ok")

    await _get_response(
        OllamaAdapter(max_output_tokens=512), daemon, max_tokens=requested,
    )

    assert daemon.num_predict == 512


# --------------------------------------------------------------------------
# Stopping on the cap is a failed attempt
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_response_stopped_by_the_route_cap_is_a_failure_not_an_answer():
    daemon = _replying("loop loop loop", done_reason="length")

    with pytest.raises(OutputCapReachedError) as raised:
        await _get_response(OllamaAdapter(max_output_tokens=128), daemon)

    assert isinstance(raised.value, IncompleteGenerationError)
    assert raised.value.cap == 128 and raised.value.model == MODEL
    # The error names the route's knob and carries none of the response text.
    assert "max_output_tokens" in str(raised.value)
    assert "loop" not in str(raised.value)


@pytest.mark.asyncio
async def test_the_tool_stream_path_fails_on_the_cap_before_yielding_anything():
    """With tools, the stream answers from one non-streaming probe: a capped
    probe must surface as the failure, with no partial text ahead of it."""
    daemon = _replying("loop loop loop", done_reason="length")
    seen: List[Any] = []

    with pytest.raises(OutputCapReachedError):
        async for item in OllamaAdapter().get_streaming_response_with_tools(
            client=daemon.client(), model=MODEL, messages=MESSAGES, tools=TOOLS,
        ):
            seen.append(item)

    assert seen == []


@pytest.mark.asyncio
async def test_a_stream_stopped_by_the_route_cap_is_a_failure_not_an_answer():
    daemon = _replying("loop loop loop", done_reason="length")

    with pytest.raises(OutputCapReachedError):
        await _stream(OllamaAdapter(), daemon)


@pytest.mark.asyncio
async def test_a_stop_on_a_callers_own_budget_keeps_its_text():
    """A caller that asked for fewer tokens chose where the response ends; its
    text stands, and the stop reason records that it was cut."""
    daemon = _replying("partial summary", done_reason="length")

    response = await _get_response(OllamaAdapter(), daemon, max_tokens=32)

    assert response.content == "partial summary"
    assert response_stop_reason(response) == "length"

    streamed = await _stream(OllamaAdapter(), _replying("partial", done_reason="length"),
                             max_tokens=32)
    [terminal] = [i for i in streamed if isinstance(i, LLMResponse)]
    assert response_stop_reason(terminal) == "length"


class _EndlessGeneration(httpx.AsyncByteStream):
    """An ``/api/chat`` stream that ignores ``num_predict`` and never ends."""

    def __init__(self) -> None:
        self.sent = 0
        self.closed = False

    async def __aiter__(self):
        while True:
            self.sent += 1
            await asyncio.sleep(0)  # a real daemon's chunks arrive over I/O
            yield _line({
                "model": MODEL,
                "message": {"role": "assistant", "content": "loop "},
                "done": False,
            })

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_a_server_that_streams_forever_is_cut_at_the_cap_and_disconnected():
    """The runaway itself: a stream that would never stop on its own. The
    adapter stops it at the cap it sent, fails the attempt, and closes the
    HTTP response, which is what makes Ollama stop generating."""
    endless = _EndlessGeneration()
    daemon = _Daemon(lambda _body: httpx.Response(200, stream=endless))
    text: List[str] = []

    async def consume() -> None:
        async for item in OllamaAdapter(max_output_tokens=32).get_streaming_response_with_tools(
            client=daemon.client(), model=MODEL, messages=MESSAGES,
        ):
            if isinstance(item, str):
                text.append(item)

    with pytest.raises(OutputCapReachedError) as raised:
        await asyncio.wait_for(consume(), 10)

    assert daemon.num_predict == 32
    assert raised.value.cap == 32
    assert endless.closed, "the HTTP response to Ollama was left open"
    # It stopped one chunk past the cap rather than reading on.
    assert endless.sent <= 34
    assert len(text) <= 32


def test_incomplete_generation_is_found_through_the_service_wrappers():
    """The service reports a selected route's failure wrapped; the transports
    must still recognize the cap underneath to say what happened."""
    from kestrel_sovereign.llm.service import LLMServiceError
    from kestrel_sovereign.llm.streaming import LLMStreamingError

    cap = OutputCapReachedError(provider="ollama", model=MODEL, cap=4096)
    streamed = LLMStreamingError("Selected route ollama:local failed", underlying=cap)
    try:
        raise LLMServiceError("Selected route ollama:local failed") from cap
    except LLMServiceError as exc:
        non_streamed = exc

    assert incomplete_generation(streamed) is cap
    assert incomplete_generation(non_streamed) is cap

    # An aggregate over several routes states its own verdict.
    aggregate = LLMStreamingError("All providers failed", underlying=cap)
    aggregate.declined_wait = None
    assert incomplete_generation(aggregate) is None
    assert incomplete_generation(RuntimeError("unrelated")) is None
