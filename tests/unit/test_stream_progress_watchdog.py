"""#3357: provider progress that yields no stream item keeps a call alive.

The orchestrator's inactivity watchdog (#3300) re-armed only on items a stream
yields, and two things a Claude turn can do for minutes yield none:

* Generating one large tool call streams its arguments as ``input_json_delta``
  events. The adapter holds them until the call is whole, so it never
  announces a call that may still be cut.
* Thinking that is not displayed. With ``display: "omitted"``, the default on
  current Claude models, each thinking block streams one empty
  ``thinking_delta`` and then its ``signature_delta``. The minutes between are
  ``ping`` events, which the Anthropic SDK consumes itself. Redacted thinking
  arrives as whole blocks with no text either.

Either one tripped the bound on a turn that was still making progress. The
Anthropic adapter now reports each event it reads, and each chunk of the
response body (the only place the pings are visible), on the call-progress
side channel the watchdog listens on
(:mod:`kestrel_sovereign.llm.call_progress`). Nothing is added to the stream.

Time is virtual: the scripted provider advances the event loop's clock, so
these tests cover minutes of streaming without sleeping. The Anthropic
provider is either the real SDK client over a scripted HTTP transport
(``sdk``) or a stream of the SDK's own event objects with no HTTP body behind
it (``events``), which isolates the per-event report.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, List, Optional, Tuple

from tests.utils.anthropic_client import anthropic_model_record

import anthropic
import httpx2
import openai
import pytest
from anthropic.types import (
    InputJSONDelta,
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    RedactedThinkingBlock,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)
from anthropic.types import ThinkingDelta as SDKThinkingDelta
from anthropic.types.raw_message_delta_event import Delta

from kestrel_sdk.llm import ToolCallStarted
from kestrel_sovereign.agent.orchestrator_engine import (
    LLMCallInactivityTimeout,
    LLMCallWatchdog,
)
from kestrel_sovereign.agent.streaming import _parse_stream_sentinels
from kestrel_sovereign.llm.adapter import LLMResponse, ThinkingDelta, ToolCall
from kestrel_sovereign.llm.anthropic_adapter import (
    CLAUDE_CODE_IDENTITY,
    AnthropicAdapter,
)
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from tests.utils.virtual_clock import VirtualClockLoop

BOUND = 180
STEP = 10
MODEL = "claude-opus-5"
USER = [{"role": "user", "content": "Write the release notes to notes.md."}]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
        },
    },
}]
ARGUMENTS = {"path": "notes.md", "content": "A line of the release notes.\n" * 40}
ARGUMENT_JSON = json.dumps(ARGUMENTS)

#: A frame the Anthropic SDK consumes itself, so only the ``sdk`` provider can
#: send it.
PING = object()
#: The provider goes silent: nothing more ever arrives.
SILENCE = object()

ROUTES = {"api-key": AnthropicAdapter, "plan": ClaudeMaxAdapter}


def _pieces(text: str, count: int) -> List[str]:
    size = -(-len(text) // count)
    return [text[i:i + size] for i in range(0, len(text), size)]


# --------------------------------------------------------------------------
# The scripted Anthropic provider
# --------------------------------------------------------------------------


def _message_start() -> RawMessageStartEvent:
    return RawMessageStartEvent(type="message_start", message=Message(
        id="msg_1", type="message", role="assistant", content=[], model=MODEL,
        stop_reason=None, stop_sequence=None,
        usage=Usage(input_tokens=1200, output_tokens=1),
    ))


def _start(index: int, block: Any) -> RawContentBlockStartEvent:
    return RawContentBlockStartEvent(
        type="content_block_start", index=index, content_block=block,
    )


def _delta(index: int, delta: Any) -> RawContentBlockDeltaEvent:
    return RawContentBlockDeltaEvent(
        type="content_block_delta", index=index, delta=delta,
    )


def _stop(index: int) -> RawContentBlockStopEvent:
    return RawContentBlockStopEvent(type="content_block_stop", index=index)


def _end(stop_reason: str) -> List[Tuple[int, Any]]:
    return [
        (0, RawMessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason=stop_reason, stop_sequence=None),
            usage=MessageDeltaUsage(output_tokens=900),
        )),
        (0, RawMessageStopEvent(type="message_stop")),
    ]


def _text_block(index: int, text: str) -> List[Tuple[int, Any]]:
    return [
        (STEP, _start(index, TextBlock(type="text", text=""))),
        (0, _delta(index, TextDelta(type="text_delta", text=text))),
        (0, _stop(index)),
    ]


def _input_json(index: int, partial_json: str) -> RawContentBlockDeltaEvent:
    return _delta(index, InputJSONDelta(type="input_json_delta", partial_json=partial_json))


def _omitted_thinking(index: int, *, gaps: int) -> List[Tuple[int, Any]]:
    """A thinking block under ``display: "omitted"``: an empty delta, then
    ``gaps`` thirty-second stretches of thinking, then its signature."""
    return [
        (0, _start(index, ThinkingBlock(type="thinking", thinking="", signature=""))),
        (0, _delta(index, SDKThinkingDelta(type="thinking_delta", thinking=""))),
        *[(30, PING) for _ in range(gaps)],
        (30, _delta(index, SignatureDelta(type="signature_delta", signature="EqQBCgIYAhIM"))),
        (0, _stop(index)),
    ]


class _ScriptedProvider:
    """One provider response, sent frame by frame on a virtual clock.

    ``script`` is ``(seconds, frame)`` pairs: the provider waits ``seconds``,
    then sends ``frame``. It records when each frame was sent, what each
    request asked for, and when its read was cancelled. ``clock`` is a
    :class:`VirtualClockLoop` or a pinned loop clock; either moves only when
    the provider advances it.
    """

    def __init__(self, clock: Any, script: List[Tuple[int, Any]]) -> None:
        self.clock = clock
        self.script = script
        self.sent: List[Tuple[float, Any]] = []
        self.requests: List[httpx2.Request] = []
        self.cancelled_at: Optional[float] = None

    async def _wait(self, seconds: float) -> None:
        for _ in range(int(seconds // STEP)):
            self.clock.advance(STEP)
            await asyncio.sleep(0)

    async def frames(self) -> AsyncIterator[Any]:
        try:
            for seconds, frame in self.script:
                if frame is SILENCE:
                    await self._wait(10_000)
                    return
                await self._wait(seconds)
                self.sent.append((self.clock.time(), frame))
                yield frame
        except asyncio.CancelledError:
            self.cancelled_at = self.clock.time()
            raise

    def last_sent(self, kind: Any) -> float:
        return max(at for at, frame in self.sent if isinstance(frame, kind))


def _sse(frame: Any) -> bytes:
    if frame is PING:
        return b'event: ping\ndata: {"type": "ping"}\n\n'
    return f"event: {frame.type}\ndata: {frame.model_dump_json()}\n\n".encode()


def _sdk_client(route: str, provider: _ScriptedProvider) -> anthropic.AsyncAnthropic:
    """The real Anthropic SDK client, its HTTP transport scripted."""

    async def body() -> AsyncIterator[bytes]:
        async for frame in provider.frames():
            yield _sse(frame)

    async def handle(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            # The Models API record the adapter reads the output ceiling from.
            return httpx2.Response(200, json=anthropic_model_record(
                MODEL, max_tokens=128_000,
            ).model_dump(mode="json"))
        provider.requests.append(request)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=body(),
        )

    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    if route == "plan":
        client = anthropic.AsyncAnthropic(
            auth_token="sk-ant-oat01-test", http_client=http_client, max_retries=0,
        )
        client.api_key = None
        return client
    return anthropic.AsyncAnthropic(
        api_key="sk-ant-api03-test", http_client=http_client, max_retries=0,
    )


class _EventStream:
    """``client.messages.stream(...)`` as the SDK's event objects alone, with
    no HTTP body behind them. Pings never surface as events, so none are
    sent."""

    def __init__(self, provider: _ScriptedProvider) -> None:
        self._frames = provider.frames()

    async def __aenter__(self) -> "_EventStream":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        await self._frames.aclose()
        return False

    def __aiter__(self) -> "_EventStream":
        return self

    async def __anext__(self) -> Any:
        while True:
            frame = await self._frames.__anext__()
            if frame is not PING:
                return frame


class _EventClient:
    def __init__(self, provider: _ScriptedProvider) -> None:
        async def retrieve(model_id: str, **_: Any) -> Any:
            return anthropic_model_record(model_id, max_tokens=128_000)

        self.messages = type("Messages", (), {
            "stream": staticmethod(lambda **_: _EventStream(provider)),
        })()
        self.models = type("Models", (), {"retrieve": staticmethod(retrieve)})()


def _client(kind: str, route: str, provider: _ScriptedProvider) -> Any:
    if kind == "sdk":
        return _sdk_client(route, provider)
    return _EventClient(provider)


def _drive(
    loop: VirtualClockLoop,
    stream_factory,
    client: Any,
) -> Tuple[List[Tuple[float, Any]], Optional[LLMCallInactivityTimeout]]:
    """What a stream yielded under the watchdog, each item with the virtual
    time it arrived, and the timeout that cut it, if any."""
    received: List[Tuple[float, Any]] = []

    async def run() -> None:
        try:
            async for item in LLMCallWatchdog(BOUND).watch(stream_factory()):
                received.append((loop.time(), item))
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()

    try:
        loop.run_until_complete(run())
    except LLMCallInactivityTimeout as exc:
        return received, exc
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    return received, None


def _stream_with_tools(adapter, client):
    return lambda: adapter.get_streaming_response_with_tools(
        client=client, model=MODEL, messages=USER, tools=TOOLS,
    )


def _items(received: List[Tuple[float, Any]]) -> List[Any]:
    return [item for _, item in received]


def _longest_gap_between_items(received, provider: _ScriptedProvider) -> float:
    """The longest stretch the consumer was handed nothing while the provider
    was sending, from its first frame to its last."""
    times = sorted([
        provider.sent[0][0], provider.sent[-1][0], *(at for at, _ in received),
    ])
    return max(later - earlier for earlier, later in zip(times, times[1:]))


def _assert_route_shape(kind: str, route: str, provider: _ScriptedProvider) -> None:
    """The request really went out on the route under test."""
    if kind != "sdk":
        return
    (request,) = provider.requests
    system = json.loads(request.content).get("system") or [{"text": None}]
    first_system = system[0]["text"]
    if route == "plan":
        assert request.headers["authorization"] == "Bearer sk-ant-oat01-test"
        assert first_system == CLAUDE_CODE_IDENTITY
    else:
        assert request.headers["x-api-key"] == "sk-ant-api03-test"
        assert first_system != CLAUDE_CODE_IDENTITY


ANTHROPIC_CASES = pytest.mark.parametrize(
    "kind,route",
    [("sdk", "api-key"), ("sdk", "plan"), ("events", "api-key"), ("events", "plan")],
)


# --------------------------------------------------------------------------
# A tool call whose arguments outlast the bound
# --------------------------------------------------------------------------


def _tool_call_opening(pieces: List[str]) -> List[Tuple[int, Any]]:
    """Prose, then a tool call whose arguments arrive as ``pieces``, ten
    virtual seconds apart."""
    return [
        (0, _message_start()),
        *_text_block(0, "Writing the notes now."),
        (STEP, _start(1, ToolUseBlock(
            type="tool_use", id="toolu_1", name="write_file", input={},
        ))),
        *[(STEP, _input_json(1, piece)) for piece in pieces],
    ]


def _long_tool_call_script(pieces: List[str]) -> List[Tuple[int, Any]]:
    return [*_tool_call_opening(pieces), (STEP, _stop(1)), *_end("tool_use")]


@ANTHROPIC_CASES
def test_a_tool_call_whose_arguments_outlast_the_bound_is_not_cut(kind, route):
    """Forty argument deltas ten virtual seconds apart: 400 seconds with
    nothing yielded, more than twice the bound. The call arrives once, whole,
    and is announced only once its block has closed."""
    loop = VirtualClockLoop()
    provider = _ScriptedProvider(loop, _long_tool_call_script(_pieces(ARGUMENT_JSON, 40)))
    client = _client(kind, route, provider)

    received, failure = _drive(loop, _stream_with_tools(ROUTES[route](), client), client)

    assert failure is None
    assert _longest_gap_between_items(received, provider) > 2 * BOUND
    items = _items(received)
    assert [type(item) for item in items] == [str, ToolCallStarted, LLMResponse]
    assert items[0] == "Writing the notes now."

    (announced_at, started), = [r for r in received if isinstance(r[1], ToolCallStarted)]
    assert (started.id, started.name) == ("toolu_1", "write_file")
    assert announced_at >= provider.last_sent(RawContentBlockStopEvent)

    final = items[-1]
    assert final.tool_calls == [
        ToolCall(id="toolu_1", name="write_file", arguments=ARGUMENTS),
    ]
    assert final.content == "Writing the notes now."
    assert (final.input_tokens, final.output_tokens) == (1200, 900)
    _assert_route_shape(kind, route, provider)


# --------------------------------------------------------------------------
# Thinking that is not displayed
# --------------------------------------------------------------------------


def _answer_after(thinking: List[Tuple[int, Any]], text_index: int) -> List[Tuple[int, Any]]:
    return [
        (0, _message_start()),
        *thinking,
        *_text_block(text_index, "Done."),
        *_end("end_turn"),
    ]


@pytest.mark.parametrize("route", ["api-key", "plan"])
def test_omitted_thinking_that_outlasts_the_bound_is_not_cut(route):
    """The production shape: six minutes of thinking under ``display:
    "omitted"`` reach the client as pings between an empty thinking delta and
    the block's signature. The SDK consumes the pings; the body still carries
    them."""
    loop = VirtualClockLoop()
    provider = _ScriptedProvider(
        loop, _answer_after(_omitted_thinking(0, gaps=11), text_index=1),
    )
    client = _sdk_client(route, provider)

    received, failure = _drive(loop, _stream_with_tools(ROUTES[route](), client), client)

    assert failure is None
    assert _longest_gap_between_items(received, provider) > 2 * BOUND
    items = _items(received)
    assert [type(item) for item in items] == [str, LLMResponse]
    assert items[0] == "Done."
    assert items[1].content == "Done."
    assert items[1].tool_calls is None
    _assert_route_shape("sdk", route, provider)


HIDDEN_THINKING_BLOCKS = 5


def _hidden_thinking_events() -> List[Tuple[int, Any]]:
    """Interleaved hidden thinking: omitted blocks whose deltas carry no text,
    and a redacted block, none more than a minute apart."""
    blocks: List[Tuple[int, Any]] = []
    for index in range(HIDDEN_THINKING_BLOCKS):
        if index == 2:
            blocks += [
                (60, _start(index, RedactedThinkingBlock(
                    type="redacted_thinking", data="EmwKAhgB",
                ))),
                (0, _stop(index)),
            ]
        else:
            blocks += [
                (30, _start(index, ThinkingBlock(type="thinking", thinking="", signature=""))),
                (30, _delta(index, SDKThinkingDelta(type="thinking_delta", thinking=""))),
                (30, _delta(index, SignatureDelta(type="signature_delta", signature="EqQB"))),
                (0, _stop(index)),
            ]
    return blocks


@ANTHROPIC_CASES
def test_hidden_thinking_deltas_that_outlast_the_bound_are_not_cut(kind, route):
    loop = VirtualClockLoop()
    provider = _ScriptedProvider(
        loop, _answer_after(_hidden_thinking_events(), text_index=HIDDEN_THINKING_BLOCKS),
    )
    client = _client(kind, route, provider)

    received, failure = _drive(loop, _stream_with_tools(ROUTES[route](), client), client)

    assert failure is None
    assert _longest_gap_between_items(received, provider) > 2 * BOUND
    assert [type(item) for item in _items(received)] == [str, LLMResponse]
    assert not any(isinstance(item, ThinkingDelta) for item in _items(received))
    _assert_route_shape(kind, route, provider)


@pytest.mark.parametrize("kind", ["sdk", "events"])
def test_the_text_only_stream_counts_hidden_thinking_too(kind):
    """``get_streaming_response`` reads the same events through the same
    helper."""
    loop = VirtualClockLoop()
    thinking = (
        _omitted_thinking(0, gaps=11) if kind == "sdk" else _hidden_thinking_events()
    )
    provider = _ScriptedProvider(
        loop, _answer_after(
            thinking, text_index=1 if kind == "sdk" else HIDDEN_THINKING_BLOCKS,
        ),
    )
    client = _client(kind, "api-key", provider)
    adapter = AnthropicAdapter()

    received, failure = _drive(
        loop,
        lambda: adapter.get_streaming_response(client=client, model=MODEL, messages=USER),
        client,
    )

    assert failure is None
    assert _longest_gap_between_items(received, provider) > 2 * BOUND
    assert _items(received) == ["Done."]


def test_omitted_thinking_does_not_cut_a_non_streaming_answer():
    """The invoke path reads its answer off the same kind of stream (#3552),
    under the same bound, so its pings are progress as well."""
    loop = VirtualClockLoop()
    provider = _ScriptedProvider(
        loop, _answer_after(_omitted_thinking(0, gaps=11), text_index=1),
    )
    client = _sdk_client("api-key", provider)

    async def run() -> LLMResponse:
        try:
            return await LLMCallWatchdog(BOUND).call(AnthropicAdapter().get_response(
                client=client, model=MODEL, messages=USER,
            ))
        finally:
            await client.close()

    try:
        response = loop.run_until_complete(run())
    finally:
        loop.close()

    assert response.content == "Done."
    assert loop.time() > 2 * BOUND


# --------------------------------------------------------------------------
# Silence still trips the bound
# --------------------------------------------------------------------------


@ANTHROPIC_CASES
def test_a_stream_that_goes_silent_mid_tool_call_is_still_cut(kind, route):
    """Arguments arrive, then nothing at all, not even a ping: the bound trips
    one bound after the last frame, the provider read is cancelled, and no
    part of the unfinished call reaches the consumer."""
    loop = VirtualClockLoop()
    provider = _ScriptedProvider(loop, [
        *_tool_call_opening(_pieces(ARGUMENT_JSON, 40)[:5]), (0, SILENCE),
    ])
    client = _client(kind, route, provider)

    received, failure = _drive(loop, _stream_with_tools(ROUTES[route](), client), client)

    assert isinstance(failure, LLMCallInactivityTimeout)
    assert _items(received) == ["Writing the notes now."]
    last_frame = provider.sent[-1][0]
    assert isinstance(provider.sent[-1][1], RawContentBlockDeltaEvent)
    assert provider.cancelled_at is not None
    assert BOUND <= provider.cancelled_at - last_frame <= BOUND + 2 * STEP


@ANTHROPIC_CASES
def test_a_stream_that_never_sends_anything_is_still_cut(kind, route):
    loop = VirtualClockLoop()
    provider = _ScriptedProvider(loop, [(0, SILENCE)])
    client = _client(kind, route, provider)

    received, failure = _drive(loop, _stream_with_tools(ROUTES[route](), client), client)

    assert isinstance(failure, LLMCallInactivityTimeout)
    assert received == []
    assert provider.cancelled_at is not None
    assert BOUND <= provider.cancelled_at <= BOUND + 2 * STEP


def test_a_stream_whose_pings_stop_is_cut_one_bound_after_the_last():
    """A ping keeps the call alive for one bound, no longer: when even the
    pings stop, the call is cut one bound after the last of them."""
    loop = VirtualClockLoop()
    pings = 11
    provider = _ScriptedProvider(loop, [
        (0, _message_start()),
        (0, _start(0, ThinkingBlock(type="thinking", thinking="", signature=""))),
        *[(30, PING) for _ in range(pings)],
        (0, SILENCE),
    ])
    client = _sdk_client("plan", provider)

    received, failure = _drive(
        loop, _stream_with_tools(ClaudeMaxAdapter(), client), client,
    )

    assert isinstance(failure, LLMCallInactivityTimeout)
    assert received == []
    last_ping = provider.sent[-1][0]
    assert provider.sent[-1][1] is PING and last_ping == pings * 30
    assert BOUND <= provider.cancelled_at - last_ping <= BOUND + 2 * STEP


# --------------------------------------------------------------------------
# The OpenAI-family adapter holds tool-call arguments the same way
# --------------------------------------------------------------------------


def _openai_chunk(delta: dict, finish_reason: Optional[str] = None) -> dict:
    return {
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0,
        "model": "gpt-5", "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason},
        ],
    }


def _openai_tool_call_script() -> List[Tuple[int, dict]]:
    first = {"tool_calls": [{
        "index": 0, "id": "call_1", "type": "function",
        "function": {"name": "write_file", "arguments": ""},
    }]}
    return [
        (0, _openai_chunk({"role": "assistant", "content": "Writing the notes now."})),
        (STEP, _openai_chunk(first)),
        *[
            (STEP, _openai_chunk({"tool_calls": [
                {"index": 0, "function": {"arguments": piece}},
            ]}))
            for piece in _pieces(ARGUMENT_JSON, 40)
        ],
        (STEP, _openai_chunk({}, finish_reason="tool_calls")),
        (0, {
            "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0,
            "model": "gpt-5", "choices": [],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 900, "total_tokens": 2100},
        }),
    ]


def _openai_client(provider: _ScriptedProvider) -> openai.AsyncOpenAI:
    async def body() -> AsyncIterator[bytes]:
        async for chunk in provider.frames():
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def handle(request: httpx2.Request) -> httpx2.Response:
        provider.requests.append(request)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=body(),
        )

    return openai.AsyncOpenAI(
        api_key="sk-test", max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
    )


def test_openai_tool_call_arguments_that_outlast_the_bound_are_not_cut():
    loop = VirtualClockLoop()
    provider = _ScriptedProvider(loop, _openai_tool_call_script())
    client = _openai_client(provider)
    adapter = OpenAIAdapter()

    received, failure = _drive(
        loop,
        lambda: adapter.get_streaming_response_with_tools(
            client=client, model="gpt-5", messages=USER, tools=TOOLS,
        ),
        client,
    )

    assert failure is None
    assert _longest_gap_between_items(received, provider) > 2 * BOUND
    final = _items(received)[-1]
    assert isinstance(final, LLMResponse)
    assert final.tool_calls == [
        ToolCall(id="call_1", name="write_file", arguments=ARGUMENTS),
    ]


def test_openai_text_stream_counts_chunks_it_does_not_yield():
    """The text-only stream drops tool-call deltas; they are still the
    provider answering."""
    loop = VirtualClockLoop()
    provider = _ScriptedProvider(loop, _openai_tool_call_script()[:-1])
    client = _openai_client(provider)
    adapter = OpenAIAdapter()

    received, failure = _drive(
        loop,
        lambda: adapter.get_streaming_response(
            client=client, model="gpt-5", messages=USER, tools=TOOLS,
        ),
        client,
    )

    assert failure is None
    assert _longest_gap_between_items(received, provider) > 2 * BOUND
    assert _items(received) == ["Writing the notes now."]


# --------------------------------------------------------------------------
# A whole turn: nothing partial or extra reaches the consumer or the history
# --------------------------------------------------------------------------

SEARCH = " ".join(["sovereignty"] * 80)
SEARCH_ARGUMENTS = {"search": SEARCH}


def _constitution_call_script() -> List[Tuple[int, Any]]:
    return [
        (0, _message_start()),
        *_text_block(0, "Let me search the constitution."),
        (STEP, _start(1, ToolUseBlock(
            type="tool_use", id="toolu_1", name="constitution", input={},
        ))),
        *[
            (STEP, _input_json(1, piece))
            for piece in _pieces(json.dumps(SEARCH_ARGUMENTS), 40)
        ],
        (STEP, _stop(1)),
        *_end("tool_use"),
    ]


@pytest.mark.asyncio
async def test_a_turn_keeps_only_whole_items_through_long_progress(tmp_path, monkeypatch):
    """A real turn on the plan route: a tool call whose arguments stream for
    400 virtual seconds, then six minutes of omitted thinking before the
    answer. Neither call trips its bound; the turn shows and records the
    answer; the model's history holds the call once, with its whole
    arguments; and no partial call or progress artefact appears anywhere."""
    from kestrel_sovereign.agent import orchestrator_engine as oe
    from tests.shared.booted_agent import booted_agent
    from tests.utils.virtual_clock import pinned_loop_clock

    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", BOUND)
    bounds: List[float] = []
    for_call = LLMCallWatchdog.for_call.__func__

    def recording(cls, *args, **kwargs):
        watchdog = for_call(cls, *args, **kwargs)
        bounds.append(watchdog.timeout)
        return watchdog

    monkeypatch.setattr(LLMCallWatchdog, "for_call", classmethod(recording))

    async with booted_agent(
        tmp_path, name="Progress", provenance="test:stream_progress_watchdog",
        features=set(),
    ) as agent:
        service = agent.llm_service
        adapter = ClaudeMaxAdapter()
        with pinned_loop_clock() as clock:
            calls = [
                _ScriptedProvider(clock, _constitution_call_script()),
                _ScriptedProvider(clock, _answer_after(
                    _omitted_thinking(0, gaps=11), text_index=1,
                )),
            ]
            pending = iter(calls)

            async def plan_route(**kwargs):
                provider = next(pending)
                client = _sdk_client("plan", provider)
                try:
                    async for item in service._stream_adapter_with_usage(
                        adapter=adapter, client=client, model=MODEL,
                        messages=kwargs["messages"],
                        provider_name="anthropic:plan",
                        path="stream_with_tool_detection",
                        invocation_context=kwargs.get("invocation_context"),
                        expose_protocol_events=True,
                        tools=kwargs.get("tools"),
                        system_prompt=kwargs.get("system_prompt"),
                        cancel_token=kwargs.get("cancel_token"),
                        require_completion=True,
                    ):
                        yield item
                finally:
                    await client.close()

            monkeypatch.setattr(service, "stream_with_tool_detection", plan_route)
            received: List[Tuple[float, Any]] = []
            async for chunk in agent.process_input_streaming(
                "Where does the constitution speak of sovereignty?",
                session_id="s1",
            ):
                received.append((clock.time(), chunk))

        # Both calls ran past twice their bound with nothing reaching the
        # consumer, and neither was cut.
        assert bounds == [BOUND, BOUND]
        assert all(provider.cancelled_at is None for provider in calls)
        times = [at for at, _ in received]
        long_waits = [b - a for a, b in zip(times, times[1:]) if b - a > 2 * BOUND]
        assert len(long_waits) == 2

        # The consumer: text and in-band sentinels only, the call announced
        # once and no fragment of its arguments shown.
        chunks = [chunk for _, chunk in received]
        assert all(isinstance(chunk, str) for chunk in chunks)
        streamed = "".join(chunks)
        revises = [c for c in chunks if c.startswith("\x1eKESTREL:REVISE:")]
        assert len(revises) == 1 and '"tool_call_id":"toolu_1"' in revises[0]
        clean, tool_parts, parts = _parse_stream_sentinels(streamed)
        assert [p["phase"] for p in tool_parts] == ["start", "done"]
        assert parts == []
        assert clean.endswith("Done.")
        assert '"search"' not in streamed and "sovereignty sovereignty" not in streamed

        # The model's own history: the call once, with its whole arguments,
        # answered by its result.
        follow_up = json.loads(calls[1].requests[0].content)["messages"]
        blocks = [
            block for message in follow_up
            if isinstance(message["content"], list)
            for block in message["content"]
        ]
        assert [
            (b["id"], b["name"], b["input"]) for b in blocks if b["type"] == "tool_use"
        ] == [("toolu_1", "constitution", SEARCH_ARGUMENTS)]
        assert [
            b["tool_use_id"] for b in blocks if b["type"] == "tool_result"
        ] == ["toolu_1"]

        # The recorded turn: one assistant row, the answer, its one tool
        # result with the whole arguments.
        history = await agent.privacy_agent.get_conversation_history(
            limit=50, session_id="s1",
        )
        (answer,) = [row for row in history if row.get("role") == "assistant"]
        assert answer["content"].endswith("Done.")
        assert '"search"' not in answer["content"]
        assert [
            (result["tool_call_id"], result["arguments"])
            for result in answer["metadata"]["tool_results"]
        ] == [("toolu_1", SEARCH_ARGUMENTS)]
