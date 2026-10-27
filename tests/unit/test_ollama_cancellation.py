"""#3572: Stop abandons an Ollama chat call even while the daemon sends nothing.

A streamed turn checks Stop between stream items, so a call that has yielded
nothing never reaches that check. Historically, the tool path read a whole
non-streaming response before returning; the logical tool response remains
buffered while its native SDK stream is now consumed internally. A daemon
still evaluating a long prompt may send nothing at all. Every path must
therefore admit Stop while awaiting the next native event.

The adapter now honours the ``cancel_token`` it is passed on every chat path:
a fired token cancels the pending request, which closes the HTTP response so
the daemon stops generating, and the call raises ``CancelledError`` rather
than returning anything.

These tests drive the real ``ollama.AsyncClient`` against a stand-in daemon:
an in-process HTTP transport, and a real loopback socket for a daemon that has
not yet sent its response headers.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

import httpx
import pytest

ollama = pytest.importorskip("ollama")

from kestrel_sovereign.llm.adapter import LLMResponse  # noqa: E402
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter  # noqa: E402

MODEL = "llama3.2:1b"
MESSAGES = [{"role": "user", "content": "Where do I work?"}]
TOOLS = [{
    "type": "function",
    "function": {"name": "lookup", "description": "", "parameters": {}},
}]
#: How long a call may take to end once its token fires. The token is polled
#: every 0.1 s; a call that ignores it never ends.
PROMPTLY = 5.0


def _line(payload: Dict[str, Any]) -> bytes:
    return (json.dumps(payload) + "\n").encode()


def _piece(content: str) -> bytes:
    return _line({
        "model": MODEL,
        "message": {"role": "assistant", "content": content},
        "done": False,
    })


class _Silent(httpx.AsyncByteStream):
    """A response body that sends ``chunks``, then nothing, until it is closed.

    A daemon evaluating a long prompt sends no bytes; a stream between two
    tokens sends none either.
    """

    def __init__(self, chunks: Optional[List[bytes]] = None) -> None:
        self._chunks = list(chunks or [])
        self.closed = False
        #: Set once a read waits on the silence.
        self.silent = asyncio.Event()

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
        self.silent.set()
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


class _Daemon:
    """A stand-in Ollama daemon whose ``/api/chat`` reply is ``body``.

    ``stop_on_request`` fires the call's cancel token as the request arrives:
    Stop lands while the daemon has sent nothing.
    """

    def __init__(self, body: _Silent, *, stop_on_request: bool = True) -> None:
        self.body = body
        self.stop_on_request = stop_on_request
        self.stopped = False
        self.requests: List[Dict[str, Any]] = []

    def client(self) -> "ollama.AsyncClient":
        return ollama.AsyncClient(
            host="http://ollama.test:11434",
            transport=httpx.MockTransport(self._handle),
        )

    def cancel_token(self) -> bool:
        return self.stopped

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if self.stop_on_request:
            self.stopped = True
        return httpx.Response(
            200, headers={"content-type": "application/x-ndjson"}, stream=self.body,
        )


async def _drain(stream: AsyncIterator[Any], seen: List[Any]) -> None:
    async for item in stream:
        seen.append(item)


def _get_response(daemon: _Daemon, tools: Optional[list]) -> Awaitable[Any]:
    return OllamaAdapter().get_response(
        client=daemon.client(), model=MODEL, messages=MESSAGES, tools=tools,
        cancel_token=daemon.cancel_token,
    )


def _stream_with_tools(daemon: _Daemon, tools: Optional[list], seen: List[Any]):
    return _drain(OllamaAdapter().get_streaming_response_with_tools(
        client=daemon.client(), model=MODEL, messages=MESSAGES, tools=tools,
        cancel_token=daemon.cancel_token,
    ), seen)


def _text_stream(daemon: _Daemon, seen: List[Any]):
    return _drain(OllamaAdapter().get_streaming_response(
        client=daemon.client(), model=MODEL, messages=MESSAGES,
        cancel_token=daemon.cancel_token,
    ), seen)


async def _assert_abandoned(call: Awaitable[Any]) -> None:
    """``call`` ends promptly, with the cancellation its token raised."""
    task = asyncio.ensure_future(call)
    try:
        done, _ = await asyncio.wait({task}, timeout=PROMPTLY)
        assert task in done, "the call ignored its fired cancel token"
        # A coroutine that raises CancelledError leaves its task cancelled:
        # the call raised the cancellation, it returned nothing.
        assert task.cancelled(), f"the call ended without being cancelled: {task!r}"
    finally:
        task.cancel()


#: Every chat path, as ``(name, call)``: ``call(daemon, seen)`` makes the call.
PATHS: Dict[str, Callable[[_Daemon, List[Any]], Awaitable[Any]]] = {
    # A buffered logical call with tools: what a tool-loop follow-up makes.
    "get_response_with_tools": lambda d, seen: _get_response(d, TOOLS),
    "get_response_without_tools": lambda d, seen: _get_response(d, None),
    # A streamed turn with tools: one buffered logical chat (#3572's case).
    "stream_with_tools": lambda d, seen: _stream_with_tools(d, TOOLS, seen),
    # A streamed turn without tools: a streamed chat.
    "stream_without_tools": lambda d, seen: _stream_with_tools(d, None, seen),
    "text_stream": lambda d, seen: _text_stream(d, seen),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(PATHS))
async def test_stop_abandons_a_call_the_daemon_has_sent_nothing_of(path):
    """Stop lands while the daemon is still silent: the call raises the
    cancellation at once, nothing reaches the caller, and the HTTP response is
    closed, which is what makes Ollama stop generating."""
    body = _Silent()
    daemon = _Daemon(body)
    seen: List[Any] = []

    await _assert_abandoned(PATHS[path](daemon, seen))

    assert len(daemon.requests) == 1, "the call never reached the daemon"
    assert body.closed, "the HTTP response to Ollama was left open"
    assert seen == []


@pytest.mark.asyncio
async def test_stop_between_streamed_tokens_abandons_the_stream():
    """A stream that has delivered part of its generation and then gone
    silent ends on Stop too, without waiting for the next token."""
    body = _Silent([_piece("PARTIAL ")])
    daemon = _Daemon(body, stop_on_request=False)
    seen: List[Any] = []

    async def stop_once_silent() -> None:
        # Stop lands while the stream's read of its next token is pending.
        await body.silent.wait()
        daemon.stopped = True

    stopper = asyncio.ensure_future(stop_once_silent())
    try:
        await _assert_abandoned(_stream_with_tools(daemon, None, seen))
    finally:
        stopper.cancel()

    assert seen == ["PARTIAL "]
    assert body.closed, "the HTTP response to Ollama was left open"


class _MuteDaemon:
    """An Ollama daemon on a real loopback socket that reads each chat request
    and answers nothing, not even its response headers, as a daemon still
    evaluating a long prompt does. Stop lands once it has the request.

    It sees what a real daemon sees: whether the client's connection closed.
    That is what makes Ollama abort a generation.
    """

    def __init__(self) -> None:
        self.stopped = False
        self.requests: List[Dict[str, Any]] = []
        self.disconnected = asyncio.Event()
        self._writers: List[asyncio.StreamWriter] = []
        self._clients: List["ollama.AsyncClient"] = []

    async def __aenter__(self) -> "_MuteDaemon":
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self._port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        for client in self._clients:
            await client._client.aclose()
        for writer in self._writers:
            writer.close()
        self._server.close()

    def client(self) -> "ollama.AsyncClient":
        client = ollama.AsyncClient(
            host=f"http://127.0.0.1:{self._port}", trust_env=False,
        )
        self._clients.append(client)
        return client

    def cancel_token(self) -> bool:
        return self.stopped

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        self._writers.append(writer)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in head.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            self.requests.append(json.loads(await reader.readexactly(length)))
            self.stopped = True
            while await reader.read(65536):
                pass
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        self.disconnected.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(PATHS))
async def test_stop_before_the_daemon_answers_drops_the_connection(path):
    """Before the daemon has sent its response headers there is no HTTP
    response to close; Stop must still drop the connection."""
    async with _MuteDaemon() as daemon:
        await _assert_abandoned(PATHS[path](daemon, []))

        assert len(daemon.requests) == 1, "the call never reached the daemon"
        try:
            await asyncio.wait_for(daemon.disconnected.wait(), PROMPTLY)
        except TimeoutError:
            pytest.fail("the connection to the Ollama daemon was left open")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(PATHS))
async def test_a_call_stopped_before_it_starts_sends_nothing(path):
    body = _Silent()
    daemon = _Daemon(body)
    daemon.stopped = True

    await _assert_abandoned(PATHS[path](daemon, []))

    assert daemon.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(PATHS))
async def test_a_token_that_never_fires_changes_nothing(path):
    """The token is the adapter's own: it is not sent to Ollama, and a call
    it never stops answers as before."""
    reply = {
        "model": MODEL,
        "message": {"role": "assistant", "content": "You work at Acme."},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 11,
        "eval_count": 3,
    }

    class _Answer(_Silent):
        async def __aiter__(self):
            yield _line(reply)

    body = _Answer()
    daemon = _Daemon(body, stop_on_request=False)
    seen: List[Any] = []

    result = await asyncio.wait_for(PATHS[path](daemon, seen), PROMPTLY)

    [request] = daemon.requests
    assert "cancel_token" not in request
    answered = result if isinstance(result, LLMResponse) else None
    text = answered.content if answered is not None else "".join(
        item for item in seen if isinstance(item, str)
    )
    assert text == "You work at Acme."
