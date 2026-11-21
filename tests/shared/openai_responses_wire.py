"""An offline OpenAI Responses API for driving the real ``openai`` SDK (#4198).

:class:`ResponsesWire` is an ``httpx.MockTransport`` handler. It records every
request exactly as the SDK serialized it and answers ``POST /v1/responses``
with a server-sent event stream built from a scripted :class:`Reply`, chosen by
a ``script`` callable from the parsed request body. Both of the adapter's
entry points stream, so one wire serves both. Given a ``chat_script``, it also
answers ``POST /v1/chat/completions`` with a scripted :class:`ChatReply`, as
an ordinary completion or a stream, whichever the request asked for.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx
import openai

MODEL = "gpt-6.2-sol"
SERVED = "gpt-6.2-sol-2026-11-01"
#: A chat-completions default and the dated model the provider says served it.
CHAT_MODEL = "gpt-4.1-mini"
CHAT_SERVED = "gpt-4.1-mini-2025-04-14"


def reasoning(item_id: str, encrypted: str) -> Dict[str, Any]:
    return {"type": "reasoning", "id": item_id, "summary": [], "encrypted_content": encrypted}


def call(item_id: str, call_id: str, name: str, arguments: str) -> Dict[str, Any]:
    return {
        "type": "function_call", "id": item_id, "call_id": call_id, "name": name,
        "arguments": arguments, "status": "completed",
    }


def message(item_id: str, text: str, phase: str = "final_answer") -> Dict[str, Any]:
    return {
        "type": "message", "id": item_id, "role": "assistant", "status": "completed",
        "phase": phase,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


@dataclass
class Reply:
    """One scripted response.

    ``status`` is the terminal status (``completed``, ``incomplete`` or
    ``failed``); ``error`` the failed response's error object. ``usage`` is
    the terminal response's usage, omitted when ``None``.
    ``disconnect_after_created`` ends the body with a transport error right
    after ``response.created``; ``disconnect_after_terminal`` ends it with one
    after the terminal event, before the stream's clean end.
    ``stall_after_created`` and ``stall_after_terminal`` leave the stream open
    and silent at the same two points instead. ``error_event`` sends a stream
    ``error`` event instead of a terminal response. ``after_terminal`` events
    follow the terminal one.
    """

    output: List[Dict[str, Any]]
    response_id: str
    model: str = SERVED
    status: str = "completed"
    error: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "input_tokens": 120,
        "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 0},
        "output_tokens": 30,
        "output_tokens_details": {"reasoning_tokens": 12},
        "total_tokens": 150,
    })
    disconnect_after_created: bool = False
    disconnect_after_terminal: bool = False
    stall_after_created: bool = False
    stall_after_terminal: bool = False
    error_event: Optional[Dict[str, Any]] = None
    after_terminal: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def response(self, status: str, output: List[Dict[str, Any]]) -> Dict[str, Any]:
        body = {
            "id": self.response_id, "object": "response", "created_at": 0,
            "model": self.model, "status": status, "output": output,
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        }
        if status != "in_progress" and self.usage is not None:
            body["usage"] = self.usage
        if self.error is not None and status == "failed":
            body["error"] = self.error
        if status == "incomplete":
            body["incomplete_details"] = {"reason": "max_output_tokens"}
        body.update(self.extra if status != "in_progress" else {})
        return body

    def events(self) -> Iterator[Dict[str, Any]]:
        yield {"type": "response.created", "response": self.response("in_progress", [])}
        if self.disconnect_after_created or self.stall_after_created:
            return
        if self.error_event is not None:
            yield {"type": "error", **self.error_event}
            return
        for index, item in enumerate(self.output):
            added = dict(item)
            if item["type"] == "message":
                added["content"] = []
            if item["type"] == "function_call":
                added["arguments"] = ""
            yield {"type": "response.output_item.added", "output_index": index, "item": added}
            if item["type"] == "message":
                for part in item["content"]:
                    yield {
                        "type": "response.output_text.delta", "item_id": item["id"],
                        "output_index": index, "content_index": 0, "delta": part["text"],
                        "logprobs": [],
                    }
            if item["type"] == "function_call":
                yield {
                    "type": "response.function_call_arguments.delta", "item_id": item["id"],
                    "output_index": index, "delta": item["arguments"],
                }
            yield {"type": "response.output_item.done", "output_index": index, "item": item}
        terminal = {
            "completed": "response.completed",
            "incomplete": "response.incomplete",
            "failed": "response.failed",
        }[self.status]
        yield {"type": terminal, "response": self.response(self.status, self.output)}
        yield from self.after_terminal


class _DisconnectingBody(httpx.AsyncByteStream):
    """Sends ``data``, then drops the connection. Given ``release``, it waits
    for that event first, holding the stream open; ``waiting`` is set once
    it does (#4207)."""

    def __init__(self, data: bytes, release: Optional[asyncio.Event] = None) -> None:
        self._data = data
        self._release = release
        self.waiting = asyncio.Event()

    async def __aiter__(self):
        yield self._data
        if self._release is not None:
            self.waiting.set()
            await self._release.wait()
        raise httpx.ReadError("the connection dropped mid-stream")

    async def aclose(self) -> None:
        return None


class _FailingCloseBody(httpx.AsyncByteStream):
    """Sends ``data`` whole, then fails to close: the SDK reads an ordinary
    completion's body to its end and closes it before decoding it (#4207)."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self):
        yield self._data

    async def aclose(self) -> None:
        raise OSError("synthetic close failure after the body was read")


class _HeldBody(httpx.AsyncByteStream):
    """Sends ``data`` whole once ``release`` is set; ``waiting`` is set while
    it waits (#4207)."""

    def __init__(self, data: bytes, release: asyncio.Event) -> None:
        self._data = data
        self._release = release
        self.waiting = asyncio.Event()

    async def __aiter__(self):
        self.waiting.set()
        await self._release.wait()
        yield self._data


class SilentBody(httpx.AsyncByteStream):
    """Sends ``data``, then never another byte: the stream a provider that
    stopped answering leaves open. ``stalled`` is set once everything was
    sent; ``closed`` once the client released the stream."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.stalled = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield self._data
        self.stalled.set()
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class ChatReply:
    """One scripted chat completion: its text, its id and the model the
    provider says served it. ``None`` leaves that field out of the reply.

    ``tool_arguments`` adds a call to ``lookup`` with that argument text.
    A ``usage`` of ``None`` reports no usage at all. On a stream, ``disconnect_before_usage`` ends the body with a transport
    error right after the first chunk, and ``disconnect_after_usage`` right
    after the usage chunk, before ``[DONE]``; ``stall_before_usage`` and
    ``stall_after_usage`` leave it open and silent at the same two points
    instead. An ordinary completion is one body, so either disconnect, or
    either stall, ends it before any of it arrived; ``close_fails`` sends it
    whole and then fails to close it, before it can decode.
    """

    text: str
    response_id: Optional[str]
    model: Optional[str] = CHAT_SERVED
    usage: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "prompt_tokens": 40, "completion_tokens": 6, "total_tokens": 46,
    })
    tool_arguments: Optional[str] = None
    disconnect_before_usage: bool = False
    disconnect_after_usage: bool = False
    stall_before_usage: bool = False
    stall_after_usage: bool = False
    close_fails: bool = False

    @property
    def disconnects(self) -> bool:
        return self.disconnect_before_usage or self.disconnect_after_usage

    @property
    def stalls(self) -> bool:
        return self.stall_before_usage or self.stall_after_usage

    @property
    def _finish_reason(self) -> str:
        return "stop" if self.tool_arguments is None else "tool_calls"

    def _identified(self, body: Dict[str, Any]) -> Dict[str, Any]:
        named = {"id": self.response_id, "model": self.model}
        return {**{key: value for key, value in named.items() if value is not None}, **body}

    def completion(self) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.tool_arguments is not None:
            message["tool_calls"] = [{
                "id": "call_1", "type": "function",
                "function": {"name": "lookup", "arguments": self.tool_arguments},
            }]
        return self._identified({
            "object": "chat.completion", "created": 0, "usage": self.usage,
            "choices": [{"index": 0, "finish_reason": self._finish_reason, "message": message}],
        })

    def chunks(self) -> bytes:
        deltas: List[Dict[str, Any]] = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": self.text},
                          "finish_reason": None}]},
        ]
        if self.tool_arguments is not None:
            deltas.append({"choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "lookup", "arguments": self.tool_arguments},
            }]}}]})
        if not (self.disconnect_before_usage or self.stall_before_usage):
            deltas += [
                {"choices": [{"index": 0, "delta": {}, "finish_reason": self._finish_reason}]},
                {"choices": [], "usage": self.usage},
            ]
        data = "".join(
            "data: " + json.dumps(self._identified({
                "object": "chat.completion.chunk", "created": 0, **delta,
            })) + "\n\n"
            for delta in deltas
        )
        if not (self.disconnects or self.stalls):
            data += "data: [DONE]\n\n"
        return data.encode()


def sse(events: Iterator[Dict[str, Any]]) -> bytes:
    chunks = []
    for sequence, event in enumerate(events):
        event = {**event, "sequence_number": sequence}
        chunks.append(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n")
    return "".join(chunks).encode()


class _StallingBody(httpx.AsyncByteStream):
    """Sends ``response.created`` for ``response_id``, then never another byte."""

    def __init__(self, response_id: str, model: str) -> None:
        self._created = sse(iter([{
            "type": "response.created",
            "response": Reply([], response_id, model=model).response("in_progress", []),
        }]))

    async def __aiter__(self):
        yield self._created
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        return None


class ResponsesWire:
    """The fake server. ``script(body, index)`` picks each request's reply,
    and ``chat_script(body, index)`` each chat completion's, if given.

    ``issued`` is the ``(model, response id)`` of every reply it sent, in
    order, on either endpoint: what each call was actually served.
    ``silent`` holds the body of each reply that went silent, in order.

    Given ``release``, a reply that disconnects holds its stream open after
    its last event, and an ordinary chat completion holds its whole body,
    until that event is set; ``held`` holds each such body, in order, so
    calls can be made to overlap (#4207).
    """

    def __init__(
        self,
        script: Callable[[Dict[str, Any], int], Reply],
        chat_script: Optional[Callable[[Dict[str, Any], int], ChatReply]] = None,
        *,
        release: Optional[asyncio.Event] = None,
    ) -> None:
        self.script = script
        self.chat_script = chat_script
        self.release = release
        self.requests: List[httpx.Request] = []
        self.unexpected: List[str] = []
        self.issued: List[Any] = []
        self.silent: List[SilentBody] = []
        self.held: List[Any] = []

    def _disconnecting(self, data: bytes) -> _DisconnectingBody:
        body = _DisconnectingBody(data, self.release)
        if self.release is not None:
            self.held.append(body)
        return body

    def bodies(self, path: str = "/v1/responses") -> List[Dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]

    def raw(self, path: str = "/v1/responses") -> List[bytes]:
        return [r.content for r in self.requests if r.url.path == path]

    def requests_bytes(self) -> List[bytes]:
        """Every request body this wire received, on any path."""
        return [r.content for r in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v1/embeddings":
            # No embedding model: stored rows stay searchable by keyword only.
            return httpx.Response(404, json={"error": {"message": "not found"}})
        if request.url.path == "/v1/chat/completions" and self.chat_script is not None:
            body = json.loads(request.content)
            chat = self.chat_script(body, len(self.bodies("/v1/chat/completions")) - 1)
            self.issued.append((chat.model, chat.response_id))
            if body.get("stream"):
                headers, data = {"content-type": "text/event-stream"}, chat.chunks()
            else:
                headers, data = {"content-type": "application/json"}, b""
                if not (chat.disconnects or chat.stalls):
                    data = json.dumps(chat.completion()).encode()
            if chat.disconnects:
                return httpx.Response(200, headers=headers, stream=self._disconnecting(data))
            if chat.stalls:
                silent = SilentBody(data)
                self.silent.append(silent)
                return httpx.Response(200, headers=headers, stream=silent)
            if chat.close_fails and not body.get("stream"):
                return httpx.Response(200, headers=headers, stream=_FailingCloseBody(data))
            if self.release is not None and not body.get("stream"):
                held = _HeldBody(data, self.release)
                self.held.append(held)
                return httpx.Response(200, headers=headers, stream=held)
            return httpx.Response(200, headers=headers, content=data)
        if request.url.path != "/v1/responses":
            self.unexpected.append(f"{request.method} {request.url.path}")
            return httpx.Response(404, json={"error": {"message": "not found"}})
        body = json.loads(request.content)
        reply = self.script(body, len(self.bodies()) - 1)
        self.issued.append((reply.model, reply.response_id))
        data = sse(reply.events())
        headers = {"content-type": "text/event-stream"}
        if reply.disconnect_after_created or reply.disconnect_after_terminal:
            return httpx.Response(200, headers=headers, stream=self._disconnecting(data))
        if reply.stall_after_created or reply.stall_after_terminal:
            silent = SilentBody(data)
            self.silent.append(silent)
            return httpx.Response(200, headers=headers, stream=silent)
        return httpx.Response(200, headers=headers, content=data)

    def client(self) -> openai.AsyncOpenAI:
        """A real SDK client over this wire."""
        return openai.AsyncOpenAI(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self)),
            max_retries=0,
        )


class StallingWire(ResponsesWire):
    """A Responses endpoint that goes silent right after ``response.created``:
    the stream a provider that stops answering leaves open."""

    def __init__(self, response_id: str = "resp_stall", model: str = SERVED) -> None:
        super().__init__(lambda body, index: Reply([], response_id, model=model))
        self.response_id = response_id
        self.model = model

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/responses":
            return super().__call__(request)
        self.requests.append(request)
        self.issued.append((self.model, self.response_id))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            stream=_StallingBody(self.response_id, self.model),
        )


def tool_names(body: Dict[str, Any]) -> List[str]:
    return [tool["name"] for tool in body.get("tools") or ()]


def input_types(body: Dict[str, Any]) -> List[str]:
    return [item.get("type") for item in body["input"]]


def wire_items(raw_body: bytes, key: str = "input") -> List[str]:
    """Each element of one request's ``key`` array in canonical JSON form.

    Parsed from the bytes the SDK sent and re-encoded compactly, so key order
    and every string value (a tool call's argument text included) are kept,
    but the encoding's whitespace and escapes are normalized. Two lists of
    these are equal when the items are canonically equal, not when their
    bytes are; compare :meth:`ResponsesWire.raw` for bytes.
    """
    return [
        json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        for item in json.loads(raw_body)[key]
    ]
