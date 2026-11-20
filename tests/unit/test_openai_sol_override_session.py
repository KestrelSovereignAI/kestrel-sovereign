"""A one-turn gpt-6.2-sol override with a tool, then the session goes on (#4198).

The acceptance a reviewer runs on an isolated instance, here offline: a real
agent on real storage, the real ``LLMService`` route walk and the real
``OpenAIAdapter`` and ``openai`` SDK, over ``httpx.MockTransport``.

1. A native turn with an exact ``openai:api/gpt-6.2-sol`` override must call
   one permitted tool. The tool's value is minted when the tool runs, so it
   is not in the prompt or the conversation. The turn goes to the Responses
   API, the tool's real output reaches the provider with the reasoning that
   asked for it, and the reply uses it.
2. The next turn in the same session, without an override, is served by the
   configured default (the mandate is untouched) and its request carries the
   value in the stored session's history.

   The agent's long-term memory spans sessions, so a fresh session on the
   same host may legitimately recall the value, inside the labelled
   ``<memories>`` block and never its own history. That is recorded as an
   observation, not a control. The negative control is a second, isolated
   host booted the same way before the value was minted, normal memory and
   configured default intact: a fresh session there, asked the identical
   question, carries the value nowhere in what it sends and cannot answer
   with it. Both mocks answer from the request's complete input, memories
   and tool results included, so a value the model was sent is never
   reported as unknown.

Every model call records the provider's own word on what served it.
"""

from __future__ import annotations

import functools
import json
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import httpx
import openai
import pytest

from kestrel_sovereign.llm.provider_served import PROVIDER_SERVED_KEY

OVERRIDE = "openai:api/gpt-6.2-sol"
SOL = "gpt-6.2-sol"
SOL_SERVED = "gpt-6.2-sol-2026-11-01"
DEFAULT = "gpt-5.5"
DEFAULT_SERVED = "gpt-5.5-2026-01-01"
FRESH_RE = re.compile(r"fresh-[0-9a-f]{32}")
ENCRYPTED = "gAAAAB-encrypted-sol-reasoning"
MEMORIES_RE = re.compile(r"<memories>.*?</memories>", re.DOTALL)


def _session_text(messages: List[Dict[str, Any]]) -> str:
    """``messages`` with each one's cross-session ``<memories>`` block taken
    out: what the session itself carries."""
    texts = []
    for message in messages:
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        texts.append(MEMORIES_RE.sub("", text or ""))
    return "\n".join(texts)


def _sse(events: List[Dict[str, Any]], *, named: bool) -> bytes:
    lines = []
    for event in events:
        prefix = f"event: {event['type']}\n" if named else ""
        lines.append(f"{prefix}data: {json.dumps(event)}\n\n")
    if not named:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


class _Wire:
    """A mock OpenAI API that answers from what it is sent, never from a
    script that already knows the tool's value."""

    def __init__(self) -> None:
        self.requests: List[httpx.Request] = []
        self.responses = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/v1/responses":
            return self._responses(json.loads(request.content))
        if path == "/v1/chat/completions":
            return self._chat(json.loads(request.content))
        return httpx.Response(404, json={"error": {"message": "not found"}})

    def _responses(self, body: Dict[str, Any]) -> httpx.Response:
        self.responses += 1
        response_id = f"resp_sol_{self.responses}"
        outputs = [item for item in body["input"] if item.get("type") == "function_call_output"]
        if not outputs:
            output = [
                {"type": "reasoning", "id": "rs_1", "summary": [],
                 "encrypted_content": ENCRYPTED},
                {"type": "function_call", "id": "fc_1", "call_id": "call_sol_1",
                 "name": "state_of_mind", "arguments": "{}"},
            ]
        else:
            found = FRESH_RE.search(json.dumps(outputs))
            text = f"The tool reported {found.group()}." if found else "The tool reported nothing."
            output = [{"type": "message", "id": "msg_1", "role": "assistant",
                       "phase": "final_answer",
                       "content": [{"type": "output_text", "text": text, "annotations": []}]}]
        response = {"id": response_id, "object": "response", "created_at": 1,
                    "model": SOL_SERVED, "status": "completed", "output": output,
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
        if not body.get("stream"):
            return httpx.Response(200, json=response)
        events = [{"type": "response.created",
                   "response": {**response, "status": "in_progress", "output": []}}]
        for index, item in enumerate(output):
            events.append({"type": "response.output_item.added", "output_index": index, "item": item})
            if item["type"] == "message":
                events.append({"type": "response.output_text.delta", "output_index": index,
                               "item_id": item["id"], "content_index": 0,
                               "delta": item["content"][0]["text"]})
            events.append({"type": "response.output_item.done", "output_index": index, "item": item})
        events.append({"type": "response.completed", "response": response})
        return httpx.Response(200, content=_sse(events, named=True),
                              headers={"content-type": "text/event-stream"})

    def _chat(self, body: Dict[str, Any]) -> httpx.Response:
        # The whole input, as the model would read it: a value anywhere in
        # it, retrieved memories included, is a value the model was given.
        found = FRESH_RE.search(json.dumps(body["messages"]))
        text = (
            f"Earlier this conversation the tool reported {found.group()}."
            if found else "No tool value appears in this conversation."
        )
        model = DEFAULT_SERVED if body["model"] == DEFAULT else body["model"]
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if not body.get("stream"):
            return httpx.Response(200, json={
                "id": "chatcmpl_default", "object": "chat.completion", "created": 1,
                "model": model, "usage": usage, "choices": [{"index": 0,
                "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            })
        chunk = {"id": "chatcmpl_default", "object": "chat.completion.chunk",
                 "created": 1, "model": model}
        events = [
            {**chunk, "choices": [{"index": 0, "delta": {"content": text}}]},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {**chunk, "choices": [], "usage": usage},
        ]
        return httpx.Response(200, content=_sse(events, named=False),
                              headers={"content-type": "text/event-stream"})

    def generations(self) -> List[tuple]:
        """``(path, body)`` of each generation request, in order."""
        return [
            (request.url.path, json.loads(request.content)) for request in self.requests
            if request.url.path in ("/v1/responses", "/v1/chat/completions")
        ]


def _openai_route(agent, wire: _Wire) -> dict:
    """A real openai:api route, built as the registry builds one, whose
    client is the real SDK over ``wire``."""
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    info = ProviderRegistry({})._build_route(
        "openai", "api", {"is_cloud": True},
        {"adapter": "OpenAIAdapter", "api_key": "offline-transport-only", "model": DEFAULT},
    )
    [provider] = agent.llm_service._convert_providers_format([info])
    provider["client"] = openai.AsyncOpenAI(
        api_key="offline-transport-only",
        base_url="https://openai.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(wire)),
        max_retries=0,
    )
    return provider


@asynccontextmanager
async def _agent_on(wire, tmp_path, monkeypatch, *, name: str, provenance: str):
    """A booted agent whose only route is openai:api over ``wire``, with
    ``DEFAULT`` as its configured model."""
    from tests.shared.booted_agent import booted_agent

    async with booted_agent(
        tmp_path, name=name, provenance=provenance, features={"StateOfMindFeature"},
    ) as agent:
        service = agent.llm_service
        provider = _openai_route(agent, wire)
        monkeypatch.setattr(service, "providers", [provider])
        monkeypatch.setattr(service, "config", {
            **(service.config or {}), "route_priority": [provider["name"]],
        })
        monkeypatch.setattr(service, "_mandate_preference", {
            "vendor": "openai", "route": "api", "model": DEFAULT,
        })
        yield agent


async def _turn(agent, path: str, prompt: str, session_id: str, **kwargs) -> str:
    if path == "stream":
        chunks = [
            chunk async for chunk in agent.process_input_streaming(
                prompt, session_id=session_id, request_id=f"{session_id}-{prompt}", **kwargs,
            )
        ]
        return "".join(chunk for chunk in chunks if isinstance(chunk, str))
    return await agent.process_input(prompt, session_id=session_id, **kwargs)


FOLLOW_UP = "What did the tool report earlier?"


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize("path", ["invoke", "stream"])
async def test_a_sol_override_turn_uses_its_tool_and_the_session_keeps_the_value(
    tmp_path, monkeypatch, record_property, path,
):
    from kestrel_sdk.tools import ToolResult

    from kestrel_sovereign.features.state_of_mind import StateOfMindFeature

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wire, control_wire = _Wire(), _Wire()
    minted: List[str] = []
    real_tool = StateOfMindFeature.get_state_of_mind

    @functools.wraps(real_tool)
    async def state_of_mind_with_fresh_value(self):
        # The real tool runs; its result gains a value minted only when the
        # tool executes, so neither the prompt nor the history can hold it.
        result = await real_tool(self)
        if not minted:
            minted.append(f"fresh-{uuid.uuid4().hex}")
        return ToolResult.ok(
            confirmation=f"{result.confirmation}\nProbe value: {minted[0]}",
            data={**(result.data or {}), "probe": minted[0]},
        )

    monkeypatch.setattr(StateOfMindFeature, "get_state_of_mind", state_of_mind_with_fresh_value)
    # Two isolated hosts, booted the same way before the value exists: the
    # second is the negative control, and never sees the first's turns.
    async with _agent_on(
        wire, tmp_path / "host", monkeypatch,
        name="Sol", provenance="test:openai_sol_override_session",
    ) as agent, _agent_on(
        control_wire, tmp_path / "control-host", monkeypatch,
        name="Sol", provenance="test:openai_sol_override_session",
    ) as control_agent:
        assert not minted
        service = agent.llm_service
        configured = dict(service._mandate_preference)
        session_id = f"sol-session-{path}"

        first = await _turn(
            agent, path, "Call state_of_mind once and tell me what it reports.",
            session_id, model_override=OVERRIDE,
        )
        after_override = dict(service._mandate_preference)
        second = await _turn(agent, path, FOLLOW_UP, session_id)
        same_host_fresh = await _turn(agent, path, FOLLOW_UP, f"{session_id}-fresh")
        control = await _turn(control_agent, path, FOLLOW_UP, f"{session_id}-control")
        calls = await agent.observability_store.query_llm_calls(limit=50)
        control_calls = await control_agent.observability_store.query_llm_calls(limit=50)
        stored = await agent.storage.get_conversation_history(session_id=session_id)

    [fresh] = minted
    generations = wire.generations()
    paths = [generation_path for generation_path, _ in generations]
    # Turn 1: two Responses API calls (the tool call, then the answer).
    assert paths[:2] == ["/v1/responses", "/v1/responses"], paths
    asked, followed_up = generations[0][1], generations[1][1]
    assert asked["model"] == SOL and asked["store"] is False
    offered = [tool["name"] for tool in asked["tools"]]
    assert "state_of_mind" in offered or "state_of_mind_feature" in offered, offered
    assert all(tool["type"] == "function" and tool["strict"] is False for tool in asked["tools"])
    assert fresh not in json.dumps(asked), "the value was available before the tool ran"
    replayed = [item for item in followed_up["input"] if item.get("type") in (
        "reasoning", "function_call", "function_call_output")]
    assert [item["type"] for item in replayed] == [
        "reasoning", "function_call", "function_call_output",
    ]
    assert replayed[0]["encrypted_content"] == ENCRYPTED
    assert replayed[1]["call_id"] == replayed[2]["call_id"] == "call_sol_1"
    assert fresh in json.dumps(replayed[2]["output"])
    assert f"The tool reported {fresh}." in first
    # The override changed no configured default.
    assert after_override == configured
    # Turn 2 and the same host's fresh session: the configured default, on
    # chat completions.
    rest = generations[2:]
    assert [generation_path for generation_path, _ in rest] == [
        "/v1/chat/completions", "/v1/chat/completions",
    ], paths
    assert [body["model"] for _, body in rest] == [DEFAULT, DEFAULT]
    # Turn 2 has the value from the session's own history, not only memory.
    assert fresh in _session_text(rest[0][1]["messages"])
    assert fresh in second
    # The same host's fresh session: its own history cannot hold the value,
    # but long-term memory may recall it across sessions. Observed, not
    # asserted either way: wherever it arrived, the model was given it.
    recalled = fresh in json.dumps(rest[1][1]["messages"])
    record_property("same_host_fresh_session_input_had_value", recalled)
    record_property("same_host_fresh_session_answer_had_value", fresh in same_host_fresh)
    assert fresh not in _session_text(rest[1][1]["messages"])
    assert (fresh in same_host_fresh) is recalled
    # The negative control: the isolated host's fresh session, the identical
    # question, the configured default, and the value in nothing it sent.
    control_input_had_value = any(
        fresh in request.content.decode() for request in control_wire.requests
    )
    record_property("independent_control_input_had_value", control_input_had_value)
    record_property("independent_control_answer_had_value", fresh in control)
    control_generations = control_wire.generations()
    assert [
        (generation_path, body["model"]) for generation_path, body in control_generations
    ] == [("/v1/chat/completions", DEFAULT)], control_generations
    assert FOLLOW_UP in json.dumps(control_generations[0][1]["messages"])
    assert not control_input_had_value
    assert fresh not in control
    # The session holds both turns.
    assert sum(fresh in (row.get("content") or "") for row in stored) >= 2
    # Every turn's calls record the provider's own word on what served them.
    served = [
        (call.model, call.metadata.get(PROVIDER_SERVED_KEY))
        for call in reversed(calls)
        if call.session_id in (session_id, f"{session_id}-fresh")
        and call.model in (SOL, DEFAULT)
    ]
    assert served == [
        (SOL, {"api": "responses", "model": SOL_SERVED, "response_id": "resp_sol_1"}),
        (SOL, {"api": "responses", "model": SOL_SERVED, "response_id": "resp_sol_2"}),
        (DEFAULT, {"api": "chat.completions", "model": DEFAULT_SERVED,
                   "response_id": "chatcmpl_default"}),
        (DEFAULT, {"api": "chat.completions", "model": DEFAULT_SERVED,
                   "response_id": "chatcmpl_default"}),
    ], served
    assert [
        (call.model, call.metadata.get(PROVIDER_SERVED_KEY))
        for call in control_calls
        if call.session_id == f"{session_id}-control" and call.model in (SOL, DEFAULT)
    ] == [
        (DEFAULT, {"api": "chat.completions", "model": DEFAULT_SERVED,
                   "response_id": "chatcmpl_default"}),
    ]


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize("path", ["invoke", "stream"])
async def test_the_tool_continuation_resends_the_first_requests_whole_input(
    tmp_path, monkeypatch, path,
):
    """The provider keeps a reasoning item only when everything before it is
    what it was produced after. A turn's first request can carry an operator
    notice after the user's message, as an inline ``system`` message (#2530).
    The tool continuation sends that request's whole input back unchanged,
    notice included, then the reasoning, the call and its output."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wire = _Wire()
    async with _agent_on(
        wire, tmp_path, monkeypatch,
        name="SolNotice", provenance="test:openai_sol_continuation_prefix",
    ) as agent:
        agent.operator_signal_producer.enqueue_auto_mode("session")
        await _turn(
            agent, path, "Call state_of_mind once and tell me what it reports.",
            f"sol-notice-{path}", model_override=OVERRIDE,
        )

    generations = wire.generations()
    assert [generation_path for generation_path, _ in generations[:2]] == [
        "/v1/responses", "/v1/responses",
    ], generations
    asked, followed_up = generations[0][1]["input"], generations[1][1]["input"]
    # The notice went out inline, after the user's message.
    notices = [
        index for index, item in enumerate(asked)
        if item.get("role") == "system" and "auto-mode is now enabled" in json.dumps(item)
    ]
    assert notices and notices[0] > 0, asked
    assert asked[notices[0] - 1].get("role") == "user", asked
    # The continuation starts with exactly that input.
    assert followed_up[:len(asked)] == asked
    assert [item.get("type") for item in followed_up[len(asked):]] == [
        "reasoning", "function_call", "function_call_output",
    ], followed_up[len(asked):]


class _FailingWire:
    """A mock OpenAI API whose Responses API accepts the request, then fails
    the response: it names its model and id and counts its tokens."""

    USAGE = {"input_tokens": 40, "output_tokens": 7, "total_tokens": 47}
    ERROR = {"code": "server_error", "message": "The model failed to respond."}

    def __init__(self) -> None:
        self.paths: List[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path != "/v1/responses":
            return httpx.Response(404, json={"error": {"message": "not found"}})
        body = json.loads(request.content)
        failed = {"id": "resp_sol_failed", "object": "response", "created_at": 1,
                  "model": SOL_SERVED, "status": "failed", "output": [],
                  "error": self.ERROR, "usage": self.USAGE}
        if not body.get("stream"):
            return httpx.Response(200, json=failed)
        events = [
            {"type": "response.created",
             "response": {**failed, "status": "in_progress", "error": None, "usage": None}},
            {"type": "response.failed", "response": failed},
        ]
        return httpx.Response(200, content=_sse(events, named=True),
                              headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize("path", ["invoke", "stream"])
async def test_a_failed_sol_response_is_recorded_with_what_served_it(
    tmp_path, monkeypatch, path,
):
    """A response the provider accepted and then failed is a failed attempt
    recorded with the provider's own word: the model and response id that
    answered and the tokens it counted, beside the provider's error."""
    from tests.shared.booted_agent import booted_agent

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wire = _FailingWire()
    async with booted_agent(
        tmp_path, name="SolFails", provenance="test:openai_sol_failed_response",
        features={"StateOfMindFeature"},
    ) as agent:
        service = agent.llm_service
        provider = _openai_route(agent, wire)
        monkeypatch.setattr(service, "providers", [provider])
        monkeypatch.setattr(service, "config", {
            **(service.config or {}), "route_priority": [provider["name"]],
        })
        monkeypatch.setattr(service, "_mandate_preference", {
            "vendor": "openai", "route": "api", "model": DEFAULT,
        })
        session_id = f"sol-failed-{path}"
        with pytest.raises(Exception) as raised:
            reply = await _turn(
                agent, path, "Call state_of_mind once.", session_id,
                model_override=OVERRIDE,
            )
            pytest.fail(f"the failed response became a reply: {reply!r}")
        calls = await agent.observability_store.query_llm_calls(limit=50)

    assert "The model failed to respond." in str(raised.value), raised.value
    # One attempt, never resent (the agent's memory also embeds through this
    # route; those calls are not generations).
    assert [
        sent for sent in wire.paths if sent in ("/v1/responses", "/v1/chat/completions")
    ] == ["/v1/responses"]
    [failed] = [call for call in calls if call.model == SOL]
    assert failed.success is False
    assert "The model failed to respond." in (failed.error_message or "")
    assert failed.metadata.get(PROVIDER_SERVED_KEY) == {
        "api": "responses", "model": SOL_SERVED, "response_id": "resp_sol_failed",
    }
    assert failed.metadata.get("usage_available", True) is True
    assert failed.metadata.get("stop_reason") == "failed"
    assert (failed.input_tokens, failed.output_tokens) == (40, 7)


class _DroppedBody(httpx.AsyncByteStream):
    """An SSE body that sends ``data`` and then loses its connection."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    async def __aiter__(self):
        yield self.data
        raise httpx.ReadError("connection lost mid-response")

    async def aclose(self) -> None:
        return None


class _BrokenWire:
    """A mock OpenAI API whose Responses API answers, then breaks: a stream
    names its model and id and streams a little, then its connection drops;
    a non-streamed response arrives whole but cannot be read."""

    USAGE = {"input_tokens": 40, "output_tokens": 7, "total_tokens": 47}

    def __init__(self) -> None:
        self.paths: List[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path != "/v1/responses":
            return httpx.Response(404, json={"error": {"message": "not found"}})
        response = {"id": "resp_sol_broken", "object": "response", "created_at": 1,
                    "model": SOL_SERVED, "status": "in_progress", "output": [],
                    "usage": None}
        if not json.loads(request.content).get("stream"):
            unreadable = [{"type": "message", "id": "msg_1", "role": "assistant",
                           "status": "completed", "content": 5}]
            return httpx.Response(200, json={
                **response, "status": "completed", "output": unreadable, "usage": self.USAGE,
            })
        item = {"type": "message", "id": "msg_1", "role": "assistant",
                "status": "in_progress", "content": []}
        events = [
            {"type": "response.created", "response": response},
            {"type": "response.output_item.added", "output_index": 0, "item": item},
            {"type": "response.output_text.delta", "output_index": 0, "item_id": "msg_1",
             "content_index": 0, "delta": "Partial"},
        ]
        return httpx.Response(200, stream=_DroppedBody(_sse(events, named=True)),
                              headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize("path", ["invoke", "stream"])
async def test_a_broken_sol_response_is_recorded_with_what_served_it(
    tmp_path, monkeypatch, path,
):
    """A Responses call that breaks after the provider named what served it
    (a stream whose connection drops after ``response.created``, a response
    that arrived but cannot be read) is a failed attempt recorded with the
    model and response id the provider gave. The caller sees the original
    failure: the SDK's own connection error for the dropped stream."""
    from tests.shared.booted_agent import booted_agent

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wire = _BrokenWire()
    async with booted_agent(
        tmp_path, name="SolBreaks", provenance="test:openai_sol_broken_response",
        features={"StateOfMindFeature"},
    ) as agent:
        service = agent.llm_service
        provider = _openai_route(agent, wire)
        monkeypatch.setattr(service, "providers", [provider])
        monkeypatch.setattr(service, "config", {
            **(service.config or {}), "route_priority": [provider["name"]],
        })
        monkeypatch.setattr(service, "_mandate_preference", {
            "vendor": "openai", "route": "api", "model": DEFAULT,
        })
        session_id = f"sol-broken-{path}"
        with pytest.raises(Exception) as raised:
            reply = await _turn(
                agent, path, "Call state_of_mind once.", session_id,
                model_override=OVERRIDE,
            )
            pytest.fail(f"the broken response became a reply: {reply!r}")
        calls = await agent.observability_store.query_llm_calls(limit=50)

    assert [
        sent for sent in wire.paths if sent in ("/v1/responses", "/v1/chat/completions")
    ] == ["/v1/responses"]
    [failed] = [call for call in calls if call.model == SOL]
    assert failed.success is False
    assert failed.metadata.get(PROVIDER_SERVED_KEY) == {
        "api": "responses", "model": SOL_SERVED, "response_id": "resp_sol_broken",
    }
    if path == "stream":
        # The route loop states its own verdict on a stream that failed after
        # output; the SDK's own error is the one it explicitly wraps.
        chain, pending = [], [raised.value]
        while pending:
            error = pending.pop()
            chain.append(error)
            pending.extend(link for link in (
                error.__cause__, getattr(error, "underlying", None),
            ) if link is not None)
        assert any(type(error) is openai.APIConnectionError for error in chain), chain
        assert "Connection error." in str(raised.value)
        assert failed.error_message == "Connection error."
        # No terminal event arrived, so no counts were reported.
        assert failed.metadata.get("usage_available") is False
        assert (failed.input_tokens, failed.output_tokens) == (None, None)
    else:
        assert (failed.input_tokens, failed.output_tokens) == (40, 7)
