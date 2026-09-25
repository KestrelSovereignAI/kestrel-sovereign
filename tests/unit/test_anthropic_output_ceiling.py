"""#3300: a Claude turn's output ceiling is the model's own, and a response
cut at it is never presented as a finished one.

The adapter used to send ``max_tokens: 4096`` on every request and never read
``stop_reason``, so every Claude turn — including the orchestrator's — was cut
at 4,096 output tokens and the cut was invisible (a 233-second turn returned
an empty 200). These tests pin both halves:

* the ceiling comes from the provider's Models API record for THAT model,
  an explicit ``max_tokens`` still wins, and an unknown ceiling is an error,
  never a guess;
* ``stop_reason`` is carried on the response on every path, and a cut at the
  model's ceiling is marked in the text on the non-streaming path, the
  text-only streaming path and the tool-detecting streaming path.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.llm import ToolCallStarted
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.anthropic_adapter import (
    AnthropicAdapter,
    anthropic_model_info,
)
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter
from kestrel_sovereign.llm.model_metadata import ModelInfo
from kestrel_sovereign.llm.output_ceiling import (
    OutputCeilingUnknownError,
    context_window_notice,
    output_ceiling_notice,
    response_stop_reason,
)
from tests.utils.anthropic_client import (
    anthropic_client,
    anthropic_model_record,
    install_models_api,
    models_api,
)

USER = [{"role": "user", "content": "Answer the four design questions."}]


def _message(blocks: List[Any], *, stop_reason: str, output_tokens: int = 10):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=5, output_tokens=output_tokens),
    )


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use(call_id: str, name: str, arguments: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def _thinking() -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking="")


# ---------------------------------------------------------------------------
# The ceiling: sourced from the model, never a literal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_ceiling_is_the_models_reported_max_tokens_not_4096():
    """The issue's pin: a claude-opus-5 request does not send 4096 — it sends
    what the Models API reports for claude-opus-5."""
    client = anthropic_client(
        _message([_text("ok")], stop_reason="end_turn"), max_tokens=128_000,
    )
    await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER,
    )
    sent = client.messages.stream.call_args.kwargs
    assert sent["max_tokens"] == 128_000
    assert sent["max_tokens"] != 4096
    client.models.retrieve.assert_awaited_once_with("claude-opus-5")


@pytest.mark.asyncio
async def test_each_model_gets_its_own_ceiling_and_is_looked_up_once():
    """Per model, per adapter: two models with different ceilings each send
    their own, and a second request for a model reuses what it learned."""
    ceilings = {"claude-opus-5": 128_000, "claude-haiku-4-5": 64_000}
    client = anthropic_client(_message([_text("ok")], stop_reason="end_turn"))
    client.models.retrieve = AsyncMock(
        side_effect=lambda model_id, **_: anthropic_model_record(
            model_id, max_tokens=ceilings[model_id],
        )
    )
    adapter = AnthropicAdapter()

    sent = []
    for model in ("claude-opus-5", "claude-haiku-4-5", "claude-opus-5"):
        await adapter.get_response(client=client, model=model, messages=USER)
        sent.append(client.messages.stream.call_args.kwargs["max_tokens"])

    assert sent == [128_000, 64_000, 128_000]
    assert [c.args[0] for c in client.models.retrieve.await_args_list] == [
        "claude-opus-5", "claude-haiku-4-5",
    ]


@pytest.mark.asyncio
async def test_ceiling_lookup_uses_the_wire_model_id():
    """A route-prefixed model name asks the Models API for the id actually
    sent on the wire."""
    client = anthropic_client(_message([_text("ok")], stop_reason="end_turn"))
    await AnthropicAdapter().get_response(
        client=client, model="anthropic/claude-opus-5", messages=USER,
    )
    client.models.retrieve.assert_awaited_once_with("claude-opus-5")


@pytest.mark.asyncio
async def test_discovered_ceiling_is_used_without_another_lookup():
    """Model discovery already carries each model's ceiling; a request for a
    discovered model uses it rather than asking again."""
    client = anthropic_client(_message([_text("ok")], stop_reason="end_turn"))
    client.models.list = AsyncMock(return_value=SimpleNamespace(data=[
        anthropic_model_record("claude-opus-5", max_tokens=128_000),
        anthropic_model_record("claude-sonnet-5", max_tokens=64_000),
    ]))
    adapter = ClaudeMaxAdapter()
    adapter._ensure_fresh_oauth_token = AsyncMock()

    models = await adapter.list_models(client)
    await adapter.get_response(client=client, model="claude-sonnet-5", messages=USER)

    assert {m.id: m.output_limit for m in models} == {
        "claude-opus-5": 128_000, "claude-sonnet-5": 64_000,
    }
    assert client.messages.stream.call_args.kwargs["max_tokens"] == 64_000
    client.models.retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_key_discovery_seeds_the_ceiling_too(monkeypatch):
    """The API-key route discovers over plain HTTP; what it learns is used
    the same way as the plan route's discovery."""
    import httpx

    real_async_client = httpx.AsyncClient

    def models_endpoint(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [
            {"id": "claude-opus-5", "type": "model", "display_name": "Opus",
             "created_at": "2026-01-01T00:00:00Z", "max_tokens": 128_000},
        ]})

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kw: real_async_client(transport=httpx.MockTransport(models_endpoint), **kw),
    )
    adapter = AnthropicAdapter()
    await adapter.list_models()
    monkeypatch.setattr(httpx, "AsyncClient", real_async_client)

    client = anthropic_client(_message([_text("ok")], stop_reason="end_turn"))
    await adapter.get_response(client=client, model="claude-opus-5", messages=USER)
    assert client.messages.stream.call_args.kwargs["max_tokens"] == 128_000
    client.models.retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_max_tokens_wins_and_skips_the_lookup():
    client = anthropic_client(_message([_text("ok")], stop_reason="end_turn"))
    await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER, max_tokens=300,
    )
    assert client.messages.stream.call_args.kwargs["max_tokens"] == 300
    client.models.retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_request_option_max_tokens_still_wins():
    """The ``raw`` request-option escape hatch has always overridden the
    request body; it still overrides the model ceiling."""
    client = anthropic_client(_message([_text("ok")], stop_reason="end_turn"))
    await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER,
        request_options=SimpleNamespace(raw={"max_tokens": 777}),
    )
    assert client.messages.stream.call_args.kwargs["max_tokens"] == 777
    client.models.retrieve.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reported", [None, 0])
async def test_unknown_ceiling_is_a_named_error_not_a_guess(reported):
    """No reported ceiling → OutputCeilingUnknownError naming the model, and
    no request is sent with an invented number."""
    client = anthropic_client(
        _message([_text("ok")], stop_reason="end_turn"), max_tokens=reported,
    )
    with pytest.raises(OutputCeilingUnknownError) as excinfo:
        await AnthropicAdapter().get_response(
            client=client, model="claude-opus-5", messages=USER,
        )
    assert excinfo.value.model == "claude-opus-5"
    assert "claude-opus-5" in str(excinfo.value)
    client.messages.stream.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_ceiling_fails_the_streaming_paths_too():
    client = MagicMock()
    install_models_api(client, max_tokens=None)
    adapter = AnthropicAdapter()
    with pytest.raises(OutputCeilingUnknownError):
        async for _ in adapter.get_streaming_response(
            client=client, model="claude-opus-5", messages=USER,
        ):
            pass
    with pytest.raises(OutputCeilingUnknownError):
        async for _ in adapter.get_streaming_response_with_tools(
            client=client, model="claude-opus-5", messages=USER,
        ):
            pass
    client.messages.stream.assert_not_called()


def test_model_info_carries_the_reported_output_limit():
    info = anthropic_model_info(
        anthropic_model_record("claude-opus-5", max_tokens=128_000)
    )
    assert info.output_limit == 128_000
    assert info.context_limit == 1_000_000
    assert anthropic_model_info({"id": "x", "max_tokens": 8192}).output_limit == 8192
    assert anthropic_model_info({"id": "x"}).output_limit is None
    assert ModelInfo.from_dict(info.to_dict()).output_limit == 128_000


# ---------------------------------------------------------------------------
# The context window: the provider bounds a full-ceiling request itself
# ---------------------------------------------------------------------------

CONTEXT_BETA = "model-context-window-exceeded-2025-08-26"


def _betas(sent: Dict[str, Any]) -> List[str]:
    return (sent.get("extra_headers") or {}).get("anthropic-beta", "").split(",")


@pytest.mark.asyncio
async def test_full_ceiling_request_opts_into_context_window_stop_on_api_route():
    """The model's full output ceiling can exceed the room a long prompt
    leaves. The provider's documented answer is to accept the request and
    stop with ``model_context_window_exceeded`` (default on Claude 4.5+, this
    beta on earlier models) — no client-side estimate of the input."""
    client = anthropic_client(_message([_text("ok")], stop_reason="end_turn"))
    await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER,
    )
    assert CONTEXT_BETA in _betas(client.messages.stream.call_args.kwargs)


@pytest.mark.asyncio
async def test_plan_route_keeps_claude_code_request_shape():
    """The plan route is shaped exactly like Claude Code's requests, which do
    not carry the context-window beta; it is not added there."""
    client = anthropic_client(_message([_text("ok")], stop_reason="end_turn"))
    adapter = ClaudeMaxAdapter()
    adapter._ensure_fresh_oauth_token = AsyncMock()
    await adapter.get_response(client=client, model="claude-opus-5", messages=USER)
    assert CONTEXT_BETA not in _betas(client.messages.stream.call_args.kwargs)


@pytest.mark.asyncio
async def test_callers_budget_does_not_add_the_context_window_beta():
    client = anthropic_client(_message([_text("ok")], stop_reason="end_turn"))
    await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER, max_tokens=300,
    )
    assert CONTEXT_BETA not in _betas(client.messages.stream.call_args.kwargs)


CONTEXT_NOTICE = context_window_notice(stop_reason="model_context_window_exceeded")


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [{}, {"max_tokens": 300}])
async def test_non_streaming_context_window_stop_is_marked_incomplete(budget):
    """A context-window stop is never a budget anyone chose, so it is marked
    whether or not the caller set max_tokens; a trailing tool call it cut is
    dropped."""
    client = anthropic_client(_message(
        [_text("Partial answer"), _tool_use("call_1", "shell", {"command": "ls -"})],
        stop_reason="model_context_window_exceeded",
    ))
    response = await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER,
        tools=[{"type": "function", "function": {"name": "shell"}}], **budget,
    )
    assert response.content == f"Partial answer\n\n{CONTEXT_NOTICE}"
    assert response.tool_calls is None
    assert response_stop_reason(response) == "model_context_window_exceeded"


@pytest.mark.asyncio
async def test_streaming_context_window_stop_is_marked_and_cut_call_unannounced():
    events = _prose_then_cut_tool_events("Let me check the repo.")
    events[6] = _ev(
        "message_delta",
        delta=SimpleNamespace(stop_reason="model_context_window_exceeded"),
        usage=SimpleNamespace(output_tokens=900),
    )
    items = await _collect(AnthropicAdapter().get_streaming_response_with_tools(
        client=_streaming_client(events), model="claude-opus-5", messages=USER,
        tools=[{"type": "function", "function": {"name": "shell"}}],
    ))
    assert not any(isinstance(i, ToolCallStarted) for i in items)
    streamed = "".join(i for i in items if isinstance(i, str))
    assert streamed == f"Let me check the repo.\n\n{CONTEXT_NOTICE}"
    assert items[-1].content == streamed
    assert items[-1].tool_calls is None


# ---------------------------------------------------------------------------
# The cut: non-streaming
# ---------------------------------------------------------------------------


NOTICE = output_ceiling_notice(ceiling=128_000, stop_reason="max_tokens")


@pytest.mark.asyncio
async def test_non_streaming_cut_at_model_ceiling_is_marked_in_the_text():
    client = anthropic_client(
        _message([_text("Question 1: yes, because")], stop_reason="max_tokens"),
    )
    response = await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER,
    )
    assert response.content == f"Question 1: yes, because\n\n{NOTICE}"
    assert response_stop_reason(response) == "max_tokens"


@pytest.mark.asyncio
async def test_non_streaming_cut_with_no_text_is_not_an_empty_answer():
    """The #3300 incident shape: the whole output went to thinking, the text
    was empty, and the turn returned "" as if it had finished."""
    client = anthropic_client(_message([_thinking()], stop_reason="max_tokens"))
    response = await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER,
    )
    assert response.content == NOTICE


@pytest.mark.asyncio
async def test_non_streaming_cut_at_callers_budget_keeps_text_untouched():
    """A caller that chose a budget owns its truncation: no notice, but the
    stop reason still says where it stopped."""
    client = anthropic_client(_message([_text("A short tit")], stop_reason="max_tokens"))
    response = await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER, max_tokens=5,
    )
    assert response.content == "A short tit"
    assert response_stop_reason(response) == "max_tokens"


@pytest.mark.asyncio
async def test_non_streaming_finished_response_is_unchanged():
    client = anthropic_client(_message([_text("Done.")], stop_reason="end_turn"))
    response = await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER,
    )
    assert response.content == "Done."
    assert response_stop_reason(response) == "end_turn"


@pytest.mark.asyncio
async def test_non_streaming_cut_drops_only_the_incomplete_trailing_tool_call():
    """The cut lands in the LAST block; a tool call there has incomplete
    arguments and must not be executed. Earlier, complete calls survive."""
    client = anthropic_client(_message(
        [
            _tool_use("call_1", "read_file", {"path": "a.py"}),
            _tool_use("call_2", "shell", {"command": "rm -rf /tmp/wo"}),
        ],
        stop_reason="max_tokens",
    ))
    response = await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER,
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )
    assert [tc.id for tc in response.tool_calls] == ["call_1"]
    assert response.content == NOTICE


@pytest.mark.asyncio
async def test_non_streaming_tool_call_is_kept_when_the_response_finished():
    client = anthropic_client(_message(
        [_tool_use("call_1", "shell", {"command": "ls"})], stop_reason="tool_use",
    ))
    response = await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER,
        tools=[{"type": "function", "function": {"name": "shell"}}],
    )
    assert [tc.id for tc in response.tool_calls] == ["call_1"]
    assert response.content is None


# ---------------------------------------------------------------------------
# The cut: streaming
# ---------------------------------------------------------------------------


def _ev(event_type: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **fields)


class _Stream:
    def __init__(self, events: List[Any]) -> None:
        self._events = list(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


def _streaming_client(events: List[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **_: _Stream(events)),
        models=models_api(max_tokens=128_000),
    )


def _text_events(text: str, *, stop_reason: str) -> List[Any]:
    return [
        _ev("message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=5))),
        _ev("content_block_start", index=0, content_block=SimpleNamespace(type="text")),
        _ev("content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text=text)),
        _ev("content_block_stop", index=0),
        _ev(
            "message_delta",
            delta=SimpleNamespace(stop_reason=stop_reason),
            usage=SimpleNamespace(output_tokens=128_000),
        ),
        _ev("message_stop"),
    ]


async def _collect(agen) -> List[Any]:
    return [item async for item in agen]


@pytest.mark.asyncio
async def test_tool_stream_cut_at_model_ceiling_ends_with_the_notice():
    items = await _collect(AnthropicAdapter().get_streaming_response_with_tools(
        client=_streaming_client(_text_events("Question 1: yes", stop_reason="max_tokens")),
        model="claude-opus-5", messages=USER,
    ))
    chunks = [i for i in items if isinstance(i, str)]
    final = items[-1]
    assert chunks == ["Question 1: yes", f"\n\n{NOTICE}"]
    assert isinstance(final, LLMResponse)
    # The terminal response mirrors exactly what was streamed.
    assert final.content == "".join(chunks)
    assert response_stop_reason(final) == "max_tokens"


@pytest.mark.asyncio
async def test_tool_stream_finished_carries_its_stop_reason_and_no_notice():
    items = await _collect(AnthropicAdapter().get_streaming_response_with_tools(
        client=_streaming_client(_text_events("All done.", stop_reason="end_turn")),
        model="claude-opus-5", messages=USER,
    ))
    assert [i for i in items if isinstance(i, str)] == ["All done."]
    assert items[-1].content == "All done."
    assert response_stop_reason(items[-1]) == "end_turn"


@pytest.mark.asyncio
async def test_tool_stream_cut_at_callers_budget_has_no_notice():
    items = await _collect(AnthropicAdapter().get_streaming_response_with_tools(
        client=_streaming_client(_text_events("Short", stop_reason="max_tokens")),
        model="claude-opus-5", messages=USER, max_tokens=2,
    ))
    assert [i for i in items if isinstance(i, str)] == ["Short"]
    assert response_stop_reason(items[-1]) == "max_tokens"


@pytest.mark.asyncio
async def test_tool_stream_cut_drops_the_incomplete_trailing_tool_call():
    events = [
        _ev("content_block_start", index=0, content_block=SimpleNamespace(
            type="tool_use", id="call_1", name="read_file")),
        _ev("content_block_delta", index=0, delta=SimpleNamespace(
            type="input_json_delta", partial_json='{"path": "a.py"}')),
        _ev("content_block_stop", index=0),
        _ev("content_block_start", index=1, content_block=SimpleNamespace(
            type="tool_use", id="call_2", name="shell")),
        _ev("content_block_delta", index=1, delta=SimpleNamespace(
            type="input_json_delta", partial_json='{"command": "rm -rf /tmp/wo')),
        _ev("content_block_stop", index=1),
        _ev("message_delta", delta=SimpleNamespace(stop_reason="max_tokens"),
            usage=SimpleNamespace(output_tokens=128_000)),
    ]
    items = await _collect(AnthropicAdapter().get_streaming_response_with_tools(
        client=_streaming_client(events), model="claude-opus-5", messages=USER,
        tools=[{"type": "function", "function": {"name": "shell"}}],
    ))
    final = items[-1]
    assert [tc.id for tc in final.tool_calls] == ["call_1"]
    assert final.tool_calls[0].arguments == {"path": "a.py"}
    assert final.content == NOTICE
    # The cut call is never announced: only the whole one gets a marker.
    assert [i.id for i in items if isinstance(i, ToolCallStarted)] == ["call_1"]


def _prose_then_cut_tool_events(prose: str) -> List[Any]:
    return [
        _ev("content_block_start", index=0, content_block=SimpleNamespace(type="text")),
        _ev("content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text=prose)),
        _ev("content_block_stop", index=0),
        _ev("content_block_start", index=1, content_block=SimpleNamespace(
            type="tool_use", id="call_cut", name="shell")),
        _ev("content_block_delta", index=1, delta=SimpleNamespace(
            type="input_json_delta", partial_json='{"command": "rm -rf /tmp/wo')),
        _ev("content_block_stop", index=1),
        _ev("message_delta", delta=SimpleNamespace(stop_reason="max_tokens"),
            usage=SimpleNamespace(output_tokens=128_000)),
    ]


@pytest.mark.asyncio
async def test_tool_stream_prose_then_cut_tool_never_signals_the_dropped_call():
    """prose → tool_use start → max_tokens cut: no ToolCallStarted is
    yielded for a call that is then dropped, and what streamed equals the
    terminal response (prose + notice), with no tool calls."""
    items = await _collect(AnthropicAdapter().get_streaming_response_with_tools(
        client=_streaming_client(_prose_then_cut_tool_events("Let me check the repo.")),
        model="claude-opus-5", messages=USER,
        tools=[{"type": "function", "function": {"name": "shell"}}],
    ))
    assert not any(isinstance(i, ToolCallStarted) for i in items)
    streamed = "".join(i for i in items if isinstance(i, str))
    final = items[-1]
    assert streamed == f"Let me check the repo.\n\n{NOTICE}"
    assert final.content == streamed
    assert final.tool_calls is None


@pytest.mark.asyncio
async def test_tool_stream_whole_trailing_tool_call_is_still_announced():
    """The held marker is released when the stream ends on tool_use — a
    finished call is announced, after the prose, before the terminal
    response."""
    events = _prose_then_cut_tool_events("Let me check the repo.")
    events[4] = _ev("content_block_delta", index=1, delta=SimpleNamespace(
        type="input_json_delta", partial_json='{"command": "ls"}'))
    events[6] = _ev("message_delta", delta=SimpleNamespace(stop_reason="tool_use"),
                    usage=SimpleNamespace(output_tokens=20))
    items = await _collect(AnthropicAdapter().get_streaming_response_with_tools(
        client=_streaming_client(events), model="claude-opus-5", messages=USER,
        tools=[{"type": "function", "function": {"name": "shell"}}],
    ))
    kinds = [type(i).__name__ for i in items]
    assert kinds == ["str", "ToolCallStarted", "LLMResponse"]
    assert items[1].id == "call_cut"
    assert [tc.id for tc in items[-1].tool_calls] == ["call_cut"]
    assert items[-1].content == "Let me check the repo."


@pytest.mark.asyncio
async def test_text_stream_cut_at_model_ceiling_ends_with_the_notice():
    items = await _collect(AnthropicAdapter().get_streaming_response(
        client=_streaming_client(_text_events("Question 1: yes", stop_reason="max_tokens")),
        model="claude-opus-5", messages=USER,
    ))
    assert items == ["Question 1: yes", f"\n\n{NOTICE}"]


@pytest.mark.asyncio
async def test_text_stream_cut_with_no_text_yields_only_the_notice():
    events = [
        _ev("content_block_start", index=0, content_block=SimpleNamespace(type="thinking")),
        _ev("content_block_stop", index=0),
        _ev("message_delta", delta=SimpleNamespace(stop_reason="max_tokens"),
            usage=SimpleNamespace(output_tokens=128_000)),
    ]
    items = await _collect(AnthropicAdapter().get_streaming_response(
        client=_streaming_client(events), model="claude-opus-5", messages=USER,
    ))
    assert items == [NOTICE]


@pytest.mark.asyncio
async def test_text_stream_finished_has_no_notice():
    items = await _collect(AnthropicAdapter().get_streaming_response(
        client=_streaming_client(_text_events("All done.", stop_reason="end_turn")),
        model="claude-opus-5", messages=USER,
    ))
    assert items == ["All done."]


# ---------------------------------------------------------------------------
# Through the real Anthropic SDK (HTTP faked at the transport)
# ---------------------------------------------------------------------------


def _sse(events: List[Dict[str, Any]]) -> bytes:
    import json

    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


def _wire_events(text: str, *, stop_reason: str) -> List[Dict[str, Any]]:
    return [
        {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-opus-5", "content": [], "stop_reason": None,
            "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 1},
        }},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta",
         "delta": {"stop_reason": stop_reason, "stop_sequence": None},
         "usage": {"output_tokens": 128_000}},
        {"type": "message_stop"},
    ]


def _sdk_http_library():
    """The HTTP library the installed Anthropic SDK is built on.

    The 0.x SDK is built on ``httpx``; 1.x moved to ``httpx2`` and rejects an
    ``httpx`` client. The SDK's own ``DefaultAsyncHttpxClient`` subclasses
    whichever one it uses, so its base class names the library — the test
    serves its fake HTTP through that library on either SDK major version.
    """
    import importlib

    import anthropic

    for base in anthropic.DefaultAsyncHttpxClient.__mro__:
        root = base.__module__.split(".")[0]
        if root in ("httpx", "httpx2"):
            return importlib.import_module(root)
    raise AssertionError(
        "anthropic.DefaultAsyncHttpxClient is not built on httpx or httpx2"
    )


def _real_sdk_client(stop_reason: str, text: str = "Question 1: yes"):
    """A real ``anthropic.AsyncAnthropic`` whose HTTP is served locally: the
    Models API record for claude-opus-5, and a streamed message."""
    import json

    import anthropic

    httpx = _sdk_http_library()
    requests: List[Any] = []

    def handler(request):
        requests.append(request)
        if request.method == "GET" and request.url.path == "/v1/models/claude-opus-5":
            return httpx.Response(200, json={
                "id": "claude-opus-5", "type": "model", "display_name": "Claude Opus 5",
                "created_at": "2026-01-01T00:00:00Z",
                "max_input_tokens": 1_000_000, "max_tokens": 128_000,
            })
        if request.method == "POST" and request.url.path == "/v1/messages":
            assert json.loads(request.content)["stream"] is True
            return httpx.Response(
                200, content=_sse(_wire_events(text, stop_reason=stop_reason)),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, json={"type": "error", "error": {
            "type": "not_found_error", "message": request.url.path}})

    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        http_client=anthropic.DefaultAsyncHttpxClient(
            transport=httpx.MockTransport(handler),
        ),
        max_retries=0,
    )
    return client, requests


@pytest.mark.asyncio
async def test_real_sdk_non_streaming_request_at_full_ceiling_is_accepted():
    """At the model's 128,000-token ceiling the SDK refuses a non-streaming
    ``messages.create`` ("Streaming is required ..."), so ``get_response``
    carries its complete response over a stream. Driven through the real SDK
    client so its request guard, Models API parsing and stream parsing are
    the real ones."""
    import json

    client, requests = _real_sdk_client("max_tokens")
    response = await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER,
    )

    sent = json.loads(requests[-1].content)
    assert sent["max_tokens"] == 128_000
    assert CONTEXT_BETA in requests[-1].headers["anthropic-beta"].split(",")
    assert response.content == f"Question 1: yes\n\n{NOTICE}"
    assert response_stop_reason(response) == "max_tokens"
    assert response.output_tokens == 128_000


@pytest.mark.asyncio
async def test_real_sdk_streaming_with_tools_reads_the_wire_stop_reason():
    client, _ = _real_sdk_client("max_tokens")
    items = await _collect(AnthropicAdapter().get_streaming_response_with_tools(
        client=client, model="claude-opus-5", messages=USER,
    ))
    assert [i for i in items if isinstance(i, str)] == [
        "Question 1: yes", f"\n\n{NOTICE}",
    ]
    assert response_stop_reason(items[-1]) == "max_tokens"


@pytest.mark.asyncio
async def test_real_sdk_finished_response_is_untouched():
    client, _ = _real_sdk_client("end_turn", text="Done.")
    response = await AnthropicAdapter().get_response(
        client=client, model="claude-opus-5", messages=USER,
    )
    assert response.content == "Done."
    assert response_stop_reason(response) == "end_turn"


# ---------------------------------------------------------------------------
# End to end through the agent's streaming turn: live == persisted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streamed_turn_cut_on_a_tool_call_persists_what_was_shown():
    """prose → tool_use start → max_tokens cut, driven through the real
    ``StreamingMixin`` turn with the real adapter's output. The client sees
    the prose and the notice with no revise signal for the dropped call,
    and the persisted assistant row is exactly that text."""
    from contextlib import asynccontextmanager

    from kestrel_sovereign.agent.streaming import (
        StreamingMixin,
        _parse_stream_sentinels,
    )

    @asynccontextmanager
    async def _passthrough():
        yield

    persisted: List[Dict[str, Any]] = []
    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock(
        side_effect=lambda role, content, **kw: persisted.append(
            {"role": role, "content": content, **kw}
        )
    )
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    agent = MagicMock()
    agent.privacy_agent = privacy_agent
    agent.features = {}
    agent.did = "test-did"
    agent.extension = None
    agent._cached_features_prompt = ""
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent.emit_event = AsyncMock()
    agent._maybe_audit = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    agent.hooks_manager = None
    agent.operator_signal_producer = None
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="claude-opus-5")
    agent._build_all_tools = MagicMock(return_value=[])
    agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered prompt"
    context_result = MagicMock()
    context_result.system_prompt = "system"
    context_result.dynamic_user_context = "ctx"
    context_result.messages = []
    context_result.semantic_recall_dependencies = ()
    agent.context_manager = MagicMock()
    agent.context_manager.build_context = AsyncMock(return_value=context_result)
    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_metric = AsyncMock()
    agent._emit_revising_event = AsyncMock()

    adapter = AnthropicAdapter()

    async def stream_with_tool_detection(**_kwargs):
        async for item in adapter.get_streaming_response_with_tools(
            client=_streaming_client(_prose_then_cut_tool_events("Let me check the repo.")),
            model="claude-opus-5", messages=USER,
            tools=[{"type": "function", "function": {"name": "shell"}}],
        ):
            yield item

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = stream_with_tool_detection
    agent._handle_orchestrator_response_streaming = MagicMock(
        side_effect=AssertionError("a dropped tool call must not be dispatched"),
    )
    for name in (
        "process_input_streaming",
        "_process_input_streaming_traced_locked",
        "_persist_assistant_turn_safely",
    ):
        setattr(agent, name, getattr(StreamingMixin, name).__get__(agent))

    live = [chunk async for chunk in agent.process_input_streaming(
        "answer the design questions", session_id="sess-1",
    )]

    wire = "".join(c for c in live if isinstance(c, str))
    assert "\x1eKESTREL:REVISE:" not in wire
    agent._emit_revising_event.assert_not_awaited()
    shown = _parse_stream_sentinels(wire)[0]
    assert shown == f"Let me check the repo.\n\n{NOTICE}"

    rows = [row for row in persisted if row["role"] == "assistant"]
    assert len(rows) == 1
    assert rows[0]["content"] == shown
