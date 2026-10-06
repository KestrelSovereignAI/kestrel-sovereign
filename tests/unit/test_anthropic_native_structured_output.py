"""#3492: Anthropic structured output uses the native ``output_config`` format.

The adapter used to implement ``response_format`` by forcing a synthetic tool
with ``tool_choice``. Claude Opus 5.5 and Sonnet 5.5 refuse a forced tool
choice outright (``tool_choice: type "tool" and "any" are not supported for
this model``) and do not allow thinking to be disabled, so every structured
call to them failed. Measured live on 2026-10-06, ``output_config.format``
with a JSON schema is accepted by every model in the Anthropic catalog.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock

import anthropic
import pytest
from pydantic import BaseModel, Field

from kestrel_sdk.llm import StructuredOutputMode
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter
from tests.utils.anthropic_client import (
    FORCED_TOOL_CHOICE_REFUSAL,
    anthropic_client,
    forced_tool_refusing_client,
    models_api,
)

USER = [{"role": "user", "content": "Audit: 'The capital of France is Paris.'"}]
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]


class Verdict(BaseModel):
    risk_level: int = Field(ge=1, le=3, description="Risk 1-3")
    reasoning: str


def _message(blocks: List[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        content=blocks,
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=5, output_tokens=10),
    )


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def test_both_anthropic_routes_advertise_native_json_schema_output():
    for adapter in (AnthropicAdapter(), ClaudeMaxAdapter()):
        caps = adapter.provider_capabilities()
        assert caps.supports_structured_output is True
        assert caps.structured_output_mode == StructuredOutputMode.JSON_SCHEMA


@pytest.mark.asyncio
async def test_response_format_requests_native_output_not_a_forced_tool():
    payload = {"risk_level": 1, "reasoning": "fine"}
    client = anthropic_client(_message([_text(json.dumps(payload))]))

    response = await AnthropicAdapter().get_response(
        client=client,
        model="claude-sonnet-5-5",
        messages=USER,
        tools=TOOLS,
        response_format=Verdict,
    )

    sent = client.messages.stream.call_args.kwargs
    assert "tool_choice" not in sent
    assert "tools" not in sent
    assert sent["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": anthropic.transform_schema(Verdict),
        }
    }
    schema = sent["output_config"]["format"]["schema"]
    # The API requires closed objects and rejects numeric bounds; the SDK's
    # transform closes the object and moves the bounds into the description.
    assert schema["additionalProperties"] is False
    assert "minimum" not in schema["properties"]["risk_level"]
    assert json.loads(response.content) == payload
    assert response.tool_calls is None


@pytest.mark.asyncio
async def test_structured_content_is_the_json_text_after_a_thinking_block():
    """Opus 5.5 answers with adaptive thinking on: a thinking block precedes
    the JSON text, and only the text is the structured content."""
    payload = {"risk_level": 2, "reasoning": "borderline"}
    client = forced_tool_refusing_client(payload)

    response = await ClaudeMaxAdapter().get_response(
        client=client,
        model="claude-opus-5-5",
        messages=USER,
        response_format=Verdict,
    )

    assert Verdict.model_validate_json(response.content).model_dump() == payload
    assert len(client.messages.requests) == 1


@pytest.mark.asyncio
async def test_forced_tool_refusing_double_rejects_a_forced_tool_choice():
    """The opus-5-5 stand-in really refuses what the old adapter sent, so the
    tests that pass against it prove the new request shape, not a lenient
    double."""
    client = forced_tool_refusing_client({"risk_level": 1, "reasoning": "x"})
    for choice in ({"type": "tool", "name": "output_Verdict"}, {"type": "any"}):
        with pytest.raises(anthropic.BadRequestError) as refused:
            client.messages.stream(
                model="claude-opus-5-5",
                messages=USER,
                tools=[{"name": "output_Verdict", "input_schema": {}}],
                tool_choice=choice,
            )
        assert refused.value.status_code == 400
        assert FORCED_TOOL_CHOICE_REFUSAL in str(refused.value)


class _EventStream:
    def __init__(self, events: List[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> "_EventStream":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self._events:
            yield event


def _event(event_type: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **fields)


def test_streaming_structured_output_streams_the_json_exactly_once():
    """Native output streams the JSON as ordinary text deltas. The old
    forced-tool path re-yielded the assembled tool arguments at the end; that
    must not happen now or the JSON would be shown twice."""
    events = [
        _event("message_start", message=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=7),
        )),
        _event("content_block_start", index=0,
               content_block=SimpleNamespace(type="text")),
        _event("content_block_delta", index=0,
               delta=SimpleNamespace(type="text_delta", text='{"risk_level": 1, ')),
        _event("content_block_delta", index=0,
               delta=SimpleNamespace(type="text_delta", text='"reasoning": "ok"}')),
        _event("content_block_stop", index=0),
        _event("message_delta", usage=SimpleNamespace(output_tokens=9),
               delta=SimpleNamespace(stop_reason="end_turn")),
    ]
    messages = MagicMock()
    messages.stream = MagicMock(return_value=_EventStream(events))
    client = SimpleNamespace(messages=messages, models=models_api())

    async def _run() -> List[Any]:
        return [
            item
            async for item in ClaudeMaxAdapter().get_streaming_response_with_tools(
                client=client,
                model="claude-opus-5-5",
                messages=USER,
                tools=TOOLS,
                response_format=Verdict,
            )
        ]

    items = asyncio.run(_run())

    sent = messages.stream.call_args.kwargs
    assert "tool_choice" not in sent
    assert "tools" not in sent
    assert sent["output_config"]["format"]["type"] == "json_schema"
    expected = '{"risk_level": 1, "reasoning": "ok"}'
    assert "".join(i for i in items if isinstance(i, str)) == expected
    finals = [i for i in items if isinstance(i, LLMResponse)]
    assert len(finals) == 1
    assert finals[0].content == expected
    assert finals[0].tool_calls is None
