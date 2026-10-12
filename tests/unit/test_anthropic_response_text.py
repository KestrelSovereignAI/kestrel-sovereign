"""#3436: a non-streaming Claude response's text is every text block it has.

``AnthropicAdapter.get_response`` kept only the LAST text block, so whatever
the model wrote in an earlier block was dropped from ``LLMResponse.content``,
while the streaming paths, which join every text delta, kept it. The messages
here are the SDK's own ``Message`` type, in the shapes the plan route returns.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from anthropic.types import Message

from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter
from kestrel_sovereign.llm.output_ceiling import response_stop_reason
from tests.utils.anthropic_client import anthropic_client

USER = [{"role": "user", "content": "Which candidates answer the question?"}]


def _message(blocks: List[Dict[str, Any]], *, stop_reason: str = "end_turn") -> Message:
    return Message.model_validate({
        "id": "msg_3436",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 580, "output_tokens": 73},
    })


def _text(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text, "citations": None}


#: How the plan route returned an adaptive-thinking reply: the thinking is
#: redacted to an empty string with only its signature.
_THINKING = {"type": "thinking", "thinking": "", "signature": "c2lnbmF0dXJl"}


async def _content(adapter: AnthropicAdapter, message: Message):
    return await adapter.get_response(
        client=anthropic_client(message), model="claude-sonnet-5-5", messages=USER
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [AnthropicAdapter(), ClaudeMaxAdapter()], ids=["api", "plan"])
async def test_text_after_a_thinking_block_is_the_content(adapter):
    """The shape captured live on ``anthropic:plan`` for #3436."""
    response = await _content(
        adapter, _message([_THINKING, _text('{"answerable_ids":["c0"]}')])
    )

    assert response.content == '{"answerable_ids":["c0"]}'
    assert response.output_tokens == 73
    assert response_stop_reason(response) == "end_turn"


@pytest.mark.asyncio
async def test_every_text_block_is_kept_in_order():
    response = await _content(
        AnthropicAdapter(),
        _message([
            _text("c1 says the user moved to Austin.\n\n"),
            _THINKING,
            _text('{"answerable_ids":["c1"]}'),
        ]),
    )

    assert response.content == (
        'c1 says the user moved to Austin.\n\n{"answerable_ids":["c1"]}'
    )


@pytest.mark.asyncio
async def test_a_trailing_empty_text_block_does_not_erase_the_text():
    response = await _content(
        AnthropicAdapter(), _message([_text('{"answerable_ids":[]}'), _text("")])
    )

    assert response.content == '{"answerable_ids":[]}'


@pytest.mark.asyncio
async def test_a_response_without_text_blocks_has_no_content():
    response = await _content(
        AnthropicAdapter(),
        _message(
            [_THINKING, {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}],
            stop_reason="tool_use",
        ),
    )

    assert response.content is None
    assert [call.name for call in response.tool_calls] == ["lookup"]
