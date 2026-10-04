"""#3355: OpenRouter's model limits come from its catalog, an unreported one
stays unknown, and a ``length`` cut is visible.

``OpenRouterAdapter.list_models`` recorded ``context_length`` with a 4,096
default, so a catalog record without one was silently treated as a 4K-window
model (and that number was persisted as if reported). These tests pin:

* ``context_length`` → ``ModelInfo.context_limit`` and
  ``top_provider.max_completion_tokens`` → ``ModelInfo.output_limit``, each
  ``None`` when the catalog does not report it;
* no output budget is invented for a request (sending the model's full
  ceiling makes OpenRouter reserve credit for all of it);
* ``finish_reason`` is carried on the response, and a ``length`` cut nobody
  budgeted is marked in the text on the non-streaming, text-streaming and
  tool-streaming paths OpenRouter shares with the OpenAI-compatible base.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import httpx
import pytest

from kestrel_sovereign.llm import openrouter_adapter as openrouter_mod
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.model_metadata import ModelInfo
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from kestrel_sovereign.llm.output_ceiling import (
    output_ceiling_notice,
    response_stop_reason,
)

MODEL = "anthropic/claude-opus-5"
MESSAGES = [{"role": "user", "content": "Answer the four design questions."}]
NOTICE = output_ceiling_notice(ceiling=None, stop_reason="length")


def _adapter(monkeypatch) -> OpenRouterAdapter:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    adapter = OpenRouterAdapter()
    adapter.base_url = "https://openrouter.example/api/v1"
    return adapter


# ---------------------------------------------------------------------------
# Catalog limits
# ---------------------------------------------------------------------------


async def _discover(monkeypatch, records: List[Dict[str, Any]]) -> Dict[str, ModelInfo]:
    real_async_client = httpx.AsyncClient

    def models_endpoint(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/models"
        return httpx.Response(200, json={"data": records})

    monkeypatch.setattr(
        openrouter_mod.httpx, "AsyncClient",
        lambda **kw: real_async_client(transport=httpx.MockTransport(models_endpoint), **kw),
    )
    models = await _adapter(monkeypatch).list_models()
    return {m.id: m for m in models}


@pytest.mark.asyncio
async def test_catalog_limits_land_on_model_info(monkeypatch):
    models = await _discover(monkeypatch, [{
        "id": MODEL, "name": "Claude Opus 5", "context_length": 1_000_000,
        "top_provider": {"context_length": 1_000_000, "max_completion_tokens": 128_000},
    }])
    info = models[MODEL]
    assert info.context_limit == 1_000_000
    assert info.output_limit == 128_000
    assert ModelInfo.from_dict(info.to_dict()).output_limit == 128_000


@pytest.mark.asyncio
async def test_unreported_limits_are_unknown_not_4096(monkeypatch):
    """The issue's pin: no ``context_length`` is an unknown window, never a
    4,096-token one. The same holds for null/zero values and for a missing
    or null output ceiling."""
    models = await _discover(monkeypatch, [
        {"id": "a/no-limits"},
        {"id": "b/null-limits", "context_length": None,
         "top_provider": {"max_completion_tokens": None}},
        {"id": "c/zero-limits", "context_length": 0, "top_provider": None},
    ])
    for model_id in ("a/no-limits", "b/null-limits", "c/zero-limits"):
        assert models[model_id].context_limit is None, model_id
        assert models[model_id].output_limit is None, model_id


# ---------------------------------------------------------------------------
# Requests and the length cut
# ---------------------------------------------------------------------------


def _completion(content: str, finish_reason: Any) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=None,
    )


def _client(response: Any = None, chunks: List[Any] = None) -> SimpleNamespace:
    async def _create(**kwargs):
        if kwargs.get("stream"):
            async def _iterate():
                for chunk in chunks or []:
                    yield chunk
            return _iterate()
        return response

    create = AsyncMock(side_effect=_create)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _sent(client) -> Dict[str, Any]:
    return client.chat.completions.create.call_args.kwargs


@pytest.mark.asyncio
async def test_no_output_budget_is_invented(monkeypatch):
    """OpenRouter reserves credit for the whole requested output, so the
    adapter does not send the model's full ceiling: without a caller budget,
    no budget field reaches the wire."""
    client = _client(_completion("ok", "stop"))
    await _adapter(monkeypatch).get_response(client=client, model=MODEL, messages=MESSAGES)
    sent = _sent(client)
    assert "max_tokens" not in sent and "max_completion_tokens" not in sent
    assert "max_tokens" not in sent["extra_body"]


@pytest.mark.asyncio
async def test_length_cut_is_marked_incomplete(monkeypatch):
    client = _client(_completion("Partial answer", "length"))
    response = await _adapter(monkeypatch).get_response(
        client=client, model=MODEL, messages=MESSAGES,
    )
    assert response.content == f"Partial answer\n\n{NOTICE}"
    assert response_stop_reason(response) == "length"


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [
    {"max_tokens": 50},
    {"request_options": SimpleNamespace(raw={"max_completion_tokens": 50})},
    {"extra_body": {"max_tokens": 50}},
])
async def test_cut_at_the_callers_budget_keeps_its_text(monkeypatch, budget):
    client = _client(_completion("Short", "length"))
    response = await _adapter(monkeypatch).get_response(
        client=client, model=MODEL, messages=MESSAGES, **budget,
    )
    assert response.content == "Short"
    assert response_stop_reason(response) == "length"


@pytest.mark.asyncio
async def test_finished_response_is_untouched_and_records_its_reason(monkeypatch):
    client = _client(_completion("Done.", "stop"))
    response = await _adapter(monkeypatch).get_response(
        client=client, model=MODEL, messages=MESSAGES,
    )
    assert response.content == "Done."
    assert response_stop_reason(response) == "stop"


def _chunk(content: str = None, finish_reason: str = None, usage: Any = None):
    choices = []
    if content is not None or finish_reason is not None:
        delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None)
        choices = [SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    return SimpleNamespace(choices=choices, usage=usage)


async def _collect(stream) -> List[Any]:
    return [item async for item in stream]


@pytest.mark.asyncio
async def test_streamed_length_cut_ends_with_the_notice_and_terminal_mirrors_it(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12)
    client = _client(chunks=[
        _chunk("Partial "), _chunk("answer", "length"), _chunk(usage=usage),
    ])
    items = await _collect(_adapter(monkeypatch).get_streaming_response_with_tools(
        client=client, model=MODEL, messages=MESSAGES,
    ))
    streamed = "".join(i for i in items if isinstance(i, str))
    assert streamed == f"Partial answer\n\n{NOTICE}"
    terminal = items[-1]
    assert isinstance(terminal, LLMResponse)
    assert terminal.content == streamed
    assert response_stop_reason(terminal) == "length"


@pytest.mark.asyncio
async def test_text_only_stream_carries_the_notice_too(monkeypatch):
    client = _client(chunks=[_chunk("Partial answer"), _chunk("", "length")])
    items = await _collect(_adapter(monkeypatch).get_streaming_response(
        client=client, model=MODEL, messages=MESSAGES,
    ))
    assert "".join(i for i in items if isinstance(i, str)) == f"Partial answer\n\n{NOTICE}"


@pytest.mark.asyncio
async def test_streamed_finished_or_budgeted_response_has_no_notice(monkeypatch):
    adapter = _adapter(monkeypatch)
    finished = await _collect(adapter.get_streaming_response_with_tools(
        client=_client(chunks=[_chunk("Done.", "stop")]), model=MODEL, messages=MESSAGES,
    ))
    budgeted = await _collect(adapter.get_streaming_response_with_tools(
        client=_client(chunks=[_chunk("Short", "length")]), model=MODEL,
        messages=MESSAGES, max_tokens=50,
    ))
    assert "".join(i for i in finished if isinstance(i, str)) == "Done."
    assert response_stop_reason(finished[-1]) == "stop"
    assert "".join(i for i in budgeted if isinstance(i, str)) == "Short"
    assert response_stop_reason(budgeted[-1]) == "length"
