"""#3577: an Ollama attempt cut at its output cap still records its usage.

Ollama reports the tokens an attempt consumed (``prompt_eval_count``,
``eval_count``) on the same reply, or final stream chunk, whose
``done_reason="length"`` makes the adapter refuse the attempt (#3552). The
adapter used to raise before anyone read those counts, so a capped attempt was
recorded as a failure with unknown usage, and its tokens disappeared from
usage tracking, the token metrics and the metering callback.

Each case drives the real ``LLMService`` route walk over the real
``OllamaAdapter`` and ``ollama.AsyncClient``, with an in-process HTTP transport
standing in for the daemon, and reads what the attempt left in a real
observability store, the ``model_usage`` sink, the Prometheus series and the
metering callback. The attempt still fails, and its partial text is recorded
nowhere.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import httpx
import pytest

pytest.importorskip("ollama")

from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter  # noqa: E402
from kestrel_sovereign.llm.output_ceiling import (  # noqa: E402
    OutputCapReachedError,
    incomplete_generation,
)
from tests.unit.test_gate_rejected_attempt_telemetry import (  # noqa: E402,F401
    CONTEXT,
    INPUT_TOKENS,
    MODEL,
    OUTPUT_TOKENS,
    ROUTE,
    TOOLS,
    _assert_billed_once,
    _only_call,
    prometheus,
    service,
    store,
)
from tests.unit.test_ollama_output_cap import (  # noqa: E402
    _Daemon,
    _EndlessGeneration,
    _line,
)

CAP = 64
PARTIAL = "loop loop loop "


def _reply(**fields: Any) -> Dict[str, Any]:
    return {
        "model": MODEL,
        "message": {"role": "assistant", "content": PARTIAL},
        **fields,
    }


def _capped(body: Dict[str, Any]) -> httpx.Response:
    """Ollama's answer to a generation it stopped at ``num_predict``."""
    done = _reply(
        done=True,
        done_reason="length",
        prompt_eval_count=INPUT_TOKENS,
        eval_count=OUTPUT_TOKENS,
    )
    if body.get("stream"):
        # Text first; the counts and the stop reason arrive on the final
        # (done) chunk.
        return httpx.Response(
            200, content=_line(_reply(done=False)) + _line(done),
        )
    return httpx.Response(200, json=done)


#: How the turn reached Ollama, and whether ``/api/chat`` was streamed: a
#: logical non-streaming call, a logically buffered tool turn, and a text
#: stream all consume one native SDK stream, with the same output budget.
ENTRIES = {
    "generate_with_messages": (TOOLS, True),
    "stream_with_tools": (TOOLS, True),
    "stream": (None, True),
}


async def _call(llm, entry: str) -> None:
    tools, _streamed = ENTRIES[entry]
    messages = [{"role": "user", "content": "where do I work?"}]
    if entry == "generate_with_messages":
        await llm.generate_with_messages(
            messages=messages, tools=tools, invocation_context=CONTEXT,
        )
        return
    async for _ in llm.stream_with_tool_detection(
        messages=messages, tools=tools, invocation_context=CONTEXT,
    ):
        pass


def _install(llm, daemon: _Daemon) -> None:
    llm.install_route(OllamaAdapter(max_output_tokens=CAP), daemon.client())


def _cap_failure(raised: BaseException) -> OutputCapReachedError:
    found = incomplete_generation(raised)
    assert isinstance(found, OutputCapReachedError), repr(raised)
    return found


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", sorted(ENTRIES))
async def test_a_capped_attempt_fails_and_records_the_tokens_ollama_reported(
    service,  # noqa: F811 - the imported fixture
    store,  # noqa: F811 - the imported fixture
    prometheus,  # noqa: F811 - the imported fixture
    entry,
):
    daemon = _Daemon(_capped)
    _install(service, daemon)

    with pytest.raises(Exception) as raised:
        await _call(service, entry)

    # Still a failed attempt, never an answer.
    cap = _cap_failure(raised.value)
    assert cap.cap == CAP
    [request] = daemon.requests
    assert bool(request.get("stream")) is ENTRIES[entry][1]
    assert request["options"]["num_predict"] == CAP

    call = await _only_call(store)
    assert call.provider == ROUTE and call.model == MODEL
    assert call.success is False
    assert call.error_message == str(cap)
    assert call.metadata.get("stop_reason") == "length"
    assert "usage_available" not in call.metadata
    # The tokens Ollama reported for the attempt are kept...
    assert (call.input_tokens, call.output_tokens) == (INPUT_TOKENS, OUTPUT_TOKENS)
    # ...and the text it generated is not.
    assert call.response_preview is None
    assert "loop" not in repr(call)

    service._record_model_usage.assert_awaited_once()
    args, kwargs = service._record_model_usage.await_args
    assert args == (MODEL, ROUTE)
    assert kwargs["tokens"] == INPUT_TOKENS + OUTPUT_TOKENS

    stats = await store.get_llm_stats()
    assert stats["success_rate"] == 0
    assert stats["total_input_tokens"] == INPUT_TOKENS
    assert stats["total_output_tokens"] == OUTPUT_TOKENS

    assert prometheus["calls"] == [
        ({"provider": ROUTE, "model": MODEL, "success": "False"}, 1)
    ]
    assert prometheus["tokens"] == [
        ({"model": MODEL, "direction": "input"}, INPUT_TOKENS),
        ({"model": MODEL, "direction": "output"}, OUTPUT_TOKENS),
    ]
    # Ollama consumed those tokens whatever the adapter decided.
    _assert_billed_once(service)


@pytest.mark.asyncio
async def test_a_stream_cut_by_the_guard_without_counts_invents_no_usage(
    service,  # noqa: F811 - the imported fixture
    store,  # noqa: F811 - the imported fixture
    prometheus,  # noqa: F811 - the imported fixture
):
    """A daemon that ignores ``num_predict`` reports no counts before the
    adapter cuts it off: the failed attempt records unknown usage, not zero
    and not a guess."""
    endless = _EndlessGeneration()
    daemon = _Daemon(lambda _body: httpx.Response(200, stream=endless))
    _install(service, daemon)

    with pytest.raises(Exception) as raised:
        await asyncio.wait_for(_call(service, "stream"), 10)

    assert _cap_failure(raised.value).cap == CAP
    assert endless.closed

    call = await _only_call(store)
    assert call.success is False
    assert (call.input_tokens, call.output_tokens) == (None, None)
    assert call.metadata.get("usage_available") is False
    service._record_model_usage.assert_not_awaited()
    assert prometheus["tokens"] == []
    assert service.metered == []

