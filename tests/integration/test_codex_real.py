"""Integration tests: CodexAdapter against the live ChatGPT-backend Responses API.

Pattern mirrors ``tests/integration/test_anthropic_cache_real.py`` (#705 / #709):
gated on credentials, auto-loads ``.env``, real httpx, prints observed metrics
to stderr.

Why these exist (#841): mock-only coverage for the Codex adapter shipped a
broken #828 fix to production — the unit tests asserted on body shape produced
by helpers that were never validated against the wire. A live gate is the only
way to catch protocol drift between Chat Completions (what the orchestrator
emits) and the Responses API (what the ChatGPT backend speaks).

Skipped when neither ``CODEX_AUTH_TOKEN`` nor ``~/.codex/auth.json`` is
available, so they're safe to keep in the default integration suite.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import pytest

try:
    from dotenv import load_dotenv as _load_dotenv
    _env = Path(__file__).resolve().parents[2] / ".env"
    if _env.exists():
        _load_dotenv(_env, override=False)
except ImportError:
    pass

from kestrel_sovereign.llm.codex_adapter import CodexAdapter  # noqa: E402
from kestrel_sovereign.llm.continuation_store import (  # noqa: E402
    InMemoryContinuationStore,
)


def _resolve_codex_token() -> Optional[str]:
    """Match ``ProviderRegistry._read_codex_auth_file`` resolution order.

    Env first, then ``~/.codex/auth.json`` written by ``codex login``.
    """
    env_token = os.environ.get("CODEX_AUTH_TOKEN")
    if env_token:
        return env_token

    auth_path = Path.home() / ".codex" / "auth.json"
    if not auth_path.exists():
        return None
    try:
        data = json.loads(auth_path.read_text())
        tokens = data.get("tokens", {})
        return tokens.get("access_token") or data.get("access_token")
    except Exception:
        return None


def _default_model() -> str:
    return os.environ.get("CODEX_BENCH_MODEL", "gpt-5.5")


def _skip_if_no_creds() -> str:
    token = _resolve_codex_token()
    if not token:
        pytest.skip(
            "No Codex credentials — set CODEX_AUTH_TOKEN or run `codex login`. "
            "Skipping live-network test."
        )
    return token


@pytest.mark.asyncio
async def test_codex_single_turn_text_real_api():
    """Bare user message → text response. Smoke test for adapter wiring.

    Confirms account_id extraction from the JWT, OAuth header construction,
    SSE event parsing, and ``response.completed`` capture all work end-to-end.
    """
    token = _skip_if_no_creds()
    adapter = CodexAdapter()
    model = _default_model()

    resp = await adapter.get_response(
        client=token,
        model=model,
        messages=[
            {"role": "system", "content": "Reply with a single short sentence."},
            {"role": "user", "content": "What is 2+2?"},
        ],
    )

    print(
        f"\n[{model}] single-turn: content={resp.content!r} "
        f"input_tokens={resp.input_tokens} output_tokens={resp.output_tokens}",
        file=sys.stderr,
    )
    assert resp.content, f"Expected non-empty text content, got {resp!r}"


@pytest.mark.asyncio
async def test_codex_tool_call_round_trip_real_api():
    """The #828 live repro. Two turns:

    Turn 1: user asks the model to use a tool. Expect either a function_call
    response or a direct text answer (some sessions answer without using the
    tool — accept both, only fail on protocol error).

    Turn 2: orchestrator-style follow-up. Build the multi-turn message list
    in **Chat-Completions** format with ``role=assistant`` carrying
    ``tool_calls`` and a ``role=tool`` result — exactly what the agent loop
    emits. Pre-#828 fix this turn returned 400 ``Unknown parameter:
    'input[1].tool_calls'``. Post-fix the adapter must translate to
    ``function_call`` / ``function_call_output`` items and complete cleanly.
    """
    token = _skip_if_no_creds()
    adapter = CodexAdapter()
    model = _default_model()

    tool = {
        "type": "function",
        "function": {
            "name": "get_current_model",
            "description": "Return the name of the model currently serving this conversation.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }

    # Turn 1: ask in a way that strongly biases toward calling the tool.
    resp_t1 = await adapter.get_response(
        client=token,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a diagnostic assistant. To answer questions about "
                    "your runtime identity, call the ``get_current_model`` tool "
                    "rather than guessing."
                ),
            },
            {
                "role": "user",
                "content": "What model are you using right now? Use the tool.",
            },
        ],
        tools=[tool],
    )

    print(
        f"\n[{model}] T1: content={resp_t1.content!r} "
        f"tool_calls={[(tc.id, tc.name) for tc in (resp_t1.tool_calls or [])]}",
        file=sys.stderr,
    )

    # If the model didn't call the tool, the rest of the test isn't
    # meaningful — but we still pass turn 1, which proves the adapter
    # accepts a tool definition without erroring. Synthesize a fake tool
    # call ourselves to drive the turn-2 protocol regardless, since the
    # bug is about the *adapter* accepting Chat-Completions input shape,
    # not about whether the model chose to use the tool.
    if resp_t1.tool_calls:
        tc = resp_t1.tool_calls[0]
        synthetic = False
    else:
        # Synthetic call_id matching what the orchestrator would emit if
        # the model had returned one. Turn-2 still tests the conversion
        # path and the wire validation; the model just never produced
        # this id, but the Responses API doesn't validate that against
        # any prior server-side state when ``store=False``.
        from types import SimpleNamespace
        tc = SimpleNamespace(id="call_synthetic_1", name="get_current_model", arguments={})
        synthetic = True
        print(
            f"\n[{model}] T1 did not call the tool — synthesizing a call_id for T2 "
            "so we still exercise the Chat-Completions → Responses-API conversion.",
            file=sys.stderr,
        )

    # Turn 2: build the orchestrator's exact message shape.
    args_str = json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else (tc.arguments or "{}")
    messages_t2 = [
        {
            "role": "system",
            "content": (
                "You are a diagnostic assistant. To answer questions about "
                "your runtime identity, call the ``get_current_model`` tool "
                "rather than guessing."
            ),
        },
        {
            "role": "user",
            "content": "What model are you using right now? Use the tool.",
        },
        {
            "role": "assistant",
            "content": resp_t1.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": args_str},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": f"{model}",
        },
    ]

    # The pre-#828 failure raised RuntimeError("Codex API returned 400: ...").
    # Post-fix this turn must complete with HTTP 200 (resp.content may or may
    # not be non-empty depending on what the model decides to do, but the
    # adapter must not raise a protocol error).
    resp_t2 = await adapter.get_response(
        client=token,
        model=model,
        messages=messages_t2,
        tools=[tool],
    )

    print(
        f"\n[{model}] T2 (synthetic={synthetic}): content={resp_t2.content!r} "
        f"input_tokens={resp_t2.input_tokens}",
        file=sys.stderr,
    )

    # The whole point of #828: turn 2 must NOT raise on the
    # ``Unknown parameter: 'input[*].tool_calls'`` shape.
    # Either content or further tool_calls is acceptable — both prove the
    # request body was accepted by the Responses API.
    assert resp_t2.content is not None or resp_t2.tool_calls, (
        f"Turn 2 returned empty response; expected text or tool calls. {resp_t2!r}"
    )


@pytest.mark.asyncio
async def test_codex_tool_call_with_session_id_real_api():
    """The agent's actual production path: tool calls + session_id + replay
    on T2. This combination wasn't covered by the existing live tests and
    let the call_id-extraction bug ship to production (#857). The user's
    repro: two consecutive tool-using turns failed with 400 ""No tool output
    found for function call call_..."" because the cached function_call's
    ``call_id`` and the orchestrator's tool_call_id had drifted apart —
    the adapter was capturing the output-item ``id`` (``fc_...``) instead
    of the tool-call ``call_id`` (``call_...``).

    This test mirrors the agent's real flow: extract tool_call.id from
    ``LLMResponse.tool_calls`` (no synthesis), reuse it as ``tool_call_id``
    in the next turn's tool message, and pass ``session_id`` so the
    adapter caches T1's outputs and replays them on T2.
    """
    token = _skip_if_no_creds()
    store = InMemoryContinuationStore()
    adapter = CodexAdapter(continuation_store=store)
    model = _default_model()
    session_id = "test-codex-real-tool-call-with-session"

    tool = {
        "type": "function",
        "function": {
            "name": "get_current_model",
            "description": "Returns the current model id.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }

    resp_t1 = await adapter.get_response(
        client=token,
        model=model,
        messages=[
            {"role": "system", "content": "Use the tool to identify the runtime model."},
            {"role": "user", "content": "What model are you running on? Use the tool."},
        ],
        tools=[tool],
        session_id=session_id,
    )

    if not resp_t1.tool_calls:
        pytest.skip(
            f"T1 didn't call the tool (got content={resp_t1.content!r}); "
            "the call_id-replay invariant cannot be exercised this run."
        )

    tc = resp_t1.tool_calls[0]
    print(
        f"\n[{model}] T1 tool_call.id={tc.id!r} (must be ``call_...`` not ``fc_...``)",
        file=sys.stderr,
    )
    # Pre-fix this would be ``fc_...``. Post-fix it must be ``call_...``.
    assert tc.id.startswith("call_"), (
        f"Expected ToolCall.id to be the API's call_id (``call_...``), got {tc.id!r}. "
        "Pre-#857 the adapter captured the wrong field."
    )

    # T2: build orchestrator-style messages using the EXACT id from T1.
    # The agent layer does this same flow.
    args_str = json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else (tc.arguments or "{}")
    resp_t2 = await adapter.get_response(
        client=token,
        model=model,
        messages=[
            {"role": "system", "content": "Use the tool to identify the runtime model."},
            {"role": "user", "content": "What model are you running on? Use the tool."},
            {
                "role": "assistant",
                "content": resp_t1.content or "",
                "tool_calls": [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": args_str},
                }],
            },
            {"role": "tool", "tool_call_id": tc.id, "content": model},
        ],
        tools=[tool],
        session_id=session_id,
    )

    print(
        f"\n[{model}] T2 (with replay): content={resp_t2.content!r}",
        file=sys.stderr,
    )
    # Pre-fix: 400 ""No tool output found for function call call_..."".
    # Post-fix: clean text response or further tool call.
    assert resp_t2.content is not None or resp_t2.tool_calls


@pytest.mark.asyncio
async def test_codex_text_only_reasoning_replay_real_api():
    """Text-only reasoning continuity (#842). T1 asks GPT-5 a problem that
    reliably triggers a ``reasoning`` item but no function_calls. T2 is a
    text-only follow-up referencing T1. The cached T1 reasoning must be
    replayed as input on T2 — preserving encrypted chain-of-thought across
    plain text turns. The strict pass condition is HTTP 200; if the live
    API rejects our reasoning-item shape on T2, the adapter raises 400.

    Scoped narrowly: this test exists to prevent regression of the
    text-only replay path that the #875 fix could otherwise have removed
    if the cache clear had been overly broad.
    """
    token = _skip_if_no_creds()
    store = InMemoryContinuationStore()
    adapter = CodexAdapter(continuation_store=store)
    model = _default_model()
    session_id = "test-codex-real-text-only-replay"

    resp_t1 = await adapter.get_response(
        client=token, model=model,
        messages=[
            {"role": "system", "content": "Reason step by step. Then answer concisely."},
            {"role": "user", "content": "What is 17 multiplied by 23? Show the calculation."},
        ],
        session_id=session_id,
    )

    cursor = store.get("openai_plan", session_id)
    assert cursor is not None
    cached = json.loads(cursor.turn_outputs[0])
    types = [item.get("type") for item in cached]
    if "reasoning" not in types:
        pytest.skip(
            f"GPT-5 didn't emit a reasoning item this run (types={types}); "
            "behavior fluctuation, not an adapter bug."
        )

    resp_t2 = await adapter.get_response(
        client=token, model=model,
        messages=[
            {"role": "system", "content": "Reason step by step. Then answer concisely."},
            {"role": "user", "content": "What is 17 multiplied by 23? Show the calculation."},
            {"role": "assistant", "content": resp_t1.content or ""},
            {"role": "user", "content": "Now multiply that by 2."},
        ],
        session_id=session_id,
    )

    print(
        f"\n[{model}] text-only T2 (replay enabled): {resp_t2.content!r}",
        file=sys.stderr,
    )
    assert resp_t2.content, "T2 should produce text content"

    # Both turns recorded, cache grows to 2 entries.
    cursor2 = store.get("openai_plan", session_id)
    assert len(cursor2.turn_outputs) == 2


@pytest.mark.asyncio
async def test_codex_two_separate_agent_loops_share_session_real_api():
    """The user's #875 production scenario: two independent agent loops on
    the same ``session_id``. Loop 1 ends; the orchestrator's storage keeps
    the final assistant text only (no tool/tool_call messages). Loop 2
    starts with a fresh user message and a fresh tool exchange.

    Pre-#875 the cache from loop 1 leaked into loop 2: the converter walked
    assistant messages positionally and replayed loop 1's
    ``function_call(call_X)`` items even though loop 2's tool messages
    referenced ``call_Y``. Result: 400 ""No tool call found for function
    call output with call_id call_Y"".

    Post-#875 the converter looks up cached function_calls by ``call_id``
    against the orchestrator's current ``tool_calls``, AND the adapter
    clears the cache when the input has no tool history (start of a fresh
    loop). Both loops must complete with 200.
    """
    token = _skip_if_no_creds()
    store = InMemoryContinuationStore()
    adapter = CodexAdapter(continuation_store=store)
    model = _default_model()
    session_id = "test-codex-real-multi-loop-shared-session"

    tool = {
        "type": "function",
        "function": {
            "name": "get_current_model",
            "description": "Returns the current model id.",
            "parameters": {
                "type": "object", "properties": {}, "additionalProperties": False,
            },
        },
    }

    # --- Agent loop 1: tool call + tool result + final answer ---
    resp_l1_t1 = await adapter.get_response(
        client=token, model=model,
        messages=[
            {"role": "system", "content": "Use the tool to identify the runtime model."},
            {"role": "user", "content": "What model are you? Use the tool."},
        ],
        tools=[tool],
        session_id=session_id,
    )
    if not resp_l1_t1.tool_calls:
        pytest.skip("Loop 1 didn't call the tool; cannot exercise the bug.")
    tc1 = resp_l1_t1.tool_calls[0]
    args1 = json.dumps(tc1.arguments) if isinstance(tc1.arguments, dict) else (tc1.arguments or "{}")

    resp_l1_t2 = await adapter.get_response(
        client=token, model=model,
        messages=[
            {"role": "system", "content": "Use the tool to identify the runtime model."},
            {"role": "user", "content": "What model are you? Use the tool."},
            {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": tc1.id, "type": "function",
                    "function": {"name": tc1.name, "arguments": args1},
                }],
            },
            {"role": "tool", "tool_call_id": tc1.id, "content": model},
        ],
        tools=[tool],
        session_id=session_id,
    )
    print(
        f"\n[{model}] Loop 1 final: {resp_l1_t2.content!r}",
        file=sys.stderr,
    )

    # Cache now holds 2 turn_outputs from loop 1.
    cursor_after_loop1 = store.get("openai_plan", session_id)
    assert len(cursor_after_loop1.turn_outputs) == 2

    # --- Agent loop 2: NEW user message, no tool history in input ---
    # The orchestrator's storage keeps only the final assistant text from
    # loop 1; tool/tool_call messages are transient. So loop 2's input has
    # no ``assistant.tool_calls`` — pre-#875 this triggered the bug.
    resp_l2_t1 = await adapter.get_response(
        client=token, model=model,
        messages=[
            {"role": "system", "content": "Use the tool to identify the runtime model."},
            {"role": "user", "content": "What model are you? Use the tool."},
            {"role": "assistant", "content": resp_l1_t2.content or ""},
            {"role": "user", "content": "Confirm again — what model are you? Use the tool."},
        ],
        tools=[tool],
        session_id=session_id,
    )
    if not resp_l2_t1.tool_calls:
        pytest.skip("Loop 2 didn't call the tool; cannot exercise the round trip.")
    tc2 = resp_l2_t1.tool_calls[0]

    # The cache accumulates across loops (id-match keeps it safe; no
    # fresh-loop clear by design — preserves text-only continuity).
    args2 = json.dumps(tc2.arguments) if isinstance(tc2.arguments, dict) else (tc2.arguments or "{}")
    resp_l2_t2 = await adapter.get_response(
        client=token, model=model,
        messages=[
            {"role": "system", "content": "Use the tool to identify the runtime model."},
            {"role": "user", "content": "What model are you? Use the tool."},
            {"role": "assistant", "content": resp_l1_t2.content or ""},
            {"role": "user", "content": "Confirm again — what model are you? Use the tool."},
            {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": tc2.id, "type": "function",
                    "function": {"name": tc2.name, "arguments": args2},
                }],
            },
            {"role": "tool", "tool_call_id": tc2.id, "content": model},
        ],
        tools=[tool],
        session_id=session_id,
    )
    print(
        f"\n[{model}] Loop 2 final: {resp_l2_t2.content!r}",
        file=sys.stderr,
    )
    # Pre-#875: loop 2 raised 400 ""No tool call found for function call output
    # with call_id call_..."". Post-fix: clean text response.
    assert resp_l2_t2.content is not None or resp_l2_t2.tool_calls


@pytest.mark.asyncio
async def test_codex_continuation_cursor_written_real_api():
    """Two-turn run with the same ``session_id``. Cursor must be written on
    turn 1 with ``last_response_id`` from the live ``response.completed`` event,
    and turn 2 must complete without error.
    """
    token = _skip_if_no_creds()
    store = InMemoryContinuationStore()
    adapter = CodexAdapter(continuation_store=store)
    model = _default_model()

    session_id = "test-codex-real-continuation"

    resp_t1 = await adapter.get_response(
        client=token,
        model=model,
        messages=[
            {"role": "system", "content": "Reply with a single short sentence."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        session_id=session_id,
    )

    cursor = store.get("openai_plan", session_id)
    print(
        f"\n[{model}] T1 cursor: {cursor}",
        file=sys.stderr,
    )
    assert cursor is not None, "Cursor must be written after a successful turn"
    assert cursor.last_response_id, (
        "Cursor must capture last_response_id from the response.completed event"
    )
    assert cursor.last_message_count == 1, (
        f"Cursor watermark = user-message count after system extraction "
        f"(expected 1, got {cursor.last_message_count})"
    )

    # Turn 2: same session_id → adapter sends previous_response_id under the
    # hood. We can't observe the wire from outside the adapter, but a clean
    # 200 proves the live endpoint accepts our continuation token and the
    # delta input items.
    resp_t2 = await adapter.get_response(
        client=token,
        model=model,
        messages=[
            {"role": "system", "content": "Reply with a single short sentence."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": resp_t1.content or ""},
            {"role": "user", "content": "What is 3+3?"},
        ],
        session_id=session_id,
    )

    cursor2 = store.get("openai_plan", session_id)
    print(
        f"\n[{model}] T2 content={resp_t2.content!r} cursor={cursor2}",
        file=sys.stderr,
    )
    assert resp_t2.content, "Turn 2 should return text content"
    assert cursor2.last_response_id != cursor.last_response_id, (
        "Cursor must refresh to the new response_id on turn 2"
    )
    assert cursor2.last_message_count == 3, (
        f"Watermark on turn 2 = user-side message count (3 after system "
        f"extraction), got {cursor2.last_message_count}"
    )
