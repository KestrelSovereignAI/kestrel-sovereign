"""Integration tests: CodexAdapter against the real ``codex app-server``.

The adapter drives the official ``codex app-server`` binary over stdio
JSON-RPC. The binary owns OAuth via ``~/.codex/auth.json``. These
exercise that end-to-end with the real binary and a real ChatGPT
subscription, including the inline tool-call bridge that routes
``item/tool/call`` to a kestrel-side executor (the production wiring
calls :meth:`OrchestratorEngine.execute_named_tool`, which fires the
full PRE/POST_TOOL_USE hook stack and approval queue).

Gated on the codex binary AND ``~/.codex/auth.json`` both being present,
so they skip safely in CI (which has neither). A live gate is the only
way to catch app-server protocol drift between releases.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.codex_adapter import CodexAdapter

_BIN = os.environ.get(
    "KESTREL_CODEX_APP_SERVER_BIN",
    "/Applications/Codex.app/Contents/Resources/codex",
)
_HAVE = Path(_BIN).exists() and (Path.home() / ".codex" / "auth.json").exists()

pytestmark = pytest.mark.skipif(
    not _HAVE,
    reason="codex binary + ~/.codex/auth.json required for live app-server test",
)


@pytest.mark.asyncio
async def test_single_turn_text_real():
    adapter = CodexAdapter()
    try:
        resp = await adapter.get_response(
            client=None, model="auto",
            messages=[
                {"role": "system", "content": "Reply with one short sentence."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            session_id="it-single",
        )
    finally:
        await adapter.aclose()
    print(
        f"\nsingle-turn: content={resp.content!r} "
        f"in={resp.input_tokens} out={resp.output_tokens}",
        file=sys.stderr,
    )
    assert isinstance(resp, LLMResponse)
    assert resp.content and "4" in resp.content
    assert resp.input_tokens and resp.output_tokens


@pytest.mark.asyncio
async def test_session_reuses_thread_real():
    """Same session_id must reuse one Codex thread (server-side history)."""
    adapter = CodexAdapter()
    try:
        await adapter.get_response(
            client=None, model="auto",
            messages=[{"role": "user", "content": "Remember the word: tortoise."}],
            session_id="it-mem",
        )
        first_thread = adapter._session_threads.get("it-mem")
        r2 = await adapter.get_response(
            client=None, model="auto",
            messages=[{"role": "user",
                       "content": "What word did I ask you to remember?"}],
            session_id="it-mem",
        )
        assert first_thread
        assert adapter._session_threads.get("it-mem") == first_thread
        print(f"\nrecall: {r2.content!r}", file=sys.stderr)
        assert "tortoise" in (r2.content or "").lower()
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_streaming_text_real():
    adapter = CodexAdapter()
    chunks = []
    try:
        async for c in adapter.get_streaming_response(
            client=None, model="auto",
            messages=[{"role": "user", "content": "Count: one two three."}],
            session_id="it-stream",
        ):
            if isinstance(c, str):
                chunks.append(c)
    finally:
        await adapter.aclose()
    text = "".join(chunks)
    print(f"\nstreamed: {text!r}", file=sys.stderr)
    assert text.strip()


@pytest.mark.asyncio
async def test_tool_call_round_trip_real():
    """The decisive end-to-end test: real subscription + dynamicTools +
    server-driven item/tool/call → our executor → result relayed back →
    model uses the result.

    With the orchestrator wired in production, the executor is
    ``execute_named_tool`` which runs through the PRE/POST_TOOL_USE
    hook stack. Here we substitute a stand-in to keep the test
    self-contained, but the bridge is identical."""
    adapter = CodexAdapter()
    seen_calls = []

    async def fake_executor(name: str, args: dict):
        seen_calls.append((name, args))
        if name == "get_secret_word":
            return {"success": True, "result": "salamander"}
        return {"success": False, "error": f"unknown tool {name}"}

    try:
        resp = await adapter.get_response(
            client=None, model="auto",
            messages=[
                {"role": "system",
                 "content": "When the user asks for the secret, call get_secret_word(). Reply with only the tool's returned word."},
                {"role": "user",
                 "content": "Call get_secret_word and tell me only the word it returned."},
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_secret_word",
                    "description": "Return the secret word.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }],
            session_id="it-tool",
            tool_executor=fake_executor,
        )
    finally:
        await adapter.aclose()

    print(
        f"\ntool round-trip: calls={seen_calls} content={resp.content!r}",
        file=sys.stderr,
    )
    assert seen_calls, "the model never invoked the dynamic tool"
    assert seen_calls[0][0] == "get_secret_word"
    assert "salamander" in (resp.content or "").lower()
