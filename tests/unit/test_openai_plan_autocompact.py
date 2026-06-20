"""#1844 Stage 2: Kestrel-owned compaction for openai:plan.

Covers the adapter env knob and the agent-side pre-turn orchestration
(_maybe_compact_codex_thread): gate to the resolved CodexAdapter, fire only at
/above the occupancy threshold, and compact + reset_thread so the next turn
reseeds the compacted view.
"""
import os
from types import SimpleNamespace

import pytest

from kestrel_sovereign.llm.codex_adapter import CodexAdapter, _env_int
from kestrel_sovereign.kestrel_agent import KestrelAgent


def test_env_int_parsing(monkeypatch):
    monkeypatch.delenv("X_KS_TEST_INT", raising=False)
    assert _env_int("X_KS_TEST_INT") is None
    monkeypatch.setenv("X_KS_TEST_INT", "  240000 ")
    assert _env_int("X_KS_TEST_INT") == 240000
    monkeypatch.setenv("X_KS_TEST_INT", "nan")
    assert _env_int("X_KS_TEST_INT") is None
    monkeypatch.setenv("X_KS_TEST_INT", "")
    assert _env_int("X_KS_TEST_INT") is None


class _FakeContextManager:
    def __init__(self, result):
        self._result = result
        self.compact_calls = 0

    async def compact_session(self, llm_service, *a, **k):
        self.compact_calls += 1
        return self._result


def _fake_agent(*, primary_adapter, compact_result):
    """Minimal self for the unbound KestrelAgent helpers — no real agent init."""
    llm = SimpleNamespace(
        resolve_provider_routing=lambda: (
            ([{"adapter": primary_adapter}] if primary_adapter is not None else []),
            None,
        )
    )
    fake = SimpleNamespace(
        llm_service=llm,
        context_manager=_FakeContextManager(compact_result),
    )
    # Bind the sibling helpers the orchestrator calls.
    fake._active_codex_adapter = lambda: KestrelAgent._active_codex_adapter(fake)
    fake._codex_compact_threshold_pct = (
        lambda: KestrelAgent._codex_compact_threshold_pct(fake)
    )
    return fake


def _codex_with_occupancy(pct):
    a = CodexAdapter()
    a._session_threads["s"] = ("t1", "fp")
    # window 1000; used = pct% of it
    a._record_thread_occupancy("s", {
        "last": {"inputTokens": int(pct * 10)}, "modelContextWindow": 1000})
    return a


@pytest.mark.asyncio
async def test_compacts_and_resets_when_over_threshold():
    codex = _codex_with_occupancy(85)  # 85% >= 70 default
    agent = _fake_agent(
        primary_adapter=codex,
        compact_result={"success": True, "tokens_saved": 9000, "messages_compacted": 20},
    )
    await KestrelAgent._maybe_compact_codex_thread(agent, "s")
    assert agent.context_manager.compact_calls == 1
    # reset_thread evicted the cached thread → next turn reseeds fresh.
    assert "s" not in codex._session_threads


@pytest.mark.asyncio
async def test_no_compaction_below_threshold():
    codex = _codex_with_occupancy(40)  # below 70
    agent = _fake_agent(primary_adapter=codex, compact_result={"success": True})
    await KestrelAgent._maybe_compact_codex_thread(agent, "s")
    assert agent.context_manager.compact_calls == 0
    assert "s" in codex._session_threads  # untouched


@pytest.mark.asyncio
async def test_thread_not_reset_when_compaction_not_applied():
    codex = _codex_with_occupancy(90)
    agent = _fake_agent(
        primary_adapter=codex,
        compact_result={"success": False, "reason": "Not enough messages to compact"},
    )
    await KestrelAgent._maybe_compact_codex_thread(agent, "s")
    assert agent.context_manager.compact_calls == 1
    # Compaction was a no-op → don't throw away the live thread.
    assert "s" in codex._session_threads


@pytest.mark.asyncio
async def test_noop_when_primary_is_not_codex():
    # Switched away from openai:plan → primary adapter lacks the codex surface.
    agent = _fake_agent(primary_adapter=object(), compact_result={"success": True})
    await KestrelAgent._maybe_compact_codex_thread(agent, "s")
    assert agent.context_manager.compact_calls == 0


@pytest.mark.asyncio
async def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv("KESTREL_OPENAI_PLAN_COMPACT_THRESHOLD_PCT", "50")
    codex = _codex_with_occupancy(60)  # 60 >= 50 (override) but < 70 (default)
    agent = _fake_agent(primary_adapter=codex, compact_result={"success": True})
    await KestrelAgent._maybe_compact_codex_thread(agent, "s")
    assert agent.context_manager.compact_calls == 1


@pytest.mark.asyncio
async def test_compact_session_real_generate_signature_and_session_tagging():
    """#1844 Stage 2 P1+P2: compact_session must call generate with the REAL
    keyword-only signature (user_prompt, not prompt) — the prior prompt= call
    TypeError'd into a silent no-op — AND tag the summary marker to the
    reseeded session so it survives the fresh-thread reseed."""
    from unittest.mock import AsyncMock, MagicMock
    from kestrel_sovereign.agent.context_manager import ContextManager

    captured = {}

    async def strict_generate(*, system_prompt, user_prompt, model_override=None, **kw):
        # Keyword-only + requires user_prompt: passing prompt= would TypeError.
        captured["user_prompt"] = user_prompt
        return "SUMMARY: key points preserved."

    llm = MagicMock()
    llm.generate = strict_generate

    msgs = [
        {"id": i, "role": "user" if i % 2 == 0 else "assistant",
         "content": f"message {i} with content"}
        for i in range(20)
    ]
    storage = MagicMock()
    storage.conversation = AsyncMock()
    storage.conversation.get_conversation_history = AsyncMock(return_value=msgs)
    storage.conversation.add_conversation = AsyncMock()
    storage.conversation.update_messages_metadata = AsyncMock()
    storage.conversation.db = AsyncMock()
    storage.conversation.db.fetchone = AsyncMock(return_value=[999])
    storage.conversation.agent_id = "agent-x"

    mgr = ContextManager(storage=storage, model="gpt-4")
    result = await mgr.compact_session(
        llm_service=llm, preserve_recent=5, session_id="sess-1"
    )

    assert result["success"] is True, result
    assert captured.get("user_prompt"), "generate must be called with user_prompt"
    # Summary marker tagged to the reseeded session (P2).
    add_kwargs = storage.conversation.add_conversation.call_args.kwargs
    assert add_kwargs.get("session_id") == "sess-1"
