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
    resolve_calls = []

    def _resolve(**kw):
        resolve_calls.append(kw)
        return (
            ([{"adapter": primary_adapter}] if primary_adapter is not None else []),
            None,
        )

    llm = SimpleNamespace(resolve_provider_routing=_resolve)
    fake = SimpleNamespace(
        llm_service=llm,
        context_manager=_FakeContextManager(compact_result),
        resolve_calls=resolve_calls,
    )
    # Bind the sibling helpers the orchestrator calls.
    fake._active_codex_adapter = (
        lambda mo=None: KestrelAgent._active_codex_adapter(fake, mo)
    )
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
async def test_per_turn_model_override_is_passed_to_routing():
    # Gating must resolve with the turn's model_override, not the default
    # route (codex review r2).
    codex = _codex_with_occupancy(85)
    agent = _fake_agent(primary_adapter=codex, compact_result={"success": True})
    await KestrelAgent._maybe_compact_codex_thread(
        agent, "s", "openai:plan/gpt-5.5"
    )
    assert agent.resolve_calls, "resolver must be called"
    assert agent.resolve_calls[-1].get("model_override") == "openai:plan/gpt-5.5"
    assert "force_local_only" in agent.resolve_calls[-1]


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


@pytest.mark.asyncio
async def test_global_compaction_excludes_originals_via_id_bearing_source():
    """#1844 codex r4: global !compact (session_id=None) must read the
    id-bearing history source so originals are EXCLUDED, not just have a
    summary appended (which would grow context instead of compacting)."""
    from unittest.mock import AsyncMock, MagicMock
    from kestrel_sovereign.agent.context_manager import ContextManager

    llm = MagicMock()
    llm.generate = AsyncMock(return_value="SUMMARY: condensed.")

    msgs = [{"id": i + 1, "role": "user" if i % 2 == 0 else "assistant",
             "content": f"m{i}"} for i in range(20)]
    storage = MagicMock()
    storage.conversation = AsyncMock()
    storage.conversation.get_full_history_with_ids = AsyncMock(return_value=msgs)
    storage.conversation.add_conversation = AsyncMock()
    storage.conversation.update_messages_metadata = AsyncMock()
    storage.conversation.db = AsyncMock()
    storage.conversation.db.fetchone = AsyncMock(return_value=[777])
    storage.conversation.agent_id = "agent-x"

    mgr = ContextManager(storage=storage, model="gpt-4")
    result = await mgr.compact_session(llm_service=llm, preserve_recent=5)  # no session_id

    assert result["success"] is True, result
    # Originals must be excluded — exclusion call made with the compacted ids.
    storage.conversation.update_messages_metadata.assert_awaited()
    excluded_ids = storage.conversation.update_messages_metadata.call_args.args[0]
    assert excluded_ids == [m["id"] for m in msgs[:-5]]


@pytest.mark.asyncio
async def test_noop_in_economy_or_critical_solvency_mode():
    # codex r5: when solvency forces a local model (ECONOMY/CRITICAL), the turn
    # won't use codex — don't compact/reset on a stale-high occupancy.
    for pref in ("ECONOMY", "CRITICAL"):
        codex = _codex_with_occupancy(95)
        agent = _fake_agent(primary_adapter=codex, compact_result={"success": True})
        agent._current_model_preference = pref
        await KestrelAgent._maybe_compact_codex_thread(agent, "s")
        assert agent.context_manager.compact_calls == 0
        assert "s" in codex._session_threads  # untouched
