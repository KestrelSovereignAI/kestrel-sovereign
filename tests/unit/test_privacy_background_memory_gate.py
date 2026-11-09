"""
Regression tests for the single post-response/background memory privacy gate
(#1760).

EPHEMERAL ("nothing stored anywhere") and ISOLATED ("temporary session buffer
only") must never derive durable state — temporal patterns, concept-graph
nodes, emotional memory metadata, or embeddings — from raw chat input. Both the
non-streaming and the streaming response paths funnel their post-response
memory work through ``KestrelAgent._privacy_blocks_background_memory`` →
``_post_response_pipeline``. These tests prove the gate blocks the volatile
modes on BOTH paths and lets persistent modes through.
"""
import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode, privacy_mode_to_config


# ─────────────────────────────────────────────────────────────────────────────
# _privacy_blocks_background_memory predicate
# ─────────────────────────────────────────────────────────────────────────────


def _agent_with_mode(mode):
    """Minimal stand-in carrying just the privacy_config the gate reads."""
    agent = MagicMock()
    agent.privacy_agent = MagicMock()
    agent.privacy_agent.privacy_config = privacy_mode_to_config(mode)
    return agent


class TestPrivacyGatePredicate:
    @pytest.mark.parametrize(
        "mode",
        [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED, PrivacyMode.DEIDENTIFIED],
    )
    def test_volatile_modes_block(self, mode):
        agent = _agent_with_mode(mode)
        assert KestrelAgent._privacy_blocks_background_memory(agent) is True

    @pytest.mark.parametrize(
        "mode",
        [PrivacyMode.NORMAL, PrivacyMode.PUBLIC, PrivacyMode.ANONYMOUS],
    )
    def test_persistent_modes_allow(self, mode):
        agent = _agent_with_mode(mode)
        assert KestrelAgent._privacy_blocks_background_memory(agent) is False

    def test_missing_privacy_agent_does_not_block(self):
        agent = MagicMock()
        agent.privacy_agent = None
        assert KestrelAgent._privacy_blocks_background_memory(agent) is False


# ─────────────────────────────────────────────────────────────────────────────
# Streaming-path parity: streaming routes through the same gate
# ─────────────────────────────────────────────────────────────────────────────


async def _build_streaming_agent(mode, native_tool_effect_agent):
    """Actual privacy/storage/input lifetime with explicit context/provider seams."""
    from kestrel_sovereign.agent.context_manager import ContextResult
    from kestrel_sovereign.llm.adapter import LLMResponse

    agent = await native_tool_effect_agent(did="did:test:background-memory")
    if mode is not PrivacyMode.NORMAL:
        transition = await agent.set_privacy_mode_with_effects(mode)
        assert transition.applied
    agent.extension = None
    agent._cached_features_prompt = ""
    agent._maybe_audit = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered"
    agent.operator_signal_producer = None
    context = ContextResult(system_prompt="system", messages=[], total_tokens=0, budget_summary={})
    agent.context_manager = SimpleNamespace(build_context=AsyncMock(return_value=context))

    async def stream_no_tools(**kwargs):
        for piece in ["Hello", " ", "world"]:
            yield piece
        yield LLMResponse(content="Hello world", tool_calls=[])

    agent.llm_service.stream_with_tool_detection = stream_no_tools
    agent._post_response_pipeline = AsyncMock()
    return agent


async def _drain_stream(agent):
    out = []
    async for chunk in agent.process_input_streaming("my secret note", session_id="s1"):
        out.append(chunk)
    return "".join(out)


@pytest.mark.asyncio
async def test_streaming_normal_runs_post_response_pipeline(native_tool_effect_agent):
    agent = await _build_streaming_agent(PrivacyMode.NORMAL, native_tool_effect_agent)
    text = await _drain_stream(agent)
    assert "Hello world" in text
    agent._post_response_pipeline.assert_awaited_once()
    args = agent._post_response_pipeline.await_args[0]
    assert args[0] == "my secret note"  # raw user input
    assert args[1] == "Hello world"  # final assistant text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED])
async def test_streaming_volatile_modes_skip_post_response_pipeline(mode, native_tool_effect_agent):
    agent = await _build_streaming_agent(mode, native_tool_effect_agent)
    try:
        text = await _drain_stream(agent)
        assert "Hello world" in text
        # The single privacy gate must keep the pipeline (and therefore all
        # temporal/concept/graph/embedding writes) from ever being entered.
        agent._post_response_pipeline.assert_not_awaited()
        assert await agent._raw_storage.db.fetchval("SELECT COUNT(*) FROM conversation_history") == 0

    finally:
        # These tests own the volatile session they opened on the partial host.
        agent.privacy_agent.discard_isolated_session()
        if agent.privacy_agent.ephemeral_session is not None:
            agent.privacy_agent.ephemeral_session.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Non-streaming parity: durable derived state is never produced
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED])
async def test_nonstreaming_volatile_modes_skip_all_memory_work(mode):
    """EPHEMERAL/ISOLATED must not read raw history nor invoke the analyzer,
    linker, or emotional tagger from the non-streaming pipeline."""
    agent = MagicMock()
    agent.agent_id = "did:pkh:test"
    agent.privacy_agent = MagicMock()
    agent.privacy_agent.privacy_config = privacy_mode_to_config(mode)
    agent._privacy_blocks_background_memory = (
        KestrelAgent._privacy_blocks_background_memory.__get__(agent, KestrelAgent)
    )

    agent.memory_system = MagicMock()
    agent.memory_system.analyzer = AsyncMock()
    agent.memory_system.linker = AsyncMock()

    conv_store = AsyncMock()
    agent._raw_storage = MagicMock()
    agent._raw_storage.conversation = conv_store
    agent.context_manager = MagicMock()
    agent.context_manager.memory_manager = MagicMock()
    agent.context_manager.memory_manager.tag_exchange = AsyncMock()

    await KestrelAgent._post_response_pipeline(
        agent, "raw private input", "private response", session_id="s1"
    )
    await asyncio.sleep(0.05)

    conv_store.get_full_history_with_ids.assert_not_awaited()
    agent.context_manager.memory_manager.tag_exchange.assert_not_awaited()
    agent.memory_system.analyzer.detect_patterns.assert_not_awaited()
    agent.memory_system.linker.extract_and_link.assert_not_awaited()
