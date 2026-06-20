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
from contextlib import asynccontextmanager

import pytest
from unittest.mock import AsyncMock, MagicMock

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


@asynccontextmanager
async def _passthrough():
    yield


def _build_streaming_agent(mode):
    """Mock agent wired to run the real streaming finalize path with the
    real privacy gate, spying on _post_response_pipeline."""
    from kestrel_sovereign.agent.streaming import StreamingMixin

    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock()
    privacy_agent.privacy_config = privacy_mode_to_config(mode)
    privacy_agent.privacy_mode = MagicMock()
    privacy_agent.privacy_mode.name = mode.value
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    agent = MagicMock()
    agent.privacy_agent = privacy_agent
    agent.features = {}
    agent.did = "test-did"
    agent.extension = None
    agent._cached_features_prompt = ""
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent._maybe_audit = AsyncMock()
    agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    agent.hooks_manager = None
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered"

    ctx = MagicMock()
    ctx.system_prompt = "system"
    ctx.dynamic_user_context = ""
    ctx.messages = []
    ctx.degraded_mode = False
    ctx.warnings = []
    agent.context_manager = MagicMock()
    agent.context_manager.build_context = AsyncMock(return_value=ctx)

    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_metric = AsyncMock()

    async def mock_stream_no_tools(**kwargs):
        for piece in ["Hello", " ", "world"]:
            yield piece

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = mock_stream_no_tools

    # Spy on the post-response pipeline; bind the REAL gate so the test
    # exercises the actual privacy decision, not a mock.
    agent._post_response_pipeline = AsyncMock()
    agent._privacy_blocks_background_memory = (
        KestrelAgent._privacy_blocks_background_memory.__get__(agent, KestrelAgent)
    )

    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(agent)
    agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(agent)
    )
    agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(agent)
    )
    return agent


async def _drain_stream(agent):
    out = []
    async for chunk in agent.process_input_streaming("my secret note", session_id="s1"):
        out.append(chunk)
    return "".join(out)


@pytest.mark.asyncio
async def test_streaming_normal_runs_post_response_pipeline():
    agent = _build_streaming_agent(PrivacyMode.NORMAL)
    text = await _drain_stream(agent)
    assert "Hello world" in text
    agent._post_response_pipeline.assert_awaited_once()
    args = agent._post_response_pipeline.await_args[0]
    assert args[0] == "my secret note"  # raw user input
    assert args[1] == "Hello world"  # final assistant text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED])
async def test_streaming_volatile_modes_skip_post_response_pipeline(mode):
    agent = _build_streaming_agent(mode)
    text = await _drain_stream(agent)
    assert "Hello world" in text
    # The single privacy gate must keep the pipeline (and therefore all
    # temporal/concept/graph/embedding writes) from ever being entered.
    agent._post_response_pipeline.assert_not_awaited()


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
