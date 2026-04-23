"""Active-session privacy transition contracts."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode


class _PrivacyConfig:
    def __init__(self, allows_cloud: bool = True):
        self._allows_cloud = allows_cloud

    def allows_cloud_llm(self):
        return self._allows_cloud


class _PrivacyAgent:
    def __init__(self):
        self.privacy_config = _PrivacyConfig(allows_cloud=True)
        self.privacy_mode = PrivacyMode.NORMAL
        self.conversations = []

    async def add_conversation(self, role, content, metadata=None, session_id=None):
        self.conversations.append((role, content, metadata, session_id))

    async def get_conversation_history(self, limit=50, session_id=None):
        return []

    def set_mode(self, mode):
        self.privacy_mode = mode
        self.privacy_config = _PrivacyConfig(allows_cloud=mode.to_config().allows_cloud_llm())
        return f"Privacy mode changed to {mode.value}."


class _LLMService:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.force_local_only_values = []
        self.providers = [
            {"name": "openai", "model": "gpt-5-mini"},
            {"name": "ollama", "model": "llama3.2:3b"},
        ]

    async def stream_with_tool_detection(self, **kwargs):
        self.force_local_only_values.append(kwargs["force_local_only"])
        self.started.set()
        yield "hello"
        await self.release.wait()
        yield " world"

    def _get_local_provider_names(self):
        return ["ollama"]

    def get_model_preference(self):
        return {"provider": "openai", "model": "gpt-5-mini"}

    def set_model_preference(self, model, vendor=None, route=None):
        self.model_preference = {"provider": provider, "model": model}


class _ObservabilityStore:
    async def log_metric(self, **kwargs):
        return None

    async def log_tool_call(self, **kwargs):
        return "event-1"

    async def log_tool_response(self, **kwargs):
        return None


def _make_streaming_agent():
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.did = "did:test:privacy-active-session"
    agent._privacy_mode = PrivacyMode.NORMAL
    agent.storage = MagicMock()
    agent.storage.set_privacy_mode = MagicMock()
    agent.privacy_agent = _PrivacyAgent()
    agent.llm_service = _LLMService()
    agent.features = {}
    agent.hooks_manager = None
    agent.context_manager = MagicMock()
    # ContextResult now carries dynamic_user_context (issue #703) — per-turn
    # retrieved content that belongs in the user message, not system.
    agent.context_manager.build_context = AsyncMock(
        return_value=SimpleNamespace(
            system_prompt="system",
            messages=[],
            dynamic_user_context="",
        )
    )
    agent.user_prompt_template = "{context}{query}"
    agent._cached_features_prompt = ""
    agent.extension = None
    agent.observability_store = _ObservabilityStore()
    agent._maybe_audit = AsyncMock(return_value=None)
    agent._get_governing_constitution = AsyncMock(return_value="constitution")
    agent.check_solvency = AsyncMock(return_value="gpt-5-mini")
    agent._build_all_tools = MagicMock(return_value=[])
    return agent


@pytest.mark.asyncio
async def test_privacy_transition_waits_for_active_stream_before_switching_modes():
    agent = _make_streaming_agent()

    chunks = []

    async def consume_stream():
        async for chunk in agent.process_input_streaming("hello"):
            chunks.append(chunk)

    stream_task = asyncio.create_task(consume_stream())
    await agent.llm_service.started.wait()
    await asyncio.sleep(0)

    transition_task = asyncio.create_task(
        agent.set_privacy_mode_with_effects(PrivacyMode.ISOLATED)
    )
    await asyncio.sleep(0)

    assert transition_task.done() is False
    assert agent._privacy_mode == PrivacyMode.NORMAL
    assert agent.llm_service.force_local_only_values == [False]

    agent.llm_service.release.set()
    await stream_task
    transition = await transition_task

    assert chunks == ["hello", " world"]
    assert transition.allows_cloud_llm is False
    assert agent._privacy_mode == PrivacyMode.ISOLATED
    agent.storage.set_privacy_mode.assert_called_once_with(PrivacyMode.ISOLATED)
