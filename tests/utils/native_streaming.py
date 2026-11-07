"""Scripted streaming collaborators around a real native Agent and primary SQL."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.agent.context_manager import ContextResult


async def streaming_agent(native_tool_effect_agent, *, persisted=None, connected_backend=None):
    agent = await native_tool_effect_agent(did="did:test:streaming-component", connected_backend=connected_backend)
    context = ContextResult(system_prompt="system", messages=[], total_tokens=0,
                            budget_summary={}, dynamic_user_context="ctx")
    agent.context_manager = SimpleNamespace(build_context=AsyncMock(return_value=context))
    agent.user_prompt_template = "{context}{query}"
    agent._cached_features_prompt = ""
    agent.extension = None
    agent.operator_signal_producer = None
    agent._maybe_audit = AsyncMock(return_value=None)
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent._get_governing_constitution = AsyncMock(return_value="constitution")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent._post_response_pipeline = AsyncMock(return_value=None)
    write = agent.privacy_agent.add_conversation

    async def record(role, content, **kwargs):
        result = await write(role, content, **kwargs)
        if persisted is not None:
            persisted.append({"role": role, "content": content, **kwargs})
        return result

    agent.privacy_agent.add_conversation = AsyncMock(side_effect=record)
    return agent
