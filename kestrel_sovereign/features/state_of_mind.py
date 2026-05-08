"""
State of Mind Feature

Provides the !state-of-mind command to inspect constitutional governance mode.
"""
import logging
from typing import Any, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool

logger = logging.getLogger(__name__)


class StateOfMindFeature(Feature):
    """
    Feature for viewing the agent's current constitutional state of mind.
    """

    def __init__(self, agent):
        super().__init__(agent)

    @property
    def tool_description(self) -> str:
        """Description for A2A agent card."""
        return "View the agent's current constitutional governance state and model alignment"

    async def initialize(self):
        """Initialize the feature."""
        logger.info("StateOfMindFeature initialized successfully")

    @tool(
        name="state_of_mind",
        description="Get the current constitutional governance state for this agent",
        category=ToolCategory.SYSTEM,
        command_prefix="!state-of-mind"
    )
    async def get_state_of_mind(self) -> ToolResult:
        """
        Retrieve the current state of mind for the agent.

        This shows:
        - Current LLM provider and model
        - Governance mode (complementary, authoritative, reinforcing)
        - Delegated constitutional principles
        - Active conflicts with model constitution
        - Prompt adaptation strategy

        Usage:
            !state-of-mind

        Returns:
            Formatted state of mind report
        """
        if not hasattr(self.agent, 'llm_service'):
            return ToolResult.failed("LLM service not available")

        try:
            state = self.agent.llm_service.get_state_of_mind()

            from kestrel_sovereign.llm.constitutional_profile import get_profile_service
            profile_service = get_profile_service()
            formatted = profile_service.format_state_of_mind(state)
        except Exception as e:
            logger.error(f"Failed to get state of mind: {e}")
            return ToolResult.failed(f"Error retrieving state of mind: {str(e)}")

        # The LLM service may return a StateOfMind dataclass; ToolResult.data
        # ends up in json.dumps via _serialize_tool_result, so coerce to a
        # plain dict-or-string here. dataclasses.asdict handles the
        # canonical case; fall back to vars() / str() when it isn't a
        # dataclass instance.
        import dataclasses
        if dataclasses.is_dataclass(state) and not isinstance(state, type):
            state_serialized: Any = dataclasses.asdict(state)
        elif isinstance(state, dict):
            state_serialized = state
        else:
            state_serialized = {"raw": str(state)}

        return ToolResult.ok(
            confirmation=formatted,
            data={"state": state_serialized},
        )
