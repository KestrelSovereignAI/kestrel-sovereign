"""
State of Mind Feature

Provides the !state-of-mind command to inspect constitutional governance mode.
"""
import logging
from typing import Optional
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

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
    async def get_state_of_mind(self) -> str:
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
        try:
            # Get state of mind from LLM service
            if not hasattr(self.agent, 'llm_service'):
                return "Error: LLM service not available"

            state = self.agent.llm_service.get_state_of_mind()

            # Format using the profile service
            from kestrel_sovereign.llm.constitutional_profile import get_profile_service
            profile_service = get_profile_service()
            formatted = profile_service.format_state_of_mind(state)

            return formatted

        except Exception as e:
            logger.error(f"Failed to get state of mind: {e}")
            return f"Error retrieving state of mind: {str(e)}"
