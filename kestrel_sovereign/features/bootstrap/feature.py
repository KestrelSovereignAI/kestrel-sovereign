"""
Bootstrap Feature - Commands for agent wake-up and discovery management.

Provides commands to control the bootstrap/discovery process:
- !skip-discovery - Skip discovery and use default personality
- !restart-discovery - Reset and redo the discovery process
- !bootstrap-status - Show current bootstrap state
- !rename - Rename the agent
"""

import logging
import re
from pathlib import Path
from typing import Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory
from kestrel_sovereign.bootstrap import BootstrapState

logger = logging.getLogger(__name__)


class BootstrapFeature(Feature):
    """
    Feature for managing agent bootstrap, discovery, and identity.

    Provides commands for:
    - Skipping or restarting the discovery process
    - Checking bootstrap status
    - Renaming the agent
    """

    def __init__(self, agent):
        super().__init__(agent)

    @property
    def tool_description(self) -> str:
        """Description for A2A agent card."""
        return "Manage agent identity - skip/restart discovery, check status, rename agent"

    async def initialize(self):
        """Initialize the bootstrap feature."""
        logger.info("BootstrapFeature initialized")

    @tool(
        name="skip_discovery",
        description="Skip the discovery conversation and use default personality.",
        category=ToolCategory.SYSTEM,
        command_prefix="!skip-discovery"
    )
    async def skip_discovery(self) -> str:
        """
        Skip the discovery process and use the default personality.

        Usage:
            !skip-discovery

        This will:
        - Create a default SOUL.md with generic personality
        - Mark bootstrap as complete
        - Allow normal agent operation to begin
        """
        if not hasattr(self.agent, 'bootstrap_service') or not self.agent.bootstrap_service:
            return "Bootstrap service not available."

        state = await self.agent.bootstrap_service.get_bootstrap_state()
        if state == BootstrapState.COMPLETE:
            return "Discovery already complete. Use !restart-discovery to start over."

        result = await self.agent.bootstrap_service.skip_discovery()

        # Reload SOUL.md into context builder
        if hasattr(self.agent, 'context_builder'):
            self.agent.context_builder._load_soul_md()

        return result

    @tool(
        name="restart_discovery",
        description="Reset and restart the personality discovery process.",
        category=ToolCategory.SYSTEM,
        command_prefix="!restart-discovery"
    )
    async def restart_discovery(self) -> str:
        """
        Reset and restart the discovery process.

        Usage:
            !restart-discovery

        This will:
        - Clear the discovery conversation history
        - Delete the existing SOUL.md
        - Reset bootstrap state to 'pending'
        - The next message will trigger the wake-up greeting
        """
        if not hasattr(self.agent, 'bootstrap_service') or not self.agent.bootstrap_service:
            return "Bootstrap service not available."

        result = await self.agent.bootstrap_service.restart_discovery()

        # Clear SOUL.md from context builder
        if hasattr(self.agent, 'context_builder'):
            self.agent.context_builder._soul_content = None

        return result

    @tool(
        name="bootstrap_status",
        description="Show the current bootstrap/discovery status.",
        category=ToolCategory.SYSTEM,
        command_prefix="!bootstrap-status"
    )
    async def bootstrap_status(self) -> str:
        """
        Show the current bootstrap status.

        Usage:
            !bootstrap-status

        Shows:
        - Current bootstrap state (pending, discovery, complete)
        - Number of discovery exchanges
        - Whether SOUL.md exists
        """
        if not hasattr(self.agent, 'bootstrap_service') or not self.agent.bootstrap_service:
            return "Bootstrap service not available."

        return await self.agent.bootstrap_service.get_bootstrap_status()

    @tool(
        name="rename_agent",
        description="Rename this agent.",
        category=ToolCategory.SYSTEM,
        command_prefix="!rename"
    )
    async def rename_agent(self, new_name: str) -> str:
        """
        Rename the agent.

        Usage:
            !rename <new_name>

        Args:
            new_name: The new name for the agent (1-64 characters)

        This will:
        - Update the agent's name in metadata
        - Update the SOUL.md header if it exists
        - The change takes effect immediately
        """
        if not new_name or not new_name.strip():
            return "Please provide a new name. Usage: !rename <new_name>"

        new_name = new_name.strip()

        # Validate name
        if len(new_name) > 64:
            return "Name too long. Maximum 64 characters."

        if len(new_name) < 1:
            return "Name too short. Minimum 1 character."

        # Get current name
        old_name = getattr(self.agent, '_agent_name', 'Unknown')

        try:
            # Update in agent metadata table
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)

            await self.agent._raw_storage.db.execute(
                """
                INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.agent.agent_id, "agent_name", new_name, now),
            )

            # Update agent node properties
            agent_node = await self.agent.storage.get_node(self.agent.agent_id)
            if agent_node:
                agent_node.properties["agent_name"] = new_name
                agent_node.label = new_name
                await self.agent.storage.add_node(agent_node)

            # Update in-memory reference
            self.agent._agent_name = new_name

            # Update bootstrap service name
            if hasattr(self.agent, 'bootstrap_service') and self.agent.bootstrap_service:
                self.agent.bootstrap_service.agent_name = new_name

            # Update SOUL.md if it exists
            soul_updated = await self._update_soul_name(old_name, new_name)

            result = f"Renamed from '{old_name}' to '{new_name}'."
            if soul_updated:
                result += " SOUL.md updated."

            logger.info(f"Agent renamed: {old_name} -> {new_name}")
            return result

        except Exception as e:
            logger.error(f"Failed to rename agent: {e}")
            return f"Failed to rename: {str(e)}"

    async def _update_soul_name(self, old_name: str, new_name: str) -> bool:
        """
        Update the agent name in SOUL.md if it exists.

        Returns:
            True if SOUL.md was updated
        """
        if not hasattr(self.agent, 'bootstrap_service') or not self.agent.bootstrap_service:
            return False

        agent_data_path = self.agent.bootstrap_service.agent_data_path
        if not agent_data_path:
            return False

        soul_path = Path(agent_data_path) / "SOUL.md"
        if not soul_path.exists():
            return False

        try:
            content = soul_path.read_text(encoding="utf-8")

            # Update the header: "# SOUL.md - You Are OldName" -> "# SOUL.md - You Are NewName"
            # Also handle variations like "# SOUL.md - OldName" or "# SOUL.md — You Are OldName"
            patterns = [
                (rf"# SOUL\.md\s*[-—]\s*You Are {re.escape(old_name)}", f"# SOUL.md - You Are {new_name}"),
                (rf"# SOUL\.md\s*[-—]\s*{re.escape(old_name)}", f"# SOUL.md - {new_name}"),
                (rf"You're {re.escape(old_name)}", f"You're {new_name}"),
                (rf"I'm {re.escape(old_name)}", f"I'm {new_name}"),
            ]

            updated = False
            for pattern, replacement in patterns:
                new_content, count = re.subn(pattern, replacement, content, flags=re.IGNORECASE)
                if count > 0:
                    content = new_content
                    updated = True

            if updated:
                soul_path.write_text(content, encoding="utf-8")

                # Reload SOUL.md into context builder
                if hasattr(self.agent, 'context_builder'):
                    self.agent.context_builder._load_soul_md()

                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to update SOUL.md name: {e}")
            return False
