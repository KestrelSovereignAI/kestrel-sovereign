"""
Bootstrap Feature - Commands for agent wake-up and discovery management.

Provides commands to control the bootstrap/discovery process:
- !skip-discovery - Skip discovery and use default personality
- !restart-discovery - Reset and redo the discovery process
- !bootstrap-status - Show current bootstrap state
- !rename - Rename the agent
- !bootstrap list - Show all loaded bootstrap files
- !bootstrap reload - Force reload all bootstrap files
- !bootstrap add <path> - Add a new bootstrap file
- !bootstrap remove <name> - Remove a bootstrap file from loading
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sovereign.bootstrap import BootstrapState

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared rename helpers (used by both !rename tool and HTTP endpoint)
# ------------------------------------------------------------------

async def rename_agent_core(agent, new_name: str) -> tuple[str, bool]:
    """
    Rename an agent, updating all stores + SOUL.md.

    Args:
        agent: KestrelAgent instance
        new_name: New name (1-64 characters, pre-stripped)

    Returns:
        (result_message, soul_updated) tuple

    Raises:
        ValueError: If name is invalid
        RuntimeError: If rename fails
    """
    if not new_name or not new_name.strip():
        raise ValueError("Name cannot be empty.")
    new_name = new_name.strip()
    if len(new_name) > 64:
        raise ValueError("Name too long. Maximum 64 characters.")
    if len(new_name) < 1:
        raise ValueError("Name too short. Minimum 1 character.")

    old_name = getattr(agent, '_agent_name', 'Unknown')

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    await agent._raw_storage.db.execute(
        """
        INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (agent.agent_id, "name", new_name, now),
    )

    # Update agent node properties
    agent_node = await agent.storage.get_node(agent.agent_id)
    if agent_node:
        agent_node.properties["name"] = new_name
        agent_node.label = new_name
        await agent.storage.add_node(agent_node)

    # Update in-memory reference
    agent._agent_name = new_name

    # Update bootstrap service name
    if hasattr(agent, 'bootstrap_service') and agent.bootstrap_service:
        agent.bootstrap_service.agent_name = new_name

    # Update SOUL.md if it exists
    soul_updated = await _update_soul_name(agent, old_name, new_name)

    logger.info(f"Agent renamed: {old_name} -> {new_name}")
    result = f"Renamed from '{old_name}' to '{new_name}'."
    if soul_updated:
        result += " SOUL.md updated."
    return result, soul_updated


async def _update_soul_name(agent, old_name: str, new_name: str) -> bool:
    """Update the agent name in SOUL.md if it exists."""
    if not hasattr(agent, 'bootstrap_service') or not agent.bootstrap_service:
        return False

    agent_data_path = agent.bootstrap_service.agent_data_path
    if not agent_data_path:
        return False

    soul_path = Path(agent_data_path) / "SOUL.md"
    if not soul_path.exists():
        return False

    try:
        content = soul_path.read_text(encoding="utf-8")

        patterns = [
            (rf"# SOUL\.md\s*[-\u2014]\s*You Are {re.escape(old_name)}", f"# SOUL.md - You Are {new_name}"),
            (rf"# SOUL\.md\s*[-\u2014]\s*{re.escape(old_name)}", f"# SOUL.md - {new_name}"),
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
            if hasattr(agent, 'context_builder'):
                agent.context_builder._load_soul_md()
            return True

        return False

    except Exception as e:
        logger.warning(f"Failed to update SOUL.md name: {e}")
        return False


class BootstrapFeature(Feature):
    """
    Feature for managing agent bootstrap, discovery, and identity.

    Provides commands for:
    - Skipping or restarting the discovery process
    - Checking bootstrap status
    - Renaming the agent
    - Listing, reloading, adding, and removing bootstrap files
    """

    def __init__(self, agent):
        super().__init__(agent)

    @property
    def tool_description(self) -> str:
        """Description for A2A agent card."""
        return (
            "Manage agent identity and bootstrap files - "
            "skip/restart discovery, check status, rename agent, "
            "list/reload/add/remove bootstrap files"
        )

    async def initialize(self):
        """Initialize the bootstrap feature."""
        logger.info("BootstrapFeature initialized")

    # ------------------------------------------------------------------
    # Bootstrap file management tools
    # ------------------------------------------------------------------

    @tool(
        name="bootstrap_list",
        description="Show all loaded bootstrap files and their paths, sizes, and status.",
        category=ToolCategory.SYSTEM,
        command_prefix="!bootstrap list"
    )
    async def bootstrap_list(self) -> Dict[str, Any]:
        """
        List all bootstrap files and their loading status.

        Usage:
            !bootstrap list

        Shows each configured bootstrap file with:
        - Filename
        - Resolved path on disk
        - Character count
        - Load status (loaded, not found, skipped)
        """
        loader = self._get_loader()
        if loader is None:
            return {
                "success": False,
                "error": "Bootstrap loader not available (no context_builder).",
            }

        files = loader.list_files()
        return {
            "success": True,
            "files": files,
            "total_files": loader.file_count,
            "total_chars": loader.total_chars,
            "file_order": loader.file_order,
        }

    @tool(
        name="bootstrap_reload",
        description="Force reload all bootstrap files from disk. Use after editing SOUL.md or other bootstrap files.",
        category=ToolCategory.SYSTEM,
        command_prefix="!bootstrap reload"
    )
    async def bootstrap_reload(self) -> Dict[str, Any]:
        """
        Force reload all bootstrap files from disk.

        Usage:
            !bootstrap reload

        This re-reads every bootstrap file, picking up any edits
        made since the agent started.
        """
        loader = self._get_loader()
        if loader is None:
            return {
                "success": False,
                "error": "Bootstrap loader not available (no context_builder).",
            }

        loader.reload()

        # Also refresh the context_builder cache if it wraps the loader
        if hasattr(self.agent, 'context_builder'):
            cb = self.agent.context_builder
            if hasattr(cb, 'reload_bootstrap_files'):
                cb.reload_bootstrap_files()

        return {
            "success": True,
            "message": "Bootstrap files reloaded.",
            "loaded_count": loader.file_count,
            "total_chars": loader.total_chars,
            "files": [f["name"] for f in loader.list_files() if f["status"] == "loaded"],
        }

    @tool(
        name="bootstrap_add",
        description="Add a new bootstrap file to be loaded at startup.",
        category=ToolCategory.SYSTEM,
        command_prefix="!bootstrap add"
    )
    async def bootstrap_add(self, file_path: str) -> Dict[str, Any]:
        """
        Add a new bootstrap file to the loading convention.

        Usage:
            !bootstrap add <path>

        Args:
            file_path: Path to the file to add (absolute, or relative to agent data dir)

        The file is appended to the load order and will be included
        in the agent's system prompt on next reload.
        """
        loader = self._get_loader()
        if loader is None:
            return {
                "success": False,
                "error": "Bootstrap loader not available (no context_builder).",
            }

        # Resolve the path
        resolved = Path(file_path)
        if not resolved.is_absolute():
            # Try relative to agent data path
            agent_data = self._get_agent_data_path()
            if agent_data:
                resolved = Path(agent_data) / file_path
            else:
                return {
                    "success": False,
                    "error": f"Cannot resolve relative path '{file_path}' -- no agent data path configured.",
                }

        filename = resolved.name

        # Check for duplicate before checking file existence
        if filename in loader.file_order:
            return {
                "success": False,
                "error": f"File '{filename}' is already in the bootstrap file list.",
            }

        if not resolved.exists():
            return {
                "success": False,
                "error": f"File not found: {resolved}",
            }

        loader.add_file(filename)

        # Persist to DB if available
        db = self._get_db()
        agent_id = self.agent.did
        if db and agent_id:
            try:
                await loader.save_db_entry(
                    file_name=filename,
                    file_path=str(resolved),
                    enabled=True,
                    priority=100 + len(loader.file_order),
                )
            except Exception as e:
                logger.warning(f"Failed to persist bootstrap config to DB: {e}")

        # Reload to pick up the new file
        loader.reload()

        return {
            "success": True,
            "message": f"Added '{filename}' to bootstrap files.",
            "loaded": filename in loader.get_bootstrap_content(),
        }

    @tool(
        name="bootstrap_remove",
        description="Remove a bootstrap file from the loading convention.",
        category=ToolCategory.SYSTEM,
        command_prefix="!bootstrap remove"
    )
    async def bootstrap_remove(self, name: str) -> Dict[str, Any]:
        """
        Remove a bootstrap file from loading.

        Usage:
            !bootstrap remove <name>

        Args:
            name: Name of the file to remove (e.g. "GOALS.md")

        The file itself is not deleted from disk -- it is just
        removed from the list of files loaded into the system prompt.
        """
        loader = self._get_loader()
        if loader is None:
            return {
                "success": False,
                "error": "Bootstrap loader not available (no context_builder).",
            }

        removed = loader.remove_file(name)
        if not removed:
            return {
                "success": False,
                "error": f"File '{name}' is not in the bootstrap file list.",
                "available": loader.file_order,
            }

        # Remove from DB if available
        db = self._get_db()
        agent_id = self.agent.did
        if db and agent_id:
            try:
                await loader.delete_db_entry(name)
            except Exception as e:
                logger.warning(f"Failed to remove bootstrap config from DB: {e}")

        # Refresh context builder
        if hasattr(self.agent, 'context_builder'):
            cb = self.agent.context_builder
            if hasattr(cb, 'reload_bootstrap_files'):
                cb.reload_bootstrap_files()

        return {
            "success": True,
            "message": f"Removed '{name}' from bootstrap files.",
            "remaining": loader.file_order,
        }

    # ------------------------------------------------------------------
    # Existing discovery/identity tools
    # ------------------------------------------------------------------

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

        try:
            result, _ = await rename_agent_core(self.agent, new_name)
            return result
        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.error(f"Failed to rename agent: {e}")
            return f"Failed to rename: {str(e)}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_loader(self):
        """Get the BootstrapLoader from the context builder, if available."""
        cb = getattr(self.agent, 'context_builder', None)
        if cb is None:
            return None
        return getattr(cb, '_bootstrap_loader', None)

    def _get_agent_data_path(self) -> Optional[str]:
        """Get the agent data path from bootstrap service or context builder."""
        bs = getattr(self.agent, 'bootstrap_service', None)
        if bs and hasattr(bs, 'agent_data_path') and bs.agent_data_path:
            return str(bs.agent_data_path)
        cb = getattr(self.agent, 'context_builder', None)
        if cb and hasattr(cb, 'agent_data_path') and cb.agent_data_path:
            return str(cb.agent_data_path)
        return None

    def _get_db(self):
        """Get the async database handle if available."""
        raw = getattr(self.agent, '_raw_storage', None)
        if raw and hasattr(raw, 'db'):
            return raw.db
        return None

    async def _update_soul_name(self, old_name: str, new_name: str) -> bool:
        """Update the agent name in SOUL.md. Delegates to module-level helper."""
        return await _update_soul_name(self.agent, old_name, new_name)
