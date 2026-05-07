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

@tool methods return ``kestrel_sdk.tools.result.ToolResult`` per the
kestrel-sovereign #1042 narration-honesty contract (#1061).
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
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
    async def bootstrap_list(self) -> ToolResult:
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
            return ToolResult.failed(
                "Bootstrap loader not available (no context_builder).",
            )

        files = loader.list_files()
        loaded = sum(1 for f in files if f.get("status") == "loaded")
        return ToolResult.ok(
            confirmation=(
                f"Bootstrap catalog: {loader.file_count} file(s) configured, "
                f"{loaded} loaded, {loader.total_chars} char(s) total"
            ),
            data={
                "files": files,
                "total_files": loader.file_count,
                "total_chars": loader.total_chars,
                "file_order": loader.file_order,
                "loaded_count": loaded,
            },
        )

    @tool(
        name="bootstrap_reload",
        description="Force reload all bootstrap files from disk. Use after editing SOUL.md or other bootstrap files.",
        category=ToolCategory.SYSTEM,
        command_prefix="!bootstrap reload"
    )
    async def bootstrap_reload(self) -> ToolResult:
        """
        Force reload all bootstrap files from disk.

        Usage:
            !bootstrap reload

        This re-reads every bootstrap file, picking up any edits
        made since the agent started.
        """
        loader = self._get_loader()
        if loader is None:
            return ToolResult.failed(
                "Bootstrap loader not available (no context_builder).",
            )

        try:
            loader.reload()
        except Exception as e:
            logger.error(f"bootstrap_reload failed during loader.reload(): {e}", exc_info=True)
            return ToolResult.failed(str(e))

        # Also refresh the context_builder cache if it wraps the loader
        if hasattr(self.agent, 'context_builder'):
            cb = self.agent.context_builder
            if hasattr(cb, 'reload_bootstrap_files'):
                try:
                    cb.reload_bootstrap_files()
                except Exception as e:
                    # The loader did reload; the context_builder cache
                    # refresh is the secondary side-effect. Surface as
                    # PARTIAL so the LLM can't claim a clean reload
                    # while a stale cache is still serving prompt
                    # assembly.
                    logger.error(
                        f"bootstrap_reload: loader reloaded but "
                        f"context_builder cache refresh failed: {e}",
                        exc_info=True,
                    )
                    files = loader.list_files()
                    loaded = [f["name"] for f in files if f.get("status") == "loaded"]
                    return ToolResult.partial(
                        confirmation=(
                            f"Reloaded {loader.file_count} file(s) from disk; "
                            f"{loader.total_chars} char(s) total"
                        ),
                        error=(
                            f"context_builder cache refresh failed: {e}; "
                            "stale prompt assembly until next agent restart"
                        ),
                        data={
                            "loaded_count": loader.file_count,
                            "total_chars": loader.total_chars,
                            "files": loaded,
                            "cache_refresh_error": str(e),
                        },
                    )

        files = loader.list_files()
        loaded = [f["name"] for f in files if f.get("status") == "loaded"]
        # Honesty: BootstrapLoader.reload() catches per-file read
        # exceptions and silently drops the file from the cache.
        # ``list_files()`` reports those (and budget-exhausted entries)
        # as ``status == "skipped (budget)"``: the file is present on
        # disk but isn't in the prompt. ``status == "not found"``
        # means the file genuinely isn't on disk — that's the normal
        # state for the default-optional bootstrap files (#659) and
        # not a failure. (Round 3 codex finding.)
        dropped = [
            f["name"] for f in files
            if (f.get("status") or "").startswith("skipped")
        ]
        if dropped:
            return ToolResult.partial(
                confirmation=(
                    f"Reloaded {len(loaded)} file(s) from disk; "
                    f"{loader.total_chars} char(s) total"
                ),
                error=(
                    f"{len(dropped)} configured file(s) exist on disk "
                    f"but were dropped from the prompt "
                    f"(read failure or budget exhausted): "
                    f"{', '.join(dropped)}"
                ),
                data={
                    "loaded_count": loader.file_count,
                    "total_chars": loader.total_chars,
                    "files": loaded,
                    "dropped_files": dropped,
                },
            )
        return ToolResult.ok(
            confirmation=(
                f"Reloaded {loader.file_count} file(s) from disk; "
                f"{len(loaded)} successfully loaded, "
                f"{loader.total_chars} char(s) total"
            ),
            data={
                "loaded_count": loader.file_count,
                "total_chars": loader.total_chars,
                "files": loaded,
            },
        )

    @tool(
        name="bootstrap_add",
        description="Add a new bootstrap file to be loaded at startup.",
        category=ToolCategory.SYSTEM,
        command_prefix="!bootstrap add"
    )
    async def bootstrap_add(self, file_path: str) -> ToolResult:
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
            return ToolResult.failed(
                "Bootstrap loader not available (no context_builder).",
            )

        # Resolve the path
        resolved = Path(file_path)
        if not resolved.is_absolute():
            agent_data = self._get_agent_data_path()
            if agent_data:
                resolved = Path(agent_data) / file_path
            else:
                return ToolResult.failed(
                    f"Cannot resolve relative path '{file_path}' -- "
                    "no agent data path configured.",
                )

        filename = resolved.name

        # Check for duplicate before checking file existence
        if filename in loader.file_order:
            return ToolResult.failed(
                f"File '{filename}' is already in the bootstrap file list.",
                data={"file_order": list(loader.file_order)},
            )

        if not resolved.exists():
            return ToolResult.failed(
                f"File not found: {resolved}",
                data={"resolved_path": str(resolved)},
            )

        loader.add_file(filename)

        # Persist to DB if available.
        #
        # Honesty: ``loader.save_db_entry`` silently no-ops when the
        # loader was constructed without a ``db`` connection — that's
        # the normal ContextBuilder path. Treating "agent has a DB"
        # as "persistence happened" overstates durability — the entry
        # is in-memory only and won't survive an agent restart.
        # We mirror the loader's own gate (private fields ``_db`` /
        # ``_agent_id``) so we only claim persistence when the call
        # would actually write a row.
        db_attempted = bool(getattr(loader, "_db", None) and getattr(loader, "_agent_id", None))
        db_persisted = False
        db_persist_error: Optional[str] = None
        if db_attempted:
            try:
                await loader.save_db_entry(
                    file_name=filename,
                    file_path=str(resolved),
                    enabled=True,
                    priority=100 + len(loader.file_order),
                )
                db_persisted = True
            except Exception as e:
                logger.warning(f"Failed to persist bootstrap config to DB: {e}")
                db_persist_error = str(e)

        # Reload to pick up the new file
        loader.reload()

        loaded = filename in loader.get_bootstrap_content()

        # Honesty: the loader keys files by basename. If the user
        # passes an absolute path outside the search roots and a
        # *different* file with the same basename also exists under
        # ``agent_data`` / ``extra_paths``, the loader will populate
        # the prompt from the search-root file, not from ``resolved``.
        # Surface that mismatch as PARTIAL — the entry is technically
        # loaded but the prompt reads a different file than the DB
        # row records.
        actual_loaded_path: Optional[str] = None
        try:
            actual_loaded_path = loader._resolved_paths.get(filename)  # noqa: SLF001
        except Exception:
            actual_loaded_path = None
        path_mismatch = bool(
            loaded
            and actual_loaded_path
            and Path(actual_loaded_path).resolve() != resolved.resolve()
        )

        # Common data fields shared across the OK/PARTIAL branches.
        _data = {
            "filename": filename,
            "resolved_path": str(resolved),
            "actual_loaded_path": actual_loaded_path,
            "loaded": loaded,
            "path_mismatch": path_mismatch,
            "db_attempted": db_attempted,
            "db_persisted": db_persisted,
            "db_persist_error": db_persist_error,
        }
        if path_mismatch:
            return ToolResult.partial(
                confirmation=(
                    f"Registered '{filename}' in the bootstrap file list"
                ),
                error=(
                    f"basename collision: prompt is loading '{filename}' from "
                    f"{actual_loaded_path}, not from the path you passed "
                    f"({resolved}); the DB row references a different path "
                    "than the cached content"
                ),
                data=_data,
            )
        if not loaded:
            # The loader accepted the entry but the file's content was
            # not pulled into the bootstrap content (read failure,
            # disabled, etc.). Surface as PARTIAL so the LLM doesn't
            # claim a clean add when the file isn't actually in the
            # prompt.
            return ToolResult.partial(
                confirmation=(
                    f"Registered '{filename}' in the bootstrap file list"
                ),
                error=(
                    f"file '{filename}' is in the load order but its "
                    "content was not pulled into the bootstrap prompt; "
                    "check the loader's file status"
                ),
                data=_data,
            )
        if not db_attempted:
            # The loader has no DB wiring — the entry is in-memory
            # only. Mirrors the in-process ContextBuilder default
            # construction. PARTIAL forces the LLM to surface the
            # restart-loss caveat.
            return ToolResult.partial(
                confirmation=(
                    f"Added '{filename}' to bootstrap files in memory "
                    "(loaded into prompt)"
                ),
                error=(
                    "loader has no DB wiring; the entry is in-memory "
                    "only and will not survive an agent restart"
                ),
                data=_data,
            )
        if not db_persisted:
            return ToolResult.partial(
                confirmation=(
                    f"Added '{filename}' to bootstrap files (loaded into prompt)"
                ),
                error=(
                    f"DB persistence failed: {db_persist_error}; the entry "
                    "will not survive an agent restart"
                ),
                data=_data,
            )
        return ToolResult.ok(
            confirmation=(
                f"Added '{filename}' to bootstrap files "
                "(loaded into prompt, persisted to DB)"
            ),
            data=_data,
        )

    @tool(
        name="bootstrap_remove",
        description="Remove a bootstrap file from the loading convention.",
        category=ToolCategory.SYSTEM,
        command_prefix="!bootstrap remove"
    )
    async def bootstrap_remove(self, name: str) -> ToolResult:
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
            return ToolResult.failed(
                "Bootstrap loader not available (no context_builder).",
            )

        removed = loader.remove_file(name)
        if not removed:
            return ToolResult.failed(
                f"File '{name}' is not in the bootstrap file list.",
                data={"available": list(loader.file_order)},
            )

        # Remove from DB if the loader has DB wiring. Same honesty
        # gate as bootstrap_add — the loader's delete_db_entry no-ops
        # when constructed without a db, so we can't claim a DB
        # delete happened just because the agent has a DB connection.
        db_attempted = bool(
            getattr(loader, "_db", None) and getattr(loader, "_agent_id", None)
        )
        db_removed = False
        db_remove_error: Optional[str] = None
        if db_attempted:
            try:
                await loader.delete_db_entry(name)
                db_removed = True
            except Exception as e:
                logger.warning(f"Failed to remove bootstrap config from DB: {e}")
                db_remove_error = str(e)

        # Refresh context builder
        cache_refreshed = True
        cache_refresh_error: Optional[str] = None
        if hasattr(self.agent, 'context_builder'):
            cb = self.agent.context_builder
            if hasattr(cb, 'reload_bootstrap_files'):
                try:
                    cb.reload_bootstrap_files()
                except Exception as e:
                    logger.warning(f"Failed to refresh context_builder cache: {e}")
                    cache_refreshed = False
                    cache_refresh_error = str(e)

        _data = {
            "name": name,
            "remaining": list(loader.file_order),
            "db_attempted": db_attempted,
            "db_removed": db_removed,
            "cache_refreshed": cache_refreshed,
            "db_remove_error": db_remove_error,
            "cache_refresh_error": cache_refresh_error,
        }

        # PARTIAL conditions: any secondary side-effect failed, or the
        # loader has no DB wiring (in-memory remove only — a stale
        # row may persist in DB across restart).
        err_parts = []
        if not cache_refreshed:
            err_parts.append(f"cache refresh failed: {cache_refresh_error}")
        if db_attempted and not db_removed:
            err_parts.append(f"DB removal failed: {db_remove_error}")
        elif not db_attempted:
            err_parts.append(
                "loader has no DB wiring; in-memory remove only — "
                "any DB-persisted row for this file will resurface on restart"
            )
        if err_parts:
            return ToolResult.partial(
                confirmation=(
                    f"Removed '{name}' from in-memory load order"
                ),
                error="; ".join(err_parts),
                data=_data,
            )

        return ToolResult.ok(
            confirmation=(
                f"Removed '{name}' from bootstrap files (DB row deleted)"
            ),
            data=_data,
        )

    # ------------------------------------------------------------------
    # Existing discovery/identity tools
    # ------------------------------------------------------------------

    @tool(
        name="skip_discovery",
        description="Skip the discovery conversation and use default personality.",
        category=ToolCategory.SYSTEM,
        command_prefix="!skip-discovery"
    )
    async def skip_discovery(self) -> ToolResult:
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
            return ToolResult.failed("Bootstrap service not available.")

        try:
            state = await self.agent.bootstrap_service.get_bootstrap_state()
        except Exception as e:
            logger.error(f"skip_discovery state lookup failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if state == BootstrapState.COMPLETE:
            # Honesty: this is a no-op for the bootstrap state — but a
            # previous skip may have marked COMPLETE while
            # save_soul_md() failed (no agent_data_path, write error).
            # The user-visible "already complete" framing implies a
            # working SOUL.md; we must surface the missing file so the
            # operator knows the durable artifact isn't there.
            # (Round 4 codex finding.)
            soul_exists, soul_check_path = self._verify_soul_md_exists()
            state_str = state.value if hasattr(state, "value") else str(state)
            if not soul_exists:
                return ToolResult.partial(
                    confirmation=(
                        "Bootstrap state is already COMPLETE; nothing to skip "
                        "(use !restart-discovery to start over)"
                    ),
                    error=(
                        f"SOUL.md is missing on disk "
                        f"(checked: {soul_check_path or 'no agent_data_path configured'}); "
                        "a previous skip likely failed to write it. "
                        "The agent has no persisted personality."
                    ),
                    data={
                        "state": state_str,
                        "soul_exists": False,
                        "soul_check_path": soul_check_path,
                    },
                )
            return ToolResult.ok(
                confirmation=(
                    "Discovery already complete; nothing to skip "
                    "(use !restart-discovery to start over)"
                ),
                data={
                    "state": state_str,
                    "soul_exists": True,
                },
            )

        try:
            result = await self.agent.bootstrap_service.skip_discovery()
        except Exception as e:
            logger.error(f"skip_discovery failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        # Honesty: BootstrapService.skip_discovery() marks state
        # COMPLETE even when save_soul_md() failed (missing
        # agent_data_path, write error, etc.). The user-visible
        # message says "personality saved" but no SOUL.md exists.
        # Verify the file before claiming success. (Round 3
        # codex finding.)
        soul_exists, soul_check_path = self._verify_soul_md_exists()

        # Reload SOUL.md into context builder
        cb_reload_error: Optional[str] = None
        if hasattr(self.agent, 'context_builder'):
            try:
                self.agent.context_builder._load_soul_md()
            except Exception as e:
                logger.error(
                    f"skip_discovery: bootstrap skipped but SOUL.md "
                    f"reload failed: {e}",
                    exc_info=True,
                )
                cb_reload_error = str(e)

        if not soul_exists:
            return ToolResult.partial(
                confirmation=(
                    f"Bootstrap state set to COMPLETE: "
                    f"{str(result) if result else 'Discovery skipped'}"
                ),
                error=(
                    f"SOUL.md was not written to disk "
                    f"(checked: {soul_check_path or 'no agent_data_path configured'}); "
                    "the default personality will not take effect until "
                    "the file is created and the agent restarts"
                ),
                data={
                    "service_result": str(result),
                    "soul_exists": False,
                    "soul_check_path": soul_check_path,
                    "cb_reload_error": cb_reload_error,
                },
            )

        if cb_reload_error:
            return ToolResult.partial(
                confirmation=str(result) if result else "Discovery skipped",
                error=(
                    f"SOUL.md reload into context_builder failed: "
                    f"{cb_reload_error}; "
                    "personality will not take effect until next "
                    "agent restart"
                ),
                data={
                    "service_result": str(result),
                    "soul_exists": True,
                    "cb_reload_error": cb_reload_error,
                },
            )

        return ToolResult.ok(
            confirmation=str(result) if result else "Discovery skipped",
            data={
                "service_result": str(result),
                "soul_exists": True,
            },
        )

    @tool(
        name="restart_discovery",
        description="Reset and restart the personality discovery process.",
        category=ToolCategory.SYSTEM,
        command_prefix="!restart-discovery"
    )
    async def restart_discovery(self) -> ToolResult:
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
            return ToolResult.failed("Bootstrap service not available.")

        try:
            result = await self.agent.bootstrap_service.restart_discovery()
        except Exception as e:
            logger.error(f"restart_discovery failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        # Clear SOUL.md from context builder
        if hasattr(self.agent, 'context_builder'):
            try:
                self.agent.context_builder._soul_content = None
            except Exception as e:
                # Setting an attribute almost never fails, but if it
                # somehow did the reset is incomplete.
                logger.error(
                    f"restart_discovery: bootstrap reset but SOUL.md "
                    f"cache clear failed: {e}",
                    exc_info=True,
                )
                return ToolResult.partial(
                    confirmation=str(result) if result else "Discovery restarted",
                    error=(
                        f"SOUL.md cache clear failed: {e}; "
                        "stale personality may persist until restart"
                    ),
                    data={"service_result": str(result)},
                )

        return ToolResult.ok(
            confirmation=str(result) if result else "Discovery restarted",
            data={"service_result": str(result)},
        )

    @tool(
        name="bootstrap_status",
        description="Show the current bootstrap/discovery status.",
        category=ToolCategory.SYSTEM,
        command_prefix="!bootstrap-status"
    )
    async def bootstrap_status(self) -> ToolResult:
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
            return ToolResult.failed("Bootstrap service not available.")

        try:
            status = await self.agent.bootstrap_service.get_bootstrap_status()
        except Exception as e:
            logger.error(f"bootstrap_status failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return ToolResult.ok(
            confirmation=str(status) if status else "Bootstrap status retrieved",
            data={"status": str(status)},
        )

    @tool(
        name="rename_agent",
        description="Rename this agent.",
        category=ToolCategory.SYSTEM,
        command_prefix="!rename"
    )
    async def rename_agent(self, new_name: str) -> ToolResult:
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
        if not isinstance(new_name, str):
            return ToolResult.failed(
                f"new_name must be a string, got "
                f"{type(new_name).__name__}={new_name!r}"
            )
        if not new_name or not new_name.strip():
            return ToolResult.failed(
                "Please provide a new name. Usage: !rename <new_name>"
            )

        try:
            result, soul_updated = await rename_agent_core(self.agent, new_name)
        except ValueError as e:
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"Failed to rename agent: {e}")
            return ToolResult.failed(f"Failed to rename: {str(e)}")

        return ToolResult.ok(
            confirmation=str(result),
            data={
                "new_name": new_name.strip(),
                "soul_updated": soul_updated,
            },
        )

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

    def _verify_soul_md_exists(self) -> tuple[bool, Optional[str]]:
        """Check whether SOUL.md exists at the agent's data path.

        Used by skip_discovery to verify the bootstrap_service
        actually wrote the personality file before reporting OK.
        Returns ``(exists, checked_path_or_None)``. ``checked_path``
        is None when no agent_data_path is configured (in which
        case a SOUL.md write would have been impossible).
        """
        agent_data = self._get_agent_data_path()
        if not agent_data:
            return False, None
        soul_path = Path(agent_data) / "SOUL.md"
        try:
            return soul_path.exists(), str(soul_path)
        except Exception:
            return False, str(soul_path)

    def _get_db(self):
        """Get the async database handle if available."""
        raw = getattr(self.agent, '_raw_storage', None)
        if raw and hasattr(raw, 'db'):
            return raw.db
        return None

    async def _update_soul_name(self, old_name: str, new_name: str) -> bool:
        """Update the agent name in SOUL.md. Delegates to module-level helper."""
        return await _update_soul_name(self.agent, old_name, new_name)
