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

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.bootstrap import BootstrapState
from kestrel_sovereign.bootstrap.soul_durability import inspect_soul_file
from kestrel_sovereign.storage.agent_resource_store import (
    ExpectedCurrentVersion,
    ResourceVersionConflictError,
)
from kestrel_sovereign.storage.privacy_wrapper import (
    acquire_control_plane_capability,
    optional_transition_lock,
    _resolve_transition_lock,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared rename helpers (used by both !rename tool and HTTP endpoint)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RenameOutcome:
    success: bool
    db_row_written: bool
    graph_updated: bool
    memory_updated: bool
    soul_md_updated: bool
    error: Optional[str] = None
    # A new agent name is user-derived identity input. In a volatile privacy mode
    # ("leave no trace" / session-only) rename persists NOTHING durable and only
    # updates the live in-memory name; this flag says so explicitly so callers
    # report a deliberate skip rather than a failure or a false success (#2672 P1).
    skipped_privacy: bool = False

    @property
    def any_written(self) -> bool:
        return any((
            self.db_row_written,
            self.graph_updated,
            self.memory_updated,
            self.soul_md_updated,
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "db_row_written": self.db_row_written,
            "graph_updated": self.graph_updated,
            "memory_updated": self.memory_updated,
            "soul_md_updated": self.soul_md_updated,
            "skipped_privacy": self.skipped_privacy,
            "error": self.error,
        }


def _rename_confirmation(old_name: str, new_name: str, outcome: RenameOutcome) -> str:
    if outcome.skipped_privacy:
        return (
            f"Now going by '{new_name}' for this session (was '{old_name}'). "
            "The current privacy mode leaves no durable trace, so the name was "
            "NOT saved to metadata, the graph, or SOUL.md — it reverts on restart."
        )
    result = f"Renamed from '{old_name}' to '{new_name}'."
    if outcome.soul_md_updated:
        result += " SOUL.md updated."
    return result


def _rename_partial_error(outcome: RenameOutcome) -> str:
    caveats = []
    if outcome.db_row_written:
        caveats.append("metadata DB has the new name")
    else:
        caveats.append("metadata DB still has the old name")
    if outcome.graph_updated:
        caveats.append("agent graph node has the new name")
    else:
        caveats.append("agent graph node may still have the old name")
    if outcome.memory_updated:
        caveats.append("live agent memory has the new name")
    else:
        caveats.append("live agent memory still has the old name")
    if outcome.soul_md_updated:
        caveats.append("SOUL.md has the new name")
    else:
        caveats.append("SOUL.md was not updated")

    detail = "; ".join(caveats)
    if outcome.error:
        detail += f"; error: {outcome.error}"
    return (
        f"Rename partially applied: {detail}. "
        "Restart the agent or call !rename again after fixing the failed store."
    )


async def rename_agent_core(agent, new_name: str) -> RenameOutcome:
    """
    Rename an agent, updating all stores + SOUL.md.

    Args:
        agent: KestrelAgent instance
        new_name: New name (1-64 characters, pre-stripped)

    Returns:
        RenameOutcome with per-store progress flags.

    Raises:
        ValueError: If name is invalid
    """
    if not new_name or not new_name.strip():
        raise ValueError("Name cannot be empty.")
    new_name = new_name.strip()
    if len(new_name) > 64:
        raise ValueError("Name too long. Maximum 64 characters.")
    if len(new_name) < 1:
        raise ValueError("Name too short. Minimum 1 character.")

    old_name = getattr(agent, '_agent_name', 'Unknown')

    # Serialize the privacy check and every durable rename write under the agent's
    # privacy-transition lock so a concurrent ``set_privacy_mode`` cannot flip the
    # mode into an ``await`` gap between the check and the writes and persist the
    # new name after the mode became volatile (#2672 review P1 race). ``None`` on
    # test/CLI shapes with no running agent → unguarded.
    async with optional_transition_lock(_resolve_transition_lock(agent)):
        return await _rename_agent_core_locked(agent, old_name, new_name)


async def _rename_agent_core_locked(agent, old_name: str, new_name: str) -> RenameOutcome:
    """The rename check-and-write body, run under the privacy-transition lock.

    Split from :func:`rename_agent_core` so the privacy check and every durable
    write happen inside a single held lock (#2672 review P1 race). ``old_name`` and
    the validated ``new_name`` are passed in; the caller owns the lock.
    """
    # Privacy boundary (#2672 review P1): the new name is user-derived identity
    # input. In a volatile privacy mode, persist NOTHING durable — not the
    # ``agent_metadata`` row (the direct write that bypassed the graph boundary),
    # not the ``agent`` graph node, not SOUL.md. Gate BEFORE any side effect and
    # update only the live in-memory name so the running session reflects it, then
    # report an explicit skipped outcome (never a partial/false success). This is
    # the same source-of-truth gating used by ``persist_agent_description`` and
    # ``save_soul_md``.
    from kestrel_sovereign.features.storage_access import (
        hides_persisted_user_content,
    )
    if hides_persisted_user_content(agent):
        agent._agent_name = new_name
        if hasattr(agent, 'bootstrap_service') and agent.bootstrap_service:
            agent.bootstrap_service.agent_name = new_name
        logger.info(
            "Agent rename not persisted — durable identity writes are disabled "
            "in the current privacy mode (#2672); in-memory name updated only."
        )
        return RenameOutcome(
            success=False,
            db_row_written=False,
            graph_updated=False,
            memory_updated=True,
            soul_md_updated=False,
            skipped_privacy=True,
        )

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    db_row_written = False
    graph_updated = False
    memory_updated = False
    soul_md_updated = False

    # The SOUL first, before the name is written anywhere (#3813). The rename
    # is refused unless the SOUL version it read is still current, whether or
    # not that SOUL names the agent, and a refusal fails the rename here,
    # before the metadata, the graph or the live agent has the new name. A rename run again then takes the old
    # name from the live agent and finds it in the SOUL. Renamed last, a
    # refused SOUL left the new name live, so running the rename again renamed
    # the new name to itself and reported success over a SOUL still naming
    # the old one. A store failing after the SOUL was renamed leaves a SOUL
    # that already has the new name, which the next rename leaves as it is.
    try:
        soul_md_updated = await _update_soul_name(agent, old_name, new_name)
    except Exception as e:
        if is_execution_control_error(e):
            raise
        _raise_carried_cancellation(e)
        if isinstance(e, ResourceVersionConflictError):
            logger.warning("Agent rename refused: %s", e)
            error = (
                "the SOUL changed while the rename ran, so the rename was "
                f"refused and the name was changed nowhere ({e}). Rename again "
                "to rename the current SOUL."
            )
        else:
            logger.error(f"Agent rename failed during SOUL.md update: {e}", exc_info=True)
            error = f"SOUL.md update failed: {e}"
        return RenameOutcome(
            success=False,
            db_row_written=False,
            graph_updated=False,
            memory_updated=False,
            soul_md_updated=False,
            error=error,
        )

    try:
        await agent._raw_storage.db.execute(
            """
            INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (agent.agent_id, "name", new_name, now),
        )
        db_row_written = True
    except Exception as e:
        if is_execution_control_error(e):
            raise
        _raise_carried_cancellation(e)
        logger.error(f"Agent rename failed at the metadata write: {e}", exc_info=True)
        return RenameOutcome(
            success=False,
            db_row_written=False,
            graph_updated=False,
            memory_updated=False,
            soul_md_updated=soul_md_updated,
            error=f"metadata DB write failed: {e}",
        )

    # Update agent node properties
    try:
        from kestrel_sovereign.storage.identity_metadata import merge_identity_metadata

        updated_node = await merge_identity_metadata(
            agent.storage, agent.agent_id, {"name": new_name}, label=new_name,
            capability=acquire_control_plane_capability(),
        )
        graph_updated = updated_node is not None
    except Exception as e:
        if is_execution_control_error(e):
            raise
        _raise_carried_cancellation(e)
        logger.error(f"Agent rename failed after metadata write: {e}", exc_info=True)
        return RenameOutcome(
            success=False,
            db_row_written=db_row_written,
            graph_updated=False,
            memory_updated=False,
            soul_md_updated=soul_md_updated,
            error=f"graph update failed: {e}",
        )

    # Update in-memory reference
    try:
        agent._agent_name = new_name

        # Update bootstrap service name
        if hasattr(agent, 'bootstrap_service') and agent.bootstrap_service:
            agent.bootstrap_service.agent_name = new_name
        memory_updated = True
    except Exception as e:
        if is_execution_control_error(e):
            raise
        logger.error(f"Agent rename failed after graph update: {e}", exc_info=True)
        return RenameOutcome(
            success=False,
            db_row_written=db_row_written,
            graph_updated=graph_updated,
            memory_updated=False,
            soul_md_updated=soul_md_updated,
            error=f"in-memory update failed: {e}",
        )

    logger.info(f"Agent renamed: {old_name} -> {new_name}")
    return RenameOutcome(
        success=db_row_written and graph_updated and memory_updated,
        db_row_written=db_row_written,
        graph_updated=graph_updated,
        memory_updated=memory_updated,
        soul_md_updated=soul_md_updated,
        error=None if graph_updated else "agent graph node not found; metadata and memory were updated",
    )


def _raise_carried_cancellation(error: BaseException) -> None:
    """Raise the cancellation ``error`` carries, if this task is being cancelled.

    A storage write that fails while its task is cancelled can report its own
    error with the cancellation as its explicit cause: SQLite does when the
    cancellation arrives while it rolls back a refused SOUL write. The rename
    was cancelled then, and reporting it as a failed rename would hide that
    (#3813). A task not being cancelled was not: an expired
    ``asyncio.timeout`` raises ``TimeoutError`` from the cancellation it used
    to stop the work, and that failure is the rename's to report.
    """
    task = asyncio.current_task()
    if task is None or not task.cancelling():
        return
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, asyncio.CancelledError):
            raise current
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def _renamed_soul(content: str, old_name: str, new_name: str) -> str:
    """``content`` with the SOUL's references to ``old_name`` renamed."""
    patterns = [
        (rf"# SOUL\.md\s*[-—]\s*You Are {re.escape(old_name)}", f"# SOUL.md - You Are {new_name}"),
        (rf"# SOUL\.md\s*[-—]\s*{re.escape(old_name)}", f"# SOUL.md - {new_name}"),
        (rf"You're {re.escape(old_name)}", f"You're {new_name}"),
        (rf"I'm {re.escape(old_name)}", f"I'm {new_name}"),
    ]
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    return content


async def _update_soul_name(agent, old_name: str, new_name: str) -> bool:
    """Update the agent name in its canonical SOUL and in SOUL.md.

    The canonical SOUL, the version every boot serves, is what is renamed,
    not whatever SOUL.md holds. The file is only its seed and cache, and
    text read from it may predate a revision another connection has since
    made canonical (#3813). Before anything is written, the store checks,
    under the version lock a revision publishes under, that the version read
    is still current; it checks a SOUL with nothing to rename too, because
    the caller then renames the agent everywhere else. If another writer
    made a new version current first, the rename is refused: it is not
    retried, and nothing is written.

    A SOUL.md holding the canonical text gets the renamed text. A SOUL.md
    holding an edit that was never made canonical is renamed in place and
    stays an uncommitted edit: a rename does not make it canonical. With no
    canonical SOUL, the renamed file is published as boot would publish it.

    Returns:
        Whether the canonical SOUL or SOUL.md changed.

    Raises:
        ResourceVersionConflictError: another writer made a new SOUL version
            current after the rename read the SOUL. Neither the canonical SOUL
            nor SOUL.md was changed.
    """
    if not hasattr(agent, 'bootstrap_service') or not agent.bootstrap_service:
        return False

    agent_data_path = agent.bootstrap_service.agent_data_path
    if not agent_data_path:
        return False

    soul_path = Path(agent_data_path) / "SOUL.md"
    storage = getattr(agent, "storage", None)
    read_canonical = getattr(storage, "get_current_agent_resource", None)
    publish = getattr(storage, "promote_soul_seed", None)
    require_current = getattr(storage, "require_current_agent_resource_version", None)

    # The canonical SOUL before the file: a revision landing between the two
    # reads is then a newer current version, and the store refuses the rename.
    canonical = await read_canonical() if read_canonical is not None else None
    file_content = soul_path.read_text(encoding="utf-8") if soul_path.exists() else None
    # An empty current version is no canonical SOUL, as at boot, which then
    # serves the file.
    canonical_content = canonical.content if canonical is not None else None
    base = canonical_content or file_content
    renamed = _renamed_soul(base, old_name, new_name) if base is not None else None
    if file_content is None:
        file_renamed = None
    elif file_content == base:
        file_renamed = renamed
    else:
        file_renamed = _renamed_soul(file_content, old_name, new_name)
    publishes = renamed != base and publish is not None

    # Privacy boundary (#2672 review P1): the renamed SOUL body is user-derived
    # content. In a volatile privacy mode, do NOT rewrite it to disk or promote
    # it into the encrypted resource table / durable graph — leave no trace.
    # The on-disk / canonical SOUL simply isn't persisted while volatile.
    from kestrel_sovereign.features.storage_access import (
        hides_persisted_user_content,
    )
    if hides_persisted_user_content(agent):
        if publishes or file_renamed != file_content:
            logger.info(
                "SOUL.md name not persisted on rename — durable SOUL writes are "
                "disabled in the current privacy mode (#2672)"
            )
        return False

    # Checked even with nothing to publish: a revision naming the agent could
    # have landed since the read, and renamed everywhere else, the agent would
    # keep the old name in that SOUL with nothing left for a rename run again
    # to rename (#3813).
    expected_current = ExpectedCurrentVersion(
        canonical.version if canonical is not None else None
    )
    changed = False
    # The canonical SOUL before the file, so a refused rename leaves the file
    # as it was.
    if publishes:
        await publish(
            renamed,
            created_by=getattr(agent, "agent_id", getattr(agent, "did", "")),
            source=f"rename:{soul_path}",
            expected_current=expected_current,
        )
        changed = True
    elif read_canonical is not None and require_current is not None:
        await require_current(expected_current)
    if file_renamed != file_content:
        try:
            # Only over the text renamed above: an edit written to the file
            # since it was read is not replaced with older text.
            current_file = (
                soul_path.read_text(encoding="utf-8") if soul_path.exists() else None
            )
            if current_file == file_content:
                soul_path.write_text(file_renamed, encoding="utf-8")
                changed = True
            else:
                logger.warning(
                    "SOUL.md changed while the rename ran; left as it is, without "
                    "the new name"
                )
        except (OSError, UnicodeDecodeError) as exc:
            if not changed:
                raise
            # The canonical SOUL, the one every boot serves, has the new name;
            # SOUL.md is only its cache, and boot and !bootstrap-status report
            # that it differs.
            logger.warning(
                "SOUL.md could not be updated after the canonical SOUL was "
                "renamed: %s",
                exc,
            )
    if changed and hasattr(agent, 'context_builder'):
        loaded = False
        if hasattr(agent.context_builder, "load_canonical_soul_resource"):
            loaded = await agent.context_builder.load_canonical_soul_resource()
        if not loaded:
            agent.context_builder._load_soul_md()
    return changed


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
        description="Show all loaded bootstrap files and their paths, sizes, and per-file status (one of: loaded, partial, not found, skipped (budget)).",
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
        loaded = sum(
            1 for f in files
            if f.get("status") in {"loaded", "partial"}
        )
        truncated_files = [
            f["name"] for f in files
            if f.get("status") == "partial"
        ]
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
                "truncated_files": truncated_files,
            },
        )

    @tool(
        name="bootstrap_reload",
        description="Force reload all bootstrap files from disk into this session. A SOUL.md edit loaded this way lasts only until restart unless the Sovereign makes it canonical with !revise-soul; the result says when it will not survive.",
        category=ToolCategory.SYSTEM,
        command_prefix="!bootstrap reload"
    )
    async def bootstrap_reload(self) -> ToolResult:
        """
        Force reload all bootstrap files from disk.

        Usage:
            !bootstrap reload

        This re-reads every bootstrap file, picking up any edits
        made since the agent started. Every boot serves the canonical
        SOUL, not SOUL.md, so a reloaded SOUL.md that differs from it is
        reported as live for this session only (#3779).
        """
        loader = self._get_loader()
        if loader is None:
            return ToolResult.failed(
                "Bootstrap loader not available (no context_builder).",
            )

        try:
            loader.reload()
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logger.error(f"bootstrap_reload failed during loader.reload(): {e}", exc_info=True)
            return ToolResult.failed(str(e))

        soul_status = await inspect_soul_file(
            getattr(self.agent, "storage", None), self._get_agent_data_path()
        )
        soul_caveat = soul_status.reload_caveat()

        # Also refresh the context_builder cache if it wraps the loader
        if hasattr(self.agent, 'context_builder'):
            cb = self.agent.context_builder
            if hasattr(cb, 'reload_bootstrap_files'):
                try:
                    cb.reload_bootstrap_files()
                except Exception as e:
                    if is_execution_control_error(e):
                        raise
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
                    loaded = [
                        f["name"] for f in files
                        if f.get("status") in {"loaded", "partial"}
                    ]
                    truncated_files = [
                        f["name"] for f in files
                        if f.get("status") == "partial"
                    ]
                    return ToolResult.partial(
                        confirmation=(
                            f"Reloaded {loader.file_count} file(s) from disk; "
                            f"{loader.total_chars} char(s) total"
                        ),
                        error="; ".join(filter(None, [
                            f"context_builder cache refresh failed: {e}; "
                            "stale prompt assembly until next agent restart",
                            soul_caveat,
                        ])),
                        data={
                            "loaded_count": loader.file_count,
                            "total_chars": loader.total_chars,
                            "files": loaded,
                            "file_details": files,
                            "truncated_files": truncated_files,
                            "cache_refresh_error": str(e),
                            "soul_file": soul_status.to_dict(),
                        },
                    )

        files = loader.list_files()
        loaded = [
            f["name"] for f in files
            if f.get("status") in {"loaded", "partial"}
        ]
        truncated_files = [
            f["name"] for f in files
            if f.get("status") == "partial"
        ]
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
        if dropped or truncated_files or soul_caveat:
            errors = []
            if dropped:
                errors.append(
                    f"{len(dropped)} configured file(s) exist on disk "
                    f"but were dropped from the prompt "
                    f"(read failure or budget exhausted): "
                    f"{', '.join(dropped)}"
                )
            if truncated_files:
                errors.append(
                    f"{len(truncated_files)} configured file(s) were "
                    f"loaded partially due to bootstrap size budgets: "
                    f"{', '.join(truncated_files)}"
                )
            if soul_caveat:
                errors.append(soul_caveat)
            return ToolResult.partial(
                confirmation=(
                    f"Reloaded {len(loaded)} file(s) from disk; "
                    f"{loader.total_chars} char(s) total"
                ),
                error="; ".join(errors),
                data={
                    "loaded_count": loader.file_count,
                    "total_chars": loader.total_chars,
                    "files": loaded,
                    "file_details": files,
                    "dropped_files": dropped,
                    "truncated_files": truncated_files,
                    "soul_file": soul_status.to_dict(),
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
                "file_details": files,
                "truncated_files": [],
                "soul_file": soul_status.to_dict(),
            },
        )

    @tool(
        name="bootstrap_add",
        description="Add a new bootstrap file to be loaded at startup. file_path resolves relative to the agent data dir when not absolute; the file is registered by BASENAME, so a basename collision loads the search-root copy instead.",
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
                if is_execution_control_error(e):
                    raise
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
        except Exception as control_error:
            if is_execution_control_error(control_error):
                raise
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
        description="Remove a bootstrap file from the loading convention. name is the basename as shown by bootstrap_list (the file is not deleted from disk).",
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
                if is_execution_control_error(e):
                    raise
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
                    if is_execution_control_error(e):
                        raise
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
            if is_execution_control_error(e):
                raise
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
            if is_execution_control_error(e):
                raise
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
                loaded = False
                if hasattr(self.agent.context_builder, "load_canonical_soul_resource"):
                    loaded = await self.agent.context_builder.load_canonical_soul_resource()
                if not loaded:
                    self.agent.context_builder._load_soul_md()
            except Exception as e:
                if is_execution_control_error(e):
                    raise
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
            if is_execution_control_error(e):
                raise
            logger.error(f"restart_discovery failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        history_clear_succeeded = bool(
            getattr(result, "history_clear_succeeded", True)
        )
        history_count_after = getattr(result, "history_count_after", None)
        history_clear_error = getattr(result, "history_clear_error", None)

        try:
            strict_loader = getattr(
                type(self.agent.bootstrap_service),
                "_load_discovery_history_strict",
                None,
            )
            if strict_loader is not None:
                history_after = await (
                    self.agent.bootstrap_service._load_discovery_history_strict()
                )
            else:
                history_after = await (
                    self.agent.bootstrap_service.get_discovery_history()
                )
            history_count_after = len(history_after)
            history_clear_succeeded = (
                history_clear_succeeded and history_count_after == 0
            )
            if history_count_after != 0 and not history_clear_error:
                history_clear_error = (
                    f"discovery history still has {history_count_after} "
                    "persisted entr(ies) after reset"
                )
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logger.error(
                f"restart_discovery history verification failed: {e}",
                exc_info=True,
            )
            history_clear_succeeded = False
            history_clear_error = (
                f"failed to verify discovery history was cleared: {e}"
            )

        service_data = {
            "service_result": str(result),
            "history_clear_succeeded": history_clear_succeeded,
            "history_count_after": history_count_after,
            "history_clear_error": history_clear_error,
            "state_reset": getattr(result, "state_reset", None),
            "soul_deleted": getattr(result, "soul_deleted", None),
            "soul_path": getattr(result, "soul_path", None),
        }

        if not history_clear_succeeded:
            return ToolResult.partial(
                confirmation=(
                    f"{str(result) if result else 'Discovery restarted'} "
                    "Bootstrap state reset was attempted, but persisted "
                    "discovery history clear is unverified."
                ),
                error=(
                    history_clear_error
                    or "persisted discovery history was not confirmed empty"
                ),
                data=service_data,
            )

        # Clear SOUL.md from context builder
        if hasattr(self.agent, 'context_builder'):
            try:
                self.agent.context_builder._soul_content = None
            except Exception as e:
                if is_execution_control_error(e):
                    raise
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
                    data=service_data,
                )

        return ToolResult.ok(
            confirmation=str(result) if result else "Discovery restarted",
            data=service_data,
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
        - Whether SOUL.md is the canonical SOUL the next boot serves
        """
        if not hasattr(self.agent, 'bootstrap_service') or not self.agent.bootstrap_service:
            return ToolResult.failed("Bootstrap service not available.")

        service = self.agent.bootstrap_service
        try:
            status = await service.get_bootstrap_status()
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logger.error(f"bootstrap_status failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        # Surface the structured fields the human-readable status string
        # summarizes so the result is chainable (not a stringified blob):
        # state, discovery exchange count, soul-exists bool. These are a
        # best-effort enrichment — a sub-query failure must not sink the
        # OK envelope, which already carries the authoritative summary.
        state_value: Optional[str] = None
        exchange_count: Optional[int] = None
        soul_exists: Optional[bool] = None
        soul_file: Optional[dict] = None
        try:
            state = await service.get_bootstrap_state()
            state_value = state.value if hasattr(state, "value") else str(state)
            history = await service.get_discovery_history()
            exchange_count = sum(1 for m in history if m.get("role") == "user")
            agent_data_path = getattr(service, "agent_data_path", None)
            if agent_data_path:
                soul_exists = (Path(agent_data_path) / "SOUL.md").exists()
                soul_file = (
                    await inspect_soul_file(
                        getattr(service, "storage", None), agent_data_path
                    )
                ).to_dict()
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logger.warning(f"bootstrap_status structured fields unavailable: {e}")

        return ToolResult.ok(
            confirmation=str(status) if status else "Bootstrap status retrieved",
            data={
                "state": state_value,
                "exchange_count": exchange_count,
                "soul_exists": soul_exists,
                "soul_file": soul_file,
                "status": str(status),
            },
        )

    @tool(
        name="rename_agent",
        description="Rename this agent. new_name must be 1-64 characters.",
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
        - Update the name in the canonical SOUL and in SOUL.md, first: if
          that fails, the name is changed nowhere
        - Update the agent's name in metadata
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
            old_name = getattr(self.agent, '_agent_name', 'Unknown')
            outcome = await rename_agent_core(self.agent, new_name)
        except ValueError as e:
            return ToolResult.failed(str(e))
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logger.error(f"Failed to rename agent: {e}")
            return ToolResult.failed(f"Failed to rename: {str(e)}")

        data = {
            "new_name": new_name.strip(),
            "soul_updated": outcome.soul_md_updated,
            "rename_outcome": outcome.to_dict(),
        }
        if outcome.skipped_privacy:
            # Not a failure and not a durable rename — the in-memory name changed
            # but nothing was persisted, by privacy design (#2672 P1).
            return ToolResult.partial(
                confirmation=_rename_confirmation(old_name, new_name.strip(), outcome),
                error=(
                    "Name not persisted in the current privacy mode; it applies to "
                    "this session only and reverts on restart."
                ),
                data=data,
            )
        if not outcome.any_written:
            return ToolResult.failed(
                f"Failed to rename: {outcome.error or 'no rename stores were updated'}",
                data=data,
            )
        if not outcome.success:
            return ToolResult.partial(
                confirmation=_rename_confirmation(old_name, new_name.strip(), outcome),
                error=_rename_partial_error(outcome),
                data=data,
            )

        return ToolResult.ok(
            confirmation=_rename_confirmation(old_name, new_name.strip(), outcome),
            data=data,
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
        except Exception as control_error:
            if is_execution_control_error(control_error):
                raise
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
