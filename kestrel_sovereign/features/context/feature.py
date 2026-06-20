"""
Context Management Feature for Kestrel Agent.

Allows the agent to introspect, optimize, and manage its own context window.
This gives the agent tools to:
- See current context utilization
- Summarize specific conversation sections
- Mark content as protected or droppable
- Proactively trigger compaction
- Exclude irrelevant content (soft removal)
- Restore excluded content
- Stash context for context-switching (like git stash)

Security safeguards:
- No permanent deletion (soft exclusion only)
- Protected content cannot be excluded or stashed
- All operations logged for audit trail
- Rate limiting to prevent manipulation loops

@tool methods return ``kestrel_sdk.tools.result.ToolResult`` per the
kestrel-sovereign #1042 narration-honesty contract (see #1061).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)


def _detect_partial_signal(result: Dict[str, Any]) -> Optional[str]:
    """Inspect a manager dict for partial-success signals.

    Honesty: when the context_manager skipped some requested
    messages (because they were protected, or already excluded, or
    not found) it still returns ``success: True``. The wire-shape
    leaks this only via a count field. Translating those into
    PARTIAL forces the LLM to surface the skipped half rather than
    claim "Marked N messages" when M < N actually changed.

    Returns a human-readable error string when partial, else None.
    """
    protected = result.get("protected_count")
    if isinstance(protected, int) and protected > 0:
        return f"{protected} message(s) were protected and skipped"
    skipped = result.get("skipped_count")
    if isinstance(skipped, int) and skipped > 0:
        reason = result.get("skipped_reason") or "could not be processed"
        return f"{skipped} message(s) skipped: {reason}"
    # The action-count field varies across managers; treat any
    # known-zero with a "no-op" note as a no-op confirmation, not
    # as partial. The no-op detection lives in the caller.
    return None


def _is_noop_result(result: Dict[str, Any], action_count_keys: tuple[str, ...]) -> bool:
    """Detect a no-op manager response.

    The context_manager's stash_pop/apply/drop and restore_excluded
    happily return ``success: True, count: 0, note: "no stashes"``
    when there's nothing to do. The honest framing is "no-op", not
    "popped most recent stash". Caller passes the count keys it
    expects (e.g. ``("restored_count", "popped_count")``).
    """
    for key in action_count_keys:
        if key in result and isinstance(result[key], int) and result[key] > 0:
            return False
    # If none of the action counts were touched, this was a no-op
    # iff at least one of the count keys appeared as zero. (Avoids
    # false-positive on results that simply don't carry counts.)
    if any(key in result for key in action_count_keys):
        return True
    return False


def _wrap_manager_result(
    result: Any,
    *,
    ok_confirmation: str,
    failure_prefix: str = "context_manager",
    noop_count_keys: tuple[str, ...] = (),
    noop_confirmation: Optional[str] = None,
) -> ToolResult:
    """Translate a context_manager method's dict return into a ToolResult.

    The manager's ``mark_messages`` / ``compact_session`` / ``stash_*``
    helpers all return dicts with an inconsistent honesty shape:

      - some include ``success: True`` and the data
      - some include ``error: "..."`` and ``success: False``
      - some return data without any explicit success marker
      - some return ``success: True`` with a partial-skip count
        (``protected_count``) — those must surface as PARTIAL
      - stash_pop/apply/drop return ``success: True, count: 0``
        when there is nothing to do — the caller passes
        ``noop_count_keys`` so we can phrase the confirmation
        as a no-op rather than "popped most recent stash"

    Behavior:

      - dict with ``error`` key → ToolResult.failed(error, data=rest)
      - dict with ``success: False`` (no error) → ToolResult.failed
      - dict with a partial signal → ToolResult.partial
      - dict that satisfies ``_is_noop_result`` → ToolResult.ok with
        ``noop_confirmation`` (or a generic "no-op" message)
      - anything else → ToolResult.ok(ok_confirmation, data=result)
    """
    if isinstance(result, dict):
        err = result.get("error")
        success_flag = result.get("success", True)
        if err:
            data = {k: v for k, v in result.items() if k != "error"}
            return ToolResult.failed(str(err), data=data or None)
        if success_flag is False:
            return ToolResult.failed(
                f"{failure_prefix} returned success=False without an error message",
                data=result or None,
            )
        partial_msg = _detect_partial_signal(result)
        if partial_msg:
            return ToolResult.partial(
                confirmation=ok_confirmation,
                error=partial_msg,
                data=dict(result),
            )
        if noop_count_keys and _is_noop_result(result, noop_count_keys):
            return ToolResult.ok(
                noop_confirmation or (
                    result.get("note")
                    or f"{failure_prefix}: nothing to do (no-op)"
                ),
                data=dict(result),
            )
        return ToolResult.ok(ok_confirmation, data=dict(result))
    # Non-dict — preserve as data so the LLM can still inspect it.
    return ToolResult.ok(ok_confirmation, data={"result": result})


class ContextFeature(Feature):
    """
    Context management tools for the agent.

    Provides tools for:
    - Checking context window utilization
    - Summarizing specific conversation sections
    - Marking content priority (protected/droppable)
    - Triggering compaction
    - Excluding/restoring content
    - Stashing context for context-switching (stash/pop/apply/list/drop)
    """

    @property
    def tool_description(self) -> str:
        return "Manage context window - check status, summarize sections, mark content priority, compact, exclude/restore content"

    async def initialize(self):
        """Initialize the context feature.

        ``context_manager`` and ``llm_service`` are exposed as properties
        that read from ``self.agent`` at call time. The agent's
        ``initialize()`` creates the ContextManager (kestrel_agent.py:1034)
        AFTER it discovers and registers features (line 819), so a
        snapshot taken here would be ``None`` forever — including on
        multi-agent satellites, which is how the bug surfaced in #1382.
        Reading through ``self.agent`` at tool-call time fixes that
        without depending on registration order.
        """
        logger.info("ContextFeature initialized")

    @property
    def context_manager(self):
        """Resolve the agent's ContextManager at call time.

        Returns ``None`` until the agent finishes ``initialize()`` and
        attaches one. Each tool below checks for ``None`` and surfaces a
        precise error rather than relying on a snapshot captured during
        feature registration (which races the agent's own init order).
        """
        return getattr(getattr(self, "agent", None), "context_manager", None)

    @context_manager.setter
    def context_manager(self, value):
        # Test-only setter: a few unit tests construct a feature shell
        # with ``ContextFeature.__new__`` (no __init__) and then assign
        # a mock here. We push the mock onto the agent so the
        # property's read path still returns it.
        if getattr(self, "agent", None) is None:
            self.agent = type("AgentStub", (), {})()
        self.agent.context_manager = value

    @property
    def llm_service(self):
        """Resolve the agent's LLM service at call time. See
        ``context_manager`` above — same race, same fix."""
        return getattr(getattr(self, "agent", None), "llm_service", None)

    @llm_service.setter
    def llm_service(self, value):
        if getattr(self, "agent", None) is None:
            self.agent = type("AgentStub", (), {})()
        self.agent.llm_service = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_target_messages(
        self,
        target: str,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Parse a target string and return (messages, error).

        Target syntax:
          - "last_N"       → last N messages
          - "search:query" → semantic match
          - "ids:1,2,3"    → explicit message ids

        Returns ``(messages, None)`` on success or ``(None, error_str)`` on
        a parse / lookup failure.
        """
        if not isinstance(target, str) or not target:
            return None, f"target must be a non-empty string, got {target!r}"

        try:
            if target.startswith("last_"):
                n = target.split("_", 1)[1]
                # ``get_messages_for_selection`` accepts the criteria as
                # a string for last_n; do not pre-coerce so the manager
                # can surface its own malformed-input error.
                messages = await self.context_manager.get_messages_for_selection(
                    mode="last_n", criteria=n,
                )
            elif target.startswith("search:"):
                query = target[len("search:"):]
                messages = await self.context_manager.get_messages_for_selection(
                    mode="topic", criteria=query,
                )
            elif target.startswith("ids:"):
                ids_str = target[len("ids:"):]
                messages = await self.context_manager.get_messages_for_selection(
                    mode="messages", criteria=ids_str,
                )
            else:
                return None, (
                    f"Invalid target format: {target}. "
                    "Use 'last_N', 'search:query', or 'ids:1,2,3'"
                )
        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as e:
            return None, str(e)
        return messages, None

    @tool(
        name="context_status",
        description="Check current context window utilization. Use this to understand how much context space is available before deciding to summarize or prune.",
        category=ToolCategory.SYSTEM,
        command_prefix="!context status"
    )
    async def context_status(self) -> ToolResult:
        """Get detailed context window status."""
        if not self.context_manager:
            # The property reads ``self.agent.context_manager`` at call
            # time; ``None`` here means the agent really has no manager
            # attached, not that registration ran before init. Tell the
            # operator which side is missing instead of the historical
            # opaque "not available".
            agent_kind = type(self.agent).__name__ if self.agent else "None"
            return ToolResult.failed(
                "context_manager is not attached to this agent "
                f"(agent={agent_kind}); the agent's initialize() did "
                "not construct a ContextManager. This tool is "
                "unavailable until that wiring is in place."
            )

        try:
            context_stats = getattr(self.agent, 'context_stats', None)
            status = await self.context_manager.get_status(context_stats=context_stats)
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"context_status failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"context_status failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if not isinstance(status, dict):
            return ToolResult.failed(
                f"context_manager.get_status returned non-dict: {type(status).__name__}"
            )

        utilization = status.get("utilization_percent")
        msg_count = status.get("message_count")
        return ToolResult.ok(
            confirmation=(
                f"Context status: {msg_count} message(s), "
                f"{utilization}% utilization"
            ),
            data=dict(status),
        )

    @tool(
        name="summarize_section",
        description="Summarize a specific section of conversation history to save context space. Use this to compact verbose exchanges while preserving key information.",
        category=ToolCategory.MEMORY,
        command_prefix="!context summarize"
    )
    async def summarize_section(
        self,
        mode: str,
        criteria: str,
        preserve_key_facts: bool = True,
    ) -> ToolResult:
        """
        Summarize a section of conversation.

        Args:
            mode: Selection mode - "time_range", "topic", "messages", or "last_n"
            criteria: Selection criteria based on mode
            preserve_key_facts: Keep explicit facts, decisions, commitments (default True)
        """
        if not isinstance(preserve_key_facts, bool):
            return ToolResult.failed(
                "preserve_key_facts must be a boolean, got "
                f"{type(preserve_key_facts).__name__}={preserve_key_facts!r}"
            )

        if not self.context_manager:
            return ToolResult.failed("Context manager not available")
        if not self.llm_service:
            return ToolResult.failed("LLM service not available for summarization")

        try:
            messages = await self.context_manager.get_messages_for_selection(
                mode=mode, criteria=criteria,
            )
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.error(f"summarize_section selection failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"summarize_section selection failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if not messages:
            return ToolResult.failed(
                f"No messages found for {mode}={criteria}",
                data={"mode": mode, "criteria": criteria},
            )
        if len(messages) < 2:
            return ToolResult.failed(
                "Need at least 2 messages to summarize",
                data={"found": len(messages), "mode": mode, "criteria": criteria},
            )

        message_ids = [m["id"] for m in messages]
        try:
            result = await self.context_manager.summarize_messages(
                llm_service=self.llm_service,
                message_ids=message_ids,
                preserve_key_facts=preserve_key_facts,
            )
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.error(f"summarize_section failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"summarize_section failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return _wrap_manager_result(
            result,
            ok_confirmation=(
                f"Summarized {len(message_ids)} message(s) from "
                f"{mode}={criteria}"
            ),
            failure_prefix="summarize_messages",
        )

    @tool(
        name="mark_content",
        description="Mark conversation content for context management. Use 'protect' to ensure important content is never pruned, 'droppable' to suggest low-priority content for removal.",
        category=ToolCategory.MEMORY,
        command_prefix="!context mark"
    )
    async def mark_content(
        self,
        action: str,
        target: str,
        reason: str = "",
    ) -> ToolResult:
        """
        Mark content for context management.

        Args:
            action: "protect" (never auto-prune), "droppable" (first to remove), "clear" (remove marking)
            target: Message selection - "last_5", "search:error handling", "ids:23,24,25"
            reason: Optional reason for marking (logged for audit)
        """
        if not self.context_manager:
            return ToolResult.failed("Context manager not available")

        if action not in ("protect", "droppable", "clear"):
            return ToolResult.failed(
                f"action must be 'protect', 'droppable', or 'clear', "
                f"got {action!r}"
            )

        messages, err = await self._resolve_target_messages(target)
        if err:
            return ToolResult.failed(err)
        if not messages:
            return ToolResult.failed(
                f"No messages found for target: {target}",
                data={"target": target},
            )

        message_ids = [m["id"] for m in messages]
        try:
            result = await self.context_manager.mark_messages(
                message_ids=message_ids,
                action=action,
                reason=reason,
            )
        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as e:
            logger.error(f"mark_content failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"mark_content failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return _wrap_manager_result(
            result,
            ok_confirmation=(
                f"Marked {len(message_ids)} message(s) as {action}"
                + (f" ({reason})" if reason else "")
            ),
            failure_prefix="mark_messages",
        )

    @tool(
        name="compact_context",
        description="Compact context by summarizing older messages. Use when context utilization is high and you need space for new information.",
        category=ToolCategory.MEMORY,
        command_prefix="!context compact"
    )
    async def compact_context(
        self,
        keep_recent: int = 10,
        force: bool = False,
        dry_run: bool = False,
    ) -> ToolResult:
        """
        Compact context window by summarizing older messages.

        Args:
            keep_recent: Number of recent messages to preserve verbatim (default 10)
            force: Compact even if utilization is below threshold (default False)
            dry_run: Show what would be compacted without doing it (default False)
        """
        for name, val in (("force", force), ("dry_run", dry_run)):
            if not isinstance(val, bool):
                return ToolResult.failed(
                    f"{name} must be a boolean, got "
                    f"{type(val).__name__}={val!r}"
                )

        try:
            keep_val = int(keep_recent)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"keep_recent must be an integer, got {keep_recent!r}"
            )
        if keep_val < 0:
            return ToolResult.failed("keep_recent must be >= 0")

        if not self.context_manager:
            return ToolResult.failed("Context manager not available")

        if dry_run:
            try:
                status = await self.context_manager.check_compaction_needed()
            except (AttributeError, TypeError, ValueError) as e:
                logger.error(f"compact_context dry-run failed: {e}")
                return ToolResult.failed(str(e))
            except Exception as e:
                logger.error(f"compact_context dry-run failed: {e}", exc_info=True)
                return ToolResult.failed(str(e))

            return ToolResult.ok(
                confirmation=(
                    f"Dry run: would compact "
                    f"{max(0, status['message_count'] - keep_val)} message(s), "
                    f"preserve {min(keep_val, status['message_count'])}"
                ),
                data={
                    "dry_run": True,
                    "compaction_recommended": status.get("compaction_recommended"),
                    "utilization_percent": status.get("utilization_percent"),
                    "message_count": status.get("message_count"),
                    "would_compact": max(0, status["message_count"] - keep_val),
                    "would_preserve": min(keep_val, status["message_count"]),
                },
            )

        if not self.llm_service:
            return ToolResult.failed("LLM service not available for compaction")

        try:
            result = await self.context_manager.compact_session(
                llm_service=self.llm_service,
                preserve_recent=keep_val,
                force=force,
            )

            # Reset context stats after compaction — accumulated
            # duplicate/attribution data is stale post-compaction.
            context_stats = getattr(self.agent, 'context_stats', None)
            if context_stats is not None:
                context_stats.reset()
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"compact_context failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"compact_context failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return _wrap_manager_result(
            result,
            ok_confirmation=f"Compacted context, preserved last {keep_val}",
            failure_prefix="compact_session",
        )

    @tool(
        name="exclude_from_context",
        description="Exclude messages from context window (they remain in storage but won't be included in context). Use for redundant, superseded, or irrelevant content.",
        category=ToolCategory.MEMORY,
        command_prefix="!context exclude"
    )
    async def exclude_from_context(
        self,
        target: str,
        reason: str,
    ) -> ToolResult:
        """
        Exclude content from context assembly.

        Args:
            target: Message selection - "ids:1,2,3", "search:old debug output", "last_5"
            reason: Required reason for exclusion (logged for audit)
        """
        if not self.context_manager:
            return ToolResult.failed("Context manager not available")
        if not reason:
            return ToolResult.failed("Reason is required for exclusion")

        messages, err = await self._resolve_target_messages(target)
        if err:
            return ToolResult.failed(err)
        if not messages:
            return ToolResult.failed(
                f"No messages found for target: {target}",
                data={"target": target},
            )

        message_ids = [m["id"] for m in messages]
        try:
            result = await self.context_manager.exclude_messages(
                message_ids=message_ids, reason=reason,
            )
        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as e:
            logger.error(f"exclude_from_context failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"exclude_from_context failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return _wrap_manager_result(
            result,
            ok_confirmation=(
                f"Excluded {len(message_ids)} message(s) ({reason})"
            ),
            failure_prefix="exclude_messages",
        )

    @tool(
        name="restore_excluded",
        description="Restore previously excluded content back to context.",
        category=ToolCategory.MEMORY,
        command_prefix="!context restore"
    )
    async def restore_excluded(self, target: str = "all") -> ToolResult:
        """
        Restore excluded content.

        Args:
            target: What to restore - "all", "recent" (last exclusion), or "ids:1,2,3"
        """
        if not self.context_manager:
            return ToolResult.failed("Context manager not available")

        try:
            if target == "all":
                result = await self.context_manager.restore_messages(message_ids=None)
            elif target == "recent":
                conv_store = self.context_manager._get_conversation_store()
                if not conv_store:
                    return ToolResult.failed("Conversation store not available")
                excluded = await conv_store.get_excluded_messages(limit=10)
                if not excluded:
                    return ToolResult.ok(
                        confirmation="No excluded messages to restore",
                        data={"restored_count": 0},
                    )
                excluded.sort(
                    key=lambda m: m.get("metadata", {}).get("excluded_at", ""),
                    reverse=True,
                )
                recent_time = excluded[0].get("metadata", {}).get("excluded_at")
                recent_ids = [
                    m["id"] for m in excluded
                    if m.get("metadata", {}).get("excluded_at") == recent_time
                ]
                result = await self.context_manager.restore_messages(message_ids=recent_ids)
            elif target.startswith("ids:"):
                ids_str = target[len("ids:"):]
                try:
                    message_ids = [int(x.strip()) for x in ids_str.split(",")]
                except ValueError:
                    return ToolResult.failed(f"Invalid message IDs: {ids_str}")
                result = await self.context_manager.restore_messages(message_ids=message_ids)
            else:
                return ToolResult.failed(
                    f"Invalid target: {target}. Use 'all', 'recent', or 'ids:1,2,3'"
                )
        except ValueError as e:
            logger.error(f"restore_excluded failed: {e}")
            return ToolResult.failed(str(e))
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"restore_excluded failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"restore_excluded failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return _wrap_manager_result(
            result,
            ok_confirmation=f"Restored excluded messages (target={target})",
            failure_prefix="restore_messages",
            noop_count_keys=("restored_count", "message_count"),
            noop_confirmation=f"No excluded messages to restore (target={target})",
        )

    # =========================================================================
    # Stash Tools (Temporary Context Parking)
    # =========================================================================

    @tool(
        name="context_stash",
        description="Stash current working context (like git stash). Use when you need to context-switch to a different topic and want to restore the current discussion later.",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash"
    )
    async def context_stash(
        self,
        target: str = "last_10",
        name: str = "",
    ) -> ToolResult:
        """
        Stash messages for later restoration.

        Args:
            target: What to stash - "last_N" (e.g., "last_10") or "ids:1,2,3"
            name: Optional name for this stash (e.g., "debugging-session")
        """
        if not self.context_manager:
            return ToolResult.failed("Context manager not available")

        try:
            if target.startswith("last_"):
                try:
                    n = int(target.split("_", 1)[1])
                except ValueError:
                    return ToolResult.failed(f"Invalid last_N format: {target}")
                result = await self.context_manager.stash_messages(
                    last_n=n, name=name if name else None,
                )
            elif target.startswith("ids:"):
                ids_str = target[len("ids:"):]
                try:
                    message_ids = [int(x.strip()) for x in ids_str.split(",")]
                except ValueError:
                    return ToolResult.failed(f"Invalid message IDs: {target}")
                result = await self.context_manager.stash_messages(
                    message_ids=message_ids, name=name if name else None,
                )
            else:
                return ToolResult.failed(
                    f"Invalid target: {target}. Use 'last_N' or 'ids:1,2,3'"
                )
        except ValueError as e:
            logger.error(f"context_stash failed: {e}")
            return ToolResult.failed(str(e))
        except (AttributeError, TypeError, IndexError) as e:
            logger.error(f"context_stash failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"context_stash failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return _wrap_manager_result(
            result,
            ok_confirmation=(
                f"Stashed messages (target={target})"
                + (f" as {name!r}" if name else "")
            ),
            failure_prefix="stash_messages",
        )

    @tool(
        name="context_stash_pop",
        description="Pop the most recent stash (restore messages and remove from stash list). Like git stash pop.",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash pop"
    )
    async def context_stash_pop(self, stash_id: str = "") -> ToolResult:
        """Pop a stash - restore messages and remove from stash list."""
        if not self.context_manager:
            return ToolResult.failed("Context manager not available")

        try:
            result = await self.context_manager.stash_pop(
                stash_id=stash_id if stash_id else None,
            )
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.error(f"context_stash_pop failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"context_stash_pop failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return _wrap_manager_result(
            result,
            ok_confirmation=(
                f"Popped stash {stash_id!r}" if stash_id else "Popped most recent stash"
            ),
            failure_prefix="stash_pop",
            noop_count_keys=("restored_count", "popped_count", "message_count"),
            noop_confirmation=(
                f"No stash to pop (stash_id={stash_id!r})" if stash_id
                else "No stashes to pop"
            ),
        )

    @tool(
        name="context_stash_apply",
        description="Apply a stash without removing it (restore messages but keep stash for reuse). Like git stash apply.",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash apply"
    )
    async def context_stash_apply(self, stash_id: str = "") -> ToolResult:
        """Apply a stash - restore messages but keep stash reference."""
        if not self.context_manager:
            return ToolResult.failed("Context manager not available")

        try:
            result = await self.context_manager.stash_apply(
                stash_id=stash_id if stash_id else None,
            )
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.error(f"context_stash_apply failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"context_stash_apply failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return _wrap_manager_result(
            result,
            ok_confirmation=(
                f"Applied stash {stash_id!r}" if stash_id else "Applied most recent stash"
            ),
            failure_prefix="stash_apply",
            noop_count_keys=("restored_count", "applied_count", "message_count"),
            noop_confirmation=(
                f"No stash to apply (stash_id={stash_id!r})" if stash_id
                else "No stashes to apply"
            ),
        )

    @tool(
        name="context_stash_list",
        description="List all stashes with their names and message counts.",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash list"
    )
    async def context_stash_list(self) -> ToolResult:
        """List all available stashes."""
        if not self.context_manager:
            return ToolResult.failed("Context manager not available")

        try:
            result = await self.context_manager.stash_list()
        except (AttributeError, TypeError) as e:
            logger.error(f"context_stash_list failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"context_stash_list failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        # Honest count — derive from the manager's response so the LLM
        # cannot claim "Listed N stashes" when the manager returned 0.
        if isinstance(result, dict):
            count = len(result.get("stashes", []))
            return _wrap_manager_result(
                result,
                ok_confirmation=f"Listed {count} stash(es)",
                failure_prefix="stash_list",
            )
        return ToolResult.ok(
            confirmation="Listed stashes",
            data={"result": result},
        )

    @tool(
        name="context_stash_drop",
        description="Drop a stash without restoring (discard stashed messages). Messages become excluded from context.",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash drop"
    )
    async def context_stash_drop(self, stash_id: str = "") -> ToolResult:
        """Drop a stash without restoring messages."""
        if not self.context_manager:
            return ToolResult.failed("Context manager not available")

        try:
            result = await self.context_manager.stash_drop(
                stash_id=stash_id if stash_id else None,
            )
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.error(f"context_stash_drop failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"context_stash_drop failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return _wrap_manager_result(
            result,
            ok_confirmation=(
                f"Dropped stash {stash_id!r}" if stash_id else "Dropped most recent stash"
            ),
            failure_prefix="stash_drop",
            noop_count_keys=("dropped_count", "message_count"),
            noop_confirmation=(
                f"No stash to drop (stash_id={stash_id!r})" if stash_id
                else "No stashes to drop"
            ),
        )

    @tool(
        name="context_stash_save",
        description="Save a stash to long-term storage with semantic search capability. Use when you want to preserve context for future retrieval via !recall.",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash save"
    )
    async def context_stash_save(
        self,
        stash_id: str = "",
        name: str = "",
        summary: str = "",
        tags: str = "",
    ) -> ToolResult:
        """Save a stash to SavedItems for long-term retrieval."""
        if not self.context_manager:
            return ToolResult.failed("Context manager not available")

        try:
            tags_list = [t.strip() for t in tags.split(",")] if tags else None
            result = await self.context_manager.stash_save(
                stash_id=stash_id if stash_id else None,
                name=name if name else None,
                summary=summary if summary else None,
                tags=tags_list,
            )
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.error(f"context_stash_save failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"context_stash_save failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return _wrap_manager_result(
            result,
            ok_confirmation=(
                f"Saved stash {stash_id!r}" if stash_id
                else "Saved most recent stash"
            ),
            failure_prefix="stash_save",
        )

    @tool(
        name="context_stash_peek",
        description="Peek at stash contents without restoring. Use to explore stashed context programmatically (RLM-inspired context-as-variable).",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash peek"
    )
    async def context_stash_peek(
        self,
        stash_id: str = "",
        max_chars: int = 5000,
    ) -> ToolResult:
        """Peek at stash contents without fully restoring."""
        if not self.context_manager:
            return ToolResult.failed("Context manager not available")

        try:
            max_chars_val = int(max_chars)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"max_chars must be an integer, got {max_chars!r}"
            )
        if max_chars_val < 1:
            return ToolResult.failed("max_chars must be >= 1")

        try:
            result = await self.context_manager.stash_peek(
                stash_id=stash_id if stash_id else None,
                max_chars=max_chars_val,
            )
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.error(f"context_stash_peek failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"context_stash_peek failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return _wrap_manager_result(
            result,
            ok_confirmation=(
                f"Peeked at stash {stash_id!r}"
                if stash_id else "Peeked at most recent stash"
            )
            + f" (max_chars={max_chars_val})",
            failure_prefix="stash_peek",
        )

    # =========================================================================
    # RLM-Inspired Advanced Compaction
    # =========================================================================

    @tool(
        name="hierarchical_compact",
        description="Compact context using hierarchical tree-structured summarization (RLM-inspired). Better preserves structure than linear compaction.",
        category=ToolCategory.MEMORY,
        command_prefix="!context compact hierarchical"
    )
    async def hierarchical_compact(
        self,
        chunk_size: int = 4000,
        keep_recent: int = 5,
        max_depth: int = 3,
    ) -> ToolResult:
        """
        Hierarchical compaction using recursive summarization.

        Args:
            chunk_size: Target characters per chunk (default: 4000)
            keep_recent: Messages to preserve verbatim (default: 5)
            max_depth: Maximum recursion depth (default: 3)
        """
        if not self.context_manager:
            return ToolResult.failed("Context manager not available")
        if not self.llm_service:
            return ToolResult.failed("LLM service not available for compaction")

        try:
            chunk_val = int(chunk_size)
            keep_val = int(keep_recent)
            depth_val = int(max_depth)
        except (TypeError, ValueError):
            return ToolResult.failed(
                "chunk_size, keep_recent, max_depth must all be integers"
            )
        if chunk_val < 1 or keep_val < 0 or depth_val < 1:
            return ToolResult.failed(
                "chunk_size and max_depth must be >= 1, keep_recent >= 0"
            )

        try:
            result = await self.context_manager.hierarchical_compact(
                llm_service=self.llm_service,
                chunk_size=chunk_val,
                preserve_recent=keep_val,
                max_depth=depth_val,
            )
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"hierarchical_compact failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"hierarchical_compact failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return _wrap_manager_result(
            result,
            ok_confirmation=(
                f"Hierarchical compact complete (chunk_size={chunk_val}, "
                f"keep_recent={keep_val}, max_depth={depth_val})"
            ),
            failure_prefix="hierarchical_compact",
        )

    @tool(
        name="recursive_query",
        description="Query a subset of context using a cheaper model (RLM-inspired sub-LM call). Use for exploring large context sections, compacted originals, or excluded messages without using main model quota.",
        category=ToolCategory.MEMORY,
        command_prefix="!context query"
    )
    async def recursive_query(
        self,
        context_source: str,
        query: str,
        use_cheap_model: bool = True,
    ) -> ToolResult:
        """
        Query context subset using recursive sub-LM call.

        Args:
            context_source: Source - "stash:name", "excluded", "compacted:ID", "summary:ID", "last_N"
            query: Question to ask about the context
            use_cheap_model: Use cheaper model for query (default: True)
        """
        if not isinstance(use_cheap_model, bool):
            return ToolResult.failed(
                "use_cheap_model must be a boolean, got "
                f"{type(use_cheap_model).__name__}={use_cheap_model!r}"
            )
        if not self.context_manager:
            return ToolResult.failed("Context manager not available")
        if not self.llm_service:
            return ToolResult.failed("LLM service not available")

        try:
            context_text = ""

            if context_source.startswith("stash:"):
                stash_name = context_source[len("stash:"):]
                peek_result = await self.context_manager.stash_peek(
                    stash_id=stash_name, max_chars=10000,
                )
                if peek_result.get("success"):
                    context_text = peek_result.get("preview", "")
                else:
                    return ToolResult.failed(
                        peek_result.get("error", "Stash not found")
                    )

            elif context_source == "excluded":
                conv_store = self.context_manager._get_conversation_store()
                if not conv_store:
                    return ToolResult.failed("Conversation store not available")
                excluded = await conv_store.get_excluded_messages(limit=50)
                context_text = "\n".join([
                    f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                    for m in excluded
                ])[:10000]

            elif context_source.startswith("compacted:") or context_source.startswith("summary:"):
                try:
                    marker_id = context_source.split(":", 1)[1]
                    conv_store = self.context_manager._get_conversation_store()
                    if not conv_store:
                        return ToolResult.failed("Conversation store not available")

                    try:
                        marker_id_parsed: Any = int(marker_id)
                    except ValueError:
                        marker_id_parsed = marker_id

                    marker_messages = await conv_store.get_messages_by_ids([marker_id_parsed])
                    if not marker_messages:
                        return ToolResult.failed(
                            f"Marker message {marker_id} not found"
                        )

                    marker = marker_messages[0]
                    meta = marker.get("metadata", {})
                    original_ids = meta.get("original_message_ids", [])
                    if not original_ids:
                        return ToolResult.failed(
                            f"No original message IDs found in marker {marker_id}"
                        )

                    original_messages = await conv_store.get_messages_by_ids(original_ids)
                    if not original_messages:
                        return ToolResult.failed("Original messages not found")

                    context_text = "\n".join([
                        f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                        for m in original_messages
                    ])[:10000]
                except (IndexError, KeyError) as e:
                    return ToolResult.failed(
                        f"Invalid format: {context_source} - {e}"
                    )

            elif context_source.startswith("last_"):
                try:
                    n = int(context_source.split("_", 1)[1])
                except ValueError:
                    return ToolResult.failed(f"Invalid format: {context_source}")
                messages = await self.context_manager.get_messages_for_selection(
                    mode="last_n", criteria=str(n),
                )
                context_text = "\n".join([
                    f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                    for m in messages
                ])[:10000]
            else:
                return ToolResult.failed(
                    f"Invalid source: {context_source}. "
                    "Use 'stash:name', 'excluded', 'compacted:ID', 'summary:ID', or 'last_N'"
                )

            if not context_text:
                return ToolResult.failed("No context found to query")

            prompt = f"""Answer the following question based ONLY on this context:

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

            # Vendor-scoped cheap-model selector. We pull
            # ``<vendor>/<model>`` so routing constrains to the one
            # vendor that serves the cheap model — preventing the old
            # "broadcast one model id across every provider" cascade.
            model_override = None
            if use_cheap_model and hasattr(self.llm_service, "get_cheap_model_selector"):
                model_override = self.llm_service.get_cheap_model_selector()
            elif use_cheap_model and hasattr(self.llm_service, "get_cheap_model"):
                # Older services only expose the bare id — caller's
                # behavior without vendor scoping is best-effort.
                model_override = self.llm_service.get_cheap_model()

            # LLMService.generate is keyword-only and requires user_prompt
            # (not prompt); the old call TypeError'd into the failure path
            # below (#1844 root-cause sweep).
            response = await self.llm_service.generate(
                system_prompt="You are answering questions about conversation context. Be concise and accurate.",
                user_prompt=prompt,
                model_override=model_override,
            )
        except ValueError as e:
            logger.error(f"recursive_query failed: {e}")
            return ToolResult.failed(str(e))
        except (IndexError, KeyError) as e:
            logger.error(f"recursive_query failed: {e}")
            return ToolResult.failed(str(e))
        except (AttributeError, TypeError) as e:
            logger.error(f"recursive_query failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"recursive_query failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        answer = response.strip() if isinstance(response, str) else str(response)
        return ToolResult.ok(
            confirmation=(
                f"Answered query against {context_source} "
                f"({len(context_text)} chars of context, "
                f"model={model_override or 'default'})"
            ),
            data={
                "answer": answer,
                "context_source": context_source,
                "query": query,
                "model_used": model_override or "default",
                "context_chars": len(context_text),
            },
        )
