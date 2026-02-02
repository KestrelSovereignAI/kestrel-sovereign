"""
Sleep functionality for Kestrel Agent.

Implements human-like "sleep" cycle that:
1. Consolidates memories (creates episodes, archives decayed)
2. Exports state to sovereignty storage (IPFS/Lighthouse)
3. Returns a CID that can be used for restoration

This is inspired by how human memory consolidation occurs during sleep:
- Short-term memories are organized into long-term storage
- Unimportant details fade (forgetting curve)
- Patterns are detected and strengthened
- A "checkpoint" is created for disaster recovery
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SleepReport:
    """Result of a sleep cycle."""
    success: bool
    cid: Optional[str] = None

    # Consolidation stats
    episodes_created: int = 0
    patterns_found: int = 0
    messages_archived: int = 0
    total_messages: int = 0

    # Export stats
    shards_exported: int = 0
    total_size_bytes: int = 0
    storage_tier: str = "local"

    # Reflection stats (NEW)
    pre_reflection: Optional[Dict[str, Any]] = None
    post_reflection: Optional[Dict[str, Any]] = None
    insights_generated: int = 0

    # Timing
    consolidation_ms: int = 0
    export_ms: int = 0
    reflection_ms: int = 0

    # Error info
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "success": self.success,
            "cid": self.cid,
            "consolidation": {
                "episodes_created": self.episodes_created,
                "patterns_found": self.patterns_found,
                "messages_archived": self.messages_archived,
                "total_messages": self.total_messages,
                "duration_ms": self.consolidation_ms,
            },
            "reflection": {
                "pre_reflection": self.pre_reflection,
                "post_reflection": self.post_reflection,
                "insights_generated": self.insights_generated,
                "duration_ms": self.reflection_ms,
            },
            "export": {
                "shards_exported": self.shards_exported,
                "total_size_bytes": self.total_size_bytes,
                "storage_tier": self.storage_tier,
                "duration_ms": self.export_ms,
            },
            "error": self.error,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        if not self.success:
            return f"Sleep failed: {self.error}"

        lines = [
            "Sleep cycle complete:",
            f"  Episodes created: {self.episodes_created}",
            f"  Patterns found: {self.patterns_found}",
            f"  Memories archived: {self.messages_archived}",
            f"  Insights generated: {self.insights_generated}",
            f"  Shards exported: {self.shards_exported}",
            f"  Storage tier: {self.storage_tier}",
        ]
        if self.cid:
            lines.append(f"  CID: {self.cid}")
        return "\n".join(lines)


class SleepMixin:
    """
    Mixin class providing sleep/consolidation methods.

    Sleep combines:
    1. Memory consolidation (via MemoryConsolidator)
    2. Sovereignty export (via SovereignStorageAdapter)

    The result is a CID that represents the agent's complete state,
    which can be used for restoration or migration.

    Callback:
        Set `on_sleep_complete` to a coroutine that receives (cid: str, report: SleepReport).
        This allows platforms to update their database with the new CID.
    """

    # Optional callback for platforms to hook into sleep completion
    # Set this to an async function: async def callback(cid: str, report: SleepReport)
    on_sleep_complete: Optional[Callable] = None

    # Optional reflection hook for self-improvement during sleep
    # Set this to a ReflectionSleepHook instance
    reflection_hook: Optional[Any] = None

    async def sleep(
        self,
        tier: str = "ipfs",
        skip_consolidation: bool = False,
        skip_export: bool = False,
        skip_reflection: bool = False,
    ) -> SleepReport:
        """
        Execute a full sleep cycle.

        The sleep cycle now includes reflection for self-improvement:
        1. Pre-sleep reflection (analyze current session)
        2. Memory consolidation (create episodes, archive decayed)
        3. Post-consolidation reflection (deeper analysis with episodes)
        4. Sovereignty export (backup to IPFS/Filecoin)

        Args:
            tier: Storage tier for export ("local", "ipfs", "filecoin")
            skip_consolidation: Skip memory consolidation (just export)
            skip_export: Skip sovereignty export (just consolidate)
            skip_reflection: Skip reflection hooks

        Returns:
            SleepReport with details of operations performed
        """
        import time
        from kestrel_sovereign.filecoin_adapter import StorageTier

        report = SleepReport(success=False)
        reflection_start = time.time()

        # Note: Privacy mode checks are handled by the storage layer.
        # - EPHEMERAL/ISOLATED: Storage will raise PrivacyViolationError on export
        # - Consolidation is always allowed (reorganizes existing data)
        # We don't block sleep here - let it try and handle errors gracefully.

        tier_map = {
            "local": StorageTier.LOCAL_ONLY,
            "ipfs": StorageTier.IPFS,
            "filecoin": StorageTier.FILECOIN,
            "lighthouse": StorageTier.IPFS,  # Lighthouse uses IPFS protocol
        }
        storage_tier = tier_map.get(tier.lower(), StorageTier.LOCAL_ONLY)
        report.storage_tier = tier

        # 0. Pre-sleep reflection (analyze current session before consolidation)
        if not skip_reflection and self.reflection_hook:
            try:
                pre_result = await self.reflection_hook.on_pre_sleep(self)
                report.pre_reflection = pre_result
                if pre_result.get("success"):
                    report.insights_generated += pre_result.get("insights_generated", 0)
                    logger.info(
                        f"Pre-sleep reflection: {pre_result.get('insights_generated', 0)} insights"
                    )
            except Exception as e:
                logger.warning(f"Pre-sleep reflection failed: {e}")
                # Continue - reflection failure shouldn't block sleep

        # 1. Memory Consolidation
        if not skip_consolidation:
            start = time.time()
            try:
                consolidation_result = await self._consolidate_memories()
                report.episodes_created = consolidation_result.get("episodes_created", 0)
                report.patterns_found = consolidation_result.get("patterns_found", 0)
                report.messages_archived = consolidation_result.get("messages_archived", 0)
                report.total_messages = consolidation_result.get("total_messages_processed", 0)
                logger.info(
                    f"Consolidation complete: {report.episodes_created} episodes, "
                    f"{report.messages_archived} archived"
                )
            except Exception as e:
                logger.error(f"Consolidation failed: {e}")
                report.error = f"Consolidation failed: {e}"
                # Continue to export anyway - partial sleep is better than none
            report.consolidation_ms = int((time.time() - start) * 1000)

            # 1.5 Post-consolidation reflection (uses new episodes for deeper analysis)
            if not skip_reflection and self.reflection_hook:
                try:
                    post_result = await self.reflection_hook.on_post_consolidation(
                        self, consolidation_result
                    )
                    report.post_reflection = post_result
                    if post_result.get("success") and not post_result.get("skipped"):
                        report.insights_generated += post_result.get("insights_generated", 0)
                        logger.info(
                            f"Post-consolidation reflection: "
                            f"{post_result.get('insights_generated', 0)} insights"
                        )
                except Exception as e:
                    logger.warning(f"Post-consolidation reflection failed: {e}")
                    # Continue - reflection failure shouldn't block sleep

        # Record total reflection time
        report.reflection_ms = int((time.time() - reflection_start) * 1000) - report.consolidation_ms

        # 2. Sovereignty Export
        if not skip_export:
            start = time.time()
            try:
                export_result = await self._export_sovereignty(storage_tier)
                report.cid = export_result.get("cid")
                report.shards_exported = export_result.get("shards_exported", 0)
                report.total_size_bytes = export_result.get("total_size_bytes", 0)
                logger.info(f"Export complete: CID={report.cid}")
            except Exception as e:
                logger.error(f"Export failed: {e}")
                if report.error:
                    report.error += f"; Export failed: {e}"
                else:
                    report.error = f"Export failed: {e}"
            report.export_ms = int((time.time() - start) * 1000)

        # Success if at least one operation completed
        report.success = (
            (not skip_consolidation and report.episodes_created >= 0) or
            (not skip_export and report.cid is not None)
        )

        # Invoke callback if set (allows platform to update latest_cid)
        if report.success and report.cid and self.on_sleep_complete:
            try:
                await self.on_sleep_complete(report.cid, report)
            except Exception as e:
                logger.warning(f"on_sleep_complete callback failed: {e}")
                # Don't fail the sleep for callback errors

        return report

    async def _consolidate_memories(self) -> Dict[str, Any]:
        """
        Run memory consolidation.

        Uses MemoryConsolidator to:
        - Create narrative episodes from message clusters
        - Detect temporal patterns
        - Archive fully decayed memories
        """
        if not hasattr(self, 'memory_consolidator') or not self.memory_consolidator:
            logger.warning("MemoryConsolidator not available, skipping consolidation")
            return {"error": "MemoryConsolidator not initialized"}

        return await self.memory_consolidator.run_consolidation()

    async def _export_sovereignty(self, storage_tier) -> Dict[str, Any]:
        """
        Export agent state to sovereignty storage.

        Uses SovereignStorageAdapter for sharded, encrypted export.
        """
        from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter
        from kestrel_sovereign.filecoin_adapter import FilecoinAdapter
        import os

        # Get user secret for convergent encryption
        user_secret = os.getenv("KESTREL_USER_SECRET", "default-secret")
        if hasattr(self, 'agent_id'):
            # Include agent_id for isolation
            user_secret = f"{user_secret}:{self.agent_id}"

        # Get the raw database for direct access
        db = None
        if hasattr(self, '_raw_storage') and self._raw_storage:
            db = self._raw_storage.db
        elif hasattr(self, 'storage') and self.storage:
            if hasattr(self.storage, 'database'):
                db = self.storage.database
            elif hasattr(self.storage, 'db'):
                db = self.storage.db

        if not db:
            raise RuntimeError("No database available for sovereignty export")

        # Create adapter
        filecoin_adapter = FilecoinAdapter()
        adapter = SovereignStorageAdapter(
            db=db,
            user_secret=user_secret,
            filecoin_adapter=filecoin_adapter,
            agent_id=getattr(self, 'agent_id', '') or getattr(self, 'did', '')
        )

        # Export
        agent_did = getattr(self, 'did', None) or getattr(self, 'agent_id', 'unknown')
        cid = await adapter.export_agent(agent_did, storage_tier=storage_tier)

        # Get export stats (from manifest if available)
        return {
            "cid": cid,
            "shards_exported": len(adapter._last_manifest.shards) if hasattr(adapter, '_last_manifest') else 1,
            "total_size_bytes": sum(s.size_bytes for s in adapter._last_manifest.shards) if hasattr(adapter, '_last_manifest') else 0,
        }

    async def _command_sleep(self, user_input: str) -> str:
        """
        Handle !sleep command.

        Usage:
            !sleep                    - Full sleep (consolidate + export to IPFS)
            !sleep --tier local       - Export to local only
            !sleep --tier filecoin    - Export to Filecoin for permanent storage
            !sleep --consolidate-only - Only run memory consolidation
            !sleep --export-only      - Only run sovereignty export
        """
        parts = user_input.split()

        tier = "ipfs"
        skip_consolidation = False
        skip_export = False

        for i, part in enumerate(parts):
            if part == "--tier" and i + 1 < len(parts):
                tier = parts[i + 1]
            elif part.startswith("--tier="):
                tier = part.split("=", 1)[1]
            elif part == "--consolidate-only":
                skip_export = True
            elif part == "--export-only":
                skip_consolidation = True

        report = await self.sleep(
            tier=tier,
            skip_consolidation=skip_consolidation,
            skip_export=skip_export,
        )

        return str(report)

    async def quick_nap(self) -> Optional[str]:
        """
        Quick consolidation without full export.

        Use this for:
        - Session end (30-min inactivity)
        - Message threshold reached
        - Periodic maintenance

        Returns:
            None if nothing to consolidate, or summary string
        """
        if not hasattr(self, 'memory_consolidator') or not self.memory_consolidator:
            return None

        # Check if consolidation is needed
        should_consolidate = await self.memory_consolidator.should_create_episode()
        if not should_consolidate:
            return None

        # Create session episode
        episode = await self.memory_consolidator.create_session_episode()
        if episode:
            return f"Created episode: {episode.title}"
        return None
