"""
Audit Trail Anchoring Feature - Cryptographic proof of verifiable history.

Implements Article II Right 3 (Verifiable History) by:
1. Hashing accumulated security audit log entries with SHA-256
2. Storing serialized entries via content-addressable file storage
3. Recording anchors with hash, storage reference, and entry metadata
4. Verifying integrity by re-computing hashes against stored anchors

The security_audit_log table is managed by the SecurityFeature's
PermissionStore (separate SQLite database). This feature reads from
that database and writes anchors to the main agent storage.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.audit_anchor.hasher import AuditHasher
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database

logger = logging.getLogger(__name__)

# Auto-anchor threshold: anchor automatically after this many unanchored entries
AUTO_ANCHOR_THRESHOLD = 50


class AuditAnchorFeature(Feature):
    """
    Audit trail anchoring - cryptographic proof of verifiable history.

    Provides tools to anchor, verify, and check the status of audit trail
    hashes, ensuring the security audit log is tamper-evident.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Audit trail anchoring - cryptographic proof of verifiable history "
            "(Article II Right 3). Anchor audit logs to persistent storage, "
            "verify integrity, and check anchoring status."
        )

    async def initialize(self):
        """Initialize the audit anchor feature by creating the anchors table."""
        logger.info("Initializing AuditAnchorFeature")
        try:
            db = self._get_db()
            if db is None:
                logger.warning(
                    "AuditAnchorFeature: storage.db not available, "
                    "table creation deferred"
                )
                return

            await db.execute("""
                CREATE TABLE IF NOT EXISTS audit_anchors (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT,
                    anchor_hash TEXT NOT NULL,
                    storage_ref TEXT,
                    entries_count INTEGER NOT NULL,
                    first_entry_at TEXT,
                    last_entry_at TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            logger.info("AuditAnchorFeature initialized - audit_anchors table ready")
        except Exception as e:
            logger.warning(f"AuditAnchorFeature initialization warning: {e}")

    # =========================================================================
    # Tools
    # =========================================================================

    @tool(
        "audit_anchor",
        "Anchor current audit trail to persistent storage",
        category=ToolCategory.SYSTEM,
        command_prefix="!audit-anchor",
    )
    async def anchor_audit(self) -> ToolResult:
        """
        Anchor unanchored audit log entries to persistent storage.

        Computes a SHA-256 hash of all entries since the last anchor,
        stores the serialized entries via content-addressable storage,
        and records the anchor in the audit_anchors table.
        """
        last_anchor_at = await self._get_last_anchor_timestamp()

        entries = await self._get_audit_entries_since(last_anchor_at)
        if not entries:
            return ToolResult.ok(
                confirmation="No new audit entries since last anchor",
                data={"status": "nothing_to_anchor", "message": "No new audit entries since last anchor"},
            )

        # Compute deterministic hash
        anchor_hash = AuditHasher.hash_entries(entries)
        serialized = AuditHasher.serialize_entries(entries)

        # Store serialized entries via file storage
        storage_ref = None
        storage_failed_reason: Optional[str] = None
        try:
            storage_ref = await self.agent.storage.store_file(
                serialized, f"audit_anchor_{anchor_hash[:16]}.json"
            )
        except Exception as e:
            logger.warning(f"Could not store audit entries to file storage: {e}")
            storage_failed_reason = str(e)
            # Fall back to hash-only anchor (no file storage ref)

        # Determine entry time range
        timestamps = [
            e.get("created_at", e.get("timestamp", ""))
            for e in entries
            if e.get("created_at") or e.get("timestamp")
        ]
        first_entry_at = min(timestamps) if timestamps else None
        last_entry_at = max(timestamps) if timestamps else None

        # Record anchor in database
        anchor_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        agent_id = self.agent.did

        db = self._get_db()
        if db is not None:
            try:
                await db.execute(
                    """INSERT INTO audit_anchors
                       (id, agent_id, anchor_hash, storage_ref, entries_count,
                        first_entry_at, last_entry_at, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (anchor_id, agent_id, anchor_hash, storage_ref,
                     len(entries), first_entry_at, last_entry_at, now),
                )
            except Exception as e:
                logger.error(f"Failed to record anchor: {e}")
                return ToolResult.failed(
                    f"Failed to record anchor: {e}",
                    data={"status": "error", "message": f"Failed to record anchor: {e}"},
                )

        # Store as graph node if graph storage is available
        try:
            from kestrel_sovereign.storage import GraphNode
            node = GraphNode(
                node_id=f"audit_anchor_{anchor_id}",
                node_type="audit_anchor",
                label=f"Audit Anchor ({len(entries)} entries)",
                properties={
                    "anchor_hash": anchor_hash,
                    "storage_ref": storage_ref,
                    "entries_count": len(entries),
                    "first_entry_at": first_entry_at,
                    "last_entry_at": last_entry_at,
                    "created_at": now,
                },
            )
            await self.agent.storage.add_node(node)
        except Exception as e:
            logger.debug(f"Could not store anchor as graph node: {e}")

        result = {
            "status": "anchored",
            "anchor_id": anchor_id,
            "anchor_hash": anchor_hash,
            "entries_count": len(entries),
            "storage_ref": storage_ref,
            "first_entry_at": first_entry_at,
            "last_entry_at": last_entry_at,
            "created_at": now,
        }
        logger.info(
            f"Audit trail anchored: {len(entries)} entries, hash={anchor_hash[:16]}..."
        )

        # Honesty: when file-storage attached but failed, the anchor
        # still exists (hash recorded in audit_anchors) but the
        # serialized entries are NOT recoverable from storage_ref.
        # Surface as PARTIAL so the agent must speak that the
        # tamper-proof archival did not fully complete.
        if storage_failed_reason is not None:
            result["storage_failed_reason"] = storage_failed_reason
            return ToolResult.partial(
                confirmation=(
                    f"Anchored {len(entries)} entries (hash={anchor_hash[:16]}…)"
                ),
                error=(
                    f"file storage failed ({storage_failed_reason!r}); "
                    "the anchor row exists with hash + metadata, but the "
                    "serialized entries are not retrievable from "
                    "storage_ref. Verify-only path still works; "
                    "fully-restore path is broken for this anchor."
                ),
                data=result,
            )

        return ToolResult.ok(
            confirmation=(
                f"Anchored {len(entries)} entries (hash={anchor_hash[:16]}…, "
                f"storage_ref={storage_ref})"
            ),
            data=result,
        )

    @tool(
        "audit_verify",
        "Verify audit trail integrity against anchors",
        category=ToolCategory.SYSTEM,
        command_prefix="!audit-verify",
    )
    async def verify_audit(self) -> ToolResult:
        """
        Verify audit trail integrity by re-computing hashes for each anchor.

        For each anchor in audit_anchors, queries the original entries in
        that time range, re-computes the SHA-256 hash, and compares it
        to the stored anchor_hash. Reports pass/fail per anchor.
        """
        anchors = await self._get_all_anchors()
        if not anchors:
            return ToolResult.ok(
                confirmation="No anchors found. Run !audit-anchor first.",
                data={
                    "status": "no_anchors",
                    "message": "No anchors found. Run !audit-anchor first.",
                },
            )

        results = []
        all_passed = True

        for anchor in anchors:
            anchor_id = anchor["id"]
            stored_hash = anchor["anchor_hash"]
            first_at = anchor.get("first_entry_at")
            last_at = anchor.get("last_entry_at")

            # Get entries in this anchor's time range
            entries = await self._get_audit_entries_range(first_at, last_at)

            if not entries:
                results.append({
                    "anchor_id": anchor_id,
                    "status": "warning",
                    "message": "No entries found for time range - entries may have been deleted",
                    "stored_hash": stored_hash,
                })
                all_passed = False
                continue

            # Re-compute hash
            computed_hash = AuditHasher.hash_entries(entries)
            passed = computed_hash == stored_hash

            if not passed:
                all_passed = False

            results.append({
                "anchor_id": anchor_id,
                "status": "pass" if passed else "FAIL",
                "stored_hash": stored_hash,
                "computed_hash": computed_hash,
                "entries_count": len(entries),
                "match": passed,
            })

        passed_count = sum(
            1 for r in results
            if r.get("match", False) or r.get("status") == "pass"
        )
        failed_count = sum(1 for r in results if r.get("status") == "FAIL")
        warnings_count = sum(1 for r in results if r.get("status") == "warning")

        data = {
            "status": "verified" if all_passed else "integrity_failure",
            "total_anchors": len(anchors),
            "passed": passed_count,
            "failed": failed_count,
            "warnings": warnings_count,
            "details": results,
        }

        # Honesty: integrity failure is the headline finding; the LLM
        # must not narrate "audit verified" off a result where any
        # anchor's hash diverged. ERROR routes the failure summary
        # into the agent's ear.
        if not all_passed:
            return ToolResult.failed(
                f"Audit integrity check failed: {failed_count} hash mismatch(es), "
                f"{warnings_count} warning(s) across {len(anchors)} anchor(s)",
                data=data,
            )
        return ToolResult.ok(
            confirmation=f"Audit verified: {passed_count}/{len(anchors)} anchors passed",
            data=data,
        )

    @tool(
        "audit_anchor_status",
        "Check audit anchoring status",
        category=ToolCategory.SYSTEM,
        command_prefix="!audit-status",
    )
    async def anchor_status(self) -> ToolResult:
        """
        Check the current state of audit trail anchoring.
        """
        last_anchor_at = await self._get_last_anchor_timestamp()
        total_anchors = await self._count_anchors()
        entries_since = await self._get_audit_entries_since(last_anchor_at)
        entries_since_count = len(entries_since) if entries_since else 0

        data = {
            "last_anchor_at": last_anchor_at,
            "total_anchors": total_anchors,
            "entries_since_last": entries_since_count,
            "auto_anchor_threshold": AUTO_ANCHOR_THRESHOLD,
        }

        # Honesty: when unanchored entries exceed the auto-anchor
        # threshold, surface as PARTIAL — the read of state succeeded,
        # but the audit trail is currently in a state that warrants
        # action. The agent must speak that the threshold is exceeded
        # rather than narrate "audit-anchor status nominal".
        if entries_since_count >= AUTO_ANCHOR_THRESHOLD:
            return ToolResult.partial(
                confirmation=(
                    f"audit anchor status: {total_anchors} anchor(s); "
                    f"{entries_since_count} entries since last anchor"
                ),
                error=(
                    f"{entries_since_count} unanchored entries exceeds "
                    f"auto-anchor threshold ({AUTO_ANCHOR_THRESHOLD}); "
                    "run !audit-anchor to anchor the backlog"
                ),
                data=data,
            )
        return ToolResult.ok(
            confirmation=(
                f"audit anchor status: {total_anchors} anchor(s), "
                f"{entries_since_count} unanchored entries"
            ),
            data=data,
        )

    # =========================================================================
    # Lifecycle hooks
    # =========================================================================

    async def on_audit_complete(self, audit_result: dict):
        """
        Called after a constitution audit completes. Auto-anchors if
        the number of unanchored entries exceeds AUTO_ANCHOR_THRESHOLD.

        Args:
            audit_result: Dict with is_valid and message from the audit.
        """
        try:
            last_anchor_at = await self._get_last_anchor_timestamp()
            entries = await self._get_audit_entries_since(last_anchor_at)
            count = len(entries) if entries else 0

            if count >= AUTO_ANCHOR_THRESHOLD:
                logger.info(
                    f"Auto-anchoring: {count} unanchored entries "
                    f"(threshold={AUTO_ANCHOR_THRESHOLD})"
                )
                result = await self.anchor_audit()
                # anchor_audit returns ToolResult; the legacy
                # {"status": ...} dict lives under .data.
                status = (result.data or {}).get("status") if result.data else None
                logger.info(f"Auto-anchor result: {status}")
        except Exception as e:
            logger.warning(f"Auto-anchor failed: {e}")

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_db(self):
        """Get the main AsyncDatabase instance, or None if unavailable."""
        return resolve_feature_database(self.agent)

    def _get_permission_store(self):
        """
        Get the SecurityFeature's PermissionStore for querying the audit log.

        The security_audit_log table lives in the PermissionStore's database,
        which is separate from the main agent storage.

        Returns:
            PermissionStore instance or None if SecurityFeature not available.
        """
        try:
            features = getattr(self.agent, "features", {})
            for feature in features.values():
                if type(feature).__name__ == "SecurityFeature":
                    return getattr(feature, "permission_store", None)
        except Exception:
            pass
        return None

    async def _get_audit_entries_since(
        self, since: Optional[str]
    ) -> list:
        """
        Get audit log entries since a given timestamp.

        Queries the security_audit_log table in the PermissionStore's database.

        Args:
            since: ISO timestamp string, or None for all entries.

        Returns:
            List of entry dicts.
        """
        permission_store = self._get_permission_store()
        if permission_store is None:
            return []

        try:
            import aiosqlite
            async with aiosqlite.connect(permission_store.db_path) as db:
                db.row_factory = aiosqlite.Row
                if since:
                    cursor = await db.execute(
                        """SELECT id, feature_name, tool_name, action, decision,
                                  user_choice, args_summary, created_at
                           FROM security_audit_log
                           WHERE created_at > ?
                           ORDER BY created_at ASC""",
                        (since,),
                    )
                else:
                    cursor = await db.execute(
                        """SELECT id, feature_name, tool_name, action, decision,
                                  user_choice, args_summary, created_at
                           FROM security_audit_log
                           ORDER BY created_at ASC"""
                    )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.warning(f"Could not read audit log entries: {e}")
            return []

    async def _get_audit_entries_range(
        self, first_at: Optional[str], last_at: Optional[str]
    ) -> list:
        """
        Get audit log entries within a time range (inclusive).

        Args:
            first_at: Start timestamp (inclusive).
            last_at: End timestamp (inclusive).

        Returns:
            List of entry dicts.
        """
        permission_store = self._get_permission_store()
        if permission_store is None:
            return []

        try:
            import aiosqlite
            async with aiosqlite.connect(permission_store.db_path) as db:
                db.row_factory = aiosqlite.Row
                if first_at and last_at:
                    cursor = await db.execute(
                        """SELECT id, feature_name, tool_name, action, decision,
                                  user_choice, args_summary, created_at
                           FROM security_audit_log
                           WHERE created_at >= ? AND created_at <= ?
                           ORDER BY created_at ASC""",
                        (first_at, last_at),
                    )
                else:
                    cursor = await db.execute(
                        """SELECT id, feature_name, tool_name, action, decision,
                                  user_choice, args_summary, created_at
                           FROM security_audit_log
                           ORDER BY created_at ASC"""
                    )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.warning(f"Could not read audit log entries for range: {e}")
            return []

    async def _get_last_anchor_timestamp(self) -> Optional[str]:
        """Get the created_at timestamp of the most recent anchor, or None."""
        db = self._get_db()
        if db is None:
            return None

        try:
            # Check if table exists first
            exists = await db.table_exists("audit_anchors")
            if not exists:
                return None

            row = await db.fetchone(
                """SELECT last_entry_at FROM audit_anchors
                   ORDER BY created_at DESC LIMIT 1"""
            )
            return row[0] if row else None
        except Exception as e:
            logger.debug(f"Could not get last anchor timestamp: {e}")
            return None

    async def _count_anchors(self) -> int:
        """Count total number of anchors."""
        db = self._get_db()
        if db is None:
            return 0

        try:
            exists = await db.table_exists("audit_anchors")
            if not exists:
                return 0

            val = await db.fetchval("SELECT COUNT(*) FROM audit_anchors")
            return val or 0
        except Exception:
            return 0

    async def _get_all_anchors(self) -> list:
        """Get all anchors ordered by creation time."""
        db = self._get_db()
        if db is None:
            return []

        try:
            exists = await db.table_exists("audit_anchors")
            if not exists:
                return []

            rows = await db.fetchall(
                """SELECT id, agent_id, anchor_hash, storage_ref,
                          entries_count, first_entry_at, last_entry_at, created_at
                   FROM audit_anchors
                   ORDER BY created_at ASC"""
            )
            columns = [
                "id", "agent_id", "anchor_hash", "storage_ref",
                "entries_count", "first_entry_at", "last_entry_at", "created_at",
            ]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.warning(f"Could not retrieve anchors: {e}")
            return []
