"""
Kestrel Security - Hierarchical Permission Storage.

This module provides SQLite-backed storage for tool permissions with:
- Feature → Tool hierarchy
- Session-scoped overrides
- Rollup state calculation for UI
- Audit logging
"""

import aiosqlite
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """
    Permission levels for tools.

    - ALLOW: Always allow the tool to execute
    - AUTO: Auto-approve after earlier constitutional/honesty/security hooks
      have not blocked the call
    - DENY: Always deny the tool execution
    - ASK: Ask for user approval each time (default for new tools)
    - SESSION: Allow for the current session only (not persisted)
    """
    ALLOW = "allow"
    AUTO = "auto"
    DENY = "deny"
    ASK = "ask"
    SESSION = "session"


@dataclass
class ToolPermission:
    """Permission configuration for a single tool."""
    feature_name: str
    tool_name: str
    level: PermissionLevel
    reason: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class FeaturePermissions:
    """
    Aggregated permissions for a feature with rollup state.

    The rollup_state property calculates the aggregate state of all
    tools in this feature, used for UI display.
    """
    feature_name: str
    tools: List[ToolPermission]

    @property
    def rollup_state(self) -> str:
        """
        Calculate rollup state from children.

        Returns:
            - "allow_all": All tools are ALLOW
            - "auto_all": All tools are AUTO
            - "deny_all": All tools are DENY
            - "ask_all": All tools are ASK
            - "session_all": All tools are SESSION
            - "mixed": Tools have different settings
        """
        if not self.tools:
            return "ask_all"  # Default

        levels = {t.level for t in self.tools}

        if len(levels) == 1:
            level = list(levels)[0]
            return f"{level.value}_all"

        return "mixed"


def _snake_case(name: str) -> str:
    """Convert PascalCase to snake_case the same way `Feature.tool_name` does.
    Used at the storage boundary to look up legacy rows under both casings.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _pascal_case(name: str) -> str:
    """Best-effort inverse of ``_snake_case`` for the common snake form
    ``word_word_word``: capitalize each underscored part and concatenate.

    Lossy: e.g. ``m_c_p_agent`` → ``MCPAgent`` (correct round-trip) and
    ``mcp_agent`` → ``McpAgent`` (NOT the original class name, but still
    fine — that class name doesn't exist in the codebase anyway). Used
    only for the lookup-time variant search; we don't write under this
    form, so any oddness stays at read-time only."""
    if "_" not in name:
        return name
    parts = [p for p in name.split("_") if p]
    if not parts:
        return name
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _name_variants(name: str) -> set[str]:
    """Return the set of feature_name strings to try at lookup time:
    the input, its snake form, and a candidate PascalCase form. See the
    detailed rationale on `_lookup_rows` for why we accept both."""
    return {name, _snake_case(name), _pascal_case(name)}


# Permissive-wins ordering. ALLOW/AUTO beat ASK/SESSION beat DENY-not-found.
# DENY is a hard stop and only "wins" when it's the only thing on the row —
# if any matching row says ALLOW the operator's grant is honored. This is
# the same intent encoded in the global Auto-mode short-circuit, kept
# consistent at the per-row level.
_PERMISSIVENESS = {
    PermissionLevel.ALLOW: 4,
    PermissionLevel.AUTO: 3,
    PermissionLevel.SESSION: 2,
    PermissionLevel.ASK: 1,
    PermissionLevel.DENY: 0,
}


def _most_permissive(levels: List["PermissionLevel"]) -> "PermissionLevel":
    return max(levels, key=lambda level: _PERMISSIVENESS.get(level, 0))


class PermissionStore:
    """
    SQLite-backed hierarchical permission storage.

    Features:
    - Persistent storage of permission settings
    - Session-scoped overrides (not persisted)
    - Audit logging of permission decisions
    - Hierarchical tree retrieval for UI

    Example:
        store = PermissionStore("kestrel_prime.db")
        await store.initialize()

        # Set permission
        await store.set_permission("ModelAgent", "list_models", PermissionLevel.ALLOW)

        # Get permission (checks session overrides first)
        level = await store.get_permission("WalletAgent", "send_payment")
        # Returns: PermissionLevel.ASK (default)

        # Get full tree for UI
        tree = await store.get_permission_tree()
    """

    def __init__(self, db_path: str):
        """
        Initialize the permission store.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._session_overrides: Dict[str, PermissionLevel] = {}
        self._global_auto_mode = False
        self._initialized = False

    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS security_permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    level TEXT NOT NULL DEFAULT 'ask',
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(feature_name, tool_name)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS security_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT,
                    tool_name TEXT,
                    action TEXT,
                    decision TEXT,
                    user_choice TEXT,
                    args_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Index for faster permission lookups
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_security_permissions_lookup
                ON security_permissions(feature_name, tool_name)
            """)

            # Index for audit log queries
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_security_audit_created
                ON security_audit_log(created_at DESC)
            """)

            # Sovereign-curated auto-approve allowlist (the "Approve-and-
            # remember" store). Operator-seeded rules live in kestrel.toml;
            # these are the rows the Sovereign added via the Mews approval
            # panel and can revoke by deleting the row. See
            # kestrel_sovereign/security/auto_approve.py.
            await db.execute("""
                CREATE TABLE IF NOT EXISTS auto_approve_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT,
                    pattern TEXT NOT NULL,
                    repo_scope TEXT NOT NULL DEFAULT '',
                    added_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(agent, pattern, repo_scope)
                )
            """)

            # Full, immutable audit row for every auto-approved invocation.
            # security_audit_log lacks agent_did/command/exit_code columns;
            # this is the "no silent runs" record the constitution requires.
            # Two-phase: a row is inserted at approve-time (exit_code NULL)
            # and finalized with the real exit code once the tool returns.
            await db.execute("""
                CREATE TABLE IF NOT EXISTS auto_approve_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_did TEXT,
                    agent_name TEXT,
                    feature_name TEXT,
                    tool_name TEXT,
                    command TEXT,
                    pattern TEXT,
                    repo_scope TEXT,
                    rule_source TEXT,
                    decision TEXT NOT NULL DEFAULT 'auto_approved',
                    exit_code INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_auto_approve_audit_created
                ON auto_approve_audit(created_at DESC)
            """)

            await db.commit()

        self._initialized = True
        logger.info("PermissionStore initialized")

    async def migrate_legacy_feature_aliases(
        self,
        aliases: Dict[str, str],
    ) -> int:
        """One-time consolidation of legacy snake_case/alias rows into
        ``feature.name`` (PascalCase) rows.

        Different code paths historically wrote ``security_permissions``
        under different feature_name conventions: the subagent path used
        ``feature.name`` (class name, e.g. ``ComputerUseFeature``); the
        direct-tool path used ``feature.tool_name`` (snake-case alias,
        e.g. ``computer_use``); the operator's ``!security-set`` call used
        whatever the user typed. The orchestrator now normalizes to
        PascalCase (#1427), but agents already in operation have months of
        snake-rowed grants the operator intended to keep.

        This migration copies the more permissive of any (alias, tool)
        row pair into the (PascalCase, tool) row, preserving operator
        intent. The alias rows themselves are kept (audit trail) so a
        rollback can re-read them; lookups now resolve via the canonical
        PascalCase row.

        Args:
            aliases: Mapping of ``feature.tool_name`` → ``feature.name``,
                built from the loaded feature registry. The caller (agent
                init) supplies this — the store has no knowledge of which
                Feature classes are loaded.

        Returns:
            Number of (feature, tool) pairs whose canonical row was
            upserted from a more-permissive alias row.
        """
        if not aliases:
            return 0
        upserts = 0
        async with aiosqlite.connect(self.db_path) as db:
            for alias, canonical in aliases.items():
                if alias == canonical:
                    continue
                cursor = await db.execute(
                    "SELECT tool_name, level FROM security_permissions "
                    "WHERE feature_name = ?",
                    (alias,),
                )
                alias_rows = await cursor.fetchall()
                if not alias_rows:
                    continue
                for tool_name, raw_level in alias_rows:
                    try:
                        alias_level = PermissionLevel(raw_level)
                    except ValueError:
                        continue
                    cursor = await db.execute(
                        "SELECT level FROM security_permissions "
                        "WHERE feature_name = ? AND tool_name = ?",
                        (canonical, tool_name),
                    )
                    row = await cursor.fetchone()
                    if row:
                        try:
                            existing = PermissionLevel(row[0])
                        except ValueError:
                            existing = PermissionLevel.ASK
                        winner = _most_permissive([existing, alias_level])
                        if winner == existing:
                            continue
                        await db.execute(
                            "UPDATE security_permissions SET level = ?, "
                            "updated_at = CURRENT_TIMESTAMP "
                            "WHERE feature_name = ? AND tool_name = ?",
                            (winner.value, canonical, tool_name),
                        )
                    else:
                        await db.execute(
                            "INSERT INTO security_permissions "
                            "(feature_name, tool_name, level, reason) "
                            "VALUES (?, ?, ?, ?)",
                            (
                                canonical,
                                tool_name,
                                alias_level.value,
                                f"Migrated from legacy alias '{alias}' (#1427)",
                            ),
                        )
                    upserts += 1
            await db.commit()
        if upserts:
            logger.info(
                "Permission alias migration: %d row(s) consolidated into "
                "canonical PascalCase rows.", upserts,
            )
        return upserts

    async def get_permission(
        self,
        feature_name: str,
        tool_name: str
    ) -> PermissionLevel:
        """
        Get permission level for a tool.

        Checks in order:
        1. Session overrides (in-memory)
        2. Persistent storage
        3. Default (ASK)

        Args:
            feature_name: Name of the feature
            tool_name: Name of the tool

        Returns:
            PermissionLevel for the tool
        """
        key = f"{feature_name}.{tool_name}"

        # Session override takes priority. In global Auto mode, an explicit
        # DENY remains a hard stop; everything else can flow through Auto.
        if key in self._session_overrides:
            level = self._session_overrides[key]
            if self._global_auto_mode and level != PermissionLevel.DENY:
                return PermissionLevel.AUTO
            return level

        # Check persistent storage.
        #
        # The DB has historically accumulated a mix of casings for the same
        # logical feature: PascalCase (`TaskFeature.respond_to_a2a_task`,
        # written from the subagent-dispatch path and the operator
        # ``!security-set`` tool) and snake_case (`task_feature.…`, written
        # from the older direct-tool dispatch path before #1427's
        # normalization). Either form may be the source of truth on a given
        # row, so look up BOTH variants and resolve any tie by preferring
        # the more permissive level — that matches the operator's intent:
        # if they ever granted ALLOW under either name, the tool is allowed
        # (rather than re-asking just because the bookkeeping shifted).
        # The orchestrator's new lookup canonicalizes to PascalCase, so
        # future writes converge on the class-name form; this fallback
        # exists for backward compatibility with the mixed-case state on
        # already-running agents (#1427 sibling).
        rows = await self._lookup_rows(feature_name, tool_name)
        if rows:
            best = _most_permissive(rows)
            if self._global_auto_mode and best != PermissionLevel.DENY:
                return PermissionLevel.AUTO
            return best

        # Default for unregistered tools.
        # Demo servers (KESTREL_DEMO_SERVER=1) auto-allow — _register_all_tools
        # only catches sub-tools (web_search), not feature-as-subagent
        # invocations (web_search_feature, the snake-cased class name from
        # Feature.tool_name). Without this, the modal still pops for
        # subagent-level calls even though every sub-tool is ALLOW. Demo
        # subjects that aren't security shouldn't have to chase that.
        import os as _os
        if _os.environ.get("KESTREL_DEMO_SERVER", "").lower() in ("1", "true", "yes"):
            return PermissionLevel.ALLOW
        if self._global_auto_mode:
            return PermissionLevel.AUTO
        return PermissionLevel.ASK

    async def _lookup_rows(
        self,
        feature_name: str,
        tool_name: str,
    ) -> List[PermissionLevel]:
        """Look up `security_permissions` rows for both PascalCase and
        snake_case forms of ``feature_name`` (e.g. ``TaskFeature`` and
        ``task_feature``) and return all matching `PermissionLevel` values.

        See ``get_permission`` for the rationale — the DB has accumulated
        rows under both casings over time and we want operator grants to
        survive the orchestrator-side normalization to PascalCase (#1427).
        """
        names = _name_variants(feature_name)
        placeholders = ",".join(["?"] * len(names))
        query = (
            f"SELECT level FROM security_permissions "
            f"WHERE feature_name IN ({placeholders}) AND tool_name = ?"
        )
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, (*names, tool_name))
            rows = await cursor.fetchall()
        levels: List[PermissionLevel] = []
        for row in rows:
            try:
                levels.append(PermissionLevel(row[0]))
            except ValueError:
                # Stale level value from a removed enum variant — ignore.
                continue
        return levels

    async def set_permission(
        self,
        feature_name: str,
        tool_name: str,
        level: PermissionLevel,
        scope: str = "always",
        reason: Optional[str] = None,
    ) -> None:
        """
        Set permission for a tool.

        Args:
            feature_name: Name of the feature
            tool_name: Name of the tool
            level: Permission level to set
            scope: "always" (persistent), "session" (in-memory only), "once" (no storage)
            reason: Optional reason for the permission change
        """
        key = f"{feature_name}.{tool_name}"

        if scope == "session":
            self._session_overrides[key] = level
            logger.info(f"Set session permission: {key} = {level.value}")

        elif scope == "always":
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO security_permissions
                    (feature_name, tool_name, level, reason, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(feature_name, tool_name)
                    DO UPDATE SET level = excluded.level,
                                  reason = excluded.reason,
                                  updated_at = CURRENT_TIMESTAMP
                """, (feature_name, tool_name, level.value, reason))
                await db.commit()

            logger.info(f"Set persistent permission: {key} = {level.value}")

        # scope == "once" - no storage, just allow this execution

    async def set_feature_permission(
        self,
        feature_name: str,
        level: PermissionLevel,
        reason: Optional[str] = None,
    ) -> None:
        """
        Set all tools in a feature to the same level (bulk update).

        Args:
            feature_name: Name of the feature
            level: Permission level to set for all tools
            reason: Optional reason for the permission change
        """
        # Get all tools for this feature
        tree = await self.get_permission_tree()
        feature = next((f for f in tree if f.feature_name == feature_name), None)

        if feature:
            for tool in feature.tools:
                await self.set_permission(
                    feature_name,
                    tool.tool_name,
                    level,
                    scope="always",
                    reason=reason
                )

        logger.info(f"Set feature permission: {feature_name} = {level.value}")

    async def get_permission_tree(self) -> List[FeaturePermissions]:
        """
        Get full hierarchical permission tree for UI.

        Returns all features and their tools with current permission levels,
        including rollup states for display.

        Returns:
            List of FeaturePermissions with tools and rollup states
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT feature_name, tool_name, level, reason, created_at, updated_at
                FROM security_permissions
                ORDER BY feature_name, tool_name
            """)
            rows = await cursor.fetchall()

        # Group by feature
        features_dict: Dict[str, List[ToolPermission]] = {}

        for row in rows:
            feature_name = row["feature_name"]
            tool = ToolPermission(
                feature_name=feature_name,
                tool_name=row["tool_name"],
                level=PermissionLevel(row["level"]),
                reason=row["reason"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

            # Apply session override if present
            key = f"{feature_name}.{tool.tool_name}"
            if key in self._session_overrides:
                tool.level = self._session_overrides[key]

            if feature_name not in features_dict:
                features_dict[feature_name] = []
            features_dict[feature_name].append(tool)

        return [
            FeaturePermissions(feature_name=name, tools=tools)
            for name, tools in features_dict.items()
        ]

    async def register_tool(
        self,
        feature_name: str,
        tool_name: str,
        default_level: PermissionLevel = PermissionLevel.ASK,
    ) -> None:
        """
        Register a tool with default permission (if not already registered).

        Called when features are loaded to ensure all tools are in the tree.

        Args:
            feature_name: Name of the feature
            tool_name: Name of the tool
            default_level: Default permission level (default: ASK)
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR IGNORE INTO security_permissions
                (feature_name, tool_name, level)
                VALUES (?, ?, ?)
            """, (feature_name, tool_name, default_level.value))
            await db.commit()

    async def log_decision(
        self,
        feature_name: str,
        tool_name: str,
        action: str,
        decision: str,
        user_choice: Optional[str] = None,
        args_summary: Optional[str] = None,
    ) -> None:
        """
        Log a permission decision to the audit log.

        Args:
            feature_name: Name of the feature
            tool_name: Name of the tool
            action: Action type (e.g., "tool_execution", "permission_change")
            decision: Decision made (e.g., "allowed", "denied", "user_approved")
            user_choice: User's choice if they were asked (once/session/always)
            args_summary: Summary of tool arguments (truncated for privacy)
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO security_audit_log
                (feature_name, tool_name, action, decision, user_choice, args_summary)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (feature_name, tool_name, action, decision, user_choice, args_summary))
            await db.commit()

    async def get_audit_log(self, limit: int = 50) -> List[Dict]:
        """
        Get recent audit log entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of audit log entries (most recent first)
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT feature_name, tool_name, action, decision,
                       user_choice, args_summary, created_at
                FROM security_audit_log
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()

        return [
            {
                "feature": row["feature_name"],
                "tool": row["tool_name"],
                "action": row["action"],
                "decision": row["decision"],
                "user_choice": row["user_choice"],
                "args_summary": row["args_summary"],
                "timestamp": row["created_at"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Auto-approve allowlist (Sovereign-curated, revocable)
    # ------------------------------------------------------------------

    async def list_auto_approve_rules(self) -> List[Dict]:
        """Return the dynamic (DB-backed) auto-approve rules."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT id, agent, pattern, repo_scope, added_by, created_at
                FROM auto_approve_rules ORDER BY created_at DESC
            """)
            rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "agent": r["agent"],
                "pattern": r["pattern"],
                "repo_scope": r["repo_scope"],
                "added_by": r["added_by"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def add_auto_approve_rule(
        self,
        *,
        pattern: str,
        repo_scope: str = "",
        agent: Optional[str] = None,
        added_by: Optional[str] = None,
    ) -> None:
        """Add (or no-op if duplicate) a Sovereign-curated allowlist rule.

        This is what the Mews "Approve-and-remember" button calls. Revoke
        by deleting the row (:meth:`remove_auto_approve_rule`).
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR IGNORE INTO auto_approve_rules
                (agent, pattern, repo_scope, added_by)
                VALUES (?, ?, ?, ?)
            """, (agent, pattern, repo_scope, added_by))
            await db.commit()
        logger.info(
            "auto_approve: remembered rule (agent=%s, repo=%s) added_by=%s",
            agent or "*", repo_scope or "*", added_by or "?",
        )

    async def remove_auto_approve_rule(self, rule_id: int) -> bool:
        """Revoke a dynamic rule. Returns True if a row was deleted."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM auto_approve_rules WHERE id = ?", (rule_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def log_auto_approve(
        self,
        *,
        agent_did: Optional[str],
        agent_name: Optional[str],
        feature_name: str,
        tool_name: str,
        command: str,
        pattern: str,
        repo_scope: str,
        rule_source: str,
    ) -> int:
        """Insert the phase-1 audit row; return its id for finalization.

        The row is written *before* the tool runs so an auto-approved
        invocation can never execute without an audit trail. ``exit_code``
        is filled in by :meth:`finalize_auto_approve` once the tool exits.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO auto_approve_audit
                (agent_did, agent_name, feature_name, tool_name, command,
                 pattern, repo_scope, rule_source, decision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'auto_approved')
            """, (
                agent_did, agent_name, feature_name, tool_name, command,
                pattern, repo_scope, rule_source,
            ))
            await db.commit()
            return int(cursor.lastrowid)

    async def finalize_auto_approve(
        self, audit_id: int, exit_code: int
    ) -> None:
        """Stamp the real exit code + completion time on a phase-1 row."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE auto_approve_audit
                SET exit_code = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ? AND completed_at IS NULL
            """, (exit_code, audit_id))
            await db.commit()

    async def get_auto_approve_audit(self, limit: int = 50) -> List[Dict]:
        """Recent auto-approve audit rows (most recent first)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT id, agent_did, agent_name, feature_name, tool_name,
                       command, pattern, repo_scope, rule_source, decision,
                       exit_code, created_at, completed_at
                FROM auto_approve_audit ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    def clear_session_overrides(self) -> None:
        """Clear all session-scoped permission overrides and global Auto."""
        count = len(self._session_overrides)
        self._session_overrides.clear()
        self._global_auto_mode = False
        logger.info(f"Cleared {count} session overrides and disabled global Auto")

    def set_global_auto_mode(self, enabled: bool) -> None:
        """Enable or disable session-scoped global Auto mode."""
        self._global_auto_mode = bool(enabled)
        logger.warning(
            "Global security Auto mode %s",
            "enabled" if self._global_auto_mode else "disabled",
        )

    def get_global_auto_mode(self) -> bool:
        """Return whether session-scoped global Auto mode is enabled."""
        return self._global_auto_mode

    def __repr__(self) -> str:
        return (
            f"PermissionStore(db={self.db_path}, "
            f"session_overrides={len(self._session_overrides)}, "
            f"global_auto_mode={self._global_auto_mode})"
        )
