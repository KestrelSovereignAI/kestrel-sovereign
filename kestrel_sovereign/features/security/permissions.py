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

            await db.commit()

        self._initialized = True
        logger.info("PermissionStore initialized")

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

        # Check persistent storage
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT level FROM security_permissions
                   WHERE feature_name = ? AND tool_name = ?""",
                (feature_name, tool_name)
            )
            row = await cursor.fetchone()
            if row:
                level = PermissionLevel(row[0])
                if self._global_auto_mode and level != PermissionLevel.DENY:
                    return PermissionLevel.AUTO
                return level

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
