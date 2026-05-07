"""
Bridge Feature -- exposes the sovereign brain for external gateway integration.

Provides:
- !bridge status    -- show bridge configuration and connection status
- !bridge connections -- list active bridge connections
- !bridge history   -- recent bridge invocations

Database tables (created on initialize):
- bridge_sessions: maps gateway sessions to agent sessions
- bridge_log: audit log of all bridge invocations
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sdk.tools.base import ToolCategory

from .protocol import BridgeSession, ChannelType

logger = logging.getLogger(__name__)

# Maximum number of active sessions before we start pruning stale ones
MAX_ACTIVE_SESSIONS = 1000

# Sessions idle longer than this (seconds) are considered stale
SESSION_IDLE_TIMEOUT_SECONDS = 24 * 3600  # 24 hours


class BridgeFeature(Feature):
    """
    Feature for KestrelClaw Bridge integration.

    Manages bridge sessions, logs invocations, and provides status
    commands. The actual HTTP endpoints live in router.py and call
    into this feature's public methods.
    """

    @property
    def tool_description(self) -> str:
        return (
            "KestrelClaw Bridge - manage external gateway connections, "
            "view bridge status, list active sessions, and review invocation history"
        )

    async def initialize(self):
        """Initialize the bridge feature: resolve DB handle, create tables."""
        self._db = None
        self._agent_id = ""

        # In-memory session cache (gateway_session_id -> BridgeSession)
        self._sessions: Dict[str, BridgeSession] = {}

        # In-memory invocation counter (for status reporting)
        self._invocation_count = 0
        self._start_time = time.monotonic()

        self._db = resolve_feature_database(self.agent)

        # Agent identity (DID is the canonical source of truth)
        self._agent_id = self.agent.did

        # Create database tables
        if self._db:
            try:
                await self._db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bridge_sessions (
                        id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        gateway_session_id TEXT,
                        channel_type TEXT NOT NULL DEFAULT 'api',
                        sender_id TEXT,
                        created_at TEXT NOT NULL,
                        last_activity_at TEXT NOT NULL
                    )
                    """
                )
                await self._db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_bridge_sessions_gateway
                    ON bridge_sessions(gateway_session_id)
                    """
                )
                await self._db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_bridge_sessions_agent
                    ON bridge_sessions(agent_id, last_activity_at DESC)
                    """
                )
                await self._db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bridge_log (
                        id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        session_id TEXT,
                        direction TEXT NOT NULL DEFAULT 'inbound',
                        content_preview TEXT,
                        tokens_used INTEGER DEFAULT 0,
                        duration_ms INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                await self._db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_bridge_log_agent
                    ON bridge_log(agent_id, created_at DESC)
                    """
                )
                await self._db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_bridge_log_session
                    ON bridge_log(session_id, created_at DESC)
                    """
                )
                logger.info("BridgeFeature: database tables ready")
            except Exception as e:
                logger.warning(f"BridgeFeature: could not create tables: {e}")

        logger.info("BridgeFeature initialized")

    # =========================================================================
    # Tool commands (exposed to the agent via !bridge ...)
    # =========================================================================

    @tool(
        name="bridge_status",
        description="Show bridge configuration and connection status",
        category=ToolCategory.SYSTEM,
        command_prefix="!bridge status",
    )
    async def bridge_status(self) -> Dict[str, Any]:
        """Show bridge configuration and current status.

        Returns:
            Dict with uptime, session count, invocation count, and config
        """
        uptime_seconds = time.monotonic() - self._start_time

        # Count active sessions from DB if available
        db_session_count = 0
        if self._db:
            try:
                row = await self._db.fetchone(
                    "SELECT COUNT(*) FROM bridge_sessions WHERE agent_id = ?",
                    (self._agent_id,),
                )
                db_session_count = row[0] if row else 0
            except Exception:
                pass

        # Count recent log entries
        recent_invocations = 0
        if self._db:
            try:
                row = await self._db.fetchone(
                    """
                    SELECT COUNT(*) FROM bridge_log
                    WHERE agent_id = ? AND created_at > datetime('now', '-1 hour')
                    """,
                    (self._agent_id,),
                )
                recent_invocations = row[0] if row else 0
            except Exception:
                pass

        return {
            "status": "active",
            "agent_id": self._agent_id,
            "uptime_seconds": round(uptime_seconds, 1),
            "active_sessions_memory": len(self._sessions),
            "total_sessions_db": db_session_count,
            "total_invocations": self._invocation_count,
            "recent_invocations_1h": recent_invocations,
            "database_available": self._db is not None,
            "max_active_sessions": MAX_ACTIVE_SESSIONS,
            "session_idle_timeout_seconds": SESSION_IDLE_TIMEOUT_SECONDS,
        }

    @tool(
        name="bridge_connections",
        description="List active bridge connections/sessions",
        category=ToolCategory.SYSTEM,
        command_prefix="!bridge connections",
    )
    async def bridge_connections(self, limit: int = 20) -> Dict[str, Any]:
        """List active bridge sessions.

        Args:
            limit: Maximum number of sessions to return (default 20)

        Returns:
            Dict with list of active sessions
        """
        sessions = []

        # Try database first
        if self._db:
            try:
                rows = await self._db.fetchall(
                    """
                    SELECT id, gateway_session_id, channel_type, sender_id,
                           created_at, last_activity_at
                    FROM bridge_sessions
                    WHERE agent_id = ?
                    ORDER BY last_activity_at DESC
                    LIMIT ?
                    """,
                    (self._agent_id, limit),
                )
                for row in rows:
                    sessions.append({
                        "id": row[0],
                        "gateway_session_id": row[1],
                        "channel_type": row[2],
                        "sender_id": row[3],
                        "created_at": row[4],
                        "last_activity_at": row[5],
                    })
            except Exception as e:
                logger.warning(f"BridgeFeature: connections query failed: {e}")

        # Fallback to in-memory sessions
        if not sessions:
            sorted_sessions = sorted(
                self._sessions.values(),
                key=lambda s: s.last_activity_at,
                reverse=True,
            )
            for s in sorted_sessions[:limit]:
                sessions.append({
                    "id": s.id,
                    "gateway_session_id": s.gateway_session_id,
                    "channel_type": s.channel_type.value if isinstance(s.channel_type, ChannelType) else s.channel_type,
                    "sender_id": s.sender_id,
                    "created_at": s.created_at.isoformat(),
                    "last_activity_at": s.last_activity_at.isoformat(),
                })

        return {
            "sessions": sessions,
            "count": len(sessions),
        }

    @tool(
        name="bridge_history",
        description="Show recent bridge invocation history",
        category=ToolCategory.SYSTEM,
        command_prefix="!bridge history",
    )
    async def bridge_history(self, limit: int = 20) -> Dict[str, Any]:
        """Show recent bridge invocations.

        Args:
            limit: Maximum number of log entries to return (default 20)

        Returns:
            Dict with list of recent invocations
        """
        entries = []

        if self._db:
            try:
                rows = await self._db.fetchall(
                    """
                    SELECT id, session_id, direction, content_preview,
                           tokens_used, duration_ms, created_at
                    FROM bridge_log
                    WHERE agent_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (self._agent_id, limit),
                )
                for row in rows:
                    entries.append({
                        "id": row[0],
                        "session_id": row[1],
                        "direction": row[2],
                        "content_preview": row[3],
                        "tokens_used": row[4],
                        "duration_ms": row[5],
                        "created_at": row[6],
                    })
            except Exception as e:
                logger.warning(f"BridgeFeature: history query failed: {e}")

        return {
            "entries": entries,
            "count": len(entries),
        }

    # =========================================================================
    # Public API (called by router.py endpoints)
    # =========================================================================

    async def get_or_create_session(
        self,
        gateway_session_id: Optional[str],
        channel_type: ChannelType = ChannelType.API,
        sender_id: Optional[str] = None,
    ) -> BridgeSession:
        """
        Get an existing session by gateway_session_id, or create a new one.

        Args:
            gateway_session_id: The gateway's session identifier (can be None
                for one-shot requests)
            channel_type: The channel type for this session
            sender_id: The sender within the gateway

        Returns:
            BridgeSession instance
        """
        now = datetime.now(timezone.utc)

        # Try memory cache first
        if gateway_session_id and gateway_session_id in self._sessions:
            session = self._sessions[gateway_session_id]
            session.touch()
            # Persist the updated last_activity_at
            await self._persist_session(session)
            return session

        # Try database lookup
        if gateway_session_id and self._db:
            try:
                row = await self._db.fetchone(
                    """
                    SELECT id, agent_id, gateway_session_id, channel_type,
                           sender_id, created_at, last_activity_at
                    FROM bridge_sessions
                    WHERE gateway_session_id = ? AND agent_id = ?
                    """,
                    (gateway_session_id, self._agent_id),
                )
                if row:
                    session = BridgeSession(
                        id=row[0],
                        agent_id=row[1],
                        gateway_session_id=row[2],
                        channel_type=ChannelType(row[3]) if row[3] else ChannelType.API,
                        sender_id=row[4],
                        created_at=datetime.fromisoformat(row[5]) if row[5] else now,
                        last_activity_at=now,
                    )
                    self._sessions[gateway_session_id] = session
                    await self._persist_session(session)
                    return session
            except Exception as e:
                logger.warning(f"BridgeFeature: session lookup failed: {e}")

        # Create new session
        session = BridgeSession(
            id=str(uuid.uuid4()),
            agent_id=self._agent_id,
            gateway_session_id=gateway_session_id,
            channel_type=channel_type,
            sender_id=sender_id,
            created_at=now,
            last_activity_at=now,
        )

        # Cache in memory
        cache_key = gateway_session_id or session.id
        self._sessions[cache_key] = session

        # Prune stale sessions if cache is getting large
        if len(self._sessions) > MAX_ACTIVE_SESSIONS:
            await self._prune_stale_sessions()

        # Persist to database
        await self._persist_session(session, is_new=True)

        logger.info(
            f"BridgeFeature: new session {session.id} "
            f"(gateway={gateway_session_id}, channel={channel_type.value})"
        )
        return session

    async def log_invocation(
        self,
        session_id: str,
        direction: str,
        content_preview: str,
        tokens_used: int = 0,
        duration_ms: int = 0,
    ) -> None:
        """
        Log a bridge invocation (inbound request or outbound response).

        Args:
            session_id: The bridge session ID
            direction: 'inbound' or 'outbound'
            content_preview: First ~200 chars of the content
            tokens_used: Tokens consumed (0 for inbound)
            duration_ms: Processing duration in milliseconds
        """
        self._invocation_count += 1

        if not self._db:
            return

        log_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Truncate preview to 200 chars
        if len(content_preview) > 200:
            content_preview = content_preview[:197] + "..."

        try:
            await self._db.execute(
                """
                INSERT INTO bridge_log
                (id, agent_id, session_id, direction, content_preview,
                 tokens_used, duration_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log_id,
                    self._agent_id,
                    session_id,
                    direction,
                    content_preview,
                    tokens_used,
                    duration_ms,
                    now,
                ),
            )
        except Exception as e:
            logger.warning(f"BridgeFeature: failed to log invocation: {e}")

    def get_capabilities(self) -> List[Dict[str, Any]]:
        """
        Build the capabilities list from all registered agent features/tools.

        Returns:
            List of capability dicts for the capabilities endpoint
        """
        capabilities = []
        features_dict = getattr(self.agent, "features", {})

        for feature_name, feature in features_dict.items():
            try:
                for agent_tool in feature.get_tools():
                    schema = agent_tool.schema
                    capabilities.append({
                        "name": schema.name,
                        "description": schema.description,
                        "category": schema.category.value,
                        "command_prefix": schema.command_prefix,
                        "feature": feature_name,
                        "parameters": [
                            {
                                "name": p.name,
                                "type": p.type,
                                "description": p.description,
                                "required": p.required,
                            }
                            for p in schema.parameters
                        ],
                    })
            except Exception as e:
                logger.warning(
                    f"BridgeFeature: failed to get tools for {feature_name}: {e}"
                )

        return capabilities

    # =========================================================================
    # Internal helpers
    # =========================================================================

    async def _persist_session(
        self, session: BridgeSession, is_new: bool = False
    ) -> None:
        """Persist a session to the database."""
        if not self._db:
            return

        try:
            if is_new:
                await self._db.execute(
                    """
                    INSERT INTO bridge_sessions
                    (id, agent_id, gateway_session_id, channel_type,
                     sender_id, created_at, last_activity_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.id,
                        session.agent_id,
                        session.gateway_session_id,
                        session.channel_type.value if isinstance(session.channel_type, ChannelType) else session.channel_type,
                        session.sender_id,
                        session.created_at.isoformat(),
                        session.last_activity_at.isoformat(),
                    ),
                )
            else:
                await self._db.execute(
                    """
                    UPDATE bridge_sessions
                    SET last_activity_at = ?
                    WHERE id = ?
                    """,
                    (session.last_activity_at.isoformat(), session.id),
                )
        except Exception as e:
            logger.warning(f"BridgeFeature: failed to persist session: {e}")

    async def _prune_stale_sessions(self) -> None:
        """Remove sessions that have been idle beyond the timeout."""
        now = datetime.now(timezone.utc)
        stale_keys = []
        for key, session in self._sessions.items():
            idle_seconds = (now - session.last_activity_at).total_seconds()
            if idle_seconds > SESSION_IDLE_TIMEOUT_SECONDS:
                stale_keys.append(key)

        for key in stale_keys:
            del self._sessions[key]

        if stale_keys:
            logger.info(f"BridgeFeature: pruned {len(stale_keys)} stale sessions")
