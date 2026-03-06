"""
Webhook Feature -- generic webhook receiver for the Kestrel agent.

Allows features and operators to register custom HTTP webhook endpoints
with configurable authentication. All webhook receives are logged for
security audit purposes.

Tools:
    !webhooks list                            -- show registered webhooks
    !webhooks history                         -- recent webhook receive log
    !webhooks register <name> <auth_type>     -- register a new webhook endpoint
    !webhooks remove <name>                   -- unregister a webhook

Database tables (created on initialize):
    webhook_config -- persisted webhook endpoint registrations
    webhook_log    -- audit log of every received webhook request
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

from .models import WebhookAuthType, WebhookConfig
from .receiver import WebhookReceiver

logger = logging.getLogger(__name__)


class WebhookFeature(Feature):
    """
    Generic webhook receiver feature.

    On ``initialize()``, creates database tables for webhook configuration
    and audit logging, loads any persisted webhooks, and makes the
    ``WebhookReceiver`` available for router mounting.

    The receiver's FastAPI router can be obtained via ``get_webhook_router()``
    and included in the application at mount time.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Manage generic webhook endpoints - register, remove, list webhooks "
            "and view recent webhook receive history for security audit"
        )

    async def initialize(self):
        """Initialise the feature: resolve DB, create tables, load persisted webhooks."""
        self._db = None
        self._agent_id = ""
        self.receiver = WebhookReceiver()

        # Resolve database handle from agent storage
        if hasattr(self.agent, "storage") and self.agent.storage:
            if hasattr(self.agent.storage, "db"):
                self._db = self.agent.storage.db
            elif hasattr(self.agent.storage, "database"):
                self._db = self.agent.storage.database

        if self._db is None and hasattr(self.agent, "_raw_storage"):
            raw = self.agent._raw_storage
            if hasattr(raw, "db"):
                self._db = raw.db

        # Agent ID
        self._agent_id = getattr(self.agent, "agent_id", "") or getattr(
            self.agent, "did", "unknown"
        )

        # Create tables
        if self._db:
            await self._ensure_tables()
            await self._load_persisted_webhooks()

        logger.info("WebhookFeature initialized (agent=%s)", self._agent_id)

    async def shutdown(self):
        """No background tasks to stop; provided for lifecycle completeness."""
        logger.info("WebhookFeature shutdown")

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    async def _ensure_tables(self) -> None:
        """Create the webhook_config and webhook_log tables if they do not exist."""
        try:
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS webhook_config (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    auth_type TEXT NOT NULL,
                    auth_config_json TEXT NOT NULL DEFAULT '{}',
                    event_type TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    rate_limit INTEGER NOT NULL DEFAULT 60,
                    created_at TEXT NOT NULL
                )
                """
            )
            await self._db.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_webhook_config_agent_name
                ON webhook_config(agent_id, name)
                """
            )
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS webhook_log (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    webhook_name TEXT NOT NULL,
                    source_ip TEXT NOT NULL DEFAULT '',
                    authenticated INTEGER NOT NULL DEFAULT 0,
                    status_code INTEGER NOT NULL DEFAULT 200,
                    payload_hash TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            await self._db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_webhook_log_agent
                ON webhook_log(agent_id, created_at DESC)
                """
            )
            logger.info("WebhookFeature: tables ready")
        except Exception as exc:
            logger.warning("WebhookFeature: could not create tables: %s", exc)

    async def _load_persisted_webhooks(self) -> None:
        """Load webhook configurations from the database into the receiver."""
        try:
            rows = await self._db.fetchall(
                """
                SELECT id, name, auth_type, auth_config_json, event_type,
                       enabled, rate_limit, created_at
                FROM webhook_config
                WHERE agent_id = ?
                """,
                (self._agent_id,),
            )

            for row in rows:
                try:
                    auth_config = json.loads(row[3]) if row[3] else {}
                except (json.JSONDecodeError, TypeError):
                    auth_config = {}

                config = WebhookConfig(
                    id=row[0],
                    name=row[1],
                    auth_type=WebhookAuthType(row[2]),
                    auth_config=auth_config,
                    event_type=row[4] or "",
                    enabled=bool(row[5]),
                    rate_limit=row[6] if row[6] is not None else 60,
                    agent_id=self._agent_id,
                    created_at=row[7],
                )

                try:
                    self.receiver.register_webhook(config)
                except Exception as exc:
                    logger.warning(
                        "WebhookFeature: failed to register persisted webhook '%s': %s",
                        config.name,
                        exc,
                    )

            logger.info(
                "WebhookFeature: loaded %d persisted webhooks", len(rows)
            )
        except Exception as exc:
            logger.warning("WebhookFeature: failed to load webhooks: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_webhook_router(self):
        """Return the FastAPI router for webhook endpoints.

        This router should be mounted in server.py (or equivalent).
        Routes do NOT go through the server-level API key middleware
        because webhook auth is handled per-endpoint.

        Returns:
            ``fastapi.APIRouter``
        """
        return self.receiver.get_router()

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool(
        "webhooks_list",
        "List all registered webhook endpoints",
        category=ToolCategory.SYSTEM,
        command_prefix="!webhooks list",
    )
    async def webhooks_list(self) -> Dict[str, Any]:
        """List all registered webhooks for this agent.

        Returns:
            Dict with a list of webhooks and count.
        """
        webhooks = self.receiver.list_webhooks()

        # Supplement with DB data if available and receiver is empty
        if not webhooks and self._db:
            try:
                rows = await self._db.fetchall(
                    """
                    SELECT id, name, auth_type, event_type, enabled,
                           rate_limit, created_at
                    FROM webhook_config
                    WHERE agent_id = ?
                    ORDER BY created_at ASC
                    """,
                    (self._agent_id,),
                )
                for row in rows:
                    webhooks.append({
                        "id": row[0],
                        "name": row[1],
                        "auth_type": row[2],
                        "event_type": row[3],
                        "enabled": bool(row[4]),
                        "rate_limit": row[5],
                        "created_at": row[6],
                    })
            except Exception as exc:
                logger.warning("WebhookFeature: DB query failed: %s", exc)

        return {"webhooks": webhooks, "count": len(webhooks)}

    @tool(
        "webhooks_history",
        "Show recent webhook receive log for security audit",
        category=ToolCategory.SYSTEM,
        command_prefix="!webhooks history",
    )
    async def webhooks_history(self, limit: int = 20) -> Dict[str, Any]:
        """Show recent webhook events.

        Args:
            limit: Maximum number of events to return (default: 20)

        Returns:
            Dict with a list of events and count.
        """
        # Try database first
        events: list = []
        if self._db:
            try:
                rows = await self._db.fetchall(
                    """
                    SELECT id, webhook_name, source_ip, authenticated,
                           status_code, payload_hash, created_at
                    FROM webhook_log
                    WHERE agent_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (self._agent_id, limit),
                )
                for row in rows:
                    events.append({
                        "id": row[0],
                        "webhook_name": row[1],
                        "source_ip": row[2],
                        "authenticated": bool(row[3]),
                        "status_code": row[4],
                        "payload_hash": row[5],
                        "created_at": row[6],
                    })
            except Exception as exc:
                logger.warning("WebhookFeature: history query failed: %s", exc)

        # Fallback to in-memory log
        if not events:
            events = self.receiver.get_recent_events(limit)

        return {"events": events, "count": len(events)}

    @tool(
        "webhooks_register",
        "Register a new webhook endpoint with authentication",
        category=ToolCategory.SYSTEM,
        command_prefix="!webhooks register",
    )
    async def webhooks_register(
        self,
        name: str,
        auth_type: str = "none",
        event_type: str = "",
        auth_config_json: str = "{}",
        rate_limit: int = 60,
    ) -> Dict[str, Any]:
        """Register a new webhook endpoint.

        Args:
            name: Unique webhook name (used in the URL path /webhooks/{name})
            auth_type: Authentication method (none, bearer_token, hmac_sha256, ip_allowlist)
            event_type: Optional event type label for categorisation
            auth_config_json: JSON config for auth (e.g. {"token":"secret"} for bearer_token)
            rate_limit: Maximum requests per minute (default: 60, 0 = unlimited)

        Returns:
            Dict with registration status and webhook details.
        """
        # Validate auth_type
        try:
            auth_type_enum = WebhookAuthType(auth_type)
        except ValueError:
            valid = [t.value for t in WebhookAuthType]
            return {
                "success": False,
                "error": f"Invalid auth_type '{auth_type}'. Must be one of: {valid}",
            }

        # Parse auth config
        try:
            auth_config = json.loads(auth_config_json)
            if not isinstance(auth_config, dict):
                return {"success": False, "error": "auth_config_json must be a JSON object"}
        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"Invalid auth_config_json: {exc}"}

        # Validate name
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            return {
                "success": False,
                "error": "Webhook name must be alphanumeric (hyphens and underscores allowed)",
            }

        # Build config
        webhook_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        config = WebhookConfig(
            id=webhook_id,
            name=name,
            auth_type=auth_type_enum,
            auth_config=auth_config,
            event_type=event_type,
            enabled=True,
            rate_limit=max(0, rate_limit),
            agent_id=self._agent_id,
            created_at=now,
        )

        # Register in receiver
        try:
            self.receiver.register_webhook(config)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        # Persist to database
        if self._db:
            try:
                await self._db.execute(
                    """
                    INSERT INTO webhook_config
                        (id, agent_id, name, auth_type, auth_config_json,
                         event_type, enabled, rate_limit, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        webhook_id,
                        self._agent_id,
                        name,
                        auth_type,
                        auth_config_json,
                        event_type,
                        max(0, rate_limit),
                        now,
                    ),
                )
            except Exception as exc:
                logger.warning("WebhookFeature: failed to persist webhook config: %s", exc)

        logger.info(
            "Registered webhook: %s (auth=%s, rate_limit=%d/min)",
            name, auth_type, rate_limit,
        )

        return {
            "success": True,
            "webhook_id": webhook_id,
            "name": name,
            "auth_type": auth_type,
            "event_type": event_type,
            "rate_limit": max(0, rate_limit),
            "endpoint": f"/webhooks/{name}",
            "created_at": now,
        }

    @tool(
        "webhooks_remove",
        "Remove a registered webhook endpoint",
        category=ToolCategory.SYSTEM,
        command_prefix="!webhooks remove",
    )
    async def webhooks_remove(self, name: str) -> Dict[str, Any]:
        """Remove a registered webhook.

        Args:
            name: The name of the webhook to remove

        Returns:
            Dict with removal status.
        """
        # Remove from receiver
        removed = self.receiver.unregister_webhook(name)

        # Remove from database
        if self._db:
            try:
                await self._db.execute(
                    "DELETE FROM webhook_config WHERE name = ? AND agent_id = ?",
                    (name, self._agent_id),
                )
            except Exception as exc:
                logger.warning("WebhookFeature: failed to delete from DB: %s", exc)

        if removed:
            return {"success": True, "name": name, "status": "removed"}
        else:
            return {"success": False, "error": f"Webhook '{name}' not found"}

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    async def log_webhook_event(
        self,
        *,
        webhook_name: str,
        source_ip: str,
        authenticated: bool,
        status_code: int,
        payload_hash: str,
    ) -> None:
        """Persist a webhook event to the database for audit.

        Called internally by the receiver when a webhook is handled.

        Args:
            webhook_name: Name of the webhook.
            source_ip: Client IP address.
            authenticated: Whether auth passed.
            status_code: HTTP status returned.
            payload_hash: SHA-256 of the payload.
        """
        if not self._db:
            return

        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        try:
            await self._db.execute(
                """
                INSERT INTO webhook_log
                    (id, agent_id, webhook_name, source_ip, authenticated,
                     status_code, payload_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    self._agent_id,
                    webhook_name,
                    source_ip,
                    1 if authenticated else 0,
                    status_code,
                    payload_hash,
                    now,
                ),
            )
        except Exception as exc:
            logger.warning("WebhookFeature: failed to log webhook event: %s", exc)
