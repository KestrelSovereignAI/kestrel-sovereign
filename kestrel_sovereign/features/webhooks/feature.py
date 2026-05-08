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
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

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

        self._db = resolve_feature_database(self.agent)

        # Agent identity (DID is the canonical source of truth)
        self._agent_id = self.agent.did

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
    async def webhooks_list(self) -> ToolResult:
        """List all registered webhooks for this agent.

        Returns:
            ToolResult.ok with a list of webhooks + count. PARTIAL
            when the receiver is empty but the DB query failed — the
            empty list there could otherwise hide persisted webhooks
            that just weren't readable.
        """
        webhooks = self.receiver.list_webhooks()
        db_failed = False

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
                db_failed = True

        data = {"webhooks": webhooks, "count": len(webhooks)}
        confirmation = (
            "No webhooks registered."
            if not webhooks
            else f"{len(webhooks)} webhook(s) registered: "
            + ", ".join(w.get("name", "?") for w in webhooks)
            + "."
        )
        if db_failed:
            return ToolResult.partial(
                confirmation,
                (
                    "DB query for webhook_config failed; the receiver list "
                    "is empty and persisted webhooks could not be loaded — "
                    "an empty result here does NOT mean none are configured."
                ),
                data=data,
            )
        return ToolResult.ok(confirmation, data=data)

    @tool(
        "webhooks_history",
        "Show recent webhook receive log for security audit",
        category=ToolCategory.SYSTEM,
        command_prefix="!webhooks history",
    )
    async def webhooks_history(self, limit: int = 20) -> ToolResult:
        """Show recent webhook events.

        Args:
            limit: Maximum number of events to return (default: 20)

        Returns:
            ToolResult.ok with the event list. PARTIAL when the DB
            query failed (history could be missing persisted events
            and we fell back to the in-memory ring buffer) or when no
            DB is attached at all (events are NOT being persisted, so
            an empty list does not mean "no inbound webhooks").
        """
        # Try database first
        events: list = []
        db_failed = False
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
                db_failed = True

        # Fallback to in-memory log
        if not events:
            events = self.receiver.get_recent_events(limit)

        data = {"events": events, "count": len(events)}
        confirmation = f"Returned {len(events)} webhook event(s)."
        if db_failed:
            return ToolResult.partial(
                confirmation,
                (
                    "DB query for webhook_log failed; results are from the "
                    "in-memory ring buffer only and may be missing persisted "
                    "audit events."
                ),
                data=data,
            )
        if not self._db:
            return ToolResult.partial(
                confirmation,
                (
                    "no database is attached — webhook events are not being "
                    "persisted; this list is the in-memory ring buffer only."
                ),
                data=data,
            )
        return ToolResult.ok(confirmation, data=data)

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
    ) -> ToolResult:
        """Register a new webhook endpoint.

        Args:
            name: Unique webhook name (used in the URL path /webhooks/{name})
            auth_type: Authentication method (none, bearer_token, hmac_sha256, ip_allowlist)
            event_type: Optional event type label for categorisation
            auth_config_json: JSON config for auth (e.g. {"token":"secret"} for bearer_token)
            rate_limit: Maximum requests per minute (default: 60, 0 = unlimited)

        Returns:
            ToolResult.ok on a clean register; PARTIAL when the
            in-memory register succeeded but the DB persistence row
            failed (the webhook will work right now but won't survive
            a restart); ERROR for any validation or duplicate-name
            failure.
        """
        # Validate auth_type
        try:
            auth_type_enum = WebhookAuthType(auth_type)
        except ValueError:
            valid = [t.value for t in WebhookAuthType]
            return ToolResult.failed(
                error=f"Invalid auth_type '{auth_type}'. Must be one of: {valid}"
            )

        # Parse auth config
        try:
            auth_config = json.loads(auth_config_json)
            if not isinstance(auth_config, dict):
                return ToolResult.failed(error="auth_config_json must be a JSON object")
        except json.JSONDecodeError as exc:
            return ToolResult.failed(error=f"Invalid auth_config_json: {exc}")

        # Validate name
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            return ToolResult.failed(
                error="Webhook name must be alphanumeric (hyphens and underscores allowed)"
            )

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
            return ToolResult.failed(error=str(exc))

        # Persist to database
        persist_failed = False
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
                persist_failed = True

        logger.info(
            "Registered webhook: %s (auth=%s, rate_limit=%d/min)",
            name, auth_type, rate_limit,
        )

        data = {
            "webhook_id": webhook_id,
            "name": name,
            "auth_type": auth_type,
            "event_type": event_type,
            "rate_limit": max(0, rate_limit),
            "endpoint": f"/webhooks/{name}",
            "created_at": now,
            "persisted": (self._db is not None and not persist_failed),
        }
        confirmation = (
            f"Registered webhook '{name}' (auth={auth_type}, "
            f"rate_limit={max(0, rate_limit)}/min, endpoint=/webhooks/{name})."
        )
        if persist_failed:
            return ToolResult.partial(
                confirmation,
                (
                    "in-memory registration succeeded but the DB INSERT for "
                    "webhook_config failed — this webhook will accept "
                    "requests now but will NOT survive a restart."
                ),
                data=data,
            )
        return ToolResult.ok(confirmation, data=data)

    @tool(
        "webhooks_remove",
        "Remove a registered webhook endpoint",
        category=ToolCategory.SYSTEM,
        command_prefix="!webhooks remove",
    )
    async def webhooks_remove(self, name: str) -> ToolResult:
        """Remove a registered webhook.

        Args:
            name: The name of the webhook to remove

        Returns:
            ToolResult.ok on a clean remove (in-memory + persisted row).
            PARTIAL when the receiver forgot it but the DB DELETE failed,
            OR when the receiver had no record but a persisted-only row
            was cleaned up (the latter can happen after a prior partial
            failure or when a persisted webhook didn't load on startup —
            retrying ``webhooks_remove`` must still scrub the row, so
            we never short-circuit on the receiver miss alone).
            ERROR only when neither the receiver nor the database
            has any trace of the webhook.
        """
        # Remove from receiver (in-memory)
        removed_from_memory = self.receiver.unregister_webhook(name)

        # Always attempt the DB DELETE, even if the receiver didn't
        # have it. A prior call could have left a stale row behind
        # (transient DB error → PARTIAL), or a persisted webhook may
        # have failed to load into the receiver on startup. If we
        # short-circuit on the receiver miss, that row gets
        # resurrected on the next restart — codex caught this on
        # round 1.
        deleted_persisted_row = False
        delete_failed = False
        if self._db:
            try:
                # Probe first so we can tell "found-and-deleted" from
                # "no-row-existed" without depending on cursor.rowcount,
                # which is not uniformly exposed by the async DB
                # adapter. Two queries instead of one is cheap; correctness
                # is not.
                existing = await self._db.fetchone(
                    "SELECT id FROM webhook_config WHERE name = ? AND agent_id = ?",
                    (name, self._agent_id),
                )
                if existing is not None:
                    await self._db.execute(
                        "DELETE FROM webhook_config WHERE name = ? AND agent_id = ?",
                        (name, self._agent_id),
                    )
                    deleted_persisted_row = True
            except Exception as exc:
                logger.warning("WebhookFeature: failed to delete from DB: %s", exc)
                delete_failed = True

        if not removed_from_memory and not deleted_persisted_row and not delete_failed:
            return ToolResult.failed(error=f"Webhook '{name}' not found")

        data = {
            "name": name,
            "status": "removed",
            "removed_from_memory": removed_from_memory,
            "deleted_from_db": deleted_persisted_row,
        }
        confirmation = f"Webhook '{name}' removed."

        if delete_failed:
            return ToolResult.partial(
                confirmation,
                (
                    "the receiver forgot this webhook but the DB DELETE for "
                    "webhook_config failed — a restart could resurrect it "
                    "from the persisted row."
                ),
                data=data,
            )
        if not removed_from_memory and deleted_persisted_row:
            # Persisted-only cleanup — the receiver never had this
            # webhook in this process. Surface that so the LLM doesn't
            # claim it was actively serving requests right before the
            # remove.
            return ToolResult.partial(
                confirmation,
                (
                    f"webhook '{name}' was not loaded in the in-memory "
                    "receiver; cleaned up a stale row from webhook_config."
                ),
                data=data,
            )
        return ToolResult.ok(confirmation, data=data)

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
