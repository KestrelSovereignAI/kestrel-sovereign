"""
WebhookReceiver -- manages registered webhook endpoints and processes requests.

Responsibilities:
- Register / unregister named webhook endpoints
- Authenticate incoming requests using the configured auth handler
- Log all webhook receives (success and failure) for security audit
- Provide a FastAPI APIRouter with all webhook routes
- Per-webhook rate limiting
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .auth import WebhookAuth, create_auth_handler
from .models import WebhookAuthType, WebhookConfig, WebhookEvent

logger = logging.getLogger(__name__)


class WebhookReceiver:
    """Core webhook management engine.

    Holds the set of registered webhooks in-memory and delegates persistence
    to the owning Feature (which has database access).

    Attributes:
        webhooks: Mapping of webhook name -> WebhookConfig.
        auth_handlers: Mapping of webhook name -> WebhookAuth instance.
        event_log: In-memory ring buffer of recent WebhookEvent records.
    """

    # Maximum events to keep in-memory (oldest are evicted).
    MAX_IN_MEMORY_LOG = 200

    def __init__(self) -> None:
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.auth_handlers: Dict[str, WebhookAuth] = {}
        self.event_log: deque[WebhookEvent] = deque(maxlen=self.MAX_IN_MEMORY_LOG)

        # Per-webhook sliding-window rate limiter.
        # Maps webhook name -> deque of Unix timestamps of recent requests.
        self._rate_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=600))

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_webhook(self, config: WebhookConfig) -> None:
        """Register a new webhook endpoint.

        Args:
            config: Full webhook configuration.

        Raises:
            ValueError: If a webhook with the same name is already registered.
        """
        if config.name in self.webhooks:
            raise ValueError(f"Webhook '{config.name}' is already registered")

        # Build auth handler
        auth_handler = create_auth_handler(config.auth_type.value, config.auth_config)
        self.webhooks[config.name] = config
        self.auth_handlers[config.name] = auth_handler
        logger.info("Registered webhook: %s (auth=%s)", config.name, config.auth_type.value)

    def unregister_webhook(self, name: str) -> bool:
        """Remove a registered webhook.

        Args:
            name: Webhook name.

        Returns:
            True if the webhook was found and removed, False otherwise.
        """
        if name not in self.webhooks:
            return False

        del self.webhooks[name]
        self.auth_handlers.pop(name, None)
        self._rate_windows.pop(name, None)
        logger.info("Unregistered webhook: %s", name)
        return True

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    async def handle_webhook(
        self,
        name: str,
        *,
        headers: Dict[str, str],
        body: bytes,
        source_ip: str,
    ) -> Dict[str, Any]:
        """Authenticate and process an incoming webhook request.

        This method always creates a log entry regardless of outcome.

        Args:
            name: Registered webhook name.
            headers: Request headers (keys should be lower-cased).
            body: Raw request body.
            source_ip: Client IP address.

        Returns:
            Dict with ``status_code`` (int) and ``body`` (dict) for the HTTP response.
        """
        # --- Unknown webhook ---
        if name not in self.webhooks:
            event = self._create_event(
                webhook_name=name,
                source_ip=source_ip,
                authenticated=False,
                status_code=404,
                body=body,
            )
            self.event_log.append(event)
            return {"status_code": 404, "body": {"error": f"Unknown webhook: {name}"}}

        config = self.webhooks[name]

        # --- Disabled ---
        if not config.enabled:
            event = self._create_event(
                webhook_name=name,
                source_ip=source_ip,
                authenticated=False,
                status_code=503,
                body=body,
            )
            self.event_log.append(event)
            return {"status_code": 503, "body": {"error": "Webhook is disabled"}}

        # --- Rate limiting ---
        if config.rate_limit > 0 and self._is_rate_limited(name, config.rate_limit):
            event = self._create_event(
                webhook_name=name,
                source_ip=source_ip,
                authenticated=False,
                status_code=429,
                body=body,
            )
            self.event_log.append(event)
            return {"status_code": 429, "body": {"error": "Rate limit exceeded"}}

        # --- Authentication ---
        auth_handler = self.auth_handlers.get(name)
        authenticated = False
        if auth_handler is not None:
            try:
                authenticated = auth_handler.validate(
                    headers=headers,
                    body=body,
                    source_ip=source_ip,
                )
            except Exception as exc:
                logger.error("Auth handler error for webhook '%s': %s", name, exc)
                authenticated = False

        if not authenticated:
            event = self._create_event(
                webhook_name=name,
                source_ip=source_ip,
                authenticated=False,
                status_code=401,
                body=body,
            )
            self.event_log.append(event)
            return {"status_code": 401, "body": {"error": "Authentication failed"}}

        # --- Success ---
        event = self._create_event(
            webhook_name=name,
            source_ip=source_ip,
            authenticated=True,
            status_code=200,
            body=body,
        )
        self.event_log.append(event)

        logger.info(
            "Webhook received: %s from %s (authenticated, %d bytes)",
            name,
            source_ip,
            len(body),
        )

        return {
            "status_code": 200,
            "body": {
                "status": "received",
                "event_id": event.id,
                "webhook": name,
            },
        }

    # ------------------------------------------------------------------
    # FastAPI router
    # ------------------------------------------------------------------

    def get_router(self):
        """Return a FastAPI APIRouter with a catch-all webhook endpoint.

        The router mounts at ``/webhooks/{name}`` and delegates to
        ``handle_webhook``. It does NOT require the server-level API key
        authentication -- webhook auth is handled per-endpoint by the
        registered auth handler.

        Returns:
            A ``fastapi.APIRouter`` instance.
        """
        from fastapi import APIRouter, Request
        from fastapi.responses import JSONResponse

        router = APIRouter(tags=["webhooks"])

        @router.post("/webhooks/{webhook_name}")
        async def webhook_endpoint(webhook_name: str, request: Request):
            body = await request.body()
            # Normalise header keys to lower-case for consistent lookup.
            headers = {k.lower(): v for k, v in request.headers.items()}
            source_ip = request.client.host if request.client else "unknown"

            result = await self.handle_webhook(
                webhook_name,
                headers=headers,
                body=body,
                source_ip=source_ip,
            )

            return JSONResponse(
                content=result["body"],
                status_code=result["status_code"],
            )

        return router

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_webhooks(self) -> List[Dict[str, Any]]:
        """Return a list of registered webhook configs (sans secrets)."""
        return [cfg.to_dict() for cfg in self.webhooks.values()]

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent webhook events from the in-memory log.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of event dicts, newest first.
        """
        events = list(self.event_log)
        events.reverse()
        return [e.to_dict() for e in events[:limit]]

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _is_rate_limited(self, name: str, max_per_minute: int) -> bool:
        """Check whether the webhook has exceeded its per-minute rate limit.

        Uses a sliding window of request timestamps.
        """
        now = time.monotonic()
        window = self._rate_windows[name]

        # Purge entries older than 60 seconds.
        while window and (now - window[0]) > 60:
            window.popleft()

        if len(window) >= max_per_minute:
            return True

        window.append(now)
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_event(
        *,
        webhook_name: str,
        source_ip: str,
        authenticated: bool,
        status_code: int,
        body: bytes,
    ) -> WebhookEvent:
        """Build a ``WebhookEvent`` from raw request data."""
        payload_hash = hashlib.sha256(body).hexdigest() if body else ""
        return WebhookEvent(
            id=str(uuid.uuid4()),
            webhook_name=webhook_name,
            source_ip=source_ip,
            authenticated=authenticated,
            status_code=status_code,
            payload_hash=payload_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
