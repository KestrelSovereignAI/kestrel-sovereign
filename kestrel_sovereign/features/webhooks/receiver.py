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
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from .auth import WebhookAuth, create_auth_handler
from .models import WebhookConfig, WebhookEvent

logger = logging.getLogger(__name__)


def unknown_webhook_result(name: str) -> Dict[str, Any]:
    """The one public response for a name this host will not dispatch.

    Returned for an unregistered name AND for a name whose ownership is
    ambiguous (#3216), so the two cases are indistinguishable to a keyless
    caller. Every site that answers "not dispatching" builds its response
    here; a divergent body would hand out an ownership oracle.
    """
    return {"status_code": 404, "body": {"error": f"Unknown webhook: {name}"}}


def build_webhook_dispatch_router(
    receiver_provider: "Callable[[Optional[Any]], Iterable['WebhookReceiver']]",
):
    """Build a single ``POST /webhooks/{name}`` router that dispatches across
    many receivers.

    ``receiver_provider(agent)`` returns the current iterable of
    :class:`WebhookReceiver` instances to consider for THIS request. The
    ``agent`` argument is the request-scoped target agent
    (``request.state.agent``) that the multi-agent routing middleware attached
    for an agent-prefixed ``/api/agents/{name}/webhooks/{name}`` request, or
    ``None`` for the unprefixed ``/webhooks/{name}`` form (and single-agent
    mode). The provider is responsible for scoping accordingly:

    * agent-prefixed → ONLY that agent's enabled receivers, so a request
      addressed to agent *A* can never dispatch to agent *B*'s webhook even
      when *A*'s owning feature is disabled/removed and *B* registers the same
      name (kestrel-sovereign#2522);
    * unprefixed → the aggregate of every current agent's enabled receivers.

    Each request resolves ``{name}`` against the in-scope receivers:

    * exactly one receiver owns the name → dispatch to it;
    * more than one owns it → the target is ambiguous and the request is
      REFUSED without dispatching to any of them (kestrel-sovereign#3216).
      Iteration order never selects the target. The public response is the
      same ``404 Unknown webhook`` an unregistered name gets, so a keyless
      caller cannot probe which names are registered on more than one
      agent; each owning receiver audits the refusal as a 404 and the host
      log names the collision. The agent-prefixed form stays unambiguous
      because its scope is a single agent;
    * none owns it → recorded on the first in-scope receiver (so the 404 is
      still audited) or a bare 404 when no in-scope receiver exists.

    This replaces mounting each agent's own ``/webhooks/{name}`` catch-all: two
    identical routes would otherwise shadow each other (first-mounted wins),
    silently returning ``404 Unknown webhook`` for every webhook owned by a
    later-mounted agent (issue #2089 follow-up).
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

        # Scope resolution to the request's target agent (set by the
        # multi-agent routing middleware for an agent-prefixed request); the
        # provider narrows to that agent's enabled receivers, or aggregates
        # across all agents for the unprefixed form (#2522).
        target_agent = getattr(request.state, "agent", None)
        receivers = list(receiver_provider(target_agent))
        owners = [r for r in receivers if webhook_name in r.webhooks]
        if len(owners) == 1:
            result = await owners[0].handle_webhook(
                webhook_name, headers=headers, body=body, source_ip=source_ip
            )
        elif owners:
            # More than one in-scope receiver owns this name, so no rule but
            # iteration order could pick the target. Refuse without
            # dispatching to any owner (#3216): the public response is the
            # unknown-webhook 404 (no ownership oracle), every owner audits
            # the refusal, and the host log carries the collision so an
            # operator can re-address the sender to the agent-prefixed form.
            result = unknown_webhook_result(webhook_name)
            logger.warning(
                "Webhook '%s' is owned by %d enabled receivers; refusing the "
                "unscoped request from %s without dispatch. Address it as "
                "/api/agents/{agent}/webhooks/%s instead.",
                webhook_name,
                len(owners),
                source_ip,
                webhook_name,
            )
            for owner in owners:
                await owner.record_refusal(
                    webhook_name,
                    source_ip=source_ip,
                    body=body,
                    status_code=result["status_code"],
                )
        elif receivers:
            # No receiver owns this name. Route to the first receiver so the
            # unknown-webhook 404 is still audited.
            result = await receivers[0].handle_webhook(
                webhook_name, headers=headers, body=body, source_ip=source_ip
            )
        else:
            # No receivers at all: bare 404.
            result = unknown_webhook_result(webhook_name)

        return JSONResponse(content=result["body"], status_code=result["status_code"])

    return router


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

    def __init__(
        self,
        on_event: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> None:
        # Optional async callback invoked for every received event so the
        # owning Feature can persist it to the audit table. Every recording
        # path funnels through ``_record_event``, so no code path can append
        # to the ring buffer without also invoking this callback.
        self._on_event = on_event
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
            await self._record_event(event)
            return unknown_webhook_result(name)

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
            await self._record_event(event)
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
            await self._record_event(event)
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
            await self._record_event(event)
            return {"status_code": 401, "body": {"error": "Authentication failed"}}

        # --- Success ---
        event = self._create_event(
            webhook_name=name,
            source_ip=source_ip,
            authenticated=True,
            status_code=200,
            body=body,
        )
        await self._record_event(event)

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

    async def record_refusal(
        self,
        name: str,
        *,
        source_ip: str,
        body: bytes,
        status_code: int,
    ) -> None:
        """Audit a request for a webhook this receiver owns that the host
        refused to dispatch.

        Used by the shared dispatch router when ``name`` is owned by more than
        one in-scope receiver (#3216): the request is answered with
        ``status_code`` and NOT dispatched, but the receive still happened at
        this receiver's name and belongs in its audit log (source, payload
        hash, outcome) like every other failed receive. Runs no auth handler
        and touches no rate-limit window — nothing here is a dispatch.
        """
        event = self._create_event(
            webhook_name=name,
            source_ip=source_ip,
            authenticated=False,
            status_code=status_code,
            body=body,
        )
        await self._record_event(event)

    # ------------------------------------------------------------------
    # FastAPI router
    # ------------------------------------------------------------------

    def get_router(self):
        """Return a FastAPI APIRouter with a catch-all webhook endpoint.

        The router mounts at ``/webhooks/{name}`` and delegates to
        ``handle_webhook``. It does NOT require the server-level API key
        authentication -- webhook auth is handled per-endpoint by the
        registered auth handler.

        This is the single-agent router: dispatch resolves against this one
        receiver. In multi-agent processes the server mounts a shared
        :func:`build_webhook_dispatch_router` across every agent's receiver
        instead, so their identical ``/webhooks/{name}`` paths don't shadow
        one another.

        Returns:
            A ``fastapi.APIRouter`` instance.
        """
        # Single-agent mode: this one receiver serves every request regardless
        # of the (always ``None`` here) request-scoped agent argument.
        return build_webhook_dispatch_router(lambda _agent=None: (self,))

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

    async def _record_event(self, event: WebhookEvent) -> None:
        """Record a webhook event to the ring buffer and the audit sink.

        Single funnel for every recorded event: appends to the in-memory
        ring buffer AND invokes the ``on_event`` callback (if any) so the
        owning Feature can persist the event to ``webhook_log``. Every
        recording site in ``handle_webhook`` routes through here, so no
        outcome (404 / 503 / 429 / 401 / 200) can skip persistence.
        """
        self.event_log.append(event)
        if self._on_event is None:
            return
        try:
            await self._on_event(
                webhook_name=event.webhook_name,
                source_ip=event.source_ip,
                authenticated=event.authenticated,
                status_code=event.status_code,
                payload_hash=event.payload_hash,
            )
        except Exception as exc:  # never let persistence failure break the response
            logger.warning("WebhookReceiver: on_event callback failed: %s", exc)

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
