"""One SDK route registration for dispatch, advertised URLs and peer admission."""

import math
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from kestrel_sdk.native_a2a.server.routes.agent_card_routes import (
    create_agent_card_routes,
)
from kestrel_sdk.native_a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from kestrel_sdk.native_a2a.server.routes.rest_routes import create_rest_routes
from kestrel_sdk.native_a2a.types import AgentCard, AgentInterface
from starlette.applications import Starlette
from starlette.datastructures import URLPath
from starlette.routing import BaseRoute, Match, Mount
from starlette.types import Receive, Scope, Send

from kestrel_sovereign.a2a.native_context import (
    NativeRecipientBinding,
    NativeServerCallContextBuilder,
)
from kestrel_sovereign.a2a.request_handler import NativeA2ARequestHandler
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.multi_agent.route_name import encode_agent_route_name
from kestrel_sovereign.native_http import NativeHTTPRegistry


def native_route_prefix(routing_key: str) -> str:
    """Use the manager registration key, never a display name or signing DID."""
    return f"/api/agent-routes/{encode_agent_route_name(routing_key)}/a2a"


def native_card_path(routing_key: str) -> str:
    """Direct per-agent discovery; only the authority root is well-known."""
    return f"/api/agent-routes/{encode_agent_route_name(routing_key)}/agent-card.json"


def native_http_interfaces(
    public_base_url: str, *, routing_key: str, scope: NativeRecipientScope
) -> list[AgentInterface]:
    """Declare both native bindings from an explicitly configured public URL."""
    try:
        parsed = urlsplit(public_base_url)
        _ = parsed.port  # Validate the parser's deferred port checks as well.
    except ValueError as error:
        raise ValueError("Native A2A requires a valid public HTTP base URL") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in public_base_url)
    ):
        raise ValueError("Native A2A requires an explicit public HTTP base URL")
    base = public_base_url.rstrip("/") + native_route_prefix(routing_key)
    return [
        AgentInterface(
            url=base + suffix,
            protocol_binding=protocol,
            protocol_version="1.0",
            tenant=scope.tenant,
        )
        for protocol, suffix in (("JSONRPC", "/rpc"), ("HTTP+JSON", ""))
    ]


class _AdvertisedTenantRoute(BaseRoute):
    """Restrict the SDK's base/tenant aliases without replacing their dispatch."""

    def __init__(self, route: BaseRoute, tenant: str):
        self.route = route
        self.tenant = tenant

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        match, child_scope = self.route.matches(scope)
        if child_scope.get("path_params", {}).get("tenant", "") != self.tenant:
            return Match.NONE, {}
        return match, child_scope

    def url_path_for(self, name: str, /, **path_params: str) -> URLPath:
        return self.route.url_path_for(name, **path_params)

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.route.handle(scope, receive, send)


def _matches_endpoint(route: BaseRoute, scope: Scope) -> bool:
    match, child_scope = route.matches(scope)
    if match is not Match.FULL:
        return False
    target = route.route if isinstance(route, _AdvertisedTenantRoute) else route
    if isinstance(target, Mount):
        return any(
            _matches_endpoint(child, scope | child_scope) for child in target.routes
        )
    return True


class NativeA2AHTTP:
    """Mount both official bindings and expose their exact transport surface.

    ``admits`` is route classification, not principal authorization. Every
    matched request still enters the handler's proof and recipient policy.
    Host authentication must use this registration rather than a second path
    pattern. A mount keeps the SDK's tenant aliases beneath the agent prefix.
    """

    def __init__(
        self,
        handler: NativeA2ARequestHandler,
        *,
        public_base_url: str,
        shutdown_grace_period: float = 0,
        card_modifier: Callable[[AgentCard], Awaitable[AgentCard]] | None = None,
        public_discovery: bool = False,
    ):
        if not math.isfinite(shutdown_grace_period) or shutdown_grace_period < 0:
            raise ValueError(
                "Native stream shutdown grace must be finite and nonnegative"
            )
        guard = handler.guard
        interfaces = native_http_interfaces(
            public_base_url, routing_key=guard.routing_key, scope=guard.scope
        )
        advertised = list(handler.agent_card.supported_interfaces)
        if len(advertised) != len(interfaces) or any(
            advertised.count(item) != 1 for item in interfaces
        ):
            raise ValueError(
                "Native Agent Card interfaces must match its route registration"
            )
        self.prefix = native_route_prefix(guard.routing_key)
        self.routing_key = guard.routing_key

        def builder(protocol: str) -> NativeServerCallContextBuilder:
            return NativeServerCallContextBuilder(
                NativeRecipientBinding(
                    guard.routing_key,
                    guard.scope.recipient_id,
                    guard.scope.tenant,
                    protocol,
                )
            )

        routes = create_jsonrpc_routes(
            handler,
            rpc_url="/rpc",
            context_builder=builder("JSONRPC"),
            enable_v0_3_compat=False,
            shutdown_grace_period=shutdown_grace_period,
        )
        routes.extend(
            _AdvertisedTenantRoute(route, guard.scope.tenant)
            for route in create_rest_routes(
                handler,
                context_builder=builder("HTTP+JSON"),
                enable_v0_3_compat=False,
                shutdown_grace_period=shutdown_grace_period,
            )
        )
        self.route = Mount(self.prefix, routes=routes)
        if public_discovery and card_modifier is None:
            raise ValueError("Public discovery requires a live recipient card gate")
        self.card_routes = []
        if card_modifier is not None:
            paths = [native_card_path(self.routing_key)]
            if public_discovery:
                paths.append("/.well-known/agent-card.json")
            for path in paths:
                self.card_routes.extend(
                    create_agent_card_routes(
                        handler.agent_card,
                        card_modifier=card_modifier,
                        card_url=path,
                        cache_control="no-cache, max-age=0",
                    )
                )
        self.routes = (self.route, *self.card_routes)

    def admits(self, scope: Scope) -> bool:
        """Classify only methods and full endpoints in this SDK registration."""
        return scope["type"] == "http" and _matches_endpoint(self.route, scope)

    def admits_public_card(self, scope: Scope) -> bool:
        return scope["type"] == "http" and any(
            _matches_endpoint(route, scope) for route in self.card_routes
        )


class NativeA2AHTTPRegistry(NativeHTTPRegistry[NativeA2AHTTP]):
    """A2A route publication shares the host's native HTTP ownership contract."""

    def __init__(self, app: Starlette):
        super().__init__(app, protocol="A2A")

    def admits_public_card(self, scope: Scope) -> bool:
        self.require_application(scope.get("app"))
        return any(binding.admits_public_card(scope) for binding in self._current_bindings())
