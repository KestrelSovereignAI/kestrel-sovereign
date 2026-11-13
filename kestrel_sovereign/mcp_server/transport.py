"""Official Streamable HTTP transport and standard per-resource OAuth discovery."""

from contextlib import asynccontextmanager
from typing import get_args
from urllib.parse import urlsplit

from mcp.server import MCPServer
from mcp.server.auth.handlers.metadata import MetadataHandler
from mcp.server.auth.routes import (
    build_resource_metadata_url,
    cors_middleware,
    validate_issuer_url,
)
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import (
    RequestBodyLimitMiddleware,
    TransportSecuritySettings,
)
from mcp.shared.auth import OAuthMetadata
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Match, Route, request_response
from starlette.types import Scope

from kestrel_sovereign.multi_agent.route_name import encode_agent_route_name
from kestrel_sovereign.native_http import NativeHTTPRegistry

from .grants import MCPAuthorizationGrants
from .oauth import NativeMCPAuthorizationServer
from .policy import MemoryScope


class NativeMCPServer:
    """One agent's native MCP runtime; the host owns its complete lifespan."""

    def __init__(self, grants: MCPAuthorizationGrants, *, name: str, consent_url: str):
        self.authorization = NativeMCPAuthorizationServer(grants, consent_url)
        resource = grants.pending.resource_url
        issuer = grants.issuer
        settings = AuthSettings(issuer_url=issuer, resource_server_url=resource,
            validate_token_resource=True, required_scopes=[])
        if str(settings.issuer_url) != issuer or str(settings.resource_server_url) != resource:
            raise ValueError("Native OAuth metadata must preserve the exact issuer and resource")
        validate_issuer_url(settings.issuer_url)
        parsed_resource, parsed_issuer = urlsplit(resource), urlsplit(issuer)
        origin = grants.policy.console_origin
        if any(value.scheme + "://" + value.netloc != origin for value in (parsed_resource, parsed_issuer)):
            raise ValueError("Native MCP endpoints must use the configured first-party origin")
        if not parsed_resource.path or not parsed_issuer.path:
            raise ValueError("Native MCP resource and issuer require distinct explicit paths")
        resource_metadata_path = urlsplit(str(build_resource_metadata_url(settings.resource_server_url))).path
        issuer_metadata_path = "/.well-known/oauth-authorization-server" + parsed_issuer.path
        self.server = MCPServer(name, token_verifier=self.authorization, auth=settings, subscriptions=False)
        self._transport = self.server.streamable_http_app(streamable_http_path=parsed_resource.path,
            json_response=True, stateless_http=True, max_request_body_size=grants.policy.max_request_bytes,
            transport_security=TransportSecuritySettings(allowed_hosts=[parsed_resource.netloc],
                allowed_origins=[origin]))
        methods = sorted({client.token_endpoint_auth_method for client in grants.policy.clients})
        metadata = OAuthMetadata(issuer=settings.issuer_url,
            authorization_endpoint=AnyHttpUrl(issuer + "/authorize"),
            token_endpoint=AnyHttpUrl(issuer + "/token"),
            revocation_endpoint=AnyHttpUrl(issuer + "/revoke"),
            response_types_supported=["code"], grant_types_supported=["authorization_code", "refresh_token"],
            scopes_supported=list(get_args(MemoryScope)), code_challenge_methods_supported=["S256"],
            token_endpoint_auth_methods_supported=methods, revocation_endpoint_auth_methods_supported=methods,
            authorization_response_iss_parameter_supported=True)
        metadata_handler = MetadataHandler(metadata)
        routes = [Route(issuer_metadata_path,
            cors_middleware(metadata_handler.handle, ["GET", "OPTIONS"]), methods=["GET", "OPTIONS"])]
        for suffix, handler, allowed in (
            ("/authorize", self.authorization.authorization_endpoint, ["GET", "POST", "OPTIONS"]),
            ("/token", self.authorization.token_endpoint, ["POST", "OPTIONS"]),
            ("/revoke", self.authorization.revocation_endpoint, ["POST", "OPTIONS"]),
        ):
            endpoint = RequestBodyLimitMiddleware(request_response(handler), grants.policy.max_request_bytes)
            endpoint = CORSMiddleware(endpoint, allow_origins=["*"], allow_methods=allowed,
                allow_headers=["Authorization", "Content-Type"])
            routes.append(Route(parsed_issuer.path + suffix, endpoint, methods=allowed))
        # Keep the official app, its auth middleware and original full request
        # path intact. Exact outer routes do not consume another agent's paths.
        routes.extend([
            Route(parsed_resource.path, self._transport, methods=["GET", "POST", "DELETE"]),
            Route(resource_metadata_path, self._transport, methods=["GET", "OPTIONS"]),
        ])
        paths = [route.path for route in routes]
        if len(paths) != len(set(paths)):
            raise ValueError("Native MCP endpoints must have distinct routes")
        self.app = Starlette(routes=routes, lifespan=self._lifespan)

    @asynccontextmanager
    async def _lifespan(self, _app: Starlette):
        self.authorization.grants.consent_authority.require_current_host()
        async with self._transport.router.lifespan_context(self._transport):
            yield


class NativeMCPHTTP:
    """Publish and classify the runtime's exact SDK and OAuth route objects."""

    def __init__(self, routing_key: str, runtime: NativeMCPServer):
        encode_agent_route_name(routing_key)
        self.routing_key, self.runtime = routing_key, runtime
        self.routes = tuple(runtime.app.routes)

    def admits(self, scope: Scope) -> bool:
        return any(route.matches(scope)[0] is Match.FULL for route in self.routes)


class NativeMCPHTTPRegistry(NativeHTTPRegistry[NativeMCPHTTP]):
    """The MCP host retains one published generation per manager routing key."""

    def __init__(self, app: Starlette):
        super().__init__(app, protocol="MCP")
