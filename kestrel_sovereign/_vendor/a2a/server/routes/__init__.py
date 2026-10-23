# Modified by Kestrel: isolated SDK namespace; see VENDORED.json.
"""A2A Routes."""

from kestrel_sovereign._vendor.a2a.server.routes.agent_card_routes import create_agent_card_routes
from kestrel_sovereign._vendor.a2a.server.routes.common import (
    DefaultServerCallContextBuilder,
    ServerCallContextBuilder,
)
from kestrel_sovereign._vendor.a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from kestrel_sovereign._vendor.a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from kestrel_sovereign._vendor.a2a.server.routes.rest_routes import create_rest_routes


__all__ = [
    'DefaultServerCallContextBuilder',
    'ServerCallContextBuilder',
    'add_a2a_routes_to_fastapi',
    'create_agent_card_routes',
    'create_jsonrpc_routes',
    'create_rest_routes',
]
