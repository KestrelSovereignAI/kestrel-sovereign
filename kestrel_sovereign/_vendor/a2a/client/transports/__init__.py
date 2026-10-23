# Modified by Kestrel: isolated SDK namespace; see VENDORED.json.
"""A2A Client Transports."""

from kestrel_sovereign._vendor.a2a.client.transports.base import ClientTransport
from kestrel_sovereign._vendor.a2a.client.transports.jsonrpc import JsonRpcTransport
from kestrel_sovereign._vendor.a2a.client.transports.rest import RestTransport


try:
    from kestrel_sovereign._vendor.a2a.client.transports.grpc import GrpcTransport
except ImportError:
    GrpcTransport = None  # type: ignore


__all__ = [
    'ClientTransport',
    'GrpcTransport',
    'JsonRpcTransport',
    'RestTransport',
]
