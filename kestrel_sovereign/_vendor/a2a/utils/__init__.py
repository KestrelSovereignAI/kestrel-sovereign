# Modified by Kestrel: isolated SDK namespace; see VENDORED.json.
"""Utility functions for the A2A Python SDK."""

from kestrel_sovereign._vendor.a2a.utils import proto_utils
from kestrel_sovereign._vendor.a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    DEFAULT_RPC_URL,
    TransportProtocol,
)
from kestrel_sovereign._vendor.a2a.utils.proto_utils import to_stream_response


__all__ = [
    'AGENT_CARD_WELL_KNOWN_PATH',
    'DEFAULT_RPC_URL',
    'TransportProtocol',
    'proto_utils',
    'to_stream_response',
]
