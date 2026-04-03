"""
Compatibility shim — MCP feature has been extracted to kestrel-feature-mcp.

Install: pip install kestrel-feature-mcp
"""

try:
    from kestrel_feature_mcp.manager import (  # noqa: F401
        MCPToolManager,
        MCPGatewayManager,
        HEALTHCHECK_INITIAL_DELAY,
        HEALTHCHECK_MAX_DELAY,
        HEALTHCHECK_TIMEOUT,
        HEALTHCHECK_BACKOFF_FACTOR,
    )
except ImportError:
    raise ImportError(
        "kestrel-feature-mcp is not installed. "
        "Install it with: pip install kestrel-feature-mcp"
    )
