"""
Compatibility shim — MCP feature has been extracted to kestrel-feature-mcp.

Install: pip install kestrel-feature-mcp
"""

try:
    from kestrel_feature_mcp.gateway import (  # noqa: F401
        DockerMCPGateway,
        DockerMCPGatewayError,
        DockerMCPNotInstalledError,
        GATEWAY_STARTUP_TIMEOUT,
        GATEWAY_STARTUP_POLL_INTERVAL,
        list_available_servers,
        list_enabled_servers,
    )
except ImportError:
    raise ImportError(
        "kestrel-feature-mcp is not installed. "
        "Install it with: pip install kestrel-feature-mcp"
    )
