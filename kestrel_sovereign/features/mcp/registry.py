"""
Compatibility shim — MCP feature has been extracted to kestrel-feature-mcp.

Install: pip install kestrel-feature-mcp
"""

try:
    from kestrel_feature_mcp.registry import (  # noqa: F401
        ServerType,
        TransportType,
        ServerCategory,
        MCPServerEntry,
        MCPRegistry,
        get_registry,
        check_docker_mcp_available,
        list_docker_catalog_servers,
        list_enabled_docker_servers,
        search_docker_catalog,
        format_docker_catalog_summary,
    )
except ImportError:
    raise ImportError(
        "kestrel-feature-mcp is not installed. "
        "Install it with: pip install kestrel-feature-mcp"
    )
