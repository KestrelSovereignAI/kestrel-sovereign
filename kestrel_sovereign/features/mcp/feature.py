"""
Compatibility shim — MCP feature has been extracted to kestrel-feature-mcp.

Install: pip install kestrel-feature-mcp
"""

try:
    from kestrel_feature_mcp.feature import MCPAgent  # noqa: F401
except ImportError:
    raise ImportError(
        "kestrel-feature-mcp is not installed. "
        "Install it with: pip install kestrel-feature-mcp"
    )
