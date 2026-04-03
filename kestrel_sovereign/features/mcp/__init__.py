"""
Compatibility shim — MCP feature has been extracted to kestrel-feature-mcp.

Install: pip install kestrel-feature-mcp

All imports are re-exported from the extracted package so existing code
continues to work without changes.
"""

try:
    from kestrel_feature_mcp.feature import MCPAgent  # noqa: F401
except ImportError:
    raise ImportError(
        "kestrel-feature-mcp is not installed. "
        "Install it with: pip install kestrel-feature-mcp"
    )
