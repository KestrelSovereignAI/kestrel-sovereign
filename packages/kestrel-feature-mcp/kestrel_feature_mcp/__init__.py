"""
Kestrel Feature: MCP (Model Context Protocol)

MCP server management and tool bridging. Provides gateway mode
for 311+ Docker MCP servers and direct container management.

Install: pip install kestrel-feature-mcp
"""

from .feature import MCPAgent
from .registry import (
    MCPRegistry,
    MCPServerEntry,
    ServerType,
    TransportType,
    ServerCategory,
    get_registry,
    check_docker_mcp_available,
)

__all__ = [
    "MCPAgent",
    "MCPRegistry",
    "MCPServerEntry",
    "ServerType",
    "TransportType",
    "ServerCategory",
    "get_registry",
    "check_docker_mcp_available",
]
