"""
Kestrel Rookery - Multi-Agent Registry.

The rookery is the registry of agents managed by a single Kestrel Host.
Each agent has its own directory, DID, database, and configuration.
"""

from .config import RookeryConfig, HostConfig, LocalAgentConfig, RemoteAgentConfig
from .process_manager import ProcessManager, AgentProcess
from .proxy import (
    proxy_request_streaming,
    resolve_agent,
    get_agent_base_url,
    build_proxy_headers,
)

__all__ = [
    "RookeryConfig",
    "HostConfig",
    "LocalAgentConfig",
    "RemoteAgentConfig",
    "ProcessManager",
    "AgentProcess",
    "proxy_request_streaming",
    "resolve_agent",
    "get_agent_base_url",
    "build_proxy_headers",
]
