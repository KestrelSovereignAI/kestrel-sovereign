"""
Kestrel MultiAgent - registry of agents managed by a single Kestrel Host.

Each agent has its own directory, DID, database, and configuration.
The host process loads ``multi_agent.toml``, starts each local agent
as a subprocess, and proxies API requests to the right port.
"""

from .agent_manager import AgentManager
from .config import MultiAgentConfig, HostConfig, LocalAgentConfig, RemoteAgentConfig
from .process_manager import ProcessManager, AgentProcess
from .proxy import (
    proxy_request_streaming,
    resolve_agent,
    get_agent_base_url,
    build_proxy_headers,
)

__all__ = [
    "AgentManager",
    "MultiAgentConfig",
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
