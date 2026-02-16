"""
Kestrel Rookery - Multi-Agent Registry.

The rookery is the registry of agents managed by a single Kestrel Host.
Each agent has its own directory, DID, database, and configuration.
"""

from .config import RookeryConfig, HostConfig, LocalAgentConfig, RemoteAgentConfig

__all__ = ["RookeryConfig", "HostConfig", "LocalAgentConfig", "RemoteAgentConfig"]
