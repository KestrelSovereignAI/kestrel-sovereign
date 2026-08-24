"""
Kestrel MultiAgent - registry of agents managed by a single Kestrel Host.

Each agent has its own directory, DID, database, and configuration.
The host process loads ``multi_agent.toml``, starts each local agent
as a subprocess, and proxies API requests to the right port.

Public names resolve **lazily** (PEP 562). Importing a lightweight submodule
— notably ``process_manager`` for its cross-platform port/process helpers —
must not force-load :class:`AgentManager`, which pulls in the whole agent
cognition + LLM stack. Custody code (``phoenix_supervisor._port_listener_pids``)
and the process-supervision utilities need only the port lookup; eager package
imports made that lookup transitively depend on the entire agent import graph
resolving cleanly (a leaked-Phoenix reap could fail on an unrelated SDK skew).
See #2690.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# name -> submodule that defines it. Attribute access imports the submodule on
# demand and caches the resolved object back into module globals.
_LAZY_EXPORTS = {
    "AgentManager": ".agent_manager",
    "HostedIsolatedRuntimeLifecyclePolicy": ".agent_manager",
    "MultiAgentConfig": ".config",
    "HostConfig": ".config",
    "LocalAgentConfig": ".config",
    "RemoteAgentConfig": ".config",
    "ProcessManager": ".process_manager",
    "AgentProcess": ".process_manager",
    "proxy_request_streaming": ".proxy",
    "resolve_agent": ".proxy",
    "get_agent_base_url": ".proxy",
    "build_proxy_headers": ".proxy",
}


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


if TYPE_CHECKING:  # eager names for type checkers / IDEs only — no runtime cost
    from .agent_manager import AgentManager, HostedIsolatedRuntimeLifecyclePolicy
    from .config import (
        HostConfig,
        LocalAgentConfig,
        MultiAgentConfig,
        RemoteAgentConfig,
    )
    from .process_manager import AgentProcess, ProcessManager
    from .proxy import (
        build_proxy_headers,
        get_agent_base_url,
        proxy_request_streaming,
        resolve_agent,
    )

__all__ = [
    "AgentManager",
    "HostedIsolatedRuntimeLifecyclePolicy",
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
