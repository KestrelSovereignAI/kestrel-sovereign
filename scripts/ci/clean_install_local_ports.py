"""Give a disposable local clean-install agent isolated loopback ports.

The shared CI matrix starts on fresh runners, but a developer may already
have Kestrel (or another service) listening on the default host/agent ports.
This helper runs only after quickstart has written a *new* multi_agent.toml.
"""

from __future__ import annotations

import socket
from pathlib import Path

from kestrel_sovereign.multi_agent.config import (
    MULTI_AGENT_CONFIG_FILENAME,
    LocalAgentConfig,
    MultiAgentConfig,
)


def allocate_local_ports(config_path: Path, agent_name: str = "Kestrel") -> tuple[int, int]:
    config = MultiAgentConfig.load(config_path)
    if set(config.agents) != {agent_name}:
        raise ValueError("local clean-install requires exactly its disposable test agent")
    agent = config.agents[agent_name]
    if not isinstance(agent, LocalAgentConfig):
        raise ValueError("local clean-install agent must be local")

    # Hold both reservations until the validated config is saved. The launch
    # follows immediately; there is no destructive action against port owners.
    with socket.socket() as host_socket, socket.socket() as agent_socket:
        host_socket.bind(("127.0.0.1", 0))
        agent_socket.bind(("127.0.0.1", 0))
        host_port = host_socket.getsockname()[1]
        agent_port = agent_socket.getsockname()[1]
        updated = MultiAgentConfig(
            host=config.host.model_copy(
                update={"port": host_port, "bind": "127.0.0.1"}
            ),
            agents={
                agent_name: agent.model_copy(update={"port": agent_port})
            },
        )
        updated.save(config_path)
    return host_port, agent_port


if __name__ == "__main__":
    host, agent = allocate_local_ports(Path(MULTI_AGENT_CONFIG_FILENAME))
    print(f"Local clean-install ports: host={host}, agent={agent}")
