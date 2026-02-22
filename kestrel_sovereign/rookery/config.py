"""
Rookery Configuration - Registry of agents managed by a Kestrel Host.

Each agent is self-contained in its own directory with:
- DID (Decentralized Identifier)
- Database (kestrel_prime.db)
- Keys (agent-signing-key.pem, agent-encryption-key.pem)
- Config (kestrel.toml)

The rookery.toml file defines which agents exist and how to reach them.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Literal

import toml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

DEFAULT_HOST_PORT = 8888
DEFAULT_AGENT_START_PORT = 8801
ROOKERY_CONFIG_FILENAME = "rookery.toml"
AGENT_DATA_DIR = "agent_data"


class HostConfig(BaseModel):
    """Configuration for the Kestrel Host."""

    port: int = Field(
        default=DEFAULT_HOST_PORT,
        description="Port for the host UI/API",
        ge=1024,
        le=65535,
    )
    bind: str = Field(
        default="0.0.0.0",
        description="Interface to bind to (0.0.0.0 for all interfaces)",
    )


class LocalAgentConfig(BaseModel):
    """Configuration for a local agent managed by this host."""

    data_dir: Path = Field(
        description="Path to agent's data directory (contains kestrel_prime.db)"
    )
    port: int = Field(
        description="Port for this agent's API",
        ge=1024,
        le=65535,
    )
    autostart: bool = Field(
        default=True,
        description="Start this agent when the host starts",
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def coerce_data_dir(cls, v: Union[str, Path]) -> Path:
        """Convert string to Path (preserves relative paths)."""
        return Path(v)

    def validate_runtime(self, base_dir: Optional[Path] = None) -> list[str]:
        """Validate that data_dir exists and contains a database.

        Called by the process manager before starting an agent,
        NOT at config parse time (so you can pre-configure agents
        before running inception).

        Args:
            base_dir: Base directory to resolve relative data_dir against.
                      If None, resolves against CWD (for backward compat).

        Returns:
            List of error messages (empty if valid).
        """
        errors = []
        if base_dir is not None:
            resolved = (base_dir / self.data_dir).resolve()
        else:
            resolved = self.data_dir.resolve()
        if not resolved.exists():
            errors.append(f"Agent data directory does not exist: {resolved}")
        elif not resolved.is_dir():
            errors.append(f"Agent data_dir must be a directory: {resolved}")
        elif not (resolved / "kestrel_prime.db").exists():
            errors.append(
                f"Agent data directory missing kestrel_prime.db: {resolved}\n"
                f"Create an agent first with: kestrel create {self.data_dir.name}"
            )
        return errors


class RemoteAgentConfig(BaseModel):
    """Configuration for a remote agent (not managed by this host)."""

    url: str = Field(
        description="URL of the remote agent's API endpoint"
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Remote agent URL must start with http:// or https://: {v}")
        return v


class RookeryConfig(BaseModel):
    """
    Rookery configuration - registry of agents managed by a Kestrel Host.

    Example rookery.toml:
    ```toml
    [host]
    port = 8888
    bind = "0.0.0.0"

    [agents.claw]
    data_dir = "agent_data/claw"
    port = 8801
    autostart = true

    [agents.testbot]
    data_dir = "agent_data/testbot"
    port = 8802
    autostart = false

    [agents.remote-agent]
    url = "https://remote-kestrel.example.com"
    ```
    """

    host: HostConfig = Field(
        default_factory=HostConfig,
        description="Host configuration",
    )
    agents: dict[str, Union[LocalAgentConfig, RemoteAgentConfig]] = Field(
        default_factory=dict,
        description="Agent configurations keyed by agent name",
    )

    @model_validator(mode="after")
    def validate_port_conflicts(self) -> "RookeryConfig":
        """Validate that no two local agents use the same port."""
        used_ports: dict[int, str] = {}

        # Check host port
        used_ports[self.host.port] = "host"

        # Check agent ports
        for name, config in self.agents.items():
            if isinstance(config, LocalAgentConfig):
                port = config.port
                if port in used_ports:
                    raise ValueError(
                        f"Port conflict: agent '{name}' and '{used_ports[port]}' "
                        f"both use port {port}"
                    )
                used_ports[port] = name

        return self

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "RookeryConfig":
        """
        Load rookery config from a TOML file.

        Args:
            config_path: Path to rookery.toml

        Returns:
            RookeryConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Rookery config not found: {path}")

        try:
            data = toml.load(path)
        except toml.TomlDecodeError as e:
            raise ValueError(f"Invalid TOML in {path}: {e}")

        # Parse host config
        host_data = data.get("host", {})
        host = HostConfig(**host_data)

        # Parse agent configs
        agents: dict[str, Union[LocalAgentConfig, RemoteAgentConfig]] = {}
        agents_data = data.get("agents", {})

        for name, agent_data in agents_data.items():
            if "url" in agent_data:
                # Remote agent
                agents[name] = RemoteAgentConfig(**agent_data)
            elif "data_dir" in agent_data:
                # Local agent
                agents[name] = LocalAgentConfig(**agent_data)
            else:
                raise ValueError(
                    f"Agent '{name}' must have either 'url' (remote) or "
                    f"'data_dir' + 'port' (local)"
                )

        return cls(host=host, agents=agents)

    @classmethod
    def auto_discover(
        cls,
        base_dir: Union[str, Path] = AGENT_DATA_DIR,
        include_empty: bool = False,
    ) -> "RookeryConfig":
        """
        Auto-discover agents from agent_data/* subdirectories.

        This is used as a fallback when no rookery.toml exists.
        Scans for directories containing kestrel_prime.db (or any subdirectory
        when include_empty=True) and assigns ports sequentially starting from 8801.

        Args:
            base_dir: Directory to scan for agents (default: agent_data/)
            include_empty: If True, include empty subdirectories (for fresh provisioning)

        Returns:
            RookeryConfig with auto-discovered agents
        """
        base_path = Path(base_dir)
        agents: dict[str, LocalAgentConfig] = {}
        next_port = DEFAULT_AGENT_START_PORT

        if not base_path.is_dir():
            logger.warning(f"Agent data directory not found: {base_path}")
            return cls(host=HostConfig(), agents={})

        # Scan subdirectories
        for subdir in sorted(base_path.iterdir()):
            if not subdir.is_dir():
                continue

            db_path = subdir / "kestrel_prime.db"
            if not db_path.exists() and not include_empty:
                continue

            # Found an agent directory
            name = subdir.name
            agents[name] = LocalAgentConfig(
                data_dir=subdir,
                port=next_port,
                autostart=True,
            )
            logger.info(f"Auto-discovered agent: {name} at {subdir} (port {next_port})")
            next_port += 1

        return cls(host=HostConfig(), agents=agents)

    @classmethod
    def load(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        auto_discover_fallback: bool = True,
    ) -> "RookeryConfig":
        """
        Load rookery config with auto-discovery fallback.

        Args:
            config_path: Path to rookery.toml (default: ./rookery.toml)
            auto_discover_fallback: If True and config doesn't exist, auto-discover agents

        Returns:
            RookeryConfig instance
        """
        if config_path is None:
            config_path = Path.cwd() / ROOKERY_CONFIG_FILENAME

        path = Path(config_path)

        if path.exists():
            logger.info(f"Loading rookery config from {path}")
            return cls.from_file(path)

        if auto_discover_fallback:
            # Scan for agents relative to the config file's parent directory
            base_dir = path.parent / AGENT_DATA_DIR
            logger.info(f"No rookery config found at {path}, auto-discovering agents in {base_dir}...")
            return cls.auto_discover(base_dir)

        # No config and no auto-discovery
        logger.warning(f"No rookery config found at {path}")
        return cls(host=HostConfig(), agents={})

    def save(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save rookery config to a TOML file.

        Args:
            config_path: Path to save to (default: ./rookery.toml)
        """
        if config_path is None:
            config_path = Path.cwd() / ROOKERY_CONFIG_FILENAME

        path = Path(config_path)

        # Build TOML structure
        data = {
            "host": {
                "port": self.host.port,
                "bind": self.host.bind,
            },
            "agents": {},
        }

        for name, agent in self.agents.items():
            if isinstance(agent, LocalAgentConfig):
                data["agents"][name] = {
                    "data_dir": str(agent.data_dir),
                    "port": agent.port,
                    "autostart": agent.autostart,
                }
            elif isinstance(agent, RemoteAgentConfig):
                data["agents"][name] = {
                    "url": agent.url,
                }

        # Save to file
        with open(path, "w") as f:
            toml.dump(data, f)

        logger.info(f"Saved rookery config to {path}")

    def get_local_agents(self) -> dict[str, LocalAgentConfig]:
        """Get only local agents."""
        return {
            name: config
            for name, config in self.agents.items()
            if isinstance(config, LocalAgentConfig)
        }

    def get_remote_agents(self) -> dict[str, RemoteAgentConfig]:
        """Get only remote agents."""
        return {
            name: config
            for name, config in self.agents.items()
            if isinstance(config, RemoteAgentConfig)
        }

    def get_autostart_agents(self) -> dict[str, LocalAgentConfig]:
        """Get local agents with autostart=True."""
        return {
            name: config
            for name, config in self.get_local_agents().items()
            if config.autostart
        }
