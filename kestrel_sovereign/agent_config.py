"""
Agent Configuration Management.

Each agent directory can have a kestrel.toml config file:

```toml
[agent]
name = "Claw"
port = 8888

[server]
host = "0.0.0.0"
log_level = "INFO"
```

The CLI reads this config and uses it for start/stop/status commands.
"""

import logging
import os
import toml

logger = logging.getLogger(__name__)
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_PORT = 8888
DEFAULT_HOST = "0.0.0.0"
CONFIG_FILENAME = "kestrel.toml"
PID_FILENAME = ".kestrel.pid"


@dataclass
class AgentConfig:
    """Configuration for a Kestrel agent."""

    # Agent settings
    name: str = "Kestrel Agent"
    port: int = DEFAULT_PORT

    # Server settings
    host: str = DEFAULT_HOST
    log_level: str = "INFO"
    auto_reload: bool = False

    # Paths (computed from agent_dir)
    agent_dir: Path = field(default_factory=Path)
    db_path: Path = field(default_factory=Path)
    pid_file: Path = field(default_factory=Path)
    log_file: Path = field(default_factory=Path)
    config_file: Path = field(default_factory=Path)

    @classmethod
    def from_directory(cls, agent_dir: str | Path) -> "AgentConfig":
        """
        Load config from an agent directory.

        Looks for kestrel.toml in the directory. If not found, uses defaults.
        """
        agent_path = Path(agent_dir).resolve()
        config_path = agent_path / CONFIG_FILENAME

        # Start with defaults
        config = cls()
        config.agent_dir = agent_path
        config.db_path = agent_path / "kestrel_prime.db"
        config.pid_file = agent_path / PID_FILENAME
        config.log_file = agent_path / "kestrel.log"
        config.config_file = config_path

        # Load from file if exists
        if config_path.exists():
            try:
                data = toml.load(config_path)

                # Agent section
                agent = data.get("agent", {})
                config.name = agent.get("name", config.name)
                config.port = agent.get("port", config.port)

                # Server section
                server = data.get("server", {})
                config.host = server.get("host", config.host)
                config.log_level = server.get("log_level", config.log_level)
                config.auto_reload = server.get("auto_reload", config.auto_reload)

            except toml.TomlDecodeError as e:
                logger.error(f"Invalid TOML in {config_path}: {e}")
            except Exception as e:
                logger.warning(f"Failed to load {config_path}: {e}")

        return config

    def save(self) -> None:
        """Save current config to kestrel.toml in the agent directory."""
        data = {
            "agent": {
                "name": self.name,
                "port": self.port,
            },
            "server": {
                "host": self.host,
                "log_level": self.log_level,
                "auto_reload": self.auto_reload,
            }
        }

        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            toml.dump(data, f)

    def exists(self) -> bool:
        """Check if the agent directory exists and has a database."""
        return self.db_path.exists()

    def get_pid(self) -> Optional[int]:
        """Read PID from file, return None if not exists or invalid."""
        if not self.pid_file.exists():
            return None
        try:
            return int(self.pid_file.read_text().strip())
        except (ValueError, OSError):
            return None

    def save_pid(self, pid: int) -> None:
        """Save PID to file."""
        self.pid_file.write_text(str(pid))

    def clear_pid(self) -> None:
        """Remove PID file."""
        self.pid_file.unlink(missing_ok=True)

    def __str__(self) -> str:
        return f"AgentConfig({self.name} @ {self.agent_dir}, port={self.port})"


def find_agent_dir(hint: Optional[str] = None) -> Optional[Path]:
    """
    Find an agent directory.

    Search order:
    1. Explicit hint path
    2. KESTREL_DB_PATH environment variable
    3. Common locations: ./agent_data/*, ./my_agent, .
    """
    # 1. Explicit hint
    if hint:
        path = Path(hint).resolve()
        if (path / "kestrel_prime.db").exists():
            return path
        # Maybe they gave us a file path?
        if path.suffix == ".db" and path.exists():
            return path.parent

    # 2. Environment variable
    env_path = os.environ.get("KESTREL_DB_PATH")
    if env_path:
        path = Path(env_path).resolve()
        if path.is_dir() and (path / "kestrel_prime.db").exists():
            return path
        elif path.suffix == ".db" and path.exists():
            return path.parent

    # 3. Common locations
    for candidate in ["./agent_data/claw", "./my_agent", "."]:
        path = Path(candidate).resolve()
        if (path / "kestrel_prime.db").exists():
            return path

    # 4. Search agent_data subdirectories
    agent_data = Path("./agent_data")
    if agent_data.is_dir():
        for subdir in agent_data.iterdir():
            if subdir.is_dir() and (subdir / "kestrel_prime.db").exists():
                return subdir

    return None


def list_agents(base_dir: str = "./agent_data") -> list[AgentConfig]:
    """List all agents in a directory."""
    agents = []
    base = Path(base_dir)

    if not base.is_dir():
        return agents

    for subdir in base.iterdir():
        if subdir.is_dir() and (subdir / "kestrel_prime.db").exists():
            agents.append(AgentConfig.from_directory(subdir))

    return agents
