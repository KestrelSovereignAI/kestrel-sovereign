"""
MultiAgent Configuration - Registry of agents managed by a Kestrel Host.

Each agent is self-contained in its own directory with:
- DID (Decentralized Identifier)
- Database (kestrel_prime.db)
- Keys (agent-signing-key.pem, agent-encryption-key.pem)
- Config (kestrel.toml)

The multi_agent.toml file defines which agents exist and how to reach them.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union, Literal

import toml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

DEFAULT_HOST_PORT = 8888
DEFAULT_AGENT_START_PORT = 8801
MULTI_AGENT_CONFIG_FILENAME = "multi_agent.toml"
AGENT_DATA_DIR = "agent_data"

# Canonical modules for the features that form every agent's sovereignty
# foundation. Discovery imports these modules explicitly and fails closed, so
# an import/constructor failure cannot be mistaken for an optional capability
# degrading gracefully. Derive the public name set from this mapping: keeping
# two independent lists is exactly how a new mandatory feature could become
# documented but unenforced.
MANDATORY_FEATURE_MODULES = {
    "IdentityFeature": "kestrel_sovereign.features.identity.feature",
    "SecurityFeature": "kestrel_sovereign.features.security.feature",
    "PeersFeature": "kestrel_sovereign.features.peers.feature",
    "ConstitutionFeature": "kestrel_sovereign.features.constitution",
    # The single generic `wait` tool lives in its own mandatory feature so it
    # is always present — independent of optional features like Tasks/Talon
    # that register waitable providers (#1860 clean cutover).
    "WaitFeature": "kestrel_sovereign.features.wait.feature",
}
MANDATORY_FEATURES = frozenset(MANDATORY_FEATURE_MODULES)


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
    identity_export_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Optional per-agent identity export directory. Relative paths "
            "are resolved below this agent's data_dir."
        ),
    )
    features: Optional[List[str]] = Field(
        default=None,
        description=(
            "Allowed feature class names for this agent. "
            "If None, all discovered features are loaded (backward compatible). "
            "Mandatory features (Identity, Security, Peers, Constitution, Wait) "
            "are always loaded regardless of this list."
        ),
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def coerce_data_dir(cls, v: Union[str, Path]) -> Path:
        """Convert string to Path (preserves relative paths)."""
        return Path(v)

    @field_validator("identity_export_dir", mode="before")
    @classmethod
    def coerce_identity_export_dir(
        cls,
        value: Optional[Union[str, Path]],
    ) -> Optional[Path]:
        """Convert an optional export override while preserving relativity."""

        return None if value is None else Path(value)

    def resolve_data_dir(self, base_dir: Optional[Path] = None) -> Path:
        """Resolve this agent's data root using the runtime project base."""

        if base_dir is None:
            return self.data_dir.expanduser().resolve()
        return (base_dir / self.data_dir.expanduser()).resolve()

    def resolve_identity_export_dir(
        self,
        base_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Resolve the optional override, relative to this agent's data root."""

        if self.identity_export_dir is None:
            return None
        configured = self.identity_export_dir.expanduser()
        if configured.is_absolute():
            return configured.resolve()
        return (self.resolve_data_dir(base_dir) / configured).resolve()

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
        resolved = self.resolve_data_dir(base_dir)
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


class MultiAgentConfig(BaseModel):
    """
    MultiAgent configuration - registry of agents managed by a Kestrel Host.

    Example multi_agent.toml:
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
    def validate_port_conflicts(self) -> "MultiAgentConfig":
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
    def from_file(cls, config_path: Union[str, Path]) -> "MultiAgentConfig":
        """
        Load multi_agent config from a TOML file.

        Args:
            config_path: Path to multi_agent.toml

        Returns:
            MultiAgentConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"MultiAgent config not found: {path}")

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
    ) -> "MultiAgentConfig":
        """
        Auto-discover agents from agent_data/* subdirectories.

        This is used as a fallback when no multi_agent.toml exists.
        Scans for directories containing kestrel_prime.db (or any subdirectory
        when include_empty=True) and assigns ports sequentially starting from 8801.

        Args:
            base_dir: Directory to scan for agents (default: agent_data/)
            include_empty: If True, include empty subdirectories (for fresh provisioning)

        Returns:
            MultiAgentConfig with auto-discovered agents
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
    ) -> "MultiAgentConfig":
        """
        Load multi_agent config with auto-discovery fallback.

        Args:
            config_path: Path to multi_agent.toml (default: ./multi_agent.toml)
            auto_discover_fallback: If True and config doesn't exist, auto-discover agents

        Returns:
            MultiAgentConfig instance
        """
        if config_path is None:
            config_path = Path.cwd() / MULTI_AGENT_CONFIG_FILENAME

        path = Path(config_path)

        if path.exists():
            logger.info(f"Loading multi_agent config from {path}")
            return cls.from_file(path)

        if auto_discover_fallback:
            # Scan for agents relative to the config file's parent directory
            base_dir = path.parent / AGENT_DATA_DIR
            logger.info(f"No multi_agent config found at {path}, auto-discovering agents in {base_dir}...")
            return cls.auto_discover(base_dir)

        # No config and no auto-discovery
        logger.warning(f"No multi_agent config found at {path}")
        return cls(host=HostConfig(), agents={})

    def save(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save multi_agent config to a TOML file.

        Args:
            config_path: Path to save to (default: ./multi_agent.toml)
        """
        if config_path is None:
            config_path = Path.cwd() / MULTI_AGENT_CONFIG_FILENAME

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
                entry = {
                    "data_dir": str(agent.data_dir),
                    "port": agent.port,
                    "autostart": agent.autostart,
                }
                # A missing features key means "all features" on reload —
                # dropping a configured allowlist here would silently LIFT an
                # agent's feature restriction on the next boot (codex P1 on
                # #2358: the create-agent endpoint rewrites the whole file).
                if agent.features is not None:
                    entry["features"] = list(agent.features)
                if agent.identity_export_dir is not None:
                    entry["identity_export_dir"] = str(agent.identity_export_dir)
                data["agents"][name] = entry
            elif isinstance(agent, RemoteAgentConfig):
                data["agents"][name] = {
                    "url": agent.url,
                }

        # Save ATOMICALLY (codex P2 on #2358): writing the target in place
        # truncates it first — a failure mid-write (full disk, kill) leaves
        # multi_agent.toml empty/partial and the whole fleet unregistered on
        # the next boot. Write a sibling temp file and os.replace() it in.
        import os
        import tempfile
        # Write THROUGH symlinks (codex P2 round 9): os.replace on the link
        # path would swap the SYMLINK for a regular file, silently severing an
        # operator-managed config link — the in-place open('w') this replaced
        # followed the link. Resolve to the real target and replace that.
        if path.exists():
            path = path.resolve()
        # Ownership (codex P2 rounds 8-11): os.replace transfers the temp
        # file's owner onto the target, and uid can't be preserved without
        # root. When the file exists but ISN'T ours, write IN PLACE — that
        # keeps the inode, so owner/group/mode/ACLs all survive untouched;
        # the atomic strategy is reserved for files we own.
        try:
            if path.exists() and hasattr(os, "getuid") and path.stat().st_uid != os.getuid():
                with open(path, "w", encoding="utf-8") as f:
                    toml.dump(data, f)
                logger.info(f"Saved multi_agent config to {path} (in-place: preserving foreign ownership)")
                return
        except OSError:
            pass  # stat/uid unavailable — fall through to the atomic path
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
            )
        except OSError:
            # The parent directory isn't writable but the FILE may be (a
            # group-writable config under an operator-owned directory) — the
            # in-place write this strategy replaced handled that fine. Fall
            # back to it: non-atomic, but strictly better than refusing to
            # persist at all (codex P2 round 10).
            with open(path, "w", encoding="utf-8") as f:
                toml.dump(data, f)
            logger.info(f"Saved multi_agent config to {path} (in-place: parent dir not writable)")
            return
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                toml.dump(data, f)
            # mkstemp minted 0600 owned by this process — carrying that onto
            # the target would strip metadata an operator deliberately set
            # (codex P2 rounds 7-8):
            #   - EXISTING file: preserve its mode AND its group (another
            #     service account may read the config via group access);
            #     chown is best-effort — group changes need membership.
            #   - NEW file: derive from the process umask exactly like the
            #     plain open(..., 'w') this replaced (0666 & ~umask) — a
            #     hardcoded 0644 would bypass restrictive umasks and expose
            #     fleet topology / remote-agent URLs to other local users.
            try:
                if path.exists():
                    st = path.stat()
                    os.chmod(tmp_path, st.st_mode & 0o7777)
                    try:
                        # uid preservation only works as root; gid works with
                        # membership — try both, degrade per-field (round 10).
                        os.chown(tmp_path, st.st_uid, st.st_gid)
                    except (OSError, AttributeError):
                        try:
                            os.chown(tmp_path, -1, st.st_gid)
                        except (OSError, AttributeError):
                            pass  # non-POSIX or not a member of the group
                else:
                    current_umask = os.umask(0)
                    os.umask(current_umask)
                    os.chmod(tmp_path, 0o666 & ~current_umask)
            except OSError:
                pass  # never fail the save over metadata
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(f"Saved multi_agent config to {path}")

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
