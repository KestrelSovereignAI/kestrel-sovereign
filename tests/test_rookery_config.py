"""
Unit tests for Rookery configuration.

Tests config parsing, validation, and auto-discovery.
"""

import pytest
import toml
from pathlib import Path
from pydantic import ValidationError

from kestrel_sovereign.rookery import (
    RookeryConfig,
    HostConfig,
    LocalAgentConfig,
    RemoteAgentConfig,
)


@pytest.fixture
def temp_agent_dir(tmp_path):
    """Create a temporary agent directory with kestrel_prime.db"""
    agent_dir = tmp_path / "test_agent"
    agent_dir.mkdir()
    (agent_dir / "kestrel_prime.db").touch()
    return agent_dir


@pytest.fixture
def temp_rookery_config(tmp_path):
    """Create a temporary rookery.toml file"""
    config_path = tmp_path / "rookery.toml"

    def _create_config(content: dict):
        with open(config_path, "w") as f:
            toml.dump(content, f)
        return config_path

    return _create_config


class TestHostConfig:
    """Tests for HostConfig model."""

    def test_default_values(self):
        """Test default host config values."""
        config = HostConfig()
        assert config.port == 8888
        assert config.bind == "0.0.0.0"

    def test_custom_values(self):
        """Test custom host config values."""
        config = HostConfig(port=9999, bind="127.0.0.1")
        assert config.port == 9999
        assert config.bind == "127.0.0.1"

    def test_port_validation_below_range(self):
        """Test port validation rejects ports below 1024."""
        with pytest.raises(ValidationError) as exc_info:
            HostConfig(port=80)
        assert "greater than or equal to 1024" in str(exc_info.value)

    def test_port_validation_above_range(self):
        """Test port validation rejects ports above 65535."""
        with pytest.raises(ValidationError) as exc_info:
            HostConfig(port=70000)
        assert "less than or equal to 65535" in str(exc_info.value)


class TestLocalAgentConfig:
    """Tests for LocalAgentConfig model."""

    def test_valid_local_agent(self, temp_agent_dir):
        """Test creating a valid local agent config."""
        config = LocalAgentConfig(
            data_dir=temp_agent_dir,
            port=8801,
            autostart=True,
        )
        assert config.data_dir == temp_agent_dir.resolve()
        assert config.port == 8801
        assert config.autostart is True

    def test_default_autostart(self, temp_agent_dir):
        """Test autostart defaults to True."""
        config = LocalAgentConfig(
            data_dir=temp_agent_dir,
            port=8801,
        )
        assert config.autostart is True

    def test_missing_data_dir(self, tmp_path):
        """Test validation fails for non-existent data_dir."""
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(ValidationError) as exc_info:
            LocalAgentConfig(
                data_dir=nonexistent,
                port=8801,
            )
        assert "does not exist" in str(exc_info.value)

    def test_missing_database(self, tmp_path):
        """Test validation fails when kestrel_prime.db is missing."""
        agent_dir = tmp_path / "no_db"
        agent_dir.mkdir()
        # No kestrel_prime.db created

        with pytest.raises(ValidationError) as exc_info:
            LocalAgentConfig(
                data_dir=agent_dir,
                port=8801,
            )
        assert "kestrel_prime.db" in str(exc_info.value)

    def test_data_dir_is_file(self, tmp_path):
        """Test validation fails when data_dir is a file, not a directory."""
        file_path = tmp_path / "file.txt"
        file_path.touch()

        with pytest.raises(ValidationError) as exc_info:
            LocalAgentConfig(
                data_dir=file_path,
                port=8801,
            )
        assert "must be a directory" in str(exc_info.value)

    def test_port_validation(self, temp_agent_dir):
        """Test port validation."""
        with pytest.raises(ValidationError):
            LocalAgentConfig(data_dir=temp_agent_dir, port=80)

        with pytest.raises(ValidationError):
            LocalAgentConfig(data_dir=temp_agent_dir, port=70000)


class TestRemoteAgentConfig:
    """Tests for RemoteAgentConfig model."""

    def test_valid_http_url(self):
        """Test valid HTTP URL."""
        config = RemoteAgentConfig(url="http://example.com")
        assert config.url == "http://example.com"

    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        config = RemoteAgentConfig(url="https://example.com:8888")
        assert config.url == "https://example.com:8888"

    def test_invalid_url_no_scheme(self):
        """Test validation fails for URL without http:// or https://."""
        with pytest.raises(ValidationError) as exc_info:
            RemoteAgentConfig(url="example.com")
        assert "must start with http://" in str(exc_info.value)

    def test_invalid_url_wrong_scheme(self):
        """Test validation fails for non-http schemes."""
        with pytest.raises(ValidationError) as exc_info:
            RemoteAgentConfig(url="ftp://example.com")
        assert "must start with http://" in str(exc_info.value)


class TestRookeryConfig:
    """Tests for RookeryConfig model."""

    def test_empty_config(self):
        """Test creating an empty rookery config."""
        config = RookeryConfig()
        assert config.host.port == 8888
        assert config.host.bind == "0.0.0.0"
        assert len(config.agents) == 0

    def test_config_with_local_agent(self, temp_agent_dir):
        """Test config with a local agent."""
        config = RookeryConfig(
            agents={
                "test_agent": LocalAgentConfig(
                    data_dir=temp_agent_dir,
                    port=8801,
                )
            }
        )
        assert "test_agent" in config.agents
        assert isinstance(config.agents["test_agent"], LocalAgentConfig)

    def test_config_with_remote_agent(self):
        """Test config with a remote agent."""
        config = RookeryConfig(
            agents={
                "remote": RemoteAgentConfig(url="https://example.com")
            }
        )
        assert "remote" in config.agents
        assert isinstance(config.agents["remote"], RemoteAgentConfig)

    def test_config_with_mixed_agents(self, temp_agent_dir):
        """Test config with both local and remote agents."""
        config = RookeryConfig(
            agents={
                "local": LocalAgentConfig(
                    data_dir=temp_agent_dir,
                    port=8801,
                ),
                "remote": RemoteAgentConfig(url="https://example.com"),
            }
        )
        assert len(config.agents) == 2
        assert isinstance(config.agents["local"], LocalAgentConfig)
        assert isinstance(config.agents["remote"], RemoteAgentConfig)

    def test_port_conflict_detection(self, temp_agent_dir, tmp_path):
        """Test that port conflicts are detected."""
        agent_dir2 = tmp_path / "agent2"
        agent_dir2.mkdir()
        (agent_dir2 / "kestrel_prime.db").touch()

        with pytest.raises(ValidationError) as exc_info:
            RookeryConfig(
                agents={
                    "agent1": LocalAgentConfig(
                        data_dir=temp_agent_dir,
                        port=8801,
                    ),
                    "agent2": LocalAgentConfig(
                        data_dir=agent_dir2,
                        port=8801,  # Same port!
                    ),
                }
            )
        assert "Port conflict" in str(exc_info.value)

    def test_host_port_conflict(self, temp_agent_dir):
        """Test that host port conflicts with agent ports are detected."""
        with pytest.raises(ValidationError) as exc_info:
            RookeryConfig(
                host=HostConfig(port=8801),
                agents={
                    "agent": LocalAgentConfig(
                        data_dir=temp_agent_dir,
                        port=8801,  # Same as host!
                    ),
                }
            )
        assert "Port conflict" in str(exc_info.value)


class TestRookeryConfigLoading:
    """Tests for loading rookery config from files."""

    def test_load_from_file(self, temp_rookery_config, temp_agent_dir):
        """Test loading config from a TOML file."""
        config_data = {
            "host": {"port": 9999, "bind": "127.0.0.1"},
            "agents": {
                "test_agent": {
                    "data_dir": str(temp_agent_dir),
                    "port": 8801,
                    "autostart": True,
                }
            },
        }
        config_path = temp_rookery_config(config_data)

        config = RookeryConfig.from_file(config_path)
        assert config.host.port == 9999
        assert config.host.bind == "127.0.0.1"
        assert "test_agent" in config.agents
        assert config.agents["test_agent"].port == 8801

    def test_load_missing_file(self, tmp_path):
        """Test loading from a non-existent file raises error."""
        config_path = tmp_path / "does_not_exist.toml"
        with pytest.raises(FileNotFoundError):
            RookeryConfig.from_file(config_path)

    def test_load_invalid_toml(self, tmp_path):
        """Test loading invalid TOML raises error."""
        config_path = tmp_path / "invalid.toml"
        config_path.write_text("invalid toml content {{{")

        with pytest.raises(ValueError) as exc_info:
            RookeryConfig.from_file(config_path)
        assert "Invalid TOML" in str(exc_info.value)

    def test_load_remote_agent(self, temp_rookery_config):
        """Test loading config with remote agents."""
        config_data = {
            "agents": {
                "remote": {
                    "url": "https://example.com"
                }
            }
        }
        config_path = temp_rookery_config(config_data)

        config = RookeryConfig.from_file(config_path)
        assert "remote" in config.agents
        assert isinstance(config.agents["remote"], RemoteAgentConfig)
        assert config.agents["remote"].url == "https://example.com"

    def test_load_agent_without_required_fields(self, temp_rookery_config):
        """Test that agents without url or data_dir raise error."""
        config_data = {
            "agents": {
                "invalid": {
                    "port": 8801  # Missing data_dir
                }
            }
        }
        config_path = temp_rookery_config(config_data)

        with pytest.raises(ValueError) as exc_info:
            RookeryConfig.from_file(config_path)
        assert "must have either 'url'" in str(exc_info.value)


class TestAutoDiscovery:
    """Tests for auto-discovery of agents."""

    def test_auto_discover_empty_dir(self, tmp_path):
        """Test auto-discovery with no agents."""
        agent_data = tmp_path / "agent_data"
        agent_data.mkdir()

        config = RookeryConfig.auto_discover(agent_data)
        assert len(config.agents) == 0

    def test_auto_discover_single_agent(self, tmp_path):
        """Test auto-discovery with one agent."""
        agent_data = tmp_path / "agent_data"
        agent_data.mkdir()

        agent1 = agent_data / "agent1"
        agent1.mkdir()
        (agent1 / "kestrel_prime.db").touch()

        config = RookeryConfig.auto_discover(agent_data)
        assert len(config.agents) == 1
        assert "agent1" in config.agents
        assert config.agents["agent1"].port == 8801
        assert config.agents["agent1"].autostart is True

    def test_auto_discover_multiple_agents(self, tmp_path):
        """Test auto-discovery with multiple agents."""
        agent_data = tmp_path / "agent_data"
        agent_data.mkdir()

        # Create 3 agents
        for i in range(1, 4):
            agent_dir = agent_data / f"agent{i}"
            agent_dir.mkdir()
            (agent_dir / "kestrel_prime.db").touch()

        config = RookeryConfig.auto_discover(agent_data)
        assert len(config.agents) == 3

        # Check sequential port assignment
        assert config.agents["agent1"].port == 8801
        assert config.agents["agent2"].port == 8802
        assert config.agents["agent3"].port == 8803

    def test_auto_discover_skips_invalid_dirs(self, tmp_path):
        """Test that auto-discovery skips directories without kestrel_prime.db."""
        agent_data = tmp_path / "agent_data"
        agent_data.mkdir()

        # Valid agent
        valid = agent_data / "valid"
        valid.mkdir()
        (valid / "kestrel_prime.db").touch()

        # Invalid - no database
        invalid = agent_data / "invalid"
        invalid.mkdir()

        # File, not directory
        (agent_data / "file.txt").touch()

        config = RookeryConfig.auto_discover(agent_data)
        assert len(config.agents) == 1
        assert "valid" in config.agents

    def test_auto_discover_nonexistent_dir(self, tmp_path):
        """Test auto-discovery with non-existent directory."""
        nonexistent = tmp_path / "does_not_exist"

        config = RookeryConfig.auto_discover(nonexistent)
        assert len(config.agents) == 0


class TestRookeryConfigLoad:
    """Tests for the load() method with fallback."""

    def test_load_existing_config(self, temp_rookery_config, temp_agent_dir, tmp_path):
        """Test load() uses existing config file."""
        config_data = {
            "agents": {
                "test": {
                    "data_dir": str(temp_agent_dir),
                    "port": 8801,
                }
            }
        }
        config_path = temp_rookery_config(config_data)

        config = RookeryConfig.load(config_path)
        assert "test" in config.agents

    def test_load_auto_discover_fallback(self, tmp_path):
        """Test load() falls back to auto-discovery when config doesn't exist."""
        agent_data = tmp_path / "agent_data"
        agent_data.mkdir()

        agent = agent_data / "discovered"
        agent.mkdir()
        (agent / "kestrel_prime.db").touch()

        # Try to load non-existent config
        config_path = tmp_path / "rookery.toml"

        with pytest.MonkeyPatch.context() as mp:
            mp.chdir(tmp_path)
            config = RookeryConfig.load(config_path, auto_discover_fallback=True)

        assert "discovered" in config.agents

    def test_load_no_fallback(self, tmp_path):
        """Test load() without auto-discovery fallback."""
        config_path = tmp_path / "rookery.toml"
        config = RookeryConfig.load(config_path, auto_discover_fallback=False)
        assert len(config.agents) == 0


class TestRookeryConfigSave:
    """Tests for saving rookery config."""

    def test_save_to_file(self, tmp_path, temp_agent_dir):
        """Test saving config to a TOML file."""
        config = RookeryConfig(
            host=HostConfig(port=9999),
            agents={
                "test": LocalAgentConfig(
                    data_dir=temp_agent_dir,
                    port=8801,
                    autostart=False,
                )
            },
        )

        config_path = tmp_path / "rookery.toml"
        config.save(config_path)

        # Load and verify
        loaded = RookeryConfig.from_file(config_path)
        assert loaded.host.port == 9999
        assert "test" in loaded.agents
        assert loaded.agents["test"].port == 8801
        assert loaded.agents["test"].autostart is False

    def test_save_with_remote_agents(self, tmp_path):
        """Test saving config with remote agents."""
        config = RookeryConfig(
            agents={
                "remote": RemoteAgentConfig(url="https://example.com")
            }
        )

        config_path = tmp_path / "rookery.toml"
        config.save(config_path)

        # Verify TOML structure
        data = toml.load(config_path)
        assert data["agents"]["remote"]["url"] == "https://example.com"


class TestRookeryConfigHelpers:
    """Tests for helper methods."""

    def test_get_local_agents(self, temp_agent_dir):
        """Test get_local_agents() filters correctly."""
        config = RookeryConfig(
            agents={
                "local": LocalAgentConfig(
                    data_dir=temp_agent_dir,
                    port=8801,
                ),
                "remote": RemoteAgentConfig(url="https://example.com"),
            }
        )

        local = config.get_local_agents()
        assert len(local) == 1
        assert "local" in local

    def test_get_remote_agents(self, temp_agent_dir):
        """Test get_remote_agents() filters correctly."""
        config = RookeryConfig(
            agents={
                "local": LocalAgentConfig(
                    data_dir=temp_agent_dir,
                    port=8801,
                ),
                "remote": RemoteAgentConfig(url="https://example.com"),
            }
        )

        remote = config.get_remote_agents()
        assert len(remote) == 1
        assert "remote" in remote

    def test_get_autostart_agents(self, temp_agent_dir, tmp_path):
        """Test get_autostart_agents() filters correctly."""
        agent2 = tmp_path / "agent2"
        agent2.mkdir()
        (agent2 / "kestrel_prime.db").touch()

        config = RookeryConfig(
            agents={
                "auto": LocalAgentConfig(
                    data_dir=temp_agent_dir,
                    port=8801,
                    autostart=True,
                ),
                "manual": LocalAgentConfig(
                    data_dir=agent2,
                    port=8802,
                    autostart=False,
                ),
                "remote": RemoteAgentConfig(url="https://example.com"),
            }
        )

        autostart = config.get_autostart_agents()
        assert len(autostart) == 1
        assert "auto" in autostart
