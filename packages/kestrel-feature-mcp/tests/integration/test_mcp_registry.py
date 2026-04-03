"""
Integration tests for MCP Registry.

Tests the server catalog, search functionality, and registry operations.
No Docker required - these test the registry metadata layer only.
"""

import pytest
from pathlib import Path

from kestrel_feature_mcp.registry import (
    MCPRegistry,
    MCPServerEntry,
    ServerType,
    ServerCategory,
    TransportType,
    get_registry,
)


class TestMCPServerEntry:
    """Tests for MCPServerEntry dataclass."""

    def test_builtin_server_entry(self):
        """Test creating a built-in server entry."""
        entry = MCPServerEntry(
            name="filesystem",
            description="Local filesystem access",
            tools=["read_file", "write_file"],
            categories=[ServerCategory.FILESYSTEM],
            server_type=ServerType.BUILTIN,
        )

        assert entry.name == "filesystem"
        assert entry.is_builtin is True
        assert entry.image is None
        assert entry.full_image is None
        assert ServerCategory.FILESYSTEM in entry.categories

    def test_docker_server_entry(self):
        """Test creating a Docker-based server entry."""
        entry = MCPServerEntry(
            name="postgres",
            description="PostgreSQL database access",
            tools=["query"],
            categories=[ServerCategory.DATABASE],
            server_type=ServerType.DOCKER,
            image="mcp/postgres:latest",
            env_required=["POSTGRES_CONNECTION_STRING"],
        )

        assert entry.name == "postgres"
        assert entry.is_builtin is False
        assert entry.image == "mcp/postgres:latest"
        assert entry.full_image == "mcp/postgres:latest"
        assert "POSTGRES_CONNECTION_STRING" in entry.env_required

    def test_private_registry_entry(self):
        """Test private registry image resolution."""
        entry = MCPServerEntry(
            name="fhir",
            description="FHIR healthcare data",
            tools=["get_patient"],
            categories=[ServerCategory.HEALTHCARE],
            server_type=ServerType.DOCKER,
            image="mcp-fhir:latest",
            private=True,
            registry="gcr.io/YOUR_PROJECT_ID",
        )

        assert entry.private is True
        assert entry.full_image == "gcr.io/YOUR_PROJECT_ID/mcp-fhir:latest"

    def test_matches_query_name(self):
        """Test query matching against name."""
        entry = MCPServerEntry(
            name="postgres",
            description="Database access",
            tools=["query"],
            categories=[ServerCategory.DATABASE],
        )

        assert entry.matches_query("postgres") is True
        assert entry.matches_query("POST") is True  # case insensitive
        assert entry.matches_query("mysql") is False

    def test_matches_query_description(self):
        """Test query matching against description."""
        entry = MCPServerEntry(
            name="pg",
            description="PostgreSQL database access for SQL queries",
            tools=["query"],
            categories=[ServerCategory.DATABASE],
        )

        assert entry.matches_query("SQL") is True
        assert entry.matches_query("database") is True

    def test_matches_query_tools(self):
        """Test query matching against tool names."""
        entry = MCPServerEntry(
            name="fs",
            description="File operations",
            tools=["read_file", "write_file", "list_directory"],
            categories=[ServerCategory.FILESYSTEM],
        )

        assert entry.matches_query("read") is True
        assert entry.matches_query("write") is True
        assert entry.matches_query("delete") is False

    def test_matches_query_categories(self):
        """Test query matching against categories."""
        entry = MCPServerEntry(
            name="pg",
            description="DB",
            tools=["query"],
            categories=[ServerCategory.DATABASE, ServerCategory.DEVTOOLS],
        )

        assert entry.matches_query("database") is True
        assert entry.matches_query("devtools") is True
        assert entry.matches_query("web") is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        entry = MCPServerEntry(
            name="test",
            description="Test server",
            tools=["tool1", "tool2"],
            categories=[ServerCategory.UTILITY],
            server_type=ServerType.DOCKER,
            image="test:latest",
        )

        d = entry.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test server"
        assert d["tools"] == ["tool1", "tool2"]
        assert d["categories"] == ["utility"]
        assert d["server_type"] == "docker"
        assert d["builtin"] is False


class TestMCPRegistry:
    """Tests for MCP Registry."""

    @pytest.fixture
    def registry(self):
        """Get a fresh registry instance loaded from the real catalog."""
        # Reset the singleton
        import kestrel_feature_mcp.registry as reg_module
        reg_module._registry = None
        return get_registry()

    def test_registry_loads_catalog(self, registry):
        """Test that the registry loads servers from catalog.toml."""
        assert len(registry.servers) > 0
        # Check that some expected servers are present
        assert "filesystem" in registry.servers
        assert "fetch" in registry.servers

    def test_get_server_by_name(self, registry):
        """Test getting a server by name."""
        server = registry.get("filesystem")
        assert server is not None
        assert server.name == "filesystem"
        assert server.is_builtin is True

    def test_get_nonexistent_server(self, registry):
        """Test getting a server that doesn't exist."""
        server = registry.get("nonexistent-server-xyz")
        assert server is None

    def test_list_all_servers(self, registry):
        """Test listing all servers."""
        all_servers = registry.list_all()
        assert len(all_servers) > 0
        names = [s.name for s in all_servers]
        assert "filesystem" in names
        assert "fetch" in names

    def test_list_by_category(self, registry):
        """Test listing servers by category."""
        db_servers = registry.list_by_category(ServerCategory.DATABASE)
        assert len(db_servers) > 0
        for server in db_servers:
            assert ServerCategory.DATABASE in server.categories

    def test_list_docker_servers(self, registry):
        """Test listing only Docker-based servers."""
        docker_servers = registry.list_docker()
        assert len(docker_servers) > 0
        for server in docker_servers:
            assert server.server_type == ServerType.DOCKER
            assert server.is_builtin is False

    def test_list_builtin_servers(self, registry):
        """Test listing only built-in servers."""
        builtin_servers = registry.list_builtin()
        assert len(builtin_servers) > 0
        for server in builtin_servers:
            assert server.server_type == ServerType.BUILTIN
            assert server.is_builtin is True

    def test_search_by_name(self, registry):
        """Test searching for servers by name."""
        results = registry.search("postgres")
        assert len(results) > 0
        names = [s.name for s in results]
        assert "postgres" in names

    def test_search_by_capability(self, registry):
        """Test searching for servers by capability/tool."""
        results = registry.search("query")
        assert len(results) > 0
        # Both sqlite and postgres should have query-related tools
        names = [s.name for s in results]
        assert any("sql" in name or "postgres" in name for name in names)

    def test_search_by_category(self, registry):
        """Test searching for servers by category name."""
        results = registry.search("database")
        assert len(results) > 0
        for server in results:
            # Either matches category or description
            has_db_category = ServerCategory.DATABASE in server.categories
            has_db_in_desc = "database" in server.description.lower()
            assert has_db_category or has_db_in_desc

    def test_search_no_results(self, registry):
        """Test search that returns no results."""
        results = registry.search("xyznonexistent123")
        assert len(results) == 0

    def test_find_by_tool(self, registry):
        """Test finding servers that provide a specific tool."""
        results = registry.find_by_tool("fetch")
        assert len(results) > 0
        # At least the fetch server should have the fetch tool
        names = [s.name for s in results]
        assert "fetch" in names

    def test_get_required_env(self, registry):
        """Test getting required environment variables."""
        env_vars = registry.get_required_env("postgres")
        assert "POSTGRES_CONNECTION_STRING" in env_vars

    def test_get_required_env_nonexistent(self, registry):
        """Test getting env for nonexistent server."""
        env_vars = registry.get_required_env("nonexistent")
        assert env_vars == []

    def test_format_server_info(self, registry):
        """Test formatting server information."""
        server = registry.get("postgres")
        assert server is not None

        info = registry.format_server_info(server)
        assert "postgres" in info
        assert "PostgreSQL" in info or "database" in info.lower()
        assert "Docker" in info

    def test_format_catalog(self, registry):
        """Test formatting the entire catalog."""
        catalog = registry.format_catalog()
        assert "Available MCP Servers" in catalog
        assert "Built-in" in catalog
        assert "Docker" in catalog


class TestRegistryCatalogContent:
    """Tests to verify the catalog contains expected servers."""

    @pytest.fixture
    def registry(self):
        """Get registry instance."""
        import kestrel_feature_mcp.registry as reg_module
        reg_module._registry = None
        return get_registry()

    def test_has_builtin_filesystem(self, registry):
        """Verify filesystem built-in server exists."""
        server = registry.get("filesystem")
        assert server is not None
        assert server.is_builtin is True
        assert "read_file" in server.tools

    def test_has_builtin_bash(self, registry):
        """Verify bash built-in server exists."""
        server = registry.get("bash")
        assert server is not None
        assert server.is_builtin is True
        assert "execute" in server.tools

    def test_has_fetch_server(self, registry):
        """Verify fetch Docker server exists."""
        server = registry.get("fetch")
        assert server is not None
        assert server.server_type == ServerType.DOCKER
        assert server.transport == TransportType.STDIO  # Uses stdio, needs wrapper
        assert server.image is not None
        assert "mcp/fetch" in server.image

    def test_has_test_server_sse(self, registry):
        """Verify test-server has SSE transport."""
        server = registry.get("test-server")
        assert server is not None
        assert server.transport == TransportType.SSE  # Native SSE support

    def test_has_sqlite_server(self, registry):
        """Verify sqlite Docker server exists."""
        server = registry.get("sqlite")
        assert server is not None
        assert server.server_type == ServerType.DOCKER
        assert "read_query" in server.tools or "query" in server.tools

    def test_has_memory_server(self, registry):
        """Verify memory Docker server exists."""
        server = registry.get("memory")
        assert server is not None
        assert server.server_type == ServerType.DOCKER
        assert "store" in server.tools

    def test_has_private_fhir_server(self, registry):
        """Verify private FHIR server entry exists."""
        server = registry.get("fhir")
        assert server is not None
        assert server.private is True
        assert server.registry == "gcr.io/YOUR_PROJECT_ID"
        assert ServerCategory.HEALTHCARE in server.categories

    def test_minimum_server_count(self, registry):
        """Verify we have a reasonable number of servers defined."""
        all_servers = registry.list_all()
        assert len(all_servers) >= 10, "Expected at least 10 servers in catalog"

    def test_all_docker_servers_have_images(self, registry):
        """Verify all Docker servers have image defined."""
        docker_servers = registry.list_docker()
        for server in docker_servers:
            assert server.image is not None, f"Server {server.name} missing image"

    def test_all_servers_have_descriptions(self, registry):
        """Verify all servers have descriptions."""
        for server in registry.list_all():
            assert server.description, f"Server {server.name} missing description"

    def test_all_servers_have_tools(self, registry):
        """Verify all servers have at least one tool defined."""
        for server in registry.list_all():
            assert len(server.tools) > 0, f"Server {server.name} has no tools"
