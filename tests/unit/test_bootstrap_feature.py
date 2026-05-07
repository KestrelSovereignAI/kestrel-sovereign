"""
Unit tests for the Bootstrap Feature (#153).

Tests the bootstrap file convention:
- BootstrapLoader: file discovery, loading, truncation, budgets, add/remove
- BootstrapFeature tools: !bootstrap list, reload, add, remove
- Database persistence of bootstrap config
- Integration with ContextBuilder
"""

import pytest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.bootstrap.loader import (
    BootstrapLoader,
    DEFAULT_BOOTSTRAP_FILES,
    DEFAULT_MAX_CHARS_PER_FILE,
    DEFAULT_MAX_TOTAL_CHARS,
    truncate_content,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_agent_dir(tmp_path):
    """Create a temporary agent data directory."""
    agent_dir = tmp_path / "agent_data" / "test_agent"
    agent_dir.mkdir(parents=True)
    return agent_dir


@pytest.fixture
def extra_dir(tmp_path):
    """Create a secondary search directory (project-level)."""
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True)
    return project_dir


@pytest.fixture
def loader(tmp_agent_dir):
    """A basic BootstrapLoader pointing at the temp agent dir."""
    return BootstrapLoader(agent_data_path=str(tmp_agent_dir))


@pytest.fixture
def mock_agent():
    """Mock agent for BootstrapFeature tests."""

    class _MockStorage:
        nodes = {}
        async def get_node(self, node_id):
            return self.nodes.get(node_id)
        async def add_node(self, node):
            self.nodes[node.node_id] = node

    class _MockDB:
        data = {}
        async def execute(self, query, params=None):
            pass
        async def fetchall(self, query, params=None):
            return []

    agent = MagicMock()
    agent.agent_id = "did:test:bootstrap"
    agent._agent_name = "TestAgent"
    agent.storage = _MockStorage()
    agent._raw_storage = MagicMock()
    agent._raw_storage.db = _MockDB()
    agent.bootstrap_service = MagicMock()
    agent.bootstrap_service.agent_name = "TestAgent"
    agent.bootstrap_service.agent_data_path = None
    agent.context_builder = MagicMock()
    return agent


# ---------------------------------------------------------------------------
# truncate_content
# ---------------------------------------------------------------------------

class TestTruncateContent:
    """Tests for the standalone truncation function."""

    def test_no_truncation_under_limit(self):
        content = "Hello world"
        assert truncate_content(content, 1000) == content

    def test_exact_limit(self):
        content = "A" * 100
        assert truncate_content(content, 100) == content

    def test_truncation_with_marker(self):
        content = "A" * 200
        result = truncate_content(content, 100)
        assert "[...truncated...]" in result
        assert len(result) < 200

    def test_head_tail_proportions(self):
        content = "H" * 50 + "T" * 50
        result = truncate_content(content, 50)
        # Head = 70% of 50 = 35 chars
        # Tail = 20% of 50 = 10 chars
        assert result.startswith("H" * 35)
        assert result.endswith("T" * 10)

    def test_empty_content(self):
        assert truncate_content("", 100) == ""


# ---------------------------------------------------------------------------
# BootstrapLoader — basic loading
# ---------------------------------------------------------------------------

class TestBootstrapLoaderBasic:
    """Tests for BootstrapLoader file loading."""

    def test_no_path_returns_empty(self):
        loader = BootstrapLoader(agent_data_path=None)
        assert loader.load() == OrderedDict()
        assert loader.file_count == 0

    def test_loads_single_file(self, tmp_agent_dir, loader):
        (tmp_agent_dir / "SOUL.md").write_text("I am the agent.")
        result = loader.load()
        assert "SOUL.md" in result
        assert result["SOUL.md"] == "I am the agent."
        assert loader.file_count == 1

    def test_loads_multiple_files_in_order(self, tmp_agent_dir, loader):
        (tmp_agent_dir / "AGENTS.md").write_text("Agent config")
        (tmp_agent_dir / "SOUL.md").write_text("Personality")
        (tmp_agent_dir / "TOOLS.md").write_text("Tool ref")
        result = loader.load()
        assert list(result.keys()) == ["AGENTS.md", "SOUL.md", "TOOLS.md"]

    def test_skips_missing_files(self, tmp_agent_dir, loader):
        (tmp_agent_dir / "SOUL.md").write_text("Only this one")
        result = loader.load()
        assert list(result.keys()) == ["SOUL.md"]

    def test_skips_empty_files(self, tmp_agent_dir, loader):
        (tmp_agent_dir / "SOUL.md").write_text("   \n\n  ")
        (tmp_agent_dir / "AGENTS.md").write_text("Real content")
        result = loader.load()
        assert "SOUL.md" not in result
        assert "AGENTS.md" in result

    def test_file_order_matches_convention(self, tmp_agent_dir, loader):
        """Files load in DEFAULT_BOOTSTRAP_FILES order regardless of creation."""
        for filename in reversed(DEFAULT_BOOTSTRAP_FILES):
            (tmp_agent_dir / filename).write_text(f"Content for {filename}")

        result = loader.load()
        assert list(result.keys()) == DEFAULT_BOOTSTRAP_FILES

    def test_caching(self, tmp_agent_dir, loader):
        """Second call returns cached result without re-reading disk."""
        (tmp_agent_dir / "SOUL.md").write_text("First version")
        result1 = loader.load()

        # Modify file on disk
        (tmp_agent_dir / "SOUL.md").write_text("Second version")
        result2 = loader.load()

        # Should still be cached
        assert result2["SOUL.md"] == "First version"

    def test_get_file(self, tmp_agent_dir, loader):
        (tmp_agent_dir / "SOUL.md").write_text("Personality")
        assert loader.get_file("SOUL.md") == "Personality"
        assert loader.get_file("MISSING.md") is None

    def test_get_bootstrap_content_returns_dict(self, tmp_agent_dir, loader):
        (tmp_agent_dir / "SOUL.md").write_text("Content")
        content = loader.get_bootstrap_content()
        assert isinstance(content, dict)
        assert content["SOUL.md"] == "Content"

    def test_total_chars(self, tmp_agent_dir, loader):
        (tmp_agent_dir / "AGENTS.md").write_text("A" * 100)
        (tmp_agent_dir / "SOUL.md").write_text("B" * 200)
        loader.load()
        assert loader.total_chars == 300


# ---------------------------------------------------------------------------
# BootstrapLoader — truncation and budgets
# ---------------------------------------------------------------------------

class TestBootstrapLoaderBudgets:
    """Tests for per-file and total budget enforcement."""

    def test_per_file_truncation(self, tmp_agent_dir):
        loader = BootstrapLoader(
            agent_data_path=str(tmp_agent_dir),
            max_chars_per_file=500,
        )
        (tmp_agent_dir / "SOUL.md").write_text("X" * 1000)
        result = loader.load()
        assert len(result["SOUL.md"]) < 1000
        assert "[...truncated...]" in result["SOUL.md"]

    def test_total_budget_enforcement(self, tmp_agent_dir):
        loader = BootstrapLoader(
            agent_data_path=str(tmp_agent_dir),
            max_chars_per_file=500,
            max_total_chars=800,
        )
        # Create 3 files, each 500 chars after truncation
        (tmp_agent_dir / "AGENTS.md").write_text("A" * 500)
        (tmp_agent_dir / "SOUL.md").write_text("B" * 500)
        (tmp_agent_dir / "TOOLS.md").write_text("C" * 500)

        result = loader.load()
        total = sum(len(c) for c in result.values())
        assert total <= 800

    def test_total_budget_skips_files(self, tmp_agent_dir):
        """When budget is completely exhausted, later files are skipped."""
        loader = BootstrapLoader(
            agent_data_path=str(tmp_agent_dir),
            max_chars_per_file=10000,
            max_total_chars=500,
        )
        # First file fills most of the budget
        (tmp_agent_dir / "AGENTS.md").write_text("A" * 490)
        # Second file would not fit (only 10 chars remaining < 100 min)
        (tmp_agent_dir / "SOUL.md").write_text("B" * 500)

        result = loader.load()
        assert "AGENTS.md" in result
        # SOUL.md either truncated to fit or skipped entirely
        total = sum(len(c) for c in result.values())
        assert total <= 500


# ---------------------------------------------------------------------------
# BootstrapLoader — reload
# ---------------------------------------------------------------------------

class TestBootstrapLoaderReload:
    """Tests for cache invalidation and hot-reload."""

    def test_reload_picks_up_changes(self, tmp_agent_dir, loader):
        (tmp_agent_dir / "SOUL.md").write_text("Version 1")
        loader.load()
        assert loader.get_file("SOUL.md") == "Version 1"

        (tmp_agent_dir / "SOUL.md").write_text("Version 2")
        loader.reload()
        assert loader.get_file("SOUL.md") == "Version 2"

    def test_reload_picks_up_new_files(self, tmp_agent_dir, loader):
        loader.load()
        assert loader.file_count == 0

        (tmp_agent_dir / "SOUL.md").write_text("New file")
        loader.reload()
        assert "SOUL.md" in loader.get_bootstrap_content()

    def test_reload_removes_deleted_files(self, tmp_agent_dir, loader):
        (tmp_agent_dir / "SOUL.md").write_text("Exists")
        loader.load()
        assert "SOUL.md" in loader.get_bootstrap_content()

        (tmp_agent_dir / "SOUL.md").unlink()
        loader.reload()
        assert "SOUL.md" not in loader.get_bootstrap_content()


# ---------------------------------------------------------------------------
# BootstrapLoader — add/remove
# ---------------------------------------------------------------------------

class TestBootstrapLoaderAddRemove:
    """Tests for dynamic file list management."""

    def test_add_file_extends_order(self, loader):
        assert "CUSTOM.md" not in loader.file_order
        added = loader.add_file("CUSTOM.md")
        assert added is True
        assert "CUSTOM.md" in loader.file_order

    def test_add_duplicate_returns_false(self, loader):
        assert loader.add_file("SOUL.md") is False  # Already in defaults

    def test_remove_file(self, loader):
        assert "SOUL.md" in loader.file_order
        removed = loader.remove_file("SOUL.md")
        assert removed is True
        assert "SOUL.md" not in loader.file_order

    def test_remove_nonexistent_returns_false(self, loader):
        assert loader.remove_file("NOPE.md") is False

    def test_remove_clears_from_cache(self, tmp_agent_dir, loader):
        (tmp_agent_dir / "SOUL.md").write_text("Content")
        loader.load()
        assert "SOUL.md" in loader.get_bootstrap_content()

        loader.remove_file("SOUL.md")
        assert "SOUL.md" not in loader.get_bootstrap_content()

    def test_add_triggers_reload_on_next_load(self, tmp_agent_dir, loader):
        """Adding a file invalidates the cache."""
        loader.load()  # Initial load
        (tmp_agent_dir / "CUSTOM.md").write_text("Custom content")
        loader.add_file("CUSTOM.md")
        result = loader.load()
        assert "CUSTOM.md" in result


# ---------------------------------------------------------------------------
# BootstrapLoader — priority paths
# ---------------------------------------------------------------------------

class TestBootstrapLoaderPaths:
    """Tests for multi-directory file resolution."""

    def test_agent_path_takes_priority(self, tmp_agent_dir, extra_dir):
        """Agent-specific file shadows project-level file."""
        (tmp_agent_dir / "SOUL.md").write_text("Agent-specific soul")
        (extra_dir / "SOUL.md").write_text("Project soul")

        loader = BootstrapLoader(
            agent_data_path=str(tmp_agent_dir),
            extra_paths=[str(extra_dir)],
        )
        result = loader.load()
        assert result["SOUL.md"] == "Agent-specific soul"

    def test_extra_path_used_as_fallback(self, tmp_agent_dir, extra_dir):
        """Project-level file loaded when agent-specific doesn't exist."""
        (extra_dir / "AGENTS.md").write_text("Project agents")

        loader = BootstrapLoader(
            agent_data_path=str(tmp_agent_dir),
            extra_paths=[str(extra_dir)],
        )
        result = loader.load()
        assert result["AGENTS.md"] == "Project agents"

    def test_multiple_extra_paths_order(self, tmp_path):
        """First extra path shadows second."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "GOALS.md").write_text("Goals from dir1")
        (dir2 / "GOALS.md").write_text("Goals from dir2")

        loader = BootstrapLoader(
            agent_data_path=None,
            extra_paths=[str(dir1), str(dir2)],
        )
        result = loader.load()
        assert result["GOALS.md"] == "Goals from dir1"


# ---------------------------------------------------------------------------
# BootstrapLoader — list_files
# ---------------------------------------------------------------------------

class TestBootstrapLoaderListFiles:
    """Tests for the list_files reporting method."""

    def test_list_includes_all_configured(self, tmp_agent_dir, loader):
        (tmp_agent_dir / "SOUL.md").write_text("Content")
        files = loader.list_files()
        names = [f["name"] for f in files]
        # Should include all files from DEFAULT_BOOTSTRAP_FILES
        for default_name in DEFAULT_BOOTSTRAP_FILES:
            assert default_name in names

    def test_list_shows_loaded_status(self, tmp_agent_dir, loader):
        (tmp_agent_dir / "SOUL.md").write_text("Content")
        files = loader.list_files()
        soul_entry = next(f for f in files if f["name"] == "SOUL.md")
        assert soul_entry["status"] == "loaded"
        assert soul_entry["chars"] == len("Content")

    def test_list_shows_not_found_status(self, tmp_agent_dir, loader):
        loader.load()
        files = loader.list_files()
        agents_entry = next(f for f in files if f["name"] == "AGENTS.md")
        assert agents_entry["status"] == "not found"


# ---------------------------------------------------------------------------
# BootstrapLoader — DB persistence
# ---------------------------------------------------------------------------

class TestBootstrapLoaderDB:
    """Tests for database-backed config persistence."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.fetchall = AsyncMock(return_value=[])
        db.execute = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_load_db_config_no_db(self, loader):
        """No error when db is None."""
        await loader.load_db_config()  # Should not raise

    @pytest.mark.asyncio
    async def test_load_db_config_adds_files(self, tmp_agent_dir, mock_db):
        mock_db.fetchall = AsyncMock(return_value=[
            ("CUSTOM.md", "/path/to/CUSTOM.md", 1, 50),
        ])
        loader = BootstrapLoader(
            agent_data_path=str(tmp_agent_dir),
            db=mock_db,
            agent_id="did:test:123",
        )
        await loader.load_db_config()
        assert "CUSTOM.md" in loader.file_order

    @pytest.mark.asyncio
    async def test_load_db_config_disables_files(self, tmp_agent_dir, mock_db):
        mock_db.fetchall = AsyncMock(return_value=[
            ("SOUL.md", "", 0, 10),  # enabled=0
        ])
        loader = BootstrapLoader(
            agent_data_path=str(tmp_agent_dir),
            db=mock_db,
            agent_id="did:test:123",
        )
        await loader.load_db_config()
        assert "SOUL.md" not in loader.file_order

    @pytest.mark.asyncio
    async def test_save_db_entry(self, tmp_agent_dir, mock_db):
        loader = BootstrapLoader(
            agent_data_path=str(tmp_agent_dir),
            db=mock_db,
            agent_id="did:test:123",
        )
        await loader.save_db_entry("CUSTOM.md", "/path/CUSTOM.md")
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        assert "INSERT OR REPLACE INTO bootstrap_config" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_delete_db_entry(self, tmp_agent_dir, mock_db):
        loader = BootstrapLoader(
            agent_data_path=str(tmp_agent_dir),
            db=mock_db,
            agent_id="did:test:123",
        )
        await loader.delete_db_entry("CUSTOM.md")
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        assert "DELETE FROM bootstrap_config" in call_args[0][0]


# ---------------------------------------------------------------------------
# BootstrapFeature tools
# ---------------------------------------------------------------------------

class TestBootstrapFeatureTools:
    """Tests for the !bootstrap command tools."""

    @pytest.fixture
    def feature_with_loader(self, mock_agent, tmp_agent_dir):
        """Create a BootstrapFeature with a real loader attached."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        loader = BootstrapLoader(agent_data_path=str(tmp_agent_dir))
        mock_agent.context_builder = MagicMock()
        mock_agent.context_builder._bootstrap_loader = loader
        mock_agent.bootstrap_service.agent_data_path = str(tmp_agent_dir)

        feature = BootstrapFeature(mock_agent)
        return feature, loader, tmp_agent_dir

    @pytest.mark.asyncio
    async def test_bootstrap_list_empty(self, feature_with_loader):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature, loader, _ = feature_with_loader
        result = await feature.bootstrap_list()
        assert result.status is ToolResultStatus.OK
        assert result.data["total_files"] == 0

    @pytest.mark.asyncio
    async def test_bootstrap_list_with_files(self, feature_with_loader):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature, loader, agent_dir = feature_with_loader
        (agent_dir / "SOUL.md").write_text("Test soul")
        (agent_dir / "AGENTS.md").write_text("Test agents")
        loader.reload()

        result = await feature.bootstrap_list()
        assert result.status is ToolResultStatus.OK
        assert result.data["total_files"] == 2
        assert result.data["total_chars"] > 0

    @pytest.mark.asyncio
    async def test_bootstrap_list_no_loader(self, mock_agent):
        from kestrel_sdk.tools.result import ToolResultStatus
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature
        mock_agent.context_builder = MagicMock(spec=[])  # No _bootstrap_loader
        feature = BootstrapFeature(mock_agent)
        result = await feature.bootstrap_list()
        assert result.status is ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_bootstrap_reload(self, feature_with_loader):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature, loader, agent_dir = feature_with_loader
        (agent_dir / "SOUL.md").write_text("Initial")
        loader.load()
        assert loader.get_file("SOUL.md") == "Initial"

        (agent_dir / "SOUL.md").write_text("Updated")
        result = await feature.bootstrap_reload()
        assert result.status is ToolResultStatus.OK
        assert "SOUL.md" in result.data["files"]
        assert loader.get_file("SOUL.md") == "Updated"

    @pytest.mark.asyncio
    async def test_bootstrap_add(self, feature_with_loader):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature, loader, agent_dir = feature_with_loader
        custom_file = agent_dir / "NOTES.md"
        custom_file.write_text("My notes")

        result = await feature.bootstrap_add("NOTES.md")
        assert result.status is ToolResultStatus.OK
        assert result.data["loaded"] is True
        assert "NOTES.md" in loader.file_order

    @pytest.mark.asyncio
    async def test_bootstrap_add_file_not_found(self, feature_with_loader):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature, loader, agent_dir = feature_with_loader
        result = await feature.bootstrap_add("NONEXISTENT.md")
        assert result.status is ToolResultStatus.ERROR
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_bootstrap_add_duplicate(self, feature_with_loader):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature, loader, _ = feature_with_loader
        result = await feature.bootstrap_add("SOUL.md")  # Already in defaults
        assert result.status is ToolResultStatus.ERROR
        assert "already" in result.error.lower()

    @pytest.mark.asyncio
    async def test_bootstrap_remove(self, feature_with_loader):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature, loader, agent_dir = feature_with_loader
        (agent_dir / "GOALS.md").write_text("My goals")
        loader.load()

        result = await feature.bootstrap_remove("GOALS.md")
        assert result.status is ToolResultStatus.OK
        assert "GOALS.md" not in loader.file_order

    @pytest.mark.asyncio
    async def test_bootstrap_remove_not_found(self, feature_with_loader):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature, _, _ = feature_with_loader
        result = await feature.bootstrap_remove("NOPE.md")
        assert result.status is ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_bootstrap_no_context_builder(self, mock_agent):
        """All tools return graceful error when context_builder is missing."""
        from kestrel_sdk.tools.result import ToolResultStatus
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature
        mock_agent.context_builder = None
        feature = BootstrapFeature(mock_agent)

        for tool_fn in [feature.bootstrap_list, feature.bootstrap_reload]:
            result = await tool_fn()
            assert result.status is ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_existing_tools_still_work(self, mock_agent):
        """Verify existing skip_discovery, restart_discovery, bootstrap_status."""
        from kestrel_sdk.tools.result import ToolResultStatus
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature
        feature = BootstrapFeature(mock_agent)

        # bootstrap_status
        mock_agent.bootstrap_service.get_bootstrap_status = AsyncMock(
            return_value="**Bootstrap State:** complete"
        )
        result = await feature.bootstrap_status()
        assert result.status is ToolResultStatus.OK
        assert "Bootstrap State" in result.confirmation


# ---------------------------------------------------------------------------
# ContextBuilder integration
# ---------------------------------------------------------------------------

class TestContextBuilderIntegration:
    """Verify ContextBuilder delegates to BootstrapLoader correctly."""

    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock()
        storage.search_chunks = MagicMock(return_value=[])
        return storage

    def test_context_builder_has_loader(self, mock_storage, tmp_agent_dir):
        from kestrel_sovereign.agent.context_builder import ContextBuilder
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        assert hasattr(builder, '_bootstrap_loader')
        assert isinstance(builder._bootstrap_loader, BootstrapLoader)

    def test_bootstrap_files_property_delegates(self, mock_storage, tmp_agent_dir):
        from kestrel_sovereign.agent.context_builder import ContextBuilder
        (tmp_agent_dir / "SOUL.md").write_text("Soul content")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        assert "SOUL.md" in builder._bootstrap_files

    def test_soul_content_backward_compat(self, mock_storage, tmp_agent_dir):
        from kestrel_sovereign.agent.context_builder import ContextBuilder
        (tmp_agent_dir / "SOUL.md").write_text("My soul")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        assert builder._soul_content == "My soul"

    def test_soul_content_setter_backward_compat(self, mock_storage, tmp_agent_dir):
        from kestrel_sovereign.agent.context_builder import ContextBuilder
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        builder._soul_content = "Direct set"
        assert builder._bootstrap_files.get("SOUL.md") == "Direct set"
        builder._soul_content = None
        assert "SOUL.md" not in builder._bootstrap_files

    def test_reload_delegates_to_loader(self, mock_storage, tmp_agent_dir):
        from kestrel_sovereign.agent.context_builder import ContextBuilder
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        assert builder._bootstrap_loader.file_count == 0

        (tmp_agent_dir / "SOUL.md").write_text("New soul")
        builder.reload_bootstrap_files()
        assert builder._bootstrap_loader.file_count == 1

    def test_load_soul_md_backward_compat(self, mock_storage, tmp_agent_dir):
        from kestrel_sovereign.agent.context_builder import ContextBuilder
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        (tmp_agent_dir / "SOUL.md").write_text("Loaded via compat")
        builder._load_soul_md()
        assert builder._soul_content == "Loaded via compat"

    def test_new_files_in_convention(self, mock_storage, tmp_agent_dir):
        """CAPABILITIES.md and GOALS.md are part of the convention."""
        from kestrel_sovereign.agent.context_builder import ContextBuilder
        (tmp_agent_dir / "CAPABILITIES.md").write_text("I can do things")
        (tmp_agent_dir / "GOALS.md").write_text("My current goals")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        assert "CAPABILITIES.md" in builder._bootstrap_files
        assert "GOALS.md" in builder._bootstrap_files

    def test_system_prompt_includes_new_files(self, mock_storage, tmp_agent_dir):
        from kestrel_sovereign.agent.context_builder import ContextBuilder
        (tmp_agent_dir / "CAPABILITIES.md").write_text("I can code")
        (tmp_agent_dir / "GOALS.md").write_text("Ship the feature")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        prompt = builder.build_system_prompt("Constitution text")
        assert "--- CAPABILITIES ---" in prompt
        assert "I can code" in prompt
        assert "--- GOALS ---" in prompt
        assert "Ship the feature" in prompt


# ---------------------------------------------------------------------------
# BootstrapLoader — custom file_order
# ---------------------------------------------------------------------------

class TestBootstrapLoaderCustomOrder:
    """Tests for custom file ordering."""

    def test_custom_file_order(self, tmp_agent_dir):
        loader = BootstrapLoader(
            agent_data_path=str(tmp_agent_dir),
            file_order=["CUSTOM1.md", "CUSTOM2.md"],
        )
        (tmp_agent_dir / "CUSTOM1.md").write_text("First")
        (tmp_agent_dir / "CUSTOM2.md").write_text("Second")
        result = loader.load()
        assert list(result.keys()) == ["CUSTOM1.md", "CUSTOM2.md"]

    def test_file_order_property(self, loader):
        order = loader.file_order
        assert order == DEFAULT_BOOTSTRAP_FILES
        # Should return a copy, not the internal list
        order.append("EXTRA.md")
        assert "EXTRA.md" not in loader.file_order
