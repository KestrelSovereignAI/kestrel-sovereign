"""
Tests for the bootstrap file convention (#153).

Verifies that ContextBuilder loads AGENTS.md, SOUL.md, TOOLS.md, etc.
from the agent data directory with truncation and budget enforcement.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock

from kestrel_sovereign.agent.context_builder import (
    ContextBuilder,
    BOOTSTRAP_FILE_ORDER,
    DEFAULT_MAX_CHARS_PER_FILE,
    DEFAULT_MAX_TOTAL_CHARS,
    truncate_bootstrap_content,
)


@pytest.fixture
def tmp_agent_dir(tmp_path):
    """Create a temporary agent data directory."""
    return tmp_path


@pytest.fixture
def mock_storage():
    storage = Mock()
    storage.search_chunks = Mock(return_value=[])
    return storage


class TestTruncateBootstrapContent:
    """Tests for the truncation utility."""

    def test_no_truncation_needed(self):
        content = "Short content"
        assert truncate_bootstrap_content(content, 1000) == content

    def test_truncation_preserves_head_and_tail(self):
        content = "A" * 100
        result = truncate_bootstrap_content(content, 50)
        assert "[...truncated...]" in result
        assert len(result) < 100
        # Head should be ~70% of 50 = 35 chars of 'A'
        assert result.startswith("A" * 30)
        # Tail should be ~20% of 50 = 10 chars of 'A'
        assert result.endswith("A" * 10)

    def test_exact_limit_no_truncation(self):
        content = "X" * 100
        assert truncate_bootstrap_content(content, 100) == content

    def test_empty_content(self):
        assert truncate_bootstrap_content("", 100) == ""


class TestBootstrapFileLoading:
    """Tests for loading bootstrap files from the agent data directory."""

    def test_loads_soul_md(self, mock_storage, tmp_agent_dir):
        """SOUL.md should be loaded and accessible via backward-compat property."""
        (tmp_agent_dir / "SOUL.md").write_text("I am a test agent.")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        assert builder._soul_content == "I am a test agent."
        assert "SOUL.md" in builder._bootstrap_files

    def test_loads_multiple_files(self, mock_storage, tmp_agent_dir):
        """Multiple bootstrap files loaded in defined order."""
        (tmp_agent_dir / "AGENTS.md").write_text("Agent instructions here")
        (tmp_agent_dir / "SOUL.md").write_text("Personality")
        (tmp_agent_dir / "TOOLS.md").write_text("Tool reference")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        assert list(builder._bootstrap_files.keys()) == [
            "AGENTS.md", "SOUL.md", "TOOLS.md"
        ]

    def test_skips_missing_files(self, mock_storage, tmp_agent_dir):
        """Only existing files are loaded."""
        (tmp_agent_dir / "SOUL.md").write_text("Personality")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        assert list(builder._bootstrap_files.keys()) == ["SOUL.md"]
        assert "AGENTS.md" not in builder._bootstrap_files

    def test_skips_empty_files(self, mock_storage, tmp_agent_dir):
        """Empty/whitespace-only files are skipped."""
        (tmp_agent_dir / "SOUL.md").write_text("  \n\n  ")
        (tmp_agent_dir / "AGENTS.md").write_text("Real content")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        assert "SOUL.md" not in builder._bootstrap_files
        assert "AGENTS.md" in builder._bootstrap_files

    def test_per_file_truncation(self, mock_storage, tmp_agent_dir):
        """Files exceeding max_chars_per_file are truncated."""
        big_content = "X" * 30_000
        (tmp_agent_dir / "SOUL.md").write_text(big_content)
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        loaded = builder._bootstrap_files["SOUL.md"]
        assert len(loaded) < 30_000
        assert "[...truncated...]" in loaded

    def test_total_budget_enforcement(self, mock_storage, tmp_agent_dir):
        """Files are skipped when total budget is exhausted."""
        # Create files that collectively exceed the default total budget
        for filename in BOOTSTRAP_FILE_ORDER:
            # Each file ~25k chars, 8 files = 200k > 150k budget
            (tmp_agent_dir / filename).write_text("Y" * 25_000)

        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        total_chars = sum(len(c) for c in builder._bootstrap_files.values())
        assert total_chars <= DEFAULT_MAX_TOTAL_CHARS

    def test_no_agent_data_path(self, mock_storage):
        """No crash when agent_data_path is None."""
        builder = ContextBuilder(mock_storage, agent_data_path=None)
        assert len(builder._bootstrap_files) == 0

    def test_reload_bootstrap_files(self, mock_storage, tmp_agent_dir):
        """Hot-reload picks up new files."""
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        assert len(builder._bootstrap_files) == 0

        (tmp_agent_dir / "SOUL.md").write_text("New personality")
        builder.reload_bootstrap_files()
        assert "SOUL.md" in builder._bootstrap_files
        assert builder._soul_content == "New personality"

    def test_backward_compat_soul_setter(self, mock_storage, tmp_agent_dir):
        """Setting _soul_content updates _bootstrap_files."""
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        builder._soul_content = "Direct set"
        assert builder._bootstrap_files.get("SOUL.md") == "Direct set"

        builder._soul_content = None
        assert "SOUL.md" not in builder._bootstrap_files

    def test_load_order_preserved(self, mock_storage, tmp_agent_dir):
        """Files load in BOOTSTRAP_FILE_ORDER regardless of filesystem order."""
        # Create files in reverse order
        for filename in reversed(BOOTSTRAP_FILE_ORDER):
            (tmp_agent_dir / filename).write_text(f"Content for {filename}")

        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        loaded_names = list(builder._bootstrap_files.keys())
        assert loaded_names == BOOTSTRAP_FILE_ORDER


class TestBuildSystemPromptWithBootstrap:
    """Tests for build_system_prompt with bootstrap files."""

    def test_soul_md_identity_wrapper(self, mock_storage, tmp_agent_dir):
        """SOUL.md content wrapped in --- YOUR IDENTITY ---."""
        (tmp_agent_dir / "SOUL.md").write_text("I am Claw.")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        prompt = builder.build_system_prompt("Constitution text")
        assert "--- YOUR IDENTITY ---" in prompt
        assert "I am Claw." in prompt
        assert "--- END IDENTITY ---" in prompt

    def test_agents_md_generic_wrapper(self, mock_storage, tmp_agent_dir):
        """Non-SOUL files get filename-based wrappers."""
        (tmp_agent_dir / "AGENTS.md").write_text("Agent config")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        prompt = builder.build_system_prompt("Constitution text")
        assert "--- AGENTS ---" in prompt
        assert "Agent config" in prompt
        assert "--- END AGENTS ---" in prompt

    def test_heartbeat_md_excluded_from_prompt(self, mock_storage, tmp_agent_dir):
        """HEARTBEAT.md should NOT appear in the normal system prompt."""
        (tmp_agent_dir / "HEARTBEAT.md").write_text("Check stuff")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        prompt = builder.build_system_prompt("Constitution text")
        assert "Check stuff" not in prompt

    def test_multiple_files_ordering(self, mock_storage, tmp_agent_dir):
        """AGENTS.md appears before SOUL.md in the system prompt."""
        (tmp_agent_dir / "AGENTS.md").write_text("AGENTS_MARKER")
        (tmp_agent_dir / "SOUL.md").write_text("SOUL_MARKER")
        (tmp_agent_dir / "USER.md").write_text("USER_MARKER")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        prompt = builder.build_system_prompt("Constitution text")
        agents_pos = prompt.index("AGENTS_MARKER")
        soul_pos = prompt.index("SOUL_MARKER")
        user_pos = prompt.index("USER_MARKER")
        assert agents_pos < soul_pos < user_pos

    def test_no_bootstrap_files_still_works(self, mock_storage, tmp_agent_dir):
        """System prompt works fine with no bootstrap files."""
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        prompt = builder.build_system_prompt("Constitution text")
        assert "--- GOVERNING CONSTITUTION ---" in prompt
        assert "Constitution text" in prompt

    def test_style_reminder_with_soul(self, mock_storage, tmp_agent_dir):
        """Style reminder included when SOUL.md exists."""
        (tmp_agent_dir / "SOUL.md").write_text("Personality here")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        prompt = builder.build_system_prompt("Constitution text")
        assert "STYLE REMINDER" in prompt

    def test_no_style_reminder_without_soul(self, mock_storage, tmp_agent_dir):
        """No style reminder when SOUL.md is absent."""
        (tmp_agent_dir / "AGENTS.md").write_text("Just agents")
        builder = ContextBuilder(mock_storage, agent_data_path=str(tmp_agent_dir))
        prompt = builder.build_system_prompt("Constitution text")
        assert "STYLE REMINDER" not in prompt
