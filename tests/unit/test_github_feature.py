"""Unit tests for GitHub feature."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import base64

from kestrel_sovereign.features.github.models import (
    FileContent,
    FileType,
    RepoFile,
    SearchResult,
    CodeDefinition,
    ComponentManifest,
)
from kestrel_sovereign.features.github.client import GitHubClient, GitHubClientError
from kestrel_sovereign.features.github.cache import GitHubCache
from kestrel_sovereign.features.github.ast_analyzer import ASTAnalyzer, extract_definition, list_definitions
from kestrel_sovereign.features.github.feature import GitHubFeature


# ============== Model Tests ==============

class TestModels:
    """Tests for data models."""
    
    def test_repo_file_is_dir(self):
        f = RepoFile(path="src", name="src", type=FileType.DIR)
        assert f.is_dir()
        assert not f.is_file()
    
    def test_repo_file_is_file(self):
        f = RepoFile(path="main.py", name="main.py", type=FileType.FILE, size=100)
        assert f.is_file()
        assert not f.is_dir()
    
    def test_file_content(self):
        fc = FileContent(
            path="test.py",
            content="print('hello')",
            sha="abc123",
            size=14,
            repo="owner/repo",
            ref="main",
        )
        assert fc.path == "test.py"
        assert fc.content == "print('hello')"
    
    def test_component_manifest_from_dict(self):
        data = {
            "description": "Test feature",
            "version": "2.0.0",
            "tools": ["tool1", "tool2"],
            "files": ["feature.py", "helper.py"],
        }
        manifest = ComponentManifest.from_dict(data, "test")
        assert manifest.feature_name == "test"
        assert manifest.description == "Test feature"
        assert manifest.version == "2.0.0"
        assert manifest.tools == ["tool1", "tool2"]

    def test_component_manifest_from_dict_with_dict_tools(self):
        """Test that tools can be specified as dicts with name/command/description."""
        data = {
            "description": "Council feature",
            "version": "1.0.0",
            "tools": [
                {"name": "council_convene", "command": "!council-convene", "description": "Convene council"},
                {"name": "council_status", "command": "!council-status", "description": "View status"},
            ],
        }
        manifest = ComponentManifest.from_dict(data, "council")
        assert manifest.feature_name == "council"
        # Tools should be normalized to just the names (strings)
        assert manifest.tools == ["council_convene", "council_status"]
        # All tools should be strings, not dicts
        assert all(isinstance(t, str) for t in manifest.tools)

    def test_component_manifest_from_dict_mixed_tools(self):
        """Test mixed format - some string, some dict tools."""
        data = {
            "description": "Mixed feature",
            "tools": [
                "simple_tool",
                {"name": "complex_tool", "description": "A complex tool"},
            ],
        }
        manifest = ComponentManifest.from_dict(data, "mixed")
        assert manifest.tools == ["simple_tool", "complex_tool"]


# ============== AST Analyzer Tests ==============

class TestASTAnalyzer:
    """Tests for AST-based code analysis."""
    
    SAMPLE_CODE = '''
"""Module docstring."""

import os
from typing import Optional

def simple_function():
    """A simple function."""
    pass

def function_with_args(a: int, b: str = "default") -> bool:
    """Function with arguments."""
    return True

async def async_function(data: dict) -> None:
    """Async function."""
    await something()

class MyClass:
    """A sample class."""
    
    def __init__(self, value: int):
        self.value = value
    
    def method(self) -> int:
        """Instance method."""
        return self.value
    
    @classmethod
    def class_method(cls) -> "MyClass":
        """Class method."""
        return cls(0)
'''
    
    def test_parse_valid_code(self):
        analyzer = ASTAnalyzer(self.SAMPLE_CODE, "test.py")
        assert analyzer.parse()
    
    def test_parse_invalid_code(self):
        analyzer = ASTAnalyzer("def broken(", "broken.py")
        assert not analyzer.parse()
    
    def test_get_definitions(self):
        analyzer = ASTAnalyzer(self.SAMPLE_CODE, "test.py")
        defs = analyzer.get_definitions()
        
        names = [d.name for d in defs]
        assert "simple_function" in names
        assert "function_with_args" in names
        assert "async_function" in names
        assert "MyClass" in names
        assert "__init__" in names
        assert "method" in names
    
    def test_get_specific_definition(self):
        analyzer = ASTAnalyzer(self.SAMPLE_CODE, "test.py")
        
        defn = analyzer.get_definition("function_with_args")
        assert defn is not None
        assert defn.name == "function_with_args"
        assert defn.type == "function"
        assert "a: int" in defn.signature
        assert "b: str" in defn.signature
        assert "-> bool" in defn.signature
        assert "Function with arguments" in defn.docstring
    
    def test_get_class_definition(self):
        analyzer = ASTAnalyzer(self.SAMPLE_CODE, "test.py")
        
        defn = analyzer.get_definition("MyClass")
        assert defn is not None
        assert defn.type == "class"
        assert "A sample class" in defn.docstring
    
    def test_get_method_is_method_type(self):
        analyzer = ASTAnalyzer(self.SAMPLE_CODE, "test.py")
        
        defn = analyzer.get_definition("method")
        assert defn is not None
        assert defn.type == "method"
    
    def test_get_imports(self):
        analyzer = ASTAnalyzer(self.SAMPLE_CODE, "test.py")
        imports = analyzer.get_imports()
        
        modules = [i["module"] for i in imports]
        assert "os" in modules
        assert "typing" in modules
    
    def test_extract_definition_convenience(self):
        defn = extract_definition(self.SAMPLE_CODE, "simple_function", "test.py")
        assert defn is not None
        assert defn.name == "simple_function"
    
    def test_list_definitions_convenience(self):
        defs = list_definitions(self.SAMPLE_CODE, "test.py")
        assert len(defs) > 0


# ============== Cache Tests ==============

class TestGitHubCache:
    """Tests for SQLite cache (async methods)."""

    @pytest.fixture
    def cache(self, tmp_path):
        db_path = str(tmp_path / "test_cache.db")
        return GitHubCache(db_path)

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        content = FileContent(
            path="test.py",
            content="print('hello')",
            sha="abc123",
            size=14,
            repo="owner/repo",
            ref="main",
        )

        await cache.set(content)

        retrieved = await cache.get("owner/repo", "test.py", "main")
        assert retrieved is not None
        assert retrieved.content == "print('hello')"
        assert retrieved.sha == "abc123"

    @pytest.mark.asyncio
    async def test_get_missing(self, cache):
        result = await cache.get("owner/repo", "nonexistent.py", "main")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_specific(self, cache):
        content = FileContent(
            path="test.py",
            content="content",
            sha="abc",
            size=7,
            repo="owner/repo",
            ref="main",
        )
        await cache.set(content)

        await cache.invalidate("owner/repo", path="test.py", ref="main")

        assert await cache.get("owner/repo", "test.py", "main") is None

    @pytest.mark.asyncio
    async def test_invalidate_repo(self, cache):
        for i in range(3):
            content = FileContent(
                path=f"file{i}.py",
                content=f"content{i}",
                sha=f"sha{i}",
                size=8,
                repo="owner/repo",
                ref="main",
            )
            await cache.set(content)

        await cache.invalidate("owner/repo")

        for i in range(3):
            assert await cache.get("owner/repo", f"file{i}.py", "main") is None

    @pytest.mark.asyncio
    async def test_stats(self, cache):
        content = FileContent(
            path="test.py",
            content="x" * 100,
            sha="abc",
            size=100,
            repo="owner/repo",
            ref="main",
        )
        await cache.set(content)

        stats = await cache.stats()
        assert stats["cached_files"] == 1
        assert stats["total_size_bytes"] == 100
        assert "owner/repo" in stats["repos"]


# ============== Client Tests ==============

class TestGitHubClient:
    """Tests for GitHub API client."""
    
    @pytest.fixture
    def client(self):
        with patch.dict("os.environ", {"GITHUB_PAT": "test_token"}):
            return GitHubClient()
    
    def test_init_without_token_creates_unconfigured_client(self):
        """Client can be created without token but will fail on use."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove any existing token env vars
            import os
            os.environ.pop("GITHUB_PAT", None)
            os.environ.pop("GITHUB_TOKEN", None)

            # Client initializes but is not configured
            client = GitHubClient()
            assert client._configured is False

            # Attempting to use it raises the error
            with pytest.raises(GitHubClientError, match="No GITHUB_PAT"):
                client._check_configured()
    
    def test_parse_repo_valid(self, client):
        owner, repo = client._parse_repo("owner/repo")
        assert owner == "owner"
        assert repo == "repo"
    
    def test_parse_repo_invalid(self, client):
        with pytest.raises(GitHubClientError, match="Invalid repo format"):
            client._parse_repo("invalid")
    
    @pytest.mark.asyncio
    async def test_get_file_content(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "type": "file",
            "content": base64.b64encode(b"print('hello')").decode(),
            "sha": "abc123",
            "size": 14,
        }
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_get_client.return_value = mock_http
            
            result = await client.get_file_content("owner/repo", "test.py")
            
            assert result.content == "print('hello')"
            assert result.sha == "abc123"
    
    @pytest.mark.asyncio
    async def test_get_file_not_found(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_get_client.return_value = mock_http
            
            with pytest.raises(GitHubClientError, match="File not found"):
                await client.get_file_content("owner/repo", "missing.py")


# ============== Feature Tests ==============

class TestGitHubFeature:
    """Tests for the main GitHub feature."""
    
    @pytest.fixture
    def feature(self):
        with patch.dict("os.environ", {"GITHUB_PAT": "test_token"}):
            f = GitHubFeature()
            # Force cache initialization to avoid lazy loading issues
            _ = f.cache
            return f
    
    def test_resolve_repo_self(self, feature):
        result = feature._resolve_repo("self")
        assert result == "Kestrel-Sovereign-AI/kestrel"
    
    def test_resolve_repo_other(self, feature):
        result = feature._resolve_repo("other/repo")
        assert result == "other/repo"
    
    def test_get_tools(self, feature):
        # Environment should still have the token from fixture
        with patch.dict("os.environ", {"GITHUB_PAT": "test_token"}):
            tools = feature.get_tools()
            
            tool_names = [t.name for t in tools]
            assert "read_github_file" in tool_names
            assert "list_github_files" in tool_names
            assert "search_github_code" in tool_names
            assert "get_code_definition" in tool_names
            assert "list_code_definitions" in tool_names
            assert "get_self_repo_info" in tool_names
            assert "list_source_components" in tool_names
            assert "get_component_source" in tool_names
    
    @pytest.mark.asyncio
    async def test_read_file_cached(self, feature):
        # Set up cache
        content = FileContent(
            path="test.py",
            content="cached content",
            sha="abc",
            size=14,
            repo="Kestrel-Sovereign-AI/kestrel",
            ref="main",
        )
        await feature.cache.set(content)

        # Call the method directly
        result = await feature.read_github_file(repo="self", path="test.py")

        assert "cached content" in result
    
    @pytest.mark.asyncio
    async def test_get_definition_non_python(self, feature):
        # Call the method directly
        result = await feature.get_code_definition(
            repo="self", 
            path="README.md", 
            name="something"
        )
        
        assert "only supports Python" in result
