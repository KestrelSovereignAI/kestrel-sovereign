"""GitHub Feature - Repository access and code introspection."""
import logging
import os
from typing import Any, Optional

import yaml

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

from .ast_analyzer import ASTAnalyzer
from .cache import GitHubCache
from .client import GitHubClient, GitHubClientError
from .models import ComponentManifest, FileType

logger = logging.getLogger(__name__)


# Configuration
GITHUB_SELF_REPO = os.getenv("GITHUB_SELF_REPO", "KestrelSovereignAI/kestrel-sovereign")
GITHUB_DEFAULT_BRANCH = os.getenv("GITHUB_DEFAULT_BRANCH", "main")


class GitHubFeature(Feature):
    """Feature for accessing GitHub repositories and code introspection.
    
    Supports:
    - Reading files from any GitHub repository
    - Listing directory contents
    - Searching code
    - AST-based code analysis for Python files
    - Component manifest discovery for self-introspection
    - "self" alias for the agent's own codebase
    """
    
    tool_name = "github"
    tool_description = "Access GitHub repositories, read source code, and analyze the agent's own codebase"
    
    def __init__(self, agent=None):
        """Initialize GitHub feature."""
        super().__init__(agent)
        self._client: Optional[GitHubClient] = None
        self._cache: Optional[GitHubCache] = None
    
    async def initialize(self):
        """Initialize the feature."""
        # Client and cache are lazily initialized
        pass

    @property
    def is_available(self) -> bool:
        """Check if the GitHub feature is available (has token configured)."""
        return self.client._configured

    @property
    def client(self) -> GitHubClient:
        """Get or create GitHub client."""
        if self._client is None:
            self._client = GitHubClient()
        return self._client
    
    @property
    def cache(self) -> GitHubCache:
        """Get or create cache."""
        if self._cache is None:
            self._cache = GitHubCache()
        return self._cache
    
    def _resolve_repo(self, repo: str) -> str:
        """Resolve 'self' alias to actual repo."""
        if repo.lower() == "self":
            return GITHUB_SELF_REPO
        return repo
    
    async def cleanup(self):
        """Clean up resources."""
        if self._client:
            await self._client.close()
    
    # ============== Tools ==============
    
    @tool(
        name="read_github_file",
        description="Read a file from a GitHub repository. Use 'self' as repo to read from the agent's own codebase.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def read_github_file(
        self,
        repo: str,
        path: str,
        ref: str = "main",
    ) -> str:
        """Read a file from GitHub.
        
        Args:
            repo: Repository in 'owner/repo' format, or 'self' for agent's codebase
            path: Path to file within the repository
            ref: Branch, tag, or commit SHA (default: main)
            
        Returns:
            File content with path header
        """
        repo = self._resolve_repo(repo)
        if ref == "main":
            ref = GITHUB_DEFAULT_BRANCH
        
        # Check cache first
        cached = await self.cache.get(repo, path, ref)
        if cached:
            return f"# {path} (cached)\n\n{cached.content}"
        
        # Fetch from GitHub
        try:
            content = await self.client.get_file_content(repo, path, ref)
            # Cache it
            await self.cache.set(content)
            return f"# {path}\n\n{content.content}"
        except GitHubClientError as e:
            return f"Error reading {path}: {e}"
    
    @tool(
        name="list_github_files",
        description="List files in a GitHub repository directory. Use 'self' as repo for agent's codebase.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def list_github_files(
        self,
        repo: str,
        path: str = "",
        ref: str = "main",
        recursive: bool = False,
    ) -> str:
        """List files in a directory.
        
        Args:
            repo: Repository in 'owner/repo' format, or 'self'
            path: Directory path (empty for root)
            ref: Branch, tag, or commit SHA
            recursive: If true, list all files recursively
            
        Returns:
            Formatted file listing
        """
        repo = self._resolve_repo(repo)
        if ref == "main":
            ref = GITHUB_DEFAULT_BRANCH
        
        try:
            if recursive:
                files = await self.client.get_tree(repo, ref, recursive=True)
                # Filter by path prefix if specified
                if path:
                    files = [f for f in files if f.path.startswith(path)]
            else:
                files = await self.client.list_directory(repo, path, ref)
            
            # Format output
            lines = [f"# Files in {repo}:{path or '/'}\n"]
            
            for f in sorted(files, key=lambda x: (not x.is_dir(), x.path)):
                if f.is_dir():
                    lines.append(f"📁 {f.path}/")
                else:
                    size = f"{f.size:,}" if f.size else "?"
                    lines.append(f"📄 {f.path} ({size} bytes)")
            
            return "\n".join(lines)
        except GitHubClientError as e:
            return f"Error listing {path}: {e}"
    
    @tool(
        name="search_github_code",
        description="Search for code in GitHub repositories. Use 'self' to search agent's codebase.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def search_github_code(
        self,
        query: str,
        repo: Optional[str] = None,
        path: Optional[str] = None,
        extension: Optional[str] = None,
        max_results: int = 20,
    ) -> str:
        """Search for code in GitHub.
        
        Args:
            query: Search query
            repo: Limit to specific repo (optional, use 'self' for agent)
            path: Limit to path prefix (optional)
            extension: Limit to file extension (e.g., 'py')
            max_results: Maximum results (default 20)
            
        Returns:
            Formatted search results
        """
        if repo:
            repo = self._resolve_repo(repo)
        
        try:
            results = await self.client.search_code(
                query, repo=repo, path=path, extension=extension, max_results=max_results
            )
            
            if not results:
                return f"No results found for: {query}"
            
            lines = [f"# Search results for: {query}\n"]
            
            for r in results:
                lines.append(f"\n## {r.repo}: {r.path}")
                lines.append(f"[View on GitHub]({r.html_url})")
                
                # Include text matches if available
                for match in r.text_matches[:2]:
                    fragment = match.get("fragment", "")
                    if fragment:
                        lines.append(f"\n```\n{fragment}\n```")
            
            return "\n".join(lines)
        except GitHubClientError as e:
            return f"Search error: {e}"
    
    @tool(
        name="get_code_definition",
        description="Get a function or class definition from a Python file. Uses AST for accurate extraction.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def get_code_definition(
        self,
        repo: str,
        path: str,
        name: str,
        ref: str = "main",
    ) -> str:
        """Get a specific function or class definition.
        
        Args:
            repo: Repository or 'self'
            path: Path to Python file
            name: Function or class name
            ref: Branch, tag, or commit SHA
            
        Returns:
            Definition with signature, docstring, and source
        """
        repo = self._resolve_repo(repo)
        if ref == "main":
            ref = GITHUB_DEFAULT_BRANCH
        
        if not path.endswith(".py"):
            return "Error: AST analysis only supports Python files (.py)"
        
        # Get file content
        try:
            cached = await self.cache.get(repo, path, ref)
            if cached:
                content = cached.content
            else:
                file_content = await self.client.get_file_content(repo, path, ref)
                await self.cache.set(file_content)
                content = file_content.content
        except GitHubClientError as e:
            return f"Error reading {path}: {e}"
        
        # Parse and find definition
        analyzer = ASTAnalyzer(content, path)
        defn = analyzer.get_definition(name)
        
        if not defn:
            # List available definitions
            all_defs = analyzer.get_definitions()
            available = [d.name for d in all_defs[:20]]
            return f"Definition '{name}' not found in {path}.\n\nAvailable: {', '.join(available)}"
        
        return f"""# {defn.type.title()}: {defn.name}

**File:** {path}
**Lines:** {defn.start_line}-{defn.end_line}
**Signature:** `{defn.signature}`

## Docstring
{defn.docstring or "(no docstring)"}

## Source
```python
{defn.source}
```"""
    
    @tool(
        name="list_code_definitions",
        description="List all functions and classes in a Python file.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def list_code_definitions(
        self,
        repo: str,
        path: str,
        ref: str = "main",
    ) -> str:
        """List all definitions in a Python file.
        
        Args:
            repo: Repository or 'self'
            path: Path to Python file
            ref: Branch, tag, or commit SHA
            
        Returns:
            Organized list of all definitions
        """
        repo = self._resolve_repo(repo)
        if ref == "main":
            ref = GITHUB_DEFAULT_BRANCH
        
        if not path.endswith(".py"):
            return "Error: AST analysis only supports Python files (.py)"
        
        # Get file content
        try:
            cached = await self.cache.get(repo, path, ref)
            if cached:
                content = cached.content
            else:
                file_content = await self.client.get_file_content(repo, path, ref)
                await self.cache.set(file_content)
                content = file_content.content
        except GitHubClientError as e:
            return f"Error reading {path}: {e}"
        
        # Parse
        analyzer = ASTAnalyzer(content, path)
        definitions = analyzer.get_definitions()
        
        if not definitions:
            return f"No function or class definitions found in {path}"
        
        lines = [f"# Definitions in {path}\n"]
        
        # Group by type
        classes = [d for d in definitions if d.type == "class"]
        functions = [d for d in definitions if d.type == "function"]
        methods = [d for d in definitions if d.type == "method"]
        
        if classes:
            lines.append("\n## Classes")
            for d in classes:
                lines.append(f"- `{d.signature}` (lines {d.start_line}-{d.end_line})")
        
        if functions:
            lines.append("\n## Functions")
            for d in functions:
                lines.append(f"- `{d.signature}` (lines {d.start_line}-{d.end_line})")
        
        if methods:
            lines.append(f"\n## Methods ({len(methods)} total)")
            for d in methods[:30]:  # Limit output
                lines.append(f"- `{d.name}` (line {d.start_line})")
            if len(methods) > 30:
                lines.append(f"  ... and {len(methods) - 30} more")
        
        return "\n".join(lines)
    
    @tool(
        name="get_self_repo_info",
        description="Get information about the agent's own source repository.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def get_self_repo_info(self) -> str:
        """Get info about the agent's own repository.
        
        Returns:
            Repository metadata
        """
        repo = GITHUB_SELF_REPO
        
        try:
            info = await self.client.get_repo_info(repo)
            
            return f"""# Agent Source Repository

**Repository:** {info.get('full_name')}
**Description:** {info.get('description', 'N/A')}
**Default Branch:** {info.get('default_branch', 'main')}
**Visibility:** {info.get('visibility', 'unknown')}
**Language:** {info.get('language', 'Python')}
**Size:** {info.get('size', 0):,} KB
**URL:** {info.get('html_url')}

## Stats
- Stars: {info.get('stargazers_count', 0)}
- Forks: {info.get('forks_count', 0)}
- Open Issues: {info.get('open_issues_count', 0)}
- Last Updated: {info.get('updated_at', 'unknown')}

Use `list_source_components` to see the feature components that make up this agent."""
        except GitHubClientError as e:
            return f"Error getting repo info: {e}"
    
    @tool(
        name="list_source_components",
        description="List all feature components in the agent's source code with their manifests.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def list_source_components(self, include_files: bool = False) -> str:
        """List all feature components.
        
        Args:
            include_files: Include file listings for each component
            
        Returns:
            Formatted component list with manifests
        """
        repo = GITHUB_SELF_REPO
        ref = GITHUB_DEFAULT_BRANCH
        
        # Get features directory listing
        try:
            files = await self.client.list_directory(repo, "features", ref)
        except GitHubClientError as e:
            return f"Could not access features directory: {e}"
        
        components = []
        
        for f in files:
            if f.is_dir() and not f.name.startswith("_"):
                # Try to get component.yaml
                manifest = None
                try:
                    manifest_content = await self.client.get_file_content(
                        repo, f"features/{f.name}/component.yaml", ref
                    )
                    manifest_data = yaml.safe_load(manifest_content.content)
                    manifest = ComponentManifest.from_dict(manifest_data, f.name)
                except GitHubClientError:
                    # No manifest, create basic info
                    manifest = ComponentManifest(
                        feature_name=f.name,
                        description="(no component.yaml)",
                    )
                
                component_info = {
                    "name": f.name,
                    "manifest": manifest,
                }
                
                if include_files:
                    # List files in component directory
                    try:
                        comp_files = await self.client.get_tree(repo, ref)
                        comp_files = [
                            cf for cf in comp_files 
                            if cf.path.startswith(f"features/{f.name}/") and cf.is_file()
                        ]
                        component_info["files"] = [cf.path for cf in comp_files]
                    except GitHubClientError:
                        component_info["files"] = []
                
                components.append(component_info)
        
        # Format output
        lines = ["# Agent Source Components\n"]
        
        for comp in components:
            m = comp["manifest"]
            lines.append(f"\n## {m.feature_name}")
            lines.append(f"**Description:** {m.description}")
            lines.append(f"**Version:** {m.version}")
            lines.append(f"**Entry Point:** features/{m.feature_name}/{m.entry_point}")
            
            if m.tools:
                lines.append(f"**Tools:** {', '.join(m.tools)}")
            
            if include_files and comp.get("files"):
                lines.append("\n**Files:**")
                for path in comp["files"][:20]:
                    lines.append(f"  - {path}")
                if len(comp["files"]) > 20:
                    lines.append(f"  ... and {len(comp['files']) - 20} more")
        
        return "\n".join(lines)
    
    @tool(
        name="get_component_source",
        description="Get all source files for a specific feature component.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def get_component_source(
        self,
        component: str,
        include_content: bool = False,
    ) -> str:
        """Get source files for a component.
        
        Args:
            component: Component name (e.g., 'compute', 'security', 'github')
            include_content: Include file contents (warning: may be large)
            
        Returns:
            Component files and optionally contents
        """
        repo = GITHUB_SELF_REPO
        ref = GITHUB_DEFAULT_BRANCH
        
        component_path = f"features/{component}"
        
        # Get all files in component
        try:
            all_files = await self.client.get_tree(repo, ref)
            comp_files = [
                f for f in all_files 
                if f.path.startswith(component_path + "/") and f.is_file()
            ]
        except GitHubClientError as e:
            return f"Could not access component '{component}': {e}"
        
        if not comp_files:
            return f"Component '{component}' not found or has no files"
        
        lines = [f"# Component: {component}\n"]
        lines.append(f"**Path:** {component_path}")
        lines.append(f"**Files:** {len(comp_files)}")
        
        # Try to get manifest
        try:
            manifest_content = await self.client.get_file_content(
                repo, f"{component_path}/component.yaml", ref
            )
            lines.append("\n## Manifest (component.yaml)")
            lines.append(f"```yaml\n{manifest_content.content}\n```")
        except GitHubClientError:
            lines.append("\n*No component.yaml manifest*")
        
        lines.append("\n## Files")
        
        for f in sorted(comp_files, key=lambda x: x.path):
            rel_path = f.path[len(component_path) + 1:]
            lines.append(f"\n### {rel_path}")
            
            if include_content and f.path.endswith(".py"):
                try:
                    content = await self.client.get_file_content(repo, f.path, ref)
                    await self.cache.set(content)
                    lines.append(f"```python\n{content.content}\n```")
                except GitHubClientError as e:
                    lines.append(f"*Could not read: {e}*")
            else:
                lines.append(f"*Size: {f.size:,} bytes*")
        
        return "\n".join(lines)
    
    @tool(
        name="invalidate_github_cache",
        description="Invalidate cached GitHub content to force fresh fetch.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def invalidate_github_cache(
        self,
        repo: str,
        path: Optional[str] = None,
    ) -> str:
        """Invalidate cache entries.
        
        Args:
            repo: Repository to invalidate (or 'self')
            path: Specific path to invalidate (optional)
            
        Returns:
            Confirmation message
        """
        repo = self._resolve_repo(repo)
        
        await self.cache.invalidate(repo, path=path)
        
        if path:
            return f"Invalidated cache for {repo}:{path}"
        return f"Invalidated all cache for {repo}"
