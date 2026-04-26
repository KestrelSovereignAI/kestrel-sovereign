# GitHub Feature Design

> **Implementation status (last verified 2026-04-25):** the GitHub feature ships at `features/github/` with 48 unit tests. Roughly half of the design below is implemented — enough for "read a file, search code, get a definition, query issues" — but the deeper static-analysis surface is not.
>
> **Shipped:** `read_github_file`, `list_github_files`, `search_github_code`, `get_code_definition`, `list_code_definitions`, `get_self_repo_info`, `list_source_components`, `get_component_source`, `invalidate_github_cache`, plus a bonus issue-access tool family (`list_github_issues`, `get_github_issue`, `get_github_issue_comments`). Caching via SQLite (`features/github/cache.py`) and basic AST extraction (`features/github/ast_analyzer.py`) work.
>
> **Not shipped (still aspirational):** `find_usages`, `get_call_graph`, `get_inheritance_tree`, `analyze_dependencies`, and the deeper symbol-resolution / call-graph tooling described under "Advanced Features" below. The current AST analyzer extracts function and class definitions; it does not resolve cross-file symbol usage or build dependency graphs.
>
> This doc is kept as the design-of-record so the deferred capabilities are discoverable. Don't archive it.

## Overview

The GitHub Feature allows the Kestrel agent to access and analyze code from GitHub repositories. This enables:
- **Self-introspection**: Agent can examine its own codebase using "self" alias
- **Research**: Agent can study other codebases for patterns, examples, implementations
- **Comparison**: "How does LangChain do X?" type queries

## Scope

- **Repositories**: Any accessible repo (public or private with token)
- **Self Alias**: `self` → `KestrelSovereignAI/kestrel-sovereign` (configurable per deployment)
- **Freshness**: Cached with configurable TTL (default: 1 hour)
- **Depth**: Advanced features including AST parsing, symbol lookup, call graphs
- **Auth**: GitHub Personal Access Token for private repos and higher rate limits

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       GitHubFeature                              │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   GitHubClient  │  │   CodeCache     │  │   ASTAnalyzer   │ │
│  │                 │  │                 │  │                 │ │
│  │ • fetch_file    │  │ • SQLite store  │  │ • parse_python  │ │
│  │ • search_code   │  │ • Per-repo keys │  │ • find_symbols  │ │
│  │ • list_contents │  │ • TTL: 1 hour   │  │ • call_graph    │ │
│  │ • rate_limiting │  │ • invalidate()  │  │ • inheritance   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
│  "self" alias ──────────────────────────▶ KestrelSovereignAI/kestrel-sovereign     │
│                                                                  │
│  Tools:                                                          │
│  • read_github_file      • get_function_definition              │
│  • search_github         • get_class_definition                 │
│  • list_github_files     • find_usages                          │
│  • get_file_info         • get_call_graph                       │
│  • get_module_structure  • refresh_cache                        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. "self" as Special Alias

```python
# Resolves to configured repo
read_github_file("self", "features/base.py")
# Equivalent to:
read_github_file("KestrelSovereignAI/kestrel-sovereign", "features/base.py")

# Kestrel agents could configure differently:
# GITHUB_SELF_REPO=KestrelSovereignAI/kestrel-sovereign
```

### 2. Flexible Repo Specification

```python
# Full owner/repo format
read_github_file("langchain-ai/langchain", "README.md")

# Self alias (default for most tools)
read_github_file("self", "features/compute/feature.py")
search_github("ComputeFeature", repo="self")

# Tools default to "self" when repo is optional
list_github_files()  # Lists self repo root
```

### 3. Access Control (Optional)

```python
# Environment variable to restrict access
GITHUB_ALLOWED_REPOS=self,psf/*,langchain-ai/*

# Or leave unrestricted for public repos
# Private repos always require token
```

## Components

### 1. GitHubClient

Handles all GitHub API interactions with proper authentication and rate limiting.

```python
class GitHubClient:
    """Client for GitHub API with caching and rate limit awareness."""
    
    def __init__(self, token: str, repo: str = "KestrelSovereignAI/kestrel-sovereign"):
        self.token = token
        self.repo = repo
        self.base_url = f"https://api.github.com/repos/{repo}"
        self._rate_limit_remaining = None
        self._rate_limit_reset = None
    
    async def fetch_file(self, path: str) -> str:
        """Fetch file content from GitHub."""
        # GET /repos/{owner}/{repo}/contents/{path}
        # Returns base64 encoded content
    
    async def list_contents(self, path: str = "") -> List[Dict]:
        """List directory contents."""
        # GET /repos/{owner}/{repo}/contents/{path}
    
    async def search_code(self, query: str, path_filter: str = None) -> List[Dict]:
        """Search code in repository."""
        # GET /search/code?q={query}+repo:{repo}
    
    async def get_tree(self, recursive: bool = True) -> List[Dict]:
        """Get full repository tree."""
        # GET /repos/{owner}/{repo}/git/trees/{branch}?recursive=1
```

### 2. CodeCache

SQLite-based cache to reduce API calls and enable offline-ish operation.

```python
class CodeCache:
    """Cache for source code with TTL-based invalidation."""
    
    # Schema
    """
    CREATE TABLE file_cache (
        path TEXT PRIMARY KEY,
        content TEXT,
        sha TEXT,
        fetched_at TIMESTAMP,
        size_bytes INTEGER
    );
    
    CREATE TABLE tree_cache (
        id INTEGER PRIMARY KEY,
        tree_json TEXT,
        fetched_at TIMESTAMP
    );
    
    CREATE TABLE symbol_index (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        type TEXT,  -- 'function', 'class', 'method', 'variable'
        file_path TEXT,
        line_start INTEGER,
        line_end INTEGER,
        signature TEXT,
        docstring TEXT
    );
    CREATE INDEX idx_symbol_name ON symbol_index(name);
    CREATE INDEX idx_symbol_type ON symbol_index(type);
    """
    
    def __init__(self, db_path: Path, ttl_seconds: int = 3600):
        self.db_path = db_path
        self.ttl = ttl_seconds
    
    async def get_file(self, path: str) -> Optional[str]:
        """Get cached file if not expired."""
    
    async def set_file(self, path: str, content: str, sha: str):
        """Cache a file."""
    
    async def is_stale(self, path: str) -> bool:
        """Check if cache entry is expired."""
    
    async def invalidate(self, path: str = None):
        """Invalidate specific file or entire cache."""
    
    async def get_symbol(self, name: str, type: str = None) -> List[SymbolInfo]:
        """Look up symbol by name."""
    
    async def index_file(self, path: str, content: str):
        """Parse and index symbols from a file."""
```

### 3. ASTAnalyzer

Python AST analysis for advanced code understanding.

```python
class ASTAnalyzer:
    """Analyze Python source code using AST."""
    
    def parse_file(self, content: str, filename: str = "<string>") -> ast.Module:
        """Parse Python source into AST."""
    
    def extract_symbols(self, content: str, filepath: str) -> List[SymbolInfo]:
        """Extract all symbols (classes, functions, methods) from source."""
    
    def get_function_info(self, content: str, func_name: str) -> Optional[FunctionInfo]:
        """Get detailed info about a function."""
        # Returns: signature, docstring, decorators, line range, complexity
    
    def get_class_info(self, content: str, class_name: str) -> Optional[ClassInfo]:
        """Get detailed info about a class."""
        # Returns: bases, methods, attributes, docstring, line range
    
    def find_imports(self, content: str) -> List[ImportInfo]:
        """Find all imports in a file."""
    
    def find_usages(self, content: str, symbol_name: str) -> List[Usage]:
        """Find all usages of a symbol in code."""
    
    def build_call_graph(self, content: str) -> Dict[str, List[str]]:
        """Build a call graph for functions in a file."""
        # Returns: {function_name: [list of functions it calls]}
```

## Data Models

```python
@dataclass
class SymbolInfo:
    name: str
    type: str  # 'function', 'class', 'method', 'variable', 'constant'
    file_path: str
    line_start: int
    line_end: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent: Optional[str] = None  # For methods, the class name

@dataclass
class FunctionInfo:
    name: str
    signature: str
    docstring: Optional[str]
    decorators: List[str]
    line_start: int
    line_end: int
    complexity: int  # Cyclomatic complexity
    parameters: List[ParameterInfo]
    returns: Optional[str]
    calls: List[str]  # Functions this function calls

@dataclass
class ClassInfo:
    name: str
    bases: List[str]
    docstring: Optional[str]
    methods: List[str]
    attributes: List[str]
    decorators: List[str]
    line_start: int
    line_end: int

@dataclass
class SearchResult:
    file_path: str
    line_number: int
    line_content: str
    context_before: List[str]
    context_after: List[str]
    match_type: str  # 'exact', 'fuzzy', 'regex'

@dataclass
class FileInfo:
    path: str
    size_bytes: int
    line_count: int
    language: str
    last_modified: Optional[datetime]
    symbols: List[str]  # Top-level symbol names
```

## Tools

### Basic File Operations

```python
@tool
async def read_github_file(
    self,
    repo: str = "self",
    path: str = "",
    branch: str = "main",
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> str:
    """
    Read a file from a GitHub repository.
    
    Args:
        repo: Repository - "self" for this agent's codebase, or "owner/repo"
        path: File path relative to repo root (e.g., "features/compute/feature.py")
        branch: Branch name (default: main)
        start_line: Optional start line (1-indexed, inclusive)
        end_line: Optional end line (1-indexed, inclusive)
    
    Returns:
        File content (or line range if specified)
    
    Examples:
        read_github_file("self", "features/base.py")
        read_github_file("self", "llm/adapter.py", start_line=50, end_line=100)
        read_github_file("langchain-ai/langchain", "README.md")
    """

@tool
async def list_github_files(
    self,
    repo: str = "self",
    path: str = "",
    pattern: Optional[str] = None,
    recursive: bool = False
) -> List[Dict[str, Any]]:
    """
    List files in a GitHub repository.
    
    Args:
        repo: Repository - "self" for this agent's codebase, or "owner/repo"
        path: Directory path (default: root)
        pattern: Optional glob pattern filter (e.g., "*.py", "test_*.py")
        recursive: Whether to list recursively
    
    Returns:
        List of file info dicts with name, path, type, size
    """

@tool
async def search_github(
    self,
    query: str,
    repo: str = "self",
    file_pattern: Optional[str] = None,
    max_results: int = 20
) -> List[Dict[str, Any]]:
    """
    Search code in a GitHub repository.
    
    Args:
        query: Search query (text, function name, etc.)
        repo: Repository - "self" for this agent's codebase, or "owner/repo"
        file_pattern: Optional file pattern (e.g., "features/**/*.py")
        max_results: Maximum results to return
    
    Returns:
        List of matches with file, line, content, and context
    
    Examples:
        search_github("ComputeFeature")  # Search self repo
        search_github("async def", repo="psf/requests")
    """

@tool
async def get_github_file_info(
    self,
    repo: str = "self",
    path: str = ""
) -> Dict[str, Any]:
    """
    Get metadata about a file in a GitHub repository.
    
    Returns:
        Dict with path, size, line_count, language, symbols
    """
```

### Symbol Operations (Python-specific)

```python
@tool
async def get_function_definition(
    self,
    function_name: str,
    repo: str = "self",
    file_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Find and return the definition of a function by name.
    
    Args:
        function_name: Name of the function to find
        repo: Repository to search (default: "self")
        file_hint: Optional file path hint to narrow search
    
    Returns:
        Dict with signature, docstring, source, file_path, line_range
    
    Examples:
        get_function_definition("write_script")  # Find in self
        get_function_definition("get", repo="psf/requests", file_hint="requests/api.py")
    """

@tool
async def get_class_definition(
    self,
    class_name: str,
    repo: str = "self",
    file_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Find and return the definition of a class by name.
    
    Args:
        class_name: Name of the class to find
        repo: Repository to search (default: "self")
        file_hint: Optional file path hint to narrow search
    
    Returns:
        Dict with name, bases, methods, attributes, docstring, source
    """

@tool
async def find_symbol(
    self,
    name: str,
    repo: str = "self",
    symbol_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find all definitions of a symbol (function, class, variable).
    
    Args:
        name: Symbol name to search for
        repo: Repository to search (default: "self")
        symbol_type: Optional type filter ('function', 'class', 'method', 'variable')
    
    Returns:
        List of matches with location and definition info
    """

@tool
async def get_module_structure(
    self,
    repo: str = "self",
    path: str = ""
) -> Dict[str, Any]:
    """
    Get the structure of a Python module.
    
    Returns:
        Dict with imports, classes, functions, constants
    """
```

### Advanced Analysis (Python-specific)

```python
@tool
async def find_usages(
    self,
    symbol_name: str,
    repo: str = "self",
    file_pattern: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find all usages of a symbol across a repository.
    
    Args:
        symbol_name: Name of symbol to find usages of
        repo: Repository to search (default: "self")
        file_pattern: Optional file pattern to limit search
    
    Returns:
        List of usages with file, line, context
    """

@tool
async def get_call_graph(
    self,
    function_name: str,
    repo: str = "self",
    depth: int = 2
) -> Dict[str, Any]:
    """
    Get the call graph for a function.
    
    Args:
        function_name: Starting function
        repo: Repository to analyze (default: "self")
        depth: How many levels deep to trace (default: 2)
    
    Returns:
        Dict with 'calls' (functions this calls) and 'called_by' (callers)
    """

@tool
async def get_inheritance_tree(
    self,
    class_name: str,
    repo: str = "self"
) -> Dict[str, Any]:
    """
    Get the inheritance hierarchy for a class.
    
    Returns:
        Dict with 'bases' (parent classes) and 'subclasses'
    """

@tool
async def analyze_dependencies(
    self,
    repo: str = "self",
    path: str = ""
) -> Dict[str, Any]:
    """
    Analyze imports and dependencies of a file or module.
    
    Returns:
        Dict with internal_imports, external_imports, dependency_graph
    """
```

### Cache Management

```python
@tool
async def refresh_github_cache(
    self,
    repo: str = "self",
    path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Refresh the code cache for a repository.
    
    Args:
        repo: Repository to refresh (default: "self")
        path: Optional specific file/directory to refresh (default: all)
    
    Returns:
        Dict with files_refreshed, symbols_indexed
    """

@tool
async def get_github_cache_status(self) -> Dict[str, Any]:
    """
    Get cache statistics across all cached repositories.
    
    Returns:
        Dict with repos_cached, total_files, total_size, oldest_entry
    """
```

## File Structure

```
features/github/
├── __init__.py
├── models.py              # SymbolInfo, FunctionInfo, ClassInfo, etc.
├── client.py              # GitHub API client with rate limiting
├── cache.py               # SQLite cache for files and symbols (per-repo)
├── ast_analyzer.py        # Python AST analysis
├── symbol_index.py        # Symbol indexing and lookup
├── feature.py             # GitHubFeature with all tools
└── tests/
    ├── test_client.py
    ├── test_cache.py
    ├── test_ast_analyzer.py
    └── test_github_feature.py
```

## Configuration

```python
# Environment variables
GITHUB_TOKEN=ghp_xxxxxxxxxxxx  # Personal Access Token with repo scope
GITHUB_SELF_REPO=KestrelSovereignAI/kestrel-sovereign  # What "self" resolves to
GITHUB_DEFAULT_BRANCH=main  # Default branch
GITHUB_CACHE_TTL=3600  # Cache TTL in seconds (1 hour)

# Optional: Restrict accessible repos
GITHUB_ALLOWED_REPOS=self,psf/*,langchain-ai/*  # Wildcards supported
```

## GitHub Token Setup

The token needs `repo` scope to access private repositories.

```bash
# Token is created via GitHub UI or CLI:
# Settings → Developer settings → Personal access tokens → Tokens (classic)
# Required scopes: repo (full control of private repositories)

# Add to .env
echo "GITHUB_TOKEN=ghp_your_token_here" >> .env
```

## Security Considerations

1. **Token Storage**: Token stored in `.env`, never committed
2. **Rate Limiting**: Respect GitHub API limits (5000/hr authenticated)
3. **Cache Security**: Cache stored in agent_data, encrypted at rest
4. **No Write Access**: Feature only reads, never writes to repo

## Implementation Order

1. **Phase 1: Core Infrastructure**
   - `models.py` - Data models
   - `client.py` - GitHub API client with auth and rate limiting
   - `cache.py` - SQLite caching (keyed by repo+path)

2. **Phase 2: Basic Tools**
   - `feature.py` with basic tools:
     - `read_github_file`
     - `list_github_files`
     - `search_github`
     - `get_github_file_info`

3. **Phase 3: AST Analysis**
   - `ast_analyzer.py` - Python parsing
   - `symbol_index.py` - Symbol indexing
   - Advanced tools:
     - `get_function_definition`
     - `get_class_definition`
     - `find_symbol`
     - `get_module_structure`

4. **Phase 4: Advanced Features**
   - `find_usages`
   - `get_call_graph`
   - `get_inheritance_tree`
   - `analyze_dependencies`

## Usage Examples

```
User: "Show me how the ComputeFeature's write_script tool works"

Agent: [calls get_function_definition("write_script", repo="self", file_hint="features/compute/feature.py")]
       [calls get_class_definition("ComputeScript", repo="self")]
       [calls find_usages("write_script", repo="self")]
       
       "The write_script tool is defined in features/compute/feature.py...
        It creates a ComputeScript model, analyzes it for security issues,
        signs it with the agent's identity, and stores it for later execution..."

User: "What classes inherit from Feature?"

Agent: [calls search_github("class.*Feature", repo="self", file_pattern="features/**/*.py")]
       [calls get_inheritance_tree("Feature", repo="self")]
       
       "I found 12 classes that inherit from Feature:
        - ComputeFeature (features/compute/feature.py)
        - SecurityFeature (features/security/feature.py)
        - WalletFeature (features/wallet/feature.py)
        ..."

User: "How does LangChain implement their agent base class?"

Agent: [calls search_github("class.*Agent", repo="langchain-ai/langchain")]
       [calls get_class_definition("Agent", repo="langchain-ai/langchain")]
       
       "LangChain's Agent base class is in langchain/agents/agent.py...
        It uses a different pattern than Kestrel - here's how they structure it..."
```

## Estimated Effort

- Phase 1 (Core): 3-4 hours
- Phase 2 (Basic Tools): 2-3 hours
- Phase 3 (AST Analysis): 3-4 hours
- Phase 4 (Advanced): 2-3 hours
- Testing: 2-3 hours

**Total: ~1.5-2 days**
