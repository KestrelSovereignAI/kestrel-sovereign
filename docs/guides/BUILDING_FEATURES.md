# Building Your First Kestrel Feature

This guide walks you through creating a feature package for Kestrel Sovereign from scratch. By the end, you will have a working feature with tools, hooks, HTTP endpoints, configuration, tests, and packaging.

## Vocabulary

Before diving in, understand the hierarchy:

| Term | Definition |
|------|-----------|
| **Tool** | An atomic callable function — one `@tool`-decorated method on a Feature |
| **Skill** | An externally discoverable tool — a tool with A2A metadata for agent-to-agent discovery |
| **Feature** | A skill bundle — a `Feature` subclass grouping related tools, hooks, and endpoints |
| **Feature package** | A deployment unit — one or more features packaged as a pip-installable package and registered through the feature entry-point group |
| **Provider package** | A backend implementation package registered through a provider-specific entry-point group, such as cloud, voice, or storage providers |

A feature is the fundamental unit of extensibility in Kestrel. It groups related capabilities into a single class that the agent discovers and loads automatically.

## The Feature Contract

Every package feature is a subclass of `Feature` from the Kestrel SDK and must implement two things:

1. `async initialize()` — called on agent startup
2. `tool_description` property — one-line summary shown to the orchestrator LLM

Here is the minimal feature:

```python
from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory


class GreeterFeature(Feature):
    """A simple greeting feature."""

    @property
    def tool_description(self) -> str:
        return "Generate greetings for users"

    async def initialize(self):
        self.greeting_count = 0

    @tool("greet", "Greet a user by name", category=ToolCategory.UTILITY)
    async def greet(self, name: str) -> dict:
        """
        Generate a greeting message.

        Args:
            name: The name of the person to greet
        """
        self.greeting_count += 1
        return {"message": f"Hello, {name}!", "total_greetings": self.greeting_count}
```

### Constructor

The constructor receives the agent instance. Always call `super().__init__(agent)`:

```python
def __init__(self, agent):
    super().__init__(agent)
    # Initialize instance variables here — NO async work
    self.my_store = None
```

The `agent` object gives you access to:
- `agent.storage` — database facade (graph, files, conversations, RAG)
- `agent.llm_service` — LLM completion API
- `agent.hooks_manager` — hook registration
- `agent.features` — dict of all loaded features (populated after all features load)
- `agent.did` — the agent's decentralized identifier

## Lifecycle Hooks

Features have a well-defined lifecycle. Methods are called in this order:

```
__init__(agent)          # Sync. Set up instance variables.
    ↓
initialize()             # Async. Connect to databases, load state.
    ↓
get_hooks()              # Sync. Return Hook instances — auto-registered.
    ↓
on_enable()              # Async. Additional setup after hooks are registered.
    ↓
post_all_features_loaded(agent)  # Async. Cross-feature wiring.
    ↓
  ... agent is running ...
    ↓
on_disable()             # Async. Teardown before hooks are unregistered.
    ↓
  (hooks auto-unregistered)
    ↓
on_remove()              # Async. Called before feature package is uninstalled.
    ↓
shutdown()               # Async. Final cleanup (close connections, files).
```

### initialize()

**Required.** Perform async setup here — open database connections, load configuration, validate preconditions.

```python
async def initialize(self):
    db_path = getattr(self.agent, 'db_path', None)
    if db_path:
        self.store = MyStore(db_path)
        await self.store.setup()
```

### on_enable() / on_disable()

Optional. Called when the feature is enabled or disabled at runtime. Hooks returned by `get_hooks()` are auto-registered *before* `on_enable()` and auto-unregistered *after* `on_disable()`, so you only need these for work beyond hook management.

### post_all_features_loaded(agent)

Optional. Called after *every* feature has been discovered and initialized. Use this for cross-feature wiring:

```python
async def post_all_features_loaded(self, agent):
    # Access another feature safely
    security = agent.features.get("SecurityFeature")
    if security:
        await security.register_tools(self.get_tools())
```

### shutdown()

Optional. Release resources — close database connections, cancel background tasks.

### on_remove()

Optional. Called before the feature package is uninstalled. Clean up any persistent data (database tables, files) that should not outlive the feature.

## Tools

Tools are the primary way a feature exposes functionality. The `@tool` decorator marks an async method as callable by the agent.

### The @tool Decorator

```python
from kestrel_sdk.features.base import tool
from kestrel_sdk.tools.base import ToolCategory

@tool(
    name="search_items",              # Unique tool name
    description="Search stored items", # Shown to the LLM
    category=ToolCategory.DATA_ACCESS, # Categorization
    command_prefix="!search"           # Optional chat command
)
async def search_items(self, query: str, limit: int = 10) -> dict:
    """
    Search for items matching a query.

    Args:
        query: The search query string
        limit: Maximum number of results to return (default: 10)
    """
    results = await self.store.search(query, limit=limit)
    return {"results": results, "count": len(results)}
```

**Parameters are auto-discovered** from the method signature:
- Type annotations map to JSON Schema types: `str` → `"string"`, `int` → `"integer"`, `bool` → `"boolean"`, `float` → `"number"`, `list` → `"array"`, `dict` → `"object"`
- Parameters without defaults are required
- Parameters with defaults are optional
- `self` is automatically excluded

**Parameter descriptions** are extracted from the docstring. Supported formats:

```python
# Google style (recommended)
"""
Args:
    query: The search query string
    limit: Maximum results to return
"""

# Sphinx style
"""
:param query: The search query string
:param limit: Maximum results to return
"""
```

### Tool Categories

Use `ToolCategory` to classify your tools:

| Category | Use for |
|----------|---------|
| `SYSTEM` | Agent internals, diagnostics |
| `DATA_ACCESS` | Reading/querying data |
| `MODEL_MANAGEMENT` | LLM model operations |
| `COMMUNICATION` | Messaging, notifications |
| `UTILITY` | General-purpose tools |

### Command Prefixes

Setting `command_prefix="!my-cmd"` lets users invoke the tool directly from chat:

```
User: !my-cmd arg1 arg2
```

### Return Values

Tools should return JSON-serializable data. The framework handles serialization of:
- Primitives (`str`, `int`, `bool`, `float`, `None`)
- Dicts and lists (recursive)
- Objects with a `to_dict()` method
- Enums (uses `.value`)
- Everything else converts to `str`

### Tool Discovery

You do not need to register tools manually. The base class `get_tools()` method auto-discovers all methods decorated with `@tool`. The A2A protocol reuses the same metadata — the `@tool` decorator is the single source of truth for both local tool invocation and agent-to-agent skill discovery.

## Hooks

Hooks let your feature intercept agent events — security checks, audit logging, request filtering, etc.

### Defining a Hook

Create a Hook subclass from the Kestrel SDK, then return it from `get_hooks()`:

```python
from kestrel_sdk.hooks.base import Hook


class AuditHook(Hook):
    """Log every tool invocation."""

    def __init__(self, store):
        self.store = store

    async def before_tool_call(self, tool_name: str, params: dict, context: dict) -> dict:
        """Called before each tool executes. Return params (possibly modified)."""
        await self.store.log_event("tool_call", {
            "tool": tool_name,
            "params": params,
        })
        return params

    async def after_tool_call(self, tool_name: str, result: dict, context: dict) -> dict:
        """Called after each tool executes. Return result (possibly modified)."""
        return result
```

### Registering Hooks

Override `get_hooks()` to return your hook instances. The framework auto-registers them when the feature is enabled:

```python
class MyFeature(Feature):
    async def initialize(self):
        self._audit_hook = AuditHook(self.store)

    def get_hooks(self) -> list:
        return [self._audit_hook] if self._audit_hook else []
```

Do not call `hooks_manager.register()` directly — let the lifecycle handle it.

## Routers (HTTP Endpoints)

Features can contribute HTTP endpoints to the agent's FastAPI server.

### Adding a Router

Create a FastAPI `APIRouter` and return it from `get_router()`:

```python
# my_feature/endpoints.py
from fastapi import APIRouter

router = APIRouter(prefix="/my-feature", tags=["my-feature"])

@router.get("/status")
async def get_status():
    return {"status": "ok"}

@router.post("/items")
async def create_item(name: str, value: str):
    return {"created": True, "name": name}
```

```python
# my_feature/feature.py
class MyFeature(Feature):
    def get_router(self):
        from my_feature.endpoints import router
        return router
```

The agent mounts the router after all features are loaded. Endpoints are available at the agent's HTTP server (e.g., `http://localhost:8888/my-feature/status`).

## Configuration

Features can declare a JSON Schema for their settings. The UI can render a configuration form from this schema.

### Declaring a Config Schema

```python
class MyFeature(Feature):
    @property
    def config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Maximum search results to return",
                    "default": 25,
                    "minimum": 1,
                    "maximum": 100
                },
                "cache_ttl": {
                    "type": "integer",
                    "description": "Cache time-to-live in seconds",
                    "default": 300
                },
                "enabled_sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Data sources to search",
                    "default": ["local"]
                }
            }
        }

    async def get_config(self) -> dict:
        return {
            "max_results": self._max_results,
            "cache_ttl": self._cache_ttl,
            "enabled_sources": self._sources,
        }

    async def set_config(self, config: dict) -> None:
        # config is already validated against config_schema
        self._max_results = config.get("max_results", 25)
        self._cache_ttl = config.get("cache_ttl", 300)
        self._sources = config.get("enabled_sources", ["local"])
```

Return `None` from `config_schema` (the default) if your feature has no user-configurable settings.

## Testing

Kestrel provides `MockAgent` and pytest fixtures so you can test features without a running agent or real database.

### Using Pytest Fixtures

Register the fixtures in your `conftest.py`:

```python
# tests/conftest.py
pytest_plugins = ["kestrel_sovereign.testing.fixtures"]
```

Then use them in tests:

```python
# tests/test_greeter.py
import pytest
from my_feature.feature import GreeterFeature


async def test_greet(mock_agent):
    feature = GreeterFeature(mock_agent)
    await feature.initialize()

    result = await feature.greet(name="Alice")
    assert result["message"] == "Hello, Alice!"
    assert result["total_greetings"] == 1


async def test_greet_with_storage(mock_agent_with_storage):
    """Test with real in-memory SQLite when your feature needs storage."""
    feature = GreeterFeature(mock_agent_with_storage)
    await feature.initialize()

    result = await feature.greet(name="Bob")
    assert result["message"] == "Hello, Bob!"
```

### Available Fixtures

| Fixture | Storage | Use for |
|---------|---------|---------|
| `mock_agent` | No | Features that don't need database access |
| `mock_agent_with_storage` | In-memory SQLite | Features that read/write to `agent.storage` |
| `mock_agent_factory` | Configurable | Multiple agents or custom LLM responses |

### MockAgent Factory

Use the factory when you need to control LLM responses:

```python
async def test_with_llm_responses(mock_agent_factory):
    agent = mock_agent_factory(
        llm_responses=["First response", "Second response"],
        did="did:test:custom-agent",
    )
    feature = GreeterFeature(agent)
    await feature.initialize()
    # First LLM call returns "First response", second returns "Second response"
```

### FeatureTestCase (unittest style)

For unittest-based tests, extend `FeatureTestCase`:

```python
from kestrel_sovereign.testing import FeatureTestCase
from my_feature.feature import GreeterFeature


class TestGreeter(FeatureTestCase):
    feature_class = GreeterFeature
    use_storage = False  # Set True if feature needs database

    async def test_greet(self):
        # self.agent and self.feature are auto-created
        result = await self.feature.greet(name="Alice")
        assert result["message"] == "Hello, Alice!"
```

`FeatureTestCase` handles `MockAgent` creation, feature initialization, and teardown automatically.

### Running Tests

```bash
# Run your feature's tests
uv run pytest tests/test_greeter.py -x

# With verbose output
uv run pytest tests/test_greeter.py -xvs
```

## Packaging

Feature packages are standard Python packages that register an entry point so Kestrel discovers them automatically.

### Project Structure

```
kestrel-feature-greeter/
├── pyproject.toml
├── SKILL.md                        # Human/LLM-readable skill descriptions
├── kestrel_feature_greeter/
│   ├── __init__.py
│   ├── feature.py                  # GreeterFeature class
│   └── endpoints.py                # Optional: FastAPI router
└── tests/
    ├── conftest.py
    └── test_greeter.py
```

### pyproject.toml

```toml
[project]
name = "kestrel-feature-greeter"
version = "0.1.0"
description = "Greeting feature for Kestrel agents"
requires-python = ">=3.11"
dependencies = [
    "kestrel-sovereign-sdk>=0.1.0",
]

[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
]

[project.entry-points."kestrel_sovereign.features"]
GreeterFeature = "kestrel_feature_greeter.feature:GreeterFeature"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

The entry point is the key: the name (`GreeterFeature`) must match the class name, and the value is the dotted import path to the class.

### Entry Point Discovery

When Kestrel starts, it:

1. Scans `kestrel_sovereign/features/` for core features (Phase 1)
2. Loads entry points from the `kestrel_sovereign.features` group (Phase 2)
3. Core features take priority — if a name collision occurs, the core feature wins

### Disabling Features

Features can be disabled via environment variable:

```bash
KESTREL_DISABLED_FEATURES=GreeterFeature,SomeOtherFeature
```

Or by passing `allowed_features` to the discovery function to whitelist specific features.

## SKILL.md Convention

Each feature directory should include a `SKILL.md` file — a human-readable and LLM-readable description of the feature's skills. This file serves as a quick reference for developers and is used by the A2A discovery protocol.

### Template

```markdown
# {Feature Name}

> {One-line description}

## Skills

### {skill_name}
- **Description**: {What this skill does}
- **Category**: {ToolCategory value}
- **Command**: `{!command-prefix}` (if applicable)
- **Parameters**:
  - `param_name` (type, required/optional): Description
- **Returns**: {Description of return value}

### {another_skill}
...

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| ... | ...  | ...     | ...         |

## Dependencies

- Requires: {other features or external services}
- Optional: {features that enhance behavior if present}
```

### Example SKILL.md

```markdown
# Greeter

> Generate personalized greetings for users.

## Skills

### greet
- **Description**: Greet a user by name
- **Category**: utility
- **Command**: `!greet`
- **Parameters**:
  - `name` (string, required): The name of the person to greet
- **Returns**: `{"message": "Hello, {name}!", "total_greetings": int}`

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| greeting_style | string | "formal" | Style of greeting (formal, casual, enthusiastic) |

## Dependencies

- Requires: none
- Optional: `ConsentFeature` (for tracking greeting consent)
```

## Submitting to the Feature Registry

The feature registry at `kestrel_sovereign/data/feature_registry.toml` is the static catalog of all known features. To add your feature:

1. Open a PR to the `kestrel-sovereign` repository
2. Add an entry to `feature_registry.toml`:

```toml
[greeter]
package = "kestrel-feature-greeter"
git = "https://github.com/your-org/kestrel-feature-greeter.git"
features = ["GreeterFeature"]
description = "Generate personalized greetings for users"
tags = ["utility", "greeting"]
icon = "hand-wave"
core = false

[greeter.skills]
greet = { description = "Greet a user by name", category = "utility", tags = ["greeting"] }
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `package` | Yes | pip package name |
| `git` | Yes | Source repository URL |
| `features` | Yes | List of Feature class names in this package |
| `description` | Yes | Human-readable summary |
| `tags` | Yes | List of tags for filtering/search |
| `icon` | No | Icon identifier for UI rendering |
| `core` | No | `true` if shipped with kestrel-sovereign itself |
| `skills` | No | Per-skill declarations for static discovery |

**Runtime status** is computed automatically:
- `available` — listed in registry but package not installed
- `installed` — package installed via entry point but not enabled
- `enabled` — loaded and active
- `disabled` — explicitly disabled via `KESTREL_DISABLED_FEATURES`

### Feature Packages vs Provider Packages

Feature packages register `Feature` classes through:

```toml
[project.entry-points."kestrel_sovereign.features"]
GreeterFeature = "kestrel_feature_greeter.feature:GreeterFeature"
```

Provider packages are different: they implement an SDK provider contract and
register with that provider's entry-point group. Do not add provider classes to
`features = [...]` in `feature_registry.toml` unless they are actual
`Feature` lifecycle classes. For example, cloud deployment providers register
with:

```toml
[project.entry-points."kestrel_sovereign.cloud_providers"]
runpod = "kestrel_cloud_runpod.provider:RunPodProvider"
```

The core cloud feature entry covers Kestrel's own `DeployFeature` and
`ComputeFeature`. Packages such as `kestrel-cloud-runpod`,
`kestrel-cloud-vastai`, and `kestrel-cloud-gcp` are provider packages, not
`kestrel-feature-cloud`.

## Complete Example

Putting it all together — a feature with a tool, a hook, an HTTP endpoint, and configuration:

```python
# kestrel_feature_greeter/feature.py
import logging
from typing import Dict, List, Optional

from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.hooks.base import Hook
from kestrel_sdk.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class GreetingHook(Hook):
    """Track greeting events for analytics."""

    def __init__(self):
        self.call_count = 0

    async def after_tool_call(self, tool_name, result, context):
        if tool_name == "greet":
            self.call_count += 1
        return result


class GreeterFeature(Feature):
    """Personalized greeting generation with analytics."""

    def __init__(self, agent):
        super().__init__(agent)
        self._hook = None
        self._style = "formal"

    @property
    def tool_description(self) -> str:
        return "Generate personalized greetings for users with analytics tracking"

    async def initialize(self):
        self._hook = GreetingHook()
        logger.info("GreeterFeature initialized")

    def get_hooks(self) -> list:
        return [self._hook] if self._hook else []

    def get_router(self):
        from kestrel_feature_greeter.endpoints import router
        return router

    async def post_all_features_loaded(self, agent):
        consent = agent.features.get("ConsentFeature")
        if consent:
            logger.info("ConsentFeature available — greeting consent tracking enabled")

    @property
    def config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "style": {
                    "type": "string",
                    "description": "Greeting style",
                    "enum": ["formal", "casual", "enthusiastic"],
                    "default": "formal"
                }
            }
        }

    async def get_config(self) -> dict:
        return {"style": self._style}

    async def set_config(self, config: dict) -> None:
        self._style = config.get("style", "formal")

    @tool("greet", "Greet a user by name", category=ToolCategory.UTILITY, command_prefix="!greet")
    async def greet(self, name: str) -> dict:
        """
        Generate a greeting message.

        Args:
            name: The name of the person to greet
        """
        greetings = {
            "formal": f"Good day, {name}.",
            "casual": f"Hey {name}!",
            "enthusiastic": f"HELLO {name}!!! So great to see you!",
        }
        return {
            "message": greetings.get(self._style, greetings["formal"]),
            "style": self._style,
        }

    @tool("greeting_stats", "Show greeting analytics", category=ToolCategory.SYSTEM)
    async def greeting_stats(self) -> dict:
        """Show how many greetings have been generated."""
        return {
            "total_greetings": self._hook.call_count if self._hook else 0,
            "current_style": self._style,
        }
```

```python
# kestrel_feature_greeter/endpoints.py
from fastapi import APIRouter

router = APIRouter(prefix="/greeter", tags=["greeter"])

@router.get("/health")
async def health():
    return {"feature": "greeter", "status": "ok"}
```

```python
# tests/conftest.py
pytest_plugins = ["kestrel_sovereign.testing.fixtures"]
```

```python
# tests/test_greeter.py
import pytest
from kestrel_feature_greeter.feature import GreeterFeature


async def test_greet_formal(mock_agent):
    feature = GreeterFeature(mock_agent)
    await feature.initialize()

    result = await feature.greet(name="Alice")
    assert result["message"] == "Good day, Alice."
    assert result["style"] == "formal"


async def test_greet_casual(mock_agent):
    feature = GreeterFeature(mock_agent)
    await feature.initialize()
    await feature.set_config({"style": "casual"})

    result = await feature.greet(name="Bob")
    assert result["message"] == "Hey Bob!"


async def test_config_schema(mock_agent):
    feature = GreeterFeature(mock_agent)
    schema = feature.config_schema
    assert "style" in schema["properties"]
    assert schema["properties"]["style"]["enum"] == ["formal", "casual", "enthusiastic"]


async def test_greeting_stats(mock_agent):
    feature = GreeterFeature(mock_agent)
    await feature.initialize()

    stats = await feature.greeting_stats()
    assert stats["total_greetings"] == 0


async def test_hooks_registered(mock_agent):
    feature = GreeterFeature(mock_agent)
    await feature.initialize()

    hooks = feature.get_hooks()
    assert len(hooks) == 1


async def test_router_returned(mock_agent):
    feature = GreeterFeature(mock_agent)
    router = feature.get_router()
    assert router is not None
```

## Quick Reference

| What | Where |
|------|-------|
| Base class | `kestrel_sdk.features.base.Feature` |
| @tool decorator | `kestrel_sdk.features.base.tool` |
| Tool categories | `kestrel_sdk.tools.base.ToolCategory` |
| Hook base class | `kestrel_sdk.hooks.base.Hook` |
| MockAgent | `kestrel_sovereign.testing.mock_agent.MockAgent` |
| Pytest fixtures | `kestrel_sovereign.testing.fixtures` |
| FeatureTestCase | `kestrel_sovereign.testing.feature_test_case.FeatureTestCase` |
| Feature registry | `kestrel_sovereign/data/feature_registry.toml` |
| Feature discovery | `kestrel_sovereign.features.discover_features()` |
| Entry point group | `kestrel_sovereign.features` |
