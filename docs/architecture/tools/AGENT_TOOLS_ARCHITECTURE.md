# Agent Tools Architecture - Design Document

> **⚠ DEPRECATED — describes a removed architecture.**
> The Phase 1-4 implementation centred on `AgentToolMixin`
> (`kestrel_agent_tools.py`), `tools/registry.py`, and the legacy
> top-level `tools/` directory. All of those have been removed.
> Tools now ship through feature packages registered via the
> `kestrel_sovereign.features` entry-point group with the `@tool`
> decorator from `kestrel_sdk.features.base`.
>
> See [`docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md`](../core/FEATURE_AGENT_FRAMEWORK.md)
> for the modern pattern.
>
> Rewrite tracked in [#1047](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1047);
> kept here meanwhile for git-archaeology context.

**Status:** IMPLEMENTED (Phases 1-4)
**Priority:** HIGH - Core agent capability
**Last Updated:** November 22, 2025

## Executive Summary

Kestrel agents use a unified tool system that enables:

1. **User-invoked commands** (`!list-models`)
2. **Agent-autonomous tool use** (agent decides when to search, pull models, etc.)
3. **Programmatic tool invocation** (other code can call tools)

**New Capability (Phase 4):** Dynamic tool loading via Docker MCP Hub. This allows agents to discover, install, and use tools from the Model Context Protocol ecosystem without code changes.

## Current State Analysis

### Implemented Components (Phases 1-4)

The following components are fully implemented and operational:

1. **Tool Infrastructure**
   - `AgentTool` base class (`tools/base.py`)
   - `ToolRegistry` (`tools/registry.py`)
   - `AgentToolMixin` (`kestrel_agent_tools.py`)

2. **Core Tools**
   - `WebSearchTool` (Tavily API)
   - `FeedbackFeature` (Diagnostics & Self-correction)
   - `ModelAgent` (Ollama management)
   - **MCPAgent** (`features/mcp/`) - *New in Phase 4*

3. **Integration**
   - Database tracking (`tool_usage`, `agent_feedback`)
   - Command routing via Registry (`!search`, `!feedback`, `!list-models`, `!mcp-load`)
   - Autonomous tool injection into prompts

### The Next Frontier: Dynamic MCP Tools

**Problem:** Adding new tools currently requires code changes (creating a new Python class, registering it).
**Solution:** Use the Model Context Protocol (MCP) and Docker to allow agents to dynamically load tools at runtime.

**Source:** [Docker MCP Hub](https://hub.docker.com/u/mcp)

## Architecture (Phase 4: MCP Integration)

### 1. MCP Tool Manager (`features/mcp/manager.py`)

A new tool that manages the lifecycle of MCP servers running in Docker containers.

**Design Decisions:**
- **Transport:** HTTP (SSE) over stdio. Stdio can be unreliable for long-running processes in production.
- **Container Management:** Python `docker` SDK (not shell commands).
- **MCP Client:** Python `mcp` SDK (`mcp.client.sse`, `mcp.client.session`).

```python
class MCPToolManager:
    """
    Manages MCP tools running in Docker containers.
    Handles container lifecycle and SSE connections.
    """
    # Uses docker SDK to run containers with port mapping
    # Uses mcp.client.sse.sse_client to connect
    # Maintains active sessions in self.active_tools
```

### 2. Architecture Changes

1. **Dependency Updates**:
   - Added `mcp` (Python SDK)
   - Added `docker` (Python SDK)

2. **Tool Registry Updates**:
   - `MCPToolManager` is initialized in `MCPAgent`.
   - Tools discovered from MCP servers are currently managed within `MCPToolManager` but exposed via `!mcp-call`.
   - *Future:* Register individual MCP tools directly into the agent's tool registry for autonomous use.

3. **MCP Client Integration**:
   - The agent uses an MCP client to communicate with the Docker containers via **HTTP (SSE)**.
   - Containers are started with port mappings (e.g., `8000:8000`).
   - The agent connects to `http://localhost:<port>/sse`.

### 3. New Commands

- `!mcp-load <image>` - Pull (if needed), start, and connect to an MCP tool container.
- `!mcp-list` - List active MCP containers and their available tools.
- `!mcp-unload <container>` - Stop an MCP container and close the connection.
- `!mcp-call <container> <tool> [json_args]` - Execute a specific tool on a container.

### 4. Autonomous Workflow (Planned Phase 5)

1. **Discovery**: Agent realizes it needs a capability (e.g., "I need to query a PostgreSQL database").
2. **Search**: Agent calls `mcp_manager.search("postgresql")` (requires Catalog integration).
3. **Selection**: Agent identifies `mcp/postgresql` image.
4. **Installation**: Agent calls `mcp_manager.install("mcp/postgresql")`.
5. **Activation**: Agent calls `mcp_manager.start("mcp/postgresql", env={"POSTGRES_URL": "..."})`.
   - Container starts, exposing SSE endpoint.
   - Agent connects via MCP Client.
   - Tools are dynamically registered in `ToolRegistry`.

## Feature-Based Tool Architecture (Phase 5: Unified Features)

**Status:** IMPLEMENTED (November 22, 2025)

To unify the development of internal tools and external capabilities, we have introduced a `Feature` based architecture.

### 1. The `Feature` Class

The `Feature` class (`features/base.py`) serves as the base for all major agent capabilities (e.g., `MCPAgent`, `ModelAgent`, `SovereigntyFeature`). It provides a standard lifecycle for initialization and tool registration.

### 2. The `@tool` Decorator

We introduced a `@tool` decorator that allows developers to define tools directly on Feature methods. This eliminates the need for separate `Tool` classes and manual registration logic.

```python
class MyFeature(Feature):
    @tool(name="my_action", description="Does something", category=ToolCategory.SYSTEM)
    async def my_action(self, arg: str) -> str:
        return "Result"
```

### 3. Dynamic Registration

The `KestrelAgent` now iterates through its registered features and automatically extracts all `@tool` decorated methods, wrapping them as `AgentTool` instances and registering them in the `ToolRegistry`.

This architecture simplifies the codebase by:
- Co-locating tool logic with the feature implementation.
- Automating the registration process.
- Providing a consistent interface for both internal and external tools.

6. **Usage**: The MCP server exposes tools (e.g., `query_db`). The agent sees these in its tool list and uses them.
7. **Cleanup**: Agent calls `mcp_manager.stop("mcp/postgresql")` when done.

## Implementation Plan (Phase 4)

### Step 1: Dependencies & Setup
- Add `mcp` and `docker` to `pyproject.toml`.
- Verify Docker is running and accessible.

### Step 2: MCP Tool Manager
- Implement `MCPToolManager` class using `docker` SDK.
- Implement Docker interactions (search, pull, run, stop).

### Step 3: MCP Client & Proxy
- Implement an adapter that connects to a running MCP container via SSE.
- Convert MCP tool definitions to Kestrel `ToolSchema`.
- Forward Kestrel tool executions to the MCP server.

### Step 4: Integration & Testing
- Register `MCPToolManager` in `KestrelAgent`.
- Test with a simple MCP server (e.g., `mcp/filesystem` or `mcp/time`).

## Security Considerations

- **Sandboxing**: MCP servers run in Docker containers, providing isolation.
- **Permissions**: We must be careful about what volumes/network access we give to these containers.
- **Approval**: Autonomous installation of tools should probably require user confirmation or a specific privacy mode (e.g., only in `NORMAL` mode, not `ISOLATED`).

---

import logging

class ToolParameter(BaseModel):
    """Parameter schema for a tool"""
    name: str
    type: str  # "string", "integer", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[list] = None  # For constrained values

class ToolSchema(BaseModel):
    """Complete tool schema"""
    name: str
    description: str
    parameters: list[ToolParameter]
    returns: dict  # JSON schema of return value

class AgentTool:
    """
    Base class for agent tools.

    Tools are capabilities that can be:
    - Invoked by user commands (!search query)
    - Called autonomously by the agent (based on reasoning)
    - Used programmatically by other code

    Each tool:
    - Has a clear schema (parameters, return type)
    - Logs usage to database (performance tracking)
    - Can be enabled/disabled based on privacy mode
    - Reports errors in structured format
    """

    def __init__(self, agent_id: str, storage: 'Storage'):
        self.agent_id = agent_id
        self.storage = storage
        self.logger = logging.getLogger(f"tool.{self.name}")

    @property
    def name(self) -> str:
        """Unique tool identifier (e.g., 'web_search', 'model_manager')"""
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Human-readable description of what this tool does"""
        raise NotImplementedError

    @property
    def schema(self) -> ToolSchema:
        """Tool schema for agent reasoning and validation"""
        raise NotImplementedError

    def can_handle_command(self, user_input: str) -> bool:
        """
        Check if this tool can handle a user command.

        Examples:
        - ModelManagerTool: "!list-models", "!pull-model phi3"
        - WebSearchTool: "!search AI news"
        """
        return False  # Override in subclass

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.

        Returns:
            {
                "success": bool,
                "data": Any,  # Tool-specific result
                "error": Optional[str],
                "execution_time_ms": int
            }
        """
        raise NotImplementedError

    async def execute_from_command(self, user_input: str) -> str:
        """
        Parse user command and execute tool, returning formatted response.

        This is the bridge between commands and tools.
        """
        raise NotImplementedError

    def is_allowed_in_privacy_mode(self, privacy_mode: 'PrivacyMode') -> bool:
        """Check if tool can be used in current privacy mode"""
        return True  # Override for privacy-sensitive tools
```

### 2. Model Management Tool

```python
# tools/model_manager.py
from .base import AgentTool, ToolSchema, ToolParameter
from llm.service import LLMService
from typing import Dict, Any
import time

class ModelManagerTool(AgentTool):
    """
    Manages LLM models - discovery, pulling, cleanup, switching.

    Commands:
    - !list-models
    - !model-info <name>
    - !pull-model <name>
    - !set-model <name>
    - !storage-info
    - !cleanup-models

    Autonomous use cases:
    - Agent detects missing model → auto-pulls
    - Agent detects low disk space → suggests cleanup
    - Agent wants to switch models → changes active model
    """

    def __init__(self, agent_id: str, storage: 'Storage', llm_service: LLMService):
        super().__init__(agent_id, storage)
        self.llm_service = llm_service

    @property
    def name(self) -> str:
        return "model_manager"

    @property
    def description(self) -> str:
        return "Manages LLM models - list, pull, cleanup, switch models"

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Action to perform",
                    enum=["list", "pull", "info", "set", "storage", "cleanup"],
                    required=True
                ),
                ToolParameter(
                    name="model_name",
                    type="string",
                    description="Model name (for pull, info, set actions)",
                    required=False
                ),
                ToolParameter(
                    name="dry_run",
                    type="boolean",
                    description="Dry run mode for cleanup (don't actually delete)",
                    required=False,
                    default=False
                )
            ],
            returns={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {"type": "object"},
                    "message": {"type": "string"}
                }
            }
        )

    def can_handle_command(self, user_input: str) -> bool:
        commands = [
            "!list-models", "!model-info", "!pull-model",
            "!set-model", "!storage-info", "!cleanup-models"
        ]
        return any(user_input.startswith(cmd) for cmd in commands)

    async def execute(self, action: str, model_name: str = None, dry_run: bool = False) -> Dict[str, Any]:
        """Execute model management action"""
        start_time = time.time()

        try:
            if action == "list":
                models = await self.llm_service.discover_all_models(use_cache=True)
                return {
                    "success": True,
                    "data": {"models": models},
                    "execution_time_ms": int((time.time() - start_time) * 1000)
                }

            elif action == "pull":
                if not model_name:
                    raise ValueError("model_name required for pull action")

                result = await self.llm_service.pull_model(model_name)
                return {
                    "success": True,
                    "data": {"model": model_name, "pulled": result},
                    "message": f"Successfully pulled {model_name}",
                    "execution_time_ms": int((time.time() - start_time) * 1000)
                }

            elif action == "storage":
                storage_info = await self.llm_service.get_storage_info(use_cache=False)
                return {
                    "success": True,
                    "data": storage_info,
                    "execution_time_ms": int((time.time() - start_time) * 1000)
                }

            elif action == "cleanup":
                deleted = await self.llm_service.cleanup_unused_models(dry_run=dry_run)
                return {
                    "success": True,
                    "data": {"deleted_models": deleted, "dry_run": dry_run},
                    "message": f"Cleaned up {len(deleted)} models" if not dry_run else f"Would clean up {len(deleted)} models",
                    "execution_time_ms": int((time.time() - start_time) * 1000)
                }

            elif action == "set":
                if not model_name:
                    raise ValueError("model_name required for set action")

                self.llm_service.set_default_model(model_name)
                return {
                    "success": True,
                    "data": {"model": model_name},
                    "message": f"Default model set to {model_name}",
                    "execution_time_ms": int((time.time() - start_time) * 1000)
                }

            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            self.logger.error(f"Model manager tool failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }

    async def execute_from_command(self, user_input: str) -> str:
        """Parse command and return formatted response"""
        parts = user_input.split()
        command = parts[0]

        if command == "!list-models":
            result = await self.execute(action="list")
            if result["success"]:
                models = result["data"]["models"]
                response = "Available Models:\n\n"
                for model in models:
                    response += f"- {model['id']} ({model['provider']}) - {model.get('description', 'N/A')}\n"
                return response
            else:
                return f"Error: {result.get('error', 'Unknown error')}"

        elif command == "!pull-model":
            if len(parts) < 2:
                return "Usage: !pull-model <model_name>"
            model_name = parts[1]
            result = await self.execute(action="pull", model_name=model_name)
            return result.get("message", "Pull completed")

        elif command == "!storage-info":
            result = await self.execute(action="storage")
            if result["success"]:
                storage = result["data"]
                return f"""Storage Info:
Total: {storage['total_gb']:.1f} GB
Used: {storage['used_gb']:.1f} GB
Available: {storage['available_gb']:.1f} GB

Models: {len(storage['models'])}"""
            else:
                return f"Error: {result.get('error', 'Unknown error')}"

        elif command == "!cleanup-models":
            dry_run = "--dry-run" in parts or "-n" in parts
            result = await self.execute(action="cleanup", dry_run=dry_run)
            return result.get("message", "Cleanup completed")

        elif command == "!set-model":
            if len(parts) < 2:
                return "Usage: !set-model <model_name>"
            model_name = parts[1]
            result = await self.execute(action="set", model_name=model_name)
            return result.get("message", "Model set")

        return "Unknown command"
```

### 3. Tool Registry

```python
# tools/registry.py
from typing import Dict, List, Optional
from .base import AgentTool
from .model_manager import ModelManagerTool
from .web_search import WebSearchTool  # Existing
from .feedback import FeedbackTool  # Existing

class ToolRegistry:
    """
    Central registry of all available tools.

    Responsibilities:
    - Register and discover tools
    - Route commands to appropriate tools
    - Enable/disable tools based on privacy mode
    - Provide tool schemas for agent reasoning
    """

    def __init__(self):
        self.tools: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool):
        """Register a tool"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[AgentTool]:
        """Get tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())

    def get_schemas(self) -> List[dict]:
        """Get all tool schemas for agent reasoning"""
        return [tool.schema.dict() for tool in self.tools.values()]

    def route_command(self, user_input: str) -> Optional[AgentTool]:
        """Find tool that can handle this command"""
        for tool in self.tools.values():
            if tool.can_handle_command(user_input):
                return tool
        return None

    def get_allowed_tools(self, privacy_mode: 'PrivacyMode') -> List[AgentTool]:
        """Get tools allowed in current privacy mode"""
        return [
            tool for tool in self.tools.values()
            if tool.is_allowed_in_privacy_mode(privacy_mode)
        ]
```

### 4. Agent Integration

```python
# kestrel_agent.py modifications

class KestrelAgent:
    def __init__(self, did: str, storage: Storage, llm_service: LLMService, privacy_mode: PrivacyMode = PrivacyMode.NORMAL):
        # ... existing init ...

        # Initialize tool registry
        self.tools = ToolRegistry()

        # Register tools
        self.tools.register(ModelManagerTool(did, storage, llm_service))
        self.tools.register(WebSearchTool(did, storage))
        self.tools.register(FeedbackTool(did, storage))

        # Update prompt to include tool descriptions
        self._update_prompt_with_tools()

    def _update_prompt_with_tools(self):
        """Inject tool schemas into agent prompt for autonomous use"""
        tool_schemas = self.tools.get_schemas()
        tools_description = "\n".join([
            f"- {schema['name']}: {schema['description']}"
            for schema in tool_schemas
        ])

        # Append to prompt_template
        self.prompt_template += f"""

You have access to the following tools:
{tools_description}

When you need to use a tool, you can call it using the format:
TOOL_CALL: tool_name(param1=value1, param2=value2)

Examples:
- TOOL_CALL: model_manager(action="list")
- TOOL_CALL: web_search(query="latest AI news")
- TOOL_CALL: model_manager(action="pull", model_name="phi3:latest")
"""

    async def _handle_command(self, user_input: str) -> Optional[str]:
        """Route commands to appropriate tools"""
        tool = self.tools.route_command(user_input)
        if tool:
            return await tool.execute_from_command(user_input)

        # Fallback to legacy command handling
        return await self._handle_legacy_command(user_input)

    async def _detect_tool_calls(self, response: str) -> str:
        """
        Detect and execute tool calls in agent responses.

        If agent response contains TOOL_CALL: ..., execute the tool
        and inject results back into the conversation.
        """
        import re

        # Pattern: TOOL_CALL: tool_name(param1=value1, param2=value2)
        pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
        matches = re.finditer(pattern, response)

        for match in matches:
            tool_name = match.group(1)
            params_str = match.group(2)

            # Parse parameters (simplified - use proper parser in production)
            params = {}
            for param in params_str.split(','):
                if '=' in param:
                    key, value = param.split('=', 1)
                    params[key.strip()] = value.strip().strip('"\'')

            # Execute tool
            tool = self.tools.get_tool(tool_name)
            if tool:
                result = await tool.execute(**params)

                # Replace TOOL_CALL with result
                result_text = f"[Tool result: {result.get('message', result.get('data', 'Success'))}]"
                response = response.replace(match.group(0), result_text)

        return response
```

## Implementation Plan

### Phase 1: Core Infrastructure (1 hour)
1. Create `tools/base.py` - AgentTool base class
2. Create `tools/registry.py` - ToolRegistry
3. Add tool system to `kestrel_agent.py`

### Phase 2: Model Manager Tool (1 hour)
1. Create `tools/model_manager.py`
2. Implement all actions (list, pull, storage, cleanup, set)
3. Add command parsing

### Phase 3: Testing (1 hour)
1. Create `tests/integration/test_agent_tools_e2e.py`
2. Test user commands (`!list-models`, etc.)
3. Test autonomous tool calls
4. Test tool chaining

### Phase 4: Documentation
1. Update `docs/AGENT_TOOLS.md` with new tools
2. Add examples to README
3. Create tool development guide

## Small Model Recommendations for Tests

**Tiny models for fast testing:**
- `qwen2.5:0.5b` - 500MB (current test model) ✅
- `tinyllama:1.1b` - 600MB
- `phi3:mini` - 2.3GB
- `gemma:2b` - 1.6GB

**Never use in tests:**
- `llama3.2:70b` - 40GB
- `deepseek-r1:70b` - 40GB
- `qwen2.5:32b` - 20GB

## Success Criteria

- ✅ User can invoke tools via `!` commands
- ✅ Agent can autonomously call tools based on context
- ✅ Tools are tracked in database (usage, performance)
- ✅ Tools respect privacy modes
- ✅ Tool schemas available for agent reasoning
- ✅ All tests pass with small models (<1GB)

## Files to Create

```
tools/
├── __init__.py          # Tool exports
├── base.py              # AgentTool base class
├── registry.py          # ToolRegistry
├── model_manager.py     # ModelManagerTool
└── web_search.py        # Migrate existing

tests/integration/
└── test_agent_tools_e2e.py  # Real tool tests
```

## Files to Modify

```
kestrel_agent.py         # Add tool system
llm/service.py           # (already done - no changes)
docs/AGENT_TOOLS.md      # Update documentation
```

## Next Session TODO

1. Read this document
2. Implement Phase 1 (core infrastructure)
3. Implement Phase 2 (model manager tool)
4. Write integration tests
5. Test Emma's autonomous model management

---

**Document Status:** READY FOR IMPLEMENTATION
**Created:** November 20, 2025
**Last Updated:** November 20, 2025
