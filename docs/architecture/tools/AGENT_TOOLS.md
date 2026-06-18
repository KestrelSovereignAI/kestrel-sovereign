---
type: Architecture Spec
title: Agent Tools
description: '**Status:** Active **Last updated:** 2026-05-14 **Related:** [Agent
  Tools Architecture](AGENT_TOOLS_ARCHITECTURE.md), [Agent Tools Implementation](AGENT_TOOLS_IMPLEMENTATION.md)...'
resource: /docs/architecture/tools/AGENT_TOOLS.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# Agent Tools

**Status:** Active
**Last updated:** 2026-05-14
**Related:** [Agent Tools Architecture](AGENT_TOOLS_ARCHITECTURE.md), [Agent Tools Implementation](AGENT_TOOLS_IMPLEMENTATION.md), [Feature Agent Framework](../core/FEATURE_AGENT_FRAMEWORK.md)

Kestrel tools are actions exposed by features. They can be called by the
orchestrator LLM, reached through command prefixes, or used by other runtime
paths after feature discovery. The tool surface is dynamic: it depends on which
core features are enabled and which optional feature packages are installed.

For the canonical live inventory, use `KESTREL_FEATURES.md` and runtime feature
discovery. This document explains the tool model and current categories rather
than trying to maintain a complete hand-written catalog.

## Tool Model

| Term | Meaning |
|---|---|
| Tool | One `@tool`-decorated async method on a feature |
| Feature | A bundle of tools, hooks, routers, config, and lifecycle behavior |
| Feature dispatcher | The high-level tool that lets the orchestrator delegate to a feature |
| Direct tool | An individual feature tool promoted into the LLM tool list |
| ToolResult | The structured result envelope used by migrated tools |

The runtime starts with feature dispatcher tools to keep the LLM context small.
Individual tools can be promoted after the feature has been explored or when a
feature opts into startup promotion.

## Categories

Tool categories come from `kestrel_sdk.tools.base.ToolCategory`:

- `MODEL_MANAGEMENT`
- `FILE_OPERATIONS`
- `WEB_SEARCH`
- `MEMORY`
- `COMMUNICATION`
- `SYSTEM`
- `DATA_ACCESS`
- `COMPUTE`
- `UTILITY`
- `AGENT_MANAGEMENT`

Categories help organize tools for schemas, docs, and UI. They are not a
permission system by themselves; policy still belongs to privacy, security,
consent, hooks, and feature-specific checks.

## Result Envelope

Migrated tools return `ToolResult`:

```python
from kestrel_sdk.tools.result import ToolResult

return ToolResult.ok(
    confirmation="Indexed 3 documents.",
    data={"documents": 3},
)
```

The envelope has four fields:

- `status`: `ok`, `partial`, or `error`
- `confirmation`: user-facing success/caveat text when available
- `error`: failure or caveat text when available
- `data`: structured machine-readable payload

Use the constructors:

- `ToolResult.ok(...)` for complete success
- `ToolResult.partial(...)` for partial success or a caveat
- `ToolResult.failed(...)` for failure

The dynamic wrapper serializes this envelope before it reaches downstream
callers.

## Commands

A tool can opt into command-style invocation with `command_prefix`:

```python
@tool(
    name="web_search",
    description="Search the web",
    category=ToolCategory.WEB_SEARCH,
    command_prefix="!search",
)
async def web_search(self, query: str) -> ToolResult:
    ...
```

Commands are convenience affordances layered on top of the same feature-owned
tool. They should not duplicate tool logic.

## Where Tools Come From

Current tool sources include:

- core features under `kestrel_sovereign/features/`
- optional feature packages registered in `kestrel_sovereign.features`
- provider implementations registered in provider-specific entry-point groups
  and surfaced by their owning features

Examples of feature-owned domains include web search, model management, memory,
security, channels, delivery, tasks, peers, spawn, workflows, compute, and
response audit. Optional packages add domains such as MCP, wallet, GitHub,
observability, voice, reflection, council, visual, code, and cloud providers
when installed.

## Adding a Tool

Add tools inside a feature:

1. Choose the owning feature or create a new feature package.
2. Add an async method decorated with `@tool`.
3. Return `ToolResult` for new or migrated tools.
4. Add tests for direct method execution and `Feature.get_tools()` execution.
5. Use a command prefix only when users or operators need stable command syntax.
6. Keep feature-specific HTTP endpoints in `get_router()`, not in unrelated
   core server imports.

Minimal shape:

```python
from kestrel_sdk.features.base import tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature


class ExampleFeature(Feature):
    @property
    def tool_description(self) -> str:
        return "Demonstrate a feature-owned tool"

    async def initialize(self):
        pass

    @tool("example_echo", "Echo text back", ToolCategory.UTILITY)
    async def example_echo(self, text: str) -> ToolResult:
        """Echo text back.

        Args:
            text: Text to echo.
        """
        return ToolResult.ok(
            confirmation="Echoed text.",
            data={"text": text},
        )
```

## Honesty Expectations

Tool narration must be grounded in observed results:

- before a tool returns, use present-progressive language such as "Saving that
  now" or "Looking that up"
- after a tool returns, reflect its actual `ToolResult`
- if the status is `partial` or `error`, surface the caveat plainly
- do not claim a save, send, delete, publish, or other action succeeded unless
  the result confirms it

The prompt, result envelope, streaming `ToolCallStarted` boundary, and
`ResponseAuditHook` all exist to make this honest behavior enforceable.

## What Remains Host-Owned

The framework owns:

- feature discovery and lifecycle
- the runtime tool catalog
- direct-tool promotion and eviction
- hook dispatch and signal dispatch
- prompt guardrails and response audit
- shared policy surfaces such as privacy, consent, and security

Feature packages own their tools and domain behavior. Provider packages own
provider implementations. Keep those boundaries intact when adding new
capability.
