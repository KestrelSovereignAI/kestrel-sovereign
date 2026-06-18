---
type: Architecture Spec
title: Agent Tools Architecture
description: '**Status:** Active **Last updated:** 2026-05-14 **Related:** [Feature
  Agent Framework](../core/FEATURE_AGENT_FRAMEWORK.md), [Modular Runtime Boundary](../core/MODULAR_RUNTIME.md...'
resource: /docs/architecture/tools/AGENT_TOOLS_ARCHITECTURE.md
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

# Agent Tools Architecture

**Status:** Active
**Last updated:** 2026-05-14
**Related:** [Feature Agent Framework](../core/FEATURE_AGENT_FRAMEWORK.md), [Modular Runtime Boundary](../core/MODULAR_RUNTIME.md), [SDK README](https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk#readme)

Kestrel tools are feature-owned capabilities. A tool is one
`@tool`-decorated method on a `Feature` subclass. The runtime discovers
features, asks each feature for its tools, and exposes those tools to the
orchestrator either as a high-level feature dispatcher or as direct tool calls
after a feature has been explored.

The architectural direction is core-to-feature neutrality:

- core owns discovery, lifecycle, policy, dispatch, and audit
- features own tools, routers, hooks, config, and domain behavior
- optional feature packages are installed independently and discovered through
  entry points
- core must not import optional `kestrel_feature_*` packages directly

## Runtime Flow

The current tool path is:

1. A feature class defines async methods decorated with
   `kestrel_sdk.features.base.tool`.
2. `Feature.get_tools()` in `kestrel_sovereign/features/base.py` inspects those
   methods and wraps each one as a `DynamicTool`.
3. `ToolRegistryMixin._build_feature_tools()` exposes each loaded feature as a
   dispatcher tool when the feature supports feature-as-tool dispatch.
4. After a successful feature dispatch, `_register_explored_feature_tools()`
   promotes that feature's individual tools into the direct tool list.
5. Direct tools are rendered to the LLM using the SDK `ToolSchema` OpenAI-format
   adapter.
6. `DynamicTool.execute()` calls the feature method and serializes a
   `ToolResult` return value into a JSON-safe envelope.

The implementation lives in these framework files:

- `kestrel_sovereign/features/base.py` for the `Feature` base, `@tool`
  wrapping, `DynamicTool`, and feature-as-tool interface
- `kestrel_sovereign/agent/tool_registry.py` for feature dispatcher tools,
  direct-tool promotion, tool hiding, and direct-tool eviction
- `kestrel_sovereign/tools/result_contract.py` for the scoped
  `ToolResult` return-contract validator
- `kestrel_sovereign/features/__init__.py` for local and entry-point feature
  discovery

## SDK Contract

The shared tool contract comes from `kestrel-sovereign-sdk`:

- `kestrel_sdk.features.base.tool`
- `kestrel_sdk.tools.base.ToolCategory`
- `kestrel_sdk.tools.base.ToolParameter`
- `kestrel_sdk.tools.base.ToolSchema`
- `kestrel_sdk.tools.base.AgentTool`
- `kestrel_sdk.tools.result.ToolResult`
- `kestrel_sdk.tools.result.ToolResultStatus`

`ToolSchema` is the LLM-facing shape: name, description, category, parameters,
examples, optional command prefix, and concurrency metadata. `ToolResult` is the
runtime result envelope: status, confirmation, error, and data.

New or migrated tools should return `ToolResult` rather than a bare dict. The
wrapper keeps compatibility with pre-migration tools, but the honesty and audit
layers are strongest when every tool returns a structured envelope.

## Feature Packages

Features can be in-tree or out-of-tree.

In-tree features live under `kestrel_sovereign/features/<name>/`. Out-of-tree
features ship as normal Python packages, usually named `kestrel-feature-*`, and
register into the feature entry-point group:

```toml
[project.entry-points."kestrel_sovereign.features"]
my_feature = "kestrel_feature_my_feature.feature:MyFeature"
```

Feature packages should depend on `kestrel-sovereign-sdk` and any compatible
framework runtime they need. The framework deliberately does not depend on those
feature packages. Installing a package makes its entry point visible at startup.

Provider packages can be narrower than features. They register implementations
for an owning registry instead of exposing a whole feature:

```toml
[project.entry-points."kestrel_sovereign.cloud_providers"]
my_cloud = "my_cloud_package.provider:MyCloudProvider"

[project.entry-points."kestrel_sovereign.voice_providers"]
my_voice = "my_voice_package.provider:MyVoiceProvider"

[project.entry-points."kestrel_sovereign.storage_providers"]
my_storage = "my_storage_package.provider:MyStorageProvider"
```

## Feature Dispatcher vs Direct Tools

Kestrel keeps the LLM tool surface small by default. A feature initially appears
as a dispatcher tool with a feature-level description. The feature can then run
its own internal tool selection and return a result.

When a feature has been explored, its individual `@tool` methods can be promoted
as direct tools. Direct-tool promotion is bounded by
`ToolRegistryMixin.MAX_DIRECT_TOOLS`; least-recently explored features are
evicted when the direct surface grows too large.

Some meta-features opt into direct startup promotion with
`promote_tools_on_startup`. That opt-in lives on the feature, so startup remains
generic and does not branch on package names.

## Hooks, Routers, and Signals

Tools are not the only feature contribution. The same feature package can also
contribute:

- hooks through `get_hooks()`
- HTTP routes through `get_router()`
- post-discovery wiring through `post_all_features_loaded(agent)`
- signal sources or handlers through the signal dispatcher contracts

Hooks intercept work in flight. Signals originate work. Use hooks for policy,
approval, audit, and transformation around an existing turn or tool call. Use
signals for outside events that wake the agent or start new work.

## Honesty Layer

Tool honesty is enforced by several layers working together:

- `TOOL_HONESTY_SYSTEM_PROMPT` tells the model not to claim tool success before
  observing a result.
- `ToolResult` gives tools a shared success, partial, and error vocabulary.
- `DynamicTool.execute()` serializes `ToolResult` and derives the legacy
  top-level `success` flag from the result status.
- `ResponseAuditHook` folds deterministic narration checks into post-response
  audit.
- Streaming captures pre-tool prose at the `ToolCallStarted` boundary so the
  audit layer can compare what was said before tool execution with what the
  tool actually returned.

The goal is simple: user-facing narration must match observed tool outcomes.

## Boundary Rules

- Do not add new top-level tool implementations outside a feature.
- Do not make core import optional feature packages.
- Do not bypass `Feature.get_tools()` for feature-owned tools.
- Do not return ambiguous success from migrated tools; use `ToolResult`.
- Do not use hooks to originate work; use signals for that.
- Do not add feature-specific startup branches when a lifecycle method,
  descriptor, provider registry, hook, router, or signal can express the
  contribution.
