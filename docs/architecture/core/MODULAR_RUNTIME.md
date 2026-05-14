# Modular Runtime Boundary

**Status:** Contract scaffold
**Parent issues:** #413, #416, #419
**Last updated:** 2026-05-14

This document names the runtime boundary Kestrel is moving toward. It is a
vocabulary and contract scaffold, not a runtime rewrite. Later issues can add
descriptor objects, router migration, state-fragment plumbing, and stronger seam
tests against this language.

The rule is simple: the permanent framework owns the kernel, and features own
capabilities. The kernel may discover, validate, mount, schedule, and observe
modules. It should not import optional feature packages or bake feature-specific
startup behavior into the host.

## Runtime Kernel

The runtime kernel is the smallest host that must exist before any optional
feature can run. It owns:

- process startup and shutdown
- configuration loading and operator policy
- feature discovery from local core modules and Python entry points
- the tool registry and feature-as-tool dispatch surface
- hook registration and unregistration lifecycle
- signal dispatch and supervised background work
- shared storage, identity envelope, audit, and security enforcement
- HTTP application composition, including mounting feature routers
- capability validation and conflict handling

The kernel can call declared feature lifecycle methods. It should not know that
MCP, wallet, GitHub, voice, observability, reflection, council, or cloud-provider
packages exist by import path. Optional packages depend on the framework or SDK;
the framework does not depend on optional packages.

## Module

A module is a capability provider loaded by the kernel. Today, most modules are
`Feature` subclasses discovered from `kestrel_sovereign/features/` or from the
`kestrel_sovereign.features` entry-point group. Provider-only packages can use a
more specific registry, such as cloud, voice, or storage provider entry points.

A module owns its implementation details:

- tools and feature-as-tool behavior
- hooks returned from `get_hooks()`
- optional FastAPI router returned from `get_router()`
- post-discovery wiring in `post_all_features_loaded(agent)`
- startup direct-tool promotion through `promote_tools_on_startup`
- config schema and persisted feature config
- feature-owned state and future export/import fragments

The host may pass shared services to a module through the agent object or SDK
contracts. A module should not require the host to import a package-specific
symbol to make the module visible.

## Stable Module ID

Every module needs a stable module id: the durable name used by registries,
configuration, permissions, audit records, migration plans, and user-facing
feature inventory. A stable module id is not a Python class name and not an
import path.

For package features, the stable id should match the feature registry key and
entry-point name when practical:

```toml
[project.entry-points."kestrel_sovereign.features"]
mcp = "kestrel_feature_mcp.feature:MCPFeature"
```

Core modules should likewise have one durable registry name even if their class
or file layout changes. Later descriptor work should make this id explicit on
the module contract. Until then, tests and docs should avoid treating class names
as the permanent identity of a feature.

## Provided Capabilities

A module's provided capabilities are the things it contributes to the runtime.
The current contract already supports these contributions:

- tools discovered from feature methods
- a feature-as-tool dispatcher via `tool_name` and `tool_description`
- hooks via `get_hooks()`
- routers via `get_router()`
- post-load wiring via `post_all_features_loaded(agent)`
- startup direct-tool promotion via `promote_tools_on_startup`
- config schema and config persistence helpers

Provider packages are also capabilities, but they do not always need to be
features. For example, cloud, voice, and storage implementations can register
under provider-specific entry-point groups and be consumed by their owning core
or feature module.

## Required Capabilities

A module's required capabilities are the services it needs from the host or from
other modules. Examples include storage, LLM routing, hooks, signals, wallet,
channel adapters, delivery adapters, or a provider registry.

Required capabilities should be declared through a stable contract instead of
hidden host assumptions. Until descriptor work lands, a module should make these
requirements visible through documentation, config schema, clear initialization
errors, and tests. Optional cross-feature wiring belongs in
`post_all_features_loaded(agent)` so discovery can complete before one module
looks up another.

## Owned State

The kernel owns shared envelopes and enforcement surfaces. Modules own their own
domain state.

Current examples:

- the identity package is the portable envelope for agent identity
- feature config persists through feature-owned config nodes
- storage, audit, and security policies are host-provided enforcement surfaces
- feature state should be accessed through module APIs or shared contracts, not
  by reaching into another module's private data

Future work should define export/import fragments so a module can contribute its
portable state to the identity or sovereignty envelope without making the kernel
understand that module's internals.

## Startup Hooks

Startup must be generic. The kernel may call the feature lifecycle:

1. discover modules
2. instantiate allowed modules
3. call `initialize()`
4. register hooks from `get_hooks()`
5. register tools and feature-as-tool dispatch
6. call `post_all_features_loaded(agent)` after discovery is complete
7. promote startup direct tools only when `promote_tools_on_startup` opts in

Feature-specific startup branches are a boundary smell. A feature that needs
special startup behavior should express it through lifecycle methods, hooks,
signals, provider registries, or an explicit descriptor in future work.

## Router Contribution

Modules that expose HTTP endpoints should return an `APIRouter` from
`get_router()`. The host mounts contributed routers during app composition. This
lets a package feature expose a control surface without adding a bespoke import
to `server.py` or `endpoints/__init__.py`.

Static core endpoints can remain in the host when they are truly kernel,
operator, or compatibility surfaces. New feature-specific endpoints should prefer
router contribution.

## Export and Import Fragments

The long-term state contract is feature-owned export/import fragments:

- the kernel owns the export/import envelope and verification path
- each module owns serialization and validation for its own portable fragment
- import should route a fragment back to the module that declared ownership
- missing optional modules should leave fragments preserved or explicitly
  skipped, not silently discarded

This scaffold does not implement that contract. It names the boundary so future
work can add it without mixing legal, wallet, voice, or other feature-specific
state into the kernel.

## Entry-Point Groups

The current extension groups are:

- `kestrel_sovereign.features` for feature packages
- `kestrel_sovereign.cloud_providers` for cloud provider packages
- `kestrel_sovereign.voice_providers` for voice provider packages
- `kestrel_sovereign.storage_providers` for storage provider packages

Feature packages should register a `Feature` subclass. Provider packages should
register provider implementations for the owning registry. Core must not import
`kestrel_feature_*` packages directly; package installation makes them visible
through entry points.

## Non-Goals for This Scaffold

This document does not:

- add a descriptor class
- change feature discovery
- move routes
- move runtime state
- change startup behavior
- extract additional source packages

Those changes belong in follow-up issues that can point back to this contract.
