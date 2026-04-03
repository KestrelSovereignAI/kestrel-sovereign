# Refactor `kestrel-sovereign` toward a smaller permanent core and capability modules

## Problem

`kestrel-sovereign` already has the beginnings of the right architecture:

- feature discovery and dynamic loading exist in [`kestrel_sovereign/features/__init__.py`](kestrel_sovereign/features/__init__.py)
- the feature/subagent contract exists in [`kestrel_sovereign/features/base.py`](kestrel_sovereign/features/base.py)
- dynamic direct-tool promotion exists in [`kestrel_sovereign/agent/tool_registry.py`](kestrel_sovereign/agent/tool_registry.py)
- the intended direction is explicitly documented in [`docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md`](docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md)

But the runtime still keeps too much behavior in permanent structures:

- `KestrelAgent.initialize()` remains a composition root plus a behavior sink, with feature-specific wiring, state setup, bootstrap, wallet creation, memory setup, pre-exploration, security registration, reflection hookup, and default schedules all mixed together in one lifecycle path
- `KestrelAgent` still knows feature names directly, including `MCPAgent`, `ModelAgent`, `SpawnFeature`, `SecurityFeature`, `TaskFeature`, `PeersFeature`, `ConsentFeature`, `SchedulerFeature`, `ReflectionFeature`, and `VisualIdentityFeature`
- `server.py` still assembles a large always-on HTTP surface around a single heavyweight runtime instead of around a small kernel plus mounted capability routers
- some capabilities exist as durable first-class concepts but are not yet treated as first-class modules in the runtime model

This creates the core contradiction in the repo today:

> Kestrel says it is a feature-driven society of agents, but the host still behaves like a monolith with plugins.

That contradiction increases cognitive load, makes feature removal expensive, and raises the odds that future additions deepen coupling instead of expanding capability cleanly.

## Why this matters

This repo should not become smaller in ambition. It should become smaller in what is permanent.

We want to preserve the possibility space:

- sovereign identity
- privacy and memory
- spawn and delegated agents
- wallets and economics
- compute and deployment
- reflection and self-improvement
- legal personhood paths such as Wyoming DAO LLC incorporation

The Wyoming DAO path is a perfect example of the goal:

- it is a legitimate capability with dedicated models and tests
- it belongs in the platform universe
- it should not force the entire runtime to know legal workflow details

Relevant references:

- incorporation tool: [`kestrel_sovereign/legal/incorporate_tool.py`](kestrel_sovereign/legal/incorporate_tool.py)
- legal entity models: [`kestrel_sovereign/legal/models.py`](kestrel_sovereign/legal/models.py)
- identity package persistence of legal status: [`kestrel_sovereign/identity/identity_package.py`](kestrel_sovereign/identity/identity_package.py)
- cryostasis path currently touching incorporation behavior: [`kestrel_sovereign/agent/sleep.py`](kestrel_sovereign/agent/sleep.py)

The right move is not "delete weird features." The right move is:

> keep the weird and powerful features, but stop making them obligations of the core runtime.

## Current red-team read

### 1. The platform already has a module system, but core bypasses it

The docs say the Feature Agent framework is the standard architecture:

- [`docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md`](docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md)

The code supports discovery:

- [`kestrel_sovereign/features/__init__.py`](kestrel_sovereign/features/__init__.py)

The base feature abstraction exists:

- [`kestrel_sovereign/features/base.py`](kestrel_sovereign/features/base.py)

The tool registry supports feature dispatch and direct tool promotion:

- [`kestrel_sovereign/agent/tool_registry.py`](kestrel_sovereign/agent/tool_registry.py)

But `KestrelAgent` still manually coordinates named features and side effects in ways that bypass pure modular boundaries:

- feature discovery and registration in [`kestrel_sovereign/kestrel_agent.py`](kestrel_sovereign/kestrel_agent.py)
- feature-specific wiring and startup assumptions in [`kestrel_sovereign/kestrel_agent.py`](kestrel_sovereign/kestrel_agent.py)

### 2. The composition root is too smart

`KestrelAgent.initialize()` currently owns too many jobs:

- storage backend selection
- privacy wrapper creation
- task-manager store creation
- feature discovery and feature registration
- feature-specific post-load wiring
- wallet and memory initialization
- prompt loading
- key activation
- heartbeat startup
- default schedule creation

That means understanding "how Kestrel starts" requires holding most of the system in your head at once.

### 3. Feature identity is still class-name driven instead of capability driven

Feature discovery and disabling depend on class names:

- [`kestrel_sovereign/features/__init__.py`](kestrel_sovereign/features/__init__.py)

Direct lookups in core also depend on class names:

- [`kestrel_sovereign/kestrel_agent.py`](kestrel_sovereign/kestrel_agent.py)

That makes refactors brittle and encourages hidden contracts like "the host knows there will be a `SecurityFeature` and that it has `_register_all_tools()`."

### 4. HTTP surface is broad and static

`server.py` mounts a large always-on router set:

- [`server.py`](server.py)

This is convenient, but it weakens the idea that capabilities are optional and composable. If a feature is absent, the routing, health, auth, and capability model should still remain coherent.

### 5. Cross-feature state ownership is not yet explicit enough

There are several pieces of state that are real platform concerns, but their ownership boundaries are still blurry:

- privacy session state
- legal entity status
- schedule defaults
- reflection hooks
- security approvals
- feature exploration and direct-tool state

We should not erase these. We should make ownership and contracts explicit.

## Target architecture

Move toward this split:

### Permanent core

The permanent core should own only:

- runtime lifecycle
- module discovery and loading
- dependency injection / service container
- capability registration
- message/event bus
- permission and auth boundaries
- storage and state interfaces
- task routing contracts
- observability primitives

### Capability modules

Modules should own domain behavior:

- privacy
- memory
- wallet
- spawn
- scheduler
- reflection
- model management
- legal / incorporation
- deployment providers
- web search
- GitHub / external integrations

### Portable state bundles

Certain state should remain portable across substrate migrations, but not force host coupling:

- identity package contents
- legal entity status
- wallet state
- memory summaries / episodes
- feature-specific exportable state

The host should provide the envelope; modules should provide their own export/import fragments.

## Design principles for the refactor

1. Keep the possibility of all current features.
2. Shrink what the host must know by name.
3. Replace implicit feature friendships with declared contracts.
4. Separate "platform identity" from "feature inventory."
5. Make modules removable without multi-file surgery in core.
6. Treat legal/economic/reflective capabilities as optional first-class modules, not hacks and not mandatory host concerns.
7. Favor migration shims over big-bang rewrites.

## Proposed execution plan

### Phase 1: Define the kernel boundary

Deliverables:

- add a short architecture document describing the kernel/module boundary
- define what belongs in the runtime kernel vs capability modules
- define module contract primitives:
  - manifest / descriptor
  - provided capabilities
  - required capabilities
  - owned state
  - optional HTTP routers
  - startup hooks
  - export/import hooks

Suggested outputs:

- `docs/architecture/core/MODULAR_RUNTIME.md`
- `tests/unit/test_module_contracts.py`

Acceptance criteria:

- there is a written kernel boundary
- every new module can declare requirements and provided capabilities without core name checks

### Phase 2: Introduce a real module descriptor layer

Today discovery instantiates classes directly from import scanning. Keep that mechanism temporarily, but add a stable descriptor model above it.

Deliverables:

- add `ModuleDescriptor` or equivalent metadata object
- support stable module ids distinct from Python class names
- record module-provided routers, hooks, tasks, export fragments, and direct-tool policies
- preserve existing feature discovery during migration

References:

- [`kestrel_sovereign/features/__init__.py`](kestrel_sovereign/features/__init__.py)
- [`kestrel_sovereign/features/base.py`](kestrel_sovereign/features/base.py)

Acceptance criteria:

- core can reason about modules by stable id, not class name
- class renames do not break module loading

### Phase 3: Move feature-specific wiring out of `KestrelAgent.initialize()`

Deliverables:

- extract startup orchestration into a slimmer composition root or runtime bootstrapper
- move feature-specific post-load behavior behind module lifecycle hooks
- remove direct feature-name checks from startup where possible

Initial candidates:

- spawn pre-exploration
- security post-registration
- reflection sleep hook wiring
- default schedules
- model/mcp convenience references

References:

- [`kestrel_sovereign/kestrel_agent.py`](kestrel_sovereign/kestrel_agent.py)

Acceptance criteria:

- `KestrelAgent.initialize()` no longer hardcodes multiple named features for normal startup
- startup behavior can be understood as kernel setup plus module hooks

### Phase 4: Split portable state from runtime behavior

Deliverables:

- define a module state export/import contract
- move feature-owned export logic behind module boundaries
- keep `AgentIdentityPackage` as a portable envelope while making feature-owned fragments explicit

Initial candidates:

- legal entity fragment
- wallet fragment
- privacy/session fragment
- reflection fragment

References:

- [`kestrel_sovereign/identity/identity_package.py`](kestrel_sovereign/identity/identity_package.py)
- [`kestrel_sovereign/legal/models.py`](kestrel_sovereign/legal/models.py)

Acceptance criteria:

- portable state remains intact
- host does not need to understand each feature's internal serialization details

### Phase 5: Make HTTP routes capability-mounted

Deliverables:

- support optional router contribution from modules
- reduce static `server.py` knowledge of feature-owned endpoints
- keep auth and health coherent even when modules are disabled

References:

- [`server.py`](server.py)
- [`endpoints/`](endpoints/)

Acceptance criteria:

- disabling a module removes its router cleanly
- the server still boots with a meaningful reduced capability surface

### Phase 6: Extract legal personhood into a first-class module

This is a proving ground, not a novelty task.

Deliverables:

- formalize incorporation/legal-entity behavior as a feature module with its own tools, lifecycle, and export/import fragment
- keep legal entity persistence inside identity portability flows
- remove any host assumptions that legal behavior lives in core runtime paths

References:

- [`kestrel_sovereign/legal/incorporate_tool.py`](kestrel_sovereign/legal/incorporate_tool.py)
- [`kestrel_sovereign/legal/models.py`](kestrel_sovereign/legal/models.py)
- [`kestrel_sovereign/agent/sleep.py`](kestrel_sovereign/agent/sleep.py)
- [`tests/unit/test_wyoming_dao.py`](tests/unit/test_wyoming_dao.py)

Acceptance criteria:

- Wyoming DAO support still works
- legal entity status still survives export/import
- the host runtime no longer contains legal workflow knowledge beyond generic module hooks

### Phase 7: Add seam tests that enforce modularity

Deliverables:

- tests that assert module removal/addition behavior
- tests that assert no new direct class-name dependencies in core
- tests that assert module routers, hooks, and export fragments behave independently

Acceptance criteria:

- regressions toward monolithic coupling fail fast in CI

## Non-goals

- removing ambitious features because they are unusual
- collapsing everything into a generic plugin abstraction with no sovereignty model
- rewriting all features in one PR
- breaking identity portability or current exports
- deleting the current feature framework before its replacement exists

## Concrete first slices for Talon

Talon should not try to solve this epic in one shot. Recommended issue sequence:

1. Write the kernel/module boundary doc and module descriptor contract.
2. Add seam tests that capture current feature-name coupling in core.
3. Introduce stable module ids and descriptor-based loading.
4. Extract startup wiring from `KestrelAgent.initialize()` into explicit hooks.
5. Migrate one proving-ground capability end-to-end.

Recommended proving-ground modules:

- `legal`
- `scheduler`
- `reflection`

These are useful because they are:

- real
- stateful
- cross-cutting
- easy to accidentally leave half-in-core

## Risks

- over-abstracting too early and losing delivery speed
- breaking startup order assumptions hidden in current feature initialization
- introducing a descriptor layer that is more complex than the problem
- moving too much at once and losing the working behavior that makes Kestrel interesting

## Mitigations

- migrate one capability at a time
- require seam tests before and after each extraction
- preserve existing discovery as a compatibility layer during the transition
- keep a small set of temporary adapter shims and delete them only after module ownership is clear

## Suggested acceptance test for the epic

This epic is complete when the following are true:

- the runtime has a documented kernel boundary
- module loading is descriptor-driven rather than class-name-driven
- core startup no longer hardcodes a growing set of named features
- at least one cross-cutting capability has been migrated end-to-end as a true module
- portable state still supports identity, wallet, and legal-entity continuity
- the reduced-core shape is enforced by tests

## Reference map

- vision and current feature-framework intent:
  - [`docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md`](docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md)
- canonical maintained surface:
  - [`KESTREL_FEATURES.md`](KESTREL_FEATURES.md)
- feature discovery:
  - [`kestrel_sovereign/features/__init__.py`](kestrel_sovereign/features/__init__.py)
- feature/subagent base contract:
  - [`kestrel_sovereign/features/base.py`](kestrel_sovereign/features/base.py)
- direct tool promotion:
  - [`kestrel_sovereign/agent/tool_registry.py`](kestrel_sovereign/agent/tool_registry.py)
- heavy composition root and named-feature wiring:
  - [`kestrel_sovereign/kestrel_agent.py`](kestrel_sovereign/kestrel_agent.py)
- static server assembly:
  - [`server.py`](server.py)
- portable identity envelope:
  - [`kestrel_sovereign/identity/identity_package.py`](kestrel_sovereign/identity/identity_package.py)
- Wyoming DAO / legal entity support:
  - [`kestrel_sovereign/legal/incorporate_tool.py`](kestrel_sovereign/legal/incorporate_tool.py)
  - [`kestrel_sovereign/legal/models.py`](kestrel_sovereign/legal/models.py)
  - [`tests/unit/test_wyoming_dao.py`](tests/unit/test_wyoming_dao.py)

## Talon handoff note

When Talon works this epic, optimize for boundary-setting and seam extraction, not feature deletion.

The question is not:

> "Which features are too much?"

The question is:

> "Which responsibilities should stop being permanent knowledge of the host runtime?"
