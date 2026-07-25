---
type: Generated Reference
title: Developer & AI Agent Reference
description: Audience-specific developer view generated from the canonical Kestrel feature inventory.
resource: /docs/generated/FEATURES_developer.md
tags:
- features
- generated-docs
- developer
timestamp: 2026-04-13T00:00:00Z
status: generated
generated: true
canonical: false
source: /KESTREL_FEATURES.md
audience: developer
generator: scripts/generate_feature_docs.py
model: anthropic/claude-sonnet-4-6
regenerate: uv run python scripts/generate_feature_docs.py --audience developer
---

<!-- BEGIN PROTECTED PACKAGE BOUNDARY CONTRACT -->
## Package Ownership and Installation Boundaries

These ownership statements are normative and must remain intact in every
audience-specific derivative:

<!-- NON_BUNDLED_SURFACE_ALIASES: voice; mcp; github integration|github app; wallet; observability; council; visual identity; legal feature; code editing|code edit; parametric self; whatsapp; runpod; vast.ai|vastai; gcp compute; elevenlabs; deepgram; kestrel-talon -->

- **Bundled Feature lifecycle modules:** Feature subclasses discovered from
  `kestrel_sovereign/features/` ship in the `kestrel-sovereign` distribution.
  They need no separate package install. The generated inventory below is the
  exact in-tree discovery snapshot.
- **Bundled non-Feature components:** Some base-install runtime services, such
  as `PrivacyAgent`, are shipped by `kestrel-sovereign` but are not Feature
  lifecycle classes. The registry labels these `bundled-component` rather than
  putting them in its `features` field.
- **Not bundled — extracted Feature packages:** Voice, MCP, GitHub integration,
  wallet, observability, reflection, council, visual identity, legal, code
  editing, parametric self, and WhatsApp transport are separate install targets.
  They register Feature subclasses through the `kestrel_sovereign.features`
  entry-point group.
- **Not bundled — provider packages:** ElevenLabs, Deepgram, OpenAI voice, xAI
  voice/realtime, RunPod, Vast.ai, GCP Compute, and external storage backends
  implement provider contracts. They use provider-specific entry-point groups;
  installing one does not make that provider a Feature lifecycle class.
- **Not bundled — standalone tool:** `kestrel-talon` is an independently
  installed command-line issue processor. The in-tree `TalonCoordinatorFeature`
  is only its bundled Kestrel control surface; the coordinator and the
  standalone executable are separately named registry rows.

The runtime catalog at `kestrel_sovereign/data/feature_registry.toml` encodes
these distinctions in `boundary`. Its `package` field is always the owning
distribution/install target for that row. The compatibility field `core` is
`true` only for `bundled` and `bundled-component` rows. `features` contains
Feature lifecycle class names only; provider implementations use
`provider_classes` plus `entry_point_groups`, and standalone tools use
`command`. Catalog status `available` means “known but not detected in this
environment,” not a claim that an external distribution is publicly reachable.
<!-- END PROTECTED PACKAGE BOUNDARY CONTRACT -->

<!-- BEGIN PROTECTED CONTEXT HONESTY CONTRACT -->
## Context Runtime and Diagnostic Boundary

These context statements are normative and must remain intact in every
audience-specific derivative:

- A production turn preloads at most the latest **50** eligible entries from
  the active session before retrieval, budgeting, and lumpy history selection.
- Production and `GET /api/agent/context-status?full=true` use the same typed
  `ContextManager` build plan over that latest-50 input. The dry-run executes
  production relevance gates, elastic finalization, lumpy anchoring,
  microcompaction, wrapper accounting, and prune decisions without committing
  access records or salvage writes.
- The cheap `full=false` status deliberately omits memory/RAG acquisition and
  reports those sections as `unknown`/`skipped`, never as measured zero.
  Provider-native framing and stateful provider-thread occupancy remain
  separate from the Kestrel context plan.
- Default lumpy pruning omits older history from the provider window while
  retaining the source rows; it does not create an automatic durable summary.
  Automatic durable salvage is disabled by default. Its feature-flagged path is
  conditional on a pruned span mapping to id-bearing persistent history, so it
  is not a fail-closed guarantee for id-less or `ISOLATED` in-memory history.
- `openai:plan` occupancy compaction is best-effort. Kestrel resets the Codex
  thread only after durable compaction reports success; a skipped or failed
  attempt lets the turn continue with the existing thread.
- The complete all-route Context C lifecycle remains aspirational, not shipped
  behavior. The canonical current-state contract is
  `docs/architecture/CONTEXT_SYSTEM_DESIGN.md`; the separate
  `docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md` page is a design record.
<!-- END PROTECTED CONTEXT HONESTY CONTRACT -->

# Kestrel Sovereign — Developer Feature Reference

> **Source of truth:** [`KESTREL_FEATURES.md`](../../KESTREL_FEATURES.md)
> **Generated docs:** [`docs/generated/README.md`](README.md)
> **Historical snapshot:** [`docs/archive/KESTREL_FEATURES_legacy.md`](../archive/KESTREL_FEATURES_legacy.md)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Maintained Surface](#maintained-surface)
3. [Feature Module System](#feature-module-system)
4. [Public HTTP Surface](#public-http-surface)
5. [Authentication Surface](#authentication-surface)
6. [Privacy Presets](#privacy-presets)
7. [Audit & Verification](#audit--verification)
8. [Known Boundaries](#known-boundaries)

---

## Architecture Overview

Discovery rules take precedence over any headline numbers:

- **Feature module discovery** — [`kestrel_sovereign/features/__init__.py`](../../kestrel_sovereign/features/__init__.py)
- **HTTP route families** — [`server.py`](../../server.py) + routers under [`endpoints/`](../../endpoints)
- **Doc generation script** — [`scripts/generate_feature_docs.py`](../../scripts/generate_feature_docs.py)

---

## Maintained Surface

### Constitutional and Sovereign Foundation

| Concern | Key files |
|---|---|
| Constitution & governance | [`kestrel_sovereign/data/KESTREL_CONSTITUTION.md`](../../kestrel_sovereign/data/KESTREL_CONSTITUTION.md), [`kestrel_sovereign/agent/constitution.py`](../../kestrel_sovereign/agent/constitution.py), [`kestrel_sovereign/features/constitution.py`](../../kestrel_sovereign/features/constitution.py) |
| DID identity & continuity | [`kestrel_sovereign/inception_service.py`](../../kestrel_sovereign/inception_service.py), [`kestrel_sovereign/identity/identity_package.py`](../../kestrel_sovereign/identity/identity_package.py), [`kestrel_sovereign/identity/signing.py`](../../kestrel_sovereign/identity/signing.py), [`kestrel_sovereign/identity/continuity_verifier.py`](../../kestrel_sovereign/identity/continuity_verifier.py) |
| Sovereignty lifecycle | [`kestrel_sovereign/graduate_service.py`](../../kestrel_sovereign/graduate_service.py), [`kestrel_sovereign/retirement_service.py`](../../kestrel_sovereign/retirement_service.py), [`kestrel_sovereign/endpoints/sovereignty.py`](../../kestrel_sovereign/endpoints/sovereignty.py) |

### Agent Runtime and Context Assembly

| Concern | Key files |
|---|---|
| Core agent orchestration | [`kestrel_sovereign/kestrel_agent.py`](../../kestrel_sovereign/kestrel_agent.py), [`kestrel_sovereign/command_handler.py`](../../kestrel_sovereign/command_handler.py), [`kestrel_sovereign/agent/tool_registry.py`](../../kestrel_sovereign/agent/tool_registry.py) |
| Context & token budgeting | [`docs/architecture/CONTEXT_SYSTEM_DESIGN.md`](../architecture/CONTEXT_SYSTEM_DESIGN.md), [`kestrel_sovereign/agent/context_manager.py`](../../kestrel_sovereign/agent/context_manager.py), [`kestrel_sovereign/agent/context_builder.py`](../../kestrel_sovereign/agent/context_builder.py), [`kestrel_sovereign/agent/context_stages.py`](../../kestrel_sovereign/agent/context_stages.py), [`kestrel_sovereign/agent/token_budget.py`](../../kestrel_sovereign/agent/token_budget.py), [`kestrel_sovereign/storage/async_conversation_store.py`](../../kestrel_sovereign/storage/async_conversation_store.py) |
| Streaming & request lifecycle | [`kestrel_sovereign/agent/streaming.py`](../../kestrel_sovereign/agent/streaming.py), [`kestrel_sovereign/endpoints/agent.py`](../../kestrel_sovereign/endpoints/agent.py) |

> **Quick start:** To invoke the agent, `POST /agent/invoke` (blocking) or `POST /agent/stream` (SSE). See [`kestrel_sovereign/endpoints/agent.py`](../../kestrel_sovereign/endpoints/agent.py).
>
> `context_status` and `GET /api/agent/context-status` render the typed
> `ContextManager` plan. `full=true` executes the production retrieval,
> budgeting, and pruning policy without writes; cheap mode leaves omitted
> memory/RAG sections explicitly unknown. Provider-native framing remains
> outside the Kestrel plan.

### Multi-LLM Platform

| Concern | Key files |
|---|---|
| Unified service & routing | [`kestrel_sovereign/llm/service.py`](../../kestrel_sovereign/llm/service.py), [`kestrel_sovereign/llm/provider_registry.py`](../../kestrel_sovereign/llm/provider_registry.py), [`kestrel_sovereign/llm/mandate.py`](../../kestrel_sovereign/llm/mandate.py) |
| Catalog, metadata, retry, usage | [`kestrel_sovereign/llm/model_catalog.py`](../../kestrel_sovereign/llm/model_catalog.py), [`kestrel_sovereign/llm/model_metadata.py`](../../kestrel_sovereign/llm/model_metadata.py), [`kestrel_sovereign/llm/retry.py`](../../kestrel_sovereign/llm/retry.py), [`kestrel_sovereign/llm/usage_tracking.py`](../../kestrel_sovereign/llm/usage_tracking.py) |

**Provider adapters** (see [`kestrel_sovereign/llm/`](../../kestrel_sovereign/llm)):
`OpenAI` · `Anthropic` · `Claude Max` · `Gemini` · `Vertex AI` · `Ollama` · `OpenRouter` · `Mock`

> **Quick start:** Enumerate available models at `GET /api/models`. Switch the active model at `POST /api/model/set`. OpenAI-compatible completions available at `POST /v1/chat/completions`.

### Privacy, Storage, and Memory

| Concern | Key files |
|---|---|
| Privacy modes & enforcement | [`kestrel_sovereign/privacy.py`](../../kestrel_sovereign/privacy.py), [`kestrel_sovereign/features/privacy/feature.py`](../../kestrel_sovereign/features/privacy/feature.py), [`kestrel_sovereign/features/privacy/`](../../kestrel_sovereign/features/privacy) |
| Storage & persistence | [`kestrel_sovereign/storage/__init__.py`](../../kestrel_sovereign/storage/__init__.py), [`kestrel_sovereign/storage/async_storage.py`](../../kestrel_sovereign/storage/async_storage.py), [`kestrel_sovereign/storage/`](../../kestrel_sovereign/storage) |
| Memory systems | [`kestrel_sovereign/agent/memory_manager.py`](../../kestrel_sovereign/agent/memory_manager.py), [`kestrel_sovereign/features/memory/`](../../kestrel_sovereign/features/memory), [`kestrel_sovereign/features/memory_agency/`](../../kestrel_sovereign/features/memory_agency) |

---

## Privacy Presets

Read or set the active preset at `GET /agent/privacy-mode` / `POST /agent/privacy-mode`.

| Preset | Storage | LLM location | Shareable | Notes |
|---|---|---|---|---|
| `ephemeral` | none | local | no | Nothing stored; local LLM only |
| `isolated` | temp | local | no | Temporary session storage; local LLM only |
| `anonymous` | scrubbed | cloud | no | Stored with PII removed; cloud LLM allowed |
| `normal` | full | cloud | no | Standard persistent storage |
| `public` | full | cloud | yes | Shareable and exportable |

---

## Feature Module System

### Discovery

Features are discovered from two sources:

1. **Bundled Feature modules** — modules under `kestrel_sovereign/features/` that export a `Feature` subclass (single-file modules, package `__init__.py`, or package `feature.py`). Only modules with a discoverable `Feature` subclass are retained.
2. **Extracted Feature packages** — installed via pip and registered as `kestrel_sovereign.features` entry points. Discovered at runtime via `discover_entrypoint_feature_classes()`. On duplicate class names, bundled modules win.

> **Note:** Some support packages reside under `kestrel_sovereign/features/` but are not discoverable features because they do not export a `Feature` subclass.

> **Quick start:** Inspect the live feature inventory at `GET /api/features` and `GET /api/features/installed`. Manage individual features at `POST /api/features/{name}/enable` and `POST /api/features/{name}/disable`.

### Bundled Feature Inventory

Audited snapshot: **37** discoverable modules · **37** exported `Feature` subclasses.

| Module | Exported class |
|---|---|
| `attachments` | `AttachmentsFeature` |
| `audit_anchor` | `AuditAnchorFeature` |
| `bootstrap` | `BootstrapFeature` |
| `bridge` | `BridgeFeature` |
| `channels` | `ChannelFeature` |
| `compute` | `ComputeFeature` |
| `computer_use` | `ComputerUseFeature` |
| `consent` | `ConsentFeature` |
| `constitution` | `ConstitutionFeature` |
| `context` | `ContextFeature` |
| `delivery` | `DeliveryFeature` |
| `deploy` | `DeployFeature` |
| `health` | `HealthFeature` |
| `identity` | `IdentityFeature` |
| `keys` | `KeyManagementFeature` |
| `memory` | `MemoryFeature` |
| `memory_agency` | `MemoryAgencyFeature` |
| `model` | `ModelAgent` |
| `peers` | `PeersFeature` |
| `response_audit` | `ResponseAuditFeature` |
| `restart_coordinator` | `RestartCoordinatorFeature` |
| `save` | `SaveFeature` |
| `scheduler` | `SchedulerFeature` |
| `security` | `SecurityFeature` |
| `skills` | `SkillsFeature` |
| `sovereignty` | `SovereigntyFeature` |
| `spawn` | `SpawnFeature` |
| `state_of_mind` | `StateOfMindFeature` |
| `strategic_memory` | `StrategicMemoryFeature` |
| `talon` | `TalonCoordinatorFeature` |
| `tasks` | `TaskFeature` |
| `todo` | `TodoFeature` |
| `wait` | `WaitFeature` |
| `web_search` | `WebSearchFeature` |
| `webhooks` | `WebhookFeature` |
| `wellness` | `WellnessFeature` |

---

## Public HTTP Surface

### App-Level Routes (`server.py`)

| Method | Path |
|---|---|
| `GET` | `/` |
| `GET` | `/api/auth/key` |
| `GET` | `/api/github/{path:path}` |
| `GET` | `/health` |
| `GET` | `/health/detailed` |

`/health` is the unauthenticated aggregate load-balancer probe and returns only
`status` plus `agent_initialized`. `/health/detailed` requires an API key,
bearer JWT, or OAuth session and returns operator diagnostics.

### Router Families

#### [`kestrel_sovereign/endpoints/auth_oauth.py`](../../kestrel_sovereign/endpoints/auth_oauth.py)

| Method | Path |
|---|---|
| `GET` | `/auth/login` |
| `GET` | `/auth/callback` |
| `GET` | `/auth/logout` |
| `GET` | `/auth/me` |
| `POST` | `/auth/token` |
| `GET` | `/auth/verify` |

#### [`kestrel_sovereign/endpoints/agent.py`](../../kestrel_sovereign/endpoints/agent.py)

| Method | Path |
|---|---|
| `POST` | `/agent/invoke` |
| `POST` | `/agent/stream` |
| `POST` | `/agent/stop` |
| `GET` | `/agent/info` |
| `GET` | `/agent/privacy-mode` |
| `POST` | `/agent/privacy-mode` |
| `GET` | `/agent/notifications` |
| `GET` | `/agent/notifications/sse` |
| `GET` | `/agent/context-status` |
| `GET` | `/agent/reflection/status` |
| `GET` | `/agent/tasks` |
| `POST` | `/agent/tasks/send` |
| `GET` | `/agent/heartbeat/status` |
| `POST` | `/agent/heartbeat/trigger` |

#### [`kestrel_sovereign/endpoints/conversations.py`](../../kestrel_sovereign/endpoints/conversations.py)

| Method | Path |
|---|---|
| `GET` | `/api/sessions` |
| `GET` | `/api/conversations` |
| `GET` | `/api/conversations/{session_id}` |
| `POST` | `/api/conversations/new` |
| `DELETE` | `/api/conversations/messages/{message_id}` |
| `GET` | `/api/conversations/{session_id}/transcript` |

#### [`kestrel_sovereign/endpoints/memories.py`](../../kestrel_sovereign/endpoints/memories.py)

| Method | Path |
|---|---|
| `GET` | `/api/memories` |
| `GET` | `/api/memories/{node_id}` |
| `GET` | `/api/identity-chain` |
| `DELETE` | `/api/memories/{node_id}` |

#### [`kestrel_sovereign/endpoints/sovereignty.py`](../../kestrel_sovereign/endpoints/sovereignty.py)

| Method | Path |
|---|---|
| `GET` | `/api/storage/stats` |
| `GET` | `/api/sovereignty/exports` |
| `POST` | `/api/sovereignty/export` |
| `POST` | `/api/sovereignty/import` |
| `GET` | `/api/sovereignty/files` |
| `GET` | `/api/sovereignty/files/{filename}` |
| `GET` | `/api/sovereignty/files/{filename}/preview` |

#### [`kestrel_sovereign/endpoints/database.py`](../../kestrel_sovereign/endpoints/database.py)

| Method | Path |
|---|---|
| `GET` | `/api/db/tables` |
| `GET` | `/api/db/tables/{table_name}` |

#### [`kestrel_sovereign/endpoints/models.py`](../../kestrel_sovereign/endpoints/models.py)

| Method | Path |
|---|---|
| `GET` | `/api/agents` |
| `POST` | `/api/agents` |
| `DELETE` | `/api/agents/{agent_name}` |
| `GET` | `/api/identity` |
| `PATCH` | `/api/identity` |
| `POST` | `/api/identity/avatar` |
| `POST` | `/api/identity/avatar/generate` |
| `GET` | `/api/constitution` |
| `GET` | `/api/ipfs/status` |
| `GET` | `/api/wallet` |
| `GET` | `/api/keys` |
| `POST` | `/api/keys` |
| `PATCH` | `/api/keys/{provider}` |
| `DELETE` | `/api/keys/{provider}` |
| `GET` | `/api/keys/{provider}/usage` |
| `GET` | `/api/models` |
| `GET` | `/api/model/current` |
| `POST` | `/api/model/set` |
| `GET` | `/v1/models` |
| `POST` | `/v1/chat/completions` |

#### [`kestrel_sovereign/endpoints/commands.py`](../../kestrel_sovereign/endpoints/commands.py)

| Method | Path |
|---|---|
| `GET` | `/api/commands` |

#### [`kestrel_sovereign/endpoints/files.py`](../../kestrel_sovereign/endpoints/files.py)

| Method | Path |
|---|---|
| `GET` | `/api/files/{content_hash}` |
| `HEAD` | `/api/files/{content_hash}` |

#### [`kestrel_sovereign/endpoints/security.py`](../../kestrel_sovereign/endpoints/security.py)

| Method | Path |
|---|---|
| `GET` | `/api/security/permissions/tree` |
| `POST` | `/api/security/permissions` |
| `POST` | `/api/security/permissions/feature` |
| `GET` | `/api/security/pending` |
| `POST` | `/api/security/approve` |
| `GET` | `/api/security/audit` |
| `POST` | `/api/security/cancel/{request_id}` |
| `POST` | `/api/security/cancel-all` |
| `POST` | `/api/security/reset-session` |

#### [`kestrel_sovereign/endpoints/metrics.py`](../../kestrel_sovereign/endpoints/metrics.py)

| Method | Path |
|---|---|
| `GET` | `/metrics` |

#### [`kestrel_sovereign/endpoints/spawn.py`](../../kestrel_sovereign/endpoints/spawn.py)

| Method | Path |
|---|---|
| `GET` | `/api/spawn/children` |

#### [`kestrel_sovereign/endpoints/observability.py`](../../kestrel_sovereign/endpoints/observability.py)

| Method | Path |
|---|---|
| `GET` | `/api/observability/summary` |

#### [`kestrel_sovereign/endpoints/rasa_shim.py`](../../kestrel_sovereign/endpoints/rasa_shim.py)

| Method | Path |
|---|---|
| `POST` | `/webhooks/rest/webhook` |

#### [`kestrel_sovereign/endpoints/saved_items.py`](../../kestrel_sovereign/endpoints/saved_items.py)

| Method | Path |
|---|---|
| `GET` | `/api/saved-items` |
| `POST` | `/api/saved-items` |
| `GET` | `/api/saved-items/stats` |
| `GET` | `/api/saved-items/schemas` |
| `GET` | `/api/saved-items/tags` |
| `GET` | `/api/saved-items/by-tag/{tag}` |
| `GET` | `/api/saved-items/by-schema/{schema_id}` |
| `GET` | `/api/saved-items/{item_id}` |
| `PATCH` | `/api/saved-items/{item_id}` |
| `DELETE` | `/api/saved-items/{item_id}` |
| `POST` | `/api/saved-items/structured` |
| `POST` | `/api/saved-items/search` |
| `POST` | `/api/saved-items/{item_id}/pin` |

#### [`kestrel_sovereign/endpoints/features.py`](../../kestrel_sovereign/endpoints/features.py)

| Method | Path |
|---|---|
| `GET` | `/api/features` |
| `GET` | `/api/features/installed` |
| `GET` | `/api/features/{name}` |
| `POST` | `/api/features/{name}/install` |
| `POST` | `/api/features/{name}/enable` |
| `POST` | `/api/features/{name}/disable` |
| `POST` | `/api/features/{name}/remove` |
| `GET` | `/api/features/{name}/config` |
| `PATCH` | `/api/features/{name}/config` |
| `GET` | `/api/features/{name}/skills` |
| `GET` | `/api/skills` |
| `GET` | `/api/skills/{skill_id}/schema` |

---

## Authentication Surface

Auth is enforced by middleware in `server.py`. The live auth classes are:

| Auth class | Applies to |
|---|---|
| `Public` | `/health` (aggregate readiness only), `/favicon.ico` |
| `Public-Localhost` | `/api/auth/key` (when bootstrap is enabled) |
| `OAuth public entrypoints` | `/auth/login`, `/auth/callback`, `/auth/logout` |
| `APIKeyOrSession` | `/health/detailed`; most protected `/agent/*` and `/api/*` routes |
| `APIKeyOrSession+SSEQuery` | `/agent/stream`, `/agent/notifications/sse` — also accept `?api_key=` query param |
| `OAuthSessionSemantic` | `/auth/me` — passes middleware via API key or session, but only returns authenticated data from a real OAuth session |
| `Browser-Conditional` | `/` — serves UI for local/browser conditions; redirects to OAuth when OAuth-required mode is enabled |

---

## Audit & Verification

Audit working papers: [`docs/audit/`](../audit)

| Test file | Coverage |
|---|---|
| [`tests/unit/test_auth_decision_table.py`](../../tests/unit/test_auth_decision_table.py) | Auth middleware decision table |
| [`tests/unit/test_endpoint_contract_suite.py`](../../tests/unit/test_endpoint_contract_suite.py) | Endpoint contract assertions |
| [`tests/unit/test_feature_doc_canonicality.py`](../../tests/unit/test_feature_doc_canonicality.py) | Feature doc vs. discovered classes |
| [`tests/unit/test_generate_feature_docs.py`](../../tests/unit/test_generate_feature_docs.py) | Doc generation pipeline |

> Generated audience docs require an LLM provider key. Dry-run validation (`scripts/generate_feature_docs.py`) must pass even when generation keys are absent.

---

## Known Boundaries

- Some packages under `kestrel_sovereign/features/` are support modules, not discoverable features — they do not export a `Feature` subclass and are excluded from the inventory.
- Entrypoint-installed feature packages extend the runtime inventory without modifying the core. They are not enumerated in this document; inspect them at runtime via `GET /api/features/installed`.
- If this document and the code disagree, fix the disagreement or mark it explicitly. Do not infer counts from stale marketing material.
