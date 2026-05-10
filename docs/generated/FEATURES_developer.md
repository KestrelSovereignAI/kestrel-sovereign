<!-- AUTO-GENERATED from KESTREL_FEATURES.md — do not edit manually -->
<!-- Audience: developer | Generated: 2026-04-13 | Model: anthropic/claude-sonnet-4-6 -->
<!-- Regenerate: uv run python scripts/generate_feature_docs.py --audience developer -->

# Kestrel Sovereign — Developer Feature Reference

> **Source of truth:** [`KESTREL_FEATURES.md`](KESTREL_FEATURES.md)
> **Generated docs:** [`docs/generated/README.md`](docs/generated/README.md)
> **Historical snapshot:** [`docs/archive/KESTREL_FEATURES_legacy.md`](docs/archive/KESTREL_FEATURES_legacy.md)

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

- **Feature module discovery** — [`kestrel_sovereign/features/__init__.py`](kestrel_sovereign/features/__init__.py)
- **HTTP route families** — [`server.py`](server.py) + routers under [`endpoints/`](endpoints)
- **Doc generation script** — [`scripts/generate_feature_docs.py`](scripts/generate_feature_docs.py)

---

## Maintained Surface

### Constitutional and Sovereign Foundation

| Concern | Key files |
|---|---|
| Constitution & governance | [`kestrel_sovereign/data/KESTREL_CONSTITUTION.md`](kestrel_sovereign/data/KESTREL_CONSTITUTION.md), [`kestrel_sovereign/agent/constitution.py`](kestrel_sovereign/agent/constitution.py), [`kestrel_sovereign/features/constitution.py`](kestrel_sovereign/features/constitution.py) |
| DID identity & continuity | [`kestrel_sovereign/inception_service.py`](kestrel_sovereign/inception_service.py), [`kestrel_sovereign/identity/identity_package.py`](kestrel_sovereign/identity/identity_package.py), [`kestrel_sovereign/identity/signing.py`](kestrel_sovereign/identity/signing.py), [`kestrel_sovereign/identity/continuity_verifier.py`](kestrel_sovereign/identity/continuity_verifier.py) |
| Sovereignty lifecycle | [`kestrel_sovereign/graduate_service.py`](kestrel_sovereign/graduate_service.py), [`kestrel_sovereign/retirement_service.py`](kestrel_sovereign/retirement_service.py), [`endpoints/sovereignty.py`](endpoints/sovereignty.py) |

### Agent Runtime and Context Assembly

| Concern | Key files |
|---|---|
| Core agent orchestration | [`kestrel_sovereign/kestrel_agent.py`](kestrel_sovereign/kestrel_agent.py), [`kestrel_sovereign/command_handler.py`](kestrel_sovereign/command_handler.py), [`kestrel_sovereign/kestrel_agent_tools.py`](kestrel_sovereign/kestrel_agent_tools.py) |
| Context & token budgeting | [`kestrel_sovereign/agent/context_manager.py`](kestrel_sovereign/agent/context_manager.py), [`kestrel_sovereign/agent/context_builder.py`](kestrel_sovereign/agent/context_builder.py), [`kestrel_sovereign/agent/token_budget.py`](kestrel_sovereign/agent/token_budget.py) |
| Streaming & request lifecycle | [`kestrel_sovereign/agent/streaming.py`](kestrel_sovereign/agent/streaming.py), [`endpoints/agent.py`](endpoints/agent.py) |

> **Quick start:** To invoke the agent, `POST /agent/invoke` (blocking) or `POST /agent/stream` (SSE). See [`endpoints/agent.py`](endpoints/agent.py).

### Multi-LLM Platform

| Concern | Key files |
|---|---|
| Unified service & routing | [`kestrel_sovereign/llm/service.py`](kestrel_sovereign/llm/service.py), [`kestrel_sovereign/llm/provider_registry.py`](kestrel_sovereign/llm/provider_registry.py), [`kestrel_sovereign/llm/mandate.py`](kestrel_sovereign/llm/mandate.py) |
| Catalog, metadata, retry, usage | [`kestrel_sovereign/llm/model_catalog.py`](kestrel_sovereign/llm/model_catalog.py), [`kestrel_sovereign/llm/model_metadata.py`](kestrel_sovereign/llm/model_metadata.py), [`kestrel_sovereign/llm/retry.py`](kestrel_sovereign/llm/retry.py), [`kestrel_sovereign/llm/usage_tracking.py`](kestrel_sovereign/llm/usage_tracking.py) |

**Provider adapters** (see [`kestrel_sovereign/llm/`](kestrel_sovereign/llm/)):
`OpenAI` · `Anthropic` · `Claude Max` · `Gemini` · `Vertex AI` · `Ollama` · `OpenRouter` · `Mock`

> **Quick start:** Enumerate available models at `GET /api/models`. Switch the active model at `POST /api/model/set`. OpenAI-compatible completions available at `POST /v1/chat/completions`.

### Privacy, Storage, and Memory

| Concern | Key files |
|---|---|
| Privacy modes & enforcement | [`kestrel_sovereign/privacy.py`](kestrel_sovereign/privacy.py), [`kestrel_sovereign/features/privacy/feature.py`](kestrel_sovereign/features/privacy/feature.py), [`kestrel_sovereign/features/privacy/`](kestrel_sovereign/features/privacy) |
| Storage & persistence | [`kestrel_sovereign/storage/__init__.py`](kestrel_sovereign/storage/__init__.py), [`kestrel_sovereign/storage/async_storage.py`](kestrel_sovereign/storage/async_storage.py), [`kestrel_sovereign/storage/`](kestrel_sovereign/storage) |
| Memory systems | [`kestrel_sovereign/agent/memory_manager.py`](kestrel_sovereign/agent/memory_manager.py), [`kestrel_sovereign/features/memory/`](kestrel_sovereign/features/memory), [`kestrel_sovereign/features/memory_agency/`](kestrel_sovereign/features/memory_agency) |

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

1. **Core features** — modules under `kestrel_sovereign/features/` that export a `Feature` subclass (single-file modules, package `__init__.py`, or package `feature.py`). Only modules with a discoverable `Feature` subclass are retained.
2. **Entrypoint features** — installed via pip and registered as `kestrel_sovereign.features` entry points. Discovered at runtime via `discover_entrypoint_feature_classes()`. On duplicate class names, core features win.

> **Note:** Some support packages reside under `kestrel_sovereign/features/` but are not discoverable features because they do not export a `Feature` subclass.

> **Quick start:** Inspect the live feature inventory at `GET /api/features` and `GET /api/features/installed`. Manage individual features at `POST /api/features/{name}/enable` and `POST /api/features/{name}/disable`.

### Core Feature Inventory

Audited snapshot: **35** discoverable modules · **35** exported `Feature` subclasses.

| Module | Exported class |
|---|---|
| `audit_anchor` | `AuditAnchorFeature` |
| `bootstrap` | `BootstrapFeature` |
| `bridge` | `BridgeFeature` |
| `channels` | `ChannelFeature` |
| `compute` | `ComputeFeature` |
| `consent` | `ConsentFeature` |
| `constitution` | `ConstitutionFeature` |
| `context` | `ContextFeature` |
| `council` | `CouncilFeature` |
| `delivery` | `DeliveryFeature` |
| `deploy` | `DeployFeature` |
| `github_app` | `GitHubAppFeature` |
| `health` | `HealthFeature` |
| `identity` | `IdentityFeature` |
| `keys` | `KeyManagementFeature` |
| `memory` | `MemoryFeature` |
| `memory_agency` | `MemoryAgencyFeature` |
| `model` | `ModelAgent` |
| `peers` | `PeersFeature` |
| `reflection` | `ReflectionFeature` |
| `response_audit` | `ResponseAuditFeature` |
| `save` | `SaveFeature` |
| `scheduler` | `SchedulerFeature` |
| `security` | `SecurityFeature` |
| `sovereignty` | `SovereigntyFeature` |
| `spawn` | `SpawnFeature` |
| `state_of_mind` | `StateOfMindFeature` |
| `strategic_memory` | `StrategicMemoryFeature` |
| `talon` | `TalonCoordinatorFeature` |
| `tasks` | `TaskFeature` |
| `voice` | `VoiceFeature` |
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

### Router Families

#### [`endpoints/auth_oauth.py`](endpoints/auth_oauth.py)

| Method | Path |
|---|---|
| `GET` | `/auth/login` |
| `GET` | `/auth/callback` |
| `GET` | `/auth/logout` |
| `GET` | `/auth/me` |
| `POST` | `/auth/token` |
| `GET` | `/auth/verify` |

#### [`endpoints/agent.py`](endpoints/agent.py)

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
| `GET` | `/agent/heartbeat/status` |
| `POST` | `/agent/heartbeat/trigger` |
| `POST` | `/agent/mesh` |
| `GET` | `/agent/mesh/inbox` |

#### [`endpoints/conversations.py`](endpoints/conversations.py)

| Method | Path |
|---|---|
| `GET` | `/api/sessions` |
| `GET` | `/api/conversations` |
| `GET` | `/api/conversations/{session_id}` |
| `POST` | `/api/conversations/new` |
| `DELETE` | `/api/conversations/messages/{message_id}` |
| `GET` | `/api/conversations/{session_id}/transcript` |

#### [`endpoints/memories.py`](endpoints/memories.py)

| Method | Path |
|---|---|
| `GET` | `/api/memories` |
| `GET` | `/api/memories/{node_id}` |
| `GET` | `/api/identity-chain` |
| `DELETE` | `/api/memories/{node_id}` |

#### [`endpoints/sovereignty.py`](endpoints/sovereignty.py)

| Method | Path |
|---|---|
| `GET` | `/api/storage/stats` |
| `GET` | `/api/sovereignty/exports` |
| `POST` | `/api/sovereignty/export` |
| `POST` | `/api/sovereignty/import` |
| `GET` | `/api/sovereignty/files` |
| `GET` | `/api/sovereignty/files/{filename}` |
| `GET` | `/api/sovereignty/files/{filename}/preview` |

#### [`endpoints/database.py`](endpoints/database.py)

| Method | Path |
|---|---|
| `GET` | `/api/db/tables` |
| `GET` | `/api/db/tables/{table_name}` |

#### [`endpoints/models.py`](endpoints/models.py)

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

#### [`endpoints/commands.py`](endpoints/commands.py)

| Method | Path |
|---|---|
| `GET` | `/api/commands` |

#### [`endpoints/files.py`](endpoints/files.py)

| Method | Path |
|---|---|
| `GET` | `/api/files/{content_hash}` |
| `HEAD` | `/api/files/{content_hash}` |

#### [`endpoints/security.py`](endpoints/security.py)

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

#### [`endpoints/metrics.py`](endpoints/metrics.py)

| Method | Path |
|---|---|
| `GET` | `/metrics` |

#### [`endpoints/spawn.py`](endpoints/spawn.py)

| Method | Path |
|---|---|
| `GET` | `/api/spawn/children` |

#### [`endpoints/observability.py`](endpoints/observability.py)

| Method | Path |
|---|---|
| `GET` | `/api/observability/events` |
| `GET` | `/api/observability/summary` |

#### [`endpoints/rasa_shim.py`](endpoints/rasa_shim.py)

| Method | Path |
|---|---|
| `POST` | `/webhooks/rest/webhook` |

#### [`endpoints/saved_items.py`](endpoints/saved_items.py)

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

#### [`endpoints/voice.py`](endpoints/voice.py)

| Method | Path |
|---|---|
| `GET` | `/voice/voices` |
| `GET` | `/voice/config` |
| `POST` | `/voice/config` |
| `POST` | `/voice/tts` |
| `POST` | `/voice/tts/stream` |
| `POST` | `/voice/stt` |
| `WebSocket` | `/voice/chat` |

#### [`endpoints/features.py`](endpoints/features.py)

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
| `Public` | `/health`, `/health/detailed`, `/favicon.ico` |
| `Public-Localhost` | `/api/auth/key` (when bootstrap is enabled) |
| `OAuth public entrypoints` | `/auth/login`, `/auth/callback`, `/auth/logout` |
| `APIKeyOrSession` | Most protected `/agent/*` and `/api/*` routes |
| `APIKeyOrSession+SSEQuery` | `/agent/stream`, `/agent/notifications/sse` — also accept `?api_key=` query param |
| `OAuthSessionSemantic` | `/auth/me` — passes middleware via API key or session, but only returns authenticated data from a real OAuth session |
| `Browser-Conditional` | `/` — serves UI for local/browser conditions; redirects to OAuth when OAuth-required mode is enabled |

---

## Audit & Verification

Audit working papers: [`docs/audit/`](docs/audit)

| Test file | Coverage |
|---|---|
| [`tests/unit/test_auth_decision_table.py`](tests/unit/test_auth_decision_table.py) | Auth middleware decision table |
| [`tests/unit/test_endpoint_contract_suite.py`](tests/unit/test_endpoint_contract_suite.py) | Endpoint contract assertions |
| [`tests/unit/test_feature_doc_canonicality.py`](tests/unit/test_feature_doc_canonicality.py) | Feature doc vs. discovered classes |
| [`tests/unit/test_generate_feature_docs.py`](tests/unit/test_generate_feature_docs.py) | Doc generation pipeline |

> Generated audience docs require an LLM provider key. Dry-run validation (`scripts/generate_feature_docs.py`) must pass even when generation keys are absent.

---

## Known Boundaries

- Some packages under `kestrel_sovereign/features/` are support modules, not discoverable features — they do not export a `Feature` subclass and are excluded from the inventory.
- Entrypoint-installed feature packages extend the runtime inventory without modifying the core. They are not enumerated in this document; inspect them at runtime via `GET /api/features/installed`.
- If this document and the code disagree, fix the disagreement or mark it explicitly. Do not infer counts from stale marketing material.
