<!-- AUTO-GENERATED from KESTREL_FEATURES.md — do not edit manually -->
<!-- Audience: developer | Generated: 2026-03-16 | Model: anthropic/claude-sonnet-4-5-20250929 -->
<!-- Regenerate: uv run python scripts/generate_feature_docs.py --audience developer -->

# Kestrel Sovereign Developer Reference

**Technical reference for the Kestrel Sovereign framework**  
Engineers and AI agents integrating with or extending Kestrel should use this document to understand the maintained surface area, feature modules, and HTTP API.

---

## Quick Navigation

| Section | Purpose |
|---------|---------|
| [Maintained Surface](#maintained-surface) | Core framework components and services |
| [Feature Modules](#feature-module-inventory) | Discoverable feature inventory (41 modules, 36 exported classes) |
| [HTTP API](#public-http-surface) | Complete REST endpoint reference |
| [Authentication](#authentication-surface) | Auth model decision table |

---

## Maintained Surface

### Constitutional and Sovereign Foundation

**Constitution and governance:**
- [`kestrel_sovereign/data/KESTREL_CONSTITUTION.md`](kestrel_sovereign/data/KESTREL_CONSTITUTION.md) — canonical constitution text
- [`kestrel_sovereign/agent/constitution.py`](kestrel_sovereign/agent/constitution.py) — constitution engine
- [`kestrel_sovereign/features/constitution.py`](kestrel_sovereign/features/constitution.py) — constitution feature

**DID identity and continuity:**
- [`kestrel_sovereign/inception_service.py`](kestrel_sovereign/inception_service.py) — identity inception
- [`kestrel_sovereign/identity/identity_package.py`](kestrel_sovereign/identity/identity_package.py) — DID package format
- [`kestrel_sovereign/identity/signing.py`](kestrel_sovereign/identity/signing.py) — cryptographic signing
- [`kestrel_sovereign/identity/continuity_verifier.py`](kestrel_sovereign/identity/continuity_verifier.py) — continuity verification

**Sovereignty lifecycle:**
- [`kestrel_sovereign/graduate_service.py`](kestrel_sovereign/graduate_service.py) — graduation service
- [`kestrel_sovereign/retirement_service.py`](kestrel_sovereign/retirement_service.py) — retirement service
- [`endpoints/sovereignty.py`](endpoints/sovereignty.py) — sovereignty HTTP routes

---

### Agent Runtime and Context Assembly

**Core agent orchestration:**
- [`kestrel_sovereign/kestrel_agent.py`](kestrel_sovereign/kestrel_agent.py) — main agent class
- [`kestrel_sovereign/command_handler.py`](kestrel_sovereign/command_handler.py) — command dispatch
- [`kestrel_sovereign/kestrel_agent_tools.py`](kestrel_sovereign/kestrel_agent_tools.py) — agent tools

**Context and token budgeting:**
- [`kestrel_sovereign/agent/context_manager.py`](kestrel_sovereign/agent/context_manager.py) — context lifecycle
- [`kestrel_sovereign/agent/context_builder.py`](kestrel_sovereign/agent/context_builder.py) — context assembly
- [`kestrel_sovereign/agent/token_budget.py`](kestrel_sovereign/agent/token_budget.py) — token budget enforcement

**Streaming and request lifecycle:**
- [`kestrel_sovereign/agent/streaming.py`](kestrel_sovereign/agent/streaming.py) — SSE streaming
- [`endpoints/agent.py`](endpoints/agent.py) — agent HTTP routes

**Quick start:** Call `POST /agent/invoke` for synchronous requests or `POST /agent/stream` for SSE streaming.

---

### Multi-LLM Platform

**Unified service and routing:**
- [`kestrel_sovereign/llm/service.py`](kestrel_sovereign/llm/service.py) — LLM service abstraction
- [`kestrel_sovereign/llm/provider_registry.py`](kestrel_sovereign/llm/provider_registry.py) — provider registry
- [`kestrel_sovereign/llm/mandate.py`](kestrel_sovereign/llm/mandate.py) — mandate routing

**Supported providers:**  
OpenAI, Anthropic, Claude Max, Gemini, Vertex AI, Ollama, OpenRouter, Mock  
See [`kestrel_sovereign/llm/`](kestrel_sovereign/llm/) for adapter implementations.

**Catalog, metadata, retry, and usage tracking:**
- [`kestrel_sovereign/llm/model_catalog.py`](kestrel_sovereign/llm/model_catalog.py) — model catalog
- [`kestrel_sovereign/llm/model_metadata.py`](kestrel_sovereign/llm/model_metadata.py) — model metadata
- [`kestrel_sovereign/llm/retry.py`](kestrel_sovereign/llm/retry.py) — retry logic
- [`kestrel_sovereign/llm/usage_tracking.py`](kestrel_sovereign/llm/usage_tracking.py) — usage tracking

**Quick start:** Set model via `POST /api/model/set`, query via `GET /api/models`.

---

### Privacy, Storage, and Memory

**Privacy modes and enforcement:**
- [`kestrel_sovereign/privacy.py`](kestrel_sovereign/privacy.py) — privacy engine
- [`kestrel_sovereign/privacy_agent.py`](kestrel_sovereign/privacy_agent.py) — privacy agent
- [`kestrel_sovereign/features/privacy/`](kestrel_sovereign/features/privacy) — privacy feature

**Canonical privacy presets:**

| Preset | Storage | LLM location | Shareable | Meaning |
|--------|---------|--------------|-----------|---------|
| `ephemeral` | none | local | no | Nothing stored, local LLM only |
| `isolated` | temp | local | no | Temporary session storage, local LLM only |
| `anonymous` | scrubbed | cloud | no | Stored with PII removed, cloud LLM allowed |
| `normal` | full | cloud | no | Standard persistent storage |
| `public` | full | cloud | yes | Shareable and exportable |

**Storage and persistence:**
- [`kestrel_sovereign/storage.py`](kestrel_sovereign/storage.py) — sync storage
- [`kestrel_sovereign/async_storage.py`](kestrel_sovereign/async_storage.py) — async storage
- [`kestrel_sovereign/storage/`](kestrel_sovereign/storage) — storage implementations

**Memory systems:**
- [`kestrel_sovereign/agent/memory_manager.py`](kestrel_sovereign/agent/memory_manager.py) — memory manager
- [`kestrel_sovereign/features/memory/`](kestrel_sovereign/features/memory) — memory feature
- [`kestrel_sovereign/features/memory_agency/`](kestrel_sovereign/features/memory_agency) — memory agency feature

**Quick start:** Get/set privacy mode via `GET /agent/privacy-mode` and `POST /agent/privacy-mode`.

---

## Feature Module Inventory

**Discovery rules:** Feature modules are discovered by scanning `kestrel_sovereign/features/` for single-file features, package `__init__.py`, and package `feature.py`. See [`kestrel_sovereign/features/__init__.py`](kestrel_sovereign/features/__init__.py) for discovery logic.

**Current snapshot:** `41` discoverable modules, `36` exported `Feature` subclasses.

### Discoverable Modules

- `audit_anchor`
- `bootstrap`
- `bridge`
- `channels`
- `code_edit`
- `compute`
- `consent`
- `constitution`
- `context`
- `council`
- `delivery`
- `deploy`
- `gcp_compute`
- `github`
- `heartbeat`
- `identity`
- `keys`
- `llm_keys`
- `mcp`
- `memory`
- `memory_agency`
- `model`
- `ollama`
- `peers`
- `privacy`
- `reflection`
- `runpod`
- `save`
- `scheduler`
- `security`
- `sovereignty`
- `state_of_mind`
- `tasks`
- `training`
- `vastai`
- `vertex_ai`
- `visual_identity`
- `wallet`
- `web_search`
- `webhooks`
- `wellness`

### Exported Feature Classes

- `AuditAnchorFeature`
- `BootstrapFeature`
- `BridgeFeature`
- `ChannelFeature`
- `CodeEditFeature`
- `ComputeFeature`
- `ConsentFeature`
- `ConstitutionFeature`
- `ContextFeature`
- `CouncilFeature`
- `DeliveryFeature`
- `DeployFeature`
- `GCPComputeFeature`
- `GitHubFeature`
- `HeartbeatFeature`
- `IdentityFeature`
- `KeyManagementFeature`
- `MCPAgent`
- `MemoryAgencyFeature`
- `MemoryFeature`
- `ModelAgent`
- `PeersFeature`
- `ReflectionFeature`
- `RunPodFeature`
- `SaveFeature`
- `SchedulerFeature`
- `SecurityFeature`
- `SovereigntyFeature`
- `StateOfMindFeature`
- `TaskFeature`
- `VastAIFeature`
- `VisualIdentityFeature`
- `WalletFeature`
- `WebSearchFeature`
- `WebhookFeature`
- `WellnessFeature`

**Note:** Some modules are discovery-visible but do not currently export a `Feature` subclass.

---

## Public HTTP Surface

### App-Level Routes

Defined in [`server.py`](server.py):

- `GET /` — root UI or OAuth redirect
- `GET /api/auth/key` — bootstrap API key generation
- `GET /health` — health check
- `GET /health/detailed` — detailed health check
- `POST /webhooks/stripe/crypto` — Stripe webhook

---

### Router Families

#### Auth OAuth [`endpoints/auth_oauth.py`](endpoints/auth_oauth.py)

- `GET /auth/login`
- `GET /auth/callback`
- `GET /auth/logout`
- `GET /auth/me`

#### Agent [`endpoints/agent.py`](endpoints/agent.py)

- `POST /agent/invoke` — synchronous agent invocation
- `POST /agent/stream` — SSE streaming agent invocation
- `POST /agent/stop` — stop agent execution
- `GET /agent/info` — agent info
- `GET /agent/privacy-mode` — get privacy mode
- `POST /agent/privacy-mode` — set privacy mode
- `GET /agent/notifications` — get notifications
- `GET /agent/notifications/sse` — SSE notifications stream
- `GET /agent/context-status` — context status
- `GET /agent/reflection/status` — reflection status
- `GET /agent/tasks` — list tasks
- `GET /agent/heartbeat/status` — heartbeat status
- `POST /agent/heartbeat/trigger` — trigger heartbeat

#### Conversations [`endpoints/conversations.py`](endpoints/conversations.py)

- `GET /api/sessions`
- `GET /api/conversations`
- `GET /api/conversations/{session_id}`
- `POST /api/conversations/new`
- `DELETE /api/conversations/messages/{message_id}`
- `GET /api/conversations/{session_id}/transcript`

#### Memories [`endpoints/memories.py`](endpoints/memories.py)

- `GET /api/memories`
- `GET /api/memories/{node_id}`
- `GET /api/identity-chain`

#### Sovereignty [`endpoints/sovereignty.py`](endpoints/sovereignty.py)

- `GET /api/storage/stats`
- `GET /api/sovereignty/exports`
- `POST /api/sovereignty/export`
- `POST /api/sovereignty/import`
- `GET /api/sovereignty/files`
- `GET /api/sovereignty/files/{filename}`
- `GET /api/sovereignty/files/{filename}/preview`

#### Database [`endpoints/database.py`](endpoints/database.py)

- `GET /api/db/tables`
- `GET /api/db/tables/{table_name}`

#### Models [`endpoints/models.py`](endpoints/models.py)

- `GET /api/agents`
- `POST /api/agents`
- `DELETE /api/agents/{agent_name}`
- `GET /api/identity`
- `GET /api/constitution`
- `GET /api/ipfs/status`
- `GET /api/wallet`
- `GET /api/keys`
- `POST /api/keys`
- `PATCH /api/keys/{provider}`
- `DELETE /api/keys/{provider}`
- `GET /api/keys/{provider}/usage`
- `GET /api/models`
- `GET /api/model/current`
- `POST /api/model/set`
- `GET /v1/models` — OpenAI-compatible
- `POST /v1/chat/completions` — OpenAI-compatible

#### Commands [`endpoints/commands.py`](endpoints/commands.py)

- `GET /api/commands`

#### Files [`endpoints/files.py`](endpoints/files.py)

- `GET /api/files/{content_hash}`
- `HEAD /api/files/{content_hash}`

#### Security [`endpoints/security.py`](endpoints/security.py)

- `GET /api/security/permissions/tree`
- `POST /api/security/permissions`
- `POST /api/security/permissions/feature`
- `GET /api/security/pending`
- `POST /api/security/approve`
- `GET /api/security/audit`
- `POST /api/security/cancel/{request_id}`
- `POST /api/security/cancel-all`
- `POST /api/security/reset-session`

#### Observability [`endpoints/observability.py`](endpoints/observability.py)

- `GET /api/observability/events`
- `GET /api/observability/summary`

#### Saved Items [`endpoints/saved_items.py`](endpoints/saved_items.py)

- `GET /api/saved-items`
- `POST /api/saved-items`
- `GET /api/saved-items/stats`
- `GET /api/saved-items/schemas`
- `GET /api/saved-items/tags`
- `GET /api/saved-items/by-tag/{tag}`
- `GET /api/saved-items/by-schema/{schema_id}`
- `GET /api/saved-items/{item_id}`
- `PATCH /api/saved-items/{item_id}`
- `DELETE /api/saved-items/{item_id}`
- `POST /api/saved-items/structured`
- `POST /api/saved-items/search`
- `POST /api/saved-items/{item_id}/pin`

---

## Authentication Surface

Kestrel uses a layered authentication model. Auth middleware in [`server.py`](server.py) enforces the following classes:

| Class | Routes | Auth Method |
|-------|--------|-------------|
| **Public** | `/health`, `/health/detailed`, `/favicon.ico`, `/webhooks/stripe/crypto` | None required |
| **Public-Localhost** | `/api/auth/key` (bootstrap mode) | Localhost only |
| **OAuth entrypoints** | `/auth/login`, `/auth/callback`, `/auth/logout` | Public OAuth flow |
| **APIKeyOrSession** | Most `/agent/*` and `/api/*` routes | API key or session cookie |
| **APIKeyOrSession+SSEQuery** | `/agent/stream`, `/agent/notifications/sse` | API key (header or query param) or session |
| **OAuthSessionSemantic** | `/auth/me` | Accepts API key/session but only returns data for real sessions |
| **Browser-Conditional** | `/` | UI for local/browser, OAuth redirect otherwise |

**Quick start:** Generate API key via `GET /api/auth/key` (localhost bootstrap) or use OAuth flow for session-based access.

---

## Test Coverage and Verification

Canonical surface verification is enforced by:

- [`tests/unit/test_auth_decision_table.py`](tests/unit/test_auth_decision_table.py) — auth model contract tests
- [`tests/unit/test_endpoint_contract_suite.py`](tests/unit/test_endpoint_contract_suite.py) — endpoint contract tests
- [`tests/unit/test_feature_doc_canonicality.py`](tests/unit/test_feature_doc_canonicality.py) — feature doc canonicality tests
- [`tests/unit/test_generate_feature_docs.py`](tests/unit/test_generate_feature_docs.py) — doc generation tests

Audit working papers: [`docs/audit/`](docs/audit)

---

## Known Boundaries

- Some modules in `kestrel_sovereign/features/` are discovery-visible but do not export a `Feature` subclass
- Some route families are thin wrappers; stronger contract tests pending
- Generated audience docs require an LLM provider key; dry-run validation passes without keys

---

## Document Lineage

- **Canonical source:** [`KESTREL_FEATURES.md`](KESTREL_FEATURES.md)
- **Generation script:** [`scripts/generate_feature_docs.py`](scripts/generate_feature_docs.py)
- **Generated docs:** [`docs/generated/README.md`](docs/generated/README.md)
- **Historical archive:** [`docs/archive/KESTREL_FEATURES_legacy.md`](docs/archive/KESTREL_FEATURES_legacy.md)