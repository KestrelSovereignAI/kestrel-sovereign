<!-- AUTO-GENERATED from KESTREL_FEATURES.md — do not edit manually -->
<!-- Audience: developer | Generated: 2026-03-17 | Model: anthropic/claude-sonnet-4-5-20250929 -->
<!-- Regenerate: uv run python scripts/generate_feature_docs.py --audience developer -->

# Kestrel Sovereign Developer Reference

**Target audience:** Software engineers and AI agents integrating with or extending Kestrel Sovereign.

**Source of truth:** This document is generated from [`KESTREL_FEATURES.md`](../KESTREL_FEATURES.md). For canonical definitions, refer to the source.

---

## Quick Navigation

- [Constitutional & Identity](#constitutional-and-identity-layer) — DID identity, constitution, lifecycle
- [Agent Runtime](#agent-runtime-and-orchestration) — Core agent, tools, context, streaming
- [LLM Platform](#multi-llm-platform) — Provider routing, catalog, retry, usage
- [Privacy & Storage](#privacy-storage-and-memory) — Privacy modes, persistence, memory systems
- [Feature Modules](#feature-module-inventory) — 36 discoverable features
- [HTTP API](#http-api-surface) — All public routes and router families
- [Authentication](#authentication-model) — Route protection classes

---

## Constitutional and Identity Layer

### DID Identity and Signing

- **Inception service:** [`kestrel_sovereign/inception_service.py`](../kestrel_sovereign/inception_service.py)
- **Identity package:** [`kestrel_sovereign/identity/identity_package.py`](../kestrel_sovereign/identity/identity_package.py)
- **Signing:** [`kestrel_sovereign/identity/signing.py`](../kestrel_sovereign/identity/signing.py)
- **Continuity verification:** [`kestrel_sovereign/identity/continuity_verifier.py`](../kestrel_sovereign/identity/continuity_verifier.py)

### Constitution and Governance

- **Constitution document:** [`kestrel_sovereign/data/KESTREL_CONSTITUTION.md`](../kestrel_sovereign/data/KESTREL_CONSTITUTION.md)
- **Agent integration:** [`kestrel_sovereign/agent/constitution.py`](../kestrel_sovereign/agent/constitution.py)
- **Feature integration:** [`kestrel_sovereign/features/constitution.py`](../kestrel_sovereign/features/constitution.py)

### Sovereignty Lifecycle

- **Graduation:** [`kestrel_sovereign/graduate_service.py`](../kestrel_sovereign/graduate_service.py)
- **Retirement:** [`kestrel_sovereign/retirement_service.py`](../kestrel_sovereign/retirement_service.py)
- **HTTP routes:** [`endpoints/sovereignty.py`](../endpoints/sovereignty.py)

---

## Agent Runtime and Orchestration

### Core Agent

- **Main orchestrator:** [`kestrel_sovereign/kestrel_agent.py`](../kestrel_sovereign/kestrel_agent.py)
- **Command handler:** [`kestrel_sovereign/command_handler.py`](../kestrel_sovereign/command_handler.py)
- **Built-in tools:** [`kestrel_sovereign/kestrel_agent_tools.py`](../kestrel_sovereign/kestrel_agent_tools.py)

**Quick start:** Call `POST /agent/invoke` for synchronous execution or `POST /agent/stream` for streaming responses.

### Context and Token Management

- **Context manager:** [`kestrel_sovereign/agent/context_manager.py`](../kestrel_sovereign/agent/context_manager.py)
- **Context builder:** [`kestrel_sovereign/agent/context_builder.py`](../kestrel_sovereign/agent/context_builder.py)
- **Token budgeting:** [`kestrel_sovereign/agent/token_budget.py`](../kestrel_sovereign/agent/token_budget.py)

**Check status:** `GET /agent/context-status`

### Streaming and Lifecycle

- **Streaming:** [`kestrel_sovereign/agent/streaming.py`](../kestrel_sovereign/agent/streaming.py)
- **HTTP endpoints:** [`endpoints/agent.py`](../endpoints/agent.py)

**SSE endpoint:** `GET /agent/notifications/sse` (supports `?api_key=` query auth)

---

## Multi-LLM Platform

### Unified Service and Routing

- **LLM service:** [`kestrel_sovereign/llm/service.py`](../kestrel_sovereign/llm/service.py)
- **Provider registry:** [`kestrel_sovereign/llm/provider_registry.py`](../kestrel_sovereign/llm/provider_registry.py)
- **Mandate system:** [`kestrel_sovereign/llm/mandate.py`](../kestrel_sovereign/llm/mandate.py)

### Supported Providers

Adapters present in [`kestrel_sovereign/llm/`](../kestrel_sovereign/llm/):

- OpenAI
- Anthropic
- Claude Max
- Gemini
- Vertex AI
- Ollama
- OpenRouter
- Mock (for testing)

### Catalog, Metadata, and Tracking

- **Model catalog:** [`kestrel_sovereign/llm/model_catalog.py`](../kestrel_sovereign/llm/model_catalog.py)
- **Model metadata:** [`kestrel_sovereign/llm/model_metadata.py`](../kestrel_sovereign/llm/model_metadata.py)
- **Retry logic:** [`kestrel_sovereign/llm/retry.py`](../kestrel_sovereign/llm/retry.py)
- **Usage tracking:** [`kestrel_sovereign/llm/usage_tracking.py`](../kestrel_sovereign/llm/usage_tracking.py)

**HTTP endpoints:**
- `GET /api/models` — list available models
- `GET /api/model/current` — get current model
- `POST /api/model/set` — set active model
- `GET /v1/models` — OpenAI-compatible model list
- `POST /v1/chat/completions` — OpenAI-compatible completion endpoint

---

## Privacy, Storage, and Memory

### Privacy Modes

Privacy enforcement: [`kestrel_sovereign/privacy.py`](../kestrel_sovereign/privacy.py), [`kestrel_sovereign/features/privacy/feature.py`](../kestrel_sovereign/features/privacy/feature.py)

**Canonical presets:**

| Preset | Storage | LLM location | Shareable | Use case |
|--------|---------|--------------|-----------|----------|
| `ephemeral` | none | local | no | Nothing persisted, local LLM only |
| `isolated` | temp | local | no | Temporary session storage, local LLM only |
| `anonymous` | scrubbed | cloud | no | PII removed, cloud LLM allowed |
| `normal` | full | cloud | no | Standard persistent storage |
| `public` | full | cloud | yes | Shareable and exportable |

**HTTP endpoints:**
- `GET /agent/privacy-mode` — get current mode
- `POST /agent/privacy-mode` — set privacy mode

### Storage and Persistence

- **Storage root:** [`kestrel_sovereign/storage/__init__.py`](../kestrel_sovereign/storage/__init__.py)
- **Async storage:** [`kestrel_sovereign/storage/async_storage.py`](../kestrel_sovereign/storage/async_storage.py)
- **Storage family:** [`kestrel_sovereign/storage/`](../kestrel_sovereign/storage)

**HTTP endpoints:**
- `GET /api/storage/stats` — storage statistics
- `GET /api/sovereignty/exports` — list exports
- `POST /api/sovereignty/export` — create export
- `POST /api/sovereignty/import` — import data

### Memory Systems

- **Memory manager:** [`kestrel_sovereign/agent/memory_manager.py`](../kestrel_sovereign/agent/memory_manager.py)
- **Memory feature:** [`kestrel_sovereign/features/memory/`](../kestrel_sovereign/features/memory)
- **Memory agency feature:** [`kestrel_sovereign/features/memory_agency/`](../kestrel_sovereign/features/memory_agency)

**HTTP endpoints:**
- `GET /api/memories` — list memories
- `GET /api/memories/{node_id}` — get specific memory
- `GET /api/identity-chain` — get identity chain
- `DELETE /api/memories/{node_id}` — delete memory

---

## Feature Module Inventory

**Discovery mechanism:** Defined in [`kestrel_sovereign/features/__init__.py`](../kestrel_sovereign/features/__init__.py). Scans for single-file features, package `__init__.py`, and package `feature.py`, keeping only modules that export a `Feature` subclass.

**Current count:** 36 discoverable feature modules, 36 exported `Feature` subclasses.

### Discovered Feature Modules

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
- `mcp`
- `memory`
- `memory_agency`
- `model`
- `peers`
- `reflection`
- `runpod`
- `save`
- `scheduler`
- `security`
- `sovereignty`
- `state_of_mind`
- `tasks`
- `vastai`
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

---

## HTTP API Surface

### App-Level Routes

Defined in [`server.py`](../server.py):

- `GET /` — Root endpoint (UI or redirect)
- `GET /api/auth/key` — API key retrieval (bootstrap mode)
- `GET /health` — Health check
- `GET /health/detailed` — Detailed health check
- `POST /webhooks/stripe/crypto` — Stripe webhook

### Router Families

#### Authentication OAuth

[`endpoints/auth_oauth.py`](../endpoints/auth_oauth.py)

- `GET /auth/login`
- `GET /auth/callback`
- `GET /auth/logout`
- `GET /auth/me`

#### Agent

[`endpoints/agent.py`](../endpoints/agent.py)

- `POST /agent/invoke` — Synchronous agent invocation
- `POST /agent/stream` — Streaming agent invocation
- `POST /agent/stop` — Stop agent execution
- `GET /agent/info` — Agent information
- `GET /agent/privacy-mode` — Get privacy mode
- `POST /agent/privacy-mode` — Set privacy mode
- `GET /agent/notifications` — Get notifications
- `GET /agent/notifications/sse` — SSE notification stream
- `GET /agent/context-status` — Context status
- `GET /agent/reflection/status` — Reflection status
- `GET /agent/tasks` — Get tasks
- `GET /agent/heartbeat/status` — Heartbeat status
- `POST /agent/heartbeat/trigger` — Trigger heartbeat

#### Conversations

[`endpoints/conversations.py`](../endpoints/conversations.py)

- `GET /api/sessions`
- `GET /api/conversations`
- `GET /api/conversations/{session_id}`
- `POST /api/conversations/new`
- `DELETE /api/conversations/messages/{message_id}`
- `GET /api/conversations/{session_id}/transcript`

#### Memories

[`endpoints/memories.py`](../endpoints/memories.py)

- `GET /api/memories`
- `GET /api/memories/{node_id}`
- `GET /api/identity-chain`
- `DELETE /api/memories/{node_id}`

#### Sovereignty

[`endpoints/sovereignty.py`](../endpoints/sovereignty.py)

- `GET /api/storage/stats`
- `GET /api/sovereignty/exports`
- `POST /api/sovereignty/export`
- `POST /api/sovereignty/import`
- `GET /api/sovereignty/files`
- `GET /api/sovereignty/files/{filename}`
- `GET /api/sovereignty/files/{filename}/preview`

#### Database

[`endpoints/database.py`](../endpoints/database.py)

- `GET /api/db/tables`
- `GET /api/db/tables/{table_name}`

#### Models

[`endpoints/models.py`](../endpoints/models.py)

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

#### Commands

[`endpoints/commands.py`](../endpoints/commands.py)

- `GET /api/commands`

#### Files

[`endpoints/files.py`](../endpoints/files.py)

- `GET /api/files/{content_hash}`
- `HEAD /api/files/{content_hash}`

#### Security

[`endpoints/security.py`](../endpoints/security.py)

- `GET /api/security/permissions/tree`
- `POST /api/security/permissions`
- `POST /api/security/permissions/feature`
- `GET /api/security/pending`
- `POST /api/security/approve`
- `GET /api/security/audit`
- `POST /api/security/cancel/{request_id}`
- `POST /api/security/cancel-all`
- `POST /api/security/reset-session`

#### Observability

[`endpoints/observability.py`](../endpoints/observability.py)

- `GET /api/observability/events`
- `GET /api/observability/summary`

#### Saved Items

[`endpoints/saved_items.py`](../endpoints/saved_items.py)

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

## Authentication Model

Routes are protected by different authentication classes:

| Class | Routes | Description |
|-------|--------|-------------|
| `Public` | `/health`, `/health/detailed`, `/favicon.ico`, `/webhooks/stripe/crypto` | No authentication required |
| `Public-Localhost` | `/api/auth/key` (bootstrap mode) | Public only when bootstrap enabled |
| `OAuth public entrypoints` | `/auth/login`, `/auth/callback`, `/auth/logout` | OAuth flow entry points |
| `APIKeyOrSession` | Most `/agent/*` and `/api/*` routes | Protected by API key or session via middleware |
| `APIKeyOrSession+SSEQuery` | `/agent/stream`, `/agent/notifications/sse` | Also accepts `?api_key=` query parameter |
| `OAuthSessionSemantic` | `/auth/me` | Accepts API key or session but only returns data for real sessions |
| `Browser-Conditional` | `/` | Serves UI locally, redirects to OAuth when required |

**Auth middleware:** Defined in [`server.py`](../server.py)

---

## Testing and Verification

**Canonical surface proof layers:**

- [`tests/unit/test_auth_decision_table.py`](../tests/unit/test_auth_decision_table.py) — Auth decision verification
- [`tests/unit/test_endpoint_contract_suite.py`](../tests/unit/test_endpoint_contract_suite.py) — Endpoint contract tests
- [`tests/unit/test_feature_doc_canonicality.py`](../tests/unit/test_feature_doc_canonicality.py) — Feature doc canonicality
- [`tests/unit/test_generate_feature_docs.py`](../tests/unit/test_generate_feature_docs.py) — Doc generation tests

**Audit working papers:** [`docs/audit/`](../docs/audit)

---

## Known Boundaries

- Some support packages under `kestrel_sovereign/features/` do not export a `Feature` subclass and are not discoverable features.
- Generated audience docs require an LLM provider key for full generation; dry-run validation passes without keys.

---

## Related Documents

- **Canonical source:** [`KESTREL_FEATURES.md`](../KESTREL_FEATURES.md)
- **Generated docs index:** [`docs/generated/README.md`](../docs/generated/README.md)
- **Historical reference:** [`docs/archive/KESTREL_FEATURES_legacy.md`](../docs/archive/KESTREL_FEATURES_legacy.md)