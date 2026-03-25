# Kestrel Sovereign Feature Inventory

> Canonical source of truth for the maintained Kestrel surface.
>
> If code and this document disagree, fix the disagreement or mark it explicitly.
> Do not keep stale marketing counts here.

## How To Read This Document

- This file is the canonical inventory consumed by audience-specific generators such as [`scripts/generate_feature_docs.py`](scripts/generate_feature_docs.py).
- Generated audience docs are derived artifacts and belong under `docs/generated/`.
- Historical catalogs belong under `docs/archive/`.
- Discovery rules matter more than headline numbers:
  - Feature module discovery is defined by [`kestrel_sovereign/features/__init__.py`](kestrel_sovereign/features/__init__.py).
  - HTTP route families are defined by [`server.py`](server.py) and the routers under [`endpoints/`](endpoints).

## Canonical Principles

- Prefer maintained surfaces over aspirational claims.
- Prefer route families and feature inventories over brittle fixed counts.
- Separate supported public surfaces from internal or partial surfaces.
- Keep the canonical source austere enough that generated docs can safely transform it.

## Maintained Surface

### Constitutional and sovereign foundation

- Constitution and governance:
  - [`kestrel_sovereign/data/KESTREL_CONSTITUTION.md`](kestrel_sovereign/data/KESTREL_CONSTITUTION.md)
  - [`kestrel_sovereign/agent/constitution.py`](kestrel_sovereign/agent/constitution.py)
  - [`kestrel_sovereign/features/constitution.py`](kestrel_sovereign/features/constitution.py)
- DID identity and continuity:
  - [`kestrel_sovereign/inception_service.py`](kestrel_sovereign/inception_service.py)
  - [`kestrel_sovereign/identity/identity_package.py`](kestrel_sovereign/identity/identity_package.py)
  - [`kestrel_sovereign/identity/signing.py`](kestrel_sovereign/identity/signing.py)
  - [`kestrel_sovereign/identity/continuity_verifier.py`](kestrel_sovereign/identity/continuity_verifier.py)
- Sovereignty lifecycle:
  - [`kestrel_sovereign/graduate_service.py`](kestrel_sovereign/graduate_service.py)
  - [`kestrel_sovereign/retirement_service.py`](kestrel_sovereign/retirement_service.py)
  - [`endpoints/sovereignty.py`](endpoints/sovereignty.py)

### Agent runtime and context assembly

- Core agent orchestration:
  - [`kestrel_sovereign/kestrel_agent.py`](kestrel_sovereign/kestrel_agent.py)
  - [`kestrel_sovereign/command_handler.py`](kestrel_sovereign/command_handler.py)
  - [`kestrel_sovereign/kestrel_agent_tools.py`](kestrel_sovereign/kestrel_agent_tools.py)
- Context and token budgeting:
  - [`kestrel_sovereign/agent/context_manager.py`](kestrel_sovereign/agent/context_manager.py)
  - [`kestrel_sovereign/agent/context_builder.py`](kestrel_sovereign/agent/context_builder.py)
  - [`kestrel_sovereign/agent/token_budget.py`](kestrel_sovereign/agent/token_budget.py)
- Streaming and request lifecycle:
  - [`kestrel_sovereign/agent/streaming.py`](kestrel_sovereign/agent/streaming.py)
  - [`endpoints/agent.py`](endpoints/agent.py)

### Multi-LLM platform

- Unified service and routing:
  - [`kestrel_sovereign/llm/service.py`](kestrel_sovereign/llm/service.py)
  - [`kestrel_sovereign/llm/provider_registry.py`](kestrel_sovereign/llm/provider_registry.py)
  - [`kestrel_sovereign/llm/mandate.py`](kestrel_sovereign/llm/mandate.py)
- Provider adapters present in tree:
  - OpenAI, Anthropic, Claude Max, Gemini, Vertex AI, Ollama, OpenRouter, Mock
  - See [`kestrel_sovereign/llm/`](kestrel_sovereign/llm/)
- Catalog, metadata, retry, and usage tracking:
  - [`kestrel_sovereign/llm/model_catalog.py`](kestrel_sovereign/llm/model_catalog.py)
  - [`kestrel_sovereign/llm/model_metadata.py`](kestrel_sovereign/llm/model_metadata.py)
  - [`kestrel_sovereign/llm/retry.py`](kestrel_sovereign/llm/retry.py)
  - [`kestrel_sovereign/llm/usage_tracking.py`](kestrel_sovereign/llm/usage_tracking.py)

### Privacy, storage, and memory

- Privacy modes and enforcement:
  - [`kestrel_sovereign/privacy.py`](kestrel_sovereign/privacy.py)
  - [`kestrel_sovereign/features/privacy/feature.py`](kestrel_sovereign/features/privacy/feature.py)
  - [`kestrel_sovereign/features/privacy/`](kestrel_sovereign/features/privacy)
- Canonical privacy presets:

| Preset | Storage | LLM location | Shareable | Meaning |
|---|---|---|---|---|
| `ephemeral` | none | local | no | Nothing stored, local LLM only |
| `isolated` | temp | local | no | Temporary session storage, local LLM only |
| `anonymous` | scrubbed | cloud | no | Stored with PII removed, cloud LLM allowed |
| `normal` | full | cloud | no | Standard persistent storage |
| `public` | full | cloud | yes | Shareable and exportable |

- Storage and persistence:
  - [`kestrel_sovereign/storage/__init__.py`](kestrel_sovereign/storage/__init__.py)
  - [`kestrel_sovereign/storage/async_storage.py`](kestrel_sovereign/storage/async_storage.py)
  - [`kestrel_sovereign/storage/`](kestrel_sovereign/storage)
- Memory systems:
  - [`kestrel_sovereign/agent/memory_manager.py`](kestrel_sovereign/agent/memory_manager.py)
  - [`kestrel_sovereign/features/memory/`](kestrel_sovereign/features/memory)
  - [`kestrel_sovereign/features/memory_agency/`](kestrel_sovereign/features/memory_agency)

## Feature Module Inventory

Feature discovery scans `kestrel_sovereign/features/` for single-file features, package `__init__.py`, and package `feature.py`, then keeps only modules that actually export a discoverable `Feature` subclass. The current discovered module inventory is:

- Current audited snapshot: `37` discoverable modules and `37` exported `Feature` subclasses.

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
- `strategic_memory`
- `tasks`
- `vastai`
- `visual_identity`
- `wallet`
- `web_search`
- `webhooks`
- `wellness`

The currently exported `Feature` subclasses discovered from those modules include:

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
- `StrategicMemoryFeature`
- `TaskFeature`
- `VastAIFeature`
- `VisualIdentityFeature`
- `WalletFeature`
- `WebSearchFeature`
- `WebhookFeature`
- `WellnessFeature`

## Public HTTP Surface

### App-level routes in `server.py`

- `GET /`
- `GET /api/auth/key`
- `GET /api/github/{path:path}`
- `GET /health`
- `GET /health/detailed`
- `POST /webhooks/stripe/crypto`

### Router families mounted by `server.py`

- [`endpoints/auth_oauth.py`](endpoints/auth_oauth.py)
  - `GET /auth/login`
  - `GET /auth/callback`
  - `GET /auth/logout`
  - `GET /auth/me`
- [`endpoints/agent.py`](endpoints/agent.py)
  - `POST /agent/invoke`
  - `POST /agent/stream`
  - `POST /agent/stop`
  - `GET /agent/info`
  - `GET /agent/privacy-mode`
  - `POST /agent/privacy-mode`
  - `GET /agent/notifications`
  - `GET /agent/notifications/sse`
  - `GET /agent/context-status`
  - `GET /agent/reflection/status`
  - `GET /agent/tasks`
  - `GET /agent/heartbeat/status`
  - `POST /agent/heartbeat/trigger`
- [`endpoints/conversations.py`](endpoints/conversations.py)
  - `GET /api/sessions`
  - `GET /api/conversations`
  - `GET /api/conversations/{session_id}`
  - `POST /api/conversations/new`
  - `DELETE /api/conversations/messages/{message_id}`
  - `GET /api/conversations/{session_id}/transcript`
- [`endpoints/memories.py`](endpoints/memories.py)
  - `GET /api/memories`
  - `GET /api/memories/{node_id}`
  - `GET /api/identity-chain`
  - `DELETE /api/memories/{node_id}`
- [`endpoints/sovereignty.py`](endpoints/sovereignty.py)
  - `GET /api/storage/stats`
  - `GET /api/sovereignty/exports`
  - `POST /api/sovereignty/export`
  - `POST /api/sovereignty/import`
  - `GET /api/sovereignty/files`
  - `GET /api/sovereignty/files/{filename}`
  - `GET /api/sovereignty/files/{filename}/preview`
- [`endpoints/database.py`](endpoints/database.py)
  - `GET /api/db/tables`
  - `GET /api/db/tables/{table_name}`
- [`endpoints/models.py`](endpoints/models.py)
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
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- [`endpoints/commands.py`](endpoints/commands.py)
  - `GET /api/commands`
- [`endpoints/files.py`](endpoints/files.py)
  - `GET /api/files/{content_hash}`
  - `HEAD /api/files/{content_hash}`
- [`endpoints/security.py`](endpoints/security.py)
  - `GET /api/security/permissions/tree`
  - `POST /api/security/permissions`
  - `POST /api/security/permissions/feature`
  - `GET /api/security/pending`
  - `POST /api/security/approve`
  - `GET /api/security/audit`
  - `POST /api/security/cancel/{request_id}`
  - `POST /api/security/cancel-all`
  - `POST /api/security/reset-session`
- [`endpoints/observability.py`](endpoints/observability.py)
  - `GET /api/observability/events`
  - `GET /api/observability/summary`
- [`endpoints/rasa_shim.py`](endpoints/rasa_shim.py)
  - `POST /webhooks/rest/webhook`
- [`endpoints/saved_items.py`](endpoints/saved_items.py)
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

## Authentication Surface

The route surface is not just public versus protected. The current live classes are:

- `Public`
  - `/health`
  - `/health/detailed`
  - `/favicon.ico`
  - `/webhooks/stripe/crypto`
- `Public-Localhost`
  - `/api/auth/key` when bootstrap is enabled
- `OAuth public entrypoints`
  - `/auth/login`
  - `/auth/callback`
  - `/auth/logout`
- `APIKeyOrSession`
  - most protected `/agent/*` and `/api/*` routes via `server.py` auth middleware
- `APIKeyOrSession+SSEQuery`
  - SSE paths that also allow `?api_key=`:
    - `/agent/stream`
    - `/agent/notifications/sse`
- `OAuthSessionSemantic`
  - `/auth/me` can pass middleware via API key or session, but only returns authenticated data from a real session
- `Browser-Conditional`
  - `/` serves UI for local/browser conditions and redirects to OAuth when OAuth-required mode is enabled

## Generated and Historical Documents

- Canonical source:
  - [`KESTREL_FEATURES.md`](KESTREL_FEATURES.md)
- Generated audience docs:
  - [`docs/generated/README.md`](docs/generated/README.md)
- Historical snapshot:
  - [`docs/archive/KESTREL_FEATURES_legacy.md`](docs/archive/KESTREL_FEATURES_legacy.md)

## Audit and Verification

- Audit working papers live under [`docs/audit/`](docs/audit).
- Fast proof layers for the canonical surface live in:
  - [`tests/unit/test_auth_decision_table.py`](tests/unit/test_auth_decision_table.py)
  - [`tests/unit/test_endpoint_contract_suite.py`](tests/unit/test_endpoint_contract_suite.py)
  - [`tests/unit/test_feature_doc_canonicality.py`](tests/unit/test_feature_doc_canonicality.py)
  - [`tests/unit/test_generate_feature_docs.py`](tests/unit/test_generate_feature_docs.py)

## Known Boundaries

- Some support packages live under `kestrel_sovereign/features/` but are not discoverable features because they do not export a `Feature` subclass.
- Generated audience docs require an LLM provider key; dry-run validation should pass even when generation keys are absent.
