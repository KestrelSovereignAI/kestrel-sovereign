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
  - HTTP route families are defined by [`kestrel_sovereign/server.py`](kestrel_sovereign/server.py) and the routers under [`kestrel_sovereign/endpoints/`](kestrel_sovereign/endpoints).

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
  - [`kestrel_sovereign/endpoints/sovereignty.py`](kestrel_sovereign/endpoints/sovereignty.py)

### Agent runtime and context assembly

- Core agent orchestration:
  - [`kestrel_sovereign/kestrel_agent.py`](kestrel_sovereign/kestrel_agent.py)
  - [`kestrel_sovereign/command_handler.py`](kestrel_sovereign/command_handler.py)
- Context and token budgeting:
  - [`kestrel_sovereign/agent/context_manager.py`](kestrel_sovereign/agent/context_manager.py)
  - [`kestrel_sovereign/agent/context_builder.py`](kestrel_sovereign/agent/context_builder.py)
  - [`kestrel_sovereign/agent/token_budget.py`](kestrel_sovereign/agent/token_budget.py)
- Streaming and request lifecycle:
  - [`kestrel_sovereign/agent/streaming.py`](kestrel_sovereign/agent/streaming.py)
  - [`kestrel_sovereign/endpoints/agent.py`](kestrel_sovereign/endpoints/agent.py)

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

Features come from two sources:

1. **Core features** — discovered from `kestrel_sovereign/features/` (single-file modules, package `__init__.py`, and package `feature.py`). Only modules that export a discoverable `Feature` subclass are kept.
2. **Package features** — installed via pip and registered as `kestrel_sovereign.features` entry points. These are discovered at runtime and extend the core inventory without modifying it. On duplicate class names, core features win.

The inventory below lists **core features only**. Installed feature packages appear at runtime via `discover_entrypoint_feature_classes()` and are not enumerated here.

- Current audited snapshot: `35` discoverable modules and `36` exported `Feature` subclasses.

- `attachments`
- `audit_anchor`
- `bootstrap`
- `bridge`
- `channels`
- `cli`
- `compute`
- `computer_use`
- `consent`
- `constitution`
- `context`
- `delivery`
- `deploy`
- `health`
- `identity`
- `keys`
- `memory`
- `memory_agency`
- `model`
- `peers`
- `response_audit`
- `restart_coordinator`
- `save`
- `scheduler`
- `security`
- `skills`
- `sovereignty`
- `spawn`
- `state_of_mind`
- `strategic_memory`
- `talon`
- `tasks`
- `web_search`
- `webhooks`
- `wellness`

The currently exported `Feature` subclasses discovered from those modules include:

- `AuditAnchorFeature`
- `BootstrapFeature`
- `BridgeFeature`
- `ChannelFeature`
- `CliFeature`
- `ComputeFeature`
- `ComputerUseFeature`
- `ConsentFeature`
- `ConstitutionFeature`
- `ContextFeature`
- `DeliveryFeature`
- `DeployFeature`
- `HealthFeature`
- `IdentityFeature`
- `KeyManagementFeature`
- `MemoryAgencyFeature`
- `MemoryFeature`
- `ModelAgent`
- `PeersFeature`
- `ProxyFeature`
- `ResponseAuditFeature`
- `RestartCoordinatorFeature`
- `SaveFeature`
- `SchedulerFeature`
- `SecurityFeature`
- `SkillsFeature`
- `SovereigntyFeature`
- `SpawnFeature`
- `StateOfMindFeature`
- `StrategicMemoryFeature`
- `TalonCoordinatorFeature`
- `TaskFeature`
- `WebSearchFeature`
- `WebhookFeature`
- `WellnessFeature`

## External App Surfaces

These are maintained Kestrel applications or services, but they are not
agent feature packages and should not be listed in
`kestrel_sovereign.features` package registry metadata.

- **Kestrel GitHub bot** — the hosted GitHub App/webhook service used by the
  project for issue and discussion automation. This is an application surface,
  not a reusable `kestrel-feature-*` capability for arbitrary agents.

## Public HTTP Surface

### App-level routes in `kestrel_sovereign/server.py`

- `GET /`
- `GET /api/auth/key`
- `GET /api/github/{path:path}`
- `GET /health`
- `GET /health/detailed`

### Router families mounted by `kestrel_sovereign/server.py`

- [`kestrel_sovereign/endpoints/auth_oauth.py`](kestrel_sovereign/endpoints/auth_oauth.py)
  - `GET /auth/login`
  - `GET /auth/callback`
  - `GET /auth/logout`
  - `GET /auth/me`
  - `POST /auth/token`
  - `GET /auth/verify`
- [`kestrel_sovereign/endpoints/agent.py`](kestrel_sovereign/endpoints/agent.py)
  - `POST /api/agent/invoke`
  - `POST /api/agent/stream`
  - `POST /api/agent/stop`
  - `GET /api/agent/info`
  - `GET /api/agent/privacy-mode`
  - `POST /api/agent/privacy-mode`
  - `GET /api/agent/notifications`
  - `GET /api/agent/notifications/sse`
  - `GET /api/agent/context-status`
  - `GET /api/agent/reflection/status`
  - `GET /api/agent/tasks`
  - `GET /api/agent/tasks/{task_id}`
  - `POST /api/agent/tasks/send`
  - `GET /api/agent/heartbeat/status`
  - `POST /api/agent/heartbeat/trigger`
  - `GET /api/agent/health/status`
  - `POST /api/agent/health/trigger`
- [`kestrel_sovereign/endpoints/conversations.py`](kestrel_sovereign/endpoints/conversations.py)
  - `GET /api/sessions`
  - `GET /api/conversations`
  - `GET /api/conversations/{session_id}`
  - `POST /api/conversations/new`
  - `DELETE /api/conversations/messages/{message_id}`
  - `DELETE /api/conversations/{session_id}`
  - `POST /api/conversations/{session_id}/restore`
  - `POST /api/conversations/{session_id}/purge`
  - `POST /api/conversations/messages/{message_id}/restore`
  - `POST /api/conversations/messages/{message_id}/purge`
  - `GET /api/trash`
  - `PATCH /api/conversations/{session_id}`
  - `GET /api/conversations/{session_id}/transcript`
- [`kestrel_sovereign/endpoints/memories.py`](kestrel_sovereign/endpoints/memories.py)
  - `GET /api/memories`
  - `GET /api/memories/{node_id}`
  - `GET /api/identity-chain`
  - `DELETE /api/memories/{node_id}`
- [`kestrel_sovereign/endpoints/sovereignty.py`](kestrel_sovereign/endpoints/sovereignty.py)
  - `GET /api/storage/stats`
  - `GET /api/sovereignty/exports`
  - `POST /api/sovereignty/export`
  - `POST /api/sovereignty/import`
  - `GET /api/sovereignty/files`
  - `GET /api/sovereignty/files/{filename}`
  - `GET /api/sovereignty/files/{filename}/preview`
- [`kestrel_sovereign/endpoints/database.py`](kestrel_sovereign/endpoints/database.py)
  - `GET /api/db/tables`
  - `GET /api/db/tables/{table_name}`
- [`kestrel_sovereign/endpoints/models.py`](kestrel_sovereign/endpoints/models.py)
  - `GET /api/agents`
  - `POST /api/agents`
  - `DELETE /api/agents/{agent_name}`
  - `GET /api/identity`
  - `PATCH /api/identity`
  - `POST /api/identity/avatar`
  - `POST /api/identity/avatar/generate`
  - `GET /api/constitution`
  - `GET /api/ipfs/status`
  - `GET /api/wallet`
  - `GET /api/keys`
  - `POST /api/keys`
  - `PATCH /api/keys/{provider}`
  - `DELETE /api/keys/{provider}`
  - `GET /api/keys/{provider}/usage`
  - `GET /api/keys/available-sources`
  - `GET /api/keys/user`
  - `POST /api/keys/user`
  - `POST /api/keys/user/verify`
  - `DELETE /api/keys/user/{provider}`
  - `GET /api/keys/platform`
  - `GET /api/models`
  - `GET /api/model/current`
  - `POST /api/model/set`
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- [`kestrel_sovereign/endpoints/commands.py`](kestrel_sovereign/endpoints/commands.py)
  - `GET /api/commands`
- [`kestrel_sovereign/endpoints/files.py`](kestrel_sovereign/endpoints/files.py)
  - `GET /api/files/{content_hash}`
  - `HEAD /api/files/{content_hash}`
- [`kestrel_sovereign/endpoints/security.py`](kestrel_sovereign/endpoints/security.py)
  - `GET /api/security/permissions/tree`
  - `POST /api/security/permissions`
  - `POST /api/security/permissions/feature`
  - `GET /api/security/pending`
  - `POST /api/security/approve`
  - `GET /api/security/audit`
  - `POST /api/security/cancel/{request_id}`
  - `POST /api/security/cancel-all`
  - `POST /api/security/reset-session`
- [`kestrel_sovereign/endpoints/metrics.py`](kestrel_sovereign/endpoints/metrics.py)
  - `GET /metrics`
- [`kestrel_sovereign/endpoints/spawn.py`](kestrel_sovereign/endpoints/spawn.py)
  - `GET /api/spawn/children`
- [`kestrel_sovereign/endpoints/observability.py`](kestrel_sovereign/endpoints/observability.py)
  - `GET /api/observability/events`
  - `GET /api/observability/summary`
- [`kestrel_sovereign/endpoints/rasa_shim.py`](kestrel_sovereign/endpoints/rasa_shim.py)
  - `POST /webhooks/rest/webhook`
- [`kestrel_sovereign/endpoints/saved_items.py`](kestrel_sovereign/endpoints/saved_items.py)
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
- `kestrel-feature-voice` optional package via the
  `kestrel_sovereign.features` entry-point group
  - `GET /voice/voices`
  - `GET /voice/providers/status` — diagnostic surface for every TTS/STT/conversation provider attempted at boot. Returns `init_error`, `available_error`, live `voice_count` for TTS, and an actionable `install_hint`. Drives the voice picker's "why is this empty?" inline reason.
  - `GET /voice/config`
  - `POST /voice/config`
  - `POST /voice/tts`
  - `POST /voice/tts/stream`
  - `POST /voice/stt`
  - `WebSocket /voice/chat`
  - `POST /realtime/session` — declared on the realtime router; **served at `POST /voice/realtime/session`**. Body accepts per-call routing overrides (`prefer_realtime`, `preferred_tts`, `preferred_stt`) so the voice picker can force Pipeline mode or pin a TTS without persisting agent config. Privacy-gated via the voice path resolver; returns 409 with fallback provider names when the active route is not realtime.
  - `POST /realtime/tools/{session_id}` — **served at `POST /voice/realtime/tools/{session_id}`**. Browser POSTs here when the Realtime model invokes a tool; runs it against the agent's enabled features and returns the result. Always 200 with a result payload (errors as `{result: {error: ...}}`) so the frontend always commits *something* back to the data channel — silence wedges the model.
  - `GET /realtime/route` — **served at `GET /voice/realtime/route`**. Pure introspection: returns the resolved voice route + the model that would actually answer (`gpt-realtime-1.5` for Realtime, your chat LLM for Pipeline) plus the available conversation/TTS/STT providers. Query params (`prefer_realtime`, `preferred_tts`, `preferred_stt`) preview alternative routes without minting. Drives the voice picker's live route-preview block.
- [`kestrel_sovereign/endpoints/features.py`](kestrel_sovereign/endpoints/features.py)
  - `GET /api/features`
  - `GET /api/features/installed`
  - `GET /api/features/{name}`
  - `POST /api/features/{name}/install`
  - `POST /api/features/{name}/enable`
  - `POST /api/features/{name}/disable`
  - `POST /api/features/{name}/remove`
  - `GET /api/features/{name}/config`
  - `PATCH /api/features/{name}/config`
  - `GET /api/features/{name}/skills`
  - `GET /api/skills`
  - `GET /api/skills/{skill_id}/schema`
- [`kestrel_sovereign/endpoints/ui.py`](kestrel_sovereign/endpoints/ui.py)
  - `GET /api/ui/theme` — UI theme + i18n labels for the active locale, with legacy-fallback reporting
  - `GET /api/ui/themes` — list of installed UI themes

## Authentication Surface

The route surface is not just public versus protected. The current live classes are:

- `Public`
  - `/health`
  - `/health/detailed`
  - `/favicon.ico`
- `Public-Localhost`
  - `/api/auth/key` when bootstrap is enabled
- `OAuth public entrypoints`
  - `/auth/login`
  - `/auth/callback`
  - `/auth/logout`
- `APIKeyOrSession`
  - most protected `/agent/*` and `/api/*` routes via `kestrel_sovereign/server.py` auth middleware
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
