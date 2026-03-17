<!-- AUTO-GENERATED from KESTREL_FEATURES.md — do not edit manually -->
<!-- Audience: developer | Generated: 2026-03-17 | Model: anthropic/claude-sonnet-4-5-20250929 -->
<!-- Regenerate: uv run python scripts/generate_feature_docs.py --audience developer -->

# Kestrel Sovereign Developer Reference

**Target audience:** Software engineers and AI agents integrating with or extending Kestrel Sovereign.

This is a technical reference derived from the canonical feature inventory. For source-of-truth maintenance, see [`KESTREL_FEATURES.md`](KESTREL_FEATURES.md).

---

## Quick Start

**Launch the agent:**
```bash
# Start the FastAPI server
python server.py
```

**Invoke agent via HTTP:**
```bash
POST /agent/invoke
POST /agent/stream  # SSE streaming
```

**Check status:**
```bash
GET /health
GET /agent/info
```

---

## Architecture Overview

Kestrel Sovereign is a constitutional AI agent framework with:

- **Constitutional governance** — all actions evaluated against [`KESTREL_CONSTITUTION.md`](kestrel_sovereign/data/KESTREL_CONSTITUTION.md)
- **DID-based identity** — cryptographic continuity via [`inception_service.py`](kestrel_sovereign/inception_service.py)
- **Multi-LLM routing** — unified interface over OpenAI, Anthropic, Gemini, Vertex AI, Ollama, OpenRouter, and more
- **Feature-driven extensibility** — discover and register features from [`kestrel_sovereign/features/`](kestrel_sovereign/features/)
- **Privacy-first storage** — five privacy modes from ephemeral to public, enforced at runtime

---

## Core Systems

### 1. Constitutional Foundation

The agent operates under a formal constitution that defines permissible behaviors, consent requirements, and sovereignty principles.

| Component | Purpose | Entry Point |
|-----------|---------|-------------|
| **Constitution text** | Human-readable governance rules | [`kestrel_sovereign/data/KESTREL_CONSTITUTION.md`](kestrel_sovereign/data/KESTREL_CONSTITUTION.md) |
| **Constitution loader** | Parses and validates constitution | [`kestrel_sovereign/agent/constitution.py`](kestrel_sovereign/agent/constitution.py) |
| **Constitution feature** | Exposes constitution via feature API | [`kestrel_sovereign/features/constitution.py`](kestrel_sovereign/features/constitution.py) |

**Key API:**
```python
from kestrel_sovereign.agent.constitution import load_constitution

constitution = await load_constitution()
```

### 2. Identity and Continuity

Decentralized identity (DID) with cryptographic signing and continuity verification.

| Component | File |
|-----------|------|
| **Inception service** | [`kestrel_sovereign/inception_service.py`](kestrel_sovereign/inception_service.py) |
| **Identity package** | [`kestrel_sovereign/identity/identity_package.py`](kestrel_sovereign/identity/identity_package.py) |
| **Signing utilities** | [`kestrel_sovereign/identity/signing.py`](kestrel_sovereign/identity/signing.py) |
| **Continuity verifier** | [`kestrel_sovereign/identity/continuity_verifier.py`](kestrel_sovereign/identity/continuity_verifier.py) |

**HTTP endpoints:**
```
GET /api/identity
GET /api/identity-chain
```

### 3. Sovereignty Lifecycle

Manage agent lifecycle from inception through graduation to retirement.

| Service | File | Purpose |
|---------|------|---------|
| **Inception** | [`kestrel_sovereign/inception_service.py`](kestrel_sovereign/inception_service.py) | Bootstrap new agent identity |
| **Graduation** | [`kestrel_sovereign/graduate_service.py`](kestrel_sovereign/graduate_service.py) | Promote agent to full autonomy |
| **Retirement** | [`kestrel_sovereign/retirement_service.py`](kestrel_sovereign/retirement_service.py) | Graceful shutdown and archival |

**HTTP router:**
- [`endpoints/sovereignty.py`](endpoints/sovereignty.py)

**Key endpoints:**
```
GET  /api/sovereignty/exports
POST /api/sovereignty/export
POST /api/sovereignty/import
GET  /api/sovereignty/files
```

---

## Agent Runtime

### Core Orchestration

| Component | File | Responsibility |
|-----------|------|----------------|
| **Agent runtime** | [`kestrel_sovereign/kestrel_agent.py`](kestrel_sovereign/kestrel_agent.py) | Main agent loop, tool dispatch |
| **Command handler** | [`kestrel_sovereign/command_handler.py`](kestrel_sovereign/command_handler.py) | Parse and route commands |
| **Agent tools** | [`kestrel_sovereign/kestrel_agent_tools.py`](kestrel_sovereign/kestrel_agent_tools.py) | Built-in tool definitions |

### Context Assembly

Context management uses a token budget system to fit prompts within model limits.

| Component | File |
|-----------|------|
| **Context manager** | [`kestrel_sovereign/agent/context_manager.py`](kestrel_sovereign/agent/context_manager.py) |
| **Context builder** | [`kestrel_sovereign/agent/context_builder.py`](kestrel_sovereign/agent/context_builder.py) |
| **Token budget** | [`kestrel_sovereign/agent/token_budget.py`](kestrel_sovereign/agent/token_budget.py) |

**Quick example:**
```python
from kestrel_sovereign.agent.context_manager import ContextManager

context_mgr = ContextManager(token_limit=8192)
context = await context_mgr.assemble_context(session_id, feature_states)
```

### Streaming and Request Lifecycle

- **Streaming handler:** [`kestrel_sovereign/agent/streaming.py`](kestrel_sovereign/agent/streaming.py)
- **HTTP router:** [`endpoints/agent.py`](endpoints/agent.py)

**SSE streaming endpoint:**
```
POST /agent/stream
```

---

## Multi-LLM Platform

Unified interface over multiple LLM providers with automatic routing, retry, and usage tracking.

### Service Layer

| Component | File | Purpose |
|-----------|------|---------|
| **LLM service** | [`kestrel_sovereign/llm/service.py`](kestrel_sovereign/llm/service.py) | Unified async interface |
| **Provider registry** | [`kestrel_sovereign/llm/provider_registry.py`](kestrel_sovereign/llm/provider_registry.py) | Register and route providers |
| **Mandate** | [`kestrel_sovereign/llm/mandate.py`](kestrel_sovereign/llm/mandate.py) | Provider selection logic |

### Provider Adapters

Available in [`kestrel_sovereign/llm/`](kestrel_sovereign/llm/):

- OpenAI
- Anthropic
- Claude Max
- Gemini
- Vertex AI
- Ollama (local)
- OpenRouter
- Mock (testing)

### Catalog and Metadata

| Component | File |
|-----------|------|
| **Model catalog** | [`kestrel_sovereign/llm/model_catalog.py`](kestrel_sovereign/llm/model_catalog.py) |
| **Model metadata** | [`kestrel_sovereign/llm/model_metadata.py`](kestrel_sovereign/llm/model_metadata.py) |
| **Retry logic** | [`kestrel_sovereign/llm/retry.py`](kestrel_sovereign/llm/retry.py) |
| **Usage tracking** | [`kestrel_sovereign/llm/usage_tracking.py`](kestrel_sovereign/llm/usage_tracking.py) |

**HTTP endpoints:**
```
GET  /api/models
GET  /api/model/current
POST /api/model/set
GET  /v1/models              # OpenAI-compatible
POST /v1/chat/completions    # OpenAI-compatible
```

---

## Privacy and Storage

### Privacy Modes

Five privacy presets enforced at runtime via [`kestrel_sovereign/privacy.py`](kestrel_sovereign/privacy.py):

| Preset | Storage | LLM Location | Shareable | Use Case |
|--------|---------|--------------|-----------|----------|
| `ephemeral` | none | local | no | Zero persistence, local LLM only |
| `isolated` | temp | local | no | Session storage, local LLM only |
| `anonymous` | scrubbed | cloud | no | PII removed, cloud LLM allowed |
| `normal` | full | cloud | no | Standard persistent storage |
| `public` | full | cloud | yes | Shareable and exportable |

**Feature implementation:**
- [`kestrel_sovereign/features/privacy/feature.py`](kestrel_sovereign/features/privacy/feature.py)
- [`kestrel_sovereign/features/privacy/`](kestrel_sovereign/features/privacy)

**HTTP endpoints:**
```
GET  /agent/privacy-mode
POST /agent/privacy-mode
```

### Storage Layer

Async storage with multi-backend support.

| Component | File |
|-----------|------|
| **Storage interface** | [`kestrel_sovereign/storage/__init__.py`](kestrel_sovereign/storage/__init__.py) |
| **Async storage** | [`kestrel_sovereign/storage/async_storage.py`](kestrel_sovereign/storage/async_storage.py) |
| **Backend modules** | [`kestrel_sovereign/storage/`](kestrel_sovereign/storage) |

**HTTP endpoints:**
```
GET /api/storage/stats
```

### Memory Systems

- **Memory manager:** [`kestrel_sovereign/agent/memory_manager.py`](kestrel_sovereign/agent/memory_manager.py)
- **Memory feature:** [`kestrel_sovereign/features/memory/`](kestrel_sovereign/features/memory)
- **Memory agency feature:** [`kestrel_sovereign/features/memory_agency/`](kestrel_sovereign/features/memory_agency)

**HTTP endpoints:**
```
GET    /api/memories
GET    /api/memories/{node_id}
DELETE /api/memories/{node_id}
```

---

## Feature Module System

Features are auto-discovered from [`kestrel_sovereign/features/`](kestrel_sovereign/features/).

**Discovery rules:**
- Single-file features: `feature_name.py`
- Package features: `feature_name/__init__.py` or `feature_name/feature.py`
- Must export a `Feature` subclass to be registered

**Discovery logic:** [`kestrel_sovereign/features/__init__.py`](kestrel_sovereign/features/__init__.py)

### Current Feature Inventory

**41 discoverable modules, 36 exported `Feature` subclasses:**

| Module | Exported Class (if any) |
|--------|-------------------------|
| `audit_anchor` | `AuditAnchorFeature` |
| `bootstrap` | `BootstrapFeature` |
| `bridge` | `BridgeFeature` |
| `channels` | `ChannelFeature` |
| `code_edit` | `CodeEditFeature` |
| `compute` | `ComputeFeature` |
| `consent` | `ConsentFeature` |
| `constitution` | `ConstitutionFeature` |
| `context` | `ContextFeature` |
| `council` | `CouncilFeature` |
| `delivery` | `DeliveryFeature` |
| `deploy` | `DeployFeature` |
| `gcp_compute` | `GCPComputeFeature` |
| `github` | `GitHubFeature` |
| `heartbeat` | `HeartbeatFeature` |
| `identity` | `IdentityFeature` |
| `keys` | `KeyManagementFeature` |
| `llm_keys` | *(no exported class)* |
| `mcp` | `MCPAgent` |
| `memory` | `MemoryFeature` |
| `memory_agency` | `MemoryAgencyFeature` |
| `model` | `ModelAgent` |
| `ollama` | *(no exported class)* |
| `peers` | `PeersFeature` |
| `privacy` | *(no exported class)* |
| `reflection` | `ReflectionFeature` |
| `runpod` | `RunPodFeature` |
| `save` | `SaveFeature` |
| `scheduler` | `SchedulerFeature` |
| `security` | `SecurityFeature` |
| `sovereignty` | `SovereigntyFeature` |
| `state_of_mind` | `StateOfMindFeature` |
| `tasks` | `TaskFeature` |
| `training` | *(no exported class)* |
| `vastai` | `VastAIFeature` |
| `vertex_ai` | *(no exported class)* |
| `visual_identity` | `VisualIdentityFeature` |
| `wallet` | `WalletFeature` |
| `web_search` | `WebSearchFeature` |
| `webhooks` | `WebhookFeature` |
| `wellness` | `WellnessFeature` |

---

## HTTP API Reference

### Server Entry Point

**File:** [`server.py`](server.py)

**App-level routes:**
```
GET  /
GET  /api/auth/key
GET  /health
GET  /health/detailed
POST /webhooks/stripe/crypto
```

### Router Families

#### Authentication (`endpoints/auth_oauth.py`)

OAuth flow for browser-based authentication.

```
GET /auth/login
GET /auth/callback
GET /auth/logout
GET /auth/me
```

#### Agent Operations (`endpoints/agent.py`)

Core agent invocation and control.

```
POST /agent/invoke
POST /agent/stream                    # SSE
POST /agent/stop
GET  /agent/info
GET  /agent/privacy-mode
POST /agent/privacy-mode
GET  /agent/notifications
GET  /agent/notifications/sse         # SSE
GET  /agent/context-status
GET  /agent/reflection/status
GET  /agent/tasks
GET  /agent/heartbeat/status
POST /agent/heartbeat/trigger
```

#### Conversations (`endpoints/conversations.py`)

Session and conversation history management.

```
GET    /api/sessions
GET    /api/conversations
GET    /api/conversations/{session_id}
POST   /api/conversations/new
DELETE /api/conversations/messages/{message_id}
GET    /api/conversations/{session_id}/transcript
```

#### Memories (`endpoints/memories.py`)

Long-term memory node access.

```
GET    /api/memories
GET    /api/memories/{node_id}
GET    /api/identity-chain
DELETE /api/memories/{node_id}
```

#### Sovereignty (`endpoints/sovereignty.py`)

Export, import, and file access for data sovereignty.

```
GET  /api/storage/stats
GET  /api/sovereignty/exports
POST /api/sovereignty/export
POST /api/sovereignty/import
GET  /api/sovereignty/files
GET  /api/sovereignty/files/{filename}
GET  /api/sovereignty/files/{filename}/preview
```

#### Database (`endpoints/database.py`)

Introspect internal database schema.

```
GET /api/db/tables
GET /api/db/tables/{table_name}
```

#### Models and Configuration (`endpoints/models.py`)

Agent configuration, model selection, and OpenAI-compatible endpoints.

```
GET    /api/agents
POST   /api/agents
DELETE /api/agents/{agent_name}
GET    /api/identity
GET    /api/constitution
GET    /api/ipfs/status
GET    /api/wallet
GET    /api/keys
POST   /api/keys
PATCH  /api/keys/{provider}
DELETE /api/keys/{provider}
GET    /api/keys/{provider}/usage
GET    /api/models
GET    /api/model/current
POST   /api/model/set
GET    /v1/models                     # OpenAI-compatible
POST   /v1/chat/completions           # OpenAI-compatible
```

#### Commands (`endpoints/commands.py`)

List available agent commands.

```
GET /api/commands
```

#### Files (`endpoints/files.py`)

Content-addressed file retrieval.

```
GET  /api/files/{content_hash}
HEAD /api/files/{content_hash}
```

#### Security (`endpoints/security.py`)

Permission management and audit log.

```
GET  /api/security/permissions/tree
POST /api/security/permissions
POST /api/security/permissions/feature
GET  /api/security/pending
POST /api/security/approve
GET  /api/security/audit
POST /api/security/cancel/{request_id}
POST /api/security/cancel-all
POST /api/security/reset-session
```

#### Observability (`endpoints/observability.py`)

Event streaming and telemetry.

```
GET /api/observability/events
GET /api/observability/summary
```

#### Saved Items (`endpoints/saved_items.py`)

Structured data persistence with schemas and tags.

```
GET    /api/saved-items
POST   /api/saved-items
GET    /api/saved-items/stats
GET    /api/saved-items/schemas
GET    /api/saved-items/tags
GET    /api/saved-items/by-tag/{tag}
GET    /api/saved-items/by-schema/{schema_id}
GET    /api/saved-items/{item_id}
PATCH  /api/saved-items/{item_id}
DELETE /api/saved-items/{item_id}
POST   /api/saved-items/structured
POST   /api/saved-items/search
POST   /api/saved-items/{item_id}/pin
```

---

## Authentication and Authorization

Kestrel uses a fine-grained auth model beyond simple "public vs protected."

### Auth Classes

| Class | Routes | Behavior |
|-------|--------|----------|
| **Public** | `/health`, `/health/detailed`, `/favicon.ico`, `/webhooks/stripe/crypto` | No auth required |
| **Public-Localhost** | `/api/auth/key` (bootstrap mode) | Localhost only when bootstrap enabled |
| **OAuth public** | `/auth/login`, `/auth/callback`, `/auth/logout` | Public OAuth flow entrypoints |
| **APIKeyOrSession** | Most `/agent/*` and `/api/*` routes | Requires API key or session token |
| **APIKeyOrSession+SSEQuery** | `/agent/stream`, `/agent/notifications/sse` | Also accepts `?api_key=` query param for SSE clients |
| **OAuthSessionSemantic** | `/auth/me` | Passes middleware via API key or session, but only returns authenticated data from session |
| **Browser-Conditional** | `/` | Serves UI for local/browser, redirects to OAuth when OAuth-required mode enabled |

**Middleware implementation:** See `server.py` auth middleware and route decorators.

---

## Testing and Verification

### Test Suites

| Test Suite | File | Verifies |
|------------|------|----------|
| **Auth decision table** | [`tests/unit/test_auth_decision_table.py`](tests/unit/test_auth_decision_table.py) | Auth class coverage |
| **Endpoint contracts** | [`tests/unit/test_endpoint_contract_suite.py`](tests/unit/test_endpoint_contract_suite.py) | HTTP route signatures |
| **Feature doc canonicality** | [`tests/unit/test_feature_doc_canonicality.py`](tests/unit/test_feature_doc_canonicality.py) | Feature inventory accuracy |
| **Feature doc generation** | [`tests/unit/test_generate_feature_docs.py`](tests/unit/test_generate_feature_docs.py) | Doc generator correctness |

**Audit working papers:** [`docs/audit/`](docs/audit)

---

## Extending Kestrel

### Adding a Feature

1. Create `kestrel_sovereign/features/my_feature.py` or `kestrel_sovereign/features/my_feature/feature.py`
2. Subclass `Feature` from `kestrel_sovereign.features.base`
3. Implement `async def execute(self, context: dict) -> dict`
4. Feature auto-discovered on next agent restart

**Example:**
```python
from kestrel_sovereign.features.base import Feature

class MyFeature(Feature):
    name = "my_feature"
    description = "Does something useful"
    
    async def execute(self, context: dict) -> dict:
        return {"status": "success"}
```

### Adding an LLM Provider

1. Create `kestrel_sovereign/llm/providers/my_provider.py`
2. Implement the provider interface (see existing providers for reference)
3. Register in [`kestrel_sovereign/llm/provider_registry.py`](kestrel_sovereign/llm/provider_registry.py)

### Adding an HTTP Endpoint

1. Create or extend a router in `endpoints/`
2. Mount router in `server.py`
3. Add auth decorator: `@auth_required(AuthClass.APIKeyOrSession)`
4. Add contract test in [`tests/unit/test_endpoint_contract_suite.py`](tests/unit/test_endpoint_contract_suite.py)

---

## Known Limitations

- Some feature modules are discoverable but do not export a `Feature` subclass (see feature inventory table)
- Some route families are thin wrappers and lack comprehensive contract tests
- Generated audience docs require an LLM provider key for full generation (dry-run validation passes without keys)

---

## Related Documentation

- **Canonical source:** [`KESTREL_FEATURES.md`](KESTREL_FEATURES.md)
- **Generated docs:** [`docs/generated/README.md`](docs/generated/README.md)
- **Historical snapshot:** [`docs/archive/KESTREL_FEATURES_legacy.md`](docs/archive/KESTREL_FEATURES_legacy.md)
- **Audit papers:** [`docs/audit/`](docs/audit)

---

**Generation script:** [`scripts/generate_feature_docs.py`](scripts/generate_feature_docs.py)