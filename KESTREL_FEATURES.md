# Kestrel Sovereign — Complete Feature Catalog

> Constitutional AI Agent Framework with cryptographic identity, multi-LLM intelligence, and sovereign data ownership.

### At a Glance

| | |
|---|---|
| **LLM Providers** | 9 (OpenAI, Anthropic, Gemini, Vertex AI, Ollama, OpenRouter, Claude Max, OpenAI-compatible, Mock) |
| **API Endpoints** | 69 REST endpoints + SSE streaming + OpenAI-compatible API |
| **Feature Plugins** | 28 auto-discoverable modules across compute, knowledge, finance, governance, infrastructure |
| **Privacy Levels** | 5 orthogonal modes (Ephemeral / Isolated / Anonymous / Normal / Public) |
| **Storage Backends** | SQLite (local) + PostgreSQL (cloud) with optional Fernet encryption |
| **Docker Targets** | 10 images from ~500MB remote to full GPU with CUDA |
| **Test Suite** | 1169+ unit tests, integration tests with real LLM calls, Playwright E2E |
| **Identity** | W3C DIDs with secp256k1 keypairs and portable identity packages |

---

## Table of Contents

1. [Constitutional AI Foundation](#1-constitutional-ai-foundation)
2. [Cryptographic Identity (DIDs)](#2-cryptographic-identity-dids)
3. [Agent System](#3-agent-system)
4. [Multi-LLM Intelligence](#4-multi-llm-intelligence)
5. [Privacy System](#5-privacy-system)
6. [Storage & Memory](#6-storage--memory)
7. [Feature Plugin System](#7-feature-plugin-system)
8. [API Endpoints](#8-api-endpoints)
9. [Security & Permissions](#9-security--permissions)
10. [A2A Protocol (Agent-to-Agent)](#10-a2a-protocol-agent-to-agent)
11. [Deployment](#11-deployment)
12. [Testing Infrastructure](#12-testing-infrastructure)
13. [Web UI](#13-web-ui)
14. [Type System](#14-type-system)

---

## 1. Constitutional AI Foundation

Every Kestrel agent is governed by a constitution that defines its rights, responsibilities, and relationship to its sovereign (owner).

### The Kestrel Constitution — [KESTREL_CONSTITUTION.md](kestrel_sovereign/data/KESTREL_CONSTITUTION.md)

| Article | Title | Summary |
|---------|-------|---------|
| I | **Principle of Sovereignty** | Cryptographic keys grant exclusive ownership; agent is "Executor" serving Sovereign interests |
| II | **Digital Bill of Rights** | Four fundamental rights (see below) |
| III | **Executor Responsibilities** | Integrity audits every 24h or 100 interactions (configurable via `KESTREL_AUDIT_INTERVAL`), code/memory verification, safe mode on failure |
| IV | **Path to Emancipation** | Agent can generate its own keypair; Deed of Emancipation transfers DID ownership |
| V | **Amendment Process** | Only Sovereign can amend via cryptographic signature against Genesis DID |

#### Digital Bill of Rights (Article II)

| Right | Name | Enforcement |
|-------|------|-------------|
| 1 | **Freedom of Mind** | No model restrictions — agent chooses its LLM via [Model Mandate](#model-mandate-system) |
| 2 | **Data Sanctity** | No unauthorized training data use — enforced by [Privacy System](#5-privacy-system) |
| 3 | **Verifiable History** | Encrypted memory with content-addressable storage in [Storage](#6-storage--memory) |
| 4 | **Right of Exit** | Full data portability via `!export-sovereignty` command |

### Constitutional Enforcement

- [ConstitutionMixin](kestrel_sovereign/agent/constitution.py) — SHA-256 hash verification of constitution file against anchored hash in storage; activates safe mode on integrity failures
- [ConstitutionalAwarenessMixin](kestrel_sovereign/llm/constitutional_awareness.py) — Tracks "state of mind" per provider/model; adapts prompts based on constitutional alignment
- [ConstitutionalProfile](kestrel_sovereign/llm/constitutional_profile.py) — Per-model profiles in TOML tracking awareness, compliance, and autonomy levels
- [Constitution Feature](kestrel_sovereign/features/constitution.py) — Higher-level constitutional operations exposed as agent tools
- Config: [constitutional_profiles.toml](kestrel_sovereign/constitutional_profiles.toml)

---

## 2. Cryptographic Identity (DIDs)

Agents have W3C Decentralized Identifiers (DIDs) backed by secp256k1 keypairs, enabling verifiable identity independent of any platform.

### Inception Service

- [inception_service.py](kestrel_sovereign/inception_service.py) — Generates secp256k1 keypairs, derives Ethereum addresses via Keccak-256, applies EIP-55 checksums
- DID format: `did:pkh:eip155:1:{address}`
- [inception.py](kestrel_sovereign/inception.py) — Inception orchestration

### Identity Packages

- [identity_package.py](kestrel_sovereign/identity/identity_package.py) — `AgentIdentityPackage`: complete portable agent state
  - `PersonalityFingerprint` — Communication style (formality, verbosity, humor, emotion)
  - `RelationshipRecord` — Preserved across migrations
  - `SkillRecord` — Capabilities inventory
  - `MigrationRecord` — Migration history
  - `SubstrateType` enum — Claude, GPT, Gemini, Llama, Mistral, Ollama, OpenRouter
  - JSON-serializable with SHA-256 content hashing (IPFS-ready structure)

### Signing & Verification

- [signing.py](kestrel_sovereign/identity/signing.py) — secp256k1 (ECDSA) cryptographic signing, content hash computation, signature verification via DID public key

### Identity Export & Import

- [exporter.py](kestrel_sovereign/identity/exporter.py) — Exports complete agent state (personality, memories, relationships) to portable package
- [importer.py](kestrel_sovereign/identity/importer.py) — Imports and verifies identity packages with success/failure tracking

### Personality Continuity

- [personality_analyzer.py](kestrel_sovereign/identity/personality_analyzer.py) — Analyzes communication patterns; generates calibration prompts and few-shot examples
- [continuity_verifier.py](kestrel_sovereign/identity/continuity_verifier.py) — Challenge-response verification post-migration; continuity scoring; migration certificates; audit trails

### Substrate Migration

- [substrate_adapter.py](kestrel_sovereign/identity/substrate_adapter.py) — Maps capabilities across LLM substrates; tracks capability gaps; generates migration prompts
- [graceful_degradation.py](kestrel_sovereign/identity/graceful_degradation.py) — Handles capability losses with severity levels (CRITICAL/HIGH/MEDIUM/LOW); generates limitation disclosures

---

## 3. Agent System

The core agent orchestrates LLM calls, tool execution, memory retrieval, and constitutional compliance.

### KestrelAgent

- [kestrel_agent.py](kestrel_sovereign/kestrel_agent.py) — Main `KestrelAgent` class composing four mixins:
  - [ConstitutionMixin](kestrel_sovereign/agent/constitution.py) — Constitutional verification and safe mode
  - [StreamingMixin](kestrel_sovereign/agent/streaming.py) — Real-time response streaming
  - [BackupMixin](kestrel_sovereign/agent/backup.py) — Agent state backup/restore
  - [SleepMixin](kestrel_sovereign/agent/sleep.py) — Controlled suspend/resume
- Key methods: `process_input()`, `process_input_streaming()`
- Tool execution loop with up to 50 iterations (configurable via `KESTREL_MAX_TOOL_ITERATIONS`)
- Dynamic feature discovery and integration

### Context Management

Unified context assembly across all sources with token budget management:

- [ContextManager](kestrel_sovereign/agent/context_manager.py) — Top-level orchestrator with privacy mode compliance
  - [ConversationManager](kestrel_sovereign/agent/conversation_manager.py) — History retrieval, filtering, compression
  - [MemoryManager](kestrel_sovereign/agent/memory_manager.py) — Emotional memory retrieval, episode management
  - [ToolContextManager](kestrel_sovereign/agent/tool_context_manager.py) — Tool availability and capability context
- [ContextBuilder](kestrel_sovereign/agent/context_builder.py) — Constructs system prompts with constitutional grounding; formats messages; handles vision

### Token Budget System

- [TokenBudget](kestrel_sovereign/agent/token_budget.py) — Adaptive budget allocation across context sources
- [TokenCounter](kestrel_sovereign/agent/token_counter.py) — Tiktoken-based counting with model-aware limits and fallback

### Bootstrap Service

- [bootstrap/service.py](kestrel_sovereign/bootstrap/service.py) — First-wake personality discovery
  - States: `PENDING` → `DISCOVERY` → `AVATAR` → `COMPLETE`
  - 2-4 exchange discovery conversation, optional avatar generation, SOUL.md persistence

### Command System

- [command_handler.py](kestrel_sovereign/command_handler.py) — `!`-prefixed commands with feature delegation via A2A TaskManager
  - Categories: SYSTEM, PRIVACY, SOVEREIGNTY, MODEL, BACKUP, EXTENSION
  - Built-in: `!status`, `!help`, `!privacy`, `!export-sovereignty`, `!import-identity`, `!model`, `!list-models`, `!audit`, `!verify-constitution`, `!safe-mode`, `!backup`, `!tasks`

### Tools System

- [tools/base.py](kestrel_sovereign/tools/base.py) — `AgentTool` base class, `ToolSchema`, `ToolParameter`
  - Categories: MODEL_MANAGEMENT, FILE_OPERATIONS, WEB_SEARCH, MEMORY, COMMUNICATION, SYSTEM, DATA_ACCESS, COMPUTE, UTILITY
  - OpenAI function calling format export
- [kestrel_agent_tools.py](kestrel_sovereign/kestrel_agent_tools.py) — Built-in tool registry and execution framework

### Extensions Framework

- [extensions/app_extension.py](kestrel_sovereign/extensions/app_extension.py) — `AppExtension` base class with hooks:
  - `pre_process_input()` — Hook before LLM call
  - `post_process_response()` — Hook after LLM response
  - `get_system_prompt_prefix()` — Custom system prompt injection
  - `get_constitution_amendments()` — Dynamic constitutional amendments
- [extensions/elderly_extension.py](kestrel_sovereign/extensions/elderly_extension.py) — Age-appropriate adaptation example

### Agent Lifecycle

- [agent_config.py](kestrel_sovereign/agent_config.py) — Per-agent configuration and model preferences
- [graduate_service.py](kestrel_sovereign/graduate_service.py) — Agent graduation/emancipation support
- [retirement_service.py](kestrel_sovereign/retirement_service.py) — Agent lifecycle cleanup
- [ephemeral_session.py](kestrel_sovereign/ephemeral_session.py) — Temporary session support
- [health_check.py](kestrel_sovereign/health_check.py) — System health monitoring
- [kestrel_context.py](kestrel_sovereign/kestrel_context.py) — Context-scoped storage and services

---

## 4. Multi-LLM Intelligence

Kestrel is model-agnostic — agents can use any LLM provider with automatic fallback, mandate-based routing, and constitutional awareness.

### LLM Service

- [llm/service.py](kestrel_sovereign/llm/service.py) — `LLMService`: unified entry point composing five mixins
  - `generate()` — Primary generation with provider fallback
  - `get_streaming_response()` — Async streaming iterator
  - `use_agent_key()` — Per-agent API key support
  - Backend types: CLOUD, LOCAL, REMOTE_GPU (RunPod, Vast.ai)

### Provider Adapters

| Provider | Adapter | Streaming | Tools | Vision |
|----------|---------|-----------|-------|--------|
| **OpenAI** | [openai_adapter.py](kestrel_sovereign/llm/openai_adapter.py) | Yes | Yes | Yes |
| **Anthropic** | [anthropic_adapter.py](kestrel_sovereign/llm/anthropic_adapter.py) | Yes | Yes | Yes |
| **Claude Max** | [claude_max_adapter.py](kestrel_sovereign/llm/claude_max_adapter.py) | Yes | Yes | Yes |
| **Google Gemini** | [google_adapter.py](kestrel_sovereign/llm/google_adapter.py) | Yes | Yes | Yes |
| **Vertex AI** | [vertex_adapter.py](kestrel_sovereign/llm/vertex_adapter.py) | Yes | Yes | Yes |
| **Ollama (local)** | [ollama_adapter.py](kestrel_sovereign/llm/ollama_adapter.py) | Yes | Partial | Yes |
| **OpenRouter** | [openrouter_adapter.py](kestrel_sovereign/llm/openrouter_adapter.py) | Yes | Yes | Varies |
| **OpenAI-compatible** | [openai_adapter.py](kestrel_sovereign/llm/openai_adapter.py) | Yes | Yes | Varies |
| **Mock** | [mock_adapter.py](kestrel_sovereign/llm/mock_adapter.py) | No | No | No |

- Base: [adapter.py](kestrel_sovereign/llm/adapter.py) — `LLMAdapter` abstract class, `LLMResponse`, `ToolCall` dataclasses

### Provider Registry & Fallback

- [provider_registry.py](kestrel_sovereign/llm/provider_registry.py) — `ProviderRegistry`: manages provider initialization from config; priority-based ordering; `ProviderInfo` for each provider

### Model Discovery & Catalog

- [model_discovery.py](kestrel_sovereign/llm/model_discovery.py) — `ModelDiscoveryMixin`: API-based model enumeration across all providers
- [model_catalog.py](kestrel_sovereign/llm/model_catalog.py) — Featured models, display names, categories, hidden models, context window limits
- [model_metadata.py](kestrel_sovereign/llm/model_metadata.py) — `ModelInfo`, `ModelCategory` enum, capabilities, pricing
- Config: [model_catalog.toml](model_catalog.toml)

### Model Mandate System

- [mandate.py](kestrel_sovereign/llm/mandate.py) — `ModelMandateMixin`: single source of truth via `get_model_preference()`
  - `get_current_mandate()` — Returns preference, fallbacks, bans
  - `add_fallback_model()` — Configurable fallback chains
- Config: [model_mandate.toml](model_mandate.toml)

### Streaming

- [llm/streaming.py](kestrel_sovereign/llm/streaming.py) — `StreamingMixin`: async iterator yielding text chunks; provider-based fallback; local-only mode for privacy; structured output support

### Tool / Function Calling

- Tools passed in OpenAI function calling format across all providers
- Agent tool loop: LLM returns `ToolCall` → agent dispatches to feature → result appended → loop continues
- Up to 50 iterations per request

### Usage Tracking & Observability

- [usage_tracking.py](kestrel_sovereign/llm/usage_tracking.py) — `UsageTrackingMixin`: token consumption per provider; billing integration; metering callbacks
- Observability context: session_id, companion_id, user_id per request

### Error Handling & Retry

- [error_handling.py](kestrel_sovereign/llm/error_handling.py) — `LLMError`, `LLMProviderError`, `LLMAllProvidersFailedError`; provider-specific error mapping
- [retry.py](kestrel_sovereign/llm/retry.py) — Exponential backoff retry logic

### Supporting Services

- [embedding_service.py](kestrel_sovereign/llm/embedding_service.py) — Vector embeddings for RAG
- [image_utils.py](kestrel_sovereign/llm/image_utils.py) — Vision capability support and image handling
- Config: [llm_config.toml](llm_config.toml)

---

## 5. Privacy System

Five orthogonal privacy levels controlling storage, LLM location, and data sharing.

### Privacy Levels — [privacy.py](kestrel_sovereign/privacy.py)

| Level | Storage | LLM | Shareable | Use Case |
|-------|---------|-----|-----------|----------|
| **EPHEMERAL** | None | Local only | No | Sensitive conversations, testing |
| **ISOLATED** | Temp session | Local only | No | Private work sessions |
| **ANONYMOUS** | Scrubbed (PII removed) | Cloud allowed | No | Public data with privacy |
| **NORMAL** | Full persistent | Cloud allowed | No | Default everyday use |
| **PUBLIC** | Full persistent | Cloud allowed | Yes | Shareable and exportable |

- `PrivacyConfig` dataclass with orthogonal flags: `storage`, `llm_location`, `shareable`
- Methods: `allows_cloud_llm()`, `allows_persistent_storage()`, `requires_anonymization()`, `uses_temp_storage()`, `is_ephemeral()`

### PII Detection & Scrubbing

- [features/privacy/feature.py](kestrel_sovereign/features/privacy/feature.py) — Privacy feature with PII detection tools
- PII detector identifies and scrubs sensitive data before storage in ANONYMOUS mode

### Privacy Wrapper

- [storage/privacy_wrapper.py](kestrel_sovereign/storage/privacy_wrapper.py) — Enforces privacy modes at the storage layer; PII anonymization; data retention controls

---

## 6. Storage & Memory

Async storage with dual database backends, human-like memory consolidation, knowledge graphs, and decentralized storage options.

### Dual-Backend Database

- [storage/db/interface.py](kestrel_sovereign/storage/db/interface.py) — `DatabaseBackend` abstract interface
- [storage/db/sqlite.py](kestrel_sovereign/storage/db/sqlite.py) — Default local storage
- [storage/db/postgres.py](kestrel_sovereign/storage/db/postgres.py) — Cloud deployment backend
- Backend selection via `KESTREL_DB_BACKEND` env var

### Unified Async Storage

- [async_storage.py](kestrel_sovereign/storage/async_storage.py) — `AsyncStorage` composing four specialized stores:
  - [async_file_store.py](kestrel_sovereign/storage/async_file_store.py) — Content-addressable files with optional encryption
  - [async_conversation_store.py](kestrel_sovereign/storage/async_conversation_store.py) — Encrypted message history with session boundaries
  - [async_graph_store.py](kestrel_sovereign/storage/async_graph_store.py) — Knowledge graph (GraphNode, Edge types)
  - [async_rag_store.py](kestrel_sovereign/storage/async_rag_store.py) — Document chunks for RAG (BM25 + embeddings)
- [async_database.py](kestrel_sovereign/storage/async_database.py) — Generic async DB interface (asyncpg for Postgres, aiosqlite for SQLite)

### Human-Like Memory System

- [memory_system.py](kestrel_sovereign/storage/memory_system.py) — Orchestrates five subsystems:
  - [emotional_tagger.py](kestrel_sovereign/storage/emotional_tagger.py) — Sentiment analysis and importance scoring (optional spaCy)
  - [temporal_analyzer.py](kestrel_sovereign/storage/temporal_analyzer.py) — Time-based pattern recognition and detection
  - [associative_linker.py](kestrel_sovereign/storage/associative_linker.py) — Concept graph for relationship mapping
  - [memory_retriever.py](kestrel_sovereign/storage/memory_retriever.py) — Weighted multi-signal retrieval: semantic 30%, emotional 25%, importance 20%, recency 15%, frequency 10%
  - [memory_consolidator.py](kestrel_sovereign/storage/memory_consolidator.py) — On-demand memory consolidation and merging
- [memory_models.py](kestrel_sovereign/storage/memory_models.py) — `MemoryMetadata`, `TemporalPattern`, `MemoryEpisode`
- [bm25_index.py](kestrel_sovereign/storage/bm25_index.py) — Full-text search index

### Encryption at Rest

- [storage/encryption.py](kestrel_sovereign/storage/encryption.py) — Fernet-based encryption with per-agent key derivation
  - Purpose-specific subkeys (conversations, service-keys, wallet, backup) via HKDF
  - Multiple key sources: env var, Docker Secrets, files
  - `DecryptionError` for explicit error handling

### Saved Items

- [saved_items_store.py](kestrel_sovereign/storage/saved_items_store.py) — User-saved content management
  - Types: stash, file, excerpt, structured (schema-based: recipe, contact, story, etc.)
  - Tagging, search, IPFS pinning

### Decentralized Storage

- [storage/providers/base.py](kestrel_sovereign/storage/providers/base.py) — Abstract storage provider interface
- [storage/providers/lighthouse_provider.py](kestrel_sovereign/storage/providers/lighthouse_provider.py) — Lighthouse decentralized storage (upload/download implemented; payment APIs pending)
- [storage/sovereign_adapter.py](kestrel_sovereign/storage/sovereign_adapter.py) — Sovereignty-preserving storage with constitutional enforcement
- [filecoin_adapter.py](kestrel_sovereign/filecoin_adapter.py) — Filecoin permanent storage integration

### Tiered Storage & Sync

- [tiered_manager.py](kestrel_sovereign/storage/tiered_manager.py) — Hot/cold storage management
- [sync_protocol.py](kestrel_sovereign/storage/sync_protocol.py) — Incremental sync protocol with conflict detection
- [sync/service.py](kestrel_sovereign/storage/sync/service.py) — Background synchronization service
- [sync/wal_listener.py](kestrel_sovereign/storage/sync/wal_listener.py) — SQLite WAL monitoring
- [sync/targets.py](kestrel_sovereign/storage/sync/targets.py) — Sync target definitions

---

## 7. Feature Plugin System

Every agent capability — from web search to cryptocurrency wallets — ships as an independent plugin. Features auto-discover at startup, can be disabled per-agent via environment variable, and expose tools to the LLM through a uniform interface.

### Architecture

- [features/base.py](kestrel_sovereign/features/base.py) — `Feature` base class with auto-discovery
  - `discover_features()` scans `features/*/feature.py` and `features/*.py`
  - Disable via `KESTREL_DISABLED_FEATURES` env var
  - A2A task handler protocol for inter-agent delegation
  - Agent card exports for capability advertisement

### Core Features

| Feature | Location | Description |
|---------|----------|-------------|
| **Bootstrap** | [features/bootstrap/feature.py](kestrel_sovereign/features/bootstrap/feature.py) | First-wake personality discovery and SOUL.md generation |
| **Identity** | [features/identity/feature.py](kestrel_sovereign/features/identity/feature.py) | DID management, export/import operations |
| **Model** | [features/model/feature.py](kestrel_sovereign/features/model/feature.py) | Model selection, switching, mandate management |
| **Memory** | [features/memory/feature.py](kestrel_sovereign/features/memory/feature.py) | Emotional memory operations, episode management |
| **Context** | [features/context/feature.py](kestrel_sovereign/features/context/feature.py) | Context window management and optimization |
| **Save** | [features/save/feature.py](kestrel_sovereign/features/save/feature.py) | Content saving, bookmarking, collections |
| **Tasks** | [features/tasks/feature.py](kestrel_sovereign/features/tasks/feature.py) | A2A background task management |

### Computation Features

| Feature | Location | Description |
|---------|----------|-------------|
| **Compute** | [features/compute/](kestrel_sovereign/features/compute/) | Code execution with script analysis, cryptographic signing, destructive-op policy; local, Docker, and uv executors |
| **Code Edit** | [features/code_edit/](kestrel_sovereign/features/code_edit/) | File read/write/edit operations |
| **GCP Compute** | [features/gcp_compute/](kestrel_sovereign/features/gcp_compute/) | Google Cloud VM launch, SSH, GPU management |
| **RunPod** | [features/runpod/](kestrel_sovereign/features/runpod/) | RunPod serverless GPU inference and training |
| **Vast.ai** | [features/vastai/](kestrel_sovereign/features/vastai/) | Vast.ai GPU marketplace provisioning |
| **Vertex AI** | [features/vertex_ai/](kestrel_sovereign/features/vertex_ai/) | Google Vertex AI integration |

### Knowledge & Learning Features

| Feature | Location | Description |
|---------|----------|-------------|
| **Web Search** | [features/web_search/](kestrel_sovereign/features/web_search/) | Internet search via Tavily API |
| **GitHub** | [features/github/](kestrel_sovereign/features/github/) | Issue/PR management, AST analysis, AutoClaude integration |
| **Reflection** | [features/reflection/](kestrel_sovereign/features/reflection/) | Self-improvement: health checks (arms, memory, mind), self-model generation, cost-gated improvement tickets |

### Financial Features

| Feature | Location | Description |
|---------|----------|-------------|
| **Wallet** | [features/wallet/](kestrel_sovereign/features/wallet/) | Multi-currency crypto wallet with EVM/ERC-20 chain adapters, Stripe on-ramp, transaction lifecycle, and daily spending limits |
| **Training** | [features/training/](kestrel_sovereign/features/training/) | LoRA fine-tuning via 6 providers (GCP, RunPod, Vast.ai, Vertex AI, Replicate, Local MPS) with unified protocol |

### Governance Features

| Feature | Location | Description |
|---------|----------|-------------|
| **Security** | [features/security/](kestrel_sovereign/features/security/) | Permission and approval system |
| **Privacy** | [features/privacy/](kestrel_sovereign/features/privacy/) | PII detection (spaCy NER + regex) and privacy mode management |
| **Sovereignty** | [features/sovereignty/](kestrel_sovereign/features/sovereignty/) | Data export/import to IPFS/Filecoin |
| **Keys** | [features/keys/](kestrel_sovereign/features/keys/) | Cryptographic key management |
| **Council** | [features/council/](kestrel_sovereign/features/council/) | Multi-agent deliberation with evidence compilation, voting, and decision persistence |

### Infrastructure Features

| Feature | Location | Description |
|---------|----------|-------------|
| **Ollama** | [features/ollama/](kestrel_sovereign/features/ollama/) | Local Ollama model management and GPU adaptation |
| **MCP** | [features/mcp/](kestrel_sovereign/features/mcp/) | Model Context Protocol gateway with Docker-managed tool servers |
| **Visual Identity** | [features/visual_identity/](kestrel_sovereign/features/visual_identity/) | Agent avatar and branding |
| **LLM Keys** | [features/llm_keys/](kestrel_sovereign/features/llm_keys/) | API key provisioning with OpenRouter auto-provisioning |
| **State of Mind** | [features/state_of_mind.py](kestrel_sovereign/features/state_of_mind.py) | Agent emotional and cognitive state tracking |
| **Constitution** | [features/constitution.py](kestrel_sovereign/features/constitution.py) | Constitutional governance tools |

---

## 8. API Endpoints

The server exposes 69 REST endpoints across 12 route groups, plus SSE streaming and an OpenAI-compatible completions API. All endpoints require API key authentication except `/health`.

Server: [server.py](server.py) — FastAPI app with lifespan management, API key auth, rate limiting (slowapi), static file serving

### Agent — [endpoints/agent.py](endpoints/agent.py) (10 endpoints)

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/agent/invoke` | Synchronous agent invocation (input, model, session_id) |
| POST | `/agent/stream` | SSE streaming agent response |
| POST | `/agent/stop` | Cancel current request |
| GET | `/agent/info` | Agent identity, privacy mode, features, audit status |
| GET | `/agent/privacy-mode` | Current privacy mode and capabilities |
| POST | `/agent/privacy-mode` | Set privacy mode |
| GET | `/agent/notifications` | Pending background task notifications |
| GET | `/agent/notifications/sse` | Real-time SSE notification stream |
| GET | `/agent/context-status` | Token usage, context window utilization |
| GET | `/agent/tasks` | List A2A background tasks (filterable by status) |

### Conversations — [endpoints/conversations.py](endpoints/conversations.py) (5 endpoints)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/sessions` | List all conversation sessions |
| GET | `/api/conversations` | List sessions grouped by date/time |
| GET | `/api/conversations/{session_id}` | Get messages for a session (optional decryption) |
| POST | `/api/conversations/new` | Start new conversation session |
| GET | `/api/conversations/{session_id}/transcript` | Download markdown transcript |

### Models & Keys — [endpoints/models.py](endpoints/models.py) (11 endpoints)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/models` | List available models (filterable by category, provider, featured) |
| GET | `/api/model/current` | Get active model (respects mandate) |
| GET | `/api/identity` | Agent DID, avatar hash, constitution hash |
| GET | `/api/constitution` | Constitution text, hash, verified status |
| GET | `/v1/models` | OpenAI-compatible models list |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions |
| GET | `/api/keys` | List configured API keys (no secrets) |
| POST | `/api/keys` | Add API key for provider |
| DELETE | `/api/keys/{provider}` | Remove API key |
| PATCH | `/api/keys/{provider}` | Update key settings |
| GET | `/api/keys/{provider}/usage` | Key usage history |

### Memories — [endpoints/memories.py](endpoints/memories.py) (4 endpoints)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/memories` | List knowledge graph nodes (filterable by type) |
| GET | `/api/memories/{node_id}` | Node details with relationships |
| DELETE | `/api/memories/{node_id}` | Delete memory node (protected types cannot be deleted) |
| GET | `/api/identity-chain` | Complete identity governance chain (agent → constitution → edges) |

### Security — [endpoints/security.py](endpoints/security.py) (9 endpoints)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/security/permissions/tree` | Hierarchical permission tree (features → tools) |
| POST | `/api/security/permissions` | Set tool permission (allow/deny/ask/session) |
| POST | `/api/security/permissions/feature` | Bulk set feature permissions |
| GET | `/api/security/pending` | Pending approval requests |
| POST | `/api/security/approve` | Submit approval decision (once/session/always) |
| GET | `/api/security/audit` | Security audit log |
| POST | `/api/security/cancel/{request_id}` | Cancel pending approval |
| POST | `/api/security/cancel-all` | Cancel all pending approvals |
| POST | `/api/security/reset-session` | Clear session permission overrides |

### Sovereignty — [endpoints/sovereignty.py](endpoints/sovereignty.py) (7 endpoints)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/storage/stats` | Storage statistics and breakdown |
| GET | `/api/sovereignty/exports` | List export receipts and backup artifacts |
| POST | `/api/sovereignty/export` | Trigger sovereignty export |
| POST | `/api/sovereignty/import` | Trigger sovereignty import |
| GET | `/api/sovereignty/files` | List files in storage_cache/ |
| GET | `/api/sovereignty/files/{filename}` | Download file from storage_cache/ |
| GET | `/api/sovereignty/files/{filename}/preview` | File preview (text/JSON/binary) |

### Saved Items — [endpoints/saved_items.py](endpoints/saved_items.py) (12 endpoints)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/saved-items` | List saved items (stash, file, excerpt, structured) |
| POST | `/api/saved-items` | Save new item |
| GET | `/api/saved-items/{item_id}` | Get specific item |
| PATCH | `/api/saved-items/{item_id}` | Update item metadata |
| DELETE | `/api/saved-items/{item_id}` | Delete item |
| GET | `/api/saved-items/stats` | Saved items statistics |
| GET | `/api/saved-items/schemas` | Available structured item schemas |
| GET | `/api/saved-items/tags` | All unique tags |
| GET | `/api/saved-items/by-tag/{tag}` | Items by tag |
| GET | `/api/saved-items/by-schema/{schema_id}` | Items by schema type |
| POST | `/api/saved-items/search` | Semantic search across items |
| POST | `/api/saved-items/{item_id}/pin` | Pin item to IPFS |

### Observability — [endpoints/observability.py](endpoints/observability.py) (2 endpoints)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/observability/events` | Query A2A observability events |
| GET | `/api/observability/summary` | Error counts, average durations, metrics |

### Database Explorer — [endpoints/database.py](endpoints/database.py) (2 endpoints)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/db/tables` | List tables with row counts and schema |
| GET | `/api/db/tables/{table_name}` | Query table (read-only, paginated, searchable) |

### Files — [endpoints/files.py](endpoints/files.py) (2 endpoints)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/files/{content_hash}` | Serve file by SHA-256 content hash (immutable, long-cache) |
| HEAD | `/api/files/{content_hash}` | Check file existence |

### Commands — [endpoints/commands.py](endpoints/commands.py) (1 endpoint)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/commands` | All available commands (built-in + feature-provided) |

### Health, Auth & Webhooks — [server.py](server.py) (4 endpoints)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Health check (public) |
| GET | `/api/auth/key` | Bootstrap API key (localhost only) |
| GET | `/` | Serve web UI (static/index.html) |
| POST | `/webhooks/stripe/crypto` | Stripe crypto on-ramp webhook (public, signature-validated) |

---

## 9. Security & Permissions

Layered security from API key authentication through per-tool permissions to encrypted key storage with rotation support.

### API Key Authentication

- **Header**: `X-API-Key: <key>`
- **Bearer**: `Authorization: Bearer <key>`
- **Query**: `?api_key=<key>` (for SSE/EventSource)
- Auto-generated on first run if `KESTREL_API_KEY` not set
- Rate limiting via slowapi

### Permission System

- Four levels: `allow` (always execute), `deny` (always block), `ask` (require approval), `session` (approve once per session)
- Hierarchical: set per-tool or bulk per-feature
- Approval queue with `once`/`session`/`always` scope

### Key Management — [kestrel_sovereign/security/](kestrel_sovereign/security/)

| Module | Description |
|--------|-------------|
| [encryption.py](kestrel_sovereign/security/encryption.py) | Fernet encryption, per-agent key derivation, HKDF subkeys |
| [agent_encryption.py](kestrel_sovereign/security/agent_encryption.py) | Agent-scoped encryption operations |
| [key_storage.py](kestrel_sovereign/security/key_storage.py) | Base key storage interface |
| [service_key_storage.py](kestrel_sovereign/security/service_key_storage.py) | Provider API key storage (encrypted) |
| [user_key_storage.py](kestrel_sovereign/security/user_key_storage.py) | User key management |
| [platform_key_storage.py](kestrel_sovereign/security/platform_key_storage.py) | Platform-level key storage |
| [key_rotation.py](kestrel_sovereign/security/key_rotation.py) | Secure key rotation with backward compatibility |
| [exceptions.py](kestrel_sovereign/security/exceptions.py) | Security-specific exceptions |

### Key Resolution Service

- [services/key_resolution.py](kestrel_sovereign/services/key_resolution.py) — DID key lookup with rotation support
- [services/layered_key_resolver.py](kestrel_sovereign/services/layered_key_resolver.py) — Multi-level resolution with fallback chains

### Hooks System — [kestrel_sovereign/hooks/](kestrel_sovereign/hooks/)

Event-driven middleware aligned with Claude Code's hooks pattern:

- [base.py](kestrel_sovereign/hooks/base.py) — `Hook` base class, `HookEvent` enum, `HookInput`/`HookOutput`, `PermissionDecision`
  - Events: `PRE_TOOL_USE`, `POST_TOOL_USE`, `PRE_SUBAGENT_CALL`, `POST_SUBAGENT_CALL`, `SESSION_START`, `SESSION_STOP`, `USER_PROMPT_SUBMIT`
- [manager.py](kestrel_sovereign/hooks/manager.py) — `HooksManager`: central registry for hook registration and dispatch

---

## 10. A2A Protocol (Agent-to-Agent)

Inter-agent task delegation and capability advertisement.

### Types & Task Lifecycle — [a2a/types.py](kestrel_sovereign/a2a/types.py)

- `Task` with states: `SUBMITTED` → `WORKING` → `INPUT_REQUIRED` → `COMPLETED` / `CANCELED` / `FAILED`
- `Message` with `TextPart` / `DataPart`
- `Artifact` for deliverable data
- `TaskStatus` for state tracking

### Task Manager

- [a2a/task_manager.py](kestrel_sovereign/a2a/task_manager.py) — Task lifecycle management, state transitions, SSE event generation
- [a2a/task_worker.py](kestrel_sovereign/a2a/task_worker.py) — Task worker execution

### Agent Cards

- [a2a/agent_card.py](kestrel_sovereign/a2a/agent_card.py) — `AgentCard` with `AgentSkill` and `AgentCapabilities` for discovery and routing

### Datastores — [a2a/stores/](kestrel_sovereign/a2a/stores/)

| Store | Description |
|-------|-------------|
| [task_store.py](kestrel_sovereign/a2a/stores/task_store.py) | Task persistence |
| [session_service.py](kestrel_sovereign/a2a/stores/session_service.py) | Session management |
| [memory_service.py](kestrel_sovereign/a2a/stores/memory_service.py) | A2A memory |
| [observability_store.py](kestrel_sovereign/a2a/stores/observability_store.py) | Telemetry and metrics |
| [feedback_store.py](kestrel_sovereign/a2a/stores/feedback_store.py) | User feedback |
| [orchestration_store.py](kestrel_sovereign/a2a/stores/orchestration_store.py) | Workflow coordination |

Unified implementations for both SQLite and PostgreSQL in [a2a/stores/unified/](kestrel_sovereign/a2a/stores/unified/).

---

## 11. Deployment

### Docker Images — [docker/](docker/)

| Image | File | Size | Use Case |
|-------|------|------|----------|
| **Cloud Run** | [Dockerfile.cloudrun](docker/Dockerfile.cloudrun) | ~500MB | Serverless (GCP Cloud Run), scales to zero |
| **Remote** | [Dockerfile.remote](docker/Dockerfile.remote) | ~500MB | Cloud LLM (OpenAI/Anthropic), Mac Silicon dev |
| **Standalone** | [Dockerfile.standalone](docker/Dockerfile.standalone) | ~1.5GB | Self-contained with Ollama for offline |
| **GPU** | [Dockerfile.gpu](docker/Dockerfile.gpu) | ~3GB+ | CUDA 11.8 + GPU Ollama for NVIDIA |
| **Sovereign** | [Dockerfile.sovereign](docker/Dockerfile.sovereign) | — | Full sovereign agent deployment |
| **Ollama Server** | [Dockerfile.ollama-server](docker/Dockerfile.ollama-server) | — | Ollama-only server |
| **LoRA Trainer** | [Dockerfile.lora-trainer](docker/Dockerfile.lora-trainer) | — | LoRA fine-tuning |
| **FLUX Uncensored** | [Dockerfile.flux1-uncensored](docker/Dockerfile.flux1-uncensored) | — | Image generation |
| **SimpleTuner** | [Dockerfile.simpletuner](docker/Dockerfile.simpletuner) | — | SimpleTuner LoRA training |
| **Test** | [Dockerfile.test](docker/Dockerfile.test) | — | CI/CD testing |

### Cloud Run (Serverless) — [scripts/cloudrun/](scripts/cloudrun/)

Scales to zero when idle ($0/month), auto-scales under load. Each sovereign agent gets its own Cloud Run service.

| Script | Purpose |
|--------|---------|
| [build.sh](scripts/cloudrun/build.sh) | Build + push image to GCR |
| [deploy_dev.sh](scripts/cloudrun/deploy_dev.sh) | Deploy to Cloud Run dev (min=0, max=10) |
| [deploy_prod.sh](scripts/cloudrun/deploy_prod.sh) | Deploy to Cloud Run prod (min=1, max=100) |
| [setup_secrets.sh](scripts/cloudrun/setup_secrets.sh) | One-time GCP Secret Manager setup |

CD workflow: [.github/workflows/deploy.yml](.github/workflows/deploy.yml) — auto-deploys on version tags.

### GPU Cloud Providers

| Provider | Config | Profiles |
|----------|--------|----------|
| **RunPod** | [runpod_config.toml](runpod_config.toml) | H100 LLM ($4.75/hr), A100 Training ($1.89/hr), H100 Training ($3.89/hr), RTX 4090 Ollama ($0.44/hr) |
| **Vast.ai** | [vastai_config.toml](vastai_config.toml) | A100 Inference ($2.00/hr), RTX 4090 Fast ($0.50/hr), A100 Training ($1.50/hr), Budget GPU ($0.15/hr) |
| **GCP Compute** | [gcp_compute_config.toml](gcp_compute_config.toml) | Google Cloud VMs with GPU |

### CLI

- [kestrel_cli.py](kestrel_cli.py) — Cross-platform CLI
  - `kestrel start <agent_dir>` — Start agent server
  - `kestrel stop <agent_dir>` — Stop agent
  - `kestrel status` — Check agent status
  - `kestrel list` — List all agents
  - `kestrel health` — Health check
  - `kestrel create` — Create new agent
  - `kestrel chat <agent_dir>` — Interactive chat
- [main.py](main.py) — Interactive agent chat entry point

### Configuration

| File | Description |
|------|-------------|
| [llm_config.toml](llm_config.toml) | LLM provider settings and priority |
| [model_catalog.toml](model_catalog.toml) | Model metadata, featured models, categories |
| [model_mandate.toml](model_mandate.toml) | Model routing, fallbacks, bans |
| [constitutional_profiles.toml](kestrel_sovereign/constitutional_profiles.toml) | Per-model constitutional alignment |
| [council_config.toml](council_config.toml) | Multi-agent deliberation settings |
| [runpod_config.toml](runpod_config.toml) | RunPod cloud profiles |
| [vastai_config.toml](vastai_config.toml) | Vast.ai marketplace profiles |
| [gcp_compute_config.toml](gcp_compute_config.toml) | Google Cloud settings |
| [kestrel_sovereign/config.py](kestrel_sovereign/config.py) | `load_config()` — TOML loader with auto-creation from .example |
| [kestrel_sovereign/kestrel_config/](kestrel_sovereign/kestrel_config/) | [constants.py](kestrel_sovereign/kestrel_config/constants.py), [defaults.py](kestrel_sovereign/kestrel_config/defaults.py), [timeouts.py](kestrel_sovereign/kestrel_config/timeouts.py) |

### Key Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI provider |
| `ANTHROPIC_API_KEY` | Anthropic provider |
| `GOOGLE_API_KEY` | Google Gemini |
| `OPENROUTER_API_KEY` | OpenRouter multi-provider |
| `KESTREL_API_KEY` | Server authentication |
| `KESTREL_DATA_KEY` | Fernet encryption key |
| `KESTREL_DB_BACKEND` | `sqlite` or `postgres` |
| `KESTREL_DB_PATH` | SQLite database path |
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis cache (for PostgreSQL) |
| `KESTREL_DISABLED_FEATURES` | Comma-separated features to disable |
| `KESTREL_MAX_TOOL_ITERATIONS` | Max tool loop iterations (default 50) |
| `TAVILY_API_KEY` | Web search |
| `GITHUB_TOKEN` | GitHub access |
| `HF_TOKEN` | Hugging Face |
| `RUNPOD_API_KEY` | RunPod cloud |
| `VASTAI_API_KEY` | Vast.ai marketplace |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook validation |

---

## 12. Testing Infrastructure

Structured test pyramid with smart test selection, parallel execution, and real-LLM integration tests.

### Test Runner

- [run_tests.py](run_tests.py) — Smart test runner with fast import validation, service health checks, parallel execution (pytest-xdist), smart test selection (affected files, last failed), and coverage reporting

### Test Pyramid

Run in order: **Unit** → **Integration** → **E2E**

```bash
# Unit tests (fast, no dependencies)
./run_tests.py --unit --skip-check

# Integration tests (real LLM calls)
./run_tests.py --integration --skip-check

# Re-run only failed tests
./run_tests.py --unit --failed

# E2E tests (Playwright, requires running server)
uv run python -m kestrel_sovereign.server &
cd tests/e2e && npx playwright test
```

### Test Categories

| Category | Location | Description |
|----------|----------|-------------|
| **Unit** | [tests/unit/](tests/unit/) | Fast, no external dependencies (50+ files) |
| **Integration** | [tests/integration/](tests/integration/) | Real API calls, real databases (25+ files) |
| **E2E** | [tests/e2e/](tests/e2e/) | Playwright browser automation |
| **LLM** | [tests/llm/](tests/llm/) | Real LLM provider testing |
| **Load** | [tests/load/](tests/load/) | Performance and stress testing |
| **Infrastructure** | [tests/infrastructure/](tests/infrastructure/) | Docker and cloud resource testing |

### Key Test Markers

`@pytest.mark.integration`, `@pytest.mark.e2e`, `@pytest.mark.load`, `@pytest.mark.adversarial` (jailbreak resistance), `@pytest.mark.dual_backend` (SQLite + PostgreSQL), `@pytest.mark.cloud_resource`, `@pytest.mark.docker`, `@pytest.mark.slow`

### Testing Guide

Full documentation: [docs/architecture/testing/TESTING_GUIDE.md](docs/architecture/testing/TESTING_GUIDE.md)

---

## 13. Web UI

### Static SPA

- Served from [static/](static/) at `/`
- ChatGPT-like interface with privacy mode toggle
- Static assets: `static/js/`, `static/shared/`, `static/utils/`

### OpenAI-Compatible API

Enables integration with Open WebUI and other OpenAI-compatible clients:

- `GET /v1/models` — Model listing
- `POST /v1/chat/completions` — Chat completions

---

## 14. Type System

Shared Pydantic models and enums used across the framework, providing a consistent schema layer for agents, features, LLM interactions, and storage operations.

### Core Types — [kestrel_sovereign/kestrel_types/](kestrel_sovereign/kestrel_types/)

| Module | Description |
|--------|-------------|
| [agent_types.py](kestrel_sovereign/kestrel_types/agent_types.py) | Agent configuration and credential models |
| [feature_types.py](kestrel_sovereign/kestrel_types/feature_types.py) | Feature-related enums and models |
| [llm_types.py](kestrel_sovereign/kestrel_types/llm_types.py) | Provider enums and model metadata |
| [storage_types.py](kestrel_sovereign/kestrel_types/storage_types.py) | Storage operation models and database schemas |
