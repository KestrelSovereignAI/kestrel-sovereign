---
type: Architecture Spec
title: Kestrel Architecture Documentation
description: Revalidated route map from architecture documents to their current
  implementation or design owners.
resource: /docs/architecture/README.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-07-24T00:00:00Z'
status: active
owner: documentation
canonical: true
generated: false
privacy: public
---

# Kestrel Architecture Documentation

This is the route map for Kestrel architecture documentation. A row in the
primary map is either a verified description of the current system or an
explicit design of record. Everything else remains available in the
reference/archive map without being presented as an implementation contract.

Feature/package availability is governed by
[`KESTREL_FEATURES.md`](../../KESTREL_FEATURES.md) and
[`kestrel_sovereign/data/feature_registry.toml`](../../kestrel_sovereign/data/feature_registry.toml).
When a document and its named implementation owner disagree, the implementation
and those inventories win; update the document and this map together.

## Status taxonomy

These are the only statuses valid in the maps below.

| Status | Contract |
|---|---|
| **Active** | Revalidated description of a current implementation. The owner column names the source module or package that is authoritative. |
| **Design of record** | Current intended architecture, including explicitly identified deferred scope. It is not a claim that every described capability ships. |
| **Experimental** | Some implementation exists, but gaps or stale integration details prevent the document from serving as a current contract. |
| **Planning** | Proposal or aspirational design with no complete current implementation contract. |
| **Historical** | Superseded PRD, point-in-time audit, old runbook, or pre-extraction implementation record retained for context. |
| **Strategy** | Product or economic reasoning, not a source-code or package contract. |

## Primary implementation map

Only **Active** and **Design of record** rows belong here.

<!-- architecture-index:primary:start -->

### Core runtime

| Document | Status | Current owner / location | Scope |
|---|---|---|---|
| [AGENT_IDENTITY_CONTRACT.md](AGENT_IDENTITY_CONTRACT.md) | **Active** | `kestrel-sovereign`: `kestrel_sovereign/kestrel_agent.py` and `kestrel_sovereign/identity/` | DID identity and the derived `agent_id` compatibility property. |
| [LLM_SERVICE_ARCHITECTURE.md](LLM_SERVICE_ARCHITECTURE.md) | **Active** | `kestrel-sovereign`: `kestrel_sovereign/llm/`; shared adapter types are owned by `kestrel-sovereign-sdk` | Canonical vendor / route / model service and routing contract. |
| [DYNAMIC_TOOL_LOADING.md](DYNAMIC_TOOL_LOADING.md) | **Active** | `kestrel-sovereign`: `kestrel_sovereign/agent/tool_registry.py` and `kestrel_sovereign/agent/orchestrator_engine.py` | Progressive promotion from feature dispatchers to direct tools. |
| [FEATURE_CLI_ADAPTERS.md](FEATURE_CLI_ADAPTERS.md) | **Active** | `kestrel-sovereign`: `kestrel_sovereign/features/cli/` | Allowlisted local CLI adapter contract; remote GitHub access is excluded. |

### LLM substrate

| Document | Status | Current owner / location | Scope |
|---|---|---|---|
| [llm/PROVIDER_PLUGINS.md](llm/PROVIDER_PLUGINS.md) | **Active** | `kestrel-sovereign-sdk`: `kestrel_sdk.llm`; discovery is owned by `kestrel_sovereign/llm/provider_registry.py` | External LLM adapter and entry-point contract. |
| [llm/HONESTY_LAYER.md](llm/HONESTY_LAYER.md) | **Active** | `kestrel-sovereign`: `kestrel_sovereign/agent/streaming.py` and `kestrel_sovereign/security/narration_check.py`; event types are in `kestrel-sovereign-sdk` | Tool-call streaming, revise events, and deterministic narration checks. |

### Memory and storage

| Document | Status | Current owner / location | Scope |
|---|---|---|---|
| [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md) | **Active** | `kestrel-sovereign`: `kestrel_sovereign/storage/memory_system.py`, `memory_retriever.py`, and `memory_consolidator.py` | Cognitive memory, scoring, consolidation, and documented deployment deviations. |
| [MEMORY_OWNERSHIP.md](MEMORY_OWNERSHIP.md) | **Active** | `kestrel-sovereign`: `kestrel_sovereign/storage/`, `kestrel_sovereign/agent/`, and bundled memory feature modules | Current facade, context, feature, RAG, and A2A ownership boundaries. |
| [storage/STORAGE_ARCHITECTURE.md](storage/STORAGE_ARCHITECTURE.md) | **Active** | `kestrel-sovereign`: `kestrel_sovereign/storage/` | Current async SQLite/PostgreSQL, SQLAlchemy, and vector-storage layers. |

### Security and privacy

| Document | Status | Current owner / location | Scope |
|---|---|---|---|
| [security/PRIVACY_MODES.md](security/PRIVACY_MODES.md) | **Active** | `kestrel-sovereign`: `kestrel_sovereign/privacy.py` and `kestrel_sovereign/storage/privacy_wrapper.py` | Privacy flags, presets, and storage/processing enforcement. |
| [security/KEY_ROTATION.md](security/KEY_ROTATION.md) | **Active** | `kestrel-sovereign`: `kestrel_sovereign/security/key_rotation.py` | Resumable `KESTREL_DATA_KEY` rotation service and CLI. |
| [security/PQ_THREAT_MODEL.md](security/PQ_THREAT_MODEL.md) | **Active** | `kestrel-sovereign`: `kestrel_sovereign/security/` and `kestrel_sovereign/identity/` | Current long-horizon cryptographic threat model and control mapping. |

### Features and tools

| Document | Status | Current owner / location | Scope |
|---|---|---|---|
| [COMPUTE_FEATURE_DESIGN.md](COMPUTE_FEATURE_DESIGN.md) | **Active** | `kestrel-sovereign`: bundled `kestrel_sovereign/features/compute/` and security approval hooks | Guarded write / sign / review / execute flow. |
| [GITHUB_FEATURE_DESIGN.md](GITHUB_FEATURE_DESIGN.md) | **Design of record** | Extracted package `kestrel-feature-github` (`kestrel_feature_github`) | Shipped repository/issue inspection plus explicitly deferred deep static analysis. |
| [WALLET_SYSTEM.md](WALLET_SYSTEM.md) | **Design of record** | Extracted package `kestrel-feature-wallet` (`kestrel_feature_wallet`) | Multi-chain wallet design; package source and tests, not this repository, establish shipped scope. |
| [tools/AGENT_TOOLS_ARCHITECTURE.md](tools/AGENT_TOOLS_ARCHITECTURE.md) | **Active** | `kestrel-sovereign`: feature discovery and `kestrel_sovereign/agent/tool_registry.py`; schemas are owned by `kestrel-sovereign-sdk` | Feature-owned tool architecture and runtime flow. |
| [tools/AGENT_TOOLS_IMPLEMENTATION.md](tools/AGENT_TOOLS_IMPLEMENTATION.md) | **Active** | `kestrel-sovereign`: `kestrel_sovereign/features/base.py`, `agent/tool_registry.py`, and `tools/result_contract.py` | File-level implementation map for the tool runtime. |
| [tools/AGENT_TOOLS.md](tools/AGENT_TOOLS.md) | **Active** | `kestrel-sovereign` plus installed feature packages; live inventory is `KESTREL_FEATURES.md` and runtime discovery | Tool concepts and categories, not a fixed tool count. |

### Testing

| Document | Status | Current owner / location | Scope |
|---|---|---|---|
| [testing/TESTING_GUIDE.md](testing/TESTING_GUIDE.md) | **Active** | `kestrel-sovereign`: `run_tests.py`, `tests/`, and repository CI | Canonical test pyramid and local runner guide. |

<!-- architecture-index:primary:end -->

## Reference and archive map

These links are retained for design archaeology and follow-up work. They are
not current implementation contracts.

<!-- architecture-index:reference:start -->

### Partial implementations and open designs

| Document | Status | Current owner / location | Why it is outside the primary map |
|---|---|---|---|
| [NIGHTLY_FORGETTING.md](NIGHTLY_FORGETTING.md) | **Experimental** | Current P1-P3 code: `kestrel_sovereign/agent/sleep.py` and `kestrel_sovereign/storage/memory_consolidator.py`; P4 is design-only | The page intentionally mixes shipped P1-P3 behavior with an unimplemented reflection-table phase. |
| [security/PRIVACY_AGENT.md](security/PRIVACY_AGENT.md) | **Experimental** | Current component: `kestrel_sovereign/features/privacy/feature.py`; presets and flags are owned by `kestrel_sovereign/privacy.py` | The original PRD maps to a shipped component but predates current PUBLIC/DEIDENTIFIED presets and transition rules. |
| [security/KEY_MANAGEMENT.md](security/KEY_MANAGEMENT.md) | **Experimental** | Current data-key code: `kestrel_sovereign/security/key_storage.py` and `kestrel_sovereign/security/encryption.py`; wallet-key separation belongs to `kestrel-feature-wallet` | The page combines implemented data-key custody with planned wallet-key and container-lifecycle designs. |
| [security/CONSTITUTION_EMBEDDING.md](security/CONSTITUTION_EMBEDDING.md) | **Experimental** | Current code: `kestrel_sovereign/constitution/`, `kestrel_sovereign/inception_service.py`, and `kestrel_sovereign/agent/constitution.py` | The anchoring flow exists, but the page describes only legacy `did:pkh` inception and omits born-hybrid and succession paths. |
| [TRAINING_PROVIDER_ARCHITECTURE.md](TRAINING_PROVIDER_ARCHITECTURE.md) | **Experimental** | Protocol/factory: `kestrel_sovereign/features/training/`; cloud managers: `kestrel-cloud-runpod`, `kestrel-cloud-vastai`, and `kestrel-cloud-gcp` | The core adapter seam exists, but paths, priority, and extracted-provider details in the page require refresh. |

### Proposals and design-only material

| Document | Status | Current owner / location | Why it is outside the primary map |
|---|---|---|---|
| [USER_LIFECYCLE_MANAGEMENT.md](USER_LIFECYCLE_MANAGEMENT.md) | **Planning** | Design-only; current agent creation/retirement code is `kestrel_sovereign/inception_service.py` and `kestrel_sovereign/retirement_service.py` | Proposed SaaS user/companion schema is not the current persistence contract. |
| [core/AGENT_ECOSYSTEM.md](core/AGENT_ECOSYSTEM.md) | **Planning** | Design-only; current inception and fleet ownership is `kestrel_sovereign/inception_service.py`, `kestrel_sovereign/multi_agent/`, and `kestrel_sovereign/fleet/` | Multi-phase Genesis Factory and Capsule Host vision. |
| [storage/DECENTRALIZED_STORAGE.md](storage/DECENTRALIZED_STORAGE.md) | **Planning** | Design-only integration narrative; current storage providers are separate `kestrel-storage-*` packages and core export is `kestrel_sovereign/storage/sovereign_adapter.py` | IPFS/Filecoin, Kavach, and x402 vision exceeds the current provider contract. |
| [economics/AGENT_ECONOMICS.md](economics/AGENT_ECONOMICS.md) | **Planning** | Design-only; implemented wallet capability belongs to `kestrel-feature-wallet` | Aspirational contracts, service catalog, and autonomous economic entity model. |
| [economics/SOVEREIGN_SOLVENCY.md](economics/SOVEREIGN_SOLVENCY.md) | **Planning** | Design-only; adjacent runtime owners are `kestrel-feature-wallet` and core retirement/sovereignty services | Draft solvency, cryostasis, and wake-up protocol. |

### Superseded PRDs, audits, and runbooks

| Document | Status | Current owner / location | Why it is outside the primary map |
|---|---|---|---|
| [core/FEATURE_AGENT_FRAMEWORK.md](core/FEATURE_AGENT_FRAMEWORK.md) | **Historical** | Superseded by `kestrel-sovereign-sdk` feature/tool contracts and core entry-point discovery in `kestrel_sovereign/features/` | Pre-SDK “Feature Agent” PRD and obsolete in-tree layout. |
| [core/MULTI_MODEL_SUPPORT.md](core/MULTI_MODEL_SUPPORT.md) | **Historical** | Superseded by `LLM_SERVICE_ARCHITECTURE.md`, `kestrel_sovereign/llm/`, and `kestrel-sovereign-sdk` | Original single-string provider PRD. |
| [core/INFRASTRUCTURE.md](core/INFRASTRUCTURE.md) | **Historical** | Historical Claude worktree workflow; current repository automation lives in `.github/`, `.kestreltalon/`, and project scripts | Removed subagent and slash-command development setup. |
| [storage/HUMAN_MEMORY_SYSTEM.md](storage/HUMAN_MEMORY_SYSTEM.md) | **Historical** | Superseded by `MEMORY_SYSTEM.md` and `storage/STORAGE_ARCHITECTURE.md` | Original human-memory design narrative. |
| [storage/SOVEREIGNTY_IMPLEMENTATION.md](storage/SOVEREIGNTY_IMPLEMENTATION.md) | **Historical** | Superseded by V3 CAR code in `kestrel_sovereign/storage/sovereign_adapter.py` | Removed V2 `SovereigntyReceipt` and `AgentSnapshot` implementation map. |
| [storage/SOVEREIGNTY_V2_TECHNICAL.md](storage/SOVEREIGNTY_V2_TECHNICAL.md) | **Historical** | Superseded by V3 CAR code in `kestrel_sovereign/storage/sovereign_adapter.py` | V2 multi-CID Merkle-forest design. |
| [security/CRYPTOGRAPHIC_ANCHORING.md](security/CRYPTOGRAPHIC_ANCHORING.md) | **Historical** | Current local anchor owner: bundled `kestrel_sovereign/features/audit_anchor/`; no external blockchain anchor is implemented in core | Original blockchain-notary PRD differs from the shipped local tamper-evident anchor. |
| [security/INTEGRITY_AUDIT_SYSTEM.md](security/INTEGRITY_AUDIT_SYSTEM.md) | **Historical** | Current response audit: `kestrel_sovereign/features/response_audit/`; wallet capability is separately owned by `kestrel-feature-wallet` | Superseded FIL-funded economic audit design. |
| [security/ANTI_CORRUPTION_ANALYSIS.md](security/ANTI_CORRUPTION_ANALYSIS.md) | **Historical** | Companion to the superseded economic audit design; current response audit is `kestrel_sovereign/features/response_audit/` | Point-in-time analysis of the old FIL fee model. |
| [security/POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md](security/POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md) | **Historical** | Shipped controls: `kestrel_sovereign/security/` and `kestrel_sovereign/identity/` | Completed migration PRD retained as a wave-by-wave build record. |
| [security/CRYPTO_INVENTORY.md](security/CRYPTO_INVENTORY.md) | **Historical** | Historical Wave 0A inventory; current primitives live in `kestrel_sovereign/security/crypto_suite.py`, `kem_suite.py`, and identity modules | Pre-migration inventory contains “future” and “none in use” statements superseded by shipped PQ code. |
| [security/SERIALIZATION_COMPATIBILITY.md](security/SERIALIZATION_COMPATIBILITY.md) | **Historical** | Historical Wave 0A matrix; current serializers/verifiers live under `kestrel_sovereign/identity/`, `security/`, and `spawn/` | Pre-migration v1-to-v2 design matrix. |
| [subagent_isolation_audit.md](subagent_isolation_audit.md) | **Historical** | Audit snapshot only; current feature boundaries are discovered from `kestrel_sovereign/features/` and external entry points | April 2026 inventory of then-current feature state access. |
| [PLAN_RUNPOD_INTEGRATION.md](PLAN_RUNPOD_INTEGRATION.md) | **Historical** | Provider ownership moved to `kestrel-cloud-runpod`; core retains adapter/CLI seams only | Pre-extraction RunPod PRD with removed in-core paths. |
| [RUNPOD_LORA_TRAINING.md](RUNPOD_LORA_TRAINING.md) | **Historical** | Historical Q1 2026 runbook; provider is `kestrel-cloud-runpod`, while training adapters live in core and specialized LoRA work is external | Reproduction notes for an old deployment, not current operations. |
| [VASTAI_TRAINING.md](VASTAI_TRAINING.md) | **Historical** | Deprioritized training experiment; general provider ownership moved to `kestrel-cloud-vastai` | Failed GCR-auth experiment and obsolete in-core path claims. |
| [FILECOIN_WALLET.md](FILECOIN_WALLET.md) | **Historical** | Current wallet owner: extracted package `kestrel-feature-wallet` (`kestrel_feature_wallet`) | Pre-extraction quick start and test paths are no longer repository-local contracts. |
| [economics/WALLET_AGENT.md](economics/WALLET_AGENT.md) | **Historical** | Superseded by extracted package `kestrel-feature-wallet` | Original in-memory, in-core `WalletAgent` PRD. |
| [economics/ECONOMICS_WORK_SESSION.md](economics/ECONOMICS_WORK_SESSION.md) | **Historical** | Design-session snapshot; no runtime owner | Scratchpad whose proposals were never an implementation contract. |
| [testing/LLM_ROUTER_TESTING_PLAN.md](testing/LLM_ROUTER_TESTING_PLAN.md) | **Historical** | Current tests live under `tests/`; current test contract is `testing/TESTING_GUIDE.md` | Planned file list includes tests that were never created or were superseded. |

### Strategy references

| Document | Status | Current owner / location | Why it is outside the primary map |
|---|---|---|---|
| [PROVIDER_ECONOMICS.md](PROVIDER_ECONOMICS.md) | **Strategy** | Product strategy; implemented payer primitives are owned by `kestrel-sovereign-sdk` and `kestrel_sovereign/services/payer_resolver.py` | Revenue and referral analysis is not a provider runtime contract. |
| [economics/ECONOMIC_INCENTIVES_DEEP_DIVE.md](economics/ECONOMIC_INCENTIVES_DEEP_DIVE.md) | **Strategy** | Product/economic narrative; wallet implementation belongs to `kestrel-feature-wallet` | Illustrative FIL funding and 90/10 ethics-pool model. |
| [economics/ECONOMIC_SYSTEM_PRACTICAL_DETAILS.md](economics/ECONOMIC_SYSTEM_PRACTICAL_DETAILS.md) | **Strategy** | Product/licensing narrative; no runtime owner | Illustrative subscriptions, buy-outs, and pricing. |

<!-- architecture-index:reference:end -->

## Related authoritative references

- **Live feature and package inventory:** [`/KESTREL_FEATURES.md`](../../KESTREL_FEATURES.md)
  and [`feature_registry.toml`](../../kestrel_sovereign/data/feature_registry.toml)
- **Full repository and package map:** [`/docs/audit/REPO_MAP.md`](../audit/REPO_MAP.md)
  and [`/docs/ECOSYSTEM.md`](../ECOSYSTEM.md)
- **Constitution:** [`/docs/principles/KESTREL_CONSTITUTION.md`](../principles/KESTREL_CONSTITUTION.md)
- **Cloud Run operations:** [`/docs/deployment/README.md`](../deployment/README.md)
- **Generated OKF inventory:** [`index.md`](index.md) and [`log.md`](log.md)
