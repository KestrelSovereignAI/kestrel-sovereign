# Kestrel Architecture Documentation

Detailed Product Requirements Documents (PRDs) and technical specifications for the Kestrel agent architecture.

> **Status conventions used below:**
> - **Active** — describes the current shipped system
> - **Experimental** — code exists and works on the happy path, but has known gaps; integration tests may skip in CI without external credentials
> - **Aspirational** — design-of-record document; partially implemented (specifics in the doc's own banner)

---

## 📂 Directory Structure

### Core framework

Foundational documents defining the agent's existence and runtime.

- **[AGENT_IDENTITY_CONTRACT.md](AGENT_IDENTITY_CONTRACT.md)** — DID as the single source of truth; `agent_id` as a property derived from the DID. *Active.*
- **[LLM_SERVICE_ARCHITECTURE.md](LLM_SERVICE_ARCHITECTURE.md)** — Vendor / route / model architecture; retry, structured output, vision, streaming. *Active (canonical).*
- **[DYNAMIC_TOOL_LOADING.md](DYNAMIC_TOOL_LOADING.md)** — Direct tool registration after subagent dispatch. *Active.*
- **[FEATURE_CLI_ADAPTERS.md](FEATURE_CLI_ADAPTERS.md)** — Feature-owned CLI adapter contract for authenticated local command-line tools. *Active.*
- **[USER_LIFECYCLE_MANAGEMENT.md](USER_LIFECYCLE_MANAGEMENT.md)** — Soft / hard delete strategies and cryo storage. *Active.*
- **[core/AGENT_ECOSYSTEM.md](core/AGENT_ECOSYSTEM.md)** — Agent creation and the Genesis factory. *Active.*
- **[core/FEATURE_AGENT_FRAMEWORK.md](core/FEATURE_AGENT_FRAMEWORK.md)** — The "Society of Agents" architecture; Feature base class and `@tool` decorators. *Active.*
- **[core/MULTI_MODEL_SUPPORT.md](core/MULTI_MODEL_SUPPORT.md)** — LLMAdapter abstraction and provider fallback. *Active.*
- **[core/INFRASTRUCTURE.md](core/INFRASTRUCTURE.md)** — Parallel-development workflow and tooling. *Active.*

### LLM substrate

Adapter contract, plugin authoring, streaming, and the constitutional honesty layer. The high-level service architecture is documented above in [LLM_SERVICE_ARCHITECTURE.md](LLM_SERVICE_ARCHITECTURE.md); these are the leaves of that tree.

- **[llm/PROVIDER_PLUGINS.md](llm/PROVIDER_PLUGINS.md)** — How to ship a third-party LLM adapter as a `pip`-installable plugin. SDK contract surface, marker emission rules, conformance suite. *Active.*
- **[llm/HONESTY_LAYER.md](llm/HONESTY_LAYER.md)** — The end-to-end streaming honesty layer: `ToolCallStarted` markers, in-band revise sentinel + SSE backup, deterministic narration check in the audit hook. Closes [#1042](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1042). *Active.*

### Memory & storage

Storage protocols, memory systems, and data ownership.

- **[MEMORY_SYSTEM.md](MEMORY_SYSTEM.md)** — Emotional tagging, Ebbinghaus decay, consolidation, memory pinning. *Active.*
- **[MEMORY_OWNERSHIP.md](MEMORY_OWNERSHIP.md)** — Memory layer assignments and the facade pattern. *Active.*
- **[storage/STORAGE_ARCHITECTURE.md](storage/STORAGE_ARCHITECTURE.md)** — Multi-tier storage (PostgreSQL / SQLite / IndexedDB / ephemeral). *Active.*
- **[storage/HUMAN_MEMORY_SYSTEM.md](storage/HUMAN_MEMORY_SYSTEM.md)** — Human-like memory with temporal patterns. *Active.*
- **[storage/DECENTRALIZED_STORAGE.md](storage/DECENTRALIZED_STORAGE.md)** — IPFS / Filecoin integration design. *Active.*
- **[storage/SOVEREIGNTY_IMPLEMENTATION.md](storage/SOVEREIGNTY_IMPLEMENTATION.md)** — Sovereignty implementation details. *Active.*
- **[storage/SOVEREIGNTY_V2_TECHNICAL.md](storage/SOVEREIGNTY_V2_TECHNICAL.md)** — Merkle Forest and convergent encryption. *Active.*

### Security & privacy

Privacy modes, key management, constitution, and cryptographic integrity.

- **[security/PRIVACY_MODES.md](security/PRIVACY_MODES.md)** — The 5-tier privacy system (EPHEMERAL / ISOLATED / ANONYMOUS / NORMAL / PUBLIC). *Active.*
- **[security/PRIVACY_AGENT.md](security/PRIVACY_AGENT.md)** — Privacy enforcement agent. *Active.*
- **[security/KEY_MANAGEMENT.md](security/KEY_MANAGEMENT.md)** — Two-layer key architecture (KESTREL_DATA_KEY + agent private key). *Active.*
- **[security/KEY_ROTATION.md](security/KEY_ROTATION.md)** — Key rotation procedures. *Active.*
- **[security/CONSTITUTION_EMBEDDING.md](security/CONSTITUTION_EMBEDDING.md)** — Cryptographic binding of the constitution. *Active.*
- **[security/CRYPTOGRAPHIC_ANCHORING.md](security/CRYPTOGRAPHIC_ANCHORING.md)** — Immutable event logging. *Active.*
- **[security/INTEGRITY_AUDIT_SYSTEM.md](security/INTEGRITY_AUDIT_SYSTEM.md)** — Economic enforcement of ethics. *Active.*
- **[security/ANTI_CORRUPTION_ANALYSIS.md](security/ANTI_CORRUPTION_ANALYSIS.md)** — Safeguards against economic corruption. *Active.*
- **[security/POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md](security/POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md)** — Quantum hardening PRD-v2 (active). Tracking [epic #921](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/921).
- **[security/CRYPTO_INVENTORY.md](security/CRYPTO_INVENTORY.md)** — Authoritative inventory of every cryptographic primitive in use. *Active (Wave 0A).*
- **[security/PQ_THREAT_MODEL.md](security/PQ_THREAT_MODEL.md)** — Post-quantum threat model: what Shor breaks, what Grover degrades, what is HNDL-relevant. *Active (Wave 0A).*
- **[security/SERIALIZATION_COMPATIBILITY.md](security/SERIALIZATION_COMPATIBILITY.md)** — Serialization compatibility matrix for signed/encrypted artifacts (v1 → v2 migration). *Active (Wave 0A).*
- **[subagent_isolation_audit.md](subagent_isolation_audit.md)** — Cross-feature subagent isolation audit (Feb 2026). *Active.*

### Features

Specific feature designs and implementations.

- **[COMPUTE_FEATURE_DESIGN.md](COMPUTE_FEATURE_DESIGN.md)** — Write / sign / review / execute pattern for sandboxed code execution. *Active.*
- **[GITHUB_FEATURE_DESIGN.md](GITHUB_FEATURE_DESIGN.md)** — GitHub code introspection. *Aspirational* — see banner; ~half of the design ships.

### Cloud & GPU compute

Documents covering cloud-provider integrations. **All flagged experimental** — see each doc's banner for ground truth on what's shipped vs. designed.

- **[PLAN_RUNPOD_INTEGRATION.md](PLAN_RUNPOD_INTEGRATION.md)** — RunPod GPU pods (direct + managed modes). *Experimental.*
- **[RUNPOD_LORA_TRAINING.md](RUNPOD_LORA_TRAINING.md)** — RunPod LoRA training operational guide (Q1 2026). *Experimental, predates training/provider library split.*
- **[VASTAI_TRAINING.md](VASTAI_TRAINING.md)** — VastAI as a training backend. *Deprioritized (Dec 2025).* Note: VastAI as a general compute provider is separate and active.
- **[TRAINING_PROVIDER_ARCHITECTURE.md](TRAINING_PROVIDER_ARCHITECTURE.md)** — Protocol + factory for training providers. *Active (library).*
- **[PROVIDER_ECONOMICS.md](PROVIDER_ECONOMICS.md)** — Middleman / referral revenue model. *Strategy doc.*

### Wallets & economics

Wallet, solvency, and economic incentives.

- **[WALLET_SYSTEM.md](WALLET_SYSTEM.md)** — Multi-chain transaction signing (Filecoin, Ethereum, Polygon), ERC-20 tokens, fiat on-ramp. *Active.*
- **[FILECOIN_WALLET.md](FILECOIN_WALLET.md)** — Filecoin-specific wallet integration. *Active.*
- **[economics/AGENT_ECONOMICS.md](economics/AGENT_ECONOMICS.md)** — Vision for sovereign economic entities. *Aspirational.*
- **[economics/WALLET_AGENT.md](economics/WALLET_AGENT.md)** — Wallet agent implementation. *Active.*
- **[economics/SOVEREIGN_SOLVENCY.md](economics/SOVEREIGN_SOLVENCY.md)** — Economic survival and dormancy protocols. *Aspirational.*
- **[economics/ECONOMIC_INCENTIVES_DEEP_DIVE.md](economics/ECONOMIC_INCENTIVES_DEEP_DIVE.md)** — Payment flows and mechanisms. *Active.*
- **[economics/ECONOMIC_SYSTEM_PRACTICAL_DETAILS.md](economics/ECONOMIC_SYSTEM_PRACTICAL_DETAILS.md)** — Licensing and buy-out mechanics. *Active.*
- **[economics/ECONOMICS_WORK_SESSION.md](economics/ECONOMICS_WORK_SESSION.md)** — Working notes from economics design sessions. *Internal.*

### Tools

- **[tools/AGENT_TOOLS_ARCHITECTURE.md](tools/AGENT_TOOLS_ARCHITECTURE.md)** — Design for the tool system. *Active.*
- **[tools/AGENT_TOOLS_IMPLEMENTATION.md](tools/AGENT_TOOLS_IMPLEMENTATION.md)** — Current tool implementations. *Active.*
- **[tools/AGENT_TOOLS.md](tools/AGENT_TOOLS.md)** — Tool catalog. *Active.*

### Testing

- **[testing/TESTING_GUIDE.md](testing/TESTING_GUIDE.md)** — Test runner, pyramid strategy, SQLite WAL notes. *Active (canonical).*
- **[testing/LLM_ROUTER_TESTING_PLAN.md](testing/LLM_ROUTER_TESTING_PLAN.md)** — LLM-router testing strategy. *Active.*

---

## Related references

- **Engineering quality program:** [`/docs/audit/`](../audit/) — feature proof matrices, seam campaigns, sync/async audit. Actively maintained.
- **Live feature inventory:** [`/KESTREL_FEATURES.md`](../../KESTREL_FEATURES.md) (canonical, consumed by `scripts/generate_feature_docs.py`)
- **Constitution:** [`/docs/principles/KESTREL_CONSTITUTION.md`](../principles/KESTREL_CONSTITUTION.md)

---

## Archive

Historical / superseded architecture documents, when present, live under [`/docs/archive/`](../archive/) (gitignored). The legacy feature inventory is preserved at [`/docs/archive/KESTREL_FEATURES_legacy.md`](../archive/KESTREL_FEATURES_legacy.md).
