# Kestrel Architecture Documentation

This directory contains detailed Product Requirements Documents (PRDs) and technical specifications for the Kestrel agent architecture.

## 📂 Directory Structure

### [Core Framework](core/)
Foundational documents defining the agent's existence and infrastructure.
- **[AGENT_ECOSYSTEM.md](core/AGENT_ECOSYSTEM.md)**: System for creating and hosting agents.
- **[FEATURE_AGENT_FRAMEWORK.md](core/FEATURE_AGENT_FRAMEWORK.md)**: The "Society of Agents" architecture.
- **[MULTI_MODEL_SUPPORT.md](core/MULTI_MODEL_SUPPORT.md)**: LLM Router and model abstraction.
- **[INFRASTRUCTURE.md](core/INFRASTRUCTURE.md)**: Development infrastructure and tooling.

### [Storage & Sovereignty](storage/)
Storage protocols, decentralization, and data ownership.
- **[STORAGE_ARCHITECTURE.md](storage/STORAGE_ARCHITECTURE.md)**: High-level multi-tier storage design.
- **[HUMAN_MEMORY_SYSTEM.md](storage/HUMAN_MEMORY_SYSTEM.md)**: Human-like memory with emotional tagging, temporal patterns, and forgetting curves.
- **[DECENTRALIZED_STORAGE.md](storage/DECENTRALIZED_STORAGE.md)**: Vision for IPFS/Filecoin integration.
- **[SOVEREIGNTY_V2_TECHNICAL.md](storage/SOVEREIGNTY_V2_TECHNICAL.md)**: Merkle Forest and convergent encryption.

### [Economics](economics/)
Wallet, solvency, and economic incentives.
- **[AGENT_ECONOMICS.md](economics/AGENT_ECONOMICS.md)**: Vision for sovereign economic entities.
- **[WALLET_AGENT.md](economics/WALLET_AGENT.md)**: Wallet implementation details.
- **[WALLET_SYSTEM.md](WALLET_SYSTEM.md)**: Multi-chain transaction signing (Filecoin, Ethereum, Polygon), ERC-20 tokens, fiat on-ramp.
- **[FILECOIN_WALLET.md](FILECOIN_WALLET.md)**: Legacy Filecoin-specific wallet integration.
- **[SOVEREIGN_SOLVENCY.md](economics/SOVEREIGN_SOLVENCY.md)**: Economic survival and dormancy protocols.
- **[ECONOMIC_INCENTIVES_DEEP_DIVE.md](economics/ECONOMIC_INCENTIVES_DEEP_DIVE.md)**: Payment flows and mechanisms.
- **[ECONOMIC_SYSTEM_PRACTICAL_DETAILS.md](economics/ECONOMIC_SYSTEM_PRACTICAL_DETAILS.md)**: Licensing and buy-out mechanics.

### [Security & Privacy](security/)
Privacy modes, constitution, and cryptographic integrity.
- **[PRIVACY_MODES.md](security/PRIVACY_MODES.md)**: The 5-tier privacy system.
- **[PRIVACY_AGENT.md](security/PRIVACY_AGENT.md)**: Privacy enforcement agent.
- **[CONSTITUTION_EMBEDDING.md](security/CONSTITUTION_EMBEDDING.md)**: Cryptographic binding of the constitution.
- **[CRYPTOGRAPHIC_ANCHORING.md](security/CRYPTOGRAPHIC_ANCHORING.md)**: Immutable event logging.
- **[INTEGRITY_AUDIT_SYSTEM.md](security/INTEGRITY_AUDIT_SYSTEM.md)**: Economic enforcement of ethics.
- **[ANTI_CORRUPTION_ANALYSIS.md](security/ANTI_CORRUPTION_ANALYSIS.md)**: Safeguards against economic corruption.
- **[POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md](security/POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md)**: Migration to NIST PQC standards.

### [Agent Tools](tools/)
Capabilities and tool definitions.
- **[AGENT_TOOLS_ARCHITECTURE.md](tools/AGENT_TOOLS_ARCHITECTURE.md)**: Design for the tool system.
- **[AGENT_TOOLS_IMPLEMENTATION.md](tools/AGENT_TOOLS_IMPLEMENTATION.md)**: Implementation details of current tools.

### [Testing](testing/)
Test plans and quality assurance.
- **[LLM_ROUTER_TESTING_PLAN.md](testing/LLM_ROUTER_TESTING_PLAN.md)**: Comprehensive testing strategy.

### Archived Content
Historical documentation has been moved to `/archive/docs/`:
- Fix summaries → `/archive/docs/fix-summaries/`
- Session notes → `/archive/session-logs/`
- Security audits → `/archive/docs/`
- Outreach materials → `/archive/docs/outreach/`
