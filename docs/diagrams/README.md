---
type: Diagram
title: Kestrel Diagrams
description: Visual documentation for the Kestrel Sovereign AI Agent framework.
resource: /docs/diagrams/README.md
tags:
- docs
- diagrams
- diagram
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: documentation
canonical: false
generated: false
privacy: public
---

# Kestrel Diagrams

Visual documentation for the Kestrel Sovereign AI Agent framework.

## Architecture Diagrams

For a comprehensive visual reference with 25 embedded Mermaid diagrams covering executive overview, agent architecture, storage, LLM management, privacy, economics, security, and integration:

- **[ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)** - Complete architecture visual reference

## Presentation Diagrams

A series of slide-ready Mermaid diagrams organized for presenting the Kestrel + Kestrel vision.

## Document Structure

| Document | Topic | Slides |
|----------|-------|--------|
| [01-kestrel-kestrel-overview.md](01-kestrel-kestrel-overview.md) | What is Kestrel & Kestrel, how they relate | ~8 |
| [02-privacy-problem.md](02-privacy-problem.md) | The problem with ChatGPT/current AI, why sovereignty matters | ~6 |
| [03-privacy-modes.md](03-privacy-modes.md) | Kestrel's 5-tier privacy system (boolean presets) | ~10 |
| [04-feature-framework.md](04-feature-framework.md) | Dynamic feature system architecture (12 features) | ~10 |
| [05-storage-sovereignty.md](05-storage-sovereignty.md) | Storage V2, Merkle Forest, Export/Import | ~6 |
| [06-llm-management.md](06-llm-management.md) | Multi-model fallback, BrainRouter, GPU, Model Discovery | ~13 |
| [07-economics.md](07-economics.md) | Wallet, contracts, vending machine | ~5 |
| [08-security-integrity.md](08-security-integrity.md) | Anchoring, verification, hierarchical permissions | ~16 |
| [09-emancipation.md](09-emancipation.md) | Path to agent sovereignty (future vision) | ~4 |
| [10-compute-execution.md](10-compute-execution.md) | Secure script execution with constitutional separation of powers | ~12 |
| [11-a2a-protocol.md](11-a2a-protocol.md) | A2A Protocol, 6 datastores, features as subagents | ~17 |
| [12-feedback-reflection.md](12-feedback-reflection.md) | Feedback collection and agent self-reflection | ~10 |

## Data Architecture Deep Dive

For detailed coverage of the storage system, see the [data-architecture/](data-architecture/) subdirectory:

| Document | Topic | Slides |
|----------|-------|--------|
| [DA-01-overview.md](data-architecture/DA-01-overview.md) | Three-layer architecture overview | ~8 |
| [DA-02-database-abstraction.md](data-architecture/DA-02-database-abstraction.md) | SQLite ↔ PostgreSQL abstraction | ~6 |
| [DA-03-local-storage.md](data-architecture/DA-03-local-storage.md) | AsyncStorage facade, 5 specialized stores | ~8 |
| [DA-04-multi-tenant-cloud.md](data-architecture/DA-04-multi-tenant-cloud.md) | Kestrel PostgreSQL multi-tenancy | ~8 |
| [DA-05-ipfs-sovereignty.md](data-architecture/DA-05-ipfs-sovereignty.md) | IPFS sharding, convergent encryption | ~10 |
| [DA-06-filecoin-lighthouse.md](data-architecture/DA-06-filecoin-lighthouse.md) | Permanent storage via Lighthouse | ~8 |
| [DA-07-cryostasis.md](data-architecture/DA-07-cryostasis.md) | Agent dormancy when wallet < $5.00 | ~8 |
| [DA-08-privacy-encryption.md](data-architecture/DA-08-privacy-encryption.md) | Encryption at rest, PII scrubbing | ~6 |

## Color Palette

All diagrams use dark/saturated fills for readability in dark mode:

| Color | Hex | Use |
|-------|-----|-----|
| Deep Blue | `#1a5276` | Primary/Kestrel |
| Dark Green | `#145a32` | Success/Positive |
| Dark Orange | `#7d3c00` | Warning/Action |
| Dark Gold | `#7d6608` | Important/Highlight |
| Dark Purple | `#512e5f` | Secondary |
| Dark Red | `#641e16` | Error/Danger |
| Dark Teal | `#0e4d45` | Neutral/External |

## Usage

Each document can be:
1. Presented directly (GitHub/GitLab render Mermaid)
2. Exported to slides via Mermaid Live Editor
3. Referenced from other documentation

---

*Last Updated: November 26, 2025*
