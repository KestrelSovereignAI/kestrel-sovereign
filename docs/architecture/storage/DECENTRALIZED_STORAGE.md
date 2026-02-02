# Decentralized Storage Vision: IPFS & Filecoin Integration

**Last Updated:** November 23, 2025

---

## Vision

The Kestrel project aims to provide **true data sovereignty** by ensuring that an agent's memory and identity can persist independently of any single platform or provider. To achieve this, we integrate with decentralized storage networks, specifically **IPFS (InterPlanetary File System)** for hot storage and **Filecoin** for long-term archival.

## Core Concepts

### 1. The "Vending Machine" Model
The Kestrel platform acts as a vending machine for compute. Users pay for the time their agent is active. However, the *state* of the agent (its memory) must be portable. Decentralized storage allows the user to "eject" their agent's state into a neutral, globally accessible layer.

### 2. Content Addressing (CIDs)
By using IPFS, we shift from location-based addressing (e.g., `https://kestrel.ai/users/123/backup`) to content-based addressing (e.g., `QmHash...`). This means:
- **Immutability:** The data at a specific CID can never change.
- **Verifiability:** The user can mathematically prove that the data they retrieved is exactly what they stored.
- **Portability:** Any IPFS node in the world can serve the data.

### 3. The "Forever" Promise (Filecoin)
While IPFS ensures addressability, it does not guarantee persistence (nodes can garbage collect data). Filecoin provides the economic incentive layer.
- **Deals:** The agent (or user) can make storage deals on the Filecoin network to ensure their data is pinned for years or decades.
- **Self-Preservation:** A sovereign agent with its own wallet can autonomously renew its own storage deals, effectively paying rent for its own existence.

## Technical Implementation

The technical realization of this vision is detailed in **[SOVEREIGNTY_V2_TECHNICAL.md](SOVEREIGNTY_V2_TECHNICAL.md)**.

### Key Components:
- **Convergent Encryption:** Allows global deduplication of common data (like system prompts) while keeping user data private.
- **Merkle DAG:** Structures agent history as a directed acyclic graph, enabling efficient incremental backups.
- **Time-Based Sharding:** Seals history into immutable monthly blocks.

## Roadmap

- [x] **Phase 1: IPFS Export** (Implemented in Storage V2)
  - Agents can export their state to a local IPFS node.
  - Returns a Root CID to the user.

- [ ] **Phase 2: Filecoin Archival** (Planned Q1 2026)
  - Automated bridging from IPFS hot storage to Filecoin cold storage.
  - Integration with Filecoin aggregators (e.g., Lighthouse, Web3.Storage).

- [ ] **Phase 3: Autonomous Renewal** (Planned Q3 2026)
  - WalletAgent integration to monitor deal expiration.
  - Autonomous funding and renewal of storage contracts.
