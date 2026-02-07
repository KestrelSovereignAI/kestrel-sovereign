# Decentralized Storage Vision: IPFS & Filecoin Integration

**Last Updated:** February 6, 2026

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

### 3. The "Forever" Promise (Lighthouse Perpetual Storage)
While IPFS ensures addressability, it does not guarantee persistence (nodes can garbage collect data). Filecoin provides the economic incentive layer, and **Lighthouse** wraps it with a perpetual storage model:
- **Endowment Pool:** When you pay Lighthouse (~$2-5/GB one-time), part funds the current Filecoin deal and the rest enters a smart contract endowment pool that auto-renews deals forever. The pool grows via FIL staking/farming.
- **No Renewal Needed:** Unlike raw Filecoin deals (~$0.00005/GB but expiring), Lighthouse perpetual storage is truly pay-once-store-forever.
- **Self-Preservation:** For cryostasis, this means a dormant agent doesn't need to wake up to renew its own storage.

### 4. Kavach Threshold Encryption
Lighthouse's Kavach SDK uses **BLS threshold cryptography** to eliminate single-point-of-failure key management:
- Encryption key is split into N shards distributed across nodes
- T-of-N shards required to decrypt (Lighthouse never holds the full key)
- Access can be gated by NFT ownership, token balance, passkeys, or zkTLS
- Supported chains: EVM, Solana, Cosmos, Coreum, Radix

**Kestrel implication:** Kavach replaces our current Fernet encryption approach (local key files in `cache_dir/key_{hash}.key` protected by `KESTREL_DATA_KEY` env var). With Kavach, cryostasis recovery is tied to the agent's wallet/DID on-chain rather than a local secret that could be lost.

### 5. x402 Pay-Per-Use Protocol
[x402](https://github.com/coinbase/x402) is a Coinbase-developed protocol using HTTP 402 ("Payment Required") for micropayments:
- Agent requests a resource, server responds with 402 + price
- Agent signs a stablecoin payment (USDC on Base), retries with payment header
- Server verifies payment, serves the resource
- No accounts, sessions, or API key quotas needed

**Kestrel implication:** Enables fully autonomous agent storage payments without pre-purchased plans. The agent wallet signs USDC on Base per-upload.

## Technical Implementation

The technical realization of this vision is detailed in **[SOVEREIGNTY_V2_TECHNICAL.md](SOVEREIGNTY_V2_TECHNICAL.md)**.

### Key Components:
- **Convergent Encryption:** Allows global deduplication of common data (like system prompts) while keeping user data private.
- **Merkle DAG:** Structures agent history as a directed acyclic graph, enabling efficient incremental backups.
- **Time-Based Sharding:** Seals history into immutable monthly blocks.

## Pricing Reality (Feb 2026)

| Storage Type | Cost | Duration | Notes |
|--------------|------|----------|-------|
| IPFS hot (Lighthouse) | $0.05/GB/month | Until unpinned | Fast retrieval |
| Raw Filecoin deal | ~$0.00005/GB | ~1 year per deal | Must manually renew |
| Lighthouse perpetual | ~$2-5/GB one-time | Forever | Endowment pool auto-renews |

**Lighthouse Lifetime Plans:**
- Free: 5 GB
- Beacon: $20 (5 GB)
- Navigator: $100 (25 GB)
- Harbor: $500 (150 GB)

**Cryostasis cost for typical agent (10-100 MB): $0.04-$0.40 one-time perpetual.**

## Roadmap

- [x] **Phase 1: IPFS Export** (Implemented in Storage V2)
  - Agents can export their state to a local IPFS node.
  - Returns a Root CID to the user.

- [x] **Phase 2: Lighthouse Integration** (Implemented)
  - `LighthouseProvider` for CLOUD_HOT (IPFS pinning) and CLOUD_COLD (Filecoin).
  - `CryostasisCapable` interface with archive/restore methods.
  - `TieredStorageManager` routes storage by privacy mode.

- [ ] **Phase 3: Kavach Encryption Migration** (Planned Q2 2026)
  - Replace Fernet local encryption with Kavach threshold cryptography.
  - Tie decryption access to agent's DID-linked wallet (on-chain gating).
  - Eliminate single-point-of-failure key recovery for cryostasis.
  - **Note:** Python SDK (lighthouseweb3 v0.1.1) is unmaintained and only
    supports `upload()`. Migration requires REST API integration or
    direct use of Kavach encryption SDK.

- [ ] **Phase 4: x402 Autonomous Payments** (Planned Q3 2026)
  - Agent wallet pays per-upload via x402 protocol (USDC on Base).
  - No pre-purchased plans or API key quotas.
  - WalletAgent monitors storage balance and autonomously funds uploads.

- [ ] **Phase 5: NFT-Gated Cryostasis** (Exploratory)
  - Agent holds an NFT (e.g., Lighthouse Turby) as "life insurance."
  - NFT grants perpetual storage allocation - agent cannot die.
  - Kavach NFT-gated access ensures only the NFT holder can decrypt.
  - Digital inheritance: pass the NFT, pass the agent.
