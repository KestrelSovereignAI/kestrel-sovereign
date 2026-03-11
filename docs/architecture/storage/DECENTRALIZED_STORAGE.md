# Decentralized Storage Vision: IPFS & Filecoin Integration

**Last Updated:** March 11, 2026

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

## Providers: Lighthouse vs Storacha

Two IPFS/Filecoin providers are now implemented. Use both, or choose one:

| | Storacha (web3.storage) | Lighthouse |
|---|---|---|
| Auth | UCAN/DID — agent's Ed25519 key signs every request | Opaque API key |
| Gateway | Open source (`local.storage`, `w3s.link`) | Closed-source hosted |
| Self-hostable | Yes (`storacha/local.storage`) | No |
| Python client | Custom w3up bridge client (in-tree) | Custom REST client (in-tree) |
| Free tier | 5 GB | 5 GB |
| Paid | ~$0.05/GB/month | ~$0.05/GB/month (hot) / ~$4/GB perpetual (cold) |
| Cold storage | IPFS + Filecoin w3up shards | Filecoin endowment pool (perpetual) |
| Cryostasis | Supported (store + retrieve) | Supported (with endowment pool) |
| DID alignment | Native (space/agent/proof are DIDs) | None |
| Confidence | **High** — preferred path | Declining (closed gateway, no DID) |

**Recommendation:** Use Storacha for new deployments. Keep Lighthouse configured as a CLOUD_COLD fallback until the Filecoin cold-storage story matures for Storacha.

### Storacha Setup (one-time per deployment)

```bash
npm install -g @web3-storage/w3cli
w3 key create                                        # → STORACHA_AGENT_KEY
w3 space create kestrel                              # → STORACHA_SPACE_DID
w3 delegation create --can '*' <agent-did> | base64  # → STORACHA_PROOF
```

The agent's Ed25519 DID key can serve double duty as the UCAN signing key —
no separate credentials to manage.

### Key Files (Storacha)

| File | Purpose |
|---|---|
| `storage/providers/storacha_ucan.py` | UCAN v1 invocation builder, CIDv1, CARv1 |
| `storage/providers/storacha_rest.py` | Async HTTP client (two-phase upload, gateway retrieval) |
| `storage/providers/storacha_provider.py` | `StorageProvider` + `CryostasisCapable` implementation |
| `storage/sync/targets.py` → `StorachaTarget` | DB snapshot backup/restore for ephemeral environments |

## Pricing Reality (Mar 2026)

| Storage Type | Cost | Duration | Notes |
|--------------|------|----------|-------|
| Storacha IPFS hot | $0/month (≤5 GB) then $0.05/GB/month | Until unpinned | w3s.link gateway |
| Lighthouse IPFS hot | $0.05/GB/month | Until unpinned | Dedicated gateway |
| Raw Filecoin deal | ~$0.00005/GB | ~1 year per deal | Must manually renew |
| Lighthouse perpetual | ~$2-5/GB one-time | Forever | Endowment pool auto-renews |

**Lighthouse Lifetime Plans:**
- Free: 5 GB
- Beacon: $20 (5 GB)
- Navigator: $100 (25 GB)
- Harbor: $500 (150 GB)

**Cryostasis cost for typical agent (10-100 MB): $0.04-$0.40 one-time perpetual (Lighthouse) or ~$0 within free tier (Storacha).**

## Roadmap

- [x] **Phase 1: IPFS Export** (Implemented in Storage V2)
  - Agents can export their state to a local IPFS node.
  - Returns a Root CID to the user.

- [x] **Phase 2: Lighthouse Integration** (Implemented)
  - `LighthouseProvider` for CLOUD_HOT (IPFS pinning) and CLOUD_COLD (Filecoin).
  - `CryostasisCapable` interface with archive/restore methods.
  - `TieredStorageManager` routes storage by privacy mode.

- [x] **Phase 2b: Storacha Integration** (Implemented Mar 2026)
  - `StorachaProvider` for CLOUD_HOT (IPFS via w3up, UCAN/DID auth).
  - `StorachaTarget` for DB snapshot backup/cold-start restore.
  - `TieredStorageManager` prefers Storacha over Lighthouse for CLOUD_HOT.
  - Reflection self-model manager accepts any `StorageProvider` (not just Lighthouse).
  - `CryostasisCapable` implemented; cold Filecoin deals via w3up (Phase 3 enhancement).

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
