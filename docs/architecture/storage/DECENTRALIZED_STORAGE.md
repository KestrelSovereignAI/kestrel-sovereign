---
type: Architecture Spec
title: 'Decentralized Storage Vision: IPFS & Filecoin Integration'
description: '**Last Updated:** March 11, 2026'
resource: /docs/architecture/storage/DECENTRALIZED_STORAGE.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-07-24T00:00:00Z'
status: aspirational
owner: architecture
canonical: false
generated: false
privacy: public
---

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

## Providers

| | Lighthouse | Filebase |
|---|---|---|
| Auth | Opaque API key | S3-compatible access key/secret |
| Gateway | Closed-source hosted | Hosted IPFS + S3 endpoint |
| Self-hostable | No | No |
| Free tier | 5 GB | 5 GB |
| Paid | ~$0.05/GB/month (hot) / ~$4/GB perpetual (cold) | S3-style usage pricing |
| Cold storage | Filecoin endowment pool (perpetual) | IPFS pinning |
| Cryostasis | Supported (with endowment pool) | Pinning only |
| DID alignment | None | None |

**Recommendation:** Lighthouse remains the default for both CLOUD_HOT pinning and CLOUD_COLD perpetual storage. Filebase is registered as a CLOUD_HOT alternative when its keys are present.

## Operator Health Checks

Use `kestrel storage health` to distinguish normal Lighthouse deal lag from
storage risk. The command checks the latest Lighthouse snapshot CID, its
Filecoin deal count, whether a no-deal upload is still inside the configured
grace window, and whether the GCS fallback bucket has a `latest.db` snapshot.

Default interpretation:

- `lighthouse: ok` means the latest matching snapshot has one or more Filecoin
  deals.
- `lighthouse: pending` means the latest snapshot has no deals yet but is less
  than 24 hours old.
- `lighthouse: warning` means the latest snapshot is older than the grace window
  and still has no Filecoin deals.
- `gcs: ok` means `GCS_BACKUP_BUCKET` is reachable and `latest.db` exists for
  the selected agent.

Storage tiers should be read as layered backups, not equivalent sovereignty:

- `sovereign-operated` is currently **none/decommissioned**. The historical
  self-hosted Kubo VM is not an active backup tier; `SOVEREIGN_IPFS_URL` must
  point at a reachable reprovisioned node before this tier is active.
- `delegated-decentralized` is Lighthouse: IPFS gateway access plus Filecoin
  deals once Lighthouse aggregation completes.
- `expedient-cloud` is GCS: fast restore on centralized cloud infrastructure,
  but not decentralized.
- `local-only` means data stays in the local cache/disk backup path.

Examples:

```bash
kestrel storage health --agent Emma
kestrel storage health --agent Emma --lighthouse-grace-hours 48
kestrel storage health --agent Emma --json
kestrel storage health --agent-id did:example:agent
```

The Lighthouse check reads each upload on the account it went to, and every
tier is read in the environment `kestrel start` gives the agent: the project
`.env` over an exported value. The agent's state snapshots are on the host's
`LIGHTHOUSE_API_KEY` account, so a shell exporting another key than the
`.env` does not move the check to an account the agent never uploads to. Its
sovereignty exports are on the account it publishes them to, whose key comes
from the agent's PayerPolicy, resolved the way boot resolves it
(`resolve_lighthouse_storage` in
`kestrel_sovereign/storage/providers/lighthouse_provider.py`), in that
environment with the agent's data directory as `KESTREL_DB_PATH` and its own
data key. `--agent` names a local agent in `multi_agent.toml`, and
`--agent-id` alone names the local agent with that DID: its database supplies
its DID and any key it holds itself (`!add-key`, a `SELF_WALLET` mint). When
no local agent has the DID there is no agent database, so only a host-level
key resolves, and the command says so. Both databases are opened read-only,
and nothing is minted. A credential that cannot be resolved (a `SELF_WALLET`
key the agent has not minted yet, say) is reported as the Lighthouse tier's
error while the other tiers are still checked.

Lighthouse upload names are agent scoped for restore safety and retention:
state snapshots use `kestrel_state__<agent_id>__<YYYYMMDD_HHMMSS>.car`, and
remote manifests use `kestrel_manifest__<agent_id>__<YYYYMMDD_HHMMSS>.json`.
Pre-cutover flat uploads such as `export.car` or `kestrel_prime.db` are not
renamed and are attributed during cleanup only through the local manifest or
legacy agent manifest (`manifest_<did>.json`); filename-only flat database
heuristics are intentionally not used for restore.

Sovereignty exports (`export_sovereignty`) are named for their agent too:
`kestrel_sovereignty__<agent_id>__<YYYYMMDD_HHMMSS>.bin`, tagged
`kestrel-sovereignty-<agent_id>`. The agent id is percent-encoded (a DID is
unchanged by it). An `ipfs`-tier export goes to the IPFS node at
`IPFS_API_URL` when it is reachable. When it is not, it goes to the agent's
Lighthouse provider, which exists only when remote-tier policy and
PayerPolicy permitted one at boot. With neither, the export stays local and
reports the downgrade. The name and tag carry the agent id and a time, never
content; the payload is the export's sealed envelope. The family is distinct
from `kestrel_state__`/`kestrel_manifest__` on purpose: cold-start restore
reads its database from those, and an export is an encrypted archive.

Every reader and pruner of an agent's sovereignty exports uses the credential
that publishes them: the key the agent's PayerPolicy resolves (an
agent-scoped key in its encrypted key storage, a `SELF_WALLET` key, or the
host key). Boot builds the agent's Lighthouse provider from that one
resolution, and nothing else reads a key of its own for them. The provider
asks the resolution for the key before each call and never falls back to the
key it last had: once none resolves (the agent's own key removed, with no
host key to stand in), it has no key, so an export stays local and says so,
and both health checks report no export credential. A resolution that fails
is raised, not papered over with the old key.

Only a key the agent does not hold falls back to the host's
`LIGHTHOUSE_API_KEY`. A key it holds but cannot read (a damaged row, an
active row with no ciphertext, or a `KESTREL_DATA_KEY` other than the one it
was stored under) is an error, never
a reason to use the host's account: every Lighthouse path asks
`KeyResolutionService.resolve_key(..., strict_storage=True)`, which raises
`KeyUnreadableError` instead of falling back. The export is then refused with
nothing stored, recorded or charged; its thinning prunes nothing; and both
health checks report the error. When boot meets it, boot builds no provider
and `!check-sovereignty-status` reports why. Other callers of `resolve_key`
keep falling back on any storage failure.

- After each export it publishes, the provider thins the agent's older
  exports on that account, among themselves, with identity retention. Only
  uploads whose name and tag attribute to the agent are considered; another
  agent's are never touched.
- State snapshots are not on that account. The snapshot target boot
  registers uploads with the host's `LIGHTHOUSE_API_KEY`, and the retention
  prune after each snapshot runs there too. That is the one account
  cold-start restore can read: it runs before the agent's database exists,
  so before any key the agent holds itself can be read. The target is
  registered when PayerPolicy permits Lighthouse storage and the host key is
  set, whichever key the agent's exports use, so a key the agent holds but
  cannot read stops its exports, not its snapshots. The prune leaves exports
  alone.
- `!check-sovereignty-status` and `kestrel storage health` report the agent's
  latest export on the export account, attributed to the agent, with its
  Filecoin deal status, whether or not this host still holds the export's
  receipt. `kestrel storage health` also reports the latest snapshot, from
  the host account, and lists an account once when both keys name it. Both
  read every page of an account's uploads, and count deals in either shape
  `deal_status` answers with (a list of deals, or the list wrapped as
  `{"data": [...]}`). With no state snapshot, the export is the upload the
  check reports; with both, the status is the worse of the two.
- `scripts/backup_cleanup.py` attributes exports by the same rule and never
  deletes an upload that is, or may be, an export: attributed exports, names
  and tags in the export family that attribute to no one agent, and legacy
  `content.bin` uploads. That holds under `--include-quarantined` and the
  `--delete-older-than-days` age purge.
- An export the provider took restores from it. `import_sovereignty` reads the
  host cache, then the IPFS node, then the agent's Lighthouse provider, which
  fetches the CID from the gateway under the same byte ceiling, reading only
  an unencoded body. Its bytes are opened and checked exactly as the node's
  are: the envelope's owner and key, and the content hash. A content hash is
  never sent to Lighthouse.
- When Lighthouse cannot take an export either, it is kept on this host and
  reported as not durable off-host, as when no provider exists.
- Exports made before the naming restore exactly as before; the name does not
  change an IPFS CID.

## Pricing Reality (Mar 2026)

| Storage Type | Cost | Duration | Notes |
|--------------|------|----------|-------|
| Lighthouse IPFS hot | $0.05/GB/month | Until unpinned | Dedicated gateway |
| Raw Filecoin deal | ~$0.00005/GB | ~1 year per deal | Must manually renew |
| Lighthouse perpetual | ~$2-5/GB one-time | Forever | Endowment pool auto-renews |

**Lighthouse Lifetime Plans:**
- Free: 5 GB
- Beacon: $20 (5 GB)
- Navigator: $100 (25 GB)
- Harbor: $500 (150 GB)

**Cryostasis cost for typical agent (10-100 MB): $0.04-$0.40 one-time perpetual (Lighthouse).**

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
