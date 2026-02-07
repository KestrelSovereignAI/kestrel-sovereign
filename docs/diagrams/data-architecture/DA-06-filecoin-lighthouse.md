# DA-06: Filecoin & Lighthouse

Permanent storage for data sovereignty.

---

## Slide 1: Storage Tier Comparison

```mermaid
graph LR
    subgraph ipfs["IPFS (Hot)"]
        I1["Fast retrieval"]
        I2["Not guaranteed permanent"]
        I3["Free (if nodes available)"]
    end

    subgraph filecoin["Filecoin (Cold)"]
        F1["Slower retrieval"]
        F2["Guaranteed permanent"]
        F3["Paid, but very cheap"]
    end

    ipfs -->|"For long-term"| filecoin

    style ipfs fill:#0e4d45,stroke:#48c9b0
    style filecoin fill:#7d6608,stroke:#f4d03f
```

**IPFS for access. Filecoin for permanence.**

---

## Slide 2: Lighthouse - Managed IPFS + Filecoin

```mermaid
graph TD
    subgraph lighthouse["Lighthouse"]
        UPLOAD["Simple upload API"]
        IPFS["IPFS pinning"]
        FILECOIN["Automatic Filecoin deals"]
        RETRIEVAL["Fast retrieval gateway"]
        KAVACH["Kavach threshold encryption"]
        X402["x402 pay-per-use uploads"]
    end

    subgraph benefit["Benefits"]
        B1["No node management"]
        B2["Automatic deal renewal via endowment pool"]
        B3["Single API for both"]
        B4["NFT/token-gated access control"]
    end

    lighthouse --> benefit

    style lighthouse fill:#7d6608,stroke:#f4d03f,stroke-width:2px
```

**One API.** IPFS + Filecoin complexity hidden.

---

## Slide 3: Pricing (Updated Feb 2026)

| Storage Type | Cost | Duration | Notes |
|--------------|------|----------|-------|
| IPFS hot (Lighthouse) | $0.05/GB/month | Until unpinned | Fast retrieval |
| Raw Filecoin deal | ~$0.00005/GB | Per deal (~1yr) | Must manually renew |
| Lighthouse perpetual | ~$2-5/GB one-time | Forever | Endowment pool auto-renews |

**Lighthouse Lifetime Plans:**

| Plan | Price | Storage |
|------|-------|---------|
| Free | $0 | 5 GB |
| Beacon | $20 | 5 GB |
| Navigator | $100 | 25 GB |
| Harbor | $500 | 150 GB |

```mermaid
graph LR
    SIZE["1 GB"] --> COST1["$0.05/month hot"]
    SIZE --> COST2["~$4 one-time perpetual"]

    subgraph endowment["How Perpetual Works"]
        PAY["You pay ~$4/GB"] --> SPLIT["Split"]
        SPLIT --> DEAL["Part → current Filecoin deal"]
        SPLIT --> POOL["Part → endowment pool"]
        POOL --> RENEW["Pool auto-renews deals forever"]
        POOL --> GROW["Pool grows via FIL staking"]
    end

    style COST2 fill:#145a32,stroke:#58d68d
    style POOL fill:#7d6608,stroke:#f4d03f
```

**Pay once, stored forever.** The endowment pool funds perpetual renewal.

> **Note:** Raw Filecoin deal cost (~$0.00005/GB) is misleadingly cheap.
> Lighthouse charges ~$2-5/GB to fund the endowment pool that auto-renews
> deals in perpetuity. This is the real cost of "permanent" storage.

---

## Slide 4: Filecoin Deal Creation

```mermaid
sequenceDiagram
    participant A as Agent
    participant L as Lighthouse
    participant M as Filecoin Miners

    A->>L: Upload encrypted data
    L->>L: Pin to IPFS
    L-->>A: CID returned

    L->>M: Propose storage deal
    M->>M: Seal data (cryptographic proof)
    M-->>L: Deal accepted

    L->>L: Monitor deal health
    L->>M: Renew before expiry
```

---

## Slide 5: Deal Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Proposed: Upload data
    Proposed --> Sealing: Miner accepts
    Sealing --> Active: Proof generated
    Active --> Active: Periodic verification
    Active --> Expiring: Near end date
    Expiring --> Active: Renewed
    Expiring --> Expired: Not renewed
    Expired --> [*]
```

**Renewable.** Lighthouse auto-renews via endowment pool.

---

## Slide 6: Kavach Threshold Encryption

```mermaid
graph TD
    subgraph kavach["Kavach Encryption"]
        ENCRYPT["Agent encrypts data"]
        SHARD["Key split into N shards<br/>(BLS threshold cryptography)"]
        DIST["Shards distributed<br/>across nodes"]
    end

    subgraph decrypt["Decryption"]
        GATHER["Gather T-of-N shards"]
        RECON["Reconstruct key"]
        ACCESS["Decrypt data"]
    end

    subgraph gate["Access Control"]
        NFT["NFT ownership check"]
        TOKEN["Token balance check"]
        PASSKEY["Passkey verification"]
        ZKTLS["zkTLS attestation"]
    end

    kavach --> decrypt
    gate -->|"Must pass to get shards"| decrypt

    style kavach fill:#512e5f,stroke:#af7ac5
    style gate fill:#7d6608,stroke:#f4d03f
```

**No single point of failure.** Lighthouse never holds the full key.

Supported gating: ERC721/ERC1155 NFTs, ERC20 tokens, passkeys, zkTLS.
Chains: EVM, Solana, Cosmos, Coreum, Radix.

> **Kestrel implication:** Kavach replaces our current Fernet encryption
> (which stores keys locally in `cache_dir/key_{hash}.key`). With Kavach,
> cryostasis recovery doesn't depend on a local key file or env var -
> access is tied to the agent's wallet/DID on-chain.

---

## Slide 7: x402 Pay-Per-Use

```mermaid
sequenceDiagram
    participant A as Agent
    participant L as Lighthouse
    participant F as x402 Facilitator

    A->>L: HTTP request (upload file)
    L-->>A: 402 Payment Required + price
    A->>A: Sign payment (USDC on Base)
    A->>L: Retry with X-PAYMENT header
    L->>F: Verify & settle payment
    F-->>L: Payment confirmed
    L-->>A: 200 OK + CID
```

x402 is a [Coinbase-developed protocol](https://github.com/coinbase/x402) using
HTTP 402 for micropayments. Agents can pay per-upload without pre-purchasing plans.

> **Kestrel implication:** Enables autonomous agent storage payments.
> Agent wallet signs USDC on Base → file stored → no API key quotas needed.
> This is the path to Phase 3 (Autonomous Renewal) in our roadmap.

---

## Slide 8: Multi-Currency Payment

```mermaid
graph TD
    subgraph payment["Payment Options"]
        FIL["FIL<br/>Native Filecoin"]
        USDC["USDC<br/>Stablecoin"]
        USDT["USDT<br/>Stablecoin"]
        CARD["Credit Card<br/>Via Lighthouse"]
    end

    subgraph conversion["Lighthouse Handles"]
        CONV["Currency conversion"]
        DEAL["Deal payment"]
    end

    payment --> conversion --> DEAL

    style FIL fill:#7d6608,stroke:#f4d03f
    style USDC fill:#145a32,stroke:#58d68d
    style USDT fill:#145a32,stroke:#58d68d
```

**Pay how you want.** Lighthouse converts.

---

## Slide 9: No Vendor Lock-in

```mermaid
graph TD
    subgraph standard["Standard CID"]
        CID["QmXyz...abc"]
        UNIVERSAL["Works on ANY IPFS gateway"]
    end

    subgraph gateways["Retrieval Options"]
        G1["Lighthouse gateway"]
        G2["ipfs.io"]
        G3["Pinata"]
        G4["Your own node"]
    end

    CID --> gateways

    style CID fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style UNIVERSAL fill:#145a32,stroke:#58d68d
```

**The CID is universal.** Not locked to Lighthouse.

---

## Slide 10: Economic Incentives

```mermaid
graph TD
    subgraph miners["Filecoin Miners"]
        STORE["Store your data"]
        PROVE["Prove storage periodically"]
        EARN["Earn FIL rewards"]
    end

    subgraph you["You"]
        PAY["Pay small fee"]
        GUARANTEE["Get storage guarantee"]
        VERIFY["Can verify anytime"]
    end

    miners <-->|"Cryptoeconomic contract"| you

    style miners fill:#7d6608,stroke:#f4d03f
    style you fill:#145a32,stroke:#58d68d
```

**Miners are incentivized.** Math, not trust.

---

*Next: [DA-07-cryostasis.md](DA-07-cryostasis.md) - Agent dormancy when wallet runs low*
