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
    end

    subgraph benefit["Benefits"]
        B1["No node management"]
        B2["Automatic deal renewal"]
        B3["Single API for both"]
    end

    lighthouse --> benefit

    style lighthouse fill:#7d6608,stroke:#f4d03f,stroke-width:2px
```

**One API.** IPFS + Filecoin complexity hidden.

---

## Slide 3: Pricing

| Storage Type | Cost | Duration |
|--------------|------|----------|
| IPFS (hot) | $0.05/GB/month | Until unpinned |
| Filecoin (cold) | $0.00005/GB/month | Permanent (renewable) |
| Lighthouse combo | ~$0.05/GB/month | Hot + cold backup |

```mermaid
graph LR
    SIZE["1 GB"] --> COST1["$0.05/month hot"]
    SIZE --> COST2["$0.0006/year permanent"]

    DECADE["10 years permanent"] --> TOTAL["$0.006 total"]

    style COST2 fill:#145a32,stroke:#58d68d
    style TOTAL fill:#7d6608,stroke:#f4d03f
```

**Permanent storage for pennies.**

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

**Renewable.** Lighthouse auto-renews if funded.

---

## Slide 6: Multi-Currency Payment

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

## Slide 7: No Vendor Lock-in

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

## Slide 8: Economic Incentives

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
