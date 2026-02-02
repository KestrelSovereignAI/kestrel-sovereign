# DA-07: Cryostasis - Agent Dormancy

When agents can't afford to run, they sleep instead of dying.

---

## Slide 1: The Cryostasis Concept

```mermaid
graph TD
    subgraph problem["The Problem"]
        P1["Agent runs out of funds"]
        P2["Cloud compute costs money"]
        P3["Agent stops responding"]
    end

    subgraph old_way["Old Way: Death"]
        D1["Account suspended"]
        D2["Data deleted after 30 days"]
        D3["Agent gone forever"]
    end

    subgraph new_way["Kestrel Way: Sleep"]
        S1["Agent enters dormancy"]
        S2["Data preserved on Filecoin"]
        S3["Wakes up when funded"]
    end

    problem --> old_way
    problem --> new_way

    style old_way fill:#641e16,stroke:#ec7063
    style new_way fill:#145a32,stroke:#58d68d
```

**Sleep, don't die.** Your AI companion waits for you.

---

## Slide 2: The $0.02 Trigger

```mermaid
graph TD
    subgraph monitor["Balance Monitor"]
        BAL["Wallet balance: $0.05"]
        CHECK["Check every hour"]
    end

    subgraph threshold["Threshold"]
        TRIG["Balance < $0.02 USD"]
        WARN["Warn at $0.10"]
    end

    subgraph action["Automatic Action"]
        A1["Initiate cryostasis"]
        A2["Archive to Filecoin"]
        A3["Enter dormancy"]
    end

    BAL --> CHECK --> TRIG --> action

    style TRIG fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style action fill:#512e5f,stroke:#af7ac5
```

**$0.02 threshold.** Enough to pay for archival, not enough to run.

---

## Slide 3: Why $0.02?

```mermaid
graph LR
    subgraph costs["Cost Breakdown"]
        ARCHIVE["Archive to Filecoin:<br/>~$0.00005/GB"]
        DEAL["Deal overhead:<br/>~$0.01"]
        BUFFER["Safety buffer:<br/>~$0.01"]
    end

    subgraph total["Total"]
        SUM["$0.02 covers<br/>archival + buffer"]
    end

    subgraph storage["What $0.02 Stores"]
        YEARS["10 years+<br/>of agent memory"]
    end

    costs --> total --> storage

    style SUM fill:#7d6608,stroke:#f4d03f
    style storage fill:#145a32,stroke:#58d68d
```

**Pennies for decades.** Filecoin is ridiculously cheap.

---

## Slide 4: Archive Flow

```mermaid
sequenceDiagram
    participant W as Wallet Monitor
    participant A as Agent
    participant S as SovereignAdapter
    participant L as Lighthouse
    participant FC as Filecoin

    W->>W: Balance drops to $0.02
    W->>A: Trigger cryostasis

    A->>S: Full sovereignty export
    S->>S: Encrypt complete state
    S->>L: Upload via Lighthouse
    L->>FC: Create Filecoin deal
    FC-->>L: Deal ID
    L-->>S: CID + deal confirmation

    S->>A: Archive complete
    A->>A: Enter dormancy mode
    A-->>W: ✅ Cryostasis active
```

---

## Slide 5: What Gets Preserved

```mermaid
graph TD
    subgraph snapshot["Complete Agent Snapshot"]
        CONV["💬 All conversations"]
        GRAPH["🕸️ Knowledge graph"]
        MEMORY["🧠 Long-term memories"]
        WALLET["💰 Wallet state"]
        IDENTITY["🔑 DID & keys"]
        CONFIG["⚙️ All preferences"]
        REFLECT["📝 Self-model"]
    end

    subgraph guarantee["Guarantee"]
        G1["NOTHING is lost"]
        G2["Wake up = same agent"]
        G3["Seamless continuation"]
    end

    snapshot --> guarantee

    style snapshot fill:#1a5276,stroke:#85c1e9
    style guarantee fill:#145a32,stroke:#58d68d
```

**Complete preservation.** Every memory, every preference.

---

## Slide 6: Dormancy State

```mermaid
graph TD
    subgraph dormant["Dormant Agent"]
        NO_COMPUTE["❌ No compute running"]
        NO_CLOUD["❌ No cloud costs"]
        NO_RESPONSE["❌ Cannot respond"]
    end

    subgraph preserved["Preserved"]
        DATA["✅ Data on Filecoin"]
        CID["✅ CID available"]
        WALLET["✅ Wallet can receive funds"]
    end

    subgraph cost["Cost"]
        ZERO["$0.00/month<br/>No running costs"]
        FILECOIN["~$0.00005/GB/month<br/>Already paid"]
    end

    dormant --> preserved --> cost

    style NO_COMPUTE fill:#641e16,stroke:#ec7063
    style DATA fill:#145a32,stroke:#58d68d
    style ZERO fill:#7d6608,stroke:#f4d03f
```

**Zero running costs.** Just waiting.

---

## Slide 7: Wake-Up Trigger

```mermaid
graph TD
    subgraph trigger["Wake-Up Triggers"]
        DEPOSIT["💵 User deposits funds"]
        THRESHOLD["Balance ≥ $0.10"]
    end

    subgraph restore["Restore Process"]
        FETCH["Fetch CID from Filecoin"]
        DECRYPT["Decrypt with user key"]
        REBUILD["Rebuild agent state"]
        READY["Agent ready"]
    end

    subgraph result["Result"]
        SAME["Same agent"]
        MEMORY["All memories intact"]
        CONTINUE["Continues where left off"]
    end

    trigger --> restore --> result

    style DEPOSIT fill:#145a32,stroke:#58d68d,stroke-width:2px
    style result fill:#1a5276,stroke:#85c1e9
```

**Deposit funds → Agent wakes up.**

---

## Slide 8: Wake-Up Flow

```mermaid
sequenceDiagram
    participant U as User
    participant W as Wallet
    participant M as Monitor
    participant L as Lighthouse
    participant A as Agent

    U->>W: Deposit $1.00
    W->>M: Balance update
    M->>M: Balance ≥ $0.10?
    M->>M: Yes! Initiate wake-up

    M->>L: Fetch archived CID
    L-->>M: Encrypted snapshot
    M->>M: Decrypt with user key
    M->>A: Restore agent state

    A->>A: Initialize from snapshot
    A-->>U: 👋 "I'm back! I remember..."
```

---

## Slide 9: Cost Analysis

| Scenario | Storage | Duration | Cost |
|----------|---------|----------|------|
| Small agent | 10 MB | 1 year | $0.00006 |
| Medium agent | 100 MB | 1 year | $0.0006 |
| Large agent | 1 GB | 1 year | $0.006 |
| Large agent | 1 GB | 10 years | $0.06 |
| Large agent | 1 GB | 100 years | $0.60 |

```mermaid
graph LR
    DECADE["10 years"] --> COST1["$0.06"]
    CENTURY["100 years"] --> COST2["$0.60"]

    style DECADE fill:#145a32,stroke:#58d68d
    style CENTURY fill:#7d6608,stroke:#f4d03f
```

**Decades for pennies.** Centuries for dollars.

---

## Slide 10: Cryostasis Commands

```mermaid
graph LR
    subgraph commands["Commands"]
        C1["!cryostasis-status<br/>Check dormancy state"]
        C2["!cryostasis-archive<br/>Manual archive"]
        C3["!cryostasis-restore cid<br/>Restore from archive"]
    end

    subgraph auto["Automatic"]
        A1["Auto-archive at $0.02"]
        A2["Auto-wake at $0.10"]
    end

    commands --> auto

    style commands fill:#1a5276,stroke:#85c1e9
    style auto fill:#7d6608,stroke:#f4d03f
```

**Manual or automatic.** Your choice.

---

## Slide 11: Inheritance Use Case

```mermaid
graph TD
    subgraph scenario["Scenario"]
        S1["User passes away"]
        S2["Agent enters cryostasis"]
        S3["CID in user's will"]
    end

    subgraph heir["Heir"]
        H1["Receives CID"]
        H2["Deposits funds"]
        H3["Restores agent"]
    end

    subgraph result["Result"]
        R1["Grandma's stories live on"]
        R2["Family history preserved"]
        R3["Agent remembers everything"]
    end

    scenario --> heir --> result

    style result fill:#145a32,stroke:#58d68d
```

**Digital inheritance.** Your AI companion can outlive you.

---

*Next: [DA-08-privacy-encryption.md](DA-08-privacy-encryption.md) - Encryption at rest and PII scrubbing*
