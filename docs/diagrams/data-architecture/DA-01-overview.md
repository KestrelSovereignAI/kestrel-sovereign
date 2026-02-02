# DA-01: Data Architecture Overview

The three-layer architecture that enables data sovereignty.

---

## Slide 1: The Data Sovereignty Problem

```mermaid
graph TD
    subgraph current["Current AI Systems"]
        USER1["Your conversations"]
        CLOUD1["Vendor's servers"]
        LOCK["🔒 Vendor lock-in"]
    end

    subgraph problems["Problems"]
        P1["❌ Data lives on vendor servers"]
        P2["❌ Vendor can delete your account"]
        P3["❌ No export to competitors"]
        P4["❌ Platform dies = data dies"]
    end

    USER1 --> CLOUD1 --> LOCK
    LOCK --> problems

    style current fill:#641e16,stroke:#ec7063
    style problems fill:#7d3c00,stroke:#f5b041
```

**You don't own your AI.** The vendor does.

---

## Slide 2: The Kestrel Solution

```mermaid
graph TD
    subgraph client["Client Layer"]
        BROWSER["🌐 Browser (IndexedDB)<br/>Cache, trial, offline"]
        MOBILE["📱 Native Mobile (SQLite)<br/>Full agent on device"]
    end

    subgraph server["Server Layer"]
        SERVER["💾 Kestrel/Kestrel Server<br/>SQLite or PostgreSQL"]
    end

    subgraph permanent["Permanent Layer"]
        DECEN["🔗 IPFS/Filecoin<br/>Vendor-independent"]
    end

    BROWSER <--> SERVER
    MOBILE <--> SERVER
    SERVER <--> DECEN

    style BROWSER fill:#1a5276,stroke:#85c1e9
    style MOBILE fill:#145a32,stroke:#58d68d
    style SERVER fill:#512e5f,stroke:#af7ac5,stroke-width:2px
    style DECEN fill:#7d6608,stroke:#f4d03f
```

**Your data, your device, your rules.**

---

## Slide 3: Data Flow - Write Path

```mermaid
sequenceDiagram
    participant U as User
    participant B as Browser (IndexedDB)
    participant S as Server (SQLite/PostgreSQL)
    participant I as IPFS/Filecoin

    U->>B: Send message
    B->>B: Store in IndexedDB
    B->>S: Sync to server
    S->>S: RAG, graph, wallet
    S-->>B: Response

    alt Sovereignty export
        S->>I: Encrypted snapshot
        I-->>S: CID returned
    end
```

**Browser stores first.** Server computes. IPFS persists.

---

## Slide 4: Data Flow - Read Path

```mermaid
graph TD
    subgraph read["Read Path Priority"]
        R1["1️⃣ Check IndexedDB cache"]
        R2["2️⃣ Fetch from server"]
        R3["3️⃣ Restore from IPFS (if needed)"]
    end

    subgraph offline["Offline Capable"]
        O1["IndexedDB works offline"]
        O2["Recent messages available"]
        O3["Sync when reconnected"]
    end

    R1 --> R2 --> R3
    read --> offline

    style R1 fill:#1a5276,stroke:#85c1e9,stroke-width:2px
    style offline fill:#145a32,stroke:#58d68d
```

**Offline first.** Always responsive.

---

## Slide 5: What Lives Where

```mermaid
graph TD
    subgraph browser["Browser (IndexedDB)"]
        B1["Message cache"]
        B2["Trial sessions"]
        B3["Device identity"]
    end

    subgraph server["Server (SQLite/PostgreSQL)"]
        S1["Full conversations"]
        S2["Knowledge graph"]
        S3["RAG embeddings"]
        S4["Wallet state"]
        S5["User accounts"]
    end

    subgraph decen["IPFS/Filecoin"]
        D1["Sovereignty snapshots"]
        D2["Cryostasis archives"]
        D3["Constitution hash"]
    end

    style browser fill:#1a5276,stroke:#85c1e9
    style server fill:#512e5f,stroke:#af7ac5
    style decen fill:#7d6608,stroke:#f4d03f
```

---

## Slide 6: Privacy Mode → Storage Tier Mapping

| Privacy Mode | Local | Cloud | IPFS |
|--------------|-------|-------|------|
| 👻 EPHEMERAL | ❌ | ❌ | ❌ |
| 🏝️ ISOLATED | ⏳ temp | ❌ | ❌ |
| 🎭 ANONYMOUS | ✅ scrubbed | ✅ scrubbed | ✅ |
| 🔄 NORMAL | ✅ | ✅ | ✅ |
| 📖 PUBLIC | ✅ | ✅ | ✅ shareable |

```mermaid
graph LR
    EPH["👻 EPHEMERAL"] --> NONE["Store nothing"]
    ISO["🏝️ ISOLATED"] --> TEMP["Temp buffer only"]
    ANON["🎭 ANONYMOUS"] --> SCRUB["Scrub PII first"]
    NORM["🔄 NORMAL"] --> FULL["Full storage"]
    PUB["📖 PUBLIC"] --> SHARE["Shareable"]

    style NONE fill:#641e16,stroke:#ec7063
    style TEMP fill:#7d6608,stroke:#f4d03f
    style SCRUB fill:#512e5f,stroke:#af7ac5
    style FULL fill:#145a32,stroke:#58d68d
    style SHARE fill:#1a5276,stroke:#85c1e9
```

---

## Slide 7: Automatic Sync Triggers

```mermaid
graph TD
    subgraph triggers["Sync Triggers"]
        T1["⏰ Time interval<br/>(every 5 minutes)"]
        T2["📊 Message count<br/>(every 100 messages)"]
        T3["🔄 Manual<br/>!sync command"]
        T4["🚪 Session end<br/>(user logs out)"]
    end

    subgraph sync["Sync Process"]
        CHECK["Check last sync timestamp"]
        DIFF["Compute delta"]
        UPLOAD["Upload to cloud"]
        CONFIRM["Update sync marker"]
    end

    triggers --> CHECK --> DIFF --> UPLOAD --> CONFIRM

    style T1 fill:#1a5276,stroke:#85c1e9
    style T2 fill:#7d6608,stroke:#f4d03f
```

**Automatic, incremental, efficient.**

---

## Slide 8: Key Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| 200 messages | 18.5 KB | Tiny storage footprint |
| Export time | 0.005s | Near-instant |
| IPFS cost | $0 | Free hot storage |
| Filecoin cost | $0.00005/GB | Decades for pennies |
| Sync interval | 5 min | Near real-time |
| Offline capability | ∞ | Works forever offline |

```mermaid
graph LR
    SIZE["18.5 KB<br/>200 messages"] --> CHEAP["Essentially free"]
    SPEED["0.005s<br/>Export time"] --> FAST["Instant backup"]
    COST["$0.00005/GB<br/>Filecoin"] --> FOREVER["Decades for pennies"]

    style CHEAP fill:#145a32,stroke:#58d68d
    style FAST fill:#1a5276,stroke:#85c1e9
    style FOREVER fill:#7d6608,stroke:#f4d03f
```

**Efficient by design.** Sovereignty costs nothing.

---

*Next: [DA-02-database-abstraction.md](DA-02-database-abstraction.md) - SQLite ↔ PostgreSQL abstraction*
