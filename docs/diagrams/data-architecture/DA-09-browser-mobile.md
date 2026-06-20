---
type: Diagram
title: 'DA-09: Browser & Mobile Storage'
description: Client-side storage architecture across browsers, mobile apps, and server
  compute.
resource: /docs/diagrams/data-architecture/DA-09-browser-mobile.md
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

# DA-09: Browser & Mobile Storage

Client-side storage architecture across browsers, mobile apps, and server compute.

---

## Slide 1: Client Storage Architecture

```mermaid
graph TD
    subgraph client["CLIENT LAYER"]
        subgraph browser["Browser"]
            IDB["IndexedDB<br/>SovereignStorage.js"]
        end
        subgraph mobile["Native Mobile"]
            SQLITE_M["SQLite<br/>Full Kestrel"]
        end
    end

    subgraph server["kestrel/KESTREL SERVER"]
        SQLITE_S["SQLite<br/>(standalone)"]
        POSTGRES["PostgreSQL<br/>(multi-tenant)"]
    end

    subgraph permanent["IPFS/FILECOIN"]
        IPFS["Sovereignty Export"]
    end

    IDB <-->|"sync"| server
    SQLITE_M <-->|"sync"| server
    server <-->|"export"| IPFS

    style IDB fill:#1a5276,stroke:#85c1e9
    style SQLITE_M fill:#145a32,stroke:#58d68d
    style server fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style IPFS fill:#512e5f,stroke:#af7ac5
```

**Three client paths.** All sync to server. All export to IPFS.

---

## Slide 2: Browser Layer - IndexedDB

```mermaid
graph TD
    subgraph indexeddb["IndexedDB (Browser-Native)"]
        STORES["Object Stores"]
        TRIAL["trialSessions"]
        MSGS["messages"]
        DEVICE["deviceIdentity"]
    end

    subgraph crypto["WebCrypto"]
        AES["AES-256-GCM"]
        KEYS["Derived keys"]
    end

    subgraph impl["SovereignStorage.js"]
        SAVE["saveTrialSession()"]
        GET["getTrialSession()"]
        SYNC["syncToCloud()"]
    end

    STORES --> TRIAL
    STORES --> MSGS
    STORES --> DEVICE
    crypto --> impl
    indexeddb --> impl

    style indexeddb fill:#1a5276,stroke:#85c1e9
    style crypto fill:#512e5f,stroke:#af7ac5
    style impl fill:#145a32,stroke:#58d68d,stroke-width:2px
```

**Browser storage.** Encrypted, offline-capable, trial-ready.

---

## Slide 3: Native Mobile Layer - SQLite

```mermaid
graph TD
    subgraph platforms["Mobile Platforms"]
        RN["React Native<br/>expo-sqlite"]
        FLUTTER["Flutter<br/>sqflite"]
        IOS["iOS Native<br/>SQLite.swift"]
        ANDROID["Android Native<br/>Room/SQLite"]
    end

    subgraph full["Full Kestrel Stack"]
        RAG["RAG embeddings"]
        GRAPH["Knowledge graph"]
        FTS["Full-text search"]
        WALLET["Wallet state"]
    end

    platforms --> full

    style platforms fill:#145a32,stroke:#58d68d
    style full fill:#7d6608,stroke:#f4d03f
```

**Native mobile runs SQLite directly.** Full agent capabilities on device.

---

## Slide 4: Server Layer - Kestrel/Kestrel

```mermaid
graph TD
    subgraph standalone["Kestrel Standalone"]
        SQLITE["SQLite<br/>Single user"]
        LOCAL["Local agent data"]
    end

    subgraph multitenant["Kestrel Multi-Tenant"]
        POSTGRES["PostgreSQL<br/>Row-level security"]
        REDIS["Redis<br/>Sessions & cache"]
    end

    subgraph compute["Compute Services"]
        RAG["RAG search"]
        GRAPH["Knowledge graph"]
        CONST["Constitutional RAG"]
        LLM["LLM orchestration"]
    end

    standalone --> compute
    multitenant --> compute

    style standalone fill:#1a5276,stroke:#85c1e9
    style multitenant fill:#512e5f,stroke:#af7ac5
    style compute fill:#7d6608,stroke:#f4d03f,stroke-width:2px
```

**Server runs the heavy compute.** Clients sync and cache.

---

## Slide 5: Data Flow

```mermaid
sequenceDiagram
    participant B as Browser (IndexedDB)
    participant S as Server (SQLite/PostgreSQL)
    participant I as IPFS/Filecoin

    B->>B: User sends message
    B->>B: Store in IndexedDB
    B->>S: Sync to server
    S->>S: Run RAG, update graph
    S-->>B: Response + context

    alt Sovereignty Export
        S->>I: Encrypted snapshot
        I-->>S: CID
        S-->>B: Store CID
    end

    Note over B,I: Works offline - syncs when connected
```

---

## Slide 6: IndexedDB Object Stores

```mermaid
graph LR
    subgraph stores["SovereignStorage Stores"]
        TRIAL["trialSessions<br/>• sessionId (key)<br/>• messages[]<br/>• companion<br/>• createdAt"]
        MSGS["messages<br/>• id (key)<br/>• sessionId<br/>• role<br/>• content<br/>• timestamp"]
        DEV["deviceIdentity<br/>• deviceId (key)<br/>• publicKey<br/>• createdAt"]
    end

    subgraph encrypted["All Encrypted"]
        AES["AES-256-GCM<br/>via WebCrypto"]
    end

    stores --> encrypted

    style TRIAL fill:#1a5276,stroke:#85c1e9
    style MSGS fill:#512e5f,stroke:#af7ac5
    style DEV fill:#145a32,stroke:#58d68d
    style encrypted fill:#7d6608,stroke:#f4d03f
```

---

## Slide 7: Server Compute Capabilities

```mermaid
graph TD
    subgraph server["Server-Side Only"]
        RAG["🔍 RAG Search<br/>Vector embeddings<br/>Semantic similarity"]
        GRAPH["🕸️ Knowledge Graph<br/>Node/edge traversal<br/>Relationship queries"]
        FTS["📝 Full-Text Search<br/>FTS5 indexing<br/>Fast text lookup"]
        CONST["⚖️ Constitutional RAG<br/>US Constitution<br/>Kestrel Constitution"]
        WALLET["💰 Wallet<br/>Transaction history<br/>Balance tracking"]
    end

    style RAG fill:#1a5276,stroke:#85c1e9
    style GRAPH fill:#512e5f,stroke:#af7ac5
    style FTS fill:#145a32,stroke:#58d68d
    style CONST fill:#7d6608,stroke:#f4d03f
    style WALLET fill:#7d3c00,stroke:#f5b041
```

**Complex queries require SQL.** Browser delegates to server.

---

## Slide 8: Offline & Sync

```mermaid
graph TD
    subgraph offline["Offline Mode"]
        CACHE["Recent messages in IndexedDB"]
        QUEUE["Pending sync queue"]
        LOCAL["Local companion state"]
    end

    subgraph online["When Connected"]
        PUSH["Push queued messages"]
        PULL["Pull server updates"]
        MERGE["Merge state"]
    end

    subgraph guarantee["Guarantee"]
        NEVER["Never lose messages"]
        ALWAYS["Always responsive"]
    end

    offline --> online --> guarantee

    style offline fill:#1a5276,stroke:#85c1e9
    style online fill:#145a32,stroke:#58d68d
    style guarantee fill:#7d6608,stroke:#f4d03f
```

**Works without network.** Syncs when available.

---

## Slide 9: Trial Mode Flow

```mermaid
sequenceDiagram
    participant U as User
    participant B as Browser (IndexedDB)
    participant S as Server

    Note over U,B: No Account Required

    U->>B: Start chatting
    B->>B: Create trial session
    B->>B: Store in IndexedDB
    B->>S: Anonymous API calls
    S-->>B: Responses (no persistence)

    Note over U,S: 25 messages later...

    U->>S: Create account
    B->>S: Migrate trial data
    S->>S: Store in PostgreSQL
    S-->>B: Account linked

    Note over B,S: Data preserved, now synced
```

---

## Slide 10: Implementation Status

| Component | Platform | Status |
|-----------|----------|--------|
| **IndexedDB wrapper** | Browser | ✅ `SovereignStorage.js` |
| **AES-256-GCM encryption** | Browser | ✅ WebCrypto API |
| **Sync endpoint** | Server | ✅ `/api/sovereign/sync` |
| **Trial mode** | Browser | ✅ Works without account |
| **SQLite** | React Native | 📋 Future |
| **SQLite** | Flutter | 📋 Future |
| **SQLite** | iOS Native | 📋 Future |
| **SQLite** | Android Native | 📋 Future |

---

*Return to [README.md](README.md) for series index*
