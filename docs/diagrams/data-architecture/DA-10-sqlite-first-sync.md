---
type: Diagram
title: 'DA-10: SQLite-First Sync Architecture'
description: 'The new architecture: SQLite as primary, sync to cloud when needed.'
resource: /docs/diagrams/data-architecture/DA-10-sqlite-first-sync.md
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

# DA-10: SQLite-First Sync Architecture

The new architecture: SQLite as primary, sync to cloud when needed.

---

## Slide 1: The Vision - Two-File Sovereign Agent

```mermaid
graph TD
    subgraph agent["Kestrel Sovereign Agent"]
        DB["agent.db<br/>(SQLite)"]
        LLM["emma.llamafile<br/>(LLM runtime)"]
    end

    subgraph data["What's in agent.db"]
        CONV["Conversations"]
        MEM["Memories"]
        TASK["Tasks"]
        OBS["Observability"]
    end

    DB --> data

    style agent fill:#145a32,stroke:#58d68d,stroke-width:3px
    style DB fill:#1a5276,stroke:#85c1e9
    style LLM fill:#512e5f,stroke:#af7ac5
```

**True sovereignty.** Your data is literally a file you own.

---

## Slide 2: Why SQLite-First?

```mermaid
graph LR
    subgraph benefits["SQLite Benefits"]
        B1["Zero config"]
        B2["Single file"]
        B3["No daemon"]
        B4["Offline works"]
        B5["Portable"]
    end

    subgraph devices["Runs Everywhere"]
        D1["Desktop"]
        D2["Phone"]
        D3["IoT"]
        D4["Browser"]
    end

    benefits --> devices

    style benefits fill:#145a32,stroke:#58d68d
```

| Feature | SQLite | PostgreSQL |
|---------|--------|------------|
| Setup | None | Install daemon |
| File | Single `.db` | Directory |
| Network | Not needed | Required |
| Backup | Copy file | pg_dump |
| Portability | Full | Limited |

---

## Slide 3: Sync Architecture

```mermaid
graph TD
    subgraph local["Local (Primary)"]
        SQLITE["SQLite<br/>agent.db"]
        WAL["WAL<br/>agent.db-wal"]
    end

    subgraph sync["Sync Service"]
        LISTENER["WAL Listener"]
        SERVICE["Sync Service"]
    end

    subgraph targets["Sync Targets"]
        S3["S3 / R2"]
        LH["Lighthouse<br/>(Filecoin)"]
        PG["PostgreSQL<br/>(Aggregation)"]
    end

    WAL --> LISTENER --> SERVICE
    SERVICE --> S3
    SERVICE --> LH
    SERVICE --> PG

    style local fill:#1a5276,stroke:#85c1e9
    style sync fill:#7d6608,stroke:#f4d03f
    style targets fill:#512e5f,stroke:#af7ac5
```

**SQLite writes locally.** Sync replicates to cloud.

---

## Slide 4: WAL Listener

```mermaid
sequenceDiagram
    participant App as Application
    participant SQLite
    participant WAL as WAL File
    participant Listener as WAL Listener
    participant Sync as Sync Service
    participant Cloud as Cloud Target

    App->>SQLite: INSERT/UPDATE
    SQLite->>WAL: Write frames
    SQLite-->>App: OK

    Note over Listener: Polls WAL for changes
    Listener->>WAL: Read new frames
    Listener->>Sync: Queue changes
    Sync->>Cloud: Upload batch
```

**Non-blocking.** Application doesn't wait for sync.

---

## Slide 5: Usage Example

```python
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.a2a.stores import TaskStore
from kestrel_sovereign.storage.sync import SyncService, S3Target

# SQLite is primary - always available
backend = SQLiteBackend("/path/to/agent.db")
await backend.connect()
task_store = TaskStore(backend)

# Sync is optional - for cloud backup
sync = SyncService("/path/to/agent.db")
sync.add_target(S3Target(bucket="my-backup"))
await sync.start()  # Non-blocking, runs in background

# Use task store normally
await task_store.save(task)  # Writes to SQLite
# Sync service automatically replicates to S3
```

**Simple API.** Sync is transparent to application.

---

## Slide 6: Sync Targets

| Target | Use Case | Features |
|--------|----------|----------|
| **S3** | Cloud backup | Litestream compatible |
| **Lighthouse** | Decentralized storage | IPFS/Filecoin |
| **PostgreSQL** | Aggregation | Multi-agent analytics |

```mermaid
graph LR
    subgraph agents["Multiple Agents"]
        A1["Agent 1<br/>SQLite"]
        A2["Agent 2<br/>SQLite"]
        A3["Agent 3<br/>SQLite"]
    end

    subgraph agg["Aggregation Layer"]
        PG["PostgreSQL"]
    end

    A1 -->|sync| PG
    A2 -->|sync| PG
    A3 -->|sync| PG

    style agents fill:#1a5276,stroke:#85c1e9
    style agg fill:#512e5f,stroke:#af7ac5
```

---

## Slide 7: Architecture Modes

```mermaid
graph TD
    subgraph sovereign["SOVEREIGN (Default)"]
        S1["SQLiteBackend<br/>(Primary)"]
        S2["SyncService"]
        S3["S3Target"]
        S4["LighthouseTarget"]
        S1 --> S2
        S2 --> S3
        S2 --> S4
    end

    subgraph advanced["ADVANCED (Server)"]
        A1["DatabaseBackend ABC"]
        A2["PostgresBackend"]
        A1 --> A2
    end

    note["Choose based on<br/>deployment needs"]

    style sovereign fill:#145a32,stroke:#58d68d
    style advanced fill:#1a5276,stroke:#85c1e9
```

**SQLite (Default):** Sovereign agents, offline-first, portable
**PostgreSQL (Advanced):** Multi-tenant, high-concurrency, server deployments

*Constitutional Council Decision (session 9282ed19):*
*SQLite approved as DEFAULT; PostgreSQL retained for advanced use cases.*
*See `kestrel_sovereign/data/council_sessions/` for full deliberation.*

---

## Slide 8: Benefits

```mermaid
graph TD
    subgraph sovereignty["Data Sovereignty"]
        S1["Your data = your file"]
        S2["No cloud required"]
        S3["Works offline"]
    end

    subgraph simplicity["Simplicity"]
        SI1["Single codebase"]
        SI2["No SQL dialect handling"]
        SI3["No placeholder conversion"]
    end

    subgraph flexibility["Flexibility"]
        F1["Optional cloud sync"]
        F2["Multiple targets"]
        F3["Choose your storage"]
    end

    style sovereignty fill:#145a32,stroke:#58d68d
    style simplicity fill:#1a5276,stroke:#85c1e9
    style flexibility fill:#512e5f,stroke:#af7ac5
```

---

*Reference: Issue #2, feedback/2026-01-12_postgres_migration_plan.md*
