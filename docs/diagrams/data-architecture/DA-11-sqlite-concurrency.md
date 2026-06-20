---
type: Diagram
title: 'DA-11: SQLite Concurrency Limitations'
description: Understanding and mitigating SQLite's single-writer constraint for complex
  agent architectures.
resource: /docs/diagrams/data-architecture/DA-11-sqlite-concurrency.md
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

# DA-11: SQLite Concurrency Limitations

Understanding and mitigating SQLite's single-writer constraint for complex agent architectures.

---

## Slide 1: The Single-Writer Constraint

```mermaid
graph TD
    subgraph sqlite["SQLite Architecture"]
        WAL["WAL Mode"]
        READERS["Multiple Readers OK"]
        WRITER["Single Writer ONLY"]
    end

    subgraph contention["Under Contention"]
        W1["Writer 1"] -->|"Acquires lock"| LOCK["Write Lock"]
        W2["Writer 2"] -->|"SQLITE_BUSY"| LOCK
        W3["Writer 3"] -->|"SQLITE_BUSY"| LOCK
    end

    style WRITER fill:#641e16,stroke:#ec7063,stroke-width:2px
    style LOCK fill:#7d6608,stroke:#f4d03f
```

**SQLite allows only one writer at a time.** Multiple processes or threads attempting concurrent writes will encounter blocking or errors.

---

## Slide 2: Symptoms of Write Contention

| Symptom | Description | Severity |
|---------|-------------|----------|
| **SQLITE_BUSY** | Database is locked by another writer | Common |
| **SQLITE_LOCKED** | Table-level lock conflict | Occasional |
| **Timeout errors** | `busy_timeout` exceeded | Configurable |
| **Slow writes** | Serialized queue buildup | Gradual |
| **Application hangs** | Deadlock in write queue | Severe |

```mermaid
sequenceDiagram
    participant A as Process A
    participant DB as SQLite
    participant B as Process B

    A->>DB: BEGIN TRANSACTION
    Note over DB: Write lock acquired
    B->>DB: INSERT (blocked)
    Note over B: Waiting...
    A->>DB: COMMIT
    Note over DB: Lock released
    B->>DB: INSERT (success)
```

---

## Slide 3: Thresholds and Benchmarks

```mermaid
graph LR
    subgraph thresholds["Write Load Thresholds"]
        LOW["< 10 writes/sec<br/>SQLite handles well"]
        MED["10-100 writes/sec<br/>Contention begins"]
        HIGH["> 100 writes/sec<br/>Bottleneck likely"]
    end

    LOW --> MED --> HIGH

    style LOW fill:#145a32,stroke:#58d68d
    style MED fill:#7d6608,stroke:#f4d03f
    style HIGH fill:#641e16,stroke:#ec7063
```

| Workload | SQLite Performance | Recommendation |
|----------|-------------------|----------------|
| Single-threaded CLI agent | Excellent | Use SQLite |
| Async agent with tools | Good with WAL | Use SQLite + mitigations |
| Multi-process agent | Fair with serialization | Consider mitigations |
| High-frequency logging | Poor | Separate DB or PostgreSQL |
| Parallel cognitive processes | Bottleneck | See Slide 7 |

---

## Slide 4: Mitigation - busy_timeout Configuration

```python
import aiosqlite

# Configure busy_timeout to wait instead of failing immediately
async def connect_with_timeout(db_path: str, timeout_ms: int = 5000):
    """
    Connect with busy_timeout to handle write contention gracefully.

    Args:
        db_path: Path to SQLite database
        timeout_ms: How long to wait for lock (default 5 seconds)
    """
    conn = await aiosqlite.connect(db_path)
    await conn.execute(f"PRAGMA busy_timeout = {timeout_ms}")
    await conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL
    return conn
```

```mermaid
graph TD
    subgraph timeout["busy_timeout Behavior"]
        REQ["Write Request"] --> LOCKED{"Lock Available?"}
        LOCKED -->|Yes| WRITE["Execute Write"]
        LOCKED -->|No| WAIT["Wait up to timeout"]
        WAIT --> RETRY{"Lock Free Now?"}
        RETRY -->|Yes| WRITE
        RETRY -->|No| ERROR["SQLITE_BUSY Error"]
    end

    style WRITE fill:#145a32,stroke:#58d68d
    style ERROR fill:#641e16,stroke:#ec7063
```

**Recommended timeout values:**
- CLI agents: 5,000 ms (5 seconds)
- Background tasks: 30,000 ms (30 seconds)
- Critical writes: 60,000 ms (60 seconds)

---

## Slide 5: Mitigation - Write Batching

```mermaid
graph TD
    subgraph before["Without Batching"]
        W1["Write 1"] --> DB1["DB"]
        W2["Write 2"] --> DB1
        W3["Write 3"] --> DB1
        W4["Write 4"] --> DB1
        Note1["4 transactions<br/>4 lock acquisitions"]
    end

    subgraph after["With Batching"]
        B1["Write 1"] --> BATCH["Batch Queue"]
        B2["Write 2"] --> BATCH
        B3["Write 3"] --> BATCH
        B4["Write 4"] --> BATCH
        BATCH --> DB2["DB"]
        Note2["1 transaction<br/>1 lock acquisition"]
    end

    style BATCH fill:#7d6608,stroke:#f4d03f
    style Note2 fill:#145a32,stroke:#58d68d
```

```python
import asyncio
from typing import List, Tuple, Any

class WriteBatcher:
    """Batch multiple writes into single transactions."""

    def __init__(self, db, flush_interval: float = 0.1, max_batch: int = 100):
        self.db = db
        self.queue: List[Tuple[str, Tuple]] = []
        self.flush_interval = flush_interval
        self.max_batch = max_batch
        self._lock = asyncio.Lock()

    async def add(self, sql: str, params: Tuple = ()):
        """Queue a write operation."""
        async with self._lock:
            self.queue.append((sql, params))
            if len(self.queue) >= self.max_batch:
                await self._flush()

    async def _flush(self):
        """Execute all queued writes in a single transaction."""
        if not self.queue:
            return

        async with self.db.execute("BEGIN IMMEDIATE"):
            for sql, params in self.queue:
                await self.db.execute(sql, params)
        await self.db.commit()
        self.queue.clear()
```

**Benefits:**
- Reduces lock contention by 10-100x
- Improves throughput for high-frequency writes
- Trade-off: Slightly increased latency per write

---

## Slide 6: Mitigation - Separate Databases Per Subsystem

```mermaid
graph TD
    subgraph agent["Kestrel Agent"]
        CORE["Core Logic"]
    end

    subgraph dbs["Separate Databases"]
        MAIN["agent.db<br/>Tasks, Sessions"]
        OBS["observability.db<br/>Logs, Metrics"]
        MEM["memory.db<br/>Long-term memory"]
        TOOL["tools.db<br/>Tool results cache"]
    end

    CORE --> MAIN
    CORE --> OBS
    CORE --> MEM
    CORE --> TOOL

    style MAIN fill:#1a5276,stroke:#85c1e9
    style OBS fill:#512e5f,stroke:#af7ac5
    style MEM fill:#145a32,stroke:#58d68d
    style TOOL fill:#7d6608,stroke:#f4d03f
```

```python
from dataclasses import dataclass
from kestrel_sovereign.storage.db import SQLiteBackend

@dataclass
class AgentDatabases:
    """Separate databases to reduce write contention."""

    main: SQLiteBackend      # Core agent state
    observability: SQLiteBackend  # High-frequency logs
    memory: SQLiteBackend    # Background memory consolidation
    tools: SQLiteBackend     # Tool execution results

    @classmethod
    async def create(cls, base_path: str) -> "AgentDatabases":
        return cls(
            main=await SQLiteBackend(f"{base_path}/agent.db").connect(),
            observability=await SQLiteBackend(f"{base_path}/observability.db").connect(),
            memory=await SQLiteBackend(f"{base_path}/memory.db").connect(),
            tools=await SQLiteBackend(f"{base_path}/tools.db").connect(),
        )
```

**Use case mapping:**
| Subsystem | Database | Write Frequency |
|-----------|----------|-----------------|
| Tasks, sessions | `agent.db` | Low-Medium |
| Logs, metrics | `observability.db` | High |
| Memory consolidation | `memory.db` | Medium |
| Tool results | `tools.db` | Variable |

---

## Slide 7: Cognitive Concurrency Patterns

Future agents may require parallel cognitive processes:

```mermaid
graph TD
    subgraph cognitive["Parallel Agent Processes"]
        PERCEPT["Perception<br/>Continuous input logging"]
        ACTION["Action<br/>Tool execution & state"]
        MEMORY["Memory Consolidation<br/>Background learning"]
        DREAM["Simulation/Dreaming<br/>Policy updates"]
    end

    subgraph conflict["Without Mitigation"]
        ALL["All writing to<br/>single agent.db"]
        BOTTLENECK["SQLITE_BUSY<br/>Cascade"]
    end

    cognitive --> conflict

    style BOTTLENECK fill:#641e16,stroke:#ec7063,stroke-width:2px
```

**Architectural patterns to avoid contention:**

```mermaid
graph TD
    subgraph pattern1["Pattern 1: Separate DBs"]
        P1["perception.db"]
        P2["action.db"]
        P3["memory.db"]
        P4["simulation.db"]
    end

    subgraph pattern2["Pattern 2: Write Queue"]
        QUEUE["Centralized<br/>Write Queue"]
        SINGLE["Single Writer<br/>Process"]
        DB["agent.db"]
        QUEUE --> SINGLE --> DB
    end

    subgraph pattern3["Pattern 3: Hybrid"]
        SQLITE["SQLite<br/>Local state"]
        PG["PostgreSQL<br/>Heavy writes"]
    end

    style pattern1 fill:#145a32,stroke:#58d68d
    style pattern2 fill:#1a5276,stroke:#85c1e9
    style pattern3 fill:#512e5f,stroke:#af7ac5
```

| Pattern | Pros | Cons |
|---------|------|------|
| **Separate DBs** | True parallelism | Join complexity |
| **Write Queue** | Single source of truth | Serialization overhead |
| **Hybrid** | Best of both | Operational complexity |

---

## Slide 8: When to Consider PostgreSQL

```mermaid
graph TD
    subgraph decision["Decision Tree"]
        START["Evaluate Workload"]
        Q1{"Multi-tenant<br/>deployment?"}
        Q2{"High write<br/>frequency?"}
        Q3{"Parallel<br/>processes?"}
        Q4{"Central<br/>analytics?"}

        START --> Q1
        Q1 -->|Yes| PG["Consider PostgreSQL"]
        Q1 -->|No| Q2
        Q2 -->|"> 100/sec"| PG
        Q2 -->|"< 100/sec"| Q3
        Q3 -->|"4+ writers"| PG
        Q3 -->|"1-3 writers"| Q4
        Q4 -->|Yes| PG
        Q4 -->|No| SQLITE["SQLite + Mitigations"]
    end

    style SQLITE fill:#145a32,stroke:#58d68d
    style PG fill:#512e5f,stroke:#af7ac5
```

**PostgreSQL advantages:**
| Feature | SQLite | PostgreSQL |
|---------|--------|------------|
| Concurrent writers | 1 | Unlimited (MVCC) |
| Connection pooling | N/A | Built-in |
| Multi-tenant isolation | Manual | Row-level security |
| Centralized analytics | Complex | Native |
| Horizontal scaling | No | Read replicas |

**When PostgreSQL makes sense:**
- Multi-agent fleets with shared analytics
- SaaS/multi-tenant deployments
- Write rates exceeding 100 ops/second sustained
- Central logging and observability requirements

---

## Slide 9: Summary - Mitigation Checklist

```mermaid
graph TD
    subgraph checklist["SQLite Concurrency Checklist"]
        C1["1. Enable WAL mode"]
        C2["2. Set busy_timeout (5-60s)"]
        C3["3. Batch high-frequency writes"]
        C4["4. Separate DBs for hot paths"]
        C5["5. Use write queues for parallelism"]
        C6["6. Monitor SQLITE_BUSY errors"]
        C7["7. Consider PostgreSQL for heavy loads"]
    end

    C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7

    style C1 fill:#145a32,stroke:#58d68d
    style C2 fill:#145a32,stroke:#58d68d
    style C3 fill:#1a5276,stroke:#85c1e9
    style C4 fill:#1a5276,stroke:#85c1e9
    style C5 fill:#7d6608,stroke:#f4d03f
    style C6 fill:#7d6608,stroke:#f4d03f
    style C7 fill:#512e5f,stroke:#af7ac5
```

**Quick Reference:**
```python
# Essential SQLite concurrency setup
async def setup_sqlite_for_concurrency(db_path: str):
    conn = await aiosqlite.connect(db_path)
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA busy_timeout = 5000")
    await conn.execute("PRAGMA synchronous = NORMAL")  # Balance durability/speed
    return conn
```

---

## Constitutional Reference

This documentation fulfills a recommendation from Constitutional Council session `9282ed19-117f-455d-8aa5-a8933be57eb0`:

> "Document the concurrency limitations of the SQLite backend clearly for developers building complex agents."

The council identified that SQLite's single-writer constraint could become a bottleneck for future "cognitive concurrency" patterns. This document provides:
- Clear explanation of the limitation
- Practical mitigation strategies
- Decision framework for when to upgrade to PostgreSQL

---

*Previous: [DA-10-sqlite-first-sync.md](DA-10-sqlite-first-sync.md) - SQLite-first architecture with sync layer*
*See also: [DA-02-database-abstraction.md](DA-02-database-abstraction.md) - Database backend interface*
