---
type: Diagram
title: 'DA-02: Database Abstraction Layer'
description: 'One codebase, two backends: SQLite and PostgreSQL.'
resource: /docs/diagrams/data-architecture/DA-02-database-abstraction.md
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

# DA-02: Database Abstraction Layer

One codebase, two backends: SQLite and PostgreSQL.

---

## Slide 1: The Problem

```mermaid
graph TD
    subgraph problem["Two Different Backends"]
        KESTREL["Kestrel Standalone<br/>Uses SQLite"]
        kestrel["Kestrel Multi-Tenant<br/>Uses PostgreSQL"]
    end

    subgraph issue["The Issue"]
        I1["Different SQL dialects"]
        I2["Different placeholders"]
        I3["Different functions"]
        I4["Code can't be shared"]
    end

    problem --> issue

    style issue fill:#641e16,stroke:#ec7063
```

**Two backends, one codebase?** We need abstraction.

---

## Slide 2: The Solution - DatabaseBackend ABC

```mermaid
graph TD
    subgraph app["Application Layer"]
        FEATURES["Features, Agents, Tools"]
    end

    subgraph interface["DatabaseBackend (ABC)"]
        EXEC["execute(sql, params)"]
        FETCH["fetch_one() / fetch_all()"]
        TRANS["transaction()"]
    end

    subgraph impl["Implementations"]
        SQLITE["SQLiteBackend<br/>(aiosqlite)"]
        POSTGRES["PostgresBackend<br/>(asyncpg)"]
    end

    app --> interface --> impl

    style interface fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style SQLITE fill:#1a5276,stroke:#85c1e9
    style POSTGRES fill:#512e5f,stroke:#af7ac5
```

**Write once.** Run on either backend.

---

## Slide 3: Placeholder Conversion

```mermaid
graph LR
    subgraph code["Your Code"]
        SQL["SELECT * FROM users<br/>WHERE id = ? AND name = ?"]
    end

    subgraph sqlite["SQLite"]
        S1["SELECT * FROM users<br/>WHERE id = ? AND name = ?"]
    end

    subgraph postgres["PostgreSQL"]
        P1["SELECT * FROM users<br/>WHERE id = $1 AND name = $2"]
    end

    code -->|"SQLiteBackend"| sqlite
    code -->|"PostgresBackend"| postgres

    style code fill:#7d6608,stroke:#f4d03f
    style sqlite fill:#1a5276,stroke:#85c1e9
    style postgres fill:#512e5f,stroke:#af7ac5
```

**Automatic conversion.** `?` → `$1, $2, $3`

---

## Slide 4: SQL Dialect Helpers

| Function | SQLite | PostgreSQL |
|----------|--------|------------|
| Current time | `datetime('now')` | `NOW()` |
| UUID generation | `hex(randomblob(16))` | `uuid_generate_v4()` |
| JSON type | `json` | `jsonb` |
| Boolean | `0/1` | `true/false` |
| Text concat | `||` | `||` |

```mermaid
graph TD
    subgraph helpers["UnifiedStoreBase Helpers"]
        NOW["now_sql() → dialect-specific"]
        UUID["uuid_sql() → dialect-specific"]
        JSON["json_type() → json or jsonb"]
    end

    style helpers fill:#1a5276,stroke:#85c1e9
```

---

## Slide 5: UnifiedStoreBase

```mermaid
graph TD
    subgraph base["UnifiedStoreBase"]
        ABSTRACT["Abstract store methods"]
        HELPERS["SQL dialect helpers"]
        CONVERT["Placeholder converter"]
    end

    subgraph stores["Concrete Stores"]
        TASK["TaskStore"]
        SESSION["SessionService"]
        MEMORY["MemoryService"]
        OBS["ObservabilityStore"]
    end

    base --> stores

    style base fill:#7d6608,stroke:#f4d03f,stroke-width:2px
```

**Base class.** All A2A stores inherit from it.

---

## Slide 6: Usage Example

```python
# Works on BOTH SQLite and PostgreSQL!

class TaskStore(UnifiedStoreBase):
    async def get_task(self, task_id: str) -> Task:
        sql = """
            SELECT * FROM tasks
            WHERE id = ? AND created_at > ?
        """
        # Automatically converts:
        # SQLite: ... WHERE id = ? AND created_at > ?
        # Postgres: ... WHERE id = $1 AND created_at > $2

        row = await self.db.fetch_one(sql, [task_id, cutoff])
        return Task(**row)
```

```mermaid
graph LR
    CODE["Single codebase"] --> BOTH["Runs on both"]
    BOTH --> SQLITE["SQLite<br/>Local agents"]
    BOTH --> POSTGRES["PostgreSQL<br/>Kestrel cloud"]

    style CODE fill:#145a32,stroke:#58d68d
```

---

*Next: [DA-03-local-storage.md](DA-03-local-storage.md) - AsyncStorage facade and specialized stores*
