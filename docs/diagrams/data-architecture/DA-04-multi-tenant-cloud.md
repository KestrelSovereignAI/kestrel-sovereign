# DA-04: Multi-Tenant Cloud (Kestrel PostgreSQL)

Kestrel cloud architecture with row-level security.

---

## Slide 1: Kestrel Multi-Tenant Architecture

```mermaid
graph TD
    subgraph users["Users"]
        U1["👤 User A"]
        U2["👤 User B"]
        U3["👤 User C"]
    end

    subgraph api["Kestrel API"]
        AUTH["JWT Authentication"]
        ROUTE["Request Router"]
    end

    subgraph db["PostgreSQL"]
        TABLE["Shared tables"]
        RLS["Row-Level Security"]
    end

    users --> api --> db

    style AUTH fill:#7d6608,stroke:#f4d03f
    style RLS fill:#512e5f,stroke:#af7ac5
```

**Shared database.** Isolated data per user.

---

## Slide 2: Core Tables

```mermaid
graph TD
    subgraph tables["PostgreSQL Tables"]
        USERS["users<br/>id, email, password_hash, tier"]
        COMPANIONS["companions<br/>id, user_id, name, personality"]
        MESSAGES["messages<br/>id, companion_id, role, content"]
        MEMORIES["memories<br/>id, companion_id, content, embedding"]
    end

    USERS --> COMPANIONS --> MESSAGES
    COMPANIONS --> MEMORIES

    style USERS fill:#1a5276,stroke:#85c1e9
    style COMPANIONS fill:#512e5f,stroke:#af7ac5
```

**Hierarchical.** User → Companions → Messages/Memories.

---

## Slide 3: Row-Level Security

```mermaid
graph TD
    subgraph policy["RLS Policy"]
        RULE["user_id = current_user()"]
    end

    subgraph effect["Effect"]
        E1["User A sees only User A's data"]
        E2["User B sees only User B's data"]
        E3["No cross-user leakage"]
    end

    policy --> effect

    style policy fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style effect fill:#145a32,stroke:#58d68d
```

**Database-enforced isolation.** Can't leak data even with SQL injection.

---

## Slide 4: Companion → KestrelAgent Mapping

```mermaid
sequenceDiagram
    participant U as User
    participant F as Kestrel API
    participant C as CompanionService
    participant K as KestrelAgent

    U->>F: POST /api/companions/{id}/chat
    F->>F: Extract user_id from JWT
    F->>C: get_companion(id, user_id)
    C->>C: Verify ownership
    C->>K: get_or_create_agent(companion_id)
    K-->>C: Agent instance
    C->>K: agent.chat(message)
    K-->>C: Response
    C-->>F: Response
    F-->>U: JSON response
```

**Each companion = One KestrelAgent.**

---

## Slide 5: Vector Embeddings (pgvector)

```mermaid
graph LR
    subgraph memory["Memory Storage"]
        TEXT["Memory text"]
        VEC["embedding vector(1536)"]
    end

    subgraph search["Semantic Search"]
        QUERY["User query"]
        QVEC["Query embedding"]
        COS["Cosine similarity"]
        RESULTS["Relevant memories"]
    end

    TEXT --> VEC
    QUERY --> QVEC --> COS --> RESULTS

    style VEC fill:#7d6608,stroke:#f4d03f
    style COS fill:#512e5f,stroke:#af7ac5
```

**pgvector extension.** Native vector similarity search in PostgreSQL.

---

## Slide 6: Connection Pooling

```mermaid
graph TD
    subgraph pool["asyncpg Pool"]
        MIN["min_size: 10"]
        MAX["max_size: 100"]
        REUSE["Connection reuse"]
    end

    subgraph perf["Performance"]
        P1["No connection overhead"]
        P2["Automatic scaling"]
        P3["Health checks"]
    end

    pool --> perf

    style pool fill:#1a5276,stroke:#85c1e9
    style perf fill:#145a32,stroke:#58d68d
```

**Efficient connections.** Pool handles concurrency.

---

## Slide 7: Subscription Tiers

| Tier | Messages | Companions | Features |
|------|----------|------------|----------|
| Trial | 25 free | 1 | Basic chat |
| Free | 100/day | 3 | + Memory |
| Premium | Unlimited | 10 | + Image gen, priority |
| Enterprise | Unlimited | 100 | + API access, SLA |

```mermaid
graph LR
    TRIAL["Trial<br/>25 msgs"] --> FREE["Free<br/>100/day"] --> PREMIUM["Premium<br/>Unlimited"] --> ENTERPRISE["Enterprise<br/>API access"]

    style TRIAL fill:#641e16,stroke:#ec7063
    style FREE fill:#7d6608,stroke:#f4d03f
    style PREMIUM fill:#512e5f,stroke:#af7ac5
    style ENTERPRISE fill:#145a32,stroke:#58d68d
```

---

## Slide 8: Cloud Infrastructure

```mermaid
graph TD
    subgraph gcp["Google Cloud"]
        RUN["Cloud Run<br/>Kestrel API"]
        SQL["Cloud SQL<br/>PostgreSQL"]
        REDIS["Memorystore<br/>Redis"]
        VPC["VPC Connector"]
    end

    RUN --> VPC --> SQL
    RUN --> VPC --> REDIS

    style RUN fill:#1a5276,stroke:#85c1e9
    style SQL fill:#512e5f,stroke:#af7ac5
    style REDIS fill:#7d3c00,stroke:#f5b041
```

**Production stack.** Cloud Run + Cloud SQL + Memorystore.

---

*Next: [DA-05-ipfs-sovereignty.md](DA-05-ipfs-sovereignty.md) - IPFS sharding and convergent encryption*
