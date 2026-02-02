# DA-03: Local Storage (SQLite)

AsyncStorage facade and five specialized stores.

---

## Slide 1: The AsyncStorage Facade

```mermaid
graph TD
    subgraph facade["AsyncStorage"]
        UNIFIED["Single interface"]
        ASYNC["Fully async"]
    end

    subgraph stores["Specialized Stores"]
        FILES["📁 FileStore"]
        CONV["💬 ConversationStore"]
        GRAPH["🕸️ GraphStore"]
        RAG["🔍 RAGStore"]
        DB["💾 DatabaseStore"]
    end

    facade --> stores

    style facade fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style FILES fill:#1a5276,stroke:#85c1e9
    style CONV fill:#512e5f,stroke:#af7ac5
    style GRAPH fill:#0e4d45,stroke:#48c9b0
    style RAG fill:#7d3c00,stroke:#f5b041
    style DB fill:#145a32,stroke:#58d68d
```

**One interface.** Five specialized backends.

---

## Slide 2: FileStore - Content-Addressable

```mermaid
graph LR
    subgraph input["File Input"]
        CONTENT["File content"]
        HASH["SHA-256 hash"]
    end

    subgraph storage["Storage"]
        KEY["Hash as key"]
        BLOB["Encrypted blob"]
    end

    subgraph benefit["Benefits"]
        DEDUP["Automatic deduplication"]
        VERIFY["Integrity verification"]
    end

    CONTENT --> HASH --> KEY --> BLOB
    storage --> benefit

    style HASH fill:#7d6608,stroke:#f4d03f
    style DEDUP fill:#145a32,stroke:#58d68d
```

**Content-addressed.** Same content = same key = deduplicated.

---

## Slide 3: ConversationStore

```mermaid
graph TD
    subgraph schema["Schema"]
        MSG["messages table"]
        TS["timestamp"]
        ROLE["role: user/assistant"]
        CONTENT["content (encrypted)"]
        META["metadata (json)"]
    end

    subgraph features["Features"]
        ENC["🔐 Encrypted at rest"]
        FTS["🔍 Full-text search"]
        PAGE["📄 Pagination"]
    end

    schema --> features

    style ENC fill:#512e5f,stroke:#af7ac5
    style FTS fill:#1a5276,stroke:#85c1e9
```

**Encrypted + searchable.** Via KESTREL_DATA_KEY.

---

## Slide 4: GraphStore - Knowledge Graph

```mermaid
graph TD
    subgraph structure["Graph Structure"]
        NODES["Nodes<br/>(entities, concepts)"]
        EDGES["Edges<br/>(relationships)"]
    end

    subgraph example["Example"]
        USER["👤 User: Alice"]
        LIKES["--likes-->"]
        TOPIC["📚 Topic: AI"]
    end

    USER --> LIKES --> TOPIC

    subgraph queries["Queries"]
        Q1["Find related nodes"]
        Q2["Traverse relationships"]
        Q3["Pattern matching"]
    end

    structure --> queries

    style NODES fill:#1a5276,stroke:#85c1e9
    style EDGES fill:#7d6608,stroke:#f4d03f
```

**Semantic relationships.** Agent's understanding of the world.

---

## Slide 5: RAGStore - Retrieval Augmented Generation

```mermaid
graph TD
    subgraph ingest["Document Ingestion"]
        DOC["Document"]
        CHUNK["Chunk into segments"]
        EMBED["Generate embeddings"]
        STORE["Store vectors"]
    end

    subgraph retrieve["Retrieval"]
        QUERY["User query"]
        VEC["Query → vector"]
        SEARCH["Similarity search"]
        RESULTS["Top-k chunks"]
    end

    DOC --> CHUNK --> EMBED --> STORE
    QUERY --> VEC --> SEARCH --> RESULTS

    style EMBED fill:#7d6608,stroke:#f4d03f
    style SEARCH fill:#512e5f,stroke:#af7ac5
```

**Semantic search.** Find relevant context for any query.

---

## Slide 6: DatabaseStore - Generic Tables

```mermaid
graph LR
    subgraph tables["Tables"]
        WALLET["wallet_state"]
        ANCHORS["log_anchors"]
        FEEDBACK["feedback"]
        CUSTOM["custom tables"]
    end

    subgraph ops["Operations"]
        CRUD["CRUD operations"]
        MIGRATE["Schema migrations"]
        BACKUP["Backup/restore"]
    end

    tables --> ops

    style tables fill:#1a5276,stroke:#85c1e9
```

**Catch-all.** For everything that isn't files, conversations, graph, or RAG.

---

## Slide 7: Directory Structure

```
agent_data/
└── {companion_id}/
    ├── kestrel.db          # Main SQLite database
    ├── files/              # Content-addressed files
    │   ├── ab/
    │   │   └── cd1234...   # SHA-256 hash prefix
    │   └── ef/
    │       └── gh5678...
    └── embeddings/         # Vector embeddings cache
```

```mermaid
graph TD
    ROOT["agent_data/"] --> CID["{companion_id}/"]
    CID --> DB["kestrel.db"]
    CID --> FILES["files/"]
    CID --> EMB["embeddings/"]

    style DB fill:#145a32,stroke:#58d68d
    style FILES fill:#1a5276,stroke:#85c1e9
```

---

## Slide 8: Async All The Way

```mermaid
sequenceDiagram
    participant A as Agent
    participant S as AsyncStorage
    participant DB as aiosqlite

    A->>S: await storage.add_message(msg)
    S->>DB: await db.execute(sql)
    DB-->>S: Result
    S-->>A: Message stored

    Note over A,DB: Non-blocking I/O
    Note over A,DB: Agent can handle other requests
```

**aiosqlite.** Non-blocking SQLite for async Python.

---

*Next: [DA-04-multi-tenant-cloud.md](DA-04-multi-tenant-cloud.md) - Kestrel PostgreSQL multi-tenancy*
