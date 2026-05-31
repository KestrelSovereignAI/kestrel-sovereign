# Kestrel Storage Architecture

**Status:** Active implementation snapshot
**Last updated:** 2026-05-31

This document describes the storage stack that is in the current repository.
Older versions of this page described a pre-async, pre-SQLAlchemy migration plan;
that plan is no longer the source of truth.

## Current Shape

Kestrel storage has three layers:

1. `AsyncStorage` is the high-level facade used by agents. It composes file,
   conversation, graph, and RAG stores.
2. `AsyncDatabase` owns schema initialization, backend-agnostic SQL execution,
   and idempotent startup migrations.
3. SQLAlchemy mappings and vector backends provide typed vector search for
   tables that need kNN retrieval.

The default local backend is SQLite. PostgreSQL is supported through the async
backend layer by setting `KESTREL_DB_BACKEND=postgres` and a PostgreSQL DSN
through `KESTREL_DATABASE_URL` or explicit configuration.

## Backend Layer

`AsyncDatabase` wraps a `DatabaseBackend`:

| Backend | Implementation | Notes |
|---|---|---|
| SQLite | `storage/db/sqlite.py` | Local default; file-backed; vector search uses Python cosine fallback today. |
| PostgreSQL | `storage/db/postgres.py` | Cloud/server-capable; vector search uses pgvector where the table has a vector column and HNSW index. |

Application code writes SQLite-style `?` placeholders. The backend layer
normalizes SQL for PostgreSQL where needed.

## SQLAlchemy Layer

SQLAlchemy is used for typed ORM views and vector search, not as a wholesale
replacement for the raw `AsyncDatabase` facade. `make_session_factory(db)`
constructs a SQLAlchemy async engine pointed at the same SQLite file or
PostgreSQL DSN and caches it on the `AsyncDatabase` for the app lifetime.

The current SQLAlchemy-mapped vector tables are:

| Table | Mapping | Vector column | Status |
|---|---|---|---|
| `saved_items` | `storage/sqla/saved_item.py` | `embedding_vec` | Active vector backend path with legacy `embedding` dual-write. |
| `document_chunks` | `storage/sqla/document_chunk.py` | `embedding_vec` | Active RAG vector backend path with legacy `embedding` dual-write. |
| `conversation_history` | `storage/sqla/conversation_message.py` | `embedding_vec` | Storage/schema groundwork landed; `MemoryRetriever` still uses keyword/concept overlap in the current tree. |

## Vector Search

The vector backend factory dispatches by SQLAlchemy dialect:

| Dialect | Backend | Behavior |
|---|---|---|
| PostgreSQL | `PgVectorBackend` | Uses pgvector `<=>` and HNSW indexes. |
| SQLite / other | `PurePythonBackend` | Reads embeddings through SQLAlchemy and computes cosine in Python. |

`SqliteVecBackend` is still a future extension point. Do not document sqlite-vec
as shipped until the backend exists in `storage/vector/`.

## Embedding Generation

Embedding storage and vector search are separate from embedding generation.
Today the in-tree `EmbeddingService` is still Ollama-backed:

- default model: `nomic-embed-text`
- default dimension: 768
- alternate local models can change the dimension

The next architecture step is standardizing embedding functions on LLM providers
so embeddings are provider capabilities rather than a hardcoded Ollama side path.
Until that work lands, docs should not say that OpenAI/Anthropic/OpenRouter/etc.
provide Kestrel embeddings through the unified LLM adapter interface.

## Startup Migrations

`AsyncDatabase._init_schema()` runs the core schema and then idempotent
migrations from `storage/sqla/migrations.py`:

- `saved_items.embedding_vec`
- `document_chunks.embedding_vec`
- `conversation_history.embedding_vec`
- `conversation_history.rendered_content`
- `conversation_history.deleted_at`

On PostgreSQL, vector migrations create the `vector` extension where needed and
add HNSW indexes. On SQLite, vector columns are `BLOB`s and use the pure-Python
backend.

Failures in vector migrations are non-fatal for startup. The affected feature
falls back to the legacy search path and logs the failure for the next boot.

## Retrieval Surfaces

| Surface | Data | Search path |
|---|---|---|
| Cognitive memory | `conversation_history` + graph metadata | `MemoryRetriever` six-factor scoring; semantic factor is currently keyword/concept overlap. |
| RAG documents | `document_chunks` | Embedding search through SQLAlchemy/vector backend plus BM25 and LIKE fallback. |
| Saved items | `saved_items` | Embedding search through SQLAlchemy/vector backend with legacy fallback. |
| A2A task archive | A2A memory service tables | Full-text search for archived task transcripts. |

## Privacy And Encryption Notes

- `EPHEMERAL` privacy mode avoids persistent conversation storage.
- Conversation content can be encrypted at rest through the conversation store.
- The SQLAlchemy vector backends operate on precomputed embedding columns and do
  not decrypt `content` / `rendered_content`.
- Export/import and backup behavior is owned by sovereignty/storage-tier code,
  not by the vector backend.

## Operational Notes

- SQLite is appropriate for local, single-agent use and test environments.
- PostgreSQL is the production/concurrent backend.
- Switching embedding models can change vector dimension. Existing vector
  columns are not automatically resized; operators need an explicit re-embedding
  or migration plan before changing dimensions on a populated deployment.
