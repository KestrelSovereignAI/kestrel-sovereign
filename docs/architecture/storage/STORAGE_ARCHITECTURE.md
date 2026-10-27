---
type: Architecture Spec
title: Kestrel Storage Architecture
description: '**Status:** Active implementation snapshot **Last updated:** 2026-05-31'
resource: /docs/architecture/storage/STORAGE_ARCHITECTURE.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-07-26T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Kestrel Storage Architecture

**Status:** Active implementation snapshot
**Last updated:** 2026-10-10

This document describes the storage stack that is in the current repository.
Older versions of this page described a pre-async, pre-SQLAlchemy migration plan;
that plan is no longer the source of truth.

> **Semantic knowledge boundary:** The current graph, RAG, saved-item, memory,
> and vector surfaces described here are implementation storage surfaces; they
> are not competing semantic truth stores. The design-of-record for the planned
> canonical assertion layer, standards profiles, lifecycle, privacy, and
> migration boundary is [Semantic Knowledge Architecture](SEMANTIC_KNOWLEDGE_ARCHITECTURE.md).
> Nothing in this page claims that its current tables already implement that
> layer.

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

**PostgreSQL 16 is the floor** (#2988). The `conversation_history.session_id`
backfill (#2958) has to *skip* metadata it cannot cast rather than raise on it:
a raise inside a startup migration rolls back the whole migration, and because
the schema is the marker, the next boot retries from the same row and fails
identically. The guard that makes skipping possible is
`pg_input_is_valid(metadata, 'jsonb')` — 16+, and it answers "would this cast
succeed?" without performing the cast.

Nothing weaker works, because the documents `jsonb` rejects cannot be
enumerated from outside the parser. `IS JSON OBJECT` (also 16+) returns *true*
for both a NUL escape and a lone surrogate, and `::jsonb` then raises on each —
measured on 16.14. `IS JSON OBJECT` is still in the statement, for the
different job of proving the document is an object before `json_object_keys` is
asked about it.

On PostgreSQL 15 the backfill fails during schema initialization rather than
degrading quietly. CI and the devcontainer pin `pgvector/pgvector:pg16`.

## Backend Layer

`AsyncDatabase` wraps a `DatabaseBackend`:

| Backend | Implementation | Notes |
|---|---|---|
| SQLite | `storage/db/sqlite.py` | Local default; file-backed; vector search uses Python cosine fallback today. |
| PostgreSQL | `storage/db/postgres.py` | Cloud/server-capable; vector search uses pgvector where the table has a vector column and HNSW index. |

Application code writes SQLite-style `?` placeholders. The backend layer
normalizes SQL for PostgreSQL where needed.

### Hosted execution custody

`kestrel_sovereign.execution_custody` carries a host-authored immutable
companion/DID/owner/runtime-generation binding. The host implements
`ExecutionFence.require_work()` for synchronous admission checks and
`lock_and_validate(connection)` for validation on the native mutation's actual
PostgreSQL connection. Validators lock authority rows before graph/file rows,
retain those locks through commit, and must not begin or commit a transaction.
They do not derive authority from telemetry, trace IDs or tool arguments.

Pass the original `ExecutionCustody` to `KestrelAgent(execution_custody=...)` or
`PostgresBackend.from_pool(..., execution_custody=...)`. It is retained on the
runtime/backend and combined with ambient admissions, so background tasks and
foreign turn executors cannot silently become unfenced. Explicit transaction
membership is fixed at entry; introducing a new admission inside it is refused
before SQL. Copied children retain irrevocable denial after retirement.

Native `advisory_locks(..., on_loss=...)` yields an `AdvisoryLease` for the exact
physical session. Physical loss is observed during acquisition and ownership;
normal release retires the handle without a loss notification. The scheduler
uses this capability, its existing claim CAS, and captured host generation for
renewal and effect admission. Replacement locks never repair old admissions.
No invocation-long transaction or additional authority pool is introduced.

Custody-bound SQLite bootstrap and SQLAlchemy execution are refused, including
cached sessions/connections. Rollback and close remain available. Hosted stream
cleanup cannot dispatch cognition/tools or perform general storage writes. The
native Stop store may settle only an existing exact generation/owner identity
through its fixed terminal CAS on the same bounded operational pool. Completed
effects whose ordinary checkpoint cannot be made durable remain in Stop's
unresolved ledger, not a clean completion. This cannot recall an external
effect already submitted before authority was lost.

`execution_commit_outcome(error)` distinguishes successful commit followed by
denial (`committed`) from lost commit acknowledgement (`unknown`), including
legacy storage exception wrapping. Reanchor reports reconciliation required
and enters SafeMode; neither outcome is described as rollback or safe to retry.
Single-query, batch and script checkouts retain commit state through pool reset
and release. Such uncertainty irreversibly denies further work in the original
admission even if a legacy caller catches its error; only trusted reconciliation
and a new admission can reopen work. Invocation, scheduler source/dispatcher,
tool and transport boundaries preserve this evidence, and joined parallel
children cannot discard it in favor of an ordinary sibling error.
Successful native COMMIT and reset are different from unresolved acknowledgement
or retirement: an observer cancellation can carry their exact owned receipt
without fencing healthy scopes. A local denial still permanently denies its
original scope. Pending reset, actual ACK/reset failure, physical debt and
unscoped control never use this settled-local classification. Only native
epilogues are joined this way; transaction bodies remain cancellable. See
[`EXECUTION_CUSTODY.md`](../../development/EXECUTION_CUSTODY.md) for receipt and
scope-provenance details.
This is an opt-in native contract, not evidence that a downstream host has
installed it or passed live rollout acceptance.

### SQLite close budget

Close observes a retained rollback drain's completion separately from its
terminal error. A drain's cancellation is not cancellation of the close caller;
real caller cancellation (including a request pending before close starts) is
delivered only after drain and native connection retirement. Original control
errors and failed native closers remain failures, not successful shutdown.

Every aiosqlite connection Kestrel opens is closed under one budget, which
covers aiosqlite's `close()` and then the exit of the connection's worker
thread. `KESTREL_AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S` sets it:

- **What it bounds.** SQLite's own close runs inside the window. When the last
  connection to a WAL database closes, SQLite checkpoints the whole `-wal`
  into the main file, fsyncs it and deletes the `-wal`. The budget therefore
  bounds that disk I/O as well as the thread exit. One 10 ms poll interval is
  added to it, so errors name `1.01s`.
- **Default:** `1.0` seconds. Values below `0.01` are raised to `0.01`.
- **Read once, at import** of `kestrel_sovereign/storage/db/sqlite.py`. Set it
  in the environment before the process starts; changing it afterwards has no
  effect.
- **It also sizes the shutdown reservation.** SQLite storage advertises the
  budget as `minimum_close_timeout_s`, and whole-agent shutdown reserves that
  much of its own budget for the storage close (`_resolve_shutdown_budget` in
  `kestrel_sovereign/kestrel_agent.py`). A larger value leaves less time for
  the shutdown steps before it, and takes the whole shutdown budget once it
  exceeds it.

A close that overruns the budget raises `ConnectionError` and logs the same
message once at WARNING, with what the close had to do (#3460):

```text
SQLite connection close did not complete within 1.01s (elapsed=1.012s budget=1.01s database=kestrel_prime.db journal_mode=wal wal_bytes=2525592 wal_pages=613)
```

- `elapsed` is measured from the start of the close; `budget` is the
  configured budget.
- `database` is the file's basename only, never its path.
- `journal_mode` is what SQLite reported when the backend set it at connect.
  It is `unknown` where the opener does not set it: cold reads, backup
  targets and SQLAlchemy engines.
- `wal_bytes` is the size of the `-wal`, from a `stat` taken before the close
  began. `wal_pages` counts the frames that size holds, one page each, using
  the page size the backend read at connect; it is `unknown` where the opener
  has no page size, on the same three paths. Both are `unknown` if the `stat`
  failed, and `absent` when there is no `-wal`. They are omitted for a
  database known not to be in WAL mode. The `-wal` is never opened: every
  close takes this reading, and on ext4 a read can update the file's access
  time, which can wait on the filesystem journal during the very fsync stalls
  this reports on.

A worker that outlives the window reports the same fields after
`SQLite worker did not terminate`.

Either overrun raises `SQLiteCloseOverrunError`, a `ConnectionError`, and the
backend keeps the close until its worker exits. A file-backed snapshot read
(one made while another task holds the write transaction) has its rows before
its connection closes. If that close overruns and the read itself succeeded,
the read returns its rows and logs a second WARNING. It does not raise a
`QueryError` (#3782). The read still fails when the read itself failed, when
the overrun carries control evidence, or when the reading task was cancelled
during the close.

## SQLAlchemy Layer

SQLAlchemy is used for typed ORM views and vector search, not as a wholesale
replacement for the raw `AsyncDatabase` facade. `make_session_factory(db)`
constructs a SQLAlchemy async engine pointed at the same SQLite file or
PostgreSQL DSN and caches it on the `AsyncDatabase` for the app lifetime.

The current SQLAlchemy-mapped vector tables are:

| Table | Mapping | Vector column | Status |
|---|---|---|---|
| `saved_items` | `storage/sqla/saved_item.py` | `embedding_vec` | Active vector backend path; the only embedding column writers use (#3411). |
| `document_chunks` | `storage/sqla/document_chunk.py` | `embedding_vec` | Active RAG vector backend path; the only embedding column writers use (#3411). |
| `conversation_history` | `storage/sqla/conversation_message.py` | `embedding_vec` | Storage/schema groundwork landed; `MemoryRetriever` still uses keyword/concept overlap in the current tree. |

`embedding_vec` is the canonical vector column. Nothing writes the legacy
`embedding` column on `saved_items` and `document_chunks` any more, and a
startup migration drops it once no row depends on it (#2684, #3411). The
former legacy readers and writers, the `embedding_vec` verify/backfill helper,
and the retirement gate are described in
[Legacy Embedding Column Inventory](EMBEDDING_COLUMN_RETIREMENT.md).

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
Saved-items and RAG now resolve embeddings through the active agent-scoped LLM
provider route when that route advertises embedding support:

- OpenAI defaults to `text-embedding-3-small` at 1536 dimensions.
- Google/Gemini and Vertex default to `text-embedding-004` at 768 dimensions.
- Ollama defaults to `nomic-embed-text` at 768 dimensions.
- Anthropic advertises no embedding API. OpenRouter is disabled by default and
  advertises embeddings only for a route with an explicit `embedding_model`
  and `embedding_dim`; otherwise storage falls back to text/BM25/LIKE search.

The provider adapter boundary is a common Python shape, not a common semantic
space: `Optional[list[float]]` for one embedding and one batch result slot per
input text. Switching embedding models or providers can change both vector
dimension and ranking behavior, so existing rows need re-embedding or matching
vector columns before mixed-model semantic search should be trusted.

## Startup Migrations

`AsyncDatabase._init_schema()` runs the core schema and then idempotent
migrations from `storage/sqla/migrations.py`:

- `saved_items.embedding_vec`
- `document_chunks.embedding_vec`
- `conversation_history.embedding_vec`
- `conversation_history.rendered_content`
- `conversation_history.deleted_at`
- `conversation_history.session_id`

On PostgreSQL, vector migrations create the `vector` extension where needed and
add HNSW indexes. On SQLite, vector columns are `BLOB`s and use the pure-Python
backend.

Failures in vector migrations are non-fatal for startup. The affected feature
falls back to the legacy search path and logs the failure for the next boot.

### The derived `session_id` column

`conversation_history.session_id` is a **derived duplicate** of
`metadata.session_id`, added so sessions can be indexed (#2958). Metadata stays
authoritative; the column exists to be queried. Its one invariant is:

> The column may be NULL where session grouping is canonical.
> It may never disagree.

Which values may be copied into it is decided in exactly one place —
`storage/session_id_column.py` — and compiled into three renderings that have to
agree: Python for the write paths, SQLite and PostgreSQL for the two halves of
the legacy backfill. The rule is deliberately narrower than the grouper's, so
anything the three readers would spell differently (a non-string JSON value, a
non-ASCII identifier, a NUL, an id too long for a B-tree entry, or a
`session_id` key that appears twice in one document) lands as NULL rather than
as a value one backend agrees with and another does not.

After insertion, `update_message_metadata` is the one door that can rewrite
`metadata.session_id`, so it moves the column in the same statement. The column
follows the metadata down to NULL wherever the JSON merge cannot speak for the
whole document — and both engines have a shape where they decline to merge
while still reporting success. A metadata root that is not an object (`42`,
`[]`, `"text"`, JSON `null`) comes back from SQLite's `json_set` with nothing
added to it, and is *concatenated into an array* rather than merged by
PostgreSQL's `jsonb ||`; on SQLite a document carrying `session_id` twice keeps
the second occurrence, which is the one session grouping reads. None of those
rows ends up carrying the incoming id, so none of them may have it stamped. A
SQL `NULL` metadata column is not one of these cases — it is an absence, the
merge's `COALESCE` makes it `{}`, and it stamps normally.

Indexes for these migrations go through `AsyncDatabase.ensure_index`, not a bare
`CREATE INDEX IF NOT EXISTS`: that spelling is idempotent in sequence but races
in parallel, and `_init_schema` runs on every `from_pool()`. `ensure_index`
returns only once the named index really exists on the named table, and raises
otherwise — `IF NOT EXISTS` is satisfied by the name alone, which is unique per
database on SQLite and per schema on PostgreSQL, so a same-named index on
another table would otherwise make the DDL a silent no-op.

## Retrieval Surfaces

| Surface | Data | Search path |
|---|---|---|
| Cognitive memory | `conversation_history` + graph metadata | `MemoryRetriever` six-factor scoring; semantic factor is currently keyword/concept overlap. |
| RAG documents | `document_chunks` | Embedding search through SQLAlchemy/vector backend plus BM25 and LIKE fallback. |
| Saved items | `saved_items` | Embedding search through SQLAlchemy/vector backend with legacy fallback. |
| A2A task archive | A2A memory service tables | Full-text search for archived task transcripts. |

These retrieval surfaces return stored material or ranked candidates. A vector
hit, property-graph node, conversation summary, or RAG chunk is not a canonical
semantic assertion merely because it is retrieved. The future assertion layer
will filter lifecycle and privacy state before ranking these projections; see
[Semantic Knowledge Architecture](SEMANTIC_KNOWLEDGE_ARCHITECTURE.md).

## Privacy And Encryption Notes

- `EPHEMERAL` privacy mode avoids persistent conversation storage.
- `DEIDENTIFIED` is a fail-closed preset: `PrivacyEnforcingStorage` denies
  every generic durable write. It is not a synonym for `NORMAL` or
  `ANONYMOUS`; the `storage=deidentified` setting alone does not authorize a
  semantic, graph, vector, memory, export, or training write. The only durable
  write it admits is `store_deidentified_records`, which persists the output of
  the `kestrel_sovereign.deidentification` pipeline together with its
  validated Safe Harbor (or Expert Determination) evidence artifact, in one
  transaction. See [Privacy Modes](../security/PRIVACY_MODES.md).
- Conversation content can be encrypted at rest through the conversation store.
- The SQLAlchemy vector backends operate on precomputed embedding columns and do
  not decrypt `content` / `rendered_content`.
- Export/import and backup behavior is owned by sovereignty/storage-tier code,
  not by the vector backend.
- `AsyncGraphStore` ownership ledgers constrain current graph rows by agent;
  they do not supply the planned assertion revision, provenance, ontology, or
  inference-retraction contract. That contract is intentionally additive.

## Operational Notes

- SQLite is appropriate for local, single-agent use and test environments.
- PostgreSQL is the production/concurrent backend.
- Switching embedding models can change vector dimension. Existing vector
  columns are not automatically resized; operators need an explicit re-embedding
  or migration plan before changing dimensions on a populated deployment.
