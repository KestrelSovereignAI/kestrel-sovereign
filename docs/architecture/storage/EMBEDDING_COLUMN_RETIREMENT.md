---
type: Architecture Spec
title: Legacy Embedding Column Inventory
description: Canonical inventory of every reader and writer of the legacy raw-SQL
  `embedding` column on `saved_items` and `document_chunks`, the verify/backfill
  helper for `embedding_vec`, and the readers moved to `embedding_vec`. Phases 1
  and 2 of the retirement tracked in #2684.
resource: /docs/architecture/storage/EMBEDDING_COLUMN_RETIREMENT.md
tags:
- docs
- architecture
- architecture-spec
- storage
timestamp: '2026-09-29T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Legacy Embedding Column Inventory

`saved_items` and `document_chunks` each store an embedding twice:

| Column | SQLite | PostgreSQL | Who uses it |
|---|---|---|---|
| `embedding` (legacy) | `BLOB`, float32 little-endian | `BYTEA`, float32 little-endian | The dual-write inserts, the startup migrations, the verify/backfill helper, and readers only while a table has no `embedding_vec` column |
| `embedding_vec` (canonical) | `BLOB`, float32 little-endian | `vector(N)` (pgvector) with an HNSW index | The SQLAlchemy ORM, the vector backends (`storage/vector`), and every raw-SQL reader ([#3409](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3409)) |

The split was the #1447 bridge: it let the pgvector path land without rewriting
the raw SQL that binds float32 bytes. Retirement is tracked in
[#2684](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2684) in
three phases:

1. **This inventory plus an idempotent verify/backfill of `embedding_vec`**
   ([#3402](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3402)).
   No reader or writer changes.
2. Pass the gate with `kestrel embeddings verify|backfill`
   ([#3405](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3405),
   see [How to run](#how-to-run)), then move every legacy reader to
   `embedding_vec`. The readers moved in
   [#3409](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3409);
   see [Readers](#readers-moved-to-embedding_vec).
3. Stop the dual writes, then retire the legacy column.

This page is the starting point for phase 3. Line numbers in the schema and
writer tables are against the tree that introduced this page; reader
locations are against #3409. Re-run the grep below before acting on them.

```bash
grep -rn "embedding" kestrel_sovereign/storage kestrel_sovereign/identity \
  kestrel_sovereign/features kestrel_sovereign/agent scripts \
  | grep -E "saved_items|document_chunks|, embedding\b|embedding IS|\.embedding\b"
```

Source-tree paths below are relative to `kestrel_sovereign/` unless they start
with `scripts/`.

## Schema and migrations

| Location | Role |
|---|---|
| `storage/async_database.py:241-246` | `CREATE TABLE document_chunks (..., embedding BLOB)`. The legacy column is line 245. |
| `storage/async_database.py:659-676` | `CREATE TABLE saved_items (..., embedding BLOB, ...)`. The legacy column is line 668. |
| `storage/db/placeholder.py:224` | Rewrites `BLOB` to `BYTEA` when the schema is created on PostgreSQL. |
| `storage/async_database.py:1506-1507` | Startup call to `migrate_saved_items_add_embedding_vec`. |
| `storage/async_database.py:1525-1526` | Startup call to `migrate_document_chunks_add_embedding_vec`. |
| `storage/sqla/migrations.py:1323` | `migrate_saved_items_add_embedding_vec`. |
| `storage/sqla/migrations.py:1401` | PG: reads `octet_length(embedding)` from one row to choose the `vector(N)` width. |
| `storage/sqla/migrations.py:1446, 1466` | PG: reads every legacy `embedding` and writes `embedding_vec`. Rows whose width differs are skipped. |
| `storage/sqla/migrations.py:1517` | SQLite: `UPDATE saved_items SET embedding_vec = embedding`. |
| `storage/sqla/migrations.py:1534` | `migrate_document_chunks_add_embedding_vec`, which delegates to the generic helpers below. |
| `storage/sqla/migrations.py:1582, 1617, 1635` | `_migrate_pg_table`: dimension sniff, legacy read, `embedding_vec` write. |
| `storage/sqla/migrations.py:1680` | `_migrate_sqlite_table`: copy `embedding` into `embedding_vec`. |

Both migrations run only when `embedding_vec` is **absent**, so neither repairs
a row that is later left with only the legacy column. On PostgreSQL the column
is not created at all until some row has a legacy embedding (the width is
sniffed from it). Until then every PG `embedding_vec` write fails and is logged,
and only the legacy column holds the vector.

## ORM mappings (`embedding_vec` only)

| Location | Role |
|---|---|
| `storage/sqla/saved_item.py:67-78` | `SavedItem.embedding` maps the SQL column `embedding_vec`. The legacy column is deliberately unmapped. |
| `storage/sqla/document_chunk.py:60-67` | `DocumentChunk.embedding` maps `embedding_vec`. The legacy column is deliberately unmapped. |
| `storage/sqla/types.py` | `PortableVector`: `vector(N)` on PG, float32 bytes on SQLite. It also accepts legacy bytes on bind. |

## Writers of the legacy `embedding` column

Every production writer dual-writes. The legacy value is written by the
`INSERT`, and `embedding_vec` by a separate follow-up `UPDATE` that is
non-fatal on failure.

| Location | Table | What it writes |
|---|---|---|
| `storage/saved_items_store.py:489, 504-518` | `saved_items` | `save_item` serializes the vector (`_serialize_embedding`, line 303) and inserts it into `embedding`. |
| `storage/saved_items_store.py:557` → `794-892` | `saved_items` | Dual-write: `_write_embedding_vec` sets `embedding_vec` (PG `?::vector` at 832, SQLite bytes at 838; single-column fallbacks at 862 and 868). |
| `storage/async_rag_store.py:216, 223, 230` | `document_chunks` | `chunk_document` inserts the serialized vector into `embedding`. |
| `storage/async_rag_store.py:282` → `407-513` | `document_chunks` | Dual-write: `_write_embedding_vec` (PG at 451, SQLite at 457; fallbacks at 481 and 487). |
| `storage/async_rag_store.py:367, 373, 380` | `document_chunks` | `store_precomputed_chunks` inserts into `embedding`. |
| `storage/async_rag_store.py:399` | `document_chunks` | Dual-write for precomputed chunks. |
| `scripts/validate_vector_lift_e2e.py:294` | `document_chunks` | Validation script seeds a legacy-only row. |

`SavedItemsStore.update_item` (`storage/saved_items_store.py:666-711`) never
recomputes an embedding, so neither column changes on edit.

## Writers of `embedding_vec` only

| Location | Effect |
|---|---|
| `storage/embedding_reindex.py:459-484` (`_write_row`, via `kestrel embeddings reindex` in `cli_embeddings.py`) | Rewrites `embedding_vec` and `embedding_profile_id` to the target profile. The legacy `embedding` column is **not** touched, so after a reindex the two representations disagree by design. Reindex also embeds rows that had no vector at all, which leaves them with `embedding_vec` only. |

## Readers (moved to `embedding_vec`)

Since [#3409](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3409)
every raw-SQL reader resolves its column through
`storage/embedding_column.py`:

- `stored_embedding_column(db, table)` asks the catalog (`column_exists`)
  whether the table has `embedding_vec`. If it does, readers select
  `embedding_vec` on SQLite and `embedding_vec::text` on PostgreSQL, and
  filter on `embedding_vec IS NOT NULL`. A row whose `embedding_vec` is NULL
  has no stored vector, whatever its legacy column holds. Nothing falls back
  to the legacy value per row.
- Only when the table has **no `embedding_vec` column at all** do readers use
  the legacy `embedding`. That is a fresh PostgreSQL database, where the
  startup migration defers `vector(N)` until a legacy row shows the width
  (see [Schema and migrations](#schema-and-migrations)). Reindex reads and
  writes `embedding_vec`, so it cannot have run against such a table, and its
  legacy bytes are current. The next boot creates the column and copies them.
- `decode_stored_embedding(value)` decodes float32 little-endian bytes and
  pgvector text. It is the one decoder; the per-store `_deserialize_embedding`
  helpers are gone.

| Location | Table | Reader |
|---|---|---|
| `storage/saved_items_store.py:249-265` | `saved_items` | `SavedItem.from_row` decodes `row[8]`, the stored embedding every `SELECT` below names through `_SAVED_ITEM_COLUMNS` (line 322) and `_item_columns` (line 719). |
| `storage/saved_items_store.py:734, 744, 754` | `saved_items` | `get_by_id`, `get_by_content_hash`, `list_by_content_hash`. |
| `storage/saved_items_store.py:784` | `saved_items` | `list_items`. |
| `storage/saved_items_store.py:1012` | `saved_items` | `_search_via_vector_backend` hydrates kNN hits (the ranking itself already used `embedding_vec`). |
| `storage/saved_items_store.py:1112-1157` | `saved_items` | `_legacy_in_python_search`: filters and scores the resolved column. |
| `storage/saved_items_store.py:1209, 1322, 1338` | `saved_items` | `_text_search`, `list_by_schema`, `list_by_tag`. |
| `storage/saved_items_store.py:1385-1413` | `saved_items` | `get_stats`: `with_embedding` counts the resolved column. |
| `storage/saved_items_store.py:724` | `saved_items` | `_stored_embedding`: `save_item` returns the vector that landed in the resolved column, not the one it computed. After a failed `embedding_vec` write the item reports no embedding. |
| `storage/async_rag_store.py:275-317` | `document_chunks` | `read_indexed_chunks`: the copy pairs `embedding_profile_id` with `embedding_vec`, the vector reindex stamped it with. |
| `storage/async_rag_store.py:722-793` | `document_chunks` | `_legacy_in_python_search`: filters and scores the resolved column. Despite its name it is the **only** embedding path for a bound (per-agent) store; the generic vector spec has no ownership join. |

The `sqla/migrations.py` reads of the legacy column (lines 1401, 1446, 1582,
1617) and `embedding_vec_backfill.py` are not readers in this sense. They copy
the legacy value into `embedding_vec` and stay until phase 3.

### Indirect consumers

These read `SavedItem.embedding` (the dataclass field), which `from_row` and
`save_item` now populate from the resolved column:

| Location | Use |
|---|---|
| `features/save/feature.py:280, 402, 497` | `has_embedding` in tool results. A row embedded only by reindex reports `true`. |
| `agent/memory_manager.py:757` | `has_embedding` in saved-item context blocks. |
| `scripts/validate_vector_lift_e2e.py:483` | Asserts a saved item has an embedding. |

### Not consumers

- `identity/exporter.py:362-380` and `identity/importer.py:830-867` select and
  insert explicit column lists that omit both embedding columns. Imported
  saved items carry no vector until they are re-embedded.
- `storage/async_rag_store.py:1116-1124` (`get_chunks_for_file`) reads
  `content` only.
- Sovereignty exports and sync snapshots copy the database file and carry both
  columns without interpreting them.

## Findings for phases 2 and 3

- **Fixed in #3409: the in-Python search scored a stale vector after a
  reindex.** Both `_legacy_in_python_search` methods filter on the current
  `embedding_profile_id` (the RAG one only when the embedding service reports
  a profile), which reindex rewrites, and then scored the legacy bytes, which
  reindex does not. A reindexed row matched the new profile and was scored
  with the old model's vector. For `document_chunks` this was every bound
  store's embedding search, not a fallback.
- **Fixed in #3409: `has_embedding` reported the legacy column.** A row
  embedded only through reindex reported `has_embedding: false`.
- **Disagreement is expected, not corruption.** After a reindex,
  `embedding_vec` is the authoritative representation. The verify helper
  counts disagreeing rows. It never rewrites them.
- **PostgreSQL can have legacy-only rows with no `embedding_vec` column.**
  Until the column exists the backfill helper reports it as absent and changes
  nothing. Creating it stays with the startup migration, which chooses the
  vector width. Readers use the legacy column in that state (see
  [Readers](#readers-moved-to-embedding_vec)).
- **Phase 3 must create `embedding_vec` before it stops the legacy write.**
  A fresh PostgreSQL database holds its first vectors only in the legacy
  column until the next boot. Once the legacy write stops, the first write
  must create the column at the vector's width (or the width must come from
  configuration). Only then can the legacy branch of
  `stored_embedding_column` be deleted with the column.

## Verify and backfill helper

`kestrel_sovereign/storage/embedding_vec_backfill.py` provides:

- `verify_embedding_vec(db, table)` is read-only and returns an
  `EmbeddingVecReport`.
- `backfill_embedding_vec(db, table, batch_size=500)` copies the legacy
  `embedding` into `embedding_vec` on rows where `embedding_vec IS NULL`, then
  reports.

`table` is `"saved_items"` or `"document_chunks"`. The report counts:

| Field | Meaning |
|---|---|
| `total_rows` | Every row in the table. |
| `rows_with_both` | Both columns are non-NULL. |
| `rows_missing_embedding_vec` | Legacy `embedding` is set and `embedding_vec` is NULL. |
| `rows_embedding_vec_only` | `embedding_vec` is set and legacy `embedding` is NULL (reindexed rows that had no vector). |
| `rows_disagreeing` | Both are set and decode to different float32 vectors. |
| `rows_backfilled` | Rows this call wrote (always 0 for `verify_embedding_vec`). |
| `rows_unbackfillable` | Missing rows that cannot be copied: a byte length that is not a positive multiple of 4, a width that differs from the PG `vector(N)` column, a NaN or infinite component (pgvector rejects it, and SQLite applies the same rule), or no `embedding_vec` column. |

Guarantees:

- **Idempotent.** Only NULL `embedding_vec` values are written, and each
  `UPDATE` re-checks `embedding_vec IS NULL`. A second run writes 0 rows, and
  a concurrent dual-write is never overwritten.
- **Resumable and batch-based.** Rows are walked by primary key in batches,
  and each batch commits in its own transaction. An interrupted run leaves
  finished batches committed, and the next run picks up the rest.
  Unbackfillable rows never stall the walk.
- **No schema change and no deletion.** It never creates, drops or nulls a
  column, and it never modifies `embedding` or an existing `embedding_vec`.

## How to run

The helper is exposed as two `kestrel embeddings` subcommands
([#3405](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3405)):

```bash
kestrel embeddings verify   [--table saved_items|document_chunks|all] [--json]
kestrel embeddings backfill [--table saved_items|document_chunks|all] [--batch-size N] [--json]
```

- `verify` calls `verify_embedding_vec` and writes nothing. `backfill` calls
  `backfill_embedding_vec` (default batch size 500). Both print every report
  field above per table. `--json` prints one JSON document with a per-table
  and an overall `gate_met`.
- They select the database like `audit` and `reindex`: `--agent-name`,
  `--data-dir`, `KESTREL_DB_PATH`, or `KESTREL_DATABASE_URL` for PostgreSQL.
- They open it **without** the startup schema initializer. The default open
  runs the `embedding_vec` startup migration, which would itself create an
  absent column and copy the legacy vectors into it. An absent column is
  reported as `ABSENT` and left for the agent's startup migration to create.
- On SQLite, `verify` opens the file read-only through the cold-read
  connection ([#3407](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3407)).
  It never sets the journal mode. A checkpointed database (no `-wal`/`-shm`)
  is opened `immutable=1`, which creates no file, so `verify` works on a
  read-only file. A live agent's WAL is read with plain `mode=ro`. If a writer
  commits while `verify` reads a checkpointed database, `verify` refuses to
  report (exit `2`). `backfill` writes, and opens the database normally.

**The phase-2 gate.** Before deploying the reader switch (#3409), run
`kestrel embeddings backfill`, then `kestrel embeddings verify`, on every
deployment. Proceed only when `verify` exits 0 and `rows_disagreeing` has been
reviewed. A disagreement is expected after a reindex, so it is reported but
does not affect the exit code. After the switch a row that `verify` counts in
`rows_missing_embedding_vec` is invisible to search and reports
`has_embedding: false` until `backfill` copies it.

| Exit code | Meaning |
|---|---|
| `0` | Gate met: every table has an `embedding_vec` column and `rows_missing_embedding_vec == rows_unbackfillable`. |
| `3` | Gate not met: a table still has rows the backfill can copy, or its `embedding_vec` column is absent (even when the table is empty). |
| `2` | Did not run: a usage error, an ambiguous or missing database, a failed connection, an unsupported backend, a PostgreSQL `embedding_vec` that is not a `vector`, or a SQLite database that changed while `verify` was reading it. |
| `1` | An unexpected error (an uncaught exception). |

An absent column never meets the gate, however few rows the table holds.
Phase 2b points readers at `embedding_vec`, and they would query a column that
does not exist. The row equality alone would not catch it: with the column
absent, the helper counts every legacy row as unbackfillable, so the counts
match while no vector has moved. On PostgreSQL the startup migration sizes
`vector(N)` from an existing legacy embedding, so a table that has never held
one has no column and cannot pass the gate until one exists and the agent
restarts.
