---
type: Architecture Spec
title: Legacy Embedding Column Inventory
description: Canonical inventory of every reader and writer of the legacy raw-SQL
  `embedding` column on `saved_items` and `document_chunks`, the verify/backfill
  helper for `embedding_vec`, the readers moved to `embedding_vec`, and the
  gated migration that drops the legacy column. Phases 1-3 of the retirement
  tracked in #2684.
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
| `embedding` (legacy) | `BLOB`, float32 little-endian | `BYTEA`, float32 little-endian | Written by nothing since [#3411](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3411). The startup migrations copy it into `embedding_vec`, the verify/backfill helper reads it, readers use it only while a table has no `embedding_vec` column, and the retirement migration drops it once the gate is met |
| `embedding_vec` (canonical) | `BLOB`, float32 little-endian | `vector(N)` (pgvector) with an HNSW index | Every writer ([#3411](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3411)), the SQLAlchemy ORM, the vector backends (`storage/vector`), and every raw-SQL reader ([#3409](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3409)) |

The split was the #1447 bridge: it let the pgvector path land without rewriting
the raw SQL that binds float32 bytes. Retirement is tracked in
[#2684](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2684) in
three phases, all complete:

1. **This inventory plus an idempotent verify/backfill of `embedding_vec`**
   ([#3402](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3402)).
   No reader or writer changes.
2. Pass the gate with `kestrel embeddings verify|backfill`
   ([#3405](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3405),
   see [How to run](#how-to-run)), then move every legacy reader to
   `embedding_vec`. The readers moved in
   [#3409](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3409);
   see [Readers](#readers-moved-to-embedding_vec).
3. Stop the dual writes, then retire the legacy column
   ([#3411](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3411)).
   Writers store vectors in `embedding_vec` only, and a startup migration
   drops the legacy column wherever the gate is met; see
   [Phase 3](#phase-3-retiring-the-legacy-column).

Line numbers in the schema table are against the tree that introduced this
page, except the phase-3 and #3414 rows; reader locations are against #3409,
the writer and phase-3 locations against #3411, and the #3414 rows against
[#3414](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3414).
Re-run the grep below before acting on them.

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
| `storage/async_database.py:1555-1571` | #3414: startup call to `migrate_backfill_legacy_embedding_vec` for each table, after the two migrations above and before the retirement. |
| `storage/sqla/migrations.py:1693` | #3414: `migrate_backfill_legacy_embedding_vec`. When both columns exist it copies every legacy vector whose `embedding_vec` is NULL, through `backfill_missing_embedding_vec`. |
| `storage/async_database.py:1548-1565` | Phase 3: startup call to `migrate_retire_legacy_embedding_column` for each table, after the two migrations above. |
| `storage/sqla/migrations.py:1707, 1723` | Phase 3: `legacy_embedding_retirement_gate_met` and `migrate_retire_legacy_embedding_column`. |
| `storage/sqla/migrations.py:1811` | Phase 3: `_legacy_only_vector_written`, the recheck under the drop's lock. It tests only whether each column is NULL and reads no legacy value. |

Both copy migrations run only when `embedding_vec` is **absent**, so neither
repairs a row that is later left with only the legacy column. On PostgreSQL the
column is not created by them at all until some row has a legacy embedding (the
width is sniffed from it). Since #3411 no row gets one, so the first embedded
write creates the column instead (see [Writers](#writers)).

A column the first write created starts empty, even when legacy rows exist. On
PostgreSQL that happens when a copy migration rolls back: it adds the column,
copies, then builds the HNSW index in one transaction, so an index that
refuses the width (more than 2000 dimensions) removes the column too. Until
[#3414](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3414)
every later boot found the column present and skipped the copy, so those rows
stayed out of vector search, and kept the legacy column from retiring, until an
operator ran `kestrel embeddings backfill`. Now `migrate_backfill_legacy_embedding_vec`
runs that copy on every boot the legacy column survives. It never creates
`embedding_vec`, and it writes only NULL values.

`CREATE TABLE` still declares the legacy column, so a fresh table starts with
it. Readers need a column to resolve until `embedding_vec` exists (see
[Readers](#readers-moved-to-embedding_vec)), and the retirement migration drops
it once that is true: on SQLite at the first boot, on PostgreSQL at the boot
after the first embedded write.

## ORM mappings (`embedding_vec` only)

| Location | Role |
|---|---|
| `storage/sqla/saved_item.py:67-78` | `SavedItem.embedding` maps the SQL column `embedding_vec`. The legacy column is deliberately unmapped. |
| `storage/sqla/document_chunk.py:60-67` | `DocumentChunk.embedding` maps `embedding_vec`. The legacy column is deliberately unmapped. |
| `storage/sqla/types.py` | `PortableVector`: `vector(N)` on PG, float32 bytes on SQLite. It also accepts legacy bytes on bind. |

## Writers

### Stopped: writers of the legacy `embedding` column

Until #3411 every production writer dual-wrote: the `INSERT` wrote the legacy
value, and a separate, non-fatal `UPDATE` wrote `embedding_vec`. Each of these
has stopped writing the legacy column. Locations are from the tree that
introduced this page.

| Former location | Table | What it wrote | Since #3411 |
|---|---|---|---|
| `storage/saved_items_store.py:489, 504-518` | `saved_items` | `save_item` serialized the vector and inserted it into `embedding`. | The `INSERT` names no embedding column. |
| `storage/async_rag_store.py:216, 223, 230` | `document_chunks` | `chunk_document` inserted the serialized vector into `embedding`. | `_insert_chunk` inserts `file_hash` and `content` only. |
| `storage/async_rag_store.py:367, 373, 380` | `document_chunks` | `store_precomputed_chunks` inserted into `embedding`. | Same `_insert_chunk`. |
| `scripts/validate_vector_lift_e2e.py:294` | `document_chunks` | The validation script seeds a legacy-only row. | Kept on purpose: it stands in for data an older release wrote, which the startup migration still lifts. The script now also checks that the boot retires the legacy column. |

`grep` for `embedding` in an `INSERT` or `SET` on either table finds no
production writer. Tests that need legacy data put the column back with
`tests/utils/legacy_embedding_column.py` and seed it directly.

### Writers of `embedding_vec`

| Location | Table | What it writes |
|---|---|---|
| `storage/saved_items_store.py:556` → `810` | `saved_items` | `save_item` creates the column if absent, then `_write_embedding_vec` sets `embedding_vec` and `embedding_profile_id` (PG `?::vector`, SQLite float32 bytes; single-column fallbacks). |
| `storage/async_rag_store.py:276` → `417` | `document_chunks` | `chunk_document`: the same, once per batch. |
| `storage/async_rag_store.py:383` → `417` | `document_chunks` | `store_precomputed_chunks`: the same. |
| `storage/embedding_column.py:84` | both | `ensure_embedding_vec_column(db, table, dimension)` creates an absent `embedding_vec`: `vector(N)` sized from the vector being written plus the HNSW index on PostgreSQL, `BLOB` on SQLite. A fresh PostgreSQL database needs it, because the startup migration sizes the column only from a legacy row and none is written any more. It runs inside `transaction()`, a savepoint under a caller's transaction, and a failure is logged, not raised. It copies no legacy vector; the next boot does (#3414). |

A failed `embedding_vec` write is logged at `WARNING` and leaves the row with no
stored vector: there is no legacy copy left to backfill from. `save_item`
reports `has_embedding: false` for it, and `kestrel embeddings reindex` embeds
it (reindex selects rows whose `embedding_vec` is NULL).

`chunk_document` stores a chunk even when it gets no vector for it: no
embedding service resolves, the embedding call fails, or the model returns no
vector or a batch of the wrong length. Since
[#3415](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3415) it
logs that at `WARNING` with the count, the file hash, the reason, and the
reindex command. Until then the no-service case logged nothing above `INFO`,
and a short batch silently dropped every chunk past its end.

`SavedItemsStore.update_item` never recomputes an embedding, so no embedding
column changes on edit.

### Reindex

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
  the legacy `embedding`. That is a PostgreSQL table no embedded write has
  reached yet: since #3411 the first one creates `embedding_vec` (see
  [Writers](#writers)), so such a table's legacy column holds only what an
  older release wrote. Reindex reads and writes `embedding_vec`, so it cannot
  have run against such a table, and those legacy bytes are current. The next
  boot creates the column and copies them. The retirement migration never
  drops the legacy column while `embedding_vec` is absent, so a reader always
  has a column to resolve.
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
| `storage/async_rag_store.py:287-338` | `document_chunks` | `read_indexed_chunks`: the copy pairs `embedding_profile_id` with `embedding_vec`, the vector reindex stamped it with. |
| `storage/async_rag_store.py:742-830` | `document_chunks` | `_legacy_in_python_search`: filters and scores the resolved column. Despite its name it is the **only** embedding path for a bound (per-agent) store; the generic vector spec has no ownership join. |

The `sqla/migrations.py` reads of the legacy column (lines 1401, 1446, 1582,
1617) and `embedding_vec_backfill.py` are not readers in this sense. They copy
the legacy value into `embedding_vec`. They stay after phase 3 for every
database whose legacy column the retirement migration has not yet dropped.

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
- `storage/async_rag_store.py:1124-1132` (`get_chunks_for_file`) reads
  `content` only.
- Sovereignty exports and sync snapshots copy the database file and carry both
  columns without interpreting them.

## Findings for phases 2 and 3

These were recorded before phase 3 and are kept for the history.

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
  [Readers](#readers-moved-to-embedding_vec)). Since #3411 only an older
  release can have written such rows.
- **Done in #3411: create `embedding_vec` before stopping the legacy write.**
  A fresh PostgreSQL database held its first vectors only in the legacy column
  until the next boot. Now the first embedded write creates the column at the
  vector's width (`ensure_embedding_vec_column`). The legacy branch of
  `stored_embedding_column` stays until no supported database can still have
  a table without `embedding_vec` (see [What remains](#what-remains)).
- **Fixed in #3414: a column the first write created was never filled from
  legacy rows.** The copy migrations skip a table that has `embedding_vec`, so
  legacy rows written before the first write created the column stayed out of
  vector search until an operator ran `kestrel embeddings backfill`. No data was
  lost, since the retirement refuses to drop the column while any row holds its
  vector only there. The startup sequence now copies them.

## Verify and backfill helper

`kestrel_sovereign/storage/embedding_vec_backfill.py` provides:

- `verify_embedding_vec(db, table)` is read-only and returns an
  `EmbeddingVecReport`.
- `backfill_embedding_vec(db, table, batch_size=500)` copies the legacy
  `embedding` into `embedding_vec` on rows where `embedding_vec IS NULL`, then
  reports.
- `backfill_missing_embedding_vec(db, table, batch_size=500)` makes the same
  copy without the report and returns `(rows_backfilled, rows_unbackfillable)`.
  It reads only the rows holding a legacy value and a NULL `embedding_vec`,
  never the vector pairs the report compares, because the startup sequence
  runs it on every boot the legacy column survives (#3414). It writes nothing
  when either column is absent.

`table` is `"saved_items"` or `"document_chunks"`. The report counts are below.
Every row is in exactly one of `rows_with_both`, `rows_missing_embedding_vec`,
`rows_embedding_vec_only` and `rows_without_any_embedding`, so those four sum
to `total_rows`. Building a report whose buckets do not sum to `total_rows`
raises `EmbeddingVecBackfillError`. Before
[#3415](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3415)
added `rows_without_any_embedding`, rows with no vector in either column were
in no bucket.

| Field | Meaning |
|---|---|
| `total_rows` | Every row in the table. |
| `rows_with_both` | Both columns are non-NULL. |
| `rows_missing_embedding_vec` | Legacy `embedding` is set and `embedding_vec` is NULL. |
| `rows_embedding_vec_only` | `embedding_vec` is set and legacy `embedding` is NULL (reindexed rows that had no vector). |
| `rows_without_any_embedding` | Both columns are NULL: the row was never embedded. Vector search cannot find it until `kestrel embeddings reindex --yes` embeds it. |
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
  absent column and copy the legacy vectors into it, since #3414 copy them
  into an existing one, and, since #3411, run the retirement migration. An absent column is reported as `ABSENT` and left for
  the agent's startup migration to create.
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
one has no column and cannot pass the gate until one exists. Since #3411 the
first embedded write creates it.

A table whose legacy column is already retired reports as if every legacy
value were NULL: `rows_embedding_vec_only` counts its stored vectors,
`rows_without_any_embedding` counts the rest, and every other count is 0, so it
meets the gate whenever `embedding_vec` exists.

`rows_without_any_embedding` is part of neither the phase-2 gate nor the
retirement gate, because a row with no vector has nothing to copy or to lose.
It is reported so that rows invisible to vector search are not also invisible
to the report. `verify` prints a note naming `kestrel embeddings reindex --yes`
whenever it is non-zero. The reindex selects every row whose `embedding_vec` is
NULL, so it embeds them.

## Phase 3: retiring the legacy column

[#3411](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3411)
stopped every legacy write (see [Writers](#writers)) and added
`migrate_retire_legacy_embedding_column(db, table)` to the startup sequence,
once per table, after the copy migrations and (since #3414) right after
`migrate_backfill_legacy_embedding_vec` copies that table's copyable legacy
vectors into an existing `embedding_vec`. It drops the legacy column with
`ALTER TABLE <table> DROP COLUMN embedding`, and only when the **retirement
gate** is met for that table on that database:

- `embedding_vec` exists, and
- `rows_missing_embedding_vec == 0` and `rows_unbackfillable == 0`, as
  `verify_embedding_vec` reports them.

This is stricter than the phase-2 gate that `kestrel embeddings verify` exits
on. That gate accepts unbackfillable rows (for example a legacy vector with a
NaN component), but those rows hold their only vector in the legacy column,
and the drop would destroy it. `rows_disagreeing` does not block the drop: a
disagreement means reindex already wrote the authoritative vector to
`embedding_vec`.

**Fail closed.** When the gate is not met the migration keeps the column and
every byte in it, logs `Keeping the legacy <table>.embedding column` with the
counts at `WARNING` (at `INFO` for a PostgreSQL table that has no
`embedding_vec` yet and no legacy vector, the normal state of a fresh
database), and the next boot tries again. A copyable legacy-only row does not
normally block it, because the boot copies it first (#3414). If that copy
failed (logged at `ERROR`) or the row landed after it, run
`kestrel embeddings backfill` or reboot. For rows the copy reports
unbackfillable, run `kestrel embeddings reindex`, which embeds every row whose
`embedding_vec` is NULL. Each table is judged on its own. A failure in the
migration is logged and non-fatal: nothing writes the legacy column any more,
so keeping it costs only disk.

**No write can slip in between.** The full gate (`verify_embedding_vec`) runs
without a lock, so a database that fails it never blocks another connection.
The drop then runs in one transaction under `BEGIN IMMEDIATE` on SQLite and
`LOCK TABLE ... IN ACCESS EXCLUSIVE MODE` on PostgreSQL. Before dropping, that
transaction rechecks with one statement:

```sql
SELECT EXISTS (SELECT 1 FROM <table>
               WHERE embedding IS NOT NULL AND embedding_vec IS NULL)
```

A release still dual-writing on the same database (a mixed-version fleet on
shared PostgreSQL) therefore cannot land a legacy-only row after the check.
For as long as the lock is held it blocks every access to the table on
PostgreSQL and every other writer on SQLite, and the boot waits with it. So the
recheck is not a second `verify_embedding_vec`. That call reads and compares
every vector pair in Python, while the recheck tests only whether each column
is NULL. It still scans the table at most once, but it transfers no vector
values.

The two checks split the gate between them:

| Refusal case | Unlocked full gate | Recheck under the lock |
|---|---|---|
| `embedding_vec` absent | yes | yes (catalog lookup) |
| A legacy value with a NULL `embedding_vec` (`rows_missing_embedding_vec`) | yes | yes |
| That value is unbackfillable (not whole float32, wrong `vector(N)` width, NaN or infinite) | yes, counted separately | yes, but not told apart: `verify_embedding_vec` counts an unbackfillable row only among the rows missing `embedding_vec`, so the recheck blocks on it without classifying it |
| PostgreSQL `embedding_vec` is not a pgvector `vector` | yes (`verify_embedding_vec` raises) | no; nothing between the checks changes a column type |

On PostgreSQL the lock wait is bounded by a 10-second `lock_timeout`; on
timeout the drop waits for the next boot. `lock_timeout` bounds only the wait,
not the hold, which is why the work under the lock is kept to one statement.
A concurrent boot that dropped the column first is detected under the lock.

**SQLite older than 3.35.0** has no `DROP COLUMN`. The migration keeps the
column there and says so at `WARNING`.

**The drop is not reversible, and it is forward-only.** An older release
inserts into the legacy column, so it fails against a table that has lost it.
Before deploying #3411, run `kestrel embeddings verify` on every data directory
and back up each database; roll back only by restoring that backup. Until a
table's column is dropped, older releases keep working against it.

## Rows that were never embedded

After #3411 deployed, `verify` on four co-hosted agent databases reported
buckets that did not sum to `total_rows`: 47 `document_chunks` rows per
database had neither column set, and the pre-drop backup showed the same 47,
so no vector was lost
([#3415](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3415)).
Grouped by `file_hash`, all 47 are one file: the chunks of
`KESTREL_CONSTITUTION.md` that `kestrel constitution reanchor` wrote when it
reanchored each agent, with a NULL `embedding_profile_id` as well.

The reanchor re-indexes RAG through `chunk_document(compute_embeddings=True)`
on a store built without the agent's `LLMService`. The embedding service is
therefore resolved by a bare process-local `LLMService()`, which never applies
the agent's persisted `embedding_route` from `agent_metadata`
(`kestrel embeddings reindex` does, through
`_apply_persisted_embedding_config`). When the first available chat route
cannot embed and names no `embedding_sibling`, that service resolves nothing,
and the affected host lists such a route first. `chunk_document` then stores
every chunk without a vector, and the only log line is the `LLMService` `INFO`
that the route "does not support embeddings", which a CLI does not print. This
is the probable cause, not a proven one: the rows record only the missing
vector, and no log of that run survives, so a failed or empty embedding call
cannot be ruled out. Every one of those branches stored the chunks the same
way.

Such rows are not intentionally excluded. `kestrel embeddings reindex --yes`
embeds them, and the gaps are now reported both when the chunks are written
and by `verify`.

## What remains

Phase 3 leaves two pieces of compatibility code, both still reachable:

- `CREATE TABLE` still declares the legacy column, and
  `stored_embedding_column` still falls back to it when a table has no
  `embedding_vec`. A PostgreSQL table reaches `embedding_vec` only through
  its first embedded write, and until then readers need a column to resolve.
  Removing both needs `embedding_vec` in the fresh schema at a known width
  (for example from configuration, as `conversation_history` does).
- The copy migrations and `migrate_backfill_legacy_embedding_vec` in
  `sqla/migrations.py`, and `embedding_vec_backfill.py`, read the legacy column
  of databases that have not yet met the gate. They can go once no supported
  deployment can still hold one.
