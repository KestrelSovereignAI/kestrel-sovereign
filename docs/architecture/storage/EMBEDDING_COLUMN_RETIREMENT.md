---
type: Architecture Spec
title: Legacy Embedding Column Inventory
description: Canonical inventory of every reader and writer of the legacy raw-SQL
  `embedding` column on `saved_items` and `document_chunks`, and the verify/backfill
  helper for `embedding_vec`. Phase 1 of the retirement tracked in #2684.
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
| `embedding` (legacy) | `BLOB`, float32 little-endian | `BYTEA`, float32 little-endian | Raw `AsyncDatabase` IO: inserts, row hydration, the in-Python fallback search, stats |
| `embedding_vec` | `BLOB`, float32 little-endian | `vector(N)` (pgvector) with an HNSW index | The SQLAlchemy ORM and the vector backends (`storage/vector`) |

The split was the #1447 bridge: it let the pgvector path land without rewriting
the raw SQL that binds float32 bytes. Retirement is tracked in
[#2684](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2684) in
three phases:

1. **This inventory plus an idempotent verify/backfill of `embedding_vec`**
   ([#3402](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3402)).
   No reader or writer changes.
2. Move every legacy reader to `embedding_vec`.
3. Stop the dual writes, then retire the legacy column.

This page is the starting point for phases 2 and 3. Line numbers are against
the tree that introduced this page. Re-run the grep below before acting on
them.

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
| `storage/async_database.py:1486-1487` | Startup call to `migrate_saved_items_add_embedding_vec`. |
| `storage/async_database.py:1505-1506` | Startup call to `migrate_document_chunks_add_embedding_vec`. |
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

## Readers of the legacy `embedding` column

| Location | Table | Reader |
|---|---|---|
| `storage/saved_items_store.py:248-262` (line 259) | `saved_items` | `SavedItem.from_row` decodes `row[8]` (legacy `embedding`) into the dataclass `embedding` field. Every `SELECT` below feeds it. |
| `storage/saved_items_store.py:717` | `saved_items` | `get_by_id`. |
| `storage/saved_items_store.py:728` | `saved_items` | `get_by_content_hash`. |
| `storage/saved_items_store.py:739` | `saved_items` | `list_by_content_hash`. |
| `storage/saved_items_store.py:775, 785` | `saved_items` | `list_items`. |
| `storage/saved_items_store.py:1081` | `saved_items` | `_search_via_vector_backend` hydrates kNN hits (the ranking itself uses `embedding_vec`). |
| `storage/saved_items_store.py:1136-1179` | `saved_items` | `_legacy_in_python_search`: `WHERE embedding IS NOT NULL` (1145, 1153) and cosine over the legacy bytes (1179). |
| `storage/saved_items_store.py:1243, 1255` | `saved_items` | `_text_search`. |
| `storage/saved_items_store.py:1313` | `saved_items` | `list_by_schema`. |
| `storage/saved_items_store.py:1335, 1345` | `saved_items` | `list_by_tag`. |
| `storage/saved_items_store.py:1400-1411` | `saved_items` | `get_stats`: `with_embedding` counts `embedding IS NOT NULL`. |
| `storage/async_rag_store.py:311-326` | `document_chunks` | `read_indexed_chunks` decodes the legacy bytes. `identity/birth_record.py:640` uses it to copy chunks, and `store_precomputed_chunks` (called at `identity/birth_record.py:868`) writes them back into both columns. |
| `storage/async_rag_store.py:773-806` | `document_chunks` | `_legacy_in_python_search`: `WHERE embedding IS NOT NULL` and cosine over the legacy bytes. It serves when the vector backend is unavailable or fails. |
| `storage/sqla/migrations.py:1401, 1446, 1582, 1617` | both | Phase-2 migrations (see above). |

### Indirect consumers of the legacy value

These read `SavedItem.embedding` (the dataclass field), which `from_row`
populates **from the legacy column**:

| Location | Use |
|---|---|
| `features/save/feature.py:280, 402, 497` | `has_embedding` in tool results. |
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

- **The legacy fallback reads a stale vector after a reindex.**
  `SavedItemsStore._legacy_in_python_search` filters on the current
  `embedding_profile_id`, which reindex rewrites. It then scores the legacy
  bytes, which reindex does not rewrite. A reindexed row can therefore match
  the new profile while its legacy vector comes from the old model.
  `AsyncRAGStore._legacy_in_python_search` scores legacy bytes without a
  profile filter at all. Moving these readers to `embedding_vec` in phase 2
  removes both problems.
- **`has_embedding` reports the legacy column.** A row embedded only through
  reindex reports `has_embedding: false`.
- **Disagreement is expected, not corruption.** After a reindex,
  `embedding_vec` is the authoritative representation. The verify helper
  counts disagreeing rows. It never rewrites them.
- **PostgreSQL can have legacy-only rows with no `embedding_vec` column.**
  Until the column exists the backfill helper reports it as absent and changes
  nothing. Creating it stays with the startup migration, which chooses the
  vector width.

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
| `rows_unbackfillable` | Missing rows that cannot be copied: a byte length that is not a positive multiple of 4, a width that differs from the PG `vector(N)` column, or no `embedding_vec` column. |

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

It has no CLI entry point yet. Phase 2 is expected to run it, and to require
`rows_missing_embedding_vec == rows_unbackfillable` and a reviewed
`rows_disagreeing`, before any reader switches columns.
