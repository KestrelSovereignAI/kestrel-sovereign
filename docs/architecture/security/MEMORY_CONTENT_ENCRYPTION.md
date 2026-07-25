---
type: Architecture Decision Record
title: Saved-item and RAG Content Encryption
description: Design of record for application-layer encryption, tenant binding, search, migration, rotation, and recovery for durable memory bodies.
resource: /docs/architecture/security/MEMORY_CONTENT_ENCRYPTION.md
tags:
- docs
- architecture
- security
- storage
- encryption
timestamp: '2026-07-25T00:00:00Z'
status: design-of-record
owner: security
canonical: true
generated: false
privacy: public
---

# Saved-item and RAG content encryption

## Status and scope

This ADR is the implementation-ready design for issue
[#2677](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2677).
It is **not an implementation claim**. Today, `saved_items.content` and
`document_chunks.content` are plaintext `TEXT` columns. The rollout is complete
only after every child boundary in [Phased delivery](#phased-delivery) lands and
the migration reports full coverage.

The decision covers the two durable user-memory bodies and every path that must
temporarily see their plaintext: CRUD, lexical and vector search, embedding,
migration, rotation, import/export, backup, and repair. It does not expand the
encrypted field set to saved-item names, summaries, tags, metadata, file names,
or embeddings.

## Decision summary

1. Add a single shared `MemoryContentCodec`; stores and maintenance jobs must
   not implement encryption independently.
2. Extend the SDK key hierarchy with the `memory-content` purpose. Derive one
   key per agent, then bind each ciphertext to its tenant, table, column, and
   row identifier with AEAD associated data (AAD).
3. Keep the legacy `content` column only as a nullable plaintext migration
   source. Store the encrypted body in a new binary `content_ciphertext`
   column with explicit envelope and key-version columns.
4. Add `document_chunks.agent_id` as the authoritative tenant boundary.
   Shared files may remain content-addressed, but derived chunks are
   tenant-owned rows and are never shared between agents.
5. Replace SQL `LIKE` over encrypted bodies with a keyed exact-token candidate
   index. RAG BM25 remains an agent-scoped in-memory index built from decrypted
   chunks. Every candidate is tenant-scoped, decrypted, and post-verified.
6. Generate embeddings from plaintext immediately before encrypting a new
   body, or immediately after an authorized decrypt during reindex. Vector
   search never needs body plaintext until tenant-scoped result hydration.
7. Use mixed-format readers and resumable per-row migration. A transaction
   verifies the newly persisted ciphertext before clearing its plaintext
   source. No failure is allowed to convert missing or unreadable ciphertext
   into empty memory.
8. Rotate ciphertext and blind-index keys together. A rotation is not complete
   while any row or index entry still requires the old key.

## Threat model

### Protected

With `KESTREL_DATA_KEY` available to the application, the design protects
saved-item and chunk bodies against:

- an attacker who obtains a stopped SQLite file, PostgreSQL dump, raw database
  snapshot, or storage volume but not the data key;
- a database reader or database administrator who can select columns but
  cannot read the application process or its key source;
- accidental plaintext disclosure through ordinary SQL inspection; and
- ciphertext substitution between agents, tables, columns, or row identifiers.
  AAD mismatch makes authentication fail.

AES-256-GCM authentication also detects ciphertext corruption. Per-agent key
derivation prevents agent A's key from decrypting agent B's body even on a
shared PostgreSQL database.

### Not protected

Application-layer encryption does not protect against:

- a compromised running Kestrel process, malicious code in that process, an
  operator who can read `KESTREL_DATA_KEY`, or an authorized API caller;
- plaintext intentionally sent to a configured embedding or LLM provider;
- unencrypted fields including saved-item names, summaries, tags, metadata,
  source references, timestamps, `content_hash`, file hashes, and vector
  embeddings;
- equality, token-frequency, corpus-size, and access-pattern leakage from
  content hashes, keyed token indexes, vectors, query timing, or row counts;
- deletion or rollback of an entire row by a database writer. AAD stops
  cross-row swaps, but a database attacker who rolls back both a row and its
  ciphertext can replay an older valid version unless a separate external
  integrity anchor detects it;
- plaintext remnants in old backups, replicas, SQLite WAL/free pages, or
  PostgreSQL MVCC/dead tuples. Migration is not certified secure erasure;
  operators must rotate or expire historical media and run backend-appropriate
  compaction/vacuum procedures; or
- plaintext logical exports. Signatures authenticate an export but do not
  encrypt it.

The blind index is a search compromise, not searchable encryption with zero
leakage. Its HMAC key prevents an offline dictionary attack without the data
key, but a running service necessarily exposes query access patterns.

## Cryptographic contract

### Key hierarchy

The SDK's `VALID_PURPOSES` gains `memory-content`. The only supported derivation
is:

```text
KESTREL_DATA_KEY
  -> SDK master-key normalization
  -> HKDF(agent_id, "kestrel-agent-master-v1")
  -> HKDF(no salt, "kestrel-memory-content-v1")
  -> 32-byte agent memory-content key
```

Core must call the SDK derivation rather than duplicate it. A key identifier is
the first 16 lowercase hex characters of
`SHA-256("kestrel:memory-content:key-id:v1\0" || derived_key)`. It is a
diagnostic/version selector, never key material.

The blind-index key is
`HMAC-SHA-256(memory_content_key,
"kestrel:memory-content:lexical-index:v1\0")`. Its version is
`v1:keyed:<key-id>`. A plaintext deployment may use
`v1:plaintext:<agent-fingerprint>` while no data key is configured, but enabling
encryption creates a new version and makes the old index ineligible for
encrypted-row coverage.

### Ciphertext and AAD

`MemoryContentCodec` uses the SDK `AEADCipher`, which writes the versioned
`KSAv2:` AES-256-GCM envelope. The database column is binary even though the
current envelope is ASCII; callers must not infer state from a prefix.

AAD is a canonical, length-prefixed byte encoding:

```text
"kestrel:memory-content:aad:v1\0"
  || field(agent_id UTF-8)
  || field(corpus: "saved_items" | "document_chunks")
  || field(column: "content")
  || field(row_id canonical UTF-8)
```

Each `field` is an unsigned 32-bit big-endian length followed by those bytes.
Saved-item UUIDs are lower-case canonical strings. Chunk identifiers are base-10
ASCII with no leading zero. Length-prefixing makes the encoding unambiguous.

AAD deliberately excludes mutable metadata, embedding profile, and key
identifier. Renaming an item, re-embedding a body, or rotating its key therefore
does not require unrelated ciphertext rewrites. Row identifiers and tenant
ownership may not change in place; import creates a destination identifier and
re-encrypts with destination AAD.

### Storage shape and invariants

Both tables gain:

```sql
content              TEXT NULL,       -- legacy plaintext migration source only
content_ciphertext   BLOB/BYTEA NULL,
content_encryption_version INTEGER NULL,
content_key_id       TEXT NULL,
lexical_index_version TEXT NULL
```

`BLOB` is the SQLite type and `BYTEA` is the PostgreSQL type. Version `1` means
the `KSAv2:` envelope plus the AAD contract above. The legal states are:

| State | Plaintext | Ciphertext/version/key id | Meaning |
|---|---:|---:|---|
| legacy plaintext | non-NULL | all NULL | readable migration source |
| encrypted | NULL | all non-NULL | normal protected row |
| staged in transaction | non-NULL | all non-NULL | migration/rotation verification only; never committed by a successful worker |
| corrupt/incomplete | any other combination | inconsistent | fail closed and report for repair |

Database checks should enforce the two durable legal states after compatibility
deployment. Mixed readers must classify by the complete marker tuple, not by a
ciphertext prefix and not by attempting to decode arbitrary plaintext.

Both current `content` columns are `NOT NULL`. PostgreSQL drops that constraint
after the mixed reader is deployed. SQLite must transactionally rebuild each
table with nullable `content`, copy every column, recreate indexes/triggers, run
`foreign_key_check` plus row/count checks, and swap only after verification.
That bounded compatibility migration runs before the service accepts traffic;
the later body backfill is the online, batched migration. It is not acceptable
to use `''` as a committed ciphertext sentinel because an overlooked legacy
reader would turn protected memory into a plausible empty body.

`saved_items.content_hash` retains its current SHA-256-of-plaintext
deduplication contract, and therefore still leaks equality. Removing that leak
is separate scope because identity packages and deduplication consume the
value.

## Tenant contract for document chunks

`document_chunks` gains a non-null `agent_id` and indexes on
`(agent_id, chunk_id)` and `(agent_id, file_hash)`. Every insert, select, vector
hydration, BM25 build, LIKE replacement, re-embedding query, and delete includes
that predicate. `file_owners` must independently prove that the same agent owns
the referenced file.

`document_chunk_owners` remains a compatibility ledger only during rollout.
The migration rules are:

1. One owner: copy that owner to `document_chunks.agent_id`.
2. Multiple owners: keep the original `chunk_id` for the lexicographically
   smallest owner and clone the full chunk/vector row for every other owner.
   Each clone gets its owner's `agent_id`, a new `chunk_id`, its own AAD, and
   its own lexical tokens. Derived chunk text is no longer shared.
3. No owner: do not guess from a global file hash and do not encrypt. Record an
   `unowned` failure requiring explicit operator assignment or deletion.
4. Once every caller and row uses `document_chunks.agent_id`, remove the
   compatibility ledger in a later schema cleanup.

The split is idempotent: migration progress records the source chunk and owner,
and a uniqueness constraint on that pair prevents duplicate clones after a
restart. `chunk_id` is an internal retrieval identifier; no supported export
contract promises that it survives tenant-splitting or import.

## Read and write contract

All logical content access goes through `MemoryContentCodec`.

### Writes

- With a data key configured, the store computes validation, plaintext hash,
  lexical tokens, and optional embedding in memory, encrypts with row AAD, and
  atomically inserts only `content_ciphertext` plus its markers. Plaintext
  `content` is `NULL`.
- Saved-item ids are allocated before encryption. For auto-increment chunk ids,
  one transaction inserts an empty, uncommitted legacy-state shell solely to
  obtain the id, encrypts with that id in AAD, replaces the shell with the
  encrypted state, verifies it, and commits. Other connections never observe
  the shell, and the source document chunk is never written to `content`.
- Without a data key configured, existing plaintext deployments remain
  supported and write the explicit legacy-plaintext state. Health reports that
  protection is disabled.
- A process without a key may not update, delete through a content-dependent
  path, reindex, export, or silently overwrite an encrypted row.
- Structured saved-item JSON is validated before encryption. Dedupe uses the
  existing tenant-scoped plaintext hash and never decrypts a different tenant's
  row.

### Reads

- The codec returns legacy plaintext only for the exact legacy state.
- The encrypted state requires a matching key identifier and successful AEAD
  authentication. A missing key raises `MemoryKeyUnavailableError`; a wrong
  key, marker mismatch, or authentication failure raises
  `MemoryContentUnreadableError`.
- List/get/search APIs must map those exceptions to the established
  non-success API error contract (service unavailable for a missing key;
  integrity/storage error for corrupt data). They must never return `None`,
  `[]`, an empty string, or a partial "success" for an affected operation.
- Before a corpus search, the store checks whether encrypted rows exist for the
  bound agent. This prevents a missing key from producing plausible empty
  recall merely because the encrypted candidates could not be hydrated.
- Logs and API errors identify the corpus and opaque row id, but never include
  plaintext, ciphertext, hashes, token digests, query text, or key material.

The same contract applies to SQLite and PostgreSQL. SQL dialect differences are
limited to binary types, additive-DDL mechanics, advisory locking, and
batch-selection syntax.

## Search and embedding contract

### Keyed lexical candidate index

Create a shared table:

```sql
CREATE TABLE memory_lexical_tokens (
    agent_id TEXT NOT NULL,
    corpus TEXT NOT NULL,
    row_id TEXT NOT NULL,
    index_version TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    term_frequency INTEGER NOT NULL,
    PRIMARY KEY (agent_id, corpus, row_id, index_version, token_hash)
);
CREATE INDEX memory_lexical_token_lookup
    ON memory_lexical_tokens
       (agent_id, corpus, index_version, token_hash, row_id);
```

A shared Unicode-normalizing tokenizer lowercases, extracts alphanumeric terms,
preserves negation, removes the agreed stopword set, and records unique token
HMACs plus per-document term frequency. Index writes and a row's
`lexical_index_version` marker commit atomically. The marker is durable coverage
evidence; an interrupted backfill leaves the row eligible for migration, not
invisible.

Saved-item lexical fallback combines ordinary SQL matching over intentionally
plaintext `name` and `summary` with blind-index content candidates. It decrypts
candidates and applies the canonical tokenizer before ranking/returning them.
Substring behavior inside the encrypted body is intentionally retired; exact
normalized tokens are the supported lexical contract.

RAG behavior is:

- when BM25 is installed, build the existing per-agent, in-process BM25 corpus
  from successfully decrypted chunks and discard it on tenant change, write,
  key change, or shutdown;
- when BM25 is unavailable, query the blind index, decrypt candidates, and
  rank verified token overlap; and
- never build one plaintext BM25 corpus across agents.

The database learns which keyed tokens repeat and which rows are candidates.
It does not receive plaintext query tokens or bodies.

### Embeddings and vectors

New-write order is validate/chunk -> embed plaintext -> encrypt -> atomic
persist. Provider routing and privacy policy are unchanged: if an operator
selects a cloud embedding route, the provider receives plaintext. Encryption at
rest is not a network privacy control.

Embeddings and `embedding_profile_id` stay outside the ciphertext. Vector search
operates as it does today, but every SQL/pgvector query and hydration is scoped
by `agent_id`. Only the top tenant-scoped rows are decrypted. A hydration
failure fails the search instead of dropping the row.

`EmbeddingReindexer` must use the codec. It stops and reports an unreadable row;
it must not count that row as unembeddable, stamp a new profile, or continue to
a misleading success.

## Online migration

### Schema and deployment order

1. Ship the sidecar columns, token/job tables, `MemoryContentCodec`, and
   mixed-format readers. Make legacy `content` nullable with the verified
   backend-specific DDL above. Do not enable encrypted writes yet.
2. Backfill authoritative `document_chunks.agent_id`; split multi-owner chunks
   and stop on unowned chunks.
3. Enable encrypted writes per store. New rows are protected while old rows
   remain readable.
4. Run the online content migrator in bounded keyset batches.
5. Require zero legacy/corrupt/unowned rows, then enable checks/non-null tenant
   constraints and retire direct `content` access.

PostgreSQL workers select tenant-scoped batches in primary-key order with
`FOR UPDATE SKIP LOCKED`; a per-agent advisory lock prevents two jobs from
rewriting the same corpus. Index creation uses the backend's non-blocking
facility where available. SQLite has one writer, uses keyset batches in short
transactions, and must yield between batches. It must checkpoint WAL only
through the existing database lifecycle; migration code must not delete WAL
files.

### Progress and row algorithm

`memory_encryption_jobs` records `job_id`, `agent_id`, `corpus`, `key_id`,
`status`, last primary key, scanned/encrypted/legacy/corrupt/unowned counts,
timestamps, and a redacted last-error code. `(agent_id, corpus, key_id)` has at
most one active job. Job state is observable through CLI status and protected
health diagnostics on both databases.

For each row:

1. Lock and re-read the row and ownership.
2. If it is already a valid encrypted state for the target key, verify decrypt
   and count it; this makes reruns idempotent.
3. If it is legacy plaintext, tokenize and encrypt it with target AAD.
4. Write ciphertext and markers while retaining plaintext in the transaction.
5. Read the persisted binary value back through the database driver, decrypt
   it, and compare its bytes to the source with a constant-time digest compare.
6. Write token rows and the coverage marker.
7. Only after steps 4-6 verify, set plaintext `content = NULL` and commit.
8. On cancellation, crash, constraint failure, wrong key, or verification
   failure, roll back the row transaction and record the failure outside that
   transaction. The previously readable representation remains.

Progress advances only past committed successes and explicit failures. Reruns
rescan failures. A dry-run classifies states without decrypting plaintext rows
or mutating the database. Completion means every in-scope row decrypts with the
target key and has current lexical coverage; row counts alone are insufficient.

Legacy detection is marker-based (`content IS NOT NULL` and all encryption
columns `NULL`), not prefix heuristics. A plaintext string beginning `KSAv2:`
therefore remains plaintext, while malformed marker combinations fail closed.

## Rotation and failure semantics

Memory-content rotation is added to `KeyRotationService` only through the
shared codec; the existing prefix-only encrypted-table walker is insufficient
because it does not supply row AAD or rotate lexical keys.

Per row, a transaction locks the old ciphertext, decrypts with the recorded old
key, writes new-key ciphertext, reads it back, verifies plaintext equality and
AAD, replaces blind-index tokens under the new index version, switches the row
markers, and commits. A failed verification rolls back to the old ciphertext.
Old token versions remain until no row references them.

Operators must retain both old and new keys until all saved-item and chunk rows
verify under the new key and all new blind-index coverage is complete.
`COMPLETED` is forbidden when any row is unreadable, corrupt, unowned, on the
old key, or missing current token coverage. After completion, a wrong or old key
causes an explicit startup/readiness degradation and read errors; there is no
plaintext fallback for an encrypted marker.

## Import, export, and backup

- **Logical saved-item export:** decrypt through the store and emit the existing
  logical plaintext field. Missing/wrong keys abort the entire export. Identity
  packages are signed but not confidential, so local plaintext packages remain
  `0600` sensitive artifacts and remote use requires an existing sealed/encrypted
  export path.
- **Logical import:** treat package content as plaintext input, validate it,
  allocate the destination row id, and call the normal destination store. Never
  persist source ciphertext because its key and AAD bind the source tenant/id.
- **RAG logical import/export:** when introduced, it follows the same
  decrypt/re-encrypt rule. This ADR does not invent a chunk export in the
  current identity package.
- **Raw SQLite/PostgreSQL backup:** copy/dump ciphertext and markers unchanged.
  Restore requires the matching data key. A snapshot taken during migration may
  be mixed-format and is supported by the mixed reader and resumable job.
- **Remote backup:** the existing outer backup encryption remains mandatory
  where already required. Inner row encryption is defense in depth, not a
  replacement for archive encryption.
- **Local backup/export:** current local export behavior remains unencrypted at
  the archive layer. The CLI must say so and require explicit operator intent;
  this ADR must not be cited as protection for plaintext logical exports.

## Observability, redaction, and recovery

Protected diagnostics expose per backend/agent/corpus counts for plaintext,
encrypted-current, encrypted-old-key, unindexed, unowned, and corrupt rows;
job state; last successful batch time; and a stable error code. Metrics use
bounded labels (`backend`, `corpus`, `state`) and do not put DIDs, row ids, key
ids, or exception text in labels.

Logs and audits may contain the agent id, corpus, opaque row id, job id, counts,
and error class. They must never contain body text, query text, ciphertext,
plaintext hashes, blind-token hashes, keys, or provider payloads.

Authentication failure never triggers automatic deletion, overwrite, empty
replacement, or plaintext fallback. The row remains untouched and a
`memory_content_corruption` reference records only corpus, row id, observed key
id, error code, and timestamps. Recovery is:

1. stop writes for the affected agent/corpus;
2. preserve a permission-restricted raw database snapshot;
3. verify the configured current/rotation keys;
4. retry an authenticated read with the correct key;
5. restore the row from a known-good encrypted backup or re-import trusted
   logical content; and
6. rebuild lexical tokens and embeddings, then clear the corruption reference
   through an audited repair command.

There is no "skip corrupt row and continue as success" mode for user-facing
recall.

## Phased delivery

Each boundary is independently reviewable and keeps public docs saying
"planned" until the final gate.

### Child A — shared crypto and schema substrate

- Add SDK `memory-content` purpose and AAD-capable test vectors.
- Add `MemoryContentCodec`, domain exceptions, content/token/job columns and
  tables, mixed-state classifier, PostgreSQL constraint migration, and verified
  SQLite table rebuild.
- No store enables encrypted writes.
- Regression:
  `uv run pytest -q tests/unit/test_memory_content_codec.py
  tests/unit/test_memory_content_schema.py
  tests/integration/test_storage_backend_parity.py`.

### Child B — authoritative RAG tenant ownership

- Add and backfill `document_chunks.agent_id`, deterministic multi-owner split,
  unowned fail-closed reporting, and tenant predicates to every current RAG,
  vector, reindex, and CLI path.
- Keep the compatibility owner ledger until its dedicated cleanup.
- Regression:
  `uv run pytest -q tests/unit/test_async_rag_store.py
  tests/unit/test_rag_store_pgvector.py
  tests/integration/test_storage_backend_parity.py`.

### Child C — saved-item encrypted vertical slice

- Route saved-item CRUD, dedupe hydration, IPFS pinning, identity import/export,
  reindex, vector hydration, and lexical fallback through the codec and blind
  index.
- Add missing/wrong-key, AAD swap, corrupt envelope, and search-parity tests.
- Regression:
  `uv run pytest -q tests/unit/test_saved_items_store.py
  tests/unit/test_saved_items.py tests/unit/test_saved_items_pgvector.py
  tests/unit/test_saved_items_sqla.py`.

### Child D — RAG encrypted vertical slice

- Encrypt chunk writes/hydration, build per-agent BM25 from decrypted bodies,
  add blind-index fallback, and update embedding reindex.
- Add SQLite and PostgreSQL parity for vector-first and lexical-only paths.
- Regression:
  `uv run pytest -q tests/unit/test_async_rag_store.py
  tests/unit/test_rag_store_pgvector.py
  tests/integration/test_saved_items_rag_smoke.py`.

### Child E — online migration and operations

- Add dry-run/status/migrate commands, resumable jobs, protected health counts,
  redacted metrics/logs, corruption references, and backend concurrency tests.
- Exercise cancellation after ciphertext write and before plaintext clear,
  rerun idempotence, malformed states, multi-worker PostgreSQL, and SQLite WAL.
- Regression:
  `uv run pytest -q tests/unit/test_memory_encryption_migration.py
  tests/integration/test_memory_encryption_migration_postgres.py`.

### Child F — rotation, backup, and export closure

- Extend rotation with AAD and blind-index rotation; gate completion on verified
  coverage. Route every logical export/import through the stores and document
  raw versus logical backup behavior.
- Remove remaining direct content-column consumers and add a static regression
  that allowlists only schema/migration/codec access.
- Regression:
  `uv run pytest -q tests/unit/test_key_rotation.py
  tests/unit/test_identity_exporter.py tests/unit/test_identity_importer.py
  tests/unit/test_embedding_reindex.py
  tests/integration/test_memory_encryption_backup_restore.py`.

## Live acceptance

After Children A-F, run the repository's
[live-agent dogfood flow](../testing/LIVE_AGENT_DOGFOODING.md) twice, once on
SQLite and once on PostgreSQL:

1. Start isolated agent Kite with a unique `KESTREL_DATA_KEY`.
2. Save a sentinel through `POST /api/saved-items`, ingest a document containing
   a different sentinel, and confirm both exact-token fallback and semantic
   retrieval return the logical plaintext.
3. Stop Kite. Inspect/dump both tables with a database-native client and assert
   neither sentinel occurs in `content`, `content_ciphertext`, a blind-index
   row, logs, or migration diagnostics; assert both bodies have complete
   encryption markers and tenant ids.
4. Restart without the key, then with a wrong key. List, search, reindex, export,
   and backup-dependent logical reads must return explicit non-success errors
   and readiness must report degraded memory encryption; none may report empty
   memory or partially successful migration.
5. Restore the correct key and confirm both memories return.
6. Interrupt migration after one committed batch, restart it, and prove counts,
   plaintext hashes, vectors, token coverage, and results are unchanged.
7. Rotate to a new key, interrupt once, resume, verify every row/index under the
   new key, retire the old key, and repeat step 2 retrieval.
8. As a second agent, attempt direct-id hydration of Kite's saved item and chunk;
   both SQLite and PostgreSQL paths must return no disclosure. Copy Kite
   ciphertext onto that agent's row and confirm AAD authentication fails.
9. Create raw and logical backups, restore them to a fresh target with the
   documented keys, and verify raw restore preserves ciphertext while logical
   import re-encrypts under destination ids/keys.

The final release gate remains:

```bash
uv run pytest -q tests/unit/test_saved_items_store.py tests/unit/test_async_rag_store.py
```

## Rejected alternatives

- **SQLCipher only:** not portable to PostgreSQL and does not protect a dump
  available to a database role. It may remain defense in depth.
- **Encrypt in the existing `content` column with prefix detection:** cannot
  distinguish arbitrary legacy plaintext safely and leaves no verified
  dual-copy migration window.
- **Decrypt-scan for every lexical query:** simple but makes latency and
  plaintext exposure proportional to the entire corpus. The keyed candidate
  index gives bounded, backend-neutral fallback.
- **Keep `document_chunk_owners` as the encryption identity:** a row with
  multiple owners has no single per-agent key or AAD. Tenant-owned chunk rows
  make the cryptographic and query boundaries agree.
- **Encrypt embeddings:** current SQLite and pgvector search require numeric
  vectors. Searchable encrypted-vector techniques are separate research and
  would change the retrieval engine.
- **Treat unreadable ciphertext as absent:** this creates plausible but false
  empty memory and violates the fail-closed requirement.
