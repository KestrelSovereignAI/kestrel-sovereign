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

### Verified current seams

This design was checked against the current shared SQLite/PostgreSQL path, not
only the default SQLite deployment:

- `CORE_SCHEMA` declares both bodies as `TEXT NOT NULL`.
  `normalize_schema(..., "postgres")` changes `BLOB` embeddings to `BYTEA` and
  auto-increment syntax to `SERIAL`; it does not change either content column.
- `SavedItemsStore.save_item()` and `AsyncRAGStore.chunk_document()` bind the
  caller's plaintext directly into those columns. Saved-item identity
  export/import and `EmbeddingReindexer` also read the columns directly.
- RAG tenancy is currently a pair of `document_chunk_owners` and `file_owners`
  semi-joins. `document_chunks` and its SQLAlchemy mapping have no `agent_id`;
  a bound RAG store consequently cannot use the pgvector path and falls back to
  an ownership-scoped in-process scan.
- `AsyncStorage.create_backup_blob()` is a raw online SQLite snapshot.
  PostgreSQL backup is explicitly unsupported by that runtime method; a
  PostgreSQL raw backup in this ADR means an operator-managed `pg_dump`, not a
  new capability already present in Kestrel.

The characterization gate writes through both production stores, selects the
raw columns, and, for SQLite, closes the database and finds the sentinel bytes
in the database file. The dual-backend integration characterization runs the
same writers and raw selects against PostgreSQL when the CI service is
available. PostgreSQL is not simulated by merely asserting on a copied schema
string.

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
8. Persist a per-agent/corpus write epoch. Rotate ciphertext and blind-index
   keys behind that epoch's write fence, and do not complete while any row or
   index entry still requires the old key.

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
the full lowercase hex encoding of
`SHA-256("kestrel:memory-content:key-id:v1\0" || derived_key)`. It is an opaque
selector, never key material. A truncated diagnostic fingerprint is not
sufficient as the durable selector for a destructive rotation.

The blind-index key is
`HMAC-SHA-256(memory_content_key,
"kestrel:memory-content:lexical-index:v1\0")`. Its version is
`v1:keyed:<key-id>`. A plaintext deployment may use
`v1:plaintext:<agent-fingerprint>` while no data key is configured, but enabling
encryption creates a new version and makes the old index ineligible for
encrypted-row coverage. The agent fingerprint is the full lowercase hex
`SHA-256("kestrel:memory-content:plaintext-agent:v1\0" || field(agent_id))`.

The current SDK API derives only from the active `KESTREL_DATA_KEY`. Child A
must therefore land and pin an SDK release that also exposes the same
normalization/derivation over explicit master-key descriptors; core rotation
must not copy private `_derive_purpose_key` logic. A descriptor contains either
a raw key file or a passphrase key file plus its matching salt file. This
preserves the SDK's current raw-key versus salted/legacy-passphrase behavior
instead of treating every string as an unsalted SHA-256 passphrase.

Normal operation treats the existing `KESTREL_DATA_KEY` sources as a one-key
ring. Rotation additionally requires a restart-stable
`KESTREL_DATA_KEYRING_FILE`, a permission-restricted JSON file containing a
maximum of eight `keys` descriptors. Descriptors point to secret files; the
database and keyring manifest contain no master key.
`MemoryContentCodec` derives candidate per-agent keys, indexes them by the full
key id above, writes only with the id selected by `memory_content_state`, and
may read with any loaded id. `KESTREL_DATA_KEY` initializes state only when no
state exists; descriptor order never overrides a persisted write id. A
duplicate derived id with different key bytes is a fatal configuration error.

Rotation never depends on an in-memory `old_key` argument surviving a process
crash. Before its write fence moves, both old and new descriptors must be
loadable from the restart-stable keyring. A process that cannot resolve every
id named by an in-progress rotation starts with degraded readiness and rejects
protected reads and writes; it does not guess from descriptor order.

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
The saved-item row id is the exact validated database `id` encoded as UTF-8.
New store-created ids are lower-case UUID strings, but legacy and identity
imports already use namespaced non-UUID ids, so a UUID-only canonicalizer would
make real rows unreadable. Chunk identifiers are base-10 ASCII with no leading
zero. Length-prefixing makes the encoding unambiguous.

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

| Body state | Plaintext | Ciphertext/version/key id | Meaning |
|---|---:|---:|---|
| legacy plaintext | non-NULL | all three NULL | readable migration source |
| encrypted | NULL | all three non-NULL | normal protected row |
| staged in transaction | non-NULL | all three non-NULL | migration verification only |
| corrupt/incomplete | any other combination | inconsistent | fail closed and report for repair |

`lexical_index_version` is an independent coverage marker, not part of the body
state tuple. A plaintext row can legitimately have a plaintext lexical version,
and an encrypted row with a missing/stale lexical version is readable but not
migration-complete. Its token rows and marker must still change in the same
transaction.

The compatibility constraint must allow the three readable/staged body states:

```sql
CHECK (
  (content IS NOT NULL AND content_ciphertext IS NULL
                       AND content_encryption_version IS NULL
                       AND content_key_id IS NULL)
  OR
  (content_ciphertext IS NOT NULL
   AND content_encryption_version = 1
   AND content_key_id IS NOT NULL)
)
```

The second arm deliberately permits both `content IS NULL` and the
transaction-local staged copy. SQLite and PostgreSQL checks are immediate; a
two-state check would reject step 4 of the migration before the worker could
verify and clear plaintext. After migration and rotation finish, a later table
constraint adds `content IS NULL` to the encrypted arm and forbids committing a
dual-copy row. Mixed readers classify by the complete three-field body marker
tuple, not by a ciphertext prefix and not by attempting to decode arbitrary
plaintext.

Both current `content` columns are `NOT NULL`. PostgreSQL drops that constraint
after the mixed reader is deployed. SQLite must transactionally rebuild each
table with nullable `content`, derive the copy list from `PRAGMA table_xinfo`
so optional vector/profile columns are not lost, recreate the exact tracked
indexes/triggers from `sqlite_schema`, run `integrity_check`,
`foreign_key_check`, row-count and per-primary-key checks, and swap only after
verification. A missing optional column, index, trigger, or row aborts and
rolls back the rebuild.
That bounded compatibility migration runs before the service accepts traffic;
the later body backfill is the online, batched migration. It is not acceptable
to use `''` as a committed ciphertext sentinel because an overlooked legacy
reader would turn protected memory into a plausible empty body.

`saved_items.content_hash` retains its current SHA-256-of-plaintext
deduplication contract, and therefore still leaks equality. Removing that leak
is separate scope because identity packages and deduplication consume the
value.

Add `memory_content_state(agent_id, corpus, protection_required,
active_key_id, key_epoch, corpus_generation, rotation_id, updated_at)` with
primary key `(agent_id, corpus)` and a corpus check limited to these two table
names. `protection_required=1` requires a non-null active key id and is
monotonic: there is no automatic transition back to plaintext. The state is
created/locked in the same transaction as a corpus write. It keeps an empty
previously-encrypted corpus from silently accepting plaintext after a key is
lost, selects the write key during rotation, and supplies the generation used
to invalidate cross-process BM25 caches.

## Tenant contract for document chunks

`document_chunks` eventually gains a non-null `agent_id` and indexes on
`(agent_id, chunk_id)` and `(agent_id, file_hash)`. Every insert, select, vector
hydration, BM25 build, LIKE replacement, re-embedding query, and delete includes
that predicate. `file_owners` must independently prove that the same agent owns
the referenced file.

`document_chunk_owners` remains a compatibility ledger only during rollout.
For migration, an effective owner is present in both
`document_chunk_owners(chunk_id, agent_id)` and
`file_owners(file_hash, agent_id)`. A chunk-only owner, a file-only owner, or a
ledger row referencing a missing chunk/file is inconsistent ownership, not
authority that may be guessed or unioned. The migration rules are:

1. One owner: copy that owner to `document_chunks.agent_id`.
2. Multiple owners: keep the original `chunk_id` for the lexicographically
   smallest owner and clone the full chunk/vector row for every other owner.
   Each clone gets its owner's `agent_id`, a new `chunk_id`, its own AAD, and
   its own lexical tokens. Rewrite the compatibility ledger so the original
   and every clone each have exactly their one matching owner. Derived chunk
   text is no longer shared.
3. No owner: do not guess from a global file hash and do not encrypt. Record an
   `unowned` failure requiring explicit operator assignment or deletion.
4. Any inconsistent owner evidence is reported separately and blocks the
   non-null constraint until an operator repairs or deletes it.
5. Once every caller and row uses `document_chunks.agent_id`, remove the
   compatibility ledger in a later schema cleanup.

The split is idempotent through
`document_chunk_tenant_migrations(source_chunk_id, agent_id,
destination_chunk_id, status, error_code, updated_at)`. Its primary key is
`(source_chunk_id, agent_id)` and `destination_chunk_id` is unique. The clone,
ledger rewrite, and progress row commit together; a restart either sees the
whole split or none of it. `chunk_id` is an internal retrieval identifier; no
supported export contract promises that it survives tenant-splitting or
import.

During Child B, `agent_id` is nullable and writers dual-write it plus the
compatibility ledger. The non-null constraint and tenant-required SQLAlchemy
vector filter land only after zero unowned/inconsistent rows. This lets Child B
deploy before encryption without forcing the content migration into the same
release.

## Read and write contract

All logical content access goes through `MemoryContentCodec`.

### Writes

- With a data key configured, the store computes validation, plaintext hash,
  lexical tokens, and optional embedding in memory, encrypts with row AAD, and
  atomically inserts only `content_ciphertext` plus its markers. The body,
  legacy/vector embedding columns, token rows, coverage marker, tenant id, and
  corpus-generation increment commit in one database transaction. Plaintext
  `content` is `NULL`.
- Saved-item ids are allocated before encryption. For auto-increment chunk ids,
  one transaction inserts an empty, uncommitted legacy-state shell solely to
  obtain the id, encrypts with that id in AAD, replaces the shell with the
  encrypted state, verifies it, and commits. Other connections never observe
  the shell, and the source document chunk is never written to `content`.
- Without a data key configured, existing plaintext deployments remain
  supported only while the locked `memory_content_state` says protection has
  never been required. They write the explicit legacy-plaintext state and a
  deterministic plaintext lexical version; health reports that protection is
  disabled.
- Once `protection_required=1` or any protected/malformed row exists for that
  agent/corpus, a process without every required key may not insert, update,
  delete, reindex, export, migrate, rotate, or silently overwrite a row.
  Emptying a corpus does not reset that durable requirement.
- Structured saved-item JSON is validated before encryption. Dedupe uses the
  existing tenant-scoped plaintext hash and never decrypts a different tenant's
  row.
- `pin_to_ipfs=True` is an explicit outbound disclosure distinct from database
  encryption. The current `_pin_to_ipfs(..., encrypt=False)` publishes saved
  content as plaintext; Child C must either preserve that behavior with an
  explicit plaintext/public warning and consent or route it through an
  encrypted storage tier. Row encryption must never be presented as
  confidentiality for that IPFS object.

### Reads

- The codec returns legacy plaintext only for the exact legacy state.
- The encrypted state requires a matching key identifier and successful AEAD
  authentication. No loaded key matching the recorded id (including a
  wrongly-configured active key) raises `MemoryKeyUnavailableError`; a marker
  mismatch or authentication failure under bytes that derive the matching id
  raises `MemoryContentUnreadableError`.
- A dual-copy staged state is available only to the transaction-local
  migrator. If an ordinary reader ever observes it committed, it reports a
  malformed state rather than choosing either copy.
- HTTP list/get/search routes map a missing required key to status 503 and
  stable code `memory_key_unavailable`; malformed markers, a matching-key
  authentication failure, or an unverified staged row map to status 500 and
  `memory_content_unreadable`. CLI operations exit nonzero with the same stable
  code. They must never return `None`, `[]`, an empty string, or a partial
  "success" for an affected operation.
- Before a corpus search, the store checks whether encrypted rows exist for the
  bound agent and whether an unresolved corruption reference exists. This
  prevents a missing key or quarantined row from producing plausible empty or
  partial recall merely because candidates could not be hydrated.
- Logs and API errors identify the corpus and opaque row id, but never include
  plaintext, ciphertext, hashes, token digests, query text, or key material.

The same contract applies to SQLite and PostgreSQL. SQL dialect differences are
limited to binary types, additive-DDL mechanics, advisory locking, and
batch-selection syntax. Endpoint tests cover both exception mappings; store
tests cover list, direct-id get, lexical/vector search, delete, reindex,
logical export, migration, and rotation with missing, wrong, and corrupt keys.

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
preserves negation, and records unique token HMACs plus per-document term
frequency. Version 1 is exact: Unicode NFKC, `casefold()`, maximal runs for
which every code point is alphanumeric, discard runs shorter than two code
points, and no stopword removal (so `no`, `not`, and other negation terms
remain). Documents index every token within the stores' existing validated body
size. Queries are limited to 100 unique normalized tokens; the API rejects a
query that exceeds the cap instead of silently changing its meaning.

For a keyed row, `token_hash` is
`HMAC-SHA-256(blind_index_key, "kestrel:memory-content:token:v1\0" ||
field(corpus) || field(token))` encoded as lowercase hex. The corpus field
prevents equality linkage between saved items and chunks. For a plaintext
deployment, the same operation uses the deterministic non-secret key
`SHA-256("kestrel:memory-content:plaintext-index:v1\0" || field(agent_id))`;
this adds no confidentiality claim while the body itself is plaintext.

Index writes and a row's `lexical_index_version` marker commit atomically. The
marker is durable coverage evidence; an interrupted backfill leaves the row
eligible for migration, not invisible. Deletes remove token rows in the same
transaction. Child A extracts the HMAC/version/advisory-lock primitives already
implemented by
`kestrel_sovereign/storage/lexical_memory_index.py`; it must not create a
second independent key-normalization implementation. The existing conversation
table and conversation-specific tokenizer/search threshold remain compatible.

Saved-item lexical fallback combines ordinary SQL matching over intentionally
plaintext `name` and `summary` with blind-index content candidates. It decrypts
candidates and applies the canonical tokenizer before ranking/returning them.
Substring behavior inside the encrypted body is intentionally retired; exact
normalized tokens are the supported lexical contract. Content candidates are
the union of rows matching any unique query token, ranked by matched-token
count, summed term frequency, then stable row id; post-verification applies the
same any-token rule. The current whole-query substring behavior for
`name`/`summary` remains unchanged and its candidates are unioned before the
requested limit is applied.

RAG behavior is:

- when BM25 is installed, build the existing per-agent, in-process BM25 corpus
  only after every selected row decrypts successfully, and discard it on
  tenant change, write, key change, or shutdown;
- when BM25 is unavailable, query the blind index, decrypt candidates, and
  rank verified any-token overlap using the ordering above; and
- never build one plaintext BM25 corpus across agents.

The current `_search_by_like()` lowercases whitespace-split words and ORs
`%substring%` predicates over plaintext content. That path is removed for
protected rows; blind-index fallback uses the exact any-token contract, so its
intentional substring behavior change is covered separately from BM25 ranking.

The database learns which keyed tokens repeat and which rows are candidates.
It does not receive plaintext query tokens or bodies.

Every content write, delete, migration, token rebuild, or rotation increments
`memory_content_state.corpus_generation` in the same transaction. A BM25 cache
records `(agent_id, key_epoch, corpus_generation)` and checks it before every
use, so a write in another PostgreSQL process cannot leave a stale plaintext
index serving results. A decrypt or build failure clears the cache and fails
the whole search; "successfully decrypted chunks" never means "skip the bad
ones."

Child D routes `BM25Index.tokenize()` through this same versioned tokenizer.
That intentionally replaces its current ASCII-only regular expression so BM25
and the no-BM25 blind-index path agree for Unicode, short terms, and negation.
Search-parity tests freeze that change rather than assuming current `LIKE`,
BM25, and the new index already have identical semantics.

### Embeddings and vectors

New-write order is validate/chunk -> hash/tokenize -> embed -> encrypt ->
atomic persist. Saved items preserve the current embedding source exactly:
plaintext `summary` when present, otherwise the first 1000 content characters.
RAG embeds each plaintext chunk. An embedding-provider failure retains the
current text-only write behavior, but a database failure in either legacy or
`embedding_vec` persistence rolls back the content transaction rather than
committing a half-indexed protected row. Provider routing and privacy policy
are unchanged: if an operator selects a cloud embedding route, the provider
receives that plaintext embedding source. Encryption at rest is not a network
privacy control.

Embeddings and `embedding_profile_id` stay outside the ciphertext. Vector search
operates as it does today, but every SQL/pgvector query and hydration is scoped
by `agent_id`. Only the top tenant-scoped rows are decrypted. A hydration
failure fails the search instead of dropping the row.

`EmbeddingReindexer` must use the codec. It stops and reports an unreadable row;
it must not count that row as unembeddable, stamp a new profile, or continue to
a misleading success. It checks corpus/key state before taking the
summary-only saved-item shortcut, so a missing key cannot reindex an otherwise
protected row without proving access to its body. Child A also updates the
nullable-content SQLAlchemy mappings; Child B makes the document-chunk vector
spec require `agent_id`, so pgvector narrows before top-k rather than hydrating
and discarding foreign candidates afterward.

## Online migration

### Schema and deployment order

1. Ship the sidecar columns, token/state tables, `MemoryContentCodec`, and
   mixed-format readers. Make legacy `content` nullable, update both SQLAlchemy
   mappings, and add the compatibility check with the verified backend-specific
   DDL above. Do not enable encrypted writes yet.
2. Backfill authoritative `document_chunks.agent_id`; split multi-owner chunks
   and stop on unowned chunks.
3. Enable encrypted writes per store. New rows are protected while old rows
   remain readable.
4. Run the online content migrator in bounded keyset batches.
5. Require zero legacy/corrupt/unowned rows, then enable checks/non-null tenant
   constraints and retire direct `content` access.

PostgreSQL queue workers claim tenant-scoped work rows with
`FOR UPDATE SKIP LOCKED`; the uniqueness rules below prevent two jobs from
targeting the same corpus/key, while leases allow multiple workers on that one
job without processing the same body. Index creation uses
`CREATE INDEX CONCURRENTLY` outside a transaction and records success only
after the catalog proves the valid index exists. SQLite has one worker, uses
keyset seeding and short per-row transactions, and yields between rows. It must
checkpoint WAL only through the existing database lifecycle; migration code
must not delete WAL files or toggle journal mode.

### Progress and row algorithm

`memory_encryption_jobs` records `job_id`, `agent_id`, `corpus`, `key_id`,
`status`, the last durably seeded primary key, scanned/encrypted/legacy/corrupt/
unowned counts, timestamps, and a redacted last-error code. A backend-specific
partial unique index on `(agent_id, corpus)` where status is `pending` or
`running` permits at most one active job regardless of target key.

`memory_encryption_job_rows` records `(job_id, row_id)` as its primary key plus
`status` (`pending`, `leased`, `done`, `failed`), attempt count, random lease
token, lease expiry, stable error code, and timestamps. Keyset discovery first
inserts candidate ids and advances the job's seed cursor in the same
transaction. Workers then claim those durable ids; a crash expires a lease, a
failure remains explicitly retryable, and no high-water mark can jump over an
unrecorded failure. A resume retries expired/failed rows before seeding farther.
Completion requires zero non-`done` work rows and a fresh terminal scan proving
no legacy, staged, malformed, unowned, old-key, or stale-index rows. Job state
is observable through CLI status and protected health diagnostics on both
databases.

For each row:

1. Claim its durable work row, then lock and re-read the content row and
   ownership in a new database transaction.
2. If it is already a valid encrypted state for the target key, verify decrypt
   and count it; this makes reruns idempotent.
3. If it is legacy plaintext, tokenize and encrypt it with target AAD.
4. Write ciphertext and markers while retaining plaintext in the transaction.
5. Read the persisted binary value back through the database driver, decrypt
   it, and compare its bytes to the source with a constant-time digest compare.
6. Write token rows and the coverage marker.
7. Only after steps 4-6 verify, set plaintext `content = NULL`.
8. Mark the work row `done`, increment the corpus generation, and commit that
   one row transaction.
9. On cancellation, crash, constraint failure, missing/wrong key, or
   verification failure, roll back the content transaction. In a separate
   short transaction, mark a still-owned lease `failed`; if the process dies
   before that write, lease expiry makes it retryable. The previously readable
   representation remains.

Progress counts only committed successes and explicit durable failures. A
dry-run selects markers/ownership without selecting plaintext bodies, reports
encrypted rows as `encrypted-unverified` unless the operator explicitly asks
for authenticated verification, and never mutates the database. Completion
means every in-scope row decrypts with the target key and has current lexical
coverage; row counts alone are insufficient.

Legacy detection is marker-based (`content IS NOT NULL` and the three body
markers are `NULL`), not prefix heuristics. `lexical_index_version` may be
present independently. A plaintext string beginning `KSAv2:` therefore remains
plaintext, while malformed marker combinations fail closed.

## Rotation and failure semantics

Memory-content rotation is added to `KeyRotationService` only through the
shared codec; the existing prefix-only encrypted-table walker is insufficient
because it does not supply row AAD or rotate lexical keys.

The restart-stable keyring manifest has this versioned shape:

```json
{
  "version": 1,
  "keys": [
    {
      "key_file": "/run/secrets/kestrel_data_key_current",
      "salt_file": "/run/secrets/kestrel_data_key_current.salt"
    },
    {
      "key_file": "/run/secrets/kestrel_data_key_next",
      "salt_file": "/run/secrets/kestrel_data_key_next.salt"
    }
  ]
}
```

`salt_file` is omitted for a raw Fernet/AEAD-shaped key and is required for a
salted passphrase descriptor. A legacy env-only unsalted passphrase is accepted
only with explicit `"legacy_unsalted": true`, mutually exclusive with
`salt_file`; this makes recovery compatible without silently downgrading a new
descriptor. Relative paths, inline key values, unreadable files,
group/world-readable secret files, duplicate descriptors, more than eight
keys, and unknown fields fail startup. The manifest is read through the SDK
key-source API, and the rotation command names the target descriptor path
explicitly. After final verification, the operator makes that target the
ordinary `KESTREL_DATA_KEY` source before removing the old descriptor.

Before any row changes, rotation verifies both key descriptors against sampled
rows and creates/resumes durable job rows. It then takes the same
agent/corpus transaction fence used by every store write:

- PostgreSQL uses a shared lock-id helper and
  `pg_advisory_xact_lock`, then locks `memory_content_state FOR UPDATE`.
- SQLite starts the normal immediate write transaction and locks the state
  through its single-writer boundary.

Every content write reads the state after acquiring that fence. Rotation
atomically advances `active_key_id`, `key_epoch`, and `rotation_id` to the new
key; therefore an in-flight old-key write commits before the fence, or observes
the new epoch and writes new-key ciphertext after it. It cannot commit an old
row behind the sweep. A restart reloads the same two descriptors and database
epoch. If it cannot, writes fail closed until the operator restores the
manifest or explicitly resumes with equivalent protected secret files.

Per row, a transaction locks the old ciphertext, decrypts with the recorded old
key, writes new-key ciphertext, reads it back, verifies plaintext equality and
AAD, replaces blind-index tokens under the new index version, switches the row
markers, increments corpus generation, marks progress, and commits. A failed
verification rolls back to the old ciphertext. Row claims use the same
lease/retry protocol as content migration. Old token versions remain until no
row references them.

Operators must retain both old and new keys until all saved-item and chunk rows
verify under the new key and all new blind-index coverage is complete.
`COMPLETED` is forbidden when any row is unreadable, corrupt, unowned, on the
old key, or missing current token coverage. The final verification takes the
write fence again, proves the epoch still names the new key, and scans under
that fence before marking complete. Only then may the operator remove the old
descriptor. After completion, a missing, wrong, or old-only keyring causes an
explicit startup/readiness degradation and read errors; there is no plaintext
fallback for an encrypted marker.

Because this is a master-key rotation, `KeyRotationService` completes only
after its pre-existing conversation/file registry and both memory corpora pass
their own verification gates. The memory epoch does not permit retiring the old
master while a non-memory encrypted table still needs it.

## Import, export, and backup

- **Logical saved-item export:** decrypt through the store and emit the existing
  logical plaintext field. `IdentityExporter._get_saved_items()` must stop
  selecting `content` directly. Missing/wrong keys abort the entire export.
  Signed JSON packages are authentic but not confidential; local plaintext
  packages remain `0600` sensitive artifacts. The existing sealed identity
  package is confidential only after its recipient-bound sealing succeeds, and
  non-local publication must keep using that path.
- **Logical import:** treat package content as plaintext input, validate it,
  allocate the destination namespaced row id, recompute rather than trust
  `content_hash`, and call a store import method that preserves allowed
  metadata/timestamps while applying the normal codec/token transaction.
  `IdentityImporter._import_saved_items()` must stop using direct
  `INSERT OR REPLACE`. Never persist source ciphertext because its key and AAD
  bind the source tenant/id.
- **RAG logical import/export:** when introduced, it follows the same
  decrypt/re-encrypt rule. This ADR does not invent a chunk export in the
  current identity package.
- **Raw SQLite backup:** the current sovereignty backup uses SQLite's online
  backup API and archives ciphertext, markers, key/corpus state, corruption
  references, and any still-legacy rows unchanged. Restore requires the
  matching keyring. A snapshot taken during migration may be mixed-format and
  is supported by the mixed reader and resumable job.
- **Raw PostgreSQL backup:** `create_backup_blob()` remains unsupported for
  PostgreSQL. An operator `pg_dump`/restore has the same raw-state contract and
  is covered by the PostgreSQL acceptance fixture; this ADR does not claim the
  runtime gained a PostgreSQL backup implementation.
- **Remote sovereignty backup:** IPFS/Filecoin defaults to outer encryption,
  but the current tool accepts an explicit `encrypt=False`. Inner row
  encryption protects only migrated saved/chunk bodies, not the rest of that
  archive and not legacy rows. Child F must preserve the outer-encryption
  default, fail when it was requested but no outer key is available, and label
  an explicit unencrypted remote archive as unencrypted; row encryption is
  defense in depth, never a substitute.
- **Local backup/export:** current local sovereignty export forcibly disables
  outer encryption. The CLI must say so and require explicit operator intent;
  this ADR must not be cited as whole-archive protection or as protection for
  plaintext logical exports.

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
`memory_content_corruptions` reference records `agent_id`, corpus, row id,
observed key id, stable error code, first/last-seen timestamps, occurrence
count, resolution status/time, and repair audit id. Its primary key is
`(agent_id, corpus, row_id)`. It contains no body, ciphertext, plaintext hash,
token hash, query, exception text, or key. Marker/authentication failures
upsert the reference in a separate short transaction after leaving the content
row untouched. Missing key material is an availability/configuration state,
not corruption, and does not create this row.

“Quarantine” means the original row stays in place and cannot be returned,
reindexed, migrated past, exported, or silently skipped. A user-facing corpus
operation fails while an unresolved reference exists. Recovery is:

1. stop writes for the affected agent/corpus;
2. preserve a permission-restricted raw database snapshot;
3. verify the configured current/rotation keys;
4. retry an authenticated read with the correct key;
5. restore the row from a known-good encrypted backup or re-import trusted
   logical content; and
6. rebuild lexical tokens and embeddings, then clear the corruption reference
   through an audited repair command that first authenticates the repaired row
   and verifies current token/key coverage.

There is no "skip corrupt row and continue as success" mode for user-facing
recall.

## Phased delivery

Each boundary is independently reviewable and keeps public docs saying
"planned" until the final gate.

### Child A — shared crypto and schema substrate

- First release and pin the SDK `memory-content` purpose, explicit-master
  derivation/keyring API, and AAD-capable test vectors. Core work does not merge
  against an unreleased SDK API.
- Add `MemoryContentCodec`, domain exceptions, content/state/token columns and
  tables, mixed-state classifier, nullable SQLAlchemy mappings, PostgreSQL
  compatibility constraint, and verified SQLite table rebuild. Extract shared
  keyed-index primitives from the conversation implementation.
- Add mixed-format store readers and endpoint error mapping for both corpora,
  but leave all writers in plaintext mode.
- No store enables encrypted writes.
- Regression:
  `uv run pytest -q tests/unit/test_memory_content_codec.py
  tests/unit/test_memory_content_schema.py
  tests/unit/test_memory_content_failures.py
  tests/integration/test_storage_backend_parity.py`.

### Child B — authoritative RAG tenant ownership

- Add nullable `document_chunks.agent_id`, dual-write it with the compatibility
  ledger, reconcile chunk/file evidence, deterministically split multi-owner
  rows, and add unowned/inconsistent fail-closed reporting plus the idempotence
  table.
- After zero ownership failures, enforce non-null and add tenant predicates to
  every current RAG, SQLAlchemy/pgvector, reindex, stats, delete, and CLI path.
- Keep the compatibility owner ledger until its dedicated cleanup.
- Regression:
  `uv run pytest -q tests/unit/test_async_rag_store.py
  tests/unit/test_document_chunk_tenant_migration.py
  tests/unit/test_rag_store_pgvector.py
  tests/integration/test_storage_backend_parity.py`.

### Child C — saved-item encrypted vertical slice

- Route saved-item CRUD, dedupe hydration, IPFS pinning, identity import/export,
  reindex, vector hydration, and lexical fallback through the codec and blind
  index.
- Make the body/embedding/tokens/state-generation write one transaction and
  remove post-schema best-effort column fallbacks from this protected path.
- Add missing/wrong-key API mapping, AAD swap, corrupt envelope/quarantine,
  plaintext IPFS warning, import/export, and exact-token versus retired-
  substring search tests on SQLite and PostgreSQL.
- Regression:
  `uv run pytest -q tests/unit/test_saved_items_store.py
  tests/unit/test_saved_items.py tests/unit/test_saved_items_pgvector.py
  tests/unit/test_saved_items_sqla.py tests/unit/test_identity_exporter.py
  tests/unit/test_identity_importer.py`.

### Child D — RAG encrypted vertical slice

- Encrypt chunk writes/hydration, build per-agent BM25 from decrypted bodies,
  validate cache generation, add blind-index fallback, and update embedding
  reindex.
- Add SQLite and PostgreSQL parity for vector-first, BM25, lexical-only,
  cross-process invalidation, corrupt/missing-key, and direct-id paths.
- Regression:
  `uv run pytest -q tests/unit/test_async_rag_store.py
  tests/unit/test_rag_store_pgvector.py
  tests/integration/test_saved_items_rag_smoke.py`.

### Child E — online migration and operations

- Add dry-run/status/migrate commands, durable job/work-row leases, protected
  health counts, redacted metrics/logs, corruption references, and backend
  concurrency tests.
- Exercise cancellation after ciphertext write and before plaintext clear,
  crash after lease claim, lease expiry, rerun idempotence, malformed/staged
  states, multi-worker PostgreSQL, and SQLite WAL. Verify the last readable copy
  after every injected failure.
- Regression:
  `uv run pytest -q tests/unit/test_memory_encryption_migration.py
  tests/integration/test_memory_encryption_migration_postgres.py`.

### Child F — rotation, backup, and export closure

- Extend rotation with the SDK keyring, persisted epoch/write fence, AAD,
  leased progress, and blind-index rotation; gate completion/old-key retirement
  on fenced verified coverage.
- Verify all logical export/import paths already route through the stores,
  exercise SQLite online snapshot plus operator-managed PostgreSQL dump/restore,
  and label local/remote outer-encryption behavior exactly.
- Remove remaining direct content-column consumers and add a static regression
  that allowlists only schema/migration/codec access.
- Regression:
  `uv run pytest -q tests/unit/test_key_rotation.py
  tests/unit/test_memory_content_keyring.py
  tests/unit/test_identity_exporter.py tests/unit/test_identity_importer.py
  tests/unit/test_embedding_reindex.py
  tests/integration/test_memory_encryption_backup_restore.py`.

Dependency order is A -> B -> D and A -> C; C and B may proceed independently.
E requires C and D. F requires E. Each child deploys with either plaintext
writes or a complete single-corpus encrypted vertical slice; no child relies on
a later child merely to keep current reads, vector search, import/export, or
startup working. Public docs continue to say “planned” until F, the terminal
migration scan, and both live backends pass.

## Live acceptance

After Children A-F, run the repository's
[live-agent dogfood flow](../testing/LIVE_AGENT_DOGFOODING.md) twice, once on
SQLite and once on PostgreSQL:

1. Start isolated agent Kite with a unique `KESTREL_DATA_KEY`.
2. Save a sentinel through `POST /api/saved-items`, ingest a document containing
   a different sentinel, and confirm saved-item exact-token fallback, RAG BM25,
   RAG blind-index fallback with BM25 disabled, and semantic retrieval return
   the logical plaintext. Confirm a substring inside a larger encrypted-body
   token no longer matches.
3. Stop Kite. Inspect/dump both tables with a database-native client and assert
   neither sentinel occurs in `content`, `content_ciphertext`, a blind-index
   row, logs, or migration diagnostics; assert both bodies have complete
   encryption markers and tenant ids.
4. Restart without the key, then with a wrong key. List, direct get, both search
   modes, reindex, logical export/import overwrite, migrate, rotate, and delete
   must return the documented stable non-success errors and readiness must
   report degraded memory encryption; none may report empty memory, mutate a
   row, or report partially successful migration.
5. Restore the correct key and confirm both memories return.
6. Inject cancellation/crash after lease claim, ciphertext write, read-back,
   token replacement, and immediately before plaintext clear. Expire leases,
   resume, and prove the old readable copy survives each rollback and final
   counts, plaintext hashes, vectors, token coverage, and results are unchanged.
7. Install a two-key manifest, rotate to the new key while concurrent writes run
   in a second process, interrupt once, restart the process, resume, verify every
   row/index under the new key behind the write fence, retire the old key, and
   repeat step 2 retrieval. No concurrent row may retain the old key id.
8. As a second agent, attempt direct-id hydration of Kite's saved item and chunk;
   both SQLite and PostgreSQL paths must return no disclosure. Copy Kite
   ciphertext onto that agent's row and confirm AAD authentication fails.
9. Create SQLite raw/local/remote and identity logical exports. For PostgreSQL,
   take an operator `pg_dump` rather than calling the unsupported runtime backup
   method. Restore to fresh targets with the documented keyring and verify raw
   restore preserves ciphertext/state while logical import re-encrypts under
   destination ids/keys. Confirm local outer archives and an explicit remote
   `encrypt=False` archive are labelled unencrypted, while default remote and
   sealed identity exports fail rather than publish if outer sealing lacks a
   key.

The pre-implementation characterization gate is:

```bash
uv run pytest -q tests/unit/test_saved_items_store.py tests/unit/test_async_rag_store.py
uv run pytest -q tests/integration/test_storage_backend_parity.py \
  -k saved_item_and_rag_bodies_are_currently_plaintext_on_both_backends
```

Children C/D intentionally replace those plaintext assertions with raw-store
ciphertext assertions. The final release gate is the union of every child
regression command above plus the two live runs; the two original unit files
alone are not evidence that PostgreSQL, rotation, backup, or migration is safe.

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
