---
type: Architecture Spec
title: Incremental Backup — Page-Aligned Content-Addressed Deltas for Active Agents
description: '**Status (2026-06-27): Design spike for #1842.** Active agents re-upload
  the entire SQLite DB every backup cycle (full sqlite3.backup() dump) because the
  only "incremental" behavior shipped under #1674 is skip-if-unchanged, not delta.
  This evaluates WAL-shipping vs SQLite changesets vs page-aligned content-addressed
  chunking and recommends the last as the cross-tier solution.'
resource: /docs/architecture/INCREMENTAL_BACKUP.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-27T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# Incremental Backup — Page-Aligned Content-Addressed Deltas for Active Agents

**Status (2026-06-27): Design spike for [#1842](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1842).** No implementation yet. Parent epic [#1836](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1836) (retention/organization/isolation/decommission) is closed, which clears this follow-up's "do not start until #1836 lands" gate.

## Problem

Even with retention, an **active** agent re-uploads its *entire* database every backup cycle. The cron `0 */4 * * *` (`features/scheduler/feature.py:290`) calls `snapshot_if_changed()`, which either skips the whole pass (DB unchanged) or runs `force_snapshot()` — a **full** `sqlite3.backup()` dump of every byte to every configured tier (`storage/sync/targets.py:57-87`). The "incremental" behavior from #1674 is binary **skip-if-unchanged**, not a delta: any write to the DB means the next cycle ships the *whole* file again.

For a busy agent whose DB changes every cycle, this is the dominant ongoing cost driver. The #1836 audit measured the resulting footprint before retention: Lighthouse 32,881 files / 158 GB (8,626 copies of `kestrel_prime.db`), GCS 12,595 objects / 141 GB of never-pruned `<ts>.db`. Retention prunes the *history*; it does nothing for the *per-cycle* upload volume of an active agent.

## Current state (what ships today)

| Concern | Today | Citation |
|---|---|---|
| Snapshot unit | Full DB via `sqlite3.backup()` (raw-read fallback) | `storage/sync/targets.py:57-87` |
| Change detection | Whole-file `size + mtime_ns` of `db`/`-wal`/`-shm`; conservative (never under-reports) | `storage/sync/service.py:350-367` |
| Skip decision | `last_fingerprint == fingerprint` **and** all current targets already covered | `storage/sync/service.py:384-387` |
| Tiers | SOVEREIGN (IPFS, **decommissioned** per #1836), DELEGATED (Lighthouse), EXPEDIENT (GCS, S3) | `storage/sync/targets.py:30-39` |
| Path-keyed targets | GCS `…/snapshots/<ts>.db`, S3 `snapshots/<ts>.db` (+ `latest` pointer) | `gcs_target.py:106-118`, `s3_target.py:101-114` |
| Content-addressed targets | Lighthouse wraps whole DB in a one-block CAR → one CID; IPFS `add` → one CID | `lighthouse_target.py:263-268`, `car_builder.py:121` |
| Manifest | Per-backup, per-target: whole-file `content_hash`, CID/blob name, size, ts | `manifest_manager.py`, `gcs_target.py:124-131`, `lighthouse_target.py:134-145` |
| Retention | GFS per data class (WORKING_MEMORY: 14d all + weekly forever) | `storage/sync/retention.py:75-85` |
| WAL | **Enabled** (`PRAGMA journal_mode=WAL`). The scheduled sync snapshots via `sqlite3.backup()` **with an active WAL** — it materializes a consistent full-DB copy but does **not** truncate the live `-wal`. `wal_checkpoint(TRUNCATE)` runs only in the *separate* `AsyncStorage.create_backup_blob` export path, not the scheduled cron. | `storage/db/sqlite.py:82`, `storage/sync/targets.py:57-87`, `async_storage.py:761` |
| `sync_wal()` | No-op on every target ("new CID on every write, not recoverable without the matching DB") | `lighthouse_target.py:439-445`, `targets.py:140-151` |

**Key structural fact:** Lighthouse/IPFS addressing is whole-file `sha256 → CID` (`car_builder.py:121`). Any single changed byte yields an entirely new CID, so there is **zero block-level reuse** across snapshots today — even though the underlying stores (IPFS/Lighthouse) are block-addressed and *would* dedup identical blocks if we gave them block-aligned content.

## Requirements & constraints

1. **Cross-tier.** The delta scheme must serve both path-keyed (GCS/S3) and content-addressed (Lighthouse/IPFS) targets without a second, divergent code path per target.
2. **Restore-correct.** Any retained snapshot must reconstruct a byte-identical DB, verified against a whole-file hash, walking tiers in trust order (`targets.py:30-39`).
3. **Non-invasive to the write path.** The backup layer already runs out-of-band via `sqlite3.backup()`; the delta scheme should not require attaching session hooks to every live write connection.
4. **Atomic.** A partial upload must never leave a restorable-looking manifest that points at missing data.
5. **Retention-compatible.** Pruning must be reference-aware: a shared block may be referenced by many retained snapshots.
6. **Idle-cheap.** Must compose with the existing `snapshot_if_changed()` fingerprint skip — an idle agent still does nothing.

## Options evaluated

### Option A — WAL-shipping / frame streaming
Ship `-wal` frames between checkpoints; periodically compact to a full base.

**Verdict: rejected as the primary mechanism.**
- The scheduled path snapshots via `sqlite3.backup()` "while the database is in use with an active WAL" (`storage/sync/targets.py:57-87`): it materializes a fully-consistent full-DB copy (WAL folded in) and does **not** expose the live `-wal` frames as a shippable artifact — every `sync_wal()` is a deliberate no-op. (`wal_checkpoint(TRUNCATE)` runs only in the separate `create_backup_blob` export, `async_storage.py:761`.) Adopting WAL-shipping would mean restructuring the snapshot discipline to capture, ship, and truncate frames every cycle — new machinery, not a tweak.
- WAL frames are only replayable against the *exact* matching base page-set and are sensitive to SQLite version/format; a broken or reordered frame chain corrupts restore. That is a poor fit for long-lived, multi-tier archival.
- It does not fit content-addressed targets at all — the existing `sync_wal()` no-op comment already documents why ("new CID on every write, not recoverable without the matching DB"). Keeping a WAL-shipping path *and* a snapshot path doubles target complexity (violates requirement 1).
- Possible niche later: a high-frequency, path-keyed-only EXPEDIENT sub-tier. Out of scope here.

### Option B — SQLite session changesets (logical row deltas)
Use `sqlite3session` to capture row-level changesets and ship those.

**Verdict: rejected.**
- Requires attaching a session to every write connection (violates requirement 3) and still misses schema changes, `VACUUM`, and large-blob churn cleanly.
- Restore = apply an ordered changeset chain onto a base; a single missing/ordering error breaks the chain (fragile over months of 4h cycles).
- Logical deltas don't map onto block-addressed stores, so it can't unify the tiers.

### Option C — Page-aligned content-addressed chunking (recommended)
Treat the DB file as an ordered list of fixed-size chunks (aligned to the SQLite page size, or a small multiple). Hash each chunk; upload only chunks whose hash is new; store a per-snapshot **manifest** = the ordered list of chunk hashes/CIDs plus the whole-file hash.

**Why page-alignment is the right chunk boundary for SQLite specifically:** SQLite mutates the file by rewriting whole pages *in place* — it does not insert bytes and shift the tail. So fixed-size, page-aligned chunks capture the delta exactly, with none of the boundary-shift problems that force general-purpose backup tools (restic/borg) to use rolling-hash content-defined chunking. We get CDC's dedup benefit with fixed-size simplicity because the storage engine already aligns its writes to page boundaries.

## Recommended design

A single chunking layer beneath both target families:

```
DB file ──► [page-aligned chunker] ──► ordered [chunk_hash...] + whole_file_hash
                                          │
                          ┌───────────────┴────────────────┐
              path-keyed (GCS/S3)              content-addressed (Lighthouse/IPFS)
              put blocks/<hash> (skip if      add each chunk as a raw block (CID);
              HEAD/manifest says present);     snapshot = CAR DAG of chunk CIDs;
              manifest = JSON {order:[hash]}   unchanged chunk CIDs are byte-identical
                                               ⇒ store dedups at the block layer
```

- **Chunk size:** start at **64 KiB** (16 × the 4096-byte default page). Tunable. Trade-off: smaller chunks → finer deltas but larger manifests and more objects/blocks per snapshot; 64 KiB keeps the manifest for a ~500 MB DB to ~8k entries while still localizing typical per-cycle writes. Validate empirically in the spike against real agent DBs.
- **Manifest is written last,** after every chunk upload/pin succeeds (requirement 4). It records: chunk size, ordered chunk hashes (path-keyed) or chunk CIDs (content-addressed), whole-file `sha256`, base DB size, and a format tag (e.g. `chunked-v1/page64k`). Restore keys entirely off the manifest.
- **Reuse `CARBuilder`** (`storage/car_builder.py`) for the content-addressed path: it already produces CID-v1 raw blocks and CARs. A chunked snapshot becomes a small unixfs-style DAG (root → chunk CIDs) instead of today's single-block CAR. Unchanged chunk CIDs are identical across snapshots, so Lighthouse/IPFS transfer only *new* blocks — the "natural fit for IPFS/CAR DAGs" #1842 calls out, finally realized.
- **Path-keyed dedup** uses a content-hash keyspace `…/blocks/<sha256>` plus a per-snapshot `…/manifests/<ts>.json`. "Already present" is answered from the local target manifest first (cheap), HEAD as backstop.

### Restore
Walk tiers in trust order (`targets.py:30-39`, SOVEREIGN now `decommissioned`/skipped): fetch the chosen snapshot manifest → fetch only chunks not already on local disk → reassemble in page order → verify whole-file `sha256` against the manifest before swap-in. A failed verify falls through to the next tier.

### Compaction & retention (reference-aware GC)
There is **no separate "full base"** to compact — the chunk set *is* the base, and every snapshot manifest is a cheap pointer into shared chunks. Retention (`retention.py`) keeps pruning *manifests* by GFS. The new requirement: a chunk is deletable only when **no retained manifest references it** (reference count / mark-and-sweep over live manifests). This must run after manifest pruning, per target. For content-addressed tiers, unpinning an unreferenced CID lets the store GC it; for path-keyed tiers, delete `blocks/<hash>` with zero referencing manifests.

### Composition with the existing skip
Unchanged DB → `snapshot_if_changed()` still skips entirely (requirement 6). Changed DB → instead of `force_snapshot()` shipping the whole file, the chunker ships only changed chunks + a new manifest. The `_compute_db_fingerprint` skip remains the cheap first gate; chunk-diffing is the second-level economizer when the DB *did* change.

## Expected savings

SQLite localizes writes: a 4h cycle on an active agent typically dirties a small fraction of pages (recent conversation rows, a few index pages, freelist). Realistic changed-page ratios of 1–10 % imply **~90–99 % reduction in bytes shipped per cycle** for active agents, plus block-level dedup *across* agents sharing identical pages on content-addressed tiers. The spike must measure the real ratio on production-shaped DBs before committing the chunk size.

## Rollout (spike-first)

1. **Spike (this issue's deliverable):** a standalone script that chunks two real agent DB snapshots 4h apart, reports changed-chunk ratio at 16/64/256 KiB, and estimates upload savings + manifest overhead. No production code yet.
2. **Chunk store + manifest format** behind a feature flag, EXPEDIENT (GCS) first — lowest blast radius, path-keyed.
3. **Content-addressed path** via `CARBuilder` DAG on Lighthouse; verify block-level reuse against the live Lighthouse account.
4. **Reference-aware GC** wired into retention; prove no live manifest is ever orphaned.
5. **Restore drill** from each tier, byte-verified, before enabling by default.

## Risks & open questions

- **Manifest growth** for very large DBs at small chunk sizes — bounded by chunk-size choice; measure in the spike.
- **GC correctness** is the sharpest edge: a bug that prunes a still-referenced chunk silently breaks an old restore. Mitigation: GC is mark-and-sweep over *all* live manifests per target, dry-run logged before first destructive run, and never deletes a chunk referenced by the newest N manifests regardless.
- **Lighthouse CAR DAG support** — confirm Lighthouse serves/retrieves multi-block CARs and pins component CIDs (vs. only whole-file CARs). If not, fall back to path-keyed-style block objects on the DELEGATED tier.
- **Page size variance** — if any agent DB uses a non-default page size, align the chunk size to a multiple of *that* DB's `PRAGMA page_size` rather than assuming 4096.

## References

- Issue: [#1842](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1842) · Parent epic: [#1836](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1836) · #1674 (skip-if-unchanged)
- Snapshot: `kestrel_sovereign/storage/sync/targets.py`
- Change detection: `kestrel_sovereign/storage/sync/service.py`
- CID/CAR: `kestrel_sovereign/storage/car_builder.py`, `kestrel_sovereign/storage/sync/lighthouse_target.py`
- Retention: `kestrel_sovereign/storage/sync/retention.py`
- Scheduler: `kestrel_sovereign/features/scheduler/feature.py`
