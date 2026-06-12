# Nightly Forgetting — Unified Cognition Maintenance

> **Status (2026-06-12): Active for P1–P3; Aspirational for P4.**
> This consolidates Kestrel's scattered nightly-maintenance crons and the
> unbounded-cognition-table problem (#1674) under one orchestrated "sleep"
> pass, and adds the missing *deletion tier* to the existing importance-decay
> forgetting curve. The decay/archive engine it builds on is real and deployed
> (see [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md)).
> **P1 (importance-aware deletion tier), P2 (relevance-based episode recall +
> access heat), and P3 (unified `sleep` cycle) are implemented.** Forgetting now
> lives in the single `MemorySystem.consolidate()` chokepoint — the manual
> `!memory consolidate` tool AND the nightly `sleep` cron both route through it,
> so there is ONE memory-maintenance path, not a cron per memory kind. The
> age-based `cognition_retention` (#1715) and the auto-seeded `memory_consolidate`
> + `reflect` crons are retired; reflection subscribes to the sleep cycle via its
> hook; backups stay on their own change-aware 4h cadence.
> **Still aspirational:** reflection-table participation (P4).

## TL;DR

1. `sleep()` (`agent/sleep.py`) is the *designed* nightly orchestrator
   (reflect → consolidate → forget → export) but **nothing schedules it**.
   Its steps were instead each given their own cron at their own cadence, and
   #1715 added a fifth. That accretion is the root problem.
2. Kestrel already has a sophisticated **importance-weighted decay** forgetting
   curve with an **archival tier** — but **no deletion tier**, and **episodes
   don't participate** in it at all.
3. The fix is *not* "add a retention cron." It is: **promote `sleep` to the
   nightly cron**, fold all forgetting into its consolidation phase, and
   **extend the decay curve with an importance- and access-aware deletion
   tier** (archived + decayed-below-delete-threshold + past grace → purge).
   Age-based hard-deletion (#1715) is the antipattern this replaces.

## Background: what's actually deployed

(Verified against code + [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md), 2026-06-12.)

### The forgetting curve already exists (for messages)

`storage/memory_retriever.py` `calculate_decay()` + the nightly
`MemoryConsolidator`:

- **Ebbinghaus decay** (`calculate_decay()`, used by consolidation):
  `strength = 0.5 ** (days_old / effective_half_life)`, where the base
  `half_life_days` (default 30, `DECAY_HALF_LIFE_DAYS`) is extended by
  importance: `effective_half_life = half_life_days * (1.0 + importance * 2.0)`
  — important memories decay slower.
- **Rehearsal + load-bearing heat** (further half-life extension, multiplied
  in): `access_count` (`× (1 + log10(n+1)·0.5)`, modest rehearsal effect) and,
  more strongly, `applied_count` (`× (1 + log10(n+1)·1.0)` — memories that
  *demonstrably changed a decision*, set via `mark_applied` from
  reflection/pre-sleep hooks, #1326/#1342). Rewards *useful* over *familiar*.
  (Consolidator + `access_count` only began running in prod at #633;
  `applied_count` at #1326.)
- **Pinned criticals:** `decay_protected=True` → `calculate_decay()` returns
  1.0 unconditionally (Memory Agency feature).
- **Archival tier:** during consolidation, messages with `strength <
  DECAY_ARCHIVE_THRESHOLD` (0.1 = 10%) are flagged `archived: true` (+
  `archived_at`, `archived_strength`). **Archived rows are NOT deleted** — they
  drop out of normal retrieval but stay recallable for compliance / export.

So the active→archival half of the three-tier model is built. **The third tier
— deletion — is absent.** Rows accumulate as `archived: true` forever.

### Episodes are outside the decay model

`memory_episodes` (consolidated narrative summaries, written by the
consolidator with a paired KG node, `node_id == episode.id`) has only
`created_at` / `emotional_arc` — **no importance, access_count, strength, or
archived fields.** Nothing decays or archives episodes today. This is why an
age-based sweep was tempting for them, and it is the substantive design gap
(see [Episode participation](#episode-participation-in-the-decay-model)).

### The scheduling is fragmented

`sleep()` orchestrates, in order, with skip-flags + graceful degradation:
pre-sleep reflection *(optional — `reflection_hook` is `None` unless the
external ReflectionFeature sets it)* → consolidation
(`memory_consolidator.run_consolidation()`) → post-consolidation reflection
*(optional)* → sovereignty export. **But no cron calls `sleep()`.** Instead:

| Cron | Cadence | Relationship to `sleep()` |
|---|---|---|
| `memory_consolidate` | nightly 04:00 | **= sleep step "consolidate"**, called standalone |
| `reflect` | every 4h | = sleep reflection steps, standalone + more frequent |
| `backup_snapshot` | every 4h | overlaps sleep "export" (a backup path) |
| `training_cycle` | nightly 03:00 | reflection-feature LoRA training |
| `cognition_retention` (#1715) | nightly 04:30 | **age-based episode purge — not in `sleep` at all** |
| `trash_retention` | every 6h | conversation-data purge (different domain — leave) |

The cadences differ *on purpose* (frequent backups/reflection vs. nightly
consolidation), which is why per-op crons accreted instead of one
parameterized orchestrator. That tension is real and the design must address
it rather than naively collapse everything to nightly.

## Design

### Principle: one forgetting pass, importance-aware deletion

All *forgetting* — decay-archival (have it) and deletion (the gap) — belongs in
one place: the consolidation phase of the nightly `sleep` pass. Deletion must
be **importance- and access-aware, not age-based**. Age-only pruning is the #1
production memory-system mistake: it GCs a six-month-old load-bearing decision
("we chose Postgres over Redis because X") while keeping a two-day-old
throwaway observation.

### 1. The deletion tier (tier 3)

Extend the existing decay machinery with a deletion threshold below the archive
threshold, plus a grace period:

```
strength = calculate_decay(memory)        # already importance + access + applied aware
if strength < DECAY_ARCHIVE_THRESHOLD:    # 0.1 — tier 2, already implemented
    mark archived
if (archived
    and strength < DELETE_THRESHOLD                # e.g. 0.02, new, configurable
    and archived_at older than DELETE_GRACE_DAYS): # e.g. 90d, new, configurable
    hard-delete row (+ paired KG node + edges)
```

`decay_protected` rows are never eligible (strength pinned at 1.0).
Frequently-recalled / load-bearing rows are never eligible (access + applied
heat keep strength up). Both thresholds and the grace window live under one
config section ([Config](#config)). The reusable `purge_*` storage primitive
from #1715 stays — its **trigger and predicate** change from "age < cutoff" to
decay-based.

**Episodes vs messages (as implemented in P1).** The pseudocode above is the
message lifecycle, which has an `archived` state. *Episodes have no `archived`
state* — nothing archives them today — so the implemented episode predicate
(`AsyncStorage.purge_decayed_episodes`) drops the `archived ∧` clause and
measures grace from `created_at` directly:

```
strength = calculate_decay(created_at, importance)   # P1: importance + age
                                                     # (access/applied/pin: P2)
if strength < delete_threshold and age(created_at) > delete_grace_days:
    hard-delete episode row (+ paired KG node + edges)
```

The deletion runs inside the nightly consolidation pass (the `memory_consolidate`
tool, which both the cron and `!memory consolidate` invoke), under the same
`ResourceLock.MEMORY` as consolidation, so it never races a concurrent run. It
is best-effort: a failure in the deletion tier never fails the consolidation it
rides on. The robust KG mechanics from #1715 are preserved verbatim — the paired
node is deleted before the row, the row is removed only if its node delete
succeeded (no orphans), and a non-positive cap is a safe no-op.

### 2. Episode participation in the decay model

Episodes must carry a decayable signal before they can be deleted by
importance. Options, in increasing cost:

- **(E1) Derive episode importance at consolidation time** from its
  constituents — the consolidator already computes `avg_importance` over a
  cluster's `key_message_ids` when deciding to create an episode — and stamp it
  on the episode row (additive `importance` column). Decay then runs on
  `(episode_importance, created_at)`. *Smallest schema change; importance is a
  snapshot. RECOMMENDED for P1.*
- **(E2) Track episode access** — increment an `access_count` when an episode
  is retrieved/surfaced (Memories panel, retrieval), feeding the rehearsal
  effect so consulted episodes resist deletion. *Requires a write path on
  episode read. P2.*
- **(E3) Full parity** — give episodes the same `archived`/`strength` lifecycle
  as messages. *Most work; cleanest long-term.*

Recommended: **E1 (P1) then E2 (P2)**. Ties into #1342 (pre-sleep attestation
of which retrieved memories were load-bearing) as the access/importance signal.

### 3. reflection_insights — the (a)/(b) deadlock dissolves

`reflection_insights` (kestrel-feature-reflection) has `actionable` but no
`consumed`/`ticketed` marker, which framed retention as binary: (a) keep all
actionable forever (unbounded) vs (b) add a `consumed_at` column. The
decay-with-access model dissolves it: an **unconsumed actionable insight that
hasn't been retrieved in N days is stale regardless of its original
importance** — let it decay and become deletion-eligible like anything else.
This lives in the reflection feature's own maintenance hook (core must not
import features); the *model* is shared, the *execution* is per-package.
`temporal_patterns` stays self-bounded (upsert) — leave it.

### 4. Scheduling: `sleep` as the nightly cron

Promote `sleep()` to a scheduled built-in (replacing the standalone
`memory_consolidate` and `cognition_retention` crons), running the full
forgetting pass nightly. Reflection and backup keep their own faster cadences
**unless** decided otherwise — `sleep` calls them with `skip_*` so it never
double-runs what a dedicated cron owns:

- `sleep(skip_reflection=...)` — leave the 4h `reflect` cron as the frequent
  cadence; sleep's deep post-consolidation reflection stays optional/external.
- `sleep(skip_export=...)` — leave `backup_snapshot` as the frequent
  disaster-recovery snapshot; sleep optionally takes the nightly checkpoint.

This makes `sleep` the single home for the *forgetting* family while respecting
the genuinely-different cadences of backup and reflection.

## Config

One section, replacing the #1715 `[retention.cognition]` age window. The archive
threshold (tier 2) is *not* a `[forgetting]` key — it stays the code constant
`DECAY_ARCHIVE_THRESHOLD` (0.1); `[forgetting]` configures only the new
deletion tier (tier 3):

```toml
[forgetting]
delete_threshold  = 0.02      # tier 3 — strength below this (and past grace) is eligible
delete_grace_days = 90        # minimum lifetime (from created_at) regardless of strength
enabled           = false     # opt-in until the Sovereign turns it on
```

`load_forgetting_config()` always returns a fully-populated dict — malformed or
non-positive values fall back to the compiled-in defaults with a warning rather
than silently disabling the rail, and a non-bool `enabled` fails safe to OFF.

Deletion stays **opt-in/off by default** — episodes are autobiographical
long-term memory; the Sovereign enables forgetting deliberately.

## Migration

- #1715's `purge_episodes_older_than` primitive + KG-node/edge scrubbing are
  **kept** (correct deletion mechanics — node deleted before row, row deleted
  only if node delete succeeded, cap guard, idempotent).
- The age-based predicate + the standalone `cognition_retention` cron are
  **removed** in favor of the decay-aware predicate inside the consolidation
  pass. Clean cutover (the cron is opt-in/off; nothing in prod depends on it).
- `memory_consolidate` + auto-seeded `reflect` crons are absorbed by the `sleep`
  cron (P3). They remain schedulable TOOLS, so the cutover removes only rows that
  exactly match the old core auto-seed (name+cron+args), preserving any
  user-customized schedule.

## Decisions (resolved 2026-06-12)

1. **Backup cadence** — KEEP the intra-day `backup_snapshot` (every 4h) for
   disaster recovery; `sleep` runs `skip_export=True` so it never double-
   snapshots. Backups are now **change-aware** (`SyncService.snapshot_if_changed`
   skips an unchanged DB) so idle agents stop re-dumping (notably the full S3
   upload). Explicit `!backup` / shutdown still call unconditional `force_snapshot`.
2. **Reflection cadence** — reflection rides the nightly `sleep` cycle via its
   `reflection_hook`; the auto-seeded 4h `reflect` cron is retired.
3. **Episode importance derivation** — E1 (P1) + E2 (P2) shipped.
4. **Threshold/grace values** — `delete_threshold=0.02`, `delete_grace_days=90`.

## Phasing

- **P1** — *(Done.)* Decay-aware episode deletion: episode importance (E1),
  deletion tier (`purge_decayed_episodes`), removed `cognition_retention` cron,
  `[forgetting]` config. Opt-in/off.
- **P2** — *(Done.)* Relevance-based episode recall (reusing the shared vector
  backend) + access tracking so consulted episodes resist deletion (ties #1342).
- **P3** — *(Done.)* Scheduling unification: forgetting relocated into the single
  `MemorySystem.consolidate()` chokepoint; `sleep` promoted to the nightly cron
  (skip_export); auto-seeded `memory_consolidate` + `reflect` crons retired
  (reflection subscribes via hook); change-aware backups; activity-gated sleep
  (idle agents skip the reflection pass). Relates #626 dynamic scheduler.
- **P4** — reflection feature adopts the shared model for `reflection_insights`
  / `reflection_sessions` in its own maintenance hook (kestrel-feature-reflection).

## References

- [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md) — the deployed decay/archive engine.
- #1674 — the unbounded-cognition-tables issue this reframes.
- #1715 — the opt-in age-based stopgap this supersedes.
- #633 — made the nightly consolidator + `access_count` actually run.
- #1326 / #1342 — `applied_count` / pre-sleep attestation of load-bearing memories.
- #626 — dynamic scheduler (relevant to P3 cadence).
