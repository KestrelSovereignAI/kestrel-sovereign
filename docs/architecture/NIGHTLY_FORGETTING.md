# Nightly Forgetting — Unified Cognition Maintenance

> **Status (2026-06-12): Aspirational — design-of-record, partially implemented.**
> This proposes consolidating Kestrel's scattered nightly-maintenance crons
> and the unbounded-cognition-table problem (#1674) under one orchestrated
> "sleep" pass, and adding the missing *deletion tier* to the existing
> importance-decay forgetting curve. The decay/archive engine it builds on is
> real and deployed (see [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md)); the deletion
> tier, episode participation, and the unified `sleep` cron are not yet built.
> The opt-in age-based `cognition_retention` cron (#1715) is a **stopgap** this
> design supersedes — see [Migration](#migration).

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

`storage/memory_retriever.py` + the nightly `MemoryConsolidator`:

- **Ebbinghaus decay** (`calculate_decay()`, used by consolidation):
  `strength = 0.5 ** (days_old / effective_half_life)`, where the base
  `half_life_days` (default 30) is extended by importance:
  `effective_half_life = half_life_days * (1.0 + importance * 2.0)` — important
  memories decay slower.
- **Rehearsal + load-bearing heat** (further half-life extension, multiplied
  in): `access_count` (`× (1 + log10(n+1)·0.5)`, modest rehearsal effect) and,
  more strongly, `applied_count` (`× (1 + log10(n+1)·1.0)` — memories that
  *demonstrably changed a decision*, set via `mark_applied` from
  reflection/pre-sleep hooks, #1326/#1342). The system rewards being *useful*
  over merely *familiar*. (The nightly consolidator + `access_count` only began
  running in production at #633; `applied_count` at #1326.)
- **Pinned criticals:** `decay_protected=True` → `calculate_decay()` returns
  1.0 unconditionally (Memory Agency feature).
- **Archival tier:** during consolidation, messages with `strength <
  DECAY_ARCHIVE_THRESHOLD` (10%) are flagged `archived: true` (+ `archived_at`,
  `archived_strength`). **Archived rows are NOT deleted** — they drop out of
  normal retrieval but stay recallable for compliance / export.

So the active→archival half of the three-tier model that production memory
systems converge on is built. **The third tier — deletion — is absent.** Rows
accumulate in `archived: true` forever.

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
strength = calculate_decay(memory)        # already importance + access aware
if strength < DECAY_ARCHIVE_THRESHOLD:    # 10% — tier 2, already implemented
    mark archived
if (archived
    and strength < DECAY_DELETE_THRESHOLD          # e.g. 2%, new, configurable
    and archived_at older than DELETE_GRACE_DAYS): # e.g. 90d, new, configurable
    hard-delete row (+ paired KG node + edges)
```

`decay_protected` rows are never eligible (strength is pinned at 1.0).
Frequently-recalled rows are never eligible (access heat keeps strength up).
Both thresholds and the grace window live under a single config section
(see [Config](#config)). The reusable storage primitive `purge_*_older_than`
introduced in #1715 stays — its **trigger and predicate** change from
"age < cutoff" to "archived ∧ decayed-below-delete ∧ past grace."

### 2. Episode participation in the decay model

Episodes must carry a decayable signal before they can be deleted by
importance. Options, in increasing cost:

- **(E1) Derive episode importance at consolidation time** from its
  constituents — e.g. `max`/`weighted-mean` of `key_message_ids`' importance,
  plus `emotional_arc` intensity — and stamp it on the episode row. Decay then
  runs on `(episode_importance, created_at, episode_access_count)`. *Smallest
  schema change; importance is a snapshot.*
- **(E2) Track episode access** — increment an `access_count` when an episode
  is retrieved/surfaced (Memories panel, retrieval), feeding the rehearsal
  effect so consulted episodes resist deletion. *Requires a write path on
  episode read.*
- **(E3) Full parity** — give episodes the same `archived`/`strength` lifecycle
  as messages. *Most work; cleanest long-term.*

Recommended: **E1 + E2** (importance at write, access on read) — enough to make
episode deletion importance-aware without a full lifecycle rewrite. Ties into
#1342 (pre-sleep attestation of which retrieved memories were load-bearing),
which is a natural source of the access/importance signal.

### 3. reflection_insights — the (a)/(b) deadlock dissolves

`reflection_insights` (kestrel-feature-reflection) has `actionable` but no
`consumed`/`ticketed` marker, which framed retention as a binary: (a) keep all
actionable forever (unbounded) vs (b) add a `consumed_at` column. The
decay-with-access model dissolves it: an **unconsumed actionable insight that
hasn't been retrieved in N days is stale regardless of its original
importance** — let it decay and become deletion-eligible like anything else.
This still lives in the reflection feature's own maintenance hook (core must
not import features); the *model* is shared, the *execution* is per-package.
`temporal_patterns` stays self-bounded (upsert) — leave it.

### 4. Scheduling: `sleep` as the nightly cron

Promote `sleep()` to a scheduled built-in (replacing the standalone
`memory_consolidate` and `cognition_retention` crons), running the full
forgetting pass nightly. Reflection and backup keep their own faster cadences
**unless** we decide otherwise — `sleep` calls them with `skip_*` so it never
double-runs what a dedicated cron owns:

- `sleep(skip_reflection=...)` — leave the 4h `reflect` cron as the frequent
  cadence; sleep's deep post-consolidation reflection stays optional/external.
- `sleep(skip_export=...)` — leave `backup_snapshot` as the frequent
  disaster-recovery snapshot; sleep optionally takes the nightly sovereignty
  checkpoint.

This makes `sleep` the single home for the *forgetting* family while respecting
the genuinely-different cadences of backup and reflection.

## Config

One section, replacing the #1715 `[retention.cognition]` age window:

```toml
[forgetting]
archive_threshold = 0.10      # tier 2 — already DECAY_ARCHIVE_THRESHOLD
delete_threshold  = 0.02      # tier 3 — new; below this AND archived AND past grace
delete_grace_days = 90        # min time in archived state before deletion
enabled           = false     # opt-in until the Sovereign turns it on
```

Deletion stays **opt-in/off by default** — episodes are autobiographical
long-term memory; the Sovereign enables forgetting deliberately.

## Migration

- #1715's `purge_episodes_older_than` primitive + KG-node/edge scrubbing are
  **kept** (they're the correct deletion mechanics).
- The age-based predicate + the standalone `cognition_retention` cron are
  **removed** in favor of the decay-aware predicate inside the `sleep` pass.
  Clean cutover (the cron is opt-in/off, so nothing in production depends on
  its current behavior).
- `memory_consolidate` cron is absorbed by the `sleep` cron.

## Open decisions

1. **Backup cadence** — keep frequent intra-day `backup_snapshot` (sleep skips
   export) or fold export into nightly `sleep` only?
2. **Reflection cadence** — keep the 4h `reflect` cron, or make reflection
   nightly-in-`sleep` only?
3. **Episode importance derivation** — E1+E2 (recommended) vs full E3 parity?
4. **Threshold/grace values** — `delete_threshold`, `delete_grace_days`
   defaults.

## Phasing

- **Phase 1** — Schedule `sleep` as the nightly cron; absorb `memory_consolidate`;
  remove `cognition_retention`. No behavior change to deletion yet (deletion
  `enabled=false`).
- **Phase 2** — Add the deletion tier (tier 3) for messages (they already have
  strength/archived); wire `[forgetting]` config.
- **Phase 3** — Episode participation (E1+E2) so episodes are importance-aware
  deletion-eligible.
- **Phase 4** — reflection feature adopts the shared model for
  `reflection_insights` / `reflection_sessions` in its own maintenance hook.

## References

- [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md) — the deployed decay/archive engine.
- #1674 — the unbounded-cognition-tables issue this reframes.
- #1715 — the opt-in age-based stopgap this supersedes.
- #633 — made the nightly consolidator + `access_count` actually run.
- #1342 — pre-sleep attestation of load-bearing memories (access/importance signal source).
