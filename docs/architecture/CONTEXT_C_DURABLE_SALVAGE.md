# Context C — Unify Auto-Prune With Durable Compression

> Auto-prune is a silent decision. A fact in turn 47 — a constraint,
> a decision, a number — was true to the model on turn 47 and gone
> from the model on turn 48, and nothing in the system noticed. The
> goal of C is to make pruning loud: every byte that leaves the
> model's view either survives as a durable summary or as a lossless
> pointer to where the original lives, and the operator can tell which.

> **Status (2026-05-21):** Design doc, pending review by Emma. No
> code in this branch — this is the reviewable surface for the C
> ticket of epic [#1307](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1307).
> The A/B/D first-track is merged ([#1339](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1339),
> [#1341](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1341),
> [#1343](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1343));
> the popup is unconditionally surfacing
> `silently-pruned path still active` because C has not shipped.
> **Implementation is a separate sub-ticket** filed after this design
> is acked. Every claim about current behavior cites a source file
> and line.

---

## Table of Contents

1. [Why C exists](#why-c-exists)
2. [Current behavior (evidence-cited)](#current-behavior-evidence-cited)
3. [Core invariant (Emma 2026-05-20)](#core-invariant-emma-2026-05-20)
4. [Architecture](#architecture)
   - [Sync salvage record](#sync-salvage-record)
   - [State machine for a salvaged span](#state-machine-for-a-salvaged-span)
   - [Async summarization](#async-summarization)
   - [Interaction with episodes](#interaction-with-episodes)
   - [Interaction with `!compress`](#interaction-with-compress)
   - [`restore_excluded` contract](#restore_excluded-contract)
5. [UI surfacing (D's badge slots get real signals)](#ui-surfacing)
6. [Performance budget](#performance-budget)
7. [Failure modes + ops](#failure-modes--ops)
8. [Pre-C history (what about the spans we have already lost?)](#pre-c-history)
9. [Acceptance criteria](#acceptance-criteria)
10. [Open questions for Emma](#open-questions-for-emma)
11. [Source files](#source-files)

---

## Why C exists

Three uncoordinated mechanisms shape the context window today — auto-prune
(transient + silent), additive episodes (emotionally gated), and manual
`!compress` (durable fold). Only `!compress` produces a durable
restorable artifact; only auto-prune actually runs on every turn that
needs it. The non-emotional but factually important content in old turns
falls into the gap between them: episodes don't capture it (they require
emotional salience), auto-prune drops it silently from the model view,
and `!compress` is manual. The system has had the right machinery to
salvage these turns since the original `compress_session` shipped — it
just isn't wired into the prune path.

**Tortoise framing.** Symptom: facts present in old turns are silently
absent from the model on later turns. Disease: the prune path emits no
durable artifact; episodes are gated on emotion; only manual `!compress`
folds. C unifies the three mechanisms onto one durable-salvage substrate.

Emma's 2026-05-20 review made C **release-blocking** for any claim that
"context correctness is fixed" (see [`CONTEXT_SYSTEM_DESIGN.md`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/design/context-system/docs/architecture/CONTEXT_SYSTEM_DESIGN.md)
"Review record" section). The popup's `silently-pruned path still active`
auto-detect flag, surfaced by D in
[`agent.py:get_context_status`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/kestrel_sovereign/endpoints/agent.py),
flips off the day C ships.

---

## Current behavior (evidence-cited)

### Two silent-prune sites in `ContextManager.build_context`

1. **Pre-trim** (introduced by B / #1309) — when
   `format_conversation_history` overshoots `max_tokens` because
   wrap-overhead is added after its per-message budget check
   (`context_builder.py:format_conversation_history` ~lines 462-468),
   `ElasticTokenBudget.use("history", …)` returns False without
   recording usage. `ContextManager.build_context` then drops oldest
   messages until the byte cost fits the effective ceiling
   (`context_manager.py:632-660`). The dropped messages still live in
   the database; they simply do not appear in the LLM-bound
   `formatted_history` list for this turn — or any subsequent turn
   that re-encounters the same overshoot.

2. **Post-budget prune** (legacy) — when `budget.total_used >
   budget.total_budget` after all sections settle,
   `ContextManager.build_context` drops oldest messages until the
   total fits (`context_manager.py:671-687`). Same in-memory mutation;
   same silent loss-from-model-view.

In both cases the per-row metadata stays untouched: no
`excluded_from_context` flag, no `summarized_into` link, no anything.
The `restore_excluded` tool ([`features/context/feature.py:537-593`](kestrel_sovereign/features/context/feature.py))
finds nothing to restore — because nothing was marked.

### What `compress_session` already does (the machinery we will reuse)

`ConversationManager.compress_session`
([`conversation_manager.py:79-243`](kestrel_sovereign/agent/conversation_manager.py))
already implements the durable-fold pattern:

- LLM-summarises the older messages.
- Writes a system row with `metadata.type = "compression"`,
  `original_message_ids`, `message_range: {first, last}`,
  `tokens_before/after`, `compressed_at`.
- Marks originals with
  `{excluded_from_context: True, excluded_at, excluded_reason,
  summarized_into: <marker_id>}` via `update_messages_metadata`.

`restore_excluded` flips `excluded_from_context` back to `False`. The
contract is non-destructive, reversible, agent-reachable.

The C design **does not invent a new compression mechanism**. It
extracts the "what to do when bytes leave the model view"
invariant from `compress_session` and applies it to the prune path
with one additional layer (sync salvage record) so the LLM call does
not block on a summarization round trip.

### Async work substrate already exists

`SignalDispatcher` exposes `enqueue_signal` with durable execution,
retry, back-pressure, causation, supervised tasks
([`signals/dispatcher.py:363-377`](kestrel_sovereign/signals/dispatcher.py)).
C's deferred-summarization can ride on it instead of duplicating that
machinery (Tortoise #2 from `WORKFLOWS_FEATURE_DESIGN.md` — *the
symptom is "every multi-stage epic reinvents orchestration"; the
disease is "no Workflow primitive"*).

### Episodes are orthogonal

`MemoryConsolidator._create_episode_from_messages`
([`storage/memory_consolidator.py:194-202`](kestrel_sovereign/storage/memory_consolidator.py))
only fires when an emotionally-significant cluster is detected. C
covers the *non-emotional* spans episodes never look at. Both can
coexist as long as the design prevents double-summarising the same
span (see [Interaction with episodes](#interaction-with-episodes)).

---

## Core invariant (Emma 2026-05-20)

> **No model-visible pruning without a synchronous durable artifact
> or lossless pointer. The summary may be async; the salvage record
> must be sync.**

Restated as code:

```
PRE: rows R = [r_first .. r_last] are about to leave the model-visible
     formatted_history for this turn.

ATOMIC, SYNCHRONOUS, BEFORE the LLM call is issued:
  1. Write a salvage record S binding {session_id, message_range,
     reason, token_estimate, salvaged_at}.
  2. Mark r in R as excluded_from_context=True with
     summarized_into = S.id and salvage_state = "pointer-only".

POST: the LLM call may now proceed. R is no longer model-visible. S
      is durable, restorable via restore_excluded(S), and detectable
      via the popup as "pointer-only-salvage".

LATER, ASYNCHRONOUSLY, ON A SEPARATE TASK:
  3. Generate a summary of R, attach it to S, flip
     salvage_state = "durable-folded".
  4. Or fail with a recorded reason; flip salvage_state = "failed-fold".
```

Two consequences worth surfacing:

- The LLM call **never** sees an uncovered byte loss again. Worst case
  the model loses verbatim access to R but the salvage record exists
  immediately, the agent's `restore_excluded` tool can pull R back the
  same turn if needed, and the popup labels the slice
  `pointer-only-salvage` until summarization runs.
- The summarization LLM round-trip is **not on the hot path**. The
  Anthropic prompt-cache prefix is unaffected; the user-facing turn
  latency is unchanged modulo a single small DB write.

Emma's hardening invariant from her ack: **the UI never implies
"compression saved this" when only a pointer-only-salvage exists.** D
already enforces this via the `silently-pruned path still active`
auto-detect; once C ships, the auto-detect can read salvage_state
per-span and surface accurate labels.

---

## Architecture

### Sync salvage record

A new metadata `type` on the existing `conversation_history` table —
reusing the same shape `compress_session` uses, with one added
discriminator:

```python
salvage_marker = {
    "role": "system",
    "content": "",                       # filled in by async summary; empty for pointer-only
    "metadata": {
        "type": "salvage",               # vs "compression" for manual !compress
        "salvage_state": "pointer-only", # "pending-summary" | "durable-folded" | "failed-fold"
        "salvage_reason": "auto-prune-pretrim" |
                          "auto-prune-postbudget" |
                          "manual-compress" |    # !compress takes this path too post-C
                          "session-end-fold",    # future hook
        "original_message_ids": [...],
        "message_range": {"first": <id>, "last": <id>},
        "salvaged_at": <iso8601>,
        "summarized_at": <iso8601 | null>,
        "token_estimate": <int>,         # cheap counter.count_messages at salvage time
        "model_at_salvage": <str>,       # which model's context window forced the prune
        "session_id": <str>,
        "summary_attempts": 0,
        "last_attempt_error": null,
    }
}
```

Originals get the existing exclusion metadata:

```python
{
    "excluded_from_context": True,
    "excluded_at": <iso8601>,
    "excluded_reason": "salvage:auto-prune-pretrim" |
                       "salvage:auto-prune-postbudget" | …,
    "summarized_into": str(salvage_marker_id),
}
```

**Why reuse `conversation_history`** instead of a new table:

- `restore_excluded` already walks this table and respects
  `excluded_from_context` + `summarized_into`. Zero contract drift.
- The salvage marker is queryable as a regular row, so existing
  history queries (`get_full_history_with_ids(include_excluded=True)`)
  surface it for free.
- `format_conversation_history` already skips `excluded_from_context`
  rows, so once a span is marked, it is gone from the next turn's
  model view by the same path that handles `compress_session`
  outputs today.

**Why not a separate `salvages` table:**
- Would duplicate the exclude/restore contract.
- `summarized_into` is already a foreign-key-ish pointer; making it
  point to a different table breaks `restore_excluded`'s lookup.

The added discriminator (`metadata.type == "salvage"` vs
`"compression"`) is what lets the popup distinguish the two — and lets
the async worker pick salvages to summarise without looking at manual
compression markers.

### State machine for a salvaged span

```
                  ┌───────────────────────────────────┐
   prune happens  │  pointer-only                     │
   ──────────────►│  (sync — salvage row written;     │
                  │   originals marked excluded)      │
                  └─────────────┬─────────────────────┘
                                │ async worker picks it up
                                ▼
                  ┌───────────────────────────────────┐
                  │  pending-summary                  │
                  │  (LLM round-trip in flight)       │
                  └──────────┬────────────┬───────────┘
                             │ success    │ exhausted retries
                             ▼            ▼
                  ┌─────────────────┐ ┌──────────────────┐
                  │ durable-folded  │ │ failed-fold      │
                  │ (content: …)    │ │ (last_attempt_   │
                  │                 │ │  error: …)       │
                  └─────────────────┘ └──────────────────┘
                       │                          │
                       └────► restore_excluded()  │
                                  reverts to      │
                                  the source rows │
                                                  │
                              UI must label "no summary;
                              span only recoverable via
                              restore_excluded"
```

`restore_excluded(salvage_marker_id)` works in **every** state. It
sets `excluded_from_context: False` on the originals and effectively
demotes the salvage marker (the existing implementation already
unwinds compression markers; the same code path applies). The salvage
marker row stays for audit.

### Async summarization

Rides on `SignalDispatcher`:

- Salvage write enqueues a `Signal(type="SALVAGE_SUMMARIZE",
  payload={salvage_marker_id, session_id, model_at_salvage})` via
  `enqueue_signal` ([`signals/dispatcher.py:363`](kestrel_sovereign/signals/dispatcher.py)).
- Handler: load originals via `original_message_ids`, run the same
  summarization prompt `compress_session` uses
  ([`conversation_manager.py:142-154`](kestrel_sovereign/agent/conversation_manager.py)),
  write the summary text to `salvage_marker.content`, flip
  `salvage_state` to `durable-folded`, increment `summary_attempts`,
  set `summarized_at`.
- Retry/back-pressure/durability come from the dispatcher — we do not
  reinvent them (Tortoise #1).
- Cap: `max_summary_attempts = 3`. On exhaustion → `failed-fold`,
  record `last_attempt_error`. UI surfaces the failure as a
  first-class badge.

**Batching.** The dispatcher already supports coalesced signals; if
multiple salvages fire in close succession (a long-overflow turn that
drops a big chunk all at once), the handler can batch the originals
into a single summary call to amortise the LLM cost. Batching is an
optimisation, not a contract requirement — the simple per-salvage
worker is correct.

**Model selection.** Summary uses the same model the salvage happened
under (`model_at_salvage`). The summary prompt is the existing
`compress_session` prompt verbatim — known-good, reviewed.

### Interaction with episodes

`MemoryConsolidator._create_episode_from_messages` looks for
emotionally-significant clusters of messages
([`memory_consolidator.py:208`](kestrel_sovereign/storage/memory_consolidator.py)).
After C lands, those clusters can include `excluded_from_context`
rows because the originals are still in the table. To prevent double
summarising, the consolidator:

1. Skips spans where every member row has `summarized_into` set —
   the salvage's summary already encodes the narrative.
2. May *use* the salvage summary as input to episode generation
   (episodes are emotional narratives; salvages are factual prose;
   the episode generator can quote the salvage summary when the
   underlying span lacks emotional weight on its own).

Concretely: episodes look for emotion; salvages capture everything
else. They are additive, not competing. The doc's "Memories —
episodes" row in D's popup keeps showing episode counts; the
"Conversation" row's `pointer-only-salvage` / `pending-fold` /
`durable-folded` badges show salvage state. Two orthogonal axes,
both attributable to specific spans.

### Interaction with `!compress`

`!compress` becomes a **tuning knob, not a safety mechanism.** Post-C,
the prune path always salvages, so manual compression is no longer
required to avoid silent loss. The command still works and still
provides:

- **Force**: salvage spans the auto-prune would not have touched yet
  (e.g. user wants to compress at 50% utilization to keep the prefix
  smaller for cache stability).
- **Keep-N**: same semantics as today — fold everything except the
  recent N turns.
- **Bypass-async**: synchronous summarisation when the user wants the
  summary text immediately (the only case the LLM round-trip is on
  the operator-blocking path; explicit).

Implementation: `compress_session` is refactored to call the same
salvage primitive as the prune path, with `salvage_state` flipping
straight to `pending-summary` (sync mode) or queued
(async mode, default). The metadata discriminator stays
`"compression"` so the UI can still distinguish operator-driven
folds from prune-driven folds when useful.

### `restore_excluded` contract

`restore_excluded` ([`features/context/feature.py:537-593`](kestrel_sovereign/features/context/feature.py))
**continues to work unchanged** for salvage markers. The function
already operates on `excluded_from_context` and `summarized_into`
metadata — neither field's shape changes. The user-facing tool
behavior is identical: pull excluded originals back into the
model-visible context.

Added: a per-span undo via `restore_excluded(target=<salvage_marker_id>)`.
Already supported by the existing implementation.

---

## UI surfacing

D / #1310 shipped badge slots ready for these states ([chat.js
`renderContextBreakdown`](kestrel_sovereign/static/js/chat.js)):

| Salvage state | Conversation-row badge | Source of label |
|---|---|---|
| (no salvages this session) | — | n/a |
| `pointer-only` | `pointer-only salvage` (cyan) | `breakdown.sections.history.salvages[].pointer_only_count` |
| `pending-summary` | `pending fold` (orange) | …`pending_count` |
| `durable-folded` | `folded` (green) | …`folded_count` |
| `failed-fold` | `failed fold — restore via !context restore` (red) | …`failed_count` |

`silently_pruned_path_active` flips to `False` in the endpoint
([`endpoints/agent.py:get_context_status`](kestrel_sovereign/endpoints/agent.py))
the day C ships — the auto-detect invariant Emma added in her ack
becomes "fact: no longer active" rather than the current
"fact: still active."

The popup's "Save older turns into a durable note (!compress)"
button label changes to **"Compress now (synchronous)"** since the
default async salvage already runs — the button is for operators
who want the summary text immediately rather than after the worker
catches up.

---

## Performance budget

The sync salvage write must be cheap enough to sit on the hot turn
path. Budget for one prune event:

- 1× INSERT into `conversation_history` (the salvage marker).
- 1× UPDATE on `conversation_history` rows by id-list (mark
  originals excluded). Already batched in `compress_session`
  ([`conversation_manager.py:225-228`](kestrel_sovereign/agent/conversation_manager.py)).
- 1× enqueue into the SignalDispatcher's queue.

Total expected wall time on SQLite local: < 5ms p99 for typical span
sizes. Anthropic prompt-cache prefix is unaffected because the
salvaged originals were already going to be missing from the next
turn's prompt — C only changes whether they leave silently or with a
record.

Async summarization is bounded by the dispatcher's per-handler
concurrency and rate-limits. The worker is non-blocking on the user;
worst case the popup shows `pending fold` until summarization
catches up.

---

## Failure modes + ops

| Failure | Behaviour | Operator visibility |
|---|---|---|
| Sync salvage INSERT raises | LLM call aborts; warning surfaced in `ContextResult.warnings`; degraded-mode style fail-closed (Emma 2026-05-20 hardening) | Popup shows banner "salvage write failed — see logs" |
| Async summary LLM call fails once | Retry per dispatcher policy | `pending fold` badge |
| Async summary exhausts retries | `salvage_state = failed-fold`; `last_attempt_error` recorded | `failed fold` badge with note that span is still recoverable via `!context restore` |
| Dispatcher backed up | Salvages queue normally; `pointer-only` count climbs | Popup shows `pointer-only salvage` count + a note |
| restore_excluded called on a `pending-summary` span | Originals come back; salvage marker stays with `salvage_state = "restored"` (new terminal state); pending summary task self-cancels by checking the state | Popup shows the restored count |
| Concurrent prune + restore for the same span | Salvage write is idempotent (UNIQUE on `(session_id, message_range)` recommended); race-loser updates the existing marker | No user-visible effect |

**Telemetry to add:** counters for salvages_created, salvages_summarised,
salvages_failed; histogram of pointer-only-to-folded latency. Surfaces in
the existing observability endpoint (separate sub-ticket).

---

## Pre-C history

A real question: what about the spans that the legacy silent-prune
already dropped before C ships? Those rows are still in
`conversation_history` (the prune was in-memory only — see "Current
behavior" above) but they are not marked excluded and they will get
re-pruned silently on subsequent turns until C lands.

Two options:

1. **No backfill.** Pre-C spans get pruned silently until the user's
   first post-C turn. From then on, every prune produces a salvage.
   Older spans stay in the DB and remain reachable via
   `get_full_history_with_ids(include_excluded=True)` — they are not
   actually lost. We just do not retro-salvage them.
2. **One-shot backfill on first post-C turn.** When build_context
   first encounters a session whose history would overflow under the
   new path, it salvages the *entire pre-C tail in one go* —
   pointer-only marker covering everything older than the budget
   allows. The async worker then summarises that one large span.

Recommended: **Option 1.** Backfill is a write to every active session
on first post-C turn and creates a worst-case-large summarisation job
on a path the user is actively waiting on. Option 1 has zero
migration cost and the worst case is "the popup shows
`silently-pruned path still active` until first post-C salvage fires
per session," which the operator can see clearly.

Option 2 is filed as a follow-up if the operator wants explicit
backfill control via a CLI command (`!context backfill-salvage`).

---

## Acceptance criteria

C implementation (separate sub-ticket) is accepted when:

- [ ] Sync salvage marker shape (`metadata.type == "salvage"`,
  `salvage_state` enum, `salvage_reason` enum, `original_message_ids`,
  `message_range`, `salvaged_at`, `token_estimate`,
  `model_at_salvage`, `session_id`, `summary_attempts`,
  `last_attempt_error`) is implemented with the existing
  `excluded_from_context` + `summarized_into` link.
- [ ] Both prune sites in `ContextManager.build_context`
  ([pre-trim](kestrel_sovereign/agent/context_manager.py),
  [post-budget](kestrel_sovereign/agent/context_manager.py))
  call the salvage primitive **synchronously before** any LLM call
  for the turn proceeds.
- [ ] LLM call **cannot proceed** if the sync salvage write fails;
  warning surfaced; `ContextResult.warnings` populated. (Fail closed,
  Emma's hardening invariant from B.)
- [ ] Async summarization rides on `SignalDispatcher.enqueue_signal`,
  with retry policy and per-attempt error recording.
- [ ] `restore_excluded` operates on salvage markers identically to
  compression markers (no behavior change for the operator).
- [ ] `compress_session` refactored to share the salvage primitive;
  manual `!compress` retains `force`/`keep-N` semantics; default
  becomes async (`pending-summary`) with an explicit
  `--sync`/`--bypass-async` flag for the operator-blocking case.
- [ ] `MemoryConsolidator` skips spans where all rows already have
  `summarized_into`; may consume salvage summaries as episode inputs.
- [ ] `get_context_status` endpoint flips
  `silently_pruned_path_active` to `False`; the breakdown
  `history.salvages` block reports per-state counts.
- [ ] D's popup reads `history.salvages` and renders the four state
  badges (`pointer-only` / `pending fold` / `folded` / `failed fold`)
  in addition to the existing pruning warning.
- [ ] Tests: sync record written before LLM call (asserted by a
  monitoring patch of the LLM service); async summary updates the
  record; failure exhaustion → `failed-fold`; concurrent prune +
  restore is idempotent; `restore_excluded` on salvage markers
  matches compression-marker behavior; no episode + salvage
  double-summarization of the same span.

The popup's `silently-pruned path still active` auto-detect flag
flipping off is **the release-gate signal**: when that flag goes
false in production, the C release gate from epic #1307 has lifted.

---

## Open questions for Emma

1. **Backfill default**: Option 1 (no backfill, only new prunes
   salvaged) or Option 2 (one-shot backfill on first post-C turn per
   session)? Recommended Option 1; ack or override.
2. **!compress default mode**: post-C should `!compress` default to
   async (`pending-summary`, queued) or sync (LLM round-trip on the
   operator's turn)? Async preserves the same hot-path budget as the
   prune-driven salvage; sync gives the operator immediate visibility
   of the summary text. Recommended async-by-default with explicit
   `--sync` opt-in; ack or override.
3. **Failed-fold UX**: when a span exhausts summarization retries,
   should the popup nudge the operator to run `!context restore`
   automatically, or just label the failure and let the operator
   decide? Recommended label-only — operator agency over their
   context window. Ack or override.
4. **Episode-as-input vs episode-skip**: when an emotionally-significant
   cluster overlaps a salvage span, should the episode generator
   consume the salvage summary as input (one consolidated narrative)
   or skip the span entirely (two parallel records)? Either works;
   the design assumes the former. Ack or override.
5. **Sub-ticket scope**: should the C implementation be one ticket or
   split into "sync salvage record + prune wiring" / "async worker +
   summarization" / "popup wiring + auto-detect flip"? Recommended
   one ticket because the invariant only holds end-to-end. Ack or
   override.

---

## Source files

- `kestrel_sovereign/agent/context_manager.py:632-687` — the two
  silent-prune sites C replaces (pre-trim + post-budget auto-prune).
- `kestrel_sovereign/agent/conversation_manager.py:79-243` —
  `compress_session`, the existing durable-fold machinery C extends.
- `kestrel_sovereign/storage/async_conversation_store.py:1102` —
  `excluded_from_context` filter the LLM path reads through.
- `kestrel_sovereign/features/context/feature.py:537-593` —
  `restore_excluded` contract C preserves.
- `kestrel_sovereign/signals/dispatcher.py:363-377` —
  `enqueue_signal`, the async substrate C rides on.
- `kestrel_sovereign/storage/memory_consolidator.py:194-202` —
  the emotional-significance gate episodes use; the orthogonal axis.
- `kestrel_sovereign/agent/context_builder.py` — `measure_context_breakdown`
  (#1308 source of truth) which will gain a `history.salvages` block.
- `kestrel_sovereign/static/js/chat.js` — `renderContextBreakdown`,
  D's popup with the badge slots already ready for the new states.
- `kestrel_sovereign/endpoints/agent.py:get_context_status` — where
  `silently_pruned_path_active` flips off the day C ships.
- Related: [`CONTEXT_SYSTEM_DESIGN.md`](CONTEXT_SYSTEM_DESIGN.md)
  ("Review record" + "C — Unify auto-prune with durable compression"
  sections — the parent design Emma acked).
