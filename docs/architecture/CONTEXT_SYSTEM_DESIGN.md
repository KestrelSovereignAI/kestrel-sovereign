# Kestrel Context System — Assessment & Redesign

> A context window is not a buffer you fill until it overflows. It is a
> claim about what the model can currently *see and reason over*. Today
> Kestrel makes that claim in three uncoordinated ways and surfaces it in
> none.

> **Status (2026-05-19):** Design doc, pending review by Emma. No code in
> this branch — this is the reviewable surface for the `[EPIC]` that
> follows. Every claim about current behavior below cites a source file
> and line range; if a claim lacks evidence, treat it as a hypothesis to
> verify before implementation. Scope agreed with the Sovereign:
> **A + D + B this session-track; C is design-first follow-up.**

---

## Table of Contents

1. [Why](#why)
2. [Current Behavior (evidence-cited)](#current-behavior-evidence-cited)
3. [Assessment](#assessment)
4. [Redesign](#redesign)
   - [A — Measurement source of truth](#a--measurement-source-of-truth)
   - [B — Elastic token budget](#b--elastic-token-budget)
   - [C — Unify auto-prune with durable compression](#c--unify-auto-prune-with-durable-compression)
   - [D — Legible context: clickable breakdown popup](#d--legible-context-clickable-breakdown-popup)
5. [Non-goals](#non-goals)
6. [Review questions for Emma](#review-questions-for-emma)
7. [Source Files](#source-files)

---

## Why

A user looked at the chat footer pill — `● 34 msgs · 74%` with a
**Compress** button — and asked a reasonable question: *what does
"compress" even mean when context is already dynamically managed, and
isn't old history already condensed into summary nodes?*

The honest answer required reading three subsystems, because **the
context model is implicit** — it is not expressed in any single
component, surfaced in any UI, or reconciled across the mechanisms that
shape it. That is the disease. The pill, the Compress button, and the
`%` are symptoms of it.

Tortoise Doctrine framing: the symptom is "users (and agents) cannot
tell what the model actually sees or why"; the disease is "three
summarization mechanisms with no unifying model and no measurement
source of truth."

---

## Current Behavior (evidence-cited)

Kestrel shapes the context window with **three distinct mechanisms that
do not compose**:

### 1. Per-turn budget + pruning — automatic, transient, non-destructive

Every turn, `create_budget()` does static-percentage adaptive allocation
across `system / history / episodes / memories / rag`
(`kestrel_sovereign/agent/token_budget.py`). `build_full_context()`
measures each section as it assembles it and calls `budget.use(...)`
(`kestrel_sovereign/agent/context_builder.py:755-838`).
`ContextManager.build_context` then **auto-prunes the oldest formatted
messages** to fit the history slice, logging
`Auto-pruned N tokens from history`
(`kestrel_sovereign/agent/context_manager.py:513-524`).

This is **transient**: it only shapes *this turn's* request. Nothing is
deleted; full history stays in the DB; it is recomputed from scratch
next turn. Consequence: the request can never actually overflow — but
the drop is **silent and leaves no durable artifact**. Old turns simply
stop being sent.

### 2. Episodes / consolidation — automatic, additive, gated

`MemoryConsolidator` runs nightly and at thresholds
(`SESSION_EPISODE_THRESHOLD = 20`,
`kestrel_sovereign/storage/memory_consolidator.py:47-49`). It clusters
**emotionally significant** messages into narrative `MemoryEpisode`s and
writes them to the KG (`_create_episodes`, lines 123-206 — note the
"Only create episode if emotionally significant" gate at ~line 194).
`get_episode_context()` injects up to 5 episode summaries into the
**system** slice once `message_count >= 10`
(`context_builder.py:709-755`).

This is **additive and gated**: episodes are written *alongside*
originals, only for emotionally weighted clusters, and land in the
*system* slice. They do **not** shrink the verbatim history slice. The
conversation is therefore *not* generally "condensed into summary
nodes" — only the emotionally salient fraction, and only as extra
system-prompt context.

### 3. `!compress` — manual, LLM-driven, the only durable fold

`ConversationManager.compress_session()`
(`kestrel_sovereign/agent/conversation_manager.py:79-243`) takes all but
the recent N messages, sends them to the LLM for one summary, writes a
`[COMPRESSED CONTEXT - N messages summarized]` system marker into the
conversation, and marks the originals
`excluded_from_context: true` with `summarized_into` pointing at the
marker id (lines 195-228).

**Originals are preserved and bidirectionally linked.**
`excluded_from_context` is a metadata flag only — the row stays in
`conversation_history`. The history read skips it *only* when
`include_excluded=False`
(`kestrel_sovereign/storage/async_conversation_store.py:1102`);
`get_full_history_with_ids(include_excluded=True, include_stashed=True)`
returns it (line 1046). The marker carries `original_message_ids` +
`message_range`; each original carries `summarized_into`. A
`restore_excluded` tool / `!context restore` exists
(`kestrel_sovereign/features/context/feature.py:536-540`). So
compression is **fold, not delete** — durable, reversible,
agent-reachable.

### Sessions are soft, not hard

`session_id` is client-side (`pane.sessionId`, starts `null`,
`kestrel_sovereign/static/js/chat.js:435`). On the first message the
server resolves an effective id and the client adopts it
(`kestrel_sovereign/endpoints/agent.py:103-105, 1148-1159`).
A session is a **tag/filter** on rows in one shared per-agent
`conversation_history` table — not a separate store or hard boundary.
"New conversation" sets `pane.sessionId = null`; old rows persist and
the agent's `get_full_history` is cross-session. Issue #713 (see the
comment at `endpoints/agent.py:622-628`) is the symptom: passing
`session_id=None` summed the agent's *entire* cross-session memory and
falsely reported 100%. Unlike ChatGPT-style hard threads, Kestrel is
**one continuous agent memory with soft session tags**, plus
episodic/semantic layers. This is the correct design for an agent — but
it is invisible, which is why it confuses.

### Measurement is fragmented

- `/api/agent/context-status` measures **history only** (effective vs
  history budget) — `endpoints/agent.py:587-706`.
- `ToolContextManager.get_status()` reports **budget allocations only**
  for system/episodes/memories/rag, not actual usage —
  `kestrel_sovereign/agent/tool_context_manager.py:61-147`.
- Episode count is a heuristic: `episode_context.count("**") // 2`
  (`context_manager.py:438`).
- **Tool-schema tokens are never measured anywhere**, yet they are sent
  to the LLM every turn and are often a large slice.

No single component knows the true whole-window composition.

---

## Assessment

The primitives are right; the **unifying model and the measurement
source of truth are missing**. Four concrete defects follow from that:

1. **Three mechanisms, no reconciliation.** Transient prune (silent,
   lossy *to the model*), additive episodes (gated on emotion, system
   slice), durable `!compress` (manual fold). Nothing composes them; no
   component knows the net state.

2. **The silent-prune gap is a correctness hole, not just UX.**
   Auto-prune drops the oldest verbatim turns every turn with **no
   durable artifact**. Episodes only capture emotionally significant
   clusters. So a non-emotional but important fact in an old turn — a
   decision, a number, a constraint — is silently gone from the model's
   view unless a human happens to hit `!compress` in time. The machinery
   to fix this already exists (`compress_session`'s
   summarize→exclude→link); it simply is not wired into the prune path.

3. **Static budget throttles history for no reason.** The
   `15/40/20/10/15` split is fixed regardless of whether
   episodes/memories/RAG have *any* content. An empty conversation
   reserves ~45% of the window for sections with nothing in them while
   history is capped at 40% and prunes early. This is the most likely
   reason the pill reads high so easily — not "out of room," but "the
   budget will not let history use the room."

4. **The Compress affordance is mis-framed.** It fires at history-slice
   ≥70% implying overflow risk; auto-prune means overflow is impossible.
   The real value of `!compress` is the opposite: *salvage old turns
   into a durable summary before the silent prune drops them.* The
   label and trigger should say what it does.

The session model (soft sessions + continuous memory) is **sound** and
should not change — it just needs to be made legible.

---

## Redesign

Scope agreed with the Sovereign: **A + D + B** are coherent, contained,
and shippable on the epic's first track. **C** is the architectural
spine, design-first, separately reviewed, because it touches the hot
turn path.

### A — Measurement source of truth

**Problem:** measurement is fragmented (history-only / budget-only /
`**` heuristic / tool tokens never counted).

**Change:** add `ContextBuilder.measure_context_breakdown(query,
history, constitution, message_count)` — a read-only path that runs the
same assembly as `build_full_context` (no LLM call, no side effects) and
returns **real measured tokens per section**:

- `system` (sub-split: constitution / briefing / soul / state-of-mind
  where cheap to attribute)
- `tools` — count the serialized tool schemas the agent would send
  (currently unmeasured)
- `history` — verbatim, with `messages_kept_after_pruning`
- `episodes` — measured, not the `**` heuristic
- `rag` — labeled query-dependent / estimate
- per-section budgets, model, context limit, response reserve

Refactor `build_full_context()` to call this so the live path and the
reported path **cannot drift**. This is the prerequisite for B, C, and
D being honest.

**Acceptance:** section-sum + tools ≈ what the LLM call path actually
sends (unit test asserts within tolerance); `build_full_context` shares
the code path; `**` heuristic deleted.

### B — Elastic token budget

**Problem:** static `15/40/20/10/15` reserves budget for empty sections
and throttles history → premature prune / premature "Compress" pressure.

**Change:** after assembly measures each section, **reallocate unused
section budget to history** (or to whatever section has overflowing
demand), within a safe ceiling so the system slice can never be starved.
Localized to `token_budget.py` + the assembly call site. No persistence
change, no turn-path risk beyond allocation arithmetic.

**Acceptance:** with no episodes/RAG/memories present, history may use
materially more than 40% of post-reserve budget; system slice retains a
hard floor; existing budget tests updated to the elastic contract (not
bypassed — per the no-cop-outs rule, tests assert the documented elastic
behavior).

### C — Unify auto-prune with durable compression

> **Design-first follow-up. Not in the A/B/D track. Separately reviewed
> because it changes the hot turn path.**

**Problem:** auto-prune silently drops old turns with no durable
artifact (defect #2).

**Direction (to be specified in its own design before implementation):**
when the prune path is about to drop verbatim turns, it should emit the
**same durable artifact `compress_session` produces** —
summarize→exclude→`summarized_into` link — instead of dropping silently.
Consequences to work through in the C design:

- manual `!compress` becomes a *tuning knob* (force / keep-N / earlier),
  not a safety requirement;
- summarization in the turn path implies an LLM call on the hot path —
  needs an async/deferred or batched strategy so it does not stall the
  turn;
- interaction with episodes (avoid double-summarizing the same span);
- idempotency and the `restore_excluded` contract must still hold.

C's acceptance criteria are defined in the C design doc, not here.

### D — Legible context: clickable breakdown popup

**Problem:** the model is invisible; the pill's `%` is history-only and
misleading; there are no users to preserve familiarity for (greenfield)
so the number should be made *correct*, not kept compatible.

**Change:**

- The footer pill `%` becomes **honest whole-window utilization** — sum
  of all measured sections (system + tools + history + episodes + RAG)
  over `context_limit − response_reserve`, from A's breakdown. Not the
  history-only slice.
- `#context-status` becomes clickable
  (`kestrel_sovereign/static/js/chat.js:1294-1365`), opening
  `Modal.show()` (`kestrel_sovereign/static/js/ui.js:291-490`).
- The popup surfaces the **layered model** so it is finally legible:
  - **System** (constitution / briefing / soul / state-of-mind sub-rows)
  - **Tools** (the previously invisible schema cost)
  - **History** split into: *verbatim turns* (what `!compress` would
    fold) · *existing `[COMPRESSED CONTEXT]` markers* (already folded) ·
    *the tail auto-prune is currently dropping this turn* (the actual
    argument for compressing — silently lost otherwise)
  - **Episodes** (system slice, additive, not affected by `!compress`)
  - **RAG** (query-dependent, labeled estimate)
  - footer: model · context limit · response reserve · raw-vs-effective
    note · soft-session note ("this slice is session X; the agent also
    carries cross-session episodic + semantic memory")
- The **Compress** affordance is relabeled/retooltip'd to state what it
  actually does ("summarize older turns into a durable, restorable note
  before pruning drops them"), not implied overflow.

**Performance:** keep the cheap frequent footer poll cheap. If measuring
RAG/episodes on every poll is too expensive (to be measured, not
guessed), the poll returns everything except RAG and the popup makes one
on-demand call that adds it. Decided by a real measurement during
implementation.

**Acceptance:** pill `%` equals A's whole-window figure; pill is
clickable; popup renders the layered breakdown incl. tool tokens and the
"about-to-be-pruned" tail; idle/no-session shows the #713-safe shape;
frontend + endpoint tests cover idle + populated session.

---

## Non-goals

- Changing the soft-session / continuous-memory model — it is correct;
  D only makes it legible.
- Hard per-conversation isolation (ChatGPT-style) — explicitly rejected;
  Kestrel is an agent with continuous memory, not a threaded chatbot.
- Removing `!compress` — C may demote it to a tuning knob, but the
  durable-fold + `restore_excluded` contract stays.
- Any change to episode emotional-significance gating (separate concern,
  see `MEMORY_SYSTEM.md`).

---

## Review questions for Emma

1. Is the A/B/D vs C split right — is C genuinely safe to defer, or does
   the silent-prune correctness hole (defect #2) warrant pulling C
   forward despite the hot-path risk?
2. B's reallocation policy: should idle budget flow **only** to history,
   or to any over-demanded section (e.g. RAG-heavy turns)? What is the
   correct hard floor for the system slice?
3. C's hot-path LLM-call concern: is deferred/async summarization on
   prune acceptable, or must the durable artifact be synchronous to
   guarantee no fact is ever observed-then-lost within a single turn?
4. Does the layered popup taxonomy in D match how you want the context
   model explained to operators, or is there a canonical taxonomy to
   align to?

---

## Source Files

- `kestrel_sovereign/agent/token_budget.py` — budget allocation (B)
- `kestrel_sovereign/agent/context_builder.py:755-838` —
  `build_full_context`, target of A's refactor
- `kestrel_sovereign/agent/context_manager.py:513-524` — silent
  auto-prune (C)
- `kestrel_sovereign/agent/conversation_manager.py:79-243` —
  `compress_session`, the durable-fold machinery C reuses
- `kestrel_sovereign/storage/async_conversation_store.py:1046,1102` —
  `include_excluded` read path
- `kestrel_sovereign/storage/memory_consolidator.py:47-206` — episodes
- `kestrel_sovereign/features/context/feature.py:536-540` —
  `restore_excluded`
- `kestrel_sovereign/endpoints/agent.py:587-706` — `context-status`
  endpoint (D)
- `kestrel_sovereign/static/js/chat.js:1294-1365` — footer pill (D)
- `kestrel_sovereign/static/js/ui.js:291-490` — `Modal` (D)
- Related: `docs/architecture/MEMORY_SYSTEM.md` (episodes, sessions),
  `docs/TORTOISE_DOCTRINE.md`
