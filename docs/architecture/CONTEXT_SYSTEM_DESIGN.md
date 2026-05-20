# Kestrel Context System — Assessment & Redesign

> A context window is not a buffer you fill until it overflows. It is a
> claim about what the model can currently *see and reason over*. Today
> Kestrel makes that claim in three uncoordinated ways and surfaces it in
> none.

> **Status (2026-05-20, rev. after Emma's review):** Design doc, signed
> off by Emma on 2026-05-20 with conditions folded in below. No code in
> this branch — this is the reviewable surface for the `[EPIC]` that
> follows. Every claim about current behavior below cites a source file
> and line range; if a claim lacks evidence, treat it as a hypothesis to
> verify before implementation. Scope: **A + B + D is the first
> implementation track; C is the *next correctness track* — design-first,
> immediately next, and release-blocking for any claim that context
> correctness is fixed** (Emma's refinement; was previously framed as a
> looser follow-up).

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

Scope: **A + B + D** are the first implementation track — coherent,
contained, and shippable. **C** is the architectural spine and the
**next correctness track**: design-first, immediately next after the
A/B/D track lands, and **release-blocking for any claim that context
correctness is fixed**. C touches the hot turn path, so it gets its own
reviewed design doc before implementation — but it is *not* an optional
follow-up. (Refinement adopted from Emma's 2026-05-20 review.)

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
and throttles other sections → premature prune / premature "Compress"
pressure.

**Change (per Emma's reallocation policy):** budget is computed
**bottom-up against measured demand**, not top-down by fixed percentage.
Allocation order each turn:

1. **Mandatory floors first.** Compute the **measured token cost of
   mandatory system/governance content for this agent/model** —
   constitution, identity/bootstrap, response policy, feature routing,
   serialization overhead. This floor is **non-borrowable**: history,
   RAG, episodes, etc. may not crowd it out. If mandatory content cannot
   fit, that is a **hard-failure / degraded-mode condition**, surfaced
   loudly, not silently absorbed.
2. **Satisfy current turn + governance invariants.**
3. **Allocate eligible sections up to measured demand.** Sections that
   want less than their default share release the slack.
4. **Distribute slack to over-demanded sections by priority.** Default
   priority gives the slack to *history* because that is where the
   correctness hole hurts most — but a RAG-heavy turn, tool-heavy turn,
   or memory-recall turn legitimately deserves slack too. Priority is
   determined by turn intent.
5. **Expose the decision in the breakdown** (D) so an operator can see
   *which* section borrowed slack, from where, and why.

Optional system material (extras beyond the mandatory floor) can carry a
budget that *is* borrowable. The split between mandatory and optional
system content lives in `token_budget.py` + the system-prompt assembler.
Localized; no persistence change.

**Fail-closed degraded mode (Emma 2026-05-20 hardening):** when the
mandatory floor cannot fit, the system **fails closed** before prompt
assembly issues a model call under a false "normal" status. It must not
silently drop governance, tools, memories, or conversation to make the
call appear valid. The breakdown surface (D) reports the degraded-mode
condition explicitly.

**Acceptance:**
- Mandatory system content is measured per agent/model and treated as a
  non-borrowable floor. Below-floor is a hard-failure / degraded-mode
  condition, never silent absorption.
- Idle section budget flows to any over-demanded eligible section, not
  only history; history is the default beneficiary; turn-intent priority
  is documented.
- With no episodes/RAG/memories present, conversation may use materially
  more than the legacy 40% of post-reserve budget.
- Existing budget tests updated to the elastic contract (not bypassed —
  per the no-cop-outs rule, tests assert the documented elastic
  behavior).

### C — Unify auto-prune with durable compression

> **Next correctness track.** Design-first, immediately after A/B/D
> lands. **Release-blocking for any claim that context correctness is
> fixed.** Touches the hot turn path, so it gets its own reviewed design
> doc before implementation — but it is not optional. (Framing adopted
> from Emma's 2026-05-20 review.)

**Problem:** auto-prune silently drops old turns with no durable
artifact (defect #2). A non-emotional but important fact in an old turn
is silently gone from the model's view.

**Core invariant (Emma's formulation):**

> **No model-visible pruning without a synchronous durable artifact or
> lossless pointer. The summary may be async; the salvage record must be
> sync.**

That is: before anything is removed from the model-visible slice by the
prune path, the system must **synchronously commit a durable, lossless
prune record** — session id, message ids/ranges, token estimate, reason,
timestamp, and enough pointer/raw-span information to reconstruct or
summarize later. The LLM-driven summarization that converts the raw
salvage record into the normal
`summarize → excluded_from_context → summarized_into` shape may run
**async / deferred / batched** to keep the hot turn path cheap.

If async summarization fails or has not yet run, the UI surfaces a
**pending fold** or **failed fold** in the breakdown popup — never
"compression saved this" when only silent-prune happened (D honesty
invariant).

**Consequences to specify in C's own design doc:**

- manual `!compress` becomes a *tuning knob* (force / keep-N / earlier),
  not a safety requirement;
- async/deferred/batched summarization strategy (queue, worker, retry,
  back-pressure) — bounded by the sync salvage record so no fact is ever
  observed-then-lost mid-turn;
- interaction with episodes (avoid double-summarizing the same span);
- idempotency and the `restore_excluded` contract must still hold;
- UI states: durable-folded · pending-fold · failed-fold ·
  pointer-only-salvage.

C's full acceptance criteria are defined in the C design doc, not here.

### D — Legible context: clickable breakdown popup

**Problem:** the model is invisible; the pill's `%` is history-only and
misleading. Greenfield (no users to preserve familiarity for) → make the
number *correct*, not compatible.

**Change:**

- The footer pill `%` becomes **honest whole-window utilization** — sum
  of all measured sections (System+Tools+Conversation+Memories+RAG +
  Reserve/Overhead) over `context_limit − response_reserve`, from A's
  breakdown. Not the history-only slice.
- `#context-status` becomes clickable
  (`kestrel_sovereign/static/js/chat.js:1294-1365`), opening
  `Modal.show()` (`kestrel_sovereign/static/js/ui.js:291-490`).
- The popup uses a **canonical taxonomy that separates source ×
  visibility-state × budget-behavior** (refined from Emma's review — the
  earlier history-as-source / about-to-be-pruned-as-state cut conflated
  the axes):
  - **System / Governance** — mandatory system prompt, constitution,
    identity/bootstrap, response policy, feature routing/system extras.
    *Mandatory* (non-borrowable floor per B) shown distinctly from
    *optional*.
  - **Tools** — tool schemas exposed to the model, tool-call scaffolding,
    and any tool-result payloads injected outside normal history. Tool
    results that live in conversation rows are counted under
    Conversation but cross-attributed here.
  - **Conversation** — current user turn, recent verbatim turns, visible
    folded summaries, excluded/folded originals (linked by
    `summarized_into`), **pending-prune spans** (sync-salvaged but not
    yet summarized, per C), **out-of-window** spans (dropped by the
    legacy silent-prune path while C is still pending — surfaced
    honestly, not hidden).
  - **Memories** — episodes, reflections, KG facts, pinned memories,
    retrieved autobiographical context. Episodes counted first-class (no
    `**` heuristic per A).
  - **Retrieval / RAG** — document chunks, citations, search results,
    repo/doc context. Query-dependent, labeled estimate.
  - **Reserve / Overhead** — serialization overhead, model formatting
    overhead if measurable, unused budget, response safety reserve.
- The popup surfaces explicit **warnings/state labels** per section
  where applicable: `not counted`, `estimated`, `pending fold`,
  `failed fold`, `silently-pruned path still active` (until C lands).
- footer: model · context limit · response reserve · raw-vs-effective
  note · soft-session note ("this slice is session X; the agent also
  carries cross-session episodic + semantic memory") · which section
  borrowed slack from where this turn (B's transparency requirement).
- The **Compress** affordance is relabeled/retooltip'd to state what it
  actually does ("summarize older turns into a durable, restorable note
  before pruning drops them"), not implied overflow.

**UI honesty invariant (Emma):** **no UI element may imply "compression
saved this" when only the silent-prune path executed and no durable fold
exists.** Out-of-window spans are labeled out-of-window, not folded.

**Auto-detection invariant (Emma 2026-05-20 hardening):** while C is not
yet shipped, the breakdown surface must **automatically detect** that
the legacy silent-prune path is an active implementation state — not
merely offer the label as a possibility. If the running builder can
still drop out-of-window spans without a synchronous salvage record, the
UI surfaces `silently-pruned path still active` unconditionally for that
agent/model/turn, until C closes the gap.

**Performance:** keep the cheap frequent footer poll cheap. If measuring
RAG/episodes on every poll is too expensive (to be measured, not
guessed), the poll returns everything except RAG and the popup makes one
on-demand call that adds it. Decided by a real measurement during
implementation.

**Acceptance:** pill `%` equals A's whole-window figure; pill is
clickable; popup renders the canonical taxonomy above incl. tool tokens
and the pending-fold / out-of-window states; warning labels appear where
counts are estimated, missing, pending, or failed; no surface implies
durable compression where only silent prune happened; idle/no-session
shows the #713-safe shape; frontend + endpoint tests cover idle +
populated session + the warning-label states.

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

## Review record

**Emma — 2026-05-20 (kestrel-sovereign session `context-system-epic-review-20260520`):**
Conditional sign-off on A + B + D as the first track. Conditions and
refinements folded into the sections above:

1. **A/B/D vs C split.** Right *only if C is treated as the next
   correctness track, not a nice-to-have follow-up.* C is design-first,
   immediately next, and release-blocking for any claim that context
   correctness is fixed. → folded into [Redesign intro](#redesign) and
   [C](#c--unify-auto-prune-with-durable-compression).
2. **B reallocation policy.** Idle budget flows to *any* over-demanded
   eligible section by turn-intent priority (not only history). System
   floor is the **measured token cost of mandatory governance content
   for this agent/model**, non-borrowable; below-floor is a hard-failure
   / degraded-mode condition. → folded into
   [B](#b--elastic-token-budget).
3. **C hot-path concern.** Async summarization is acceptable; async
   durable capture is not. Invariant: *no model-visible pruning without
   a synchronous durable artifact or lossless pointer; the summary may
   be async, the salvage record must be sync.* → folded into
   [C](#c--unify-auto-prune-with-durable-compression).
4. **D popup taxonomy.** Refined to separate source / visibility-state /
   budget-behavior, with explicit warning labels (`not counted`,
   `estimated`, `pending fold`, `failed fold`,
   `silently-pruned path still active`) and the UI-honesty invariant
   that no surface may imply "compression saved this" when only
   silent-prune happened. → folded into [D](#d--legible-context-clickable-breakdown-popup).

**Ack received (Emma, 2026-05-20):** confirmed reading head `ac26c324`
(blob SHA `0dd8680e…` verified); the folded text matches intent on all
four points. Sub-tickets **#1308 / #1309 / #1310 are unblocked for
claim**. **#1311 remains design-first** and must not be treated as
implemented by A/B/D.

In the same ack, Emma added two hardening invariants — explicitly
"refinements, not blockers." Both folded in above:

- **Fail-closed degraded mode** (in [B](#b--elastic-token-budget)) —
  if the mandatory floor cannot fit, fail closed before prompt assembly
  issues a model call under a false "normal" status; never silently drop
  governance/tools/memories/conversation to make the call appear valid.
- **Auto-detection of the legacy silent-prune path** (in
  [D](#d--legible-context-clickable-breakdown-popup)) — while C is not
  yet shipped, the breakdown UI must *automatically* surface
  `silently-pruned path still active`, not merely offer the label.

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
