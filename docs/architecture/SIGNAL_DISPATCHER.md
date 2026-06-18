---
type: Architecture Spec
title: Signal Dispatcher — Design
description: '**Status:** Draft v3 — second-pass review incorporated, ready for epic
  creation **Date:** 2026-05-01 **Author:** opus-4.7 (with @UncleSaurus, reviewed
  by sonnet-4.6)'
resource: /docs/architecture/SIGNAL_DISPATCHER.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# Signal Dispatcher — Design

**Status:** Draft v3 — second-pass review incorporated, ready for epic creation
**Date:** 2026-05-01
**Author:** opus-4.7 (with @UncleSaurus, reviewed by sonnet-4.6)

## Changelog

### v3.1 (this draft)
- **CONVERSATION has a single owner: the turn lifecycle.** The registry now FORBIDS COGNITION sources from declaring `CONVERSATION` in `resources` (v3 had it required). The shared turn lifecycle — used by HTTP user input AND by the dispatcher's COGNITION route — is the sole acquirer. Eliminates the v3 self-deadlock where dispatcher and turn lifecycle both tried to acquire `CONVERSATION`.
- **Lock order invariant clarified.** Dispatcher acquires the source's registered resources first (lex order), then the turn lifecycle acquires `CONVERSATION` inside. `CONVERSATION` is treated as the highest-lock-order acquisition system-wide.
- **Acceptance criterion 11 corrected.** UI side-channel renders system-initiated turns where `visibility != INTERNAL`. `INTERNAL` stays log-only, consistent with Concern #4.

### v3
- **ARTIFACT is handler-driven, not template-driven.** Existing `morning_signal`, `reflect`, `memory_consolidate` are feature workflows that fetch data, may make multiple internal calls, and return an artifact — not one-shot LLM completions. Registry now requires `artifact_handler: Callable[[Signal], Awaitable[ArtifactResult]]` for ARTIFACT sources. A `template_artifact_handler(path)` helper covers the simple template-render-and-complete case.
- **No separate turn lock. `CONVERSATION` is one resource lock among many.** Single ordered lock manager, lexicographic acquisition. Eliminates the deadlock surface from v2's two-lock split. COGNITION sources declare `CONVERSATION` in their resources; ARTIFACT/ACTION sources don't unless they actually touch conversation state.
- **Turn serialization covers streaming and non-streaming.** Acceptance criterion now names both `process_input` and `process_input_streaming`. The lock lives on the turn lifecycle, not the entry method.
- **Registry schema completed.** `coalescing_window` and `attention_policy` are now explicit fields (v2 referenced them in dispatcher behavior but didn't put them in the dataclass).
- **Tracked dispatch, no raw `create_task`.** New `enqueue_signal(signal) -> SignalHandle` for supervised fire-and-forget, using the agent's existing background task tracker. Raw `asyncio.create_task` at call sites is forbidden by code review.
- **Cycle rule made precise.** Append-before-check; `depth > TTL` (default TTL 5) → reject; same `(target_agent, source)` pair appearing earlier in chain → reject; per-source `allow_self_loops` opt-out for legitimate self-emit cases.

### v2
- Three modes (ACTION / ARTIFACT / COGNITION) replacing the v1 two-mode split.
- Source Registry promoted to v1 boundary.
- `dispatch_signal` async, returns `SignalResult`.
- Envelope grew: `target_agent`, `session_id`, `caller`, `visibility`, full `causation_chain`.
- `signal_log` privacy rules first-class.
- ACTION resource locks (v3 generalizes this further).
- Quiet-hours generalized to per-source attention policy.

## Problem

The bird has at least five mechanisms that legitimately wake it up or trigger work. They share nothing.

| Source | Lives at | What it does today | What it does NOT do |
|---|---|---|---|
| Heartbeat | [heartbeat.py:154](../../kestrel_sovereign/heartbeat.py#L154) | Calls `process_input(prompt)` | Check if a turn is in flight |
| Scheduled cron | [features/scheduler/runner.py:68](../../kestrel_sovereign/features/scheduler/runner.py#L68) | Invokes a feature tool | Trigger cognition |
| A2A task complete | [a2a/task_manager.py:89](../../kestrel_sovereign/a2a/task_manager.py#L89) | Queues an SSE notification | Re-enter the agent |
| Approval request | [features/security/approval_queue.py:81](../../kestrel_sovereign/features/security/approval_queue.py#L81) | Both a hook AND an event | Pick one |
| Webhook (Stripe) | [features/wallet/onramp/webhook_handler.py:80](../../kestrel_sovereign/features/wallet/onramp/webhook_handler.py#L80) | Stores + fires callback | Wake the bird |

Each was added at a different time by a different author. Same accretion shape as the model-selection postmortem in [CLAUDE.md](../../CLAUDE.md). With one signal, the abstraction is invisible. With five, overdue.

The existing `emit_event` / `add_event_listener` ([agent/event_manager.py:15-33](../../kestrel_sovereign/agent/event_manager.py#L15-L33)) looks like a bus but is **agent → UI only**. The missing direction — world → agent — was never built.

## Decisions (made)

1. **Signals carry their mode as a parameter.** ACTION, ARTIFACT, or COGNITION (defined below).
2. **The Source Registry is the v1 boundary.** Sources cannot dispatch without registering first.
3. **The dispatcher is async and returns observable results.** Tracked enqueue exists for fire-and-forget.
4. **Approval is not a signal.** It stays in the hook system as a gate release on a paused turn.
5. **`CONVERSATION` is a resource lock, not a special turn lock.** Single ordered lock manager.

## The three modes

| Mode | Definition | LLM involvement | Enters conversation history | Implementation contract | Examples |
|---|---|---|---|---|---|
| **ACTION** | Deterministic side effect | None | No | `handler(payload)` | `trash_retention`, `backup_snapshot`, `signal_dispatch` |
| **ARTIFACT** | Produces an artifact (text, JSON, etc.); does not enter a turn. May internally fetch data, mutate feature state, make one or more LLM calls. | Maybe (handler decides) | No | `artifact_handler(signal) -> ArtifactResult` | `morning_signal`, `reflect`, `memory_consolidate` |
| **COGNITION** | Full agent turn — enters conversation history, may invoke tools, may emit further signals | Yes | Yes | Renders `prompt_template` → `process_input` (or streaming variant) | Heartbeat tick, A2A task-complete, webhook-driven decisions |

**Why three modes matter:** ARTIFACT can run in quiet hours without disturbing the conversation; COGNITION cannot. ARTIFACT cost is bounded by the handler; COGNITION can chain tool calls. Quiet-hours, cost limits, and UI rendering all behave differently per mode.

**Why ARTIFACT is handler-driven:** existing scheduled work like `morning_signal` reads from features, may call out to other services, may make multiple LLM calls, and returns text. A "render template, complete once" abstraction would force a rewrite. The handler contract preserves what these features already do; the simple template case is covered by a built-in helper:

```python
async def template_artifact_handler(template_path: Path) -> ArtifactHandler:
    """Built-in helper for simple template-render → one-shot LLM completion."""
    ...
```

## Classification of existing signals

| Signal | Mode | Notes |
|---|---|---|
| Heartbeat tick | COGNITION | Already a turn |
| Cron `morning_signal` | ARTIFACT | Feature workflow; returns briefing text |
| Cron `reflect` | ARTIFACT | LLM-authored, no follow-up cognition |
| Cron `signal_dispatch` | ACTION | Existing side-effect tool |
| Cron `trash_retention` | ACTION | Pure ops |
| Cron `backup_snapshot` | ACTION | Pure ops |
| Cron `memory_consolidate` | ARTIFACT (likely) | Feature owner confirms during migration |
| Cron `training_cycle` | ACTION | Long-running ops |
| A2A task complete | COGNITION | Bird decides "what now" |
| Stripe `deposit_complete` | COGNITION | Bird may want to act |
| Approval grant/deny | **not a signal** | Gate release on paused turn — hook system |
| Future: GitHub PR comment | COGNITION | |
| Future: inbound email | COGNITION (likely) | |
| Future: file watcher | ACTION (mostly) | |

## The Source Registry

Every signal source registers once at startup. The registry is the v1 boundary — default-deny, audit surface, dispatcher's source of truth.

```python
@dataclass
class SourceRegistration:
    # Identity
    name: str                         # "heartbeat", "cron.morning_signal", "webhook.stripe", ...
    schema: type[BaseModel]           # validation for payload

    # Mode
    default_mode: SignalMode
    allowed_modes: set[SignalMode]    # most sources have one; cron may have several

    # Behavior contracts (mode-specific; exactly one path per allowed mode)
    handler: Callable[[dict], Awaitable[Any]] | None              # required if ACTION in allowed_modes
    artifact_handler: Callable[[Signal], Awaitable[ArtifactResult]] | None  # required if ARTIFACT
    prompt_template: Path | None                                  # required if COGNITION

    # Trust & sanitization
    trust: Trust                      # TRUSTED | UNTRUSTED
    sanitizer: Callable | None        # required if trust == UNTRUSTED and any non-ACTION mode allowed

    # Throttling
    rate_limit: RateLimit             # per-minute, per-hour, burst
    coalescing_window: timedelta | None  # None = use global default (5s)

    # Attention
    attention_policy: AttentionPolicy
    # AttentionPolicy fields:
    #   quiet_hours: tuple[time, time] | None
    #   modes_governed: set[SignalMode]    # which modes the quiet window applies to
    #   urgency_override: Urgency          # signals at-or-above this urgency bypass quiet hours

    # Concurrency
    resources: set[ResourceLock]      # what this source's handler touches (e.g. MEMORY, WALLET)
                                      # COGNITION sources MUST NOT include CONVERSATION —
                                      # the turn lifecycle is the sole owner of that lock (Concern #1).
                                      # ARTIFACT/ACTION sources include only what they touch.
    allow_self_loops: bool = False    # opt-out of (target_agent, source) cycle check

    # Privacy & audit
    log_redaction: RedactionPolicy    # required; no defaults
    retention_days: int               # per-source signal_log retention

    # Prompt selection
    allow_prompt_override: bool = False  # opt-in for per-signal COGNITION templates
```

A source not in the registry cannot dispatch. Registration is an explicit code change reviewable as a unit, not a one-liner from inside a feature.

`allow_prompt_override` is available via
`kestrel_sovereign.signals.SourceRegistrationWithPromptOverride` while
the upstream SDK dataclass catches up; ordinary SDK registrations keep
the safe default of `False`.

## The Signal envelope

```python
@dataclass
class Signal:
    # Identity
    id: SignalId
    source: str                  # must match a SourceRegistration.name
    kind: str                    # source-specific sub-type
    mode: SignalMode             # ACTION | ARTIFACT | COGNITION
    payload: dict                # validated against registration.schema

    # Routing
    target_agent: AgentId        # which bird receives this
    session_id: SessionId | None # conversation/session to attach to; None = system-initiated
    caller: CallerIdentity | None
    visibility: Visibility       # INTERNAL | USER_VISIBLE | ADMIN_VISIBLE

    # Behavior
    urgency: Urgency             # LOW | NORMAL | HIGH
    dedupe_key: str | None
    origin_trust: Trust          # inherited from source registration
    prompt_template_override: Path | None  # honored only when registration opts in

    # Causation
    causation_chain: list[CausationFrame]

    # Audit
    arrived_at: datetime
```

`prompt_template_override` is available via
`kestrel_sovereign.signals.SignalWithPromptTemplateOverride` while the
upstream SDK dataclass catches up. The dispatcher still treats it as a
normal `Signal` envelope.

```python
@dataclass(frozen=True)
class CausationFrame:
    agent_id: AgentId
    source: str
    signal_id: SignalId
    turn_id: TurnId | None       # None if this frame's signal was ACTION/ARTIFACT
    depth: int                   # 1-indexed; first hop is depth 1
    emitted_at: datetime
```

## The dispatcher contract

Two entry points:

```python
async def dispatch_signal(self, signal: Signal) -> SignalResult: ...
async def enqueue_signal(self, signal: Signal) -> SignalHandle: ...
```

**`dispatch_signal`** awaits the full lifecycle and returns the result. Used by the scheduler (which records result/duration), heartbeat, and any caller that needs the outcome.

**`enqueue_signal`** returns immediately with a tracked handle. The work runs as an agent-owned background task using the existing background task tracker — exceptions are logged, tasks are cancellable on shutdown, no leakage. Used by webhook handlers (HTTP path returns 200 quickly) and any caller that doesn't need the outcome but must not lose it.

**Raw `asyncio.create_task(dispatch_signal(...))` at call sites is forbidden.** The point of `enqueue_signal` is supervised lifetime; ad-hoc tasks defeat that.

```python
@dataclass
class SignalResult:
    signal_id: SignalId
    status: Status               # OK | COALESCED | DROPPED_RATE_LIMIT
                                 # | DROPPED_QUIET_HOURS | DROPPED_CYCLE
                                 # | DROPPED_VALIDATION | FAILED
    mode: SignalMode
    turn_id: TurnId | None       # if COGNITION
    artifact: ArtifactResult | None  # if ARTIFACT
    action_result: Any | None    # if ACTION
    duration_ms: int
    error: ErrorInfo | None
```

The dispatcher pipeline:

1. **Validate** against registration. Unknown source → `DROPPED_VALIDATION`. Mode not in `allowed_modes` → `DROPPED_VALIDATION`. Schema fail → `DROPPED_VALIDATION`. UNTRUSTED with no sanitizer for non-ACTION mode → `DROPPED_VALIDATION`.
2. **Append-and-cycle-check** — see Concern #6 for exact rules.
3. **Quiet-hours-check** against the source's `attention_policy`. `urgency >= attention_policy.urgency_override` bypasses.
4. **Coalesce** by `dedupe_key` within `coalescing_window` (or global default).
5. **Acquire registered resource locks** in lexicographic order of lock name (single ordered lock manager — see Concern #2). `CONVERSATION` is **never** in this set; it is acquired downstream by the turn lifecycle for COGNITION (see Concern #1).
6. **Route**:
   - ACTION → `await registration.handler(payload)`
   - ARTIFACT → `await registration.artifact_handler(signal)`
   - COGNITION → select the registration `prompt_template`, or the signal's `prompt_template_override` only when the registration has `allow_prompt_override=True`; render with the signal envelope → `await agent.process_input_or_streaming(prompt, ...)`. The entry point itself acquires `CONVERSATION` at the shared turn lifecycle (Concern #1) — the dispatcher does not pre-acquire it. Streaming vs non-streaming is selected by the calling context; both share the same lifecycle boundary.
7. **Release locks** in reverse acquisition order.
8. **Log** per the source's redaction policy.

The dispatcher lives as a sibling component the agent holds a reference to. Easier to test than another mixin.

## Logging & Privacy (`signal_log`)

`signal_log` is a privacy surface, not just a debugging convenience.

**Schema:**
```
signal_log
  id, source, kind, mode, urgency, dedupe_key
  target_agent, session_id, caller_redacted, visibility
  dispatched_at, completed_at, status, duration_ms
  turn_id_if_cognition, artifact_digest_if_artifact, action_result_digest_if_action
  payload_digest          -- sha256(payload) regardless of trust
  payload_redacted        -- per-source RedactionPolicy applied
  causation_chain_digest  -- sha256 of chain for audit
  prompt_template_hash    -- sha256 of chosen template body for COGNITION
  retention_until         -- computed from registration.retention_days
```

**Rules:**
- **UNTRUSTED raw payloads are never stored.** Only digest + redacted summary.
- **TRUSTED payloads** stored in full only if registration explicitly opts in.
- **Encryption parity** — `signal_log` honors the same privacy mode as the agent's primary storage (defer to existing privacy infrastructure).
- **Retention** is per-source; a retention sweep runs as an ACTION signal.
- **Caller identity** redacted by default to role/scope; opt-in for full identifier.
- **Causation chain** stored as digest; full chain reconstructible from individual entries.
- **Prompt-template provenance** is stored for COGNITION dispatches after prompt render. The hash covers the exact template body chosen for that dispatch, including per-signal overrides.

If a registration doesn't specify a redaction policy, the dispatcher refuses to register the source. No defaults; this is too important to default.

## Concerns

### 1. Concurrency is already broken — v1 fixes it for both code paths
[kestrel_agent.py:1187](../../kestrel_sovereign/kestrel_agent.py#L1187) `process_input` has no turn lock; the streaming variant `process_input_streaming` (the main UI path — runs the same hooks/tools/history writes) has the same problem. [heartbeat.py:247](../../kestrel_sovereign/heartbeat.py#L247) doesn't check whether a turn is in flight. Two turns can interleave today; conversation history is racy.

The fix lives on the **turn lifecycle**, not the entry method: both `process_input` and `process_input_streaming` pass through a shared turn-begin / turn-end boundary, and the `CONVERSATION` resource lock is acquired/released there. All entry points — direct user input via HTTP, dispatcher-routed COGNITION signals, heartbeat — go through the same boundary, and **the turn lifecycle is the sole owner of `CONVERSATION`**. The dispatcher does not pre-acquire it for COGNITION sources (see Concern #2). **In scope for v1.**

### 2. Single ordered lock manager — no separate turn lock
v2 had two lock concepts (turn lock + resource locks) and the routing acquired resource locks before the turn lock. That's a deadlock surface — another path holding the turn lock and then needing a resource lock would self-deadlock against this one. v3 collapses them: there is **one** lock manager. `CONVERSATION` is one named lock among `MEMORY`, `WALLET`, `SCHEDULER`, etc. Acquisition is a single sorted pass (lexicographic by lock name); release is reverse. No deadlock possible because there is no second order.

COGNITION sources MUST NOT declare `CONVERSATION` in `resources` — the shared turn lifecycle is the sole acquirer of that lock (Concern #1). The dispatcher acquires the source's other registered resources (e.g. `MEMORY`, `WALLET`) in lex order, then enters the turn entry point which acquires `CONVERSATION` last. ARTIFACT and ACTION sources declare only what they touch and never include `CONVERSATION`. The registry validator enforces "COGNITION sources must NOT include CONVERSATION."

`CONVERSATION` is treated as the highest-lock-order acquisition in the system: any caller acquiring multiple locks acquires `CONVERSATION` last. This keeps the global lex-order invariant intact at the lifecycle boundary regardless of how the lex-name happens to sort.

### 3. Cost
Every COGNITION source is potential LLM spend; ARTIFACT sources are bounded but non-zero. Per-source `rate_limit` from the registry is the throttle. **Default-deny for COGNITION on any new source.** Operators tune `rate_limit` explicitly.

### 4. Conversation coherence (UI side channel)
A COGNITION turn produced mid-user-thread by a task-complete signal produces an assistant message the user didn't ask for. `visibility` and `session_id` distinguish:

- `session_id == current chat` + `visibility == USER_VISIBLE` → render inline with attribution
- `session_id == None` (system-initiated) → render in a side channel
- `visibility == INTERNAL` → no UI rendering; log only

Without this, spontaneous cognition will look like a hijacked chat. Frontend work item this design forces.

### 5. Approval is not a signal
The recent commit `61b431a4 fix(approval): take the watchdog clock off human-input hooks` is the team hitting the confusion of treating approval as both hook and event.

- **Approval is a gate release** on a paused turn.
- It belongs in the **hook system**, not the dispatcher.
- The dispatcher does **not** own it.

If approval became a signal, you'd get a second turn waking up to "approval was granted" while the original turn is still sitting waiting for that approval. Document loudly in the dispatcher module's docstring.

### 6. Cycle detection — precise rules
The chain represents the lineage of hops. Each frame is "agent X processed signal Y in turn Z at depth D." When agent X emits a new signal during turn Z, the new signal's chain is the receiving chain extended by X's frame.

**Order of operations on dispatch:**

1. Compute the would-be new frame for the receiving agent: `depth = max_existing_depth + 1` (or 1 if chain empty).
2. Build a working copy of the chain with the new frame appended.
3. Cycle check against the working copy:
   - If `depth > TTL` (default 5) → `DROPPED_CYCLE`.
   - If a frame with the same `(agent_id, source)` pair appears earlier in the working copy → `DROPPED_CYCLE`, **unless** `registration.allow_self_loops == True`.
4. If passing, the working copy becomes the chain stored on the executing turn.

**Why same `(agent_id, source)` and not just `agent_id`:** A may legitimately receive a heartbeat and later (via causation chain from B) receive an a2a_complete — different sources, no cycle. A receiving the same source twice in a chain is the ping-pong pattern.

**Heartbeat semantics under this rule:** heartbeat fires with empty chain. At dispatch the working copy is `[{A, heartbeat, ..., depth=1}]`. No earlier frames → no cycle. Pass. If during that turn A self-emits a new heartbeat-source signal, the chain becomes `[{A, heartbeat, 1}, {A, heartbeat, 2}]` → cycle detected → reject. Correct behavior.

**Self-emit escape valve:** sources with legitimate same-agent same-source re-entry (e.g. a deliberate self-tick during a turn) set `allow_self_loops = True` in registration; only TTL applies.

### 7. The scheduler stays
The dispatcher does not replace the scheduler. Cron owns *when*; dispatcher owns *what*. The scheduler's executor changes from "call this tool" to "await dispatch_signal(...)" — the registration declares the mode, so cron config doesn't have to. The persistent `scheduled_tasks` table is unchanged. Scheduler's existing result/duration/status records continue to work because `dispatch_signal` returns `SignalResult`.

### 8. Hooks are not signals
Hooks intercept an in-flight turn (PRE/POST tool use, USER_PROMPT_SUBMIT, STOP, etc.). Signals originate work.

- **Signals** wake the bird or perform side effects (this dispatcher).
- **Hooks** modify the flow of work the bird is already doing (existing `HookManager`).

A signal can produce a turn that fires hooks. A hook never produces a signal. One-way arrow.

### 9. Prompt translation layer (COGNITION only)
COGNITION sources register a `prompt_template: Path` rendered with the Signal envelope into a structured prompt with explicit fences for UNTRUSTED payload. Templates live under `prompts/signals/`.

ARTIFACT sources do not use `prompt_template` — they own their handler's prompt construction (because they may make multiple LLM calls or none). The template helper `template_artifact_handler(path)` exists for ARTIFACT sources that genuinely are one-shot template completions; everything else writes a real handler.

### 10. Quiet-hours / attention policy is general
v1 had `active_hours` as a heartbeat property. Under the registry, `attention_policy` is per-source: quiet window, which modes it governs (commonly "ARTIFACT and COGNITION, not ACTION"), and an urgency override threshold. Heartbeat becomes one producer governed by this policy.

### 11. Logging privacy (covered above)
UNTRUSTED raw payloads never stored. Per-source redaction. Encryption parity with privacy mode. Retention sweep.

### 12. Migration is incremental
- Heartbeat and scheduler keep working through the dispatcher with no behavior change.
- New COGNITION sources (A2A task-complete, Stripe deposit) light up only after registration.
- The old `emit_event` SSE path stays alive — orthogonal to this design.
- Approval stays in the hook system; nothing moves.

No big-bang merge. Each source migrates as its own PR.

## Out of scope (explicit)

- Distributed dispatch across multiple agent processes (single-process for now; envelope is future-proof via `target_agent`)
- Signal priorities beyond LOW / NORMAL / HIGH
- Replay-from-log
- Replacing the hook system
- Cross-agent signal routing beyond what A2A already provides

## Open questions

1. **`memory_consolidate` mode** — likely ARTIFACT; feature owner confirms during migration.
2. **Coalescing window default** — 5s global default with per-source override. Confirm 5s is the right number.
3. **Out-of-quiet-hours HIGH-urgency COGNITION** — immediate dispatch. Confirm.
4. **`session_id == None` resolution** — per-source rolling daily session for tidy UI grouping. Confirm.
5. **Resource lock granularity** — coarse (`MEMORY`, `WALLET`, ...) for v1; finer keys (`MEMORY:user_123`) as a follow-up. Confirm.
6. **Visibility default** — `INTERNAL`. Sources opt into user visibility. Confirm.

## v1 acceptance criteria

The epic is shippable when:

1. `Signal`, `SignalMode`, `SignalResult`, `SignalHandle`, `CausationFrame`, `SourceRegistration`, `RedactionPolicy`, `AttentionPolicy`, `RateLimit`, `ResourceLock` defined and unit-tested.
2. `SignalDispatcher` implemented with: validation, append-and-cycle-check (with documented order), per-source quiet-hours, dedupe coalescing, single ordered lock manager, three-mode routing, structured logging.
3. **Shared turn lifecycle for `process_input` and `process_input_streaming`.** `CONVERSATION` lock acquired/released at the lifecycle boundary. Direct HTTP user input, heartbeat, and dispatcher-routed COGNITION all pass through it. Race regression test: two concurrent turns must serialize.
4. `enqueue_signal` uses the agent's existing background task tracker; raw `create_task` forbidden by lint/review.
5. `signal_log` table created with privacy rules enforced; registration without `RedactionPolicy` fails to register.
6. `template_artifact_handler` helper provided.
7. **Heartbeat migrated** to register and dispatch (behavior unchanged; now race-safe).
8. **Scheduler migrated** — cron tasks register sources, scheduler awaits `dispatch_signal`, existing result/duration contract preserved.
9. **A2A `task_manager.py`** `on_task_complete` migrated — task-complete becomes COGNITION with full causation chain; A2A propagates chain via task metadata.
10. **Stripe webhook** migrated — `deposit_complete` becomes COGNITION (UNTRUSTED, with sanitizer) via `enqueue_signal`.
11. **UI side-channel rendering** for system-initiated turns where `session_id == None` AND `visibility != INTERNAL`. `visibility == INTERNAL` turns remain unrendered (log only) per Concern #4.
12. Documentation: dispatcher module docstring covers hooks-vs-signals; source registration guide written; cycle-detection rules documented with worked examples.

## Appendix: why this wasn't done originally

The bird started as a chat bot — one entry point, one mental model. Heartbeat was the first crack in that frame and was conceived as a periodic timer, not as one-of-N signals. Every signal added afterward solved a local problem in its author's vocabulary: scheduler author thought "cron"; A2A author thought "callback"; wallet author thought "webhook"; security author thought "hook." Three different problems, four different solutions, no consolidation.

From inside the timeline this is invisible; from outside, it's the same accretion pattern documented in CLAUDE.md for model selection. The design is obvious in retrospect for the same reason every consolidation is obvious in retrospect: the abstraction only becomes visible once enough instances exist to suggest it.
