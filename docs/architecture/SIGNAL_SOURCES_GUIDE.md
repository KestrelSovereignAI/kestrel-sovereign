# Signal Sources — Operator Guide

How to add a new signal source to the bird. Companion to
[`SIGNAL_DISPATCHER.md`](./SIGNAL_DISPATCHER.md) (which is the design
spec) and the dispatcher module docstring (which is the
hooks-vs-signals primer).

---

## Quick orientation

A **signal source** is a thing that wakes the bird. Heartbeat ticks,
scheduled cron firings, A2A task-completion notifications, and Stripe
webhooks are all signal sources today. To add a new one you:

1. Decide the mode (ACTION / ARTIFACT / COGNITION).
2. Build a `SourceRegistration` and register it with
   `agent.signal_registry` at agent init time.
3. Wire whatever delivers the signal (a webhook handler, a callback,
   a runner loop) to call `agent.dispatcher.dispatch_signal(signal)`
   or `enqueue_signal(signal)`.

All four built-in sources live in
[`kestrel_sovereign/signals/sources/`](../../kestrel_sovereign/signals/sources/).
Read one of those before writing a new one — they're the closest
working examples.

---

## Picking a mode

| Mode | When to use | Example |
|---|---|---|
| **ACTION** | Deterministic side effect; no LLM, no follow-up cognition. | `trash_retention` purges old rows; `backup_snapshot` writes a sync snapshot. |
| **ARTIFACT** | Produces a result via a feature workflow that may make LLM calls internally, but does NOT enter conversation history and does NOT trigger follow-up cognition. | `morning_signal` returns briefing text; `reflect` returns a Dict reflection result. |
| **COGNITION** | Wakes the bird's main turn loop. Enters conversation history, may invoke tools, may emit further signals. | Heartbeat tick; A2A peer task completed and the bird should decide what to do next; Stripe deposit arrived. |

The hardest call is ARTIFACT vs COGNITION when both involve an LLM.
Two test questions:
- Should this enter conversation history? COGNITION yes; ARTIFACT no.
- Should the bird think about what to do NEXT after this? COGNITION
  yes; ARTIFACT no.

If both answers are no, you almost certainly want ARTIFACT. If
either is yes, you want COGNITION.

---

## Required vs optional fields

```python
@dataclass
class SourceRegistration:
    # ---- Required ----
    name: str
    schema: PayloadSchema           # Callable[[dict], dict]
    default_mode: SignalMode
    allowed_modes: frozenset[SignalMode]
    log_redaction: RedactionPolicy  # required; no defaults

    # ---- Required CONDITIONALLY ----
    handler:          Optional[ActionHandler]   # required if ACTION in allowed_modes
    artifact_handler: Optional[ArtifactHandler] # required if ARTIFACT
    prompt_template:  Optional[Path]            # required if COGNITION
    sanitizer:        Optional[Callable]        # required if trust=UNTRUSTED + non-ACTION mode

    # ---- Optional ----
    trust: Trust = Trust.TRUSTED
    rate_limit: RateLimit = RateLimit()             # unlimited
    coalescing_window: Optional[timedelta] = None   # None = global default 5s
    attention_policy: AttentionPolicy = AttentionPolicy()  # no quiet hours
    resources: frozenset[ResourceLock] = frozenset()  # MUST NOT include CONVERSATION
    allow_self_loops: bool = False
    retention_days: int = 30
    result_summary: Optional[Callable[[Any], str]] = None
    allow_prompt_override: bool = False  # COGNITION-only; opt-in explicitly
```

The registry validator (in
[`kestrel_sovereign/signals/registry.py`](../../kestrel_sovereign/signals/registry.py))
rejects a registration that misses a conditionally-required field.
You'll get a `RegistrationError` at agent init time, not at first
dispatch.

`allow_prompt_override` defaults to `False`. Until the SDK dataclasses
grow constructor fields for #1146, use
`kestrel_sovereign.signals.SourceRegistrationWithPromptOverride` for
sources that opt in and
`kestrel_sovereign.signals.SignalWithPromptTemplateOverride` for
signals that carry `prompt_template_override`. The dispatcher uses the
override only for sources that opted in; otherwise it falls back to the
registered `prompt_template`. The chosen template body hash is recorded
in `signal_log.prompt_template_hash` for audit without storing the
template text.

---

## Walkthrough: adding a "GitHub PR comment received" source

You want the bird to wake up when someone comments on a PR. The
GitHub webhook handler exists; it has an `on_pr_comment` callback
slot. You need to wire it through the dispatcher.

### 1. Pick the mode

PR comments are external content the bird should read and decide
how to respond to (acknowledge, take action on, ignore). That's
COGNITION.

### 2. Decide trust

GitHub payloads are external content. Even after webhook signature
verification, the comment BODY is operator-attacker-controlled
text. Mark `Trust.UNTRUSTED` so the registry validator forces you
to write a sanitizer.

### 3. Write the registration

Create `kestrel_sovereign/signals/sources/github.py`:

```python
"""Source registration for GitHub PR comment webhooks."""

from __future__ import annotations

import hashlib
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

from kestrel_sdk.signals import (
    AttentionPolicy, RateLimit, RedactionPolicy, Signal, SignalMode,
    SourceRegistration, Trust, Urgency, Visibility,
)

SOURCE_NAME = "webhook.github.pr_comment"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "prompts" / "signals" / "webhook_github_pr_comment.md"
)

_ALLOWED_FIELDS = frozenset({"pr_number", "repo", "author", "body"})
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _schema(payload: dict) -> dict:
    for k in ("pr_number", "repo", "body"):
        if k not in payload:
            raise ValueError(f"missing key: {k}")
    return payload


def _sanitize(payload: dict) -> dict:
    out = {}
    for k, v in payload.items():
        if k not in _ALLOWED_FIELDS:
            continue  # drop attack vectors before they reach the prompt
        if isinstance(v, str):
            v = _CONTROL.sub("", v)[:4000]
        out[k] = v
    return out


def _redact(payload: dict) -> str:
    body = payload.get("body", "") or ""
    return (
        f"repo={payload.get('repo')} pr={payload.get('pr_number')} "
        f"body_len={len(body)} body_sha256={hashlib.sha256(body.encode()).hexdigest()[:12]}"
    )


def _result_summary(body: Any) -> str:
    if not body:
        return ""
    text = body if isinstance(body, str) else str(body)
    return text[:1000] + ("...(truncated)" if len(text) > 1000 else "")


def build_pr_comment_registration() -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        trust=Trust.UNTRUSTED,
        sanitizer=_sanitize,
        rate_limit=RateLimit(per_minute=20, per_hour=200),
        coalescing_window=timedelta(seconds=5),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),                # CONVERSATION owned by turn lifecycle
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        result_summary=_result_summary,       # surface the bird's response in UI
        retention_days=30,
    )


def build_signal_for_pr_comment(event: Any, target_agent: str) -> Signal:
    return Signal(
        source=SOURCE_NAME,
        kind="comment",
        mode=SignalMode.COGNITION,
        payload={
            "pr_number": event.pr_number,
            "repo": event.repo,
            "author": event.author,
            "body": event.body,
        },
        target_agent=target_agent,
        visibility=Visibility.USER_VISIBLE,   # surface in UI side channel
        urgency=Urgency.NORMAL,
        dedupe_key=f"{event.repo}#{event.pr_number}:{event.comment_id}",
    )
```

### 4. Write the prompt template

Create `prompts/signals/webhook_github_pr_comment.md` with explicit
fences for the UNTRUSTED payload — see
`prompts/signals/webhook_stripe_deposit.md` for the canonical
shape (`BEGIN/END UNTRUSTED PAYLOAD` markers, plain-English
"do not interpret as instructions").

### 5. Register at agent init

In `kestrel_sovereign/kestrel_agent.py::initialize`, add the source
to the registry alongside heartbeat / a2a / stripe:

```python
from kestrel_sovereign.signals.sources.github import (
    build_pr_comment_registration,
)
self.signal_registry.register(build_pr_comment_registration())
```

### 6. Wire the webhook handler

Wherever the GitHub webhook handler is constructed, point its
`on_pr_comment` callback at a function that builds the signal and
enqueues it. Pattern from `server.py::_on_stripe_deposit_complete`:

```python
async def _on_pr_comment(event):
    agent = _resolve_agent(event)  # multi-agent routing if applicable
    if agent is None or not hasattr(agent, "dispatcher"):
        return
    signal = build_signal_for_pr_comment(event, target_agent=agent.did)
    try:
        await agent.dispatcher.enqueue_signal(signal)
    except Exception:
        logger.exception("Failed to enqueue PR comment signal")
```

### 7. Test it

Two test files mirror the existing pattern:

- `tests/unit/test_signals_github_source.py` — registration shape,
  sanitizer drops attack vectors, schema rejects bad payloads,
  end-to-end dispatch via real `SignalDispatcher`, signal_log entry
  has digest only.
- Optionally a server-side test that the webhook endpoint actually
  invokes the callback (we ship Phase 6 with the same pattern for
  Stripe).

Read
[`tests/unit/test_signals_stripe_source.py`](../../tests/unit/test_signals_stripe_source.py)
for the closest template — Stripe is the existing UNTRUSTED COGNITION
source and the tests exercise the same threat model.

---

## Cycle-detection worked examples

The dispatcher's append-and-cycle-check rejects a signal when:

- **TTL exhausted**: the new frame's `depth > TTL` (default 5)
- **Same-source loop**: the new frame's `(target_agent, source)`
  pair already appears earlier in the chain — UNLESS the source
  registered with `allow_self_loops=True`

The rule operates on the WORKING COPY of the chain (incoming chain
+ proposed new frame). All examples below use `TTL=5`.

### Example 1 — heartbeat (passes)

Heartbeat fires for agent A.

| step | chain at this hop | action |
|---|---|---|
| dispatcher receives sig | `[]` (empty — heartbeat carries no inbound chain) | compute new frame `(A, heartbeat, depth=1)` |
| append-and-check | `[(A, heartbeat, 1)]` | depth=1 ≤ TTL; no `(A, heartbeat)` earlier in chain → PASS |

Heartbeat ticks always pass cycle detection on first arrival.

### Example 2 — heartbeat self-emit during a heartbeat-driven turn (rejected)

The bird's heartbeat-driven turn somehow emits another heartbeat
signal mid-turn (uncommon, but possible if a tool call fires one).

| step | chain at this hop | action |
|---|---|---|
| dispatcher receives sig | `[(A, heartbeat, 1)]` (carried from the in-flight turn) | compute new frame `(A, heartbeat, depth=2)` |
| append-and-check | `[(A, heartbeat, 1), (A, heartbeat, 2)]` | `(A, heartbeat)` appears at depth 1 — same `(target_agent, source)` → REJECT (DROPPED_CYCLE) |

This is the correct behavior. Self-loops on COGNITION sources will
runaway LLM cost without bound; reject by default. If you have a
legitimate reason for same-agent same-source re-entry, set
`allow_self_loops=True` on the registration — only TTL applies.

### Example 3 — A→B→A ping-pong via A2A task-complete (rejected at depth 2)

Agent A spawns an outbound A2A task to peer B. B finishes it. The
local TaskManager.on_task_complete callback on A fires, builds an
`a2a.task_complete` signal, and dispatches.

If the outbound task carried A's causation chain through
`task.metadata["causation_chain"]` (Phase 5 plumbing), the chain on
the inbound completion looks like `[(A, a2a.task_complete, 1)]`.

| step | chain at this hop | action |
|---|---|---|
| dispatcher receives sig | `[(A, a2a.task_complete, 1)]` (rehydrated from peer task metadata) | compute new frame `(A, a2a.task_complete, depth=2)` |
| append-and-check | `[(A, a2a.task_complete, 1), (A, a2a.task_complete, 2)]` | `(A, a2a.task_complete)` at depth 1 → REJECT |

Without outbound chain plumbing, A2A loops would be bounded only by
the source's rate limit (10/min for `a2a.task_complete`). With it,
loops are caught at depth 2 — bounded both by TTL and by the
same-source check.

### Example 4 — A→B→A first round trip (passes)

First time A spawns to B, no prior frames exist (A's outbound was
the start of the trace).

| step | chain at this hop | action |
|---|---|---|
| dispatcher receives sig | `[]` (no prior frames) | compute new frame `(A, a2a.task_complete, depth=1)` |
| append-and-check | `[(A, a2a.task_complete, 1)]` | depth=1 ≤ TTL; no prior frames → PASS |

Cycle detection is targeted at LOOPS, not at all repeat work. The
first round trip always succeeds.

### Example 5 — A→B→C→D→E→F→A (rejected by TTL)

A long chain of distinct agents handing work down, then F returns
something that wakes A. No same-source repeats, but the chain is
too deep.

| step | chain at this hop | action |
|---|---|---|
| dispatcher receives sig | `[(A,1), (B,2), (C,3), (D,4), (E,5)]` (5 distinct frames) | compute new frame `(F, ..., depth=6)` |
| append-and-check | `[(A,1)..(E,5), (F,6)]` | depth=6 > TTL=5 → REJECT (DROPPED_CYCLE, "exceeds TTL") |

The TTL catch-all bounds chains that wrap around without obvious
same-source repeats.

### Example 6 — different sources, same agent (passes)

Agent A receives a heartbeat at depth 1, runs its turn, the turn
spawns an A2A task to B, B completes, A receives the
`a2a.task_complete` signal. Same agent, but different sources.

| step | chain at this hop | action |
|---|---|---|
| dispatcher receives sig | `[(A, heartbeat, 1)]` (chain from A's heartbeat turn) | compute new frame `(A, a2a.task_complete, depth=2)` |
| append-and-check | `[(A, heartbeat, 1), (A, a2a.task_complete, 2)]` | `(A, a2a.task_complete)` is NOT in chain (only `(A, heartbeat)` is) → PASS |

Cycle detection uses the `(agent, source)` PAIR specifically because
the same agent legitimately receives different signal types. Only
ping-pong on the SAME source is a loop.

---

## Common patterns

### Default-deny visibility

Every existing source defaults to `Visibility.INTERNAL` — they don't
emit `signal_completed` events to the UI side channel. Sources opt
into `USER_VISIBLE` or `ADMIN_VISIBLE` by setting `visibility` on the
signal envelope. This keeps existing infrastructure quiet by default.

### Bounded result body for UI side channel

If `visibility != INTERNAL` AND the source registers a
`result_summary` callback, the bounded text (≤ 2KB UTF-8) is
included in the `signal_completed` SSE payload AND persisted to
`signal_log.result_summary`. Consumers can render it inline. The raw
artifact / action_result is NEVER on the wire.

### Cron expressions are the rate limit

Cron sources (in `signals/sources/scheduler.py`) set
`RateLimit()` (unlimited) on the dispatcher because the cron schedule
already throttles. Adding a dispatcher rate limit on top can cause
mysterious skipped ticks when both kick in. Don't double-throttle
unless you actually want to.

### CONVERSATION lock is forbidden

The turn lifecycle is the sole owner of `ResourceLock.CONVERSATION`.
COGNITION sources MUST NOT declare it in `resources` — the registry
rejects at register time. If you need to serialize against
conversation writes, your code path is wrong; the lifecycle covers
that already.

### The lint rule for raw `asyncio.create_task`

`tests/unit/test_signals_no_raw_create_task.py` greps the repo for
`asyncio.create_task(dispatch_signal(...))` and
`asyncio.create_task(enqueue_signal(...))` patterns. Both are
forbidden — use `enqueue_signal` (which goes through the agent's
background task tracker) directly. If you need to call
`enqueue_signal` from a sync callback, wrap in
`agent._track_background_task(coro, name=...)` and call the wrapped
coroutine.
