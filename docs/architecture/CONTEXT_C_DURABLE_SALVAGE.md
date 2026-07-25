---
type: Architecture Spec
title: Context C Durable Salvage Design Record
description: Aspirational design record for a complete automatic durable-salvage lifecycle; only a feature-flagged subset is implemented.
resource: /docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md
tags:
- docs
- architecture
- architecture-spec
- context
- design
timestamp: '2026-07-25T00:00:00Z'
status: aspirational
owner: context-runtime
canonical: false
generated: false
privacy: public
---

# Context C Durable Salvage Design Record

> **Status: aspirational design with a conditional partial implementation.**
> This page is not the current context runtime contract. Read
> [Kestrel Context Management Contract](CONTEXT_SYSTEM_DESIGN.md) for shipped
> behavior, including the default lumpy-pruning path, the feature flag, and the
> diagnostic honesty boundary. The complete state machine described here has
> not shipped.

## Purpose

Context C is the intended end state for automatic history eviction: no message
should disappear from the model-visible window until Kestrel has synchronously
established a durable, recoverable representation of the evicted span.

The design remains useful because the current default preserves original rows
in storage but can omit them from the provider window without first creating a
summary. That is safe from physical deletion, but it is not a complete
durable-salvage guarantee.

## Status matrix

| Capability | Current status |
|---|---|
| Lumpy selection drops an older history chunk while retaining database rows | **Shipped default** |
| Manual context compaction writes a summary marker and excludes originals | **Shipped when invoked** |
| Codex occupancy handling compacts durable session history before resetting a Codex thread | **Shipped on `openai:plan` when its threshold is crossed** |
| Automatic prune writes a salvage marker/span before omission | **Conditional**, behind `KESTREL_CONTEXT_C_DURABLE_SALVAGE` |
| Background processing and janitor use `SalvageWorker` | **Conditional**, behind the same flag |
| Automatic salvage is the default for all routes | **Not shipped** |
| Original SignalDispatcher-based orchestration | **Not shipped**; the partial implementation uses `SalvageWorker` |
| Complete recovery, operator UX, and every state transition in this design | **Aspirational** |

The conditional implementation lives in
[`kestrel_sovereign/agent/salvage.py`](../../kestrel_sovereign/agent/salvage.py)
and is called from
[`kestrel_sovereign/agent/context_manager.py`](../../kestrel_sovereign/agent/context_manager.py).
The manual compaction path is owned by
[`kestrel_sovereign/features/context/feature.py`](../../kestrel_sovereign/features/context/feature.py)
and the conversation manager.

## Intended invariant

The target invariant is:

> Before an automatic prune can remove a span from the provider-visible
> history, a synchronous transaction records exactly which canonical messages
> are being displaced and how they can be recovered. If that write fails, the
> turn fails closed rather than silently pruning.

This design distinguishes:

- **canonical rows**, which remain the durable conversation record;
- **model-visible history**, which is the selected rendered suffix;
- **salvage markers**, which name the displaced canonical span;
- **derived summaries**, which help later turns recover the span's meaning; and
- **worker state**, which records asynchronous summary progress and failure.

## Intended lifecycle

The full design aims for these stages:

1. Compute the lumpy prune boundary from the production token budget.
2. Materialize the exact canonical span represented by that boundary.
3. In one durable transaction, create an idempotent salvage record and bind it
   to the span.
4. Permit the current turn to omit the span only after the durable write
   succeeds.
5. Summarize or otherwise condense the span asynchronously.
6. Attach the derived artifact to the salvage record without replacing the
   originals.
7. Retry or quarantine failures with explicit state and operator visibility.
8. Allow recovery, inspection, and reprocessing from canonical rows.

The intended state vocabulary is conceptually:

```text
pending -> processing -> complete
                    \-> failed -> retrying/quarantined
```

Exact schema and transition names remain implementation details until this
design is promoted and revalidated against the runtime.

## Safety properties

A complete implementation should preserve these properties:

- **Fail closed:** inability to write the durable marker prevents automatic
  omission on the feature-enabled path.
- **Idempotence:** retrying the same prune span cannot create conflicting
  salvage records.
- **Canonical preservation:** derived summaries never overwrite original
  conversation rows.
- **Session isolation:** salvage records and workers cannot cross agent or
  session boundaries.
- **Replay clarity:** summaries are distinguishable from original user and
  assistant turns.
- **Observable failure:** pending, failed, retried, and completed work can be
  inspected without inferring state from an absent summary.
- **Bounded work:** worker retries, summary size, and janitor scans have
  explicit limits.

## What the partial implementation proves

With `KESTREL_CONTEXT_C_DURABLE_SALVAGE` enabled, the current production
coordinator computes the pruned span, attempts the synchronous salvage
write/marker, and schedules the process-local `SalvageWorker`. It fails closed
when the prerequisite durable write is unavailable or unsuccessful.

That is meaningful implementation evidence, but it does **not** prove:

- that automatic salvage is active in default deployments;
- that a `SignalDispatcher` owns background work;
- that every provider and restart topology has completed recovery testing;
- that the Context feature's manual tools are the automatic state machine;
- that context-status is an exact view of the production prune; or
- that the broader UX and operational lifecycle in this design is complete.

`GET /api/agent/context-status` reports best-effort salvage counts and whether
the silent-prune path is active, but its measurement remains a projection.
The canonical contract documents the limitation and links open
[#2534](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2534).

## Promotion criteria

This page should move from `aspirational` only after all of the following have
runtime and test evidence:

- the intended invariant is enforced on the supported automatic prune paths;
- the enabled/default rollout state is explicit and tested;
- restart, retry, idempotence, and quarantine behavior is verified;
- diagnostics report real salvage state without implying an exact prompt
  simulation;
- operator recovery and inspection are documented; and
- the canonical context contract can label the lifecycle **Shipped** without
  relying on a feature flag or design inference.

Until then, implementation changes must be documented first in
[Kestrel Context Management Contract](CONTEXT_SYSTEM_DESIGN.md), with this
page retained as the design record rather than presented as current truth.
