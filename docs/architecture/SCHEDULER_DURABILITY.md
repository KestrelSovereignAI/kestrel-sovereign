---
type: Architecture Spec
title: Durable Scheduler Execution
description: Durable, multi-replica-safe scheduler execution with claimed leases,
  idempotent occurrences, one-shot deadlines, IANA time zones, and explicit
  misfire behavior.
resource: /docs/architecture/SCHEDULER_DURABILITY.md
tags:
- docs
- architecture
- architecture-spec
- scheduler
timestamp: '2026-07-24T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Durable scheduler execution

The scheduler is a durable wake transport. A schedule row represents its next
occurrence; an execution-log row represents that occurrence's stable identity.
Runners claim before dispatching, renew a finite lease while dispatching, and
complete through a compare-and-set on the same claim token. Multiple runners
may poll the same PostgreSQL database: at most one can claim a due occurrence.

An expired lease means the owner died or stopped renewing. A later runner may
recover the same occurrence, preserving its execution ID and idempotency key.
This prevents a completed occurrence from running again. It cannot make an
uncooperative external side effect exactly-once across a crash after dispatch
but before the outcome commit; target tools must use the supplied idempotency
key at their effect boundary. During dispatch they can retrieve
`get_current_scheduler_execution()` from
`kestrel_sovereign.features.scheduler.runner` for the execution ID, occurrence,
attempt number, and idempotency key.

For isolated-feature tools, Core translates that trusted scheduler identity to
the public SDK `ToolExecutionContext` JSON-RPC envelope. The envelope is never
merged into tool arguments. A scheduled invocation fails before dispatch when
the isolated service has not explicitly advertised execution-context support;
ordinary interactive/API calls retain their legacy wire format. Isolated tools
should use the SDK's `get_tool_execution_context()` only while their handler is
active and apply `idempotency_key` at the external-effect boundary.

## Schedule kinds and time zones

`schedule_add` creates a recurring cron schedule. Its `timezone_name` accepts
an IANA name and defaults to `UTC`, preserving existing schedules. Cron is
matched against local wall-clock time but `next_run_at` is stored as UTC.

- A spring-forward non-existent local time is skipped.
- A fall-back ambiguous local time fires once, at the earlier (`fold=0`)
  occurrence.

`schedule_add_deadline(run_at=...)` creates a one-shot absolute deadline.
`run_at` must have a UTC offset. After a claimed deadline reaches a terminal
outcome, it is disabled with `next_run_at = NULL`; it cannot be resumed.

## Misfires

Every schedule visibly stores one of these policies:

- `skip` — the compatible default. If lateness exceeds its per-schedule grace
  (or the host's `[scheduler] misfire_grace_seconds` when omitted), record a
  `skipped_misfire` outcome and re-anchor to the next future run.
- `fire_once` — execute one late occurrence, then re-anchor to a future run.
- `catch_up` — advance from the missed occurrence, allowing successive missed
  occurrences to be handled on later polls.

Set `[scheduler] lease_seconds` to control recovery speed. It must be positive;
the runner renews at roughly one-third of that interval.

## Hosted execution

`SchedulerRunner` remains the standalone per-agent executor by default. A
host with a shared database can pass `agent_id=None` plus a
`HostedSchedulerExecutor`, or the supplied
`AgentManagerHostedSchedulerExecutor`. The latter maps durable agent DIDs to
host configuration and invokes `AgentManager.load_agent` under a per-agent
lock when the target is cold. The host owns waking an unloaded agent; the
scheduler still owns claim, lease, idempotency, and completion semantics.
