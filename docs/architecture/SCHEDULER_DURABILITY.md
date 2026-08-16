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

Windows does not ship the IANA zoneinfo database. The base distribution
therefore declares `tzdata` directly on Windows; `zoneinfo` can resolve both
the default `UTC` and named zones such as `America/Chicago` without installing
the optional Pandas or Phoenix extras.

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
the runner renews at roughly one-third of that interval, strictly before
expiry, and keeps the renewal task alive until its terminal claim CAS is
complete. On SQLite and PostgreSQL, the database is the timing authority for
due/expiry checks and lease publication: PostgreSQL uses statement-time
`clock_timestamp()` rather than a transaction-start or host timestamp, and
SQLite uses its statement-time clock. A claimant locks the schedule row and any
recovery execution-log row before publishing its lease, then rechecks the same
claim token immediately before an effect is admitted. Renewals and the terminal
CAS use the same token-guarded, database-time boundary, so lock waits and host
clock skew cannot publish an already-expired real-backend lease.

A successful recurring completion resets the schedule attempt counter for its
next occurrence; recovery of an uncompleted occurrence retains the execution
identity and increments that occurrence's attempt. Renewal remains alive until
the terminal CAS finishes, including when the caller is being cancelled.

## Worker supervision and liveness

The polling loop is a replaceable worker owned by a scheduler supervisor. An
infrastructure exception escapes that worker, is reported as `restarting`, and
causes a restart with exponential backoff bounded between one and thirty
seconds. The escaped failure also latches host readiness fail-closed even if a
replacement worker later completes a tick; recovery does not erase the durable
evidence that scheduler infrastructure failed during this process. Runtime
telemetry runs on an independently-owned heartbeat, so a long cognition task
does not make a live worker look stale. Those writes are bounded, best-effort
observers and cannot kill or indefinitely delay the polling loop.
Occurrence-level failures remain isolated inside a tick and are
finalized independently, so one failed scheduled task neither cancels its
siblings nor kills the polling worker.

Every poller emits one row per `(authorized agent, worker owner)` pair to
`scheduler_runtime_status`, including agents with zero schedules. Publication
and freshness use database statement time, so skew between PostgreSQL replica
process clocks cannot make a current worker stale or preserve an old one. The
reader selects only the current freshness window (plus at most one prior-owner
diagnostic row), and publication reaps reports older than 24 hours, so random
per-process owner IDs cannot grow health-query work without bound. Fleet
inventory is loaded once per heartbeat and owner reports are batch-upserted.
The normal PostgreSQL bootstrap only inspects the existing composite key; an
exclusive table lock is taken only for the legacy single-key migration and has
a five-second lock timeout. The report contains worker state, tick timestamps,
restart/failure counts, and
exact runnable, executing, configured-enabled, paused, protocol-fenced, and
scheduler-disabled counts. Consequently, these states remain distinct:

- no runtime report was received;
- a current worker explicitly reports zero schedule rows;
- schedules exist but were deliberately paused by an operator;
- an occurrence is executing under a current claim lease;
- schedules are temporarily fenced by scheduler protocol; and
- scheduler safety policy disabled one or more schedules.

The Health feature reads these reports through the same scheduler-status helper
used by `schedule_list`. It aggregates only fresh workers and remains healthy
when any current replica is running, so a failing or stopping peer cannot
overwrite or flap a healthy peer. Missing or stale telemetry, no running worker,
or unclaimed work older than both the telemetry threshold and its misfire grace
fails the critical `scheduler_liveness` check. A due batch owned by a current
tick remains healthy while rows wait behind the bounded concurrency semaphore,
but that exemption expires after the telemetry threshold plus one claim lease
so a wedged tick cannot hide an overdue queue forever. With heterogeneous
misfire grace, the reported overdue age belongs to the row with the earliest
expired deadline rather than an unrelated older row still inside its grace.
Tick boundaries are stamped from database time as well.
A current worker with zero schedules passes. A live claim is reported as
executing and remains configured-enabled; only an absent or expired claim lease
is classified as protocol recovery. Scheduler-disabled rows warn with their
reason; operator pauses and completed one-shot history do not impersonate a
worker outage. Scheduler safety reasons remain disablements even on one-shot
rows and retain their explicit recovery contract. In particular, an
`execution_log_inconsistent` one-shot cannot be replayed at its old deadline or
have its marker cleared by `schedule_resume`; the execution log must be
reconciled and a new one-shot created. Durable reports carry a per-runner owner
ID. Standalone startup grace is bounded from feature initialization, while a
successful arm replaces that anchor with its database-time arm timestamp. A
ready hook that never arms therefore fails health after the finite grace, and a
prior `stopped`, stale, or same-owner row cannot mask missing current telemetry.

## Protocol-v2 rollout fence

The old scheduler only selected `agent_id`, `enabled`, and `next_run_at`. It
could therefore execute an occurrence that a v2 runner had claimed but not yet
made invisible to a stale read. Claim leases alone cannot make arbitrary
old/new rolling overlap safe.

Protocol v2 uses a durable, agent-scoped `scheduler_protocol_rollout` control
row and intentionally leaves `scheduled_tasks.scheduler_protocol_version`
nullable. Every v2 writer explicitly records version 2; a row written by an
old binary that omits that column is observable as unsafe state rather than
silently being accepted. The control row and schedule predicates are scoped by
the authorized agent DID on both SQLite and PostgreSQL.

It also records database-global schema provenance in the singleton
`scheduler_protocol_schema` row. PostgreSQL establishes that row under a
transaction advisory lock; SQLite establishes it under its writer transaction.
The provenance distinguishes a genuinely fresh v2 table from a pre-v2 table:
the first configured DID cannot make a second newly configured DID look like a
legacy upgrade merely because the shared table now exists. In a shared
PostgreSQL host, a non-polling preflight resolves every configured local DID
without initializing agents, establishes that global provenance, and seeds all
resolvable DIDs' rollout rows before concurrent feature runners start. A newly
added DID is therefore active immediately only when the global provenance is
already `fresh-v2`; an absent, unknown, or legacy provenance remains a
quiesced migration. A configured DID that cannot be resolved without creating
identity state is excluded from scheduler authority and latches a readiness
failure instead of creating a blank identity database or taking healthy peers
down with it.

If no global provenance exists and `scheduled_tasks` already exists (even if it
has no rows), first v2 startup records legacy-unknown provenance, enters
`quiescing`, creates a random activation nonce, disables/fences the applicable
rows, and refuses to run schedules. This strict transition prevents a legacy
replica with no current row for an agent from later selecting a new v2 row.

An operator must perform this explicit, safe transition:

1. Drain and stop every legacy scheduler poller **and every legacy process
   that can add, resume, or otherwise make a schedule runnable**. Remove their
   auto-restart paths and wait for their in-flight dispatches to finish.
2. Start a v2 candidate without an acknowledgement. Its controlled startup
   failure/log gives the per-agent activation nonce and leaves the agent
   quiesced; it is not a healthy mixed-version replica.
3. Verify the old processes cannot run or write, then restart/deploy v2 with
   `KESTREL_SCHEDULER_ROLLOUT_ACK=<nonce>` (a comma-separated list is accepted
   only when acknowledging multiple independently quiesced agents).
4. Only after that v2 activation may replicas and normal schedule writers be
   brought back. Remove the acknowledgement from later steady-state deploys.

Activation is a transactionally guarded control-state CAS followed by
conversion of precisely the rows fenced under that nonce. If any row appears
or changes after fencing (including an old writer's `NULL` version/nonce row),
v2 rotates the nonce and remains quiesced; the old acknowledgement is no
longer accepted. This makes the acknowledgement proof of an actual drain, not
a documentation-only warning. Never bypass the fence by editing the control
tables manually or by running old and new scheduler images together.

The fence covers both legacy-visible runnable rows and active v2 claim-bearing
rows. A claim whose fence wins has its ownership token revoked but keeps its
execution/occurrence identity as a disabled recovery claim; activation does
not re-enable it. Before an executor is called, the runner holds the DID's
active-control admission boundary and renews the exact token again. Thus a
fence that wins prevents the effect from starting; an already admitted effect
finishes its terminal CAS before the fence can commit. After acknowledgement,
a new v2 runner can recover the preserved occurrence with a fresh token and
the same execution/idempotency identity, rather than allowing an old in-memory
worker to finalize a recurring row back to legacy-visible `enabled=1`.

There is one intentionally conservative activation outcome. An origin/main
worker can (1) select an enabled overdue row, (2) dispatch it, then (3) re-read
`enabled` before its normal reschedule. If the fence changes that flag between
steps 1 and 3, the legacy worker leaves the old `next_run_at` overdue. V2
cannot prove whether that external effect happened, so a fenced non-claim row
that was already due at activation remains disabled with terminal status
`rollout_ambiguous_legacy_occurrence`. An operator must reconcile the effect
and explicitly resume that schedule; activation never guesses or replays it.
`schedule_list` identifies this as a scheduler-originated, recoverable
disablement and names `schedule_resume` as the recovery action. Resume is
serialized against the active rollout control row, recomputes the next cron
occurrence from the database clock, and clears the terminal reason only when
the caller passes `acknowledge_ambiguous_effect=true`. Invalid
persisted idempotency keys are not blindly re-enabled; they must be repaired or
the schedule recreated first.

Pausing remains permitted during quiescence and is serialized with the control
row, so a fence is not mistaken for an already-paused user request and later
activation cannot undo a real pause. Creating, resuming, and updating a
schedule require an active control row; update reads and writes its enabled
state under that same lock so it cannot resurrect a concurrent pause or resume.

Schedule idempotency bases are also bounded before persistence: the SDK allows
512 UTF-8 bytes and v2 reserves 65 bytes for `:<sha256-occurrence>`, so a base
may use at most 447 UTF-8 bytes. This preserves stable user-supplied keys
across lease recovery without truncating or hashing them silently.

## Hosted execution

`SchedulerRunner` remains the standalone per-agent executor by default. A
host with a shared database can pass `agent_id=None` plus a
`HostedSchedulerExecutor`, or the supplied
`AgentManagerHostedSchedulerExecutor`. The latter maps durable agent DIDs to
host configuration and invokes `AgentManager.load_agent` under a per-agent
lock when the target is cold. The host owns waking an unloaded agent; the
scheduler still owns claim, lease, idempotency, and completion semantics.

The host rechecks live scheduler authority at claim, dispatch, renewal, and
finalization time, rather than treating startup configuration as permanent
permission. Removing an agent revokes that authority under the same per-DID
lifecycle lock used for a cold wake, so a due occurrence cannot resurrect an
administratively removed tenant. Every cold registration also receives the
same app-owned A2A, feature-route, and asset onboarding as an autostart agent.
Protocol failures and unresolved scheduler identities are safety outages: they
are latched into public readiness as a sanitized HTTP 503, with detailed
diagnostics exposed only through the authenticated readiness surface.
