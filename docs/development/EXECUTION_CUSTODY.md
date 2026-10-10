---
type: Developer Guide
title: Transaction-bound hosted execution custody
description: Native host generation validation, exact advisory custody and cleanup boundaries.
status: proposed
privacy: public
tags:
- development
- storage
- authority
---

# Transaction-bound hosted execution custody

Core issue #3569 extends existing PostgreSQL advisory and native transaction
custody. This is a host authority contract, not a causation-derived permission.

`PostgresBackend.advisory_locks(..., on_loss=callback)` yields an exact-session
`AdvisoryLease` (or `None` for an empty, unobserved exclusion set). Registering
the callback is part of acquisition. `require_live()` rejects physical session
death, a detached asyncpg pool proxy, and completed retirement. A replacement
session cannot revive that capability. Callbacks are synchronous and invoked at
most once; hosts must synchronously latch denial and their fatal-loss policy.
Callbacks are notification, not instantaneous distributed fencing: asyncpg can
queue them after PostgreSQL has already released the lock.

`bind_execution_custody(fence)` owns one connection-free admission. Its host
`ExecutionFence` supplies immutable generation binding, `require_work()`, and
`lock_and_validate(connection)`. Copied children retain the same irrevocable
revocation state. A normal context exit retires that admission too. Foreign
turn executors carry the admission through the declaration-site turn registry;
binding a snapshot cannot discard a denying ancestor already on that task.

Native PostgreSQL queries validate using the actual mutation's connection and
transaction, before graph/file locks. Autocommit work becomes a short guarded
transaction only when custody is bound. Explicit transactions retain authority
locks through commit. Every query surface participates, including reads,
RETURNING, batch and script execution; SQL verbs cannot reliably classify
whether a query mutates. Validators must lock authority rows (FOR SHARE for
non-key generation updates), check the original identity/owner/generation, and
hold those locks through commit. They must not acquire a second connection,
commit the transaction or publish effects. SQL transaction-control statements
cannot escape a guarded native transaction. SQLite cannot silently execute a
PostgreSQL-authorized scope.

Tool dispatch checks admission after awaited permission hooks, immediately
before execution, and again before publishing its result. Already submitted
remote effects cannot be recalled by cancellation or process termination;
destination-side fencing is required for a stronger external guarantee.

The context owns no connection between operations. A stream retains its
original admission membership across consumer handoff, foreign continuation
and cleanup. A later consumer may add live scopes but cannot silently discard
an earlier retired/denying scope; reopening work requires a newly admitted
operation, not reusing that suspended stream. Children from a retired admission
cannot inherit its replacement. Consumers must join all owned work before
normal advisory retirement. Cleanup does not grant new cognition/tool/storage
authority; terminal identity/token CAS needs its own explicit native custody.

The native provider service retains the owning agent's runtime custody and
checks each adapter attempt and streamed chunk, including decision discovery,
pin canaries, decisions, embeddings and each internal retry after backoff.
`await_execution_work(owner, operation)` evaluates a lazy operation only after
binding original runtime custody and rechecks it before publishing a result.
Public streams pin the union of every consumer's admission through exceptions,
fallback and finalization, without leaving a binding installed across a yield.
Aborted hosted streams finalize under cleanup-only denial; ordinary usage writes
and billing callbacks cannot run there. Codex transport readers have a neutral
transport-lifetime context; each tool callback still binds its owning turn.
Authority errors are not provider failures and must never trigger fallback. The AgentManager
accepts a trusted per-agent `execution_custody_factory` before construction;
it also preserves an explicitly bound shared backend and refuses replacing
that binding. Multi-tenant hosts should use an unbound shared operational pool
and inject independent per-agent custody through the factory.

Concurrent discovery owns each lazy child: control failure cancels and joins
all siblings before choosing terminal evidence, including late uncertain commits.
Tolerant discovery returns only ordinary failures as values. Catalog worker
threads bind the original turn snapshot, and publication rechecks original
custody after lifecycle, cache and worker waits. Isolated wake/provision/restart
also checks the synchronous mutation and child-start boundaries; RPC success
does not publish a channel result after authority loss.

PostgreSQL scheduler execution records `executing` in the existing exact
occurrence log before target dispatch. Only exact-owner live finalization
resolves it into a known outcome. Lease loss, process death or post-dispatch
uncertainty does not authorize replay: recovery disables the schedule with
`unresolved_effect`, retaining the original occurrence evidence for operator
reconciliation. This deliberately trades automatic retry for duplicate-effect
safety; a supplied idempotency key alone is not proof of destination deduplication.
Renewal retains the original resolved runtime and exact advisory effect handle.
Pause retains unresolved occurrence identity, and resume/definition changes
refuse unreconciled `executing` evidence. Commit uncertainty cannot be normalized
into a terminal failed execution or clean Stop settlement. PostgreSQL carries
commit state through pool release/reset (including pinned operational sessions),
so release failure or cancellation after commit does not claim rollback.
Single-query, batch and script checkouts all carry that commit state. An
uncertain/committed outcome also irreversibly latches the original admission:
even a legacy caller that catches or stringifies the error cannot resume work
under it. Reconciliation requires a separately trusted new admission, not
clearing this state or rebinding a replacement lease. Source handlers,
dispatchers, dynamic/direct tools and provider transport callbacks preserve
cause-chained execution control errors. Parallel cleanup joins every child and
preserves any child's commit uncertainty over an ordinary sibling failure.

Provider send and isolated-tool RPC boundaries recheck after startup, traffic,
lock or wake awaits. Decision/embedding accounting retains the entire original
operation's scope; aborted accounting cannot spawn ordinary usage writes.
Every native stream forwarding layer explicitly joins its underlying iterator's
close under cleanup-only custody, including transport-handler cleanup. EOF and
close failures are classified before terminal invocation settlement. Boot's
READY transition and deferred readiness likewise retain/check original custody;
denial unwinds already committed boot phases rather than publishing readiness.

A2A boot reuses the guarded native storage backend, including DSN-only agents;
individual stores do not own its pool. Inbound verification/authorization and
peer tools retain the original runtime throughout the operation. Operational
sessions remember every participating admission until physical release and
revalidate before publishing even a successfully released result. Hosted usage
writers are joined through cancellation/timeout; Codex drains late inline-tool
control evidence before unregistering the turn sink. Scheduler cold preparation
is already an executing occurrence: uncertain bootstrap cannot advance or replay
the original schedule. A known feature-unavailable deferral restores only the
same live claim, never an uncertain occurrence under fresh authority.

Stream forwarding captures declaration-site turn capabilities before moving
execution to its source owner, including privacy-lock reentry and the requested
session. Conversation ownership is not delegated: child wakes queue for distinct
turns. A shared close record reaches that source and its
children before cancellation starts unwinding an actively advancing generator;
ordinary storage/provider/tool work is then denied, even while runtime custody
is otherwise live. Closing one stream does not revoke its reusable runtime.
Provider classification, retry and fallback preserve cause-chained control
evidence before ordinary failed-invocation accounting.

Scheduler effect markers lock the exact claim first and sample database lease
time in a separate statement after the row lock is held. Each confirmed renewal
also supplies a conservative monotonic deadline measured from before renewal
checkout. Synchronous effect guards enforce that deadline, and a separately
owned watcher expires/cancels/joins work even while renewal is blocked. A late
renewal acknowledgement cannot revive an expired admission; watcher and renewal
tasks are joined before occurrence ownership is released.

Custody-bound durable-signal boot uses transactional creation/repair of its
exact source-sequence index, retaining generation locks through commit. Unbound
host maintenance keeps concurrent index DDL. Large preexisting ledgers should
be migrated before runtime admission to avoid a blocking bootstrap index build.

## Delivery status

Forwarded streams establish their requested session before capturing the
explicit turn/privacy-reentry carriers. They do not delegate the conversation
hold: a child wake must queue for a distinct cognition turn. Scheduler cold
preparation and dispatch run in one joined owner under the original confirmed
deadline; expiry synchronously denies cancellation-resistant bootstrap children
and interrupts/joins the owner, without holding database locks across boot.
Cause-aware control classification applies before outer retries, stale-route
or canary results, timeout conversion, subagent envelopes and failed-wake
envelopes. Streaming settlement keeps unknown/committed evidence ahead of any
checkpoint denial. Ordinary provider/validation failures with live original
authority still record content-free accounting; control/cancellation cleanup
does not reopen ordinary accounting writes.

This is an implementation checkpoint, **not release approval**. Real native
tests cover session termination/release races, original-generation scheduler
renewal, retained runtime/backend denials, native graph/file/conversation writes,
fixed transaction membership/lock ordering, commit acknowledgement loss
(including cancellation), reanchor reconciliation and exact Stop settlement.
Hosted cleanup preserves failed checkpoints in the existing unresolved ledger
without reopening general work. SQLAlchemy cached/yielded execution remains
explicitly unsupported under custody; rollback/close remain available.

See the [canonical storage contract](../architecture/storage/STORAGE_ARCHITECTURE.md#hosted-execution-custody)
for installation and limitations. Full independent review, repository/CI gates,
immutable publication/verification and Frinz generation adoption/live acceptance
remain required; a source check is not evidence of downstream rollout.
