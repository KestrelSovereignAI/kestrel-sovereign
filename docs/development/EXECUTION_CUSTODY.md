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

The context owns no connection between operations. Streaming must retire and
release each admission before consumer handoff and use a fresh, separately
validated admission for continuation/send. Children from a retired admission
cannot inherit its replacement. Consumers must join all owned work before
normal advisory retirement. Cleanup does not grant new cognition/tool/storage
authority; terminal identity/token CAS needs its own explicit native custody.

## Delivery status

This is an implementation checkpoint, **not release approval**. Initial real
PostgreSQL tests cover exact session termination, immutable replacement denial,
stale generation across all native SQL methods, transfer serialization,
revocation rollback and transaction-control refusal. Scheduler loss wiring,
persistent runtime/alternate-session propagation, terminal cleanup, Frinz
generation integration and full independent gates remain before release.
