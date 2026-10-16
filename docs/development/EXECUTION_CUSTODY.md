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

Native response retirement retains each failed resource and its original close
task even when the SDK has already set an optimistic closed flag. Repeated
shutdown joins that task; it does not retry the close or erase unresolved debt.
Ordinary caller cancellation after a successful close is not cleanup failure.
Cold initialization, ingress release and reanchor cleanup preserve original
unknown/committed receipts ahead of cancellation. HTTP peer envelopes and
private ingress redaction carry only safe control/outcome classifications, not
private provider prose. Scheduler telemetry harvests cause-bearing cancelled
receipts before releasing the publisher task and latches them for readiness.

Stream lifecycle inheritance requires the actual constructing producer task;
a copied ContextVar in a sibling cannot confer privacy-frame reentry. Text
taps retain the original producer agent, admission scopes and request generation.
A TTS consumer cannot substitute its own agent or a replacement request with the
same opaque ID. Waiting for tapped text holds no privacy mutex; each native
synthesis validates original producer custody and fresh privacy policy. Direct
realtime cognition admission reads durable constitutional restrictions and the
completed hash-bound governance receipt before minting a session.
Captured constitutional validators are repeated before result/credential
publication as well as native dispatch. A tapped HTTP response retains the
same request generation through every outer forwarding layer's EOF and close.
Later revocation cannot replace an original failed native closer's debt with
an unowned denial.

Scheduler admission owns its original short database mutation through the
native COMMIT acknowledgement. Stop is reported after that owner settles and
before any scheduled effect begins; genuine commit uncertainty is preserved.
`execution_denial_scopes(error)` reports exact original scope provenance only
when all explicit authority carriers are scoped and none carries commit
uncertainty. Combining runtime and occurrence denials retains both scopes;
an unscoped authority error is never reclassified as occurrence-only retirement.

PostgreSQL moves only its native transaction exit and pool reset into original
owned completion tasks. Transaction bodies, context-manager yields and token
resets remain in the caller; arbitrary resolvers and gates are not shielded.
The strong native contract accepts an actual asyncpg pool. A caller-context
wrapper must cooperate through `PostgresPoolAdapter(pool, wrapper)`: `wrapper`
is an async context-manager factory receiving Core's managed native checkout;
it enters that manager and yields the same connection. Wrapper entry/exit and
their ContextVar tokens stay in the original caller. Core independently retains
and settles the registered native lease, even if wrapper exit omits it or tries
to suppress its terminal control. Arbitrary wrapper awaits remain cancellable.
Opaque legacy acquire wrappers retain ordinary caller-context behavior but
cannot admit execution custody, including custody introduced later inside an
initially unbound operational session. This restriction is explicit, not an
inferred native capability or a bypass of wrapper accounting.
The direct epilogues must be the actual native asyncpg context types, not
context-affine subclasses. A wrapper cannot suppress the original body failure
inside its managed lease: Core records it before invoking wrapper exit, and
does not publish after-commit callbacks for a rolled-back cancellation. After
the entire wrapper exit (including any post-reset await), all retained scopes
are revalidated at the final publication boundary.

Before issuing a committed receipt, one fixed `SELECT 1` on that same held
transaction proves it is not aborted: PostgreSQL can otherwise accept COMMIT
as a silent ROLLBACK. The proof is cancellable and precedes native commit
handoff. The proof itself is not acceptance of COMMIT: ordinary work revalidates every
retained admission after that awaited probe and immediately before native exit
ownership, without another await. Retirement at this boundary rolls the body
back; fixed native cleanup retains its separate post-retirement contract.
Known-commit custody must also own the top-level transaction, not
merely release a savepoint inside a wrapper's external transaction. Custody,
callbacks and fixed cleanup require this proof. Unfenced,
callback-free transactions retain their historical behavior but do not mint a
known-commit receipt without proof. Refusal to create an epilogue owner aborts
only its original native connection, retains the exact resource with no owner
as physical-retirement debt, preserves the original body error explicitly,
and denies the participating scopes. It is not a successful reset/rollback or
permission to retry. A failed native reset likewise aborts that original dirty
session; abort failure remains explicit evidence. The borrowed pool is never
terminated as a substitute for retiring its one lease.
Actual rollback/reset failures preserve the original body cancellation and
closer errors as explicit carriers. Failed physical reset denies every
participating scope even if no commit occurred. Error aggregation separately
retains the bounded unresolved-commit classification of legacy carriers; a
settled acknowledged receipt cannot erase unrelated reconciliation debt just
because the selected exception object changes.

`NativeExecutionObserver` is a passive, original-lifetime inventory for hosts
that hand off cancellation-resistant external providers. Bound observation
adds no execution scope or capability. PostgreSQL reserves the actual checkout
before acquisition can suspend and records only actual transaction-exit/reset
owners, before creating those tasks. Closing new observation admission cannot
cancel those epilogues. `join_accepted()` joins their physical owners and backend
classification independently of the provider's later continuation or exception
suppression; it preserves its own observer cancellation separately.

Classification notifications are not native ACKs or closers. Records remain
open across transaction exit and reset registration. A still-open acquisition,
transaction/operational body or adapter remains explicit resource debt, not an
unbounded provider join or successful physical retirement. Actual committed
facts and hidden UNKNOWN/failed-closer evidence survive alongside that debt.
Successful records are released promptly; unresolved resources remain retained.
The resource strongly retains its original checkout task and every actual
participating body task, including children using an inherited operational
lease. Completed participants are released by their actual task completion;
physical checkout completion removes the remaining participant roots and
callbacks. A detached child cannot disappear while its transaction/debt is
still rooted. Bind observation
before acquiring the original operational/transaction lease. A later observer
on an inherited or already-open unobserved connection is refused before SQL,
not silently attached to an empty inventory. Script SQL uses the same raw
transaction-control refusal as single-query SQL.
The cutoff also applies to NEW work on an already-retained connection.
Explicit transactions revalidate both original custody and observation after
checkout/operational-mutex waits, immediately before native BEGIN; savepoints
revalidate before native entry too. An already-accepted original exit/reset is
cleanup, not a new work admission, and is still physically joined.
An unscoped legacy autocommit query retains autocommit semantics (including
CONCURRENTLY DDL): observation owns its checkout/reset, not a fabricated
explicit-transaction COMMIT receipt. Opaque pools and raw transaction-control
SQL cannot bypass observation. Ordinary unobserved legacy paths are unchanged.
Native aggregation preserves an original enclosing cancellation when its
explicit cause would otherwise be selected; it never creates a reverse cause
edge or drops that cancellation just to avoid a cycle. A failing COMMIT also
retains the original observer cancellation alongside its native failure.
PostgreSQL opts its original epilogue tasks into `create_owned_task`'s
`retain_originals=True` policy at creation, before asyncio can reduce a caused
native cancellation to its cause. Generic task owners keep their established
cleanup-error priority; opting in grants no work or cleanup authority.
Errors or cancellation in wrapper code AFTER successful native reset carry
that same known commit receipt before classification, and all participating
scopes are sampled on that failure path too. They do not invent uncertainty on
healthy scopes merely because a wrapper stopped observing after retirement.
Cancellation while observing a successful native ACK is delivered on the
original cancellation object with an `AcknowledgedExecutionCommit`, not an
invented UNKNOWN. Its commit owner and shared original-reset record are
physical evidence, not an admission or a new generation. The checkout retains
one latest ACK task; earlier escaped receipts observe the same reset owner
without retaining every completed transaction in a long operational session.

After ACK/reset, authority loss is checked even when the caller was cancelled.
Only actual denied scopes retain the known-committed fact; healthy inherited
scopes are not poisoned. `execution_denial_scopes` returns an empty tuple for
a fully retired observer-only receipt, original scopes for a settled scoped
denial, and `None` for pending retirement, UNKNOWN, failed closers, legacy
unresolved commit carriers or any combined unscoped control. A known commit
must never hide another original unscoped failure. These facts remain control
evidence through wrappers and shutdown; they do not imply the whole external
effect completed or permit retry without durable reconciliation. Logical task
name prefixes aid tracing; the native owners are distinct tasks and names grant
no authority. Genuine native ACK/reset/rollback failures still fail closed.

Renewal teardown cooperatively stops the interval and physically joins its
in-flight native transaction before releasing the original advisory exclusion.
Concurrent cognition terminalization callbacks share one exact original-token
owner; its successful receipt prevents a late callback from manufacturing debt
by repeating a CAS against a token already cleared by that same owner.

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
Cancelled or explicitly closed hosted streams finalize under cleanup-only
denial; ordinary usage writes and billing callbacks cannot run there. Explicit
control errors bypass ordinary accounting even for unbound consumers. Ordinary
provider failures with live original authority still record
failed-attempt and known partial-usage accounting. Codex transport readers have a neutral
transport-lifetime context; each tool callback still binds its owning turn.
Authority errors are not provider failures and must never trigger fallback. The AgentManager
accepts a trusted per-agent `execution_custody_factory` before construction;
it also preserves an explicitly bound shared backend and refuses replacing
that binding. Multi-tenant hosts should use an unbound shared operational pool
and inject independent per-agent custody through the factory.

The built-in standalone PostgreSQL server explicitly creates one
`ProcessRuntimeExecutionFence` per agent construction. This declares only the
original process-owned runtime lifetime: it is not a tenant row, distributed
generation or scheduler occurrence. Cold boot checks the launching occurrence
as well, while published resident services retain only the original runtime.
Shutdown retires that local lifetime before yielding. It never retires borrowed
tenant/shared-backend custody; multi-tenant hosts must supply their own native
generation factory. Provider construction validates and binds the original
custody first, and manager-supplied services reuse native agent usage storage.
Both AgentManager and the no-manifest single-agent startup branch supply this
original lifetime before provider construction. Failed single-agent startup
retires it before the real boot coordinator's first rollback await and before
awaiting the owned LLM-service close, including cancellation;
SQLite startup remains unbound. Native startup regressions exercise this
production selection rather than injecting an agent's custody afterward.

Host inception and birth replay may open only their exact local
`kestrel_prime.db` through a private lexical identity-anchor capability. It
retains the SAME original admissions, never drops an ancestor, and expires
irreversibly after the owned database closes; retained handles and copied
children cannot reacquire it. Exact archive reads require the native cold-read
backend. This does not authorize tenant SQLite, arbitrary paths, or general
PostgreSQL bypass: runtime birth replication still validates the destination's
original generation on its actual PostgreSQL transaction. Native file safety
is checked before granting anchor access: symlinks, hard links, directories and
other non-regular leaves retain the public file-validation refusal without
opening, moving or modifying their targets. Inception embedding
and accounting consumers borrow the owned birth database, not a separately
created usage database, and their native provider closure joins before that
database retires. Physical closure never reconciles an uncertain native effect.

Delivery and health workers use the same explicit first-party resident handoff
as scheduler and peer services: they wait for READY and retain the original
runtime, not the completed cold-start occurrence. Ordinary task children retain
every ambient admission.

Provider finalization opens the host-key store only for deferred routes that
need it; non-master payer policies do not open it unnecessarily. Standalone
process-only custody can access the existing operator-owned SQLite `host.db`
through a private host-table accessor retaining the SAME process lifetime.
Its native SQLite authorizer excludes agent/graph tables, database attachment,
export and schema replacement; the ordinary SQLite executor remains forbidden.
Retirement still denies reads/writes, and post-commit loss preserves committed
evidence. Borrowed row/advisory/occurrence custody cannot use this accessor:
multi-tenant hosts must inject their native PostgreSQL host database. The agent
owns and closes a cached local host accessor on boot rollback or shutdown.

The mandatory EPHEMERAL shutdown sweep has a fixed native cleanup entry point
for that original retired standalone process only. It checks the canonical
initialized storage, original backend and agent identity, and freezes the
original entry watermark before acquiring one original-pool connection. Private
connection-bound copies run the existing conversation/lexical, graph ownership,
channel and projection purge algorithms, preserving their destructive audit
sink. No executor escapes, no provider or migration starts, and live storage
and copied children keep their denying original custody. Borrowed tenant
generations cannot use this local-process cleanup entry point. Failure remains
an uncertified purge, not a clean no-trace receipt.

Owned iterators retain the original terminal exception independently of
asyncio task retrieval, which otherwise discards cancellation causes after the
first read. Requested close consumes only a private, control-free interrupt.
Both later closure failures and caller cancellation preserve native authority
and irreversible commit evidence. Completed-effect checkpoint failure likewise
selects terminal evidence across the turn and checkpoint instead of making an
ordinary turn failure retryable.

Parallel tool joins select terminal evidence across the original error and
every joined sibling, including a late pure authority denial behind an ordinary
cancellation. Both ordinary and streaming request cleanup classify that control
carrier as ABANDONED, not an acknowledged Stop.

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

Normal stream EOF is distinct from an abort: successful source exhaustion does
not mark already queued, independently admitted child turns cleanup-only. Abort
or failed close still denies the source and its copied children before unwinding.

First-party resident services have an explicit runtime-lifetime handoff. Their
creation checks every current admission, and effect loops wait for successful
READY publication (which checks those admissions again). Supervision, idle
monitoring, heartbeat, resume monitoring and standalone scheduler loops then
carry the SAME immutable runtime custody, not the retired request/occurrence
that cold-booted them. Durable owner-heartbeat timers may protect admitted boot
work before READY; their rollback ownership remains explicit. No effect child or
foreign turn receives this handoff: ordinary background tasks still inherit all
denying ancestors. Revoking the original runtime still denies resident work.
Cancellation and typed Stop/self-fence carriers preserve cause-chained native
control evidence before classification: an unknown/committed cognition effect
is failed, non-ACKable and non-retryable, including retained late completions.
Renewal loss carries its original exception through cooperative cancellation
or retained late route completion, rather than reducing control to a reason
string. Exact-token failure NACKs and fallback releases use the same control
settlement as success ACKs. Managed cognition retry releases (including Hold)
retain the original owner/token, without a live lease. Both polling and exact
event claims exclude a sibling while that owner is live; the original live
scope may retry normally, and stopped/stale owners keep ordinary at-least-once
crash recovery. Lost release ACKs can therefore be settled by the fixed
original-token terminal CAS without exposing work to a sibling. A receipt
whose terminal CAS no longer matches remains explicit cleanup debt: it is not permission to stop the owner
or manufacture another delivery. No terminal receipt is written while a
retained route can still perform effects.
Renewal retirement joins the original child even under repeated cancellation
of the context body or its closer. Late native commit/control evidence is
selected before the body's cancellation can bypass receipt settlement. A
retained ordinary route also classifies failures from its eventual native
lease release. Confirmed joined terminalization clears debt before propagating
caller cancellation; a failed child retains exact debt and blocks retirement.
Detached ingress completion pins its publication-time custody and rechecks
before every RPC, including after retry backoff; a replacement runtime cannot
authorize an old callback. Cause-carried control from a cancelled facade is
harvested from the original owned operation before shield cancellation can
erase it, and does not become an ordinary completion retry.
Peers' restored-question replay and hourly backstop use one feature-owned
resident driver after READY. Restored subscriptions, deferred terminal-signal
joins and retries inherit that original runtime root. Newly committed outbound
questions likewise publish only their registered subscription as a resident
source, after POST/correlation insert under live caller admission; subscriptions
still verify the persisted outbound route and current peer scope. Ordinary
children receive no such handoff. Feature disable and boot
rollback still cancel/join exactly the feature's owned tasks.
The A2A expiry/receiver-settlement integration starts exactly one hourly
backstop, alongside the feature-owned boot settlement and lease-delay recovery.
Claim checks, renewals, confirmations, receiver reads and every replay/sweep
retry boundary propagate original control receipts, not retryable non-delivery.
Lease-renewing delivery waits and bounded receiver settlement physically join
their original native child on cancellation/deadline; a late commit receipt
wins over that interruption. Delivered-wake retirement and cancellation restore
likewise join their original row-write owner before releasing the producer.
SDK event-reader callbacks are explicit new inbound events, not continuations
of the completed boot claim. The registered exact-client handoff joins its
callback under the same original runtime; owned routing waits for READY. It
cannot redirect to a replacement client or generation. The now-tracked resume
observer is infrastructure for restart-idle classification, not permanent user
work, and its callback preserves cause-chained control failures.

Explicit cognition control errors, including cause-wrapped unknown/committed
outcomes, must escape the durable result normalizer. Its exact managed delivery
is terminal FAILED, never ordinary RETRY or terminal-ACKable; a retained route's
late control outcome uses the same rule. Native terminal cleanup is a fixed
original agent/consumer/delivery/owner/token CAS on the existing backend, not a
generic SQL executor or new admission. No new provider cursor acknowledgement
is inferred. Failed terminal writes remain exact-identity cleanup debt and block
clean owner release. Cleanup liveness can update only an existing unstopped
managed owner that still holds a leased or receipt-pending retry cognition,
under the canonical recovery
serialization key; it cannot insert or revive an owner or authorize ordinary
work. This is live-process debt ownership, not a new guarantee of durable
terminal evidence while PostgreSQL is unreachable or after process loss.
Runtime retirement also retains fixed original-backend cleanup for unactivated
initial reservations and exact original raw-handoff owner/token compensation.
It cannot create an owner, transfer a lease, clear a successor's token, or admit
general SQL. Idle shutdown can stop its existing owner and close storage after
ordinary authority has retired. Cancellation-resistant cognition instead keeps
the original owner live through cleanup-only heartbeat/re-arm until its owned
work is joined; only then is that owner marked stopped.
This protection also applies before shutdown and before terminal cleanup debt
exists: a still-running retained cognition owns the exact existing lease.
Its heartbeat and timer re-arm must not request live runtime admission after
revocation, recover deliveries, or admit replacement cognition.
Ordinary idempotent cognition retains its documented at-least-once
crash recovery contract; these explicit control outcomes require reconciliation.
Assistant persistence, response audits (including their hook manager), and
conversation embedding fallbacks preserve control evidence instead of allowing
cancellation, advisory policy or keyword fallback to erase it. A default native
LLM service reuses storage's initialized PostgreSQL usage database before boot
consumers; an externally supplied service retains its own host binding.
Optional wake-reply ID readback and the console's single-message endpoint also
propagate original native control/retirement failures before applying ordinary
lookup, decryption or HTTP-error normalization. A missing reply or ordinary
database read error keeps its established optional/UI behavior; an original
failed RESET and its resource/closer are not a missing message.

Scheduler effect markers lock the exact claim first and sample database lease
time in a separate statement after the row lock is held. Each confirmed renewal
also supplies a conservative monotonic deadline measured from before renewal
checkout. Synchronous effect guards enforce that deadline, and a separately
owned watcher expires/cancels/joins work even while renewal is blocked. A late
renewal acknowledgement cannot revive an expired admission; watcher and renewal
tasks are joined before occurrence ownership is released.
The canonical `ConfirmedLeaseDeadline` owns this monotonic lifetime algorithm;
scheduler state retains its existing stop/custody fields and delegates its
watcher to that same implementation. The deadline alone is not distributed
authority. Downstream hosts must retain their exact-token validation, attach
their backend-specific execution fence, and physically join original native
COMMIT/RESET owners even when the confirmed interval expires. Invalid/nonfinite
or already-expired confirmations fail closed and cannot revive lost work.

Custody-bound durable-signal boot uses transactional creation/repair of its
exact source-sequence index, retaining generation locks through commit. Unbound
host maintenance keeps concurrent index DDL. Large preexisting ledgers should
be migrated before runtime admission to avoid a blocking bootstrap index build.

## Delivery status

Standalone host-key SQLite writes revalidate the original process custody
after acquiring the actual writer lock and after native statement completion,
before COMMIT. Host scripts use the existing owning transaction, so retirement
cannot publish a partial multi-statement script. A successful COMMIT followed
by loss still preserves committed evidence rather than permitting replay.
Every host write form uses one native COMMIT handoff. An interrupted or failed
acknowledgment preserves `unknown` evidence on every original retained/ambient
scope before cleanup; a later rollback failure or repeated cancellation cannot
erase that evidence. Rollback after COMMIT is not proof that the effect vanished.
Selected legacy error wrappers retain their original explicit control cause:
later rollback or A2A readback failures become context, not replacement causes.
All database factories use the same joined retirement boundary; cancellation
after uncertain initialization cannot supersede the original COMMIT receipt,
and unsuccessful cleanup retains its database owner and both failure objects.
A2A producer, command, worker, completion, and best-effort projection boundaries
propagate cause-carried control instead of converting it into a task error,
retry, completion wake, or successful projection. A durable terminal token
permits ordinary transport reconciliation, never new authority after retirement.
Store retirement still attempts every owner, selecting irreversible control
ahead of an earlier ordinary close failure when reporting joined failures.
Registered feature handlers and public task tools follow the same rule:
neither a live parent admission nor a narrower child's completed native effect
permits conversion of control into a fresh FAILED result or workflow retry.
Config loading/persistence cannot normalize control as missing configuration
or a best-effort write failure. PostgreSQL native rollback and physical pool
release retain the original body receipt even if their cleanup raises another
error or cancellation. Explicit exception-group members retain their strongest
control evidence; incidental implicit handling context does not grant authority.
Background task managers harvest original child cancellation before discarding
completed tasks and retain unresolved receipts through subsequent drain/close.
Feature-owned ordinary and resident tasks share that first terminal receipt
with their agent owner. Completed-task removal cannot erase it, and repeated
joins cannot replace a cause-bearing cancellation with a fresh cancellation.
Feature retirement still unregisters its sleep hooks after a control-bearing
child failure. Full feature removal attempts every independent inverse and
selects irreversible control evidence ahead of an earlier ordinary cleanup
failure, retaining the failed feature owner for later joined retirement.
Signal terminal-delivery helpers preserve the same original dispatch receipt
before non-delivery, cancellation or detached-enqueue normalization. Peers
cannot restore a completed-effect question to WAITING on that control path;
neither a delivery supervisor's cancellation cleanup nor local finalizer may
turn it into an automatic retry. Ordinary delivery failure and control-free
shutdown still retain their documented restore/retry behavior.
The real agent shutdown joins every later resource owner before reporting that
receipt. Physical retirement is not reconciliation or an acknowledged Stop.
The earlier durable-admission Future also preserves original control instead
of publishing retryable NOT_ADMITTED: channel cognition/terminal ingress and
the isolated producer cannot issue a provider retry or ACK from that error.
All four isolated ingress owners (detached, deferred, legacy queue and provider
ACK/retry completion) harvest the same first owned-task receipt before removing
their task/client references. One strongest control receipt survives physical
proxy retirement and denies later ingress/completion; successful provider RPCs
recheck the original admission before publishing their result. Terminal joins
harvest completed owners independently of callback scheduling. Ordinary live
provider failures retain their existing bounded retry behavior.
Late facade operations, including health/lifecycle and ACKs beyond their
cancellation grace, harvest the same first canonical receipt before removing
their retained owner. Same-loop Future/Task receipts remain exact; foreign
owner loops consume private source outcomes locally and export only a newly
constructed fixed authority/commit category, never the private exception graph.
Privacy mandates, state writes, restoration and recovery reads preserve these
same native control carriers before ordinary refusal/best-effort handling.
Constructed-Agent and batch rollback select original initializer receipts only
after all independent cleanup joins. Isolated stage, promote, abort and legacy
authority reconciliation likewise quarantine without retrying a native
uncertainty as a fresh read/CAS. Lifecycle deadlines retain the first child
receipt without accumulating abandoned waiter callbacks or reader tracebacks.
Explicit terminal retirement retries serialize on the same exact-client lock;
another owner's uncertain stop cannot silently skip that fresh bounded retry.
Agent, Feature and A2A background owners retain one strongest control receipt
(unknown commit before committed before authority, first identity at equal
strength), not an append-only traceback list. Completed task graphs are released.
Model/embedding preference callbacks do not swallow native
control, and their completed-task drainer retains one strongest original task
receipt. Late unbound modality accounting keeps its existing two-second caller
deadline, but a native COMMIT/ACK failure afterward survives task removal and
is reported by drain/close. LLM close bounds each join and its cancellation
grace (100ms), retains the exact unfinished operation for later close, and
attempts independent resources before propagating strongest control. Caller
cancellation hands remaining resources to these same service-owned tasks,
without another full wait per resource. Pending cleanup raises explicit debt;
usage storage waits for physical writer settlement and never closes underneath
a retained writer. Late completions harvest one strongest receipt, and retries
join the same operation rather than duplicate an uncertain native effect;
closing physical resources never certifies reconciliation of an uncertain write.
Actual agent retirement also fences its borrowed preference/accounting storage
and local host-key accessor behind these same native LLM owners. The bounded
shutdown attempt hands the dependency to one durable continuation; the explicit
host/reaper join settles the original owners before storage or serving-record
release and reports their strongest original receipt. Failed construction
registers the unpublished provider before cleanup awaits. Pending or uncertain
cleanup keeps that exact provider and original scope in manager/server custody,
blocks replacement birth and shared-pool release, and is retried by host teardown
without recreating an uncertain native close. Ordinary settled close failures
retain their compatibility behavior. Server cancellation/group handling cannot
strip a native control cause or hide it behind an earlier ordinary failure.
Terminal host drain joins the unpublished provider's physical close owner,
not another bounded close attempt. If construction produced an Agent before
initialization failed, the host retains that Agent privately instead of closing
its provider beneath cognition, storage, or a rollback continuation. A failed
host-key database factory likewise transfers its actual retirement owner into
Agent custody before propagating the original initialization receipt.
Every lifecycle consumer joins the exact continuation before reporting any
outcome, including cancellation-containing groups. Child timeout cancellation
does not manufacture caller cancellation. Fleet/quarantine retirement and
delegated result reads retain the original strongest native control receipt;
completed-task removal or bounded diagnostic eviction cannot acknowledge it.
Retained native receipts freeze their original explicit-carrier tracebacks:
repeated CAS heartbeats or delegated-result reads preserve the same exception,
cause/group and commit outcome without retaining every later reader frame.
Batch startup, admission release and PostgreSQL pool/scheduler rollback select
the strongest native receipt only after independent cleanup has settled; an
ordinary startup failure or caller cancellation cannot replace that receipt.
Failed single-agent birth records native control on the actual retained Agent.
Private-route renewal carries the original runtime and ambient scopes across
the coordinator lock and revalidates immediately before provider touch. A
bounded host close cannot cancel the route's physical request/client drain.
Rotated and unpublished route clients use the same native-close registry as
ordinary providers; uncertain close is retained, never recreated on a retry.
Cancellation while acquiring the publication lock or draining the prior route
physically retires the newly constructed, unpublished client through that same
registry before reporting the original strongest receipt. The pinned old route
remains owned, and a started drain stays fail-closed; cancellation never
certifies its release or silently re-arms it.

Durable owner-heartbeat failure re-arms bounded cleanup liveness even before a
late cognition owner or terminalization debt is installed. Re-arm retains the
denying original context, not a fresh admission. Cognition and cleanup callbacks
harvest cancellation-containing exception groups before removing owners; native
control terminalizes the original exact delivery rather than releasing it for
retry. Spawn tools likewise preserve a distinct child's control denial when
the parent's own admission is still live.
Deferred OpenRouter provisioning validates the original admission after client
acquisition and immediately before each POST, including retries after backoff.
Cause-carried control errors bypass retry/error normalization; ordinary live
rate limits retain their existing bounded retry policy.

Idle-restart inspection ignores only the exact original live occurrence of
its own scheduled restart coordinator, carried by the existing revocable
scheduler scope across awaited dispatch tasks. Other schedules, same-named
tasks and co-hosted work still block. Delivery and health polling residents
are infrastructure; actual delivery attempts, including their durable receipt,
are separately tracked ordinary children and remain restart blockers.

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

An exact cognition terminalization or cleanup-liveness CAS returning no match
is not proof of successful cleanup. Original control debt remains retained,
shutdown remains fenced, and successor-owned deliveries are never overwritten.
Only the original retry capability is eligible for exact terminal cleanup.
The resident store also retains local release-receipt custody until its native
ACK joins: neither its own drainer nor an exact-event callback may replace that
token while the release is outstanding. Successful ACK restores normal
same-owner retry; control-bearing failure retains exclusion until exact cleanup
succeeds. Executable drain queries apply the same live-owner exclusions before
their limit, and excluded rows trigger bounded liveness polling rather than a
zero-delay database/task loop. No additional database lease or pool is created.
Repeated retries retain bounded diagnostic notes rather than
manufacturing another owner or acknowledgement. Operator reconciliation is
required when the original lease can no longer establish terminal state.

The first distributed Stop admission settles control-bearing renewal failures
as `ABANDONED` before asynchronous cleanup can claim `COMPLETED`. Both native
HTTP and bridge invocation/streaming routes classify explicit control causes
before ordinary Stop, retry, refusal or provider-error normalization. Source
closure and transport cleanup select original terminal evidence before later
cleanup noise; an abandoned stream never becomes a retry recommendation or
an acknowledged Stop.

The process-wide distributed Stop registry retains its original host/storage
custody separately from caller scopes. A pure, originally scoped tenant/turn
denial remains on that caller's receipt and does not fence unrelated active
generations. Loss of the registry's own original custody, unscoped denial,
uncertain commit, or native retirement debt still fences the shared owner;
caller-local classification never grants a replacement admission or receipt.

Explicit native retirement debt is execution control even when the original
closer raised an ordinary exception. The canonical control classifier preserves
that actual resource/closer evidence through availability and tool-error paths.
An outer tap HTTP body retains its original failure and close task: failed
body/physical retirement releases the original generation as `ABANDONED`, and
repeated close returns the same receipt rather than successful completion.

Scheduler finalization stops future renewal intervals only after acquiring its
original live-token row lock. Its own committed token clearance is not external
lease loss; original in-flight renewal/finalizer acknowledgements still join,
and the original confirmed deadline is never extended by this stop signal.
Cognition terminalizers share pending native tasks, but successful completion
retains only a weak original-delivery receipt. Completed tasks and their signal
contexts are released; the receipt expires with its original delivery owners.

Scheduler dispatch classifies every explicit execution-control carrier before
ordinary task-failure and typed feature/owner/unavailable normalization. Even
when only a child admission is denied and the outer claim remains live, the
original executing occurrence stays unresolved; no failed finalization, claim
clearance, or recurring schedule advancement can acknowledge that effect.

Local identity anchors and process host keys use one SQLite native COMMIT
handoff. It retains every original admission before queuing COMMIT, latches
uncertain acknowledgements before rollback cleanup, and checks those same
admissions after successful acknowledgement. Retirement after a successful
COMMIT is reported as committed, never as rollback. Bound scripts use that
owning transaction rather than sqlite3's implicit executescript commits.

Native SDK, HTTP transport and usage-database close owners are not cancelled
merely because a bounded caller's join expired. The original owner remains
pending until physical close finishes; terminal host retirement joins it before
closing borrowed storage. Provider is_closed flags alone are not physical-close
evidence. Batch initializer recovery joins the original initializer Tasks
directly through repeated cancellation before retiring all unpublished agents;
it does not need a new collector Task to recover a successful private result. Inception,
embedding/recall compatibility fallbacks and modality accounting preserve native
control receipts before ordinary error/cancellation normalization; uncertainty
cannot authorize another INSERT, vector rewrite or legacy retrieval retry.

Refusal to allocate an original physical closer is explicit resource/None
retirement debt, not successful closure: HTTP response inventories keep the
actual handle and database shutdown still attempts each independent backend
owner. Refused unstarted coroutines are closed. Partial iterator registration
or watcher startup retains and cancels its original dispatch-gated producer;
the async forwarding boundary joins that same producer and watcher before
reconciling only its exact constructor reservation. No source body is admitted
on registration failure. SQLAlchemy read/write sessions use one lifecycle that
explicitly aggregates original work/COMMIT denial with rollback and exit errors,
instead of relying on implicit exception context. These are candidate contracts,
not a claim of whole-branch approval or public artifact verification.

Failed boot joins the original LLM preference/accounting and native provider
owners before retiring their borrowed database. Dispatcher cleanup control is
not a request to retry cancellation indefinitely. Every independent feature
and resource inverse is attempted even when another returns a cancellation or
native exception group; FAILED is published before the strongest original
control receipt propagates. The serving-record audit remains conservative for
uncertified close owners even after physical cleanup finishes. Optional startup
initialization, inference-lease tools and model selection preserve explicit
control causes/groups before formatting ordinary failures or taking fallbacks.

Native SDK stream acquisition retains the opened resource through post-open
authority validation. A rejected handle is closed under cleanup-only custody
before its original denial escapes. Every adapter forwarding layer owns and
joins its inner iterator, and native response closure preserves original
control receipts even when Stop races a completed child or cleanup fails.

Authenticated hosts may admit a care-only Telegram acknowledgement through
the durable non-cognitive terminal path. Its sanitized content reaches
registered workflow observers, but no cognition delivery is created. A child
wire descriptor cannot request this host-only classification; a later ordinary
retry cannot upgrade its retained source event into cognition. The host must
fence the current outstanding wait, reply identity, owner, route and config
generation through admission. This does not bypass privacy, sender or Hold
policy, or replace Workflows' signal/deadline CAS.

SDK HTTP responses retain one native close owner before iteration, so an
implicit SDK/HTTPX EOF close and explicit adapter teardown join the same
physical resource retirement. SDK generators that hide their response (Ollama)
install that same owner at HTTPX's pre-body response hook, capturing each
request's own custody rather than the first caller's scope on a shared client.
The custody hook runs before existing response observers, since an observer
may itself consume/close the body before SDK parsing begins. Native HTTPX
types are retained independently of catalog test doubles and client factories.
Non-streaming chat, embeddings, model discovery, files, batch operations and
raw SDK calls install that same response hook before dispatch. HTTPX's
internal read/close therefore cannot retire the response without joining its
one retained native task. Best-effort metadata/embedding fallbacks preserve
explicit control causes/groups rather than returning an empty successful
result after uncertain cleanup.
A cancelled successful Codex session-lock
acquisition releases that exact lock before reporting cancellation.

Dispatcher withdrawal joins both monitor and cognition owners before reporting
validation withdrawal; a worker's original uncertain/committed receipt wins
over cancellation. Event persistence, activation, reservation repair, legacy
upgrade and outcome writers preserve original control evidence rather than
returning retryable failure. Outcome control is retained in one bounded
dispatcher fence even when the writer completed before shutdown joined it;
accepted writers and repairs settle before the owner can be released.
Each dispatch and outcome-drain reader restores the fence's frozen traceback
before raising the same receipt, so repeated heartbeat/retry reads cannot
retain an unbounded stack of previous consumers.

Private terminal-only ACK retries bind the original normalized envelope with
the existing per-agent keyed integrity mechanism. The terminal marker is
reasserted only after proving the retained terminal consumer row; ANONYMOUS
retains no extra plaintext, and privacy-elided channels use that verified
terminal receipt without creating a cognition delivery. Unverifiable legacy
private content and altered sender/content/correlation metadata fail closed.
Keyless NORMAL channel receipts use the retained source payload's original
sender as replay identity; their validated opaque caller label remains an
audit identity and is never compared to a raw provider sender identifier.

Inference acquisition/reconciliation captures original runtime membership before
waiting for its coordinator lock and checks it before provider acquisition;
replacing the service's current admission cannot rescue the waiting caller.
Scheduler effect/watch/renewal retirement and Codex inline-tool checkpointing
select the original strongest body/child receipt only after physical joins.
Completed health writes retain control evidence before pending-owner removal,
and repeated feature shutdown reports that same receipt. Delivery queue tools
preserve native database control instead of converting it into a failed result.

SQLite retirement retains the original unfinished close task even after its
native worker has exited. Repeated caller cancellation cannot shed that later
receipt, and a benign interrupt message cannot override an explicit control
cause. Generation/tool-loop errors and anti-repeat checkpoint errors are
selected together: a weaker checkpoint failure never replaces an original
unknown/committed result. Operator audit and fallback-conversation writers
propagate native control instead of requeueing an uncertain persisted notice.
Shared embedding discovery uses one physically joined owned task, so every
coalesced reader retrieves the same original receipt, including cancellation
carriers whose cause would otherwise be consumed by the first reader.

See the [canonical storage contract](../architecture/storage/STORAGE_ARCHITECTURE.md#hosted-execution-custody)
for installation and limitations. Full independent review, repository/CI gates,
immutable publication/verification and Frinz generation adoption/live acceptance
remain required; a source check is not evidence of downstream rollout.
## Native peer and constitutional receipts

Temporary local-peer HTTP clients and their responses retain the original
physical close owner. Repeated caller cancellation cannot discard a transport
or response body merely because HTTPX has already marked its facade closed.
Subscription deadlines may end ordinary waiting, but an explicit control
receipt carried by the timeout is propagated, never normalized to EOF.

Constitutional-state persistence likewise preserves native committed/unknown
receipts. An authorized Safe Mode exit may have committed before its connection
release acknowledgement fails; that is a reconciliation requirement, not proof
that the state was never persisted. The in-memory Safe Mode remains closed until
the original outcome has been reconciled.

## Bounded observation is not native settlement

`wait_owned_or_pending` observes an already lifecycle-retained task. The first
caller cancellation starts one finite positive grace deadline and forwards at
most one stop request; subsequent cancellation cannot extend that deadline or
reinterrupt cleanup. `PendingOwnedTask` means the observer abandoned unfinished
work. It is neither a terminal task outcome nor an unknown commit receipt.
Callers must retain the exact owner, harvest its late native evidence and join it
with `await_owned_task` at the owning lifecycle boundary.

`ReentrantTransitionLock.start_owned_native_work` retains a fixed body before
dispatch, including under eager task creation. An actual admitted foreign-mutex
owner registers that exact continuation as its frame dependency; a copied token
alone grants no mutex bypass. Other workers enter the ordinary serialized path
and recheck live policy there. Foreign-frame exit drains its dependencies before
unlocking, and the original base turn retains its existing physical drain.
`privacy_transition` joins entered native dependencies before changing policy;
queued, not-yet-admitted workers revalidate after acquiring the lock instead.
These enclosing joins can outlast the observer grace. Native unknown/committed
receipts and repeated lifecycle observations always retain the original carrier.

`ExecutionStreamLifecycle` is an opt-in physical lifetime for lazy stream
producers. Its factory receives the lazy body and original context; its start
callback registers the actual producer task before any source step, including
under eager task creation. It is inherited only within opted-in source work,
not by unrelated Core cognition streams. The lifetime admission runs before
native advancement and before returning each item to a consumer. Shutdown must
close paused iterators as well as join their exact tasks; a task join alone
cannot release a producer waiting for another consumer pull.

`capture_native_work_factory` retains the constructing owner's held privacy
frame for a deferred source without starting it. A live foreign frame permits
only its exact registered native tasks to reenter and to register transitive
native continuations. An ended frame cannot be revived, and an arbitrary child
with copied ContextVars still queues on the ordinary reentry mutex. Foreign
exit drains newly registered dependencies until the original graph is empty.

## Successful stream drain and native cognition dispatch

Text EOF is not Stop. A successfully completed tap retains one reference to
the exact original request generation before text endpoint cleanup. Its one
speech owner releases that reference only after queued text and final buffered
synthesis physically drain. Stop and abandonment continue to deny this same
generation; same-ID redelivery cannot lend a replacement. A late Stop declines
the handoff without retroactively failing an already-published text response.
Unclaimed completed queues expire after 30 seconds. The actual text privacy
owner transfers its held-span token only to a matching agent/request generation;
speech waits for text without holding that span and validates each synthesis.

`bind_native_execution_admission` carries original trusted asynchronous read-only
admission validators to native HTTP dispatch. Each request captures its validators
before observers, and the original transport runs them after every observer and
immediately before sending. Realtime cognition binds a fixed constitutional
revision/generation/hash, including through awaited model discovery. Durable
healthy rows never override locally unpersisted Safe Mode or pending restrictions.

Deferred native resources also retain their original read-only gate inventory
through `capture_native_execution_admission` and
`bind_native_execution_admission_snapshot`. Original gates join current caller
gates at a handoff; a later caller cannot replace the original resource's policy.
`require_native_execution_admission` revalidates that inventory before an owned
child's actual lazy dispatch and before final disclosure. Execution stream
forwarders retain the constructing inventory, checking both physical source work
and final consumer delivery, including buffered bytes. Trusted turn-relative lock
reentry is bound only while validating, never across delivery to a foreign caller.
This does not mint a replacement scope or recall an already-submitted effect.

Immutable launch artifacts selected inside a mutable namespace retain a durable
restrictive marker on the owned namespace, not on the VENV/BIN itself. Canonical
no-follow cleanup refuses that marker before deleting any contents, including
cold offboarding after proxies, process settings and manager witnesses disappear.
It is cleanup denial, never provisioning/launch authority or a cleanup override.
Session-only ISOLATED file bytes retain their matching metadata in session memory;
session clearing/EPHEMERAL purge clears both without persistent file writes.

Runtime-owner heartbeat teardown joins its original native SQL operation rather
than cancelling a possibly submitted COMMIT. An ordinary failed native closer is
retained background retirement debt even without a commit/authority exception;
successful physical close plus caller cancellation remains cancellation alone.

Browser realtime cancellation closes response admission even before a response
ID is known. A buffered pre-cancellation input commit cannot reopen it: new audio
requires the original ordered input-buffer-clear acknowledgement followed by a
fresh commit. Late response IDs and tool arguments remain cancelled; user input
transcripts remain independently deliverable.

Scheduler telemetry observation remains bounded, but retiring its observer joins
each already-accepted native SQL write through its original COMMIT acknowledgement.
Cancellation stops publication between statements; it does not manufacture an
unknown transaction outcome by interrupting a healthy COMMIT. Genuine native
unknown/committed receipts still fence readiness and remain available at shutdown.

A pure, originally scoped scheduler occurrence denial retires that occurrence,
not the shared host. The runner retains at most 32 original receipts without
reauthorizing an expired claim. Localization requires the exact scopes minted
by that original occurrence and its actual runner, not a shared fence class,
an inherited scope, or another admission on the same runner. Later observers
may retain the same already-proven receipt after its context has reset.
Runtime/unscoped denial, native close debt and
committed/unknown outcomes still latch host readiness failure. After-commit
callbacks preserve that committed distinction across explicit cancellation
causes and exception groups; ordinary reporting errors remain nonfatal.
Embedding-width diagnostics likewise propagate original database control rather
than treating authority/ACK loss as unavailable recall metadata.

Reanchor inspection propagates original canonical control carriers before
formatting operator diagnostics, including the initial runtime read, local
birth-anchor retargeting and bootstrap-custody inspection. An ordinary database
availability error still returns a readable `ReanchorResult`; authority denial,
native retirement debt and unknown/committed evidence do not become one.

Core delivery renewal acquires the exact delivery's PostgreSQL row lock or
SQLite writer reservation before sampling its clock. An expired token cannot
be revived by time spent waiting on that serialization point. A successful
renewal returns its row from the same transaction. Delivery disable stops new
polls and unstarted batch entries, wakes the idle interval, and physically joins
accepted delivery/status operations through their original COMMIT acknowledgement;
cancellation of the stop caller does not cancel those operations.
A resident that has not entered the queue loop may instead be cancelled while
waiting at the agent's separate READY publication barrier. No queue SQL or
delivery effect has then been accepted. The loop marks entry synchronously
before its first await; teardown checks that mark without yielding, so boot
rollback can retire the dormant resident without cancelling a native ACK.

Hosted default-VENV provisioning also consults retained immutable launch custody.
Removing an operator setting does not authorize upgrade, adoption or manifest
stamping of the same artifact. The no-follow restrictive namespace marker survives
process loss and denies default provisioning in that namespace when the specific
protected subtree can no longer be proved. It remains denial evidence, not launch
authority, and Core never clears it or mutates the operator artifact to bypass it.

Operator-notice settlement and stream closure retain the original turn failure
alongside any independently guarded audit/cleanup failure. Observability purge
keeps its per-table string diagnostics but also retains original exceptions in
the explicit cause chain; an uncertain DELETE is not optional telemetry noise.
Independent sweeps are attempted, while the original backend still refuses SQL
after native uncertainty. Reanchor embedding cleanup records its actual service
as unstarted retirement debt when closer allocation fails, preserving the body
transaction's committed/unknown disposition rather than implying rollback.
These native tests exercise the original storage operations; the embedding-span
proof does not replace signed-amendment authorization coverage.

Reentrant feature shutdown checks the live physical join graph before each
join, excluding itself and actual waiting callers without dropping their
original ownership roots. A snapshot taken before another owner enters shutdown
is not permission to join it later. Completion removes each original root only
when that Task actually finishes. This bookkeeping grants no execution authority.
Codex's abandoned per-thread
mutex acquisition releases synchronously, without requiring another Task to
admit Lock.release. Asynchronous abandoned-result cleanup uses canonical closer
admission, retaining its original returned resource if that admission fails.

Host-context bootstrap preserves original authority/commit/retirement controls
instead of returning them as degraded-state text. Ordinary capability failures
retain the separately documented degraded-context behavior. Cleanup registers
each actual resource before close-Task admission, joins accepted original close
operations, and attempts independent resources even when another close reports
native debt. Session factories retire before their borrowed databases. Failed
or unaccepted original owners remain retained across retries; another resource's
successful close cannot acknowledge their debt. The original caller owns this
drain without an extra all-resource collector Task. Local SQLite constructor/
closer fault tests cover these contracts, not live server READY publication or
hosted Workflow acceptance.

Admission-gate release is an in-memory Condition operation owned by the original
caller, not a newly admitted cleanup Task. Cancellation can retry only lock
acquisition; decrement and notification do not suspend, so native work is never
replayed and admission is released once. PostgreSQL mutation/COMMIT-ACK fault
tests cover task refusal and repeated caller withdrawal at this gate boundary;
they do not certify the complete isolated RPC or READY path.

Health shutdown likewise drains its existing loop, finite persistence and base
feature owners directly, preserving original native control alongside later
cleanup/cancellation. It does not require fresh observer Tasks before either
drain. Tests drive the actual HealthFeature public methods on an unpublished
native SQLite agent and hold the original COMMIT acknowledgement through caller
withdrawal. They prove physical write settlement and retained UNKNOWN, not live
agent readiness or provider behavior.

Health diagnostic normalizers preserve original execution control instead of
turning it into fail/warn dictionaries or generic HTTP errors. Fleet probes
retain ordinary per-agent diagnostics but use canonical original Task ownership
for control/cancellation and accepted sibling joins. Eight actual PostgreSQL
cases cover database, Scheduler, bootstrap, trigger, server fallback/outage and
fleet-reader paths after a real committed mutation loses its acknowledgement.
The fleet-reader proof uses an unpublished actual Agent, not a fabricated
manager roster or a claim of live fleet READY acceptance. Other component
normalizers and full live-host acceptance still require the whole review gate.

After an ordinary host opening failure, caller cancellation observed during
original-resource cleanup remains cancellation once that cleanup joins. The
canonical terminal selector's lifecycle opt-in preserves the opening failure
as explicit evidence; it never outranks native authority, UNKNOWN or retirement
debt. Standalone Hold startup invokes the shared original context drain directly
and propagates its actual cleanup debt alongside the startup failure. Actual
partial Agent initialization and SQLite schema/COMMIT/close faults cover these
contracts without bypassing mandatory features or publishing READY.

Agent-owned standalone Hold shutdown also enters that original context drain
directly. Caller withdrawal is reported only after the actual resource closers
join; failed native owners remain in the original context. Server phases retain
their separate cancellation-shielded Tasks, but canonical admission closes an
unaccepted operation and records the actual still-owned resources as unstarted
retirement debt. A refused phase cannot skip later independent drains or replace
an earlier native body receipt. These tests exercise actual partial Agent and
server teardown consumers with SQLite resources and PostgreSQL COMMIT-ACK
faults, not full server startup or READY. Scheduler startup/preflight allocation
sites remain separate findings, not covered by this shutdown repair.

Independent teardown is dependency-aware. Refused or failed Scheduler/Agent
phases leave their consumers' Hold context, Stop evidence and shared pools
owned and live, with explicit unstarted retirement debt. They do not prevent
independent drains such as the Phoenix proxy. Native tests exercise an actual
partial Agent and an actual prepared/polling Scheduler, original Hold reads,
SQLite workers and a loopback HTTPX proxy. They schedule no cognition and
establish neither cold-agent readiness nor complete server acceptance.
Phoenix retains the exact accepted proxy closer or its admission failure;
failed client closure cannot clear its original supervisor/client reference.
Process termination is still attempted independently. No Phoenix process is
started or adopted by the native proxy tests.

Host-Scheduler shutdown aggregates the original runner receipt with independent
storage retirement instead of replacing it in a finally block. Each original
application reference clears only after that owner's successful stop/close.
The PostgreSQL proof drives a real polling runner into native COMMIT-ACK loss,
then refuses its original backend closer; successful independent storage
closure is tested separately. Neither case claims a scheduled cognition or
server READY result.

Bounded Stop cleanup is not a physical-retirement acknowledgement. Cooperative
tail drains and distributed invocation closure raise an explicit pending-owner
timeout when their budget expires with original Tasks still unfinished. The
receipt retains the original registry and exact Tasks without cancelling them;
server teardown therefore keeps the registry and blocks dependent retirement.
Distributed closure still attempts independent owner abandonment, but does not
erase local generation bookkeeping while original settlement remains live.
Later joins preserve any actual native COMMIT/retirement failure rather than
replacing it with the earlier timeout. Cooperative drain budgets are finite
and nonnegative, and observer withdrawal is handled in the original drain
without allocating a replacement collector. Native proofs hold PostgreSQL
acknowledgements after real COMMIT through all three server registry-close
paths; they do not establish full startup, scheduled cognition or READY.

Both Stop drains use the same absolute-deadline observer for original Tasks.
Direct distributed-close withdrawal is aggregated with already-known native
control and pending-owner debt, rather than escaping as plain cancellation.
Independent owner abandonment uses the canonical owned fan-out: a refused
admission retains the registry, later epochs still get an attempt, and every
accepted child joins without sibling cancellation before the terminal result.
The native tests distinguish a pending COMMIT acknowledgement from an accepted
abandonment producer paused before its original database operation; neither
boundary manufactures a physical-retirement acknowledgement.

Health's database deadline checks original control evidence on the timeout
carrier before producing a diagnostic. Native PostgreSQL health-query tests
hold the original authority-transaction COMMIT acknowledgement through actual
deadline cancellation and prove that its COMMITTED receipt and physical pool
retirement survive both public feature and server-fallback consumers.

Finite health writes and gate transitions use canonical coroutine admission,
including closure when the original Task factory refuses admission. Health
retains its existing running-loop creation surface; an explicit foreign loop
is refused before work. Gate transitions preserve their coroutine-only input
contract and do not adopt a borrowed native Task. The gate proof covers the
actual lifecycle helper, not a complete SDK child/reload handshake.

Offline evidence fixtures also own their native resources: operator-notice
databases close in fixture teardown and assert their original worker exits;
legacy structural Health mocks close coroutines they deliberately suppress.
Those mocks are not native ACK/READY evidence. Standalone startup failure tests
use SQLite's actual schema authorizer rather than depending on which optional
feature registrations happen to be available in a larger suite.

LLM cleanup registers original clients and owned usage databases before closer
Task admission. Refusal retains the exact resource and error, closes the fresh
coroutine, and does not skip independent clients. Unpublished-provider refusal
keeps the original service in its host registry. Failed native owners cannot
be replaced by a speculative second close. Usage retirement joins the original
preference/modality writer Tasks even if their drain-wrapper admission failed;
the usage database reference clears only after physical close succeeds.
Native tests use actual SDK clients with loopback HTTPX sockets and actual
SQLite writers, not provider requests, fabricated ACKs or READY publication.

These are candidate contracts, not whole-branch review or release approval.
