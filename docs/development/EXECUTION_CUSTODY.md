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
evidence. Batch initializer recovery likewise owns its result collector through
repeated cancellation before retiring all unpublished agents. Inception,
embedding/recall compatibility fallbacks and modality accounting preserve native
control receipts before ordinary error/cancellation normalization; uncertainty
cannot authorize another INSERT, vector rewrite or legacy retrieval retry.

See the [canonical storage contract](../architecture/storage/STORAGE_ARCHITECTURE.md#hosted-execution-custody)
for installation and limitations. Full independent review, repository/CI gates,
immutable publication/verification and Frinz generation adoption/live acceptance
remain required; a source check is not evidence of downstream rollout.
