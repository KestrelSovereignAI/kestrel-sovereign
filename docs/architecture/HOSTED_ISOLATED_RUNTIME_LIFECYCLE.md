---
type: Architecture Spec
title: Hosted Isolated Runtime Lifecycle
description: Host-scoped idle retirement and sanitized telemetry for isolated feature children.
resource: /docs/architecture/HOSTED_ISOLATED_RUNTIME_LIFECYCLE.md
tags:
- docs
- architecture
- isolated-runtime
timestamp: '2026-08-22T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Hosted isolated runtime lifecycle

Hosted applications can apply idle-retirement policy and receive sanitized
process telemetry without reaching into `ProxyFeature` internals or managing a
second subprocess lifecycle.

Pass policy when constructing the exact `KestrelAgent`:

```python
agent = KestrelAgent(
    did=agent_did,
    isolated_runtime_root=runtime_root,
    isolated_runtime_namespace=runtime_namespace,
    isolated_runtime_idle_timeout_seconds=900,
    isolated_runtime_idle_timeouts={
        "TelegramFeature": None,  # polling producers must remain resident
    },
    isolated_runtime_telemetry_observer=record_snapshot,
)
```

Hosts that load agents through `AgentManager` provide the same per-agent policy
through `hosted_isolated_runtime_lifecycle_policy_factory`; the factory returns
a `HostedIsolatedRuntimeLifecyclePolicy` selected from the manager-owned agent
name, DID, and local configuration before that agent is constructed.

Lifecycle policy is accepted only with an explicit hosted root and namespace.
The default timeout applies per isolated feature only when its service calls
`IsolatedFeatureService.advertise_inbound_producer(False)` before a successful
initialize handshake. SDK 0.37 freezes that negotiated declaration for the
child generation; changing it requires a restarted child. Missing or malformed
producer metadata fails resident before the first event, because a quiet polling
child cannot prove it has an independent wake source. The immutable override
map is an explicit per-feature operator decision that can opt an ambiguous
utility feature into a different positive timeout or disable retirement for an
ingress producer.
Standalone agents retain their existing always-resident behavior.

The public lifecycle seam rejects unscoped agents and calls made after isolated
feature construction. Unknown override names are logged after discovery. Core
also refuses idle retirement whenever the child owns a channel bridge, hosted
Telegram route attestation, or negotiated external-ingress lifecycle: an inbound
producer with no independent wake source must remain resident even if a host
accidentally assigns it a timeout.
Core also latches a feature resident after it observes any canonical inbound
event name from the published child, protecting legacy producers whose
capability metadata omitted their inbound role.
The idle monitor is not started for a child that already advertises an inbound
producer, and it stops if a later reload adds one. Other non-retirement outcomes
use a bounded retry interval rather than a zero-yield loop.

Core owns retirement. At the deadline it atomically closes the existing traffic
gate only when no work is admitted, rechecks the activity generation and
monotonic deadline, and stops the exact published client under the existing
reload lock. Activity is stamped at admission and completion. A racing call
either prevents retirement or waits for a complete retirement and cold-starts
one new generation. Failed or cancellation-resistant stops seal the proxy and
retain the exact client for terminal cleanup; they are never reported as idle
success. A transient cold-start failure instead restores the idle/retryable
state, and caller cancellation cannot interrupt a wake transaction midway.

Initialization and terminal cleanup retain their original Task admission receipt,
including allocation refusal or callback-publication failure. Completed is not
the same as retired: an unresolved native outcome keeps its exact original
operation fenced, even for a shutdown already queued on the retirement lock.
Ordinary facade failures retain bounded retry after the original operation
settles; successful retirement and a settled acknowledgement alone do not create
permanent cleanup debt. Each finite observer keeps its own original withdrawal
without adding that cancellation to another observer's shared source history.
Uncertainty from an independent ingress operation still denies new work, but
does not pretend that a child's as-yet-unattempted physical stop already failed.
An earlier bounded cleanup that waited behind that ingress may finish its first
stop once the exact ingress owner settles; it cannot retry a failed native stop.

Wake preparation failures retain the existing secret-free tool error envelope;
an ancillary telemetry failure after publication leaves the live child running
and supervised rather than misclassifying it as idle.
Before a wake, Core recreates the private workspace, resolves runtime paths, and
reprovisions a missing Core-managed venv. It also invalidates the prior child's
memoized config and reads the current durable generation, so a change committed
by another replica while this process was idle reaches the new child.
Configuration changes also wake an idle-retired child before staging so its
negotiated transition validation and cleanup hooks cannot be bypassed. UI
contributions parsed from the last successful initialize handshake remain
available to the live console manifest while only the child process is idle.
If the active config prevents the child from waking, Core uses the existing
clientless staged repair path; if terminal lifecycle lands during that wake, it
uses terminal config repair. Reclaiming the environment or latching terminal
lifecycle clears the cached UI contribution before its assets can be advertised.
`cleanup_eligible` therefore means the exact child is idle-retired, Core is not
in terminal shutdown, and the embedding host may ask Core to reclaim that
feature's owned mutable workspace,
Core-managed venv, and provisioning cache through
`ProxyFeature.reclaim_idle_workspace()`. It is not permission for the host to
delete a path itself: Core's seam rechecks eligibility and performs secure
feature-scoped deletion under the same reload lock that serializes a racing
cold wake. Caller cancellation is reported only after the deletion transaction
has settled, so it cannot release that lock while a worker still mutates the
tree. Before the first destructive mutation Core durably marks the managed
environment for forced repair and preserves that marker until deletion fully
succeeds, so a partial reclaim cannot be adopted as a healthy venv on the next
wake. The seam never removes the agent namespace, sibling feature directories,
or an external immutable venv. An immutable operator venv located inside the
feature's mutable workspace makes cleanup permanently ineligible: Core refuses
reclamation with a configuration-specific error naming the selecting setting,
because deleting the workspace would otherwise delete an artifact that Core
does not own. This remains true when `BIN` is also configured; Core bases the
decision on the separately validated `VENV` artifact rather than the launch
path selected by `BIN`. Operators must relocate that immutable venv outside the
agent-scoped workspace before Core can reclaim the workspace.
If a cold wake cannot revalidate the configured immutable venv, Core retains
the last validated artifact and marks custody unproven. Cleanup remains
ineligible, and reclamation returns a distinct setting-only custody diagnostic,
until a later successful runtime-path resolution republishes the complete
launch and custody state.

Core 0.53.8 introduces a distinct interrupted-reclaim marker payload. A host
must finish or retry any pending reclaim with Core 0.53.8 or newer before
downgrading: older Core releases do not recognize that payload and will retain
the affected feature directory rather than start from potentially partial
state. This fail-closed version-skew behavior does not affect directories whose
reclaim completed and removed the marker.

`IsolatedRuntimeTelemetrySnapshot` is immutable and deliberately excludes DIDs,
tenant identifiers, paths, commands, environment variables, configuration, and
credentials. It reports feature/distribution identity, lifecycle state and
generation, restart and activity timestamps, startup/provision timing, and
best-effort RSS/CPU/FD/process counts. OS metrics degrade to `None`; the process
PID is pinned to its observed creation time so PID reuse cannot expose another
process. Health restart count is separate from planned idle-wake count, and
cold child startup timing is separate from warm admission latency. A synchronous
`runtime_telemetry_snapshot()` returns lifecycle state plus the most recent
off-loop process sample; callers that require a fresh process sample await
`sample_runtime_telemetry()`. Pull-only hosts can also request the bounded disk
sample explicitly with `sample_runtime_telemetry(refresh_disk=True)`; without
an observer or that opt-in, disk fields remain unsampled. Core measures its
owned venv, mutable workspace,
and provisioning-cache bytes off the event loop with root no-follow descriptor
traversal plus entry/time budgets;
all components in one refresh share the same total time deadline. The
`disk_telemetry_status` field distinguishes a complete sample, a budget-exceeded
sample, and an unavailable safe traversal; the first budget exhaustion is also
logged once per feature. A refresh failure reports `unavailable` and clears its
byte counters instead of presenting a stale complete sample. Externally
supplied immutable environments remain
`None` without making an otherwise complete sample ambiguous. `cache_hit` means
Core performed no venv mutation, and venv size is cached across wakes unless
Core actually provisions, upgrades, or reinstalls it. Environment and download
fields are categories rather than additive physical-storage totals: hard-linked
files are charged only to the first category traversed, while clone-based
filesystems do not expose shared extents through the safe metadata seam. An
embedding host may merge its own owner-scoped accounting after receiving the
snapshot. A failed or
uncertain retirement reports `state="retirement-uncertain"`, remains active for
capacity accounting, and is never cleanup-eligible. Telemetry and the reclaim
seam use the same live-evidence predicate. A successful non-terminal recovery
does not retain a stale uncertainty marker after its exact facade work settles.

The observer is stored on the exact agent instance. There is no process-global
registry and no lookup API that can select another agent's telemetry. Observer
delivery is advisory: failures are logged, OS sampling runs in the default
worker pool, and synchronous host observers run in a separate bounded daemon
executor so they cannot starve venv or lifecycle work or hold process exit.
Submissions are serialized per agent before entering that shared pool, and a
per-agent submission token remains held until the underlying worker Future
settles even if terminal lifecycle cancellation abandons its asyncio delivery
task. Cancelled queued work is physically discarded before retry, so one tenant
can occupy at most one worker or queued slot. The pool
remains a bounded advisory resource: enough distinct tenants with wedged
observers can saturate it, at which point other tenants degrade to retrying
telemetry rather than losing lifecycle progress or growing an unbounded queue.
A forced lifecycle snapshot rejected by Core's saturated observer pool is
retried with capped exponential backoff until delivery or terminal lifecycle
cancellation, because an idle or cleanup transition may have no later trigger.
Failures in host callback code or snapshot construction use a bounded retry
streak and then return to ordinary rate-limited delivery, preventing a broken
observer from creating a permanent process-sampling and warning loop.
Hot-path
emissions are rate-limited and scheduled through the agent background-task
registry after traffic admission is released, and an asynchronous observer is
bounded then detached and observed if it refuses to finish. A state transition
that arrives behind one slow observer is coalesced and delivered when that
observer settles, so an idle cleanup-eligibility transition is not silently
dropped. Observer deferral and emit-task deferral have separate latches: a slow
observer cannot create a reschedule loop, and ordinary coalesced traffic retains
the hot-path rate limit. Disk refresh and observer delivery are never awaited
while Core owns a reload lock. Terminal lifecycle
cancels Core-owned telemetry tasks and does not create a post-cleanup delivery.
Workspace reclaim advances a generation fence so a previously scheduled disk
sample cannot overwrite the post-reclaim accounting after deletion, and a disk
sampling failure is logged without suppressing the forced lifecycle snapshot.
Telemetry cannot acquire ownership of child traffic or lifecycle progress.
The isolated child supervisor, idle monitor, and telemetry delivery tasks are
registered as infrastructure daemons, so their intentional residency does not itself block an
`idle_agents_only` restart.
While retired, the supervisor waits on the child-publication event rather than
polling once per second; a cold wake or terminal shutdown releases that wait.
Immutable hosted BIN overrides have the same operator custody as immutable
VENV overrides. A validated executable contained in the runtime workspace
prevents both cleanup eligibility and deletion. Failed revalidation retains
the prior executable identity and makes custody unproven; it never grants
Core permission to reclaim the workspace after a failed wake.

Immutable runtime custody starts at construction, before asynchronous child
initialization. Protected VENV/BIN paths accumulate across override removal or
retargeting rather than replacing the previous operator-owned artifacts. A
failed immutable validation leaves custody uncertainty sticky; validating the
current setting does not prove ownership of the historical path or authorize
its deletion. Reclamation remains fail-closed until explicit owner reconciliation.

Finite cleanup joins preserve independent original failures and the current
caller's withdrawal without rewriting a shared native source receipt. This
also applies to configuration-fence finalizers, startup registration rollback,
single-agent onboarding rollback, and SQLite worker retirement. Ordinary
configuration body-error precedence does not suppress native finalizer control.
An accepted SDK response/context close remains the same close task even when
publishing its completion callback fails.

Every explicit native `create_owned_task` admission supplies a fresh
`OwnedTaskAdmission`. Its start gate keeps the operation unentered until
canonical callback publication succeeds, including under eager execution.
Synchronous refund/drain handoffs publish `admission.task` into their existing
record even when creation raises; their asynchronous consumers join that exact
Task. Existing unbounded asynchronous join boundaries can use
`await_owned_operation` for the same receipt-and-join protocol. Bounded tail
owners keep their original absolute deadline and explicit handoff instead.
An accepted but unstarted closer retains `(resource, original_task)` debt;
only genuine no-Task refusal uses `(resource, None)`. Debt belongs on a private
carrier, not the original native exception. Independent cleanup phases still
get their attempt before the finite terminal failure is raised. Shared source
latches never acquire another shutdown observer's actual cancellation.
The physical result of an original closer is distinct from its completion
observer's callback error: successful physical retirement closes that record,
while the observer keeps its own failure finitely. A later healthy observation
must not invent a replacement close or replay observer debt. Failed native
attempts reuse their existing private error carrier instead of growing a new
exception ancestry on every Stop. Cached publication diagnostics and subsequent
native failures are composed in isolation from their original source graphs.
Core-owned completion-observation groups carry canonical source provenance;
genuine foreign source groups remain whole. Host hook rollback returns its
current finite failures separately from retained context control. All cognition
debt writers use the same bounded source-only retention boundary, including
late route and cleanup completion callbacks. PostgreSQL and Phoenix close
accounting uses the original Task's physical result, not its joiner's aggregate.

Manager, server and dispatcher finite lifecycle owners use an explicit native
composition policy: every original source and actual current join withdrawal
remains in the finite result without decorating any original source graph.
This does not change the generic non-native caused-cancellation defaults.
The manager's two completed-cleanup writers have a different lifetime: their
shared summary keeps one strongest genuine original (UNKNOWN before COMMITTED
before other denial), fixed irreversible classification, and exact unresolved
retirement ownership. It does not keep every completed source after the bounded
128-record display history is evicted or acknowledged. Compacted independent
scope evidence fails closed at this manager's aggregate reader rather than
attributing all sources to the selected original's tenant. Exact per-operation
scopes remain unchanged. Display acknowledgement is not physical retirement:
generic debts stay conservative, and unretired typed ACK receipts keep their
mutable original retirement record, including an owner not yet published.
Only that receipt's actual retirement permits its redundant ownership record
to be compacted; a successful unrelated Task or Future is not a retirement ACK.

Coordinator and lock-acquisition Tasks need admission receipts too: their body
must not enter before completion-callback publication succeeds. A failed
publication joins the actual accepted Task, rather than describing acceptance
as no-Task refusal. Lock observation is separate from acquisition; a lock
physically acquired before an observer failure is released unless ownership
has explicitly transferred. An unentered server phase keeps private debt for
its actual resources and accepted Task (or genuine None), without modifying
the original failure. An existing Agent continuation remains owned and joined
even when admission of its new observing waiter fails.

Finite multi-owner boundaries keep originals and actual withdrawals unreduced
until later owners finish. An ordinary-first failure cannot erase cancellation
before a later native result arrives. Nested writer drains and shutdown waiters
can pass caller-local finite collectors; these lists never become shared control
history. Fleet topology and post-removal reconciliation failures preserve
conservative relationship state while independent cleanup still receives its
attempt. Forwarding an already compacted manager summary preserves its unscoped
classification instead of attributing discarded sources to its selected tenant.

Session-training acquisition and rejected-session release retain the actual
physical hook task and session when native retirement is unresolved. A later
close reports that evidence without issuing a replacement release. Ordinary
provider release failures retain their existing bounded retry behavior. Local
component tests do not prove authenticated provider termination or billing
acceptance; those remain separate provider verification boundaries.

Submission, compensation and release publish their original Task on the job
before registering fallible completion observers. A refused/unpublished owner
does not advance the physical release phase underneath an accepted task. Native
submission or compensation control is not a business failure string and cannot
skip the first independent session release; the finite result retains both
outcomes. Public training adapters preserve that control. The SQLite backend
likewise roots the exact connection and accepted closer (or an explicit absent
closer) before observer publication. A live worker with no admitted closer is
unresolved retirement, not a completed close; later probes cannot retry it or
discard its original control while that worker remains live.

Private host ingress has a caller-local Core admission rooted before task
publication. Backend observation follows actual joined task lineage, not mere
ContextVar inheritance into resident readers or supervisors. Native failure
observation follows nested TaskGroups and their actual direct/gather waiters,
including a group child running before its parent's exit wait begins. This
membership is a physical join obligation, not copied-context authority. Native
classification comes from the actual backend phase: a foreign exception's
attributes, type, cause, or group membership do not prove that this ingress
attempted COMMIT. Healthy rolled-back foreign failures release their observation.

An unresolved native ingress retains its original resource/phase owners,
admitted scope snapshot, selected client and actual caller withdrawal privately.
Public errors contain detached classifications only, including later lifecycle
and configuration rejections. Terminal shutdown still performs the first
physical child stop; an unrelated ingress failure cannot substitute for that
retirement. Every physical stop producer uses the same per-client disposition
handoff, including when settlement is harvested before its callback runs.

Reserved quiesce/resume RPCs use their existing lifecycle owner outside data
admission, with the same native observation. An unresolved outcome is monotonic:
a later failed reset cannot weaken a previously observed UNKNOWN. A fully
retired known ACK plus withdrawal, or a denial confined to a captured caller
scope, is delivered finitely without poisoning a distinct live caller's runtime.
Neither case clears an unresolved commit or physical-retirement debt.

Finite shutdown outcomes are composed only after the enclosing ownership
boundary finishes collecting its originals. Successful native work can still
have a failed completion observer: that error belongs to the finite drain, not
the physical worker's shared failure latch. Reaper, offboarding, refund,
reconciliation and unsealing outcomes must all survive independently, including
ordinary-first failures followed by native control and actual caller withdrawal.
Cold-identity admission uses the same physically balanced manager-lock helper
as the drain and handoff seal. Spawn expiry and admitted-spawn joins retain
native unwind failures without weakening signed deadlines or authorizing retry.
Admission hooks use an acceptance receipt before completion publication.
Bookkeeping-observer or expiry-fence failures still join the original hook;
they cannot return while it retains native work. An unexpected observation in
a finite group must not filter away an independent original native source.

Standalone CLI and main callers own their Hold context separately from the
Agent. They complete both safe retirements before reporting a settled Agent
failure. A genuinely pending original Agent Task instead retains that exact
context/Task dependency; it is neither a successful context close nor permission
to allocate a replacement cleanup attempt. Component native tests establish
these ownership boundaries, not whole-Agent genesis, provider billing or deployed
production acceptance.
