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
or an external immutable venv.

Core 0.53.5 introduces a distinct interrupted-reclaim marker payload. A host
must finish or retry any pending reclaim with Core 0.53.5 or newer before
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
Submissions are serialized per agent before entering that shared pool, so one
tenant can occupy at most one worker or queued slot. The pool remains a bounded
advisory resource: enough distinct tenants with wedged observers can saturate
it, at which point other tenants degrade to retrying telemetry rather than
losing lifecycle progress or growing an unbounded queue. A forced lifecycle
snapshot rejected by a saturated pool or failed during snapshot construction
is retried with bounded exponential backoff until delivery or terminal
lifecycle cancellation.
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
