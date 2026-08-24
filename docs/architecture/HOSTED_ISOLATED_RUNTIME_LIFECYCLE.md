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
The default timeout applies per isolated feature; the immutable override map can
set a different positive timeout or disable retirement for an ingress producer.
Standalone agents retain their existing always-resident behavior.

The public lifecycle seam rejects unscoped agents and calls made after isolated
feature construction. Unknown override names are logged after discovery. Core
also refuses idle retirement whenever the child owns a channel bridge, hosted
Telegram route attestation, or negotiated external-ingress lifecycle: an inbound
producer with no independent wake source must remain resident even if a host
accidentally assigns it a timeout.
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
reprovisions a missing Core-managed venv. `cleanup_eligible` therefore means the
exact child is stopped and the embedding host may reclaim that feature's owned
mutable workspace, Core-managed venv, and provisioning cache. It does not grant
permission to remove the agent namespace, sibling feature directories, or an
external immutable venv.

`IsolatedRuntimeTelemetrySnapshot` is immutable and deliberately excludes DIDs,
tenant identifiers, paths, commands, environment variables, configuration, and
credentials. It reports feature/distribution identity, lifecycle state and
generation, restart and activity timestamps, startup/provision timing, and
best-effort RSS/CPU/FD/process counts. OS metrics degrade to `None`; the process
PID is pinned to its observed creation time so PID reuse cannot expose another
process. Health restart count is separate from planned idle-wake count, and
cold child startup timing is separate from warm admission latency. Core measures
its owned venv, mutable workspace, and provisioning-cache bytes off the event
loop with root no-follow descriptor traversal plus entry/time budgets;
all components in one refresh share the same total time deadline. Externally
supplied immutable environments remain `None`. `cache_hit` means Core performed
no venv mutation, and venv size is cached across wakes unless Core actually
provisions, upgrades, or reinstalls it. An embedding host may merge its own
owner-scoped accounting after receiving the snapshot. A failed or uncertain
retirement reports `state="retirement-uncertain"`, remains active for capacity
accounting, and is never cleanup-eligible.

The observer is stored on the exact agent instance. There is no process-global
registry and no lookup API that can select another agent's telemetry. Observer
delivery is advisory: failures are logged, OS sampling runs in a worker, hot-path
emissions are rate-limited and scheduled through the agent background-task
registry after traffic admission is released, and an asynchronous observer is
bounded then detached and observed if it refuses to finish. Terminal lifecycle
cancels Core-owned telemetry tasks and does not create a post-cleanup delivery.
Telemetry cannot acquire ownership of child traffic or lifecycle progress.
The isolated child supervisor and idle monitor are registered as infrastructure
daemons, so their intentional residency does not itself block an
`idle_agents_only` restart.
