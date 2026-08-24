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

Core owns retirement. At the deadline it atomically closes the existing traffic
gate only when no work is admitted, rechecks the activity generation and
monotonic deadline, and stops the exact published client under the existing
reload lock. Activity is stamped at admission and completion. A racing call
either prevents retirement or waits for a complete retirement and cold-starts
one new generation. Failed or cancellation-resistant stops seal the proxy and
retain the exact client for terminal cleanup; they are never reported as idle
success. A transient cold-start failure instead restores the idle/retryable
state, and caller cancellation cannot interrupt a wake transaction midway.

`IsolatedRuntimeTelemetrySnapshot` is immutable and deliberately excludes DIDs,
tenant identifiers, paths, commands, environment variables, configuration, and
credentials. It reports feature/distribution identity, lifecycle state and
generation, restart and activity timestamps, startup/provision timing, and
best-effort RSS/CPU/FD/process counts. OS metrics degrade to `None`; the process
PID is pinned to its observed creation time so PID reuse cannot expose another
process. Health restart count is separate from planned idle-wake count, and
cold child startup timing is separate from warm admission latency. Core measures
its owned venv, mutable workspace, and provisioning-cache bytes off the event
loop; externally supplied immutable environments remain `None`. An embedding
host may merge its own owner-scoped accounting after receiving the snapshot.

The observer is stored on the exact agent instance. There is no process-global
registry and no lookup API that can select another agent's telemetry. Observer
delivery is advisory: failures are logged, OS sampling runs in a worker, hot-path
emissions are rate-limited, and an asynchronous observer is bounded then
detached and observed if it refuses to finish. It cannot acquire ownership of
child lifecycle progress.
