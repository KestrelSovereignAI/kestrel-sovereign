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

Core owns retirement. At the deadline it closes the existing traffic gate,
drains admitted work, rechecks the activity generation and monotonic deadline,
and stops the exact published client under the existing reload lock. A racing
call either invalidates retirement or waits for retirement and cold-starts one
new generation. Failed or cancellation-resistant stops seal the proxy and retain
the exact client for terminal cleanup; they are never reported as idle success.

`IsolatedRuntimeTelemetrySnapshot` is immutable and deliberately excludes DIDs,
tenant identifiers, paths, commands, environment variables, configuration, and
credentials. It reports feature/distribution identity, lifecycle state and
generation, restart and activity timestamps, startup/provision timing, and
best-effort RSS/CPU/FD/process counts. OS metrics degrade to `None`; the process
PID is pinned to its observed creation time so PID reuse cannot expose another
process. Environment/download/private-byte values remain `None` unless Core
itself knows them; an embedding host may merge its own owner-scoped workspace
accounting after receiving the snapshot.

The observer is stored on the exact agent instance. There is no process-global
registry and no lookup API that can select another agent's telemetry. Observer
delivery is advisory: failures are logged, and an asynchronous observer is
bounded and cancelled so it cannot acquire ownership of child lifecycle
progress.
