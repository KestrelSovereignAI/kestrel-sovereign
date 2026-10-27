---
type: Architecture Spec
title: Retained host service discovery
description: Original host service registration discovery bound to agent lifecycle and publication, independent of resource authority.
resource: /docs/architecture/HOST_SERVICE_DISCOVERY.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-10-26T00:00:00Z'
status: implementation-candidate
owner: architecture
canonical: false
generated: false
privacy: public
---

# Retained host service discovery

`OperatorRuntimeRegistry` remains the sole service registration authority.
`async with agent.host_service_operation(requirement) as operation` requires an
explicit SDK `ServiceRequirement` with `scope=HOST` and yields a typed
`ServiceOperation(service, check_current)`. The synchronous checker is valid
only inside that original frame. It pins the exact registration set and its
lifecycle seal: removing and re-registering even the same service and
registration objects never revives an old operation.

Identity-capable discovery separately advertises
`host_service_identity_api_version = 1`. Its operation also carries an opaque
`discovery_identity`. `identity.matches(other)` compares the exact original
registration, issued set and seal, agent binding generation, and host
publication object/generation. It never invokes service equality. It is
metadata only, cannot be serialized, and cannot revive a closed checker.

`operation.require_issued()` additionally requires the exact record issued
inside that still-live discovery frame. Constructing or copying a
`ServiceOperation`, replacing its checker, or copying its identity/issuer
does not satisfy this provenance check. New background-producer admission
uses fresh genuine discovery at every operation and separately validates its
current content/run policies. Existing two-field API1 consumers keep their
original `check_current` behavior. This is a trusted Python extension boundary,
not a sandbox against arbitrary Python object mutation.

The server binds the genuinely started host registry through
`bind_host_service_resolver`. Each agent retains its own `operator_registry`.
The manager propagates the host binding to constructors, the shared publication
barrier and the final cold-agent registration seam. Operations check the exact
current manager route, agent identity, host publication gate/generation and
original agent lifetime. `clear_host_service_resolver(expected)` withdraws only
that exact binding, preserving a replacement. Host and agent shutdown revoke
discovery before resource retirement; a failed independent cleanup remains a
reported failure, not a reason to strand later resources.

This is discovery and a current-registration **restriction**, not a grant.
No `OperatorContext`, capability action or delegated principal is created.
A workflow producer must independently retain its original run policy and
canonical artifact policy, composing `operation.check_current` into their
mutation and final-delivery checks. A background run lease or Scheduler
execution cannot substitute for those policies. Service objects and references
must not be cached as persistent authority after the discovery frame exits.

Single-agent and multi-agent consoles both discover the authenticated host UI
manifest. An empty manifest is an ordinary no-contribution result.
