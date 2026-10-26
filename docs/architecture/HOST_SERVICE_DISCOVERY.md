# Retained host service discovery

`OperatorRuntimeRegistry` remains the sole service registration authority.
`async with agent.host_service_operation(requirement) as operation` requires an
explicit SDK `ServiceRequirement` with `scope=HOST` and yields a typed
`ServiceOperation(service, check_current)`. The synchronous checker is valid
only inside that original frame. It pins the exact registration set and its
lifecycle seal: removing and re-registering even the same service and
registration objects never revives an old operation.

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
