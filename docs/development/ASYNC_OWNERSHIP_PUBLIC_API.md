---
type: Developer Guide
---

# Public asynchronous-tail ownership

`kestrel_sovereign.async_ownership` exposes five identity aliases of Core's
canonical ownership implementation. There is no second task owner, wrapper,
retry policy, or altered default:

| Symbol | Contract |
| --- | --- |
| `OwnedTaskOutcome` | Original result, failure, and caller-cancellation observation. |
| `await_owned_operation` | Admit and physically join one asynchronous operation at an existing unbounded join boundary, including failed task publication. |
| `await_owned_task` | Observe the same original task despite repeated caller cancellation; existing borrowing and forwarding options retain their semantics. |
| `raise_owned_outcome` | Select the original outcome with canonical control/cancellation precedence; finite owners may request `retain_originals=True`. |
| `run_blocking_operation` | Finish one blocking operation, including its synchronous finalizer, before propagating caller cancellation. |

## Cleanup is not WORK permission

Owning a tail does not admit new SQL, storage, model, provider, or business work.
Callers must use the original execution-admission boundary for new work. These
aliases do not renew expired authority or turn an acknowledged cleanup into a
reusable work grant. Work-only APIs such as
`execution_custody.run_execution_blocking_operation` check live authority and
are not substitutes for retiring an original physical owner after revocation.

`await_owned_operation` is for an existing asynchronous join boundary. It does
not create a replacement registry, observation deadline, or cleanup policy.
Bounded lifecycle owners retain their original deadline and explicit ownership
handoff. Task creation/admission internals are not part of this public facade.

## Task completion is not process-tree drain

A task or thread join proves only completion of that original operation. It
does not prove that subprocesses, nested process groups, pipes, sessions, or
workspaces have retired. Their original physical custodian must separately
observe and join them, retaining unresolved custody and original failures.
No helper here authorizes killing an unverified PID or infers physical absence
from a done task, missing record, empty debt list, or successful SQL ACK.

The underlying canonical ownership tests remain applicable without copies.
Public API tests additionally bind object identity/signatures and exercise
actual blocked-thread body/finalizer retirement through the public imports.
