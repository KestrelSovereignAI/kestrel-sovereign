---
type: Developer Guide
---

# Public asynchronous-tail ownership

`kestrel_sovereign.async_ownership` exposes ten identity aliases of Core's
canonical ownership implementation. There is no second task owner, wrapper,
retry policy, or altered default:

| Symbol | Contract |
| --- | --- |
| `OwnedTaskAdmission` | Fresh caller-local receipt of the exact accepted native Task, written before callback publication; not a second registry or work grant. |
| `OwnedTaskOutcome` | Original result, failure, and caller-cancellation observation. |
| `await_owned_operation` | Admit and physically join one asynchronous operation at an existing unbounded join boundary, including failed task publication. |
| `await_owned_task` | Observe the same original task despite repeated caller cancellation; existing borrowing and forwarding options retain their semantics. |
| `create_owned_task` | Create one gated original Task, retaining the exact accepted Task on an explicit admission receipt if later publication fails; true allocation refusal closes the unstarted original coroutine. |
| `owned_task_error` | Harvest a terminal original Task's error without replacing its original cancellation or explicit evidence; an unfinished Task is not a terminal outcome. |
| `pending_owned_retirement_error` | Snapshot unfinished original Tasks as resource-bound retirement debt without cancelling, replacing, or acknowledging them. |
| `raise_owned_outcome` | Select the original outcome with canonical control/cancellation precedence; finite owners may request `retain_originals=True`. |
| `run_blocking_operation` | Finish one blocking operation, including its synchronous finalizer, before propagating caller cancellation. |
| `wait_owned_tasks_until` | Observe exact original Tasks until one finite absolute loop-clock deadline, retaining caller withdrawal and returning explicit done/pending sets without cancelling pending producers. |

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
handoff. `wait_owned_tasks_until` returns an observation, not a successful drain:
the caller still owns every returned pending Task and must retain its original
resource/debt and terminal evidence. Repeated caller cancellation does not renew
the supplied absolute deadline. `pending_owned_retirement_error` is a snapshot;
an empty later snapshot does not erase previously retained error evidence.

With an explicit fresh `OwnedTaskAdmission`, a postaccept publication failure
may raise while `admission.task` still names the accepted Task. The original
owner must join or retain that exact Task, not infer non-admission from the
exception or create a replacement operation. If allocation truly refused,
`admission.task` remains `None`; naming a `retirement_resource` retains unstarted
retirement debt rather than inventing a physical-close acknowledgement.
Creation retains the canonical copied-context and failed-start-gate behavior;
the public aliases add no admission policy, registry, retry, or new defaults.

The original admission receipt also retains a failed admission and its first
terminal observation. Task error attributes are diagnostic caches: refusal to
publish them is reported with the original error, but cannot strand an accepted
task behind a start gate. Cancellation before the first task step cannot erase
the original admission failure, even if that diagnostic cache is unavailable.
Multiple failed publications compose into that same receipt. Its terminal
snapshot also reconciles the physical task exception, so an earlier diagnostic
body-error cache cannot hide a later completion-publication failure. The original
creator's first completion callback captures even a cancelled physical Task,
before an unrelated native await can consume its original exception. All later
readers use that one terminal receipt and the creator's original error policy.
A caller that directly awaits the native Task must recover its completed receipt
with `owned_task_error` before classifying a cause-free cancellation as Stop.
Invocation uses this same canonical reader while retaining native await
cancellation delivery, rather than adding a second observation or source owner.
Iterator terminal inspection uses that same canonical task reader, including
completion failures raised after the source has already physically closed.
Cancellation observers retain the exact private Stop they actually received.
A callback publication or completion failure does not become private Stop by
sharing its exception class, including before the observer's first task step.
Observation-publication cancellations carry an explicit publication wrapper with
the exact original cause. A pending cancellation request alone cannot authenticate
that synchronous failure as a delivered Stop. A borrowed observer's publication
failure cannot interrupt the producer it does not own. Joins reconcile pre-first-
step Stop even before the scheduled done callback runs. Repeated owner joins
retain the same failures without repeating physical source close.

A custom task factory can make allocation ambiguous. If construction raises
and the original native task registry cannot identify exactly one accepted
Task, `admission.task is None` does not prove refusal. The original owner keeps
that UNKNOWN admission and its error as retirement debt; it cannot release the
resource budget or retry the work on that evidence. Independently identified
physical owners may still finish and close their own resources. Their later
completion does not erase the original UNKNOWN admission or make lifecycle
completion successful; the original close-completion API continues to report
the retained failure.

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
They also exercise real native Task allocation/publication, bounded observation,
original-error harvesting, and resource-bound pending debt. These CPU fixtures
do not prove database COMMIT, remote ownership, or descendant-process drain.
