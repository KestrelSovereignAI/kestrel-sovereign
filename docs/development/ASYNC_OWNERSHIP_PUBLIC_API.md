---
type: Developer Guide
---

# Public asynchronous-tail ownership

`kestrel_sovereign.async_ownership` exposes identity aliases of Core's
canonical ownership implementation. There is no second task owner, wrapper,
retry policy, or altered default:

| Symbol | Contract |
| --- | --- |
| `OwnedOperation` | Retain one original operation and its admission across repeated or finite borrowed observations; an explicit fresh receipt can be reserved before construction. |
| `OwnedTaskAdmission` | Fresh caller-local receipt of the exact accepted native Task, written before callback publication; not a second registry or work grant. |
| `OwnedTaskLender` | Synchronous host-owned resource policy. Reserve against the original admission, lend to the exact gated Task, retire only from canonical completion. |
| `OwnedTaskLenderCapture` | One live binding occurrence and its actual admitted parent, captured separately from the body's execution context. |
| `OwnedTaskOutcome` | Original result, failure, and caller-cancellation observation. |
| `OwnedTaskRetirement` | Read-only snapshot of one exact admission: unknown, proven task nonacceptance, pending, or complete with its original error. |
| `await_owned_operation` | Admit and physically join one asynchronous operation at an existing unbounded join boundary, including failed task publication. |
| `await_owned_task` | Observe the same original task despite repeated caller cancellation; existing borrowing and forwarding options retain their semantics. |
| `bind_owned_task_lender` | Route the embedding host's own policy from its actual resource-owning Task. This binding never creates a resource or grants authority on its own. |
| `capture_owned_task_lender` | Retain the current exact parent for a later, explicitly delegated native admission. A copied context does not carry the task identity. |
| `create_owned_task` | Create one gated original Task, retaining the exact accepted Task on an explicit admission receipt if later publication fails; true allocation refusal closes the unstarted original coroutine. |
| `owned_task_error` | Harvest a terminal original Task's error without replacing its original cancellation or explicit evidence; an unfinished Task is not a terminal outcome. |
| `pending_owned_retirement_error` | Snapshot unfinished original Tasks as resource-bound retirement debt without cancelling, replacing, or acknowledging them. |
| `raise_owned_outcome` | Select the original outcome with canonical control/cancellation precedence; finite owners may request `retain_originals=True`. |
| `reconcile_owned_task_admission` | Observe the canonical receipt without allocating, cancelling, retrying, closing, or releasing any resource or budget. |
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
creator synchronously captures an already-finished Task before exposing it;
pending Tasks retain a completion callback. The canonical wrapper records its
own failure on that same receipt before publishing diagnostic attributes. An
earlier foreign observer of an outer factory Task can still consume native
cancellation evidence: callback registration alone cannot certify observation.
If that original physical outcome cannot be recovered, the receipt retains
explicit uncertainty and retirement debt naming the exact original Task. All
later readers use that one terminal receipt and the creator's error policy.
Implicit exception context from an observer's active error handler is not
physical completion evidence and cannot certify that capture succeeded.
Later admission publication failures extend that same combined receipt even
after physical completion was captured. They retain the original physical
failure if a diagnostic cache was withdrawn, and never retrieve the consumed
native cancellation a second time.
A caller that directly awaits the native Task must recover its completed receipt
with `owned_task_error` before classifying a cause-free cancellation as Stop.
Failed request admission joins a positively identified canonical wrapper whose
start gate already holds the refusal; it does not inject another cancellation
into that sealed gate. An opaque outer factory has no such proof. Durable
settlement uses the same original completion reader and shared terminal
classifier: SDK or audit uncertainty abandons its registration. Failed
settlement publication omits only its authenticated private cancellation and
retains every independent physical completion failure.
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

`OwnedOperation(..., admission=receipt)` consumes the same fresh receipt that
its owner reserved before construction, including when a custom factory cannot
identify an accepted Task. Consumed receipts cannot be reused for a new attempt.
Failed opening does not turn `observe(defer_cancellation=False)` into an unbounded
join: the original operation retains its finalizer while the borrowed caller can
withdraw. The default observation remains an unbounded join.
After failed opening, neither observation mode forwards a fresh cancellation
into the original owner's already-running finalizer. Factory-publication errors
and unstarted-resource debt remain on the same canonical admission receipt, not
only on the operation's observer.

Both borrow and must-join observation use the retained original receipt through
`await_owned_task(..., admission=receipt)`, even when the Task refuses diagnostic
metadata. The explicit receipt must name an exactly admitted native Task.
Actual result publication is still a separate observable step. A subclass's
refusal is delivered in the explicit admission's observation outcome so the
existing owner can close any unpublished handle before propagation. A substituted
value cannot replace the Task's actual result or certify resource delivery. Neither
publication failure rewrites the producer's original physical receipt.

`reconcile_owned_task_admission` reports UNKNOWN for a fresh unused receipt,
ambiguous acceptance, lost completion capture, or an original unretired physical
resource. `task is None` alone is never proven nonacceptance. That state requires
an observed refusal and actual close of the unstarted payload. Nonacceptance
proves only that this Task attempt was not accepted; it cannot acknowledge close
of a pre-existing external resource. Complete means the exact original Task's
completion was captured, not that its business operation succeeded. Original
errors remain attached to the observation. These snapshots do not lend execution
authority, retire a database lease, decrement accounting, or implement the
cross-task resource-loan policy; physical custodians still own those obligations.

## Explicit host-resource loans

The embedding resource owner provides the one synchronous `OwnedTaskLender`
policy and validates the actual creator, original admitted parent, resource
occurrence and capacity. Core reserves against the exact admission before
native allocation and calls `lend` before opening the payload gate.
The original parent must be live at reservation and lending. Once accepted,
the child's own receipt owns its loan: creator completion alone cannot revoke
it before the first payload step. Binding exit and withdrawal of the original
work authority still deny that step. A completed parent's old capture cannot
reserve another child. A copied or fresh context does not supply a captured
parent. Pass the exact capture to
explicitly delegate across a fresh execution context; pass `None` to suppress
lending. Native privacy factories and callback scopes retain the capture at
creation, independently of privacy scope. A detached boundary suppresses it.
Core's cleanup-only custody suppresses WORK lending even for previously
captured factories, including shared stream withdrawal and restored turn
custody. Detachment also suppresses old explicit captures; an independent
resource owner may bind its own new policy, but cleanup-only work may not.
The detachment occurrence is retained separately from that active policy and
carried through the selected body context, so binding the new owner cannot
revive a capture from before detachment. Context restrictions are read without
entering the supplied Context, including a Context already active in its caller.
The original admitted owner still retains its physical
cleanup and reports its exact outcome; no new work grant is made there.

For complete native retirement or proven nonacceptance, Core publishes one
canonical observation to the host. UNKNOWN retains the admission and its
original errors for the host. Callbacks are notification, not the resource's
owner: a host must retain and read-only reconcile its own receipts if publication
fails. Core's original one-shot publisher is the sole caller of policy `retire`.
A host drain joins original Tasks and waits for its retirement-change signal;
it must not call `retire` independently, even after observing physical COMPLETE.
This prevents a drain racing the already-queued original completion callback.
No callback runs a replacement task or acknowledges an external
resource. Opaque factory wrappers never receive a loan. Exhausted or revoked
policies must refuse synchronously, without allocating another owner or
reattributing an existing resource. A reservation refusal is atomic.

Captures retain Core's canonical read-only execution-work check, including the
same mutable stream-withdrawal records, not a new context-derived work grant.
Admission rechecks that evidence before reservation/allocation, before and after
lending, and at the actual first payload step. A retained callback in a fresh
context cannot borrow from its still-unwinding withdrawn producer. Exact native
PostgreSQL COMMIT/ROLLBACK, pool reset/close and advisory-session close suppress
WORK lending through the same cleanup restriction; physical retirement cannot
depend on spare WORK capacity. This does not relax standalone execution checks.

Only the original native CPython selector/proactor or installed uvloop default
constructor can receive a loan, with no configured task factory. Core checks
implementation identity before reservation and again before allocation; it
never silently bypasses or replaces an unsupported factory. This restriction
applies only to live, unsuppressed host loans. Generic non-lender creation keeps
its existing factory semantics and ambiguous-acceptance reconciliation. A
one-turn Task snapshot or first-entry identity check cannot prove the lifetime
of a coroutine retained by an arbitrary constructor. The exact accepted Task
and first-entry checks remain additional safeguards, not that missing proof.
Gate cancellation/refusal is settled inside the original admission transaction
before lending or dispatch; the physical cancellation remains on that receipt.

Direct owned iterators capture their original lender separately from the source
body context. `ExecutionStreamLifecycle.owner_factory` receives lazy body,
context, and keyword-only `admission`/`lender_capture`; it must forward those
exact originals to canonical construction and return the admitted producer.
There is no legacy factory fallback or replacement receipt. Captured privacy
factories accept the same explicit handoff. A forwarder checks producer capacity
before allocating its watcher/closer, while source dispatch remains gated until
all construction/registration finishes. Cooperative Event-only idle waits and
physical closers do not consume extra WORK loans.

The original source admission also retains physical-retirement facts separately
from exception presentation. An observed failed close keeps reconciliation
UNKNOWN and the host loan outstanding even when its producer Task is done;
an ordinary body failure followed by successful physical close can retire.
The same record retains producer, watcher, and original closer receipts. Native
completion and actual close progress re-evaluate the original one-shot retirement
publisher, including when completion notification preceded source settlement.
No observer timeout, lost callback, exception spelling, or replacement owner
can manufacture an acknowledgment. A failed constructor uses a fresh pending
source carrier rather than decorating its original refusal. The existing join
settles that construction fact only after the exact accepted owners settled;
independent capture, epilogue, watcher, and physical-close failures remain.
Missing original completion-capture registration is UNKNOWN even if an
unstarted wrapper's Task has since been joined; a later join does not replace
that failed registration. Constructor debt names the uncertain original
producer or watcher receipt, not a different Task that already retired.

Physical HTTP-body close (including inline response close), database-facade
disposal, and native SQLite connection close use the same cleanup-only lending
restriction as PostgreSQL epilogues. They must retire their original resources
even when no new WORK capacity is available, without granting new WORK authority
or suppressing an actual failed close.

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

For a native Task or Future admitted elsewhere, its existing retained terminal
cancellation remains stable across canonical joins. There is no Core admission
receipt to reconcile; rereading asyncio's consumed cancellation cannot supply a
new physical failure. This does not grant Core ownership of that foreign task.

Private Stop is authenticated by the cancellation token on the exact original
admission receipt. Core lifecycle cancellation uses `cancel_owned_task`; neither a
positive cancellation count nor a close flag authenticates a distinct source or
watcher error. The same check applies before source entry, during unwinding, and
at terminal close. Genuine teardown cancellation still propagates, with every
original carrier retained. Multiple pure cancellations use the canonical finite
cancellation envelope; their original graph remains intact. Invocation and Kite
recover native-await evidence through `owned_task_completion_error` before Stop
classification. Recovery also retains actual caller withdrawal received while
restoring the privacy writer; the native receipt cannot replace that finite
observation. A composed cancellation or an explicitly caused cancellation is
never an individual private Stop acknowledgement. Physical close authenticates
and omits only the producer's individual acknowledgement before composing the
remaining source and watcher evidence.

HTTP, Bridge and cognition dispatch use the shared tool terminal classifier
before acknowledging typed Stop or owner loss. SDK effect uncertainty and
unconfirmed audit evidence remain terminal even when carried by cancellation.
The existing structured tool-error response retains its original explicit cause.
Formatting and consuming that response cannot complete an uncertain request:
the same terminal evidence remains available through physical stream cleanup,
request disposition and the speech-drain handoff. Direct decorated invocations
and generators use the same terminal classifier for their own registrations.
Retirement classifies all actual outcomes after cleanup, including uncertainty
first raised during producer unwind, physical close or speech-tap completion.
An earlier cancellation snapshot cannot certify those later stages.

If publication of the original completion callback is refused, a foreign native
await may consume an independent cancelled completion before Core can capture it.
The exact lost exception cannot be reconstructed. Its admission retains explicit
unobservable-completion uncertainty and debt naming that same Task, rather than
certifying ordinary failure or successful retirement. No callback retry, new
physical operation, replacement receipt or source-only graph guess is used.

A native Task whose actual coroutine is exactly the canonical wrapper, which
never entered that wrapper, cannot have run an outer completion epilogue. This
positive identity/entry proof retains its known gate failure without inventing
completion uncertainty. An outer custom-factory coroutine does not satisfy it.

Native result consumers enclose their existing await and privacy writer restoration in `owned_native_completion`. Success and ordinary exceptions must consult the same authoritative receipt as cancellation before returning output or retiring a request. This synchronous observation scope adds no Task, waiter or await and preserves native cancellation delivery. SQLite read shielding retains its original queue-drain policy.

Session-backed lifecycle owners apply the same tool-terminal classifier through release, retry, status, compensation and close. SDK or audit uncertainty retains the original resource and physical attempt; it cannot become ordinary cancellation or authorize a replacement release. Explicit denial-scope evidence still distinguishes a veto before physical work from unresolved release, and existing deadlines remain unchanged.

Training startup reconciles the opening receipt even when its physical Task
returns successfully. If that unpublished result must be rejected, the same
original session is released before the outcome reaches the caller. Finite
SDK/audit observation errors preserve the original source graph as well as
Core control evidence; this isolation does not grant execution authority.
Training adapters propagate the shared tool-terminal classification through
status and provider error delivery.
