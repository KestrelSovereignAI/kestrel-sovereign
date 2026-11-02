---
type: Architecture Spec
title: Governed visual computer runtime
description: Governed provider-neutral visual runtime source, local browser delivery boundaries, and remaining remote and product acceptance gates.
resource: /docs/architecture/VISUAL_COMPUTER_RUNTIME.md
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

# Governed visual computer runtime

`VisualComputerFeature` owns `/api/visual-computer` and its console Computer
panel. It is independent of the legacy `ComputerUseFeature` host file/shell
grants. Default configuration exposes no visual tools or routes.

The host loads environment factories from
`kestrel_sovereign.environment_providers` and computer factories from
`kestrel_sovereign.computer_drivers`. Entry points construct their own SDK
protocol with trusted private settings. No shell/local-host fallback exists.
The built-in `local-docker-browser` adapter ships in Core; its pinned worker
image must be built and verified separately. Registering its factory is not
evidence that its worker or Docker boundary is ready.

Enable only after installing the concrete driver and building its pinned
worker image. The local Docker image reference identifies the inspected local
image configuration digest; it is not a mutable tag or an implicit pull.

```toml
[features.visual_computer]
enabled = true
granted_actions = [
  "visual.admit", "visual.status", "visual.observe", "visual.act",
  "visual.stop", "visual.takeover", "visual.handback", "visual.release",
]
permitted_locations = ["local"]
max_lifetime_seconds = 1800
max_hourly_cost_usd = "0"
max_total_cost_usd = "0"
operation_seconds = 30

[features.visual_computer.providers.local-docker-browser]
docker_socket = "/var/run/docker.sock"
image = "worker@sha256:REPLACE_WITH_LOCAL_IMAGE_CONFIGURATION_DIGEST"
profile_id = "offline-v1"

[features.visual_computer.drivers.local-docker-browser]
docker_socket = "/var/run/docker.sock"
```

Only a current explicit sovereign API-key credential is admitted. Cookie, JWT,
peer, unattended and delegated callers have no visual grant. The host derives
tenant/principal/acting-agent from the original selected agent, never JSON,
resource names, causation frames, URLs or a submitted OperatorContext. The
request context, current credential, exact feature generation and visual grant
are checked at operations and private delivery through the shared host
capability and outer ASGI transport gates. Full raw-data privacy policy applies
to views, observations, takeover and handback. Stop and physical release return
only minimal safety acknowledgements and remain usable in volatile modes.
Remote environments also need current processing/destination policy; no remote
provider is enabled by these local settings.

The original agent's AsyncDatabase stores encrypted request constraints,
accepted quote, resource lease, private connection, session, current frame
metadata, original action intents and receipts. Screen pixels are response-only;
raw recording is disabled. The existing offline key-rotation registry rewraps
the visual envelope root. Missing or incorrectly keyed original custody fails
closed. Single owning host-process semantics match the existing artifact
service; this is not a distributed authority registry.

Admission commits the request and exact accepted quote before acquisition.
Pending physical leases can open a driver, but only a subsequent provider
inspection proving ready enables interaction. Unknown acquisition/open outcomes
retain the original request/session/quote/connection and require explicit
`POST /requests/{request_id}/reconcile`. A released request is a tombstone and
cannot silently acquire another resource. Reconciliation never extends the
original lease lifetime.

Each action requires the current frame, epoch, original action ID and a deadline
within both the current grant and operation bounds. Acceptance is persisted
before input. Reusing an action ID with changed input conflicts. Accepted
uncertain input cannot be replayed; use
`GET /sessions/{session_id}/actions/{action_id}`. A fresh host process fences
normal input until `POST /sessions/{session_id}/reconcile` verifies the exact
accepted session and all unknown actions have been resolved. A replacement
worker cannot be adopted through a reused resource name.

New host action acceptance records a preparation before the one-shot durable
dispatch CAS. Original authorization, session/lease/epoch and the accepted frame
revision are checked again after that CAS and before the driver is entered.
Preparation alone permits no input: an interrupted preparation can become an
exact `denied` receipt during original-ID status reconciliation. A known local
failure after acknowledged dispatch CAS but before driver entry also records
that no input was dispatched. Legacy receipt-less records and a lost dispatch
checkpoint acknowledgement remain dispatch-possible; they are never replayed.
A generic driver conflict, timeout, cancellation or invalid receipt is not
proof of no effect. The browser worker journals exact negative receipts for
its authenticated synchronous pre-effect refusals, including stale frames and
epochs. Status reconciles those receipts using the original action ID; input
resumes only after unresolved actions have terminal receipts.

Agent and manual input have distinct original result owners. Before consuming
a manual frame, the host authenticates the full human request, installed
consumer lease and pixel-free frame against the same encrypted action, private
lease and session rows under both current input and view grants. Its fixed
negative-result owner retains the original database, key, privacy holds and
ciphertext CAS through native writer completion. Global turn retirement permits
only that original, physically joined no-dispatch fact to clear the pending
projection; it grants no ordinary SQL, input, observation or delivery. Later
frame clearing or control changes do not erase this historical fact. Agent and
human owners cannot settle each other's rows or facts.

The Docker trace includes nested manual action RPCs and marks input possible
before the first partial stdin write. A partial send or lost acknowledgement
stays unknown. When the complete original RPC reaches the worker after Stop or
handback, the worker authenticates its originally issued consumer lease before
journaling a pre-effect denied receipt; current lease eligibility is still
checked before any input or repeated act delivery. The manual composition uses
published SDK 0.45 with normal-index host and worker locks. Core successor
review, CI and actual merged-base host/worker acceptance remain release gates;
SDK installability does not establish Core product readiness.

The default limits are 256 active or unresolved requests and 4096 retained
actions across an owner's live sessions and unresolved released sessions.
Terminal action payloads are reclaimed only after verified physical release;
unknown actions and uncertain leases keep both custody and their capacity
charge. Released requests retain immutable encrypted constraints, original
session/owner/lease bindings and their IDs in the same canonical request row.
Obsolete connection, frame and control derivatives are removed. A released
session cannot accept input, control, reconciliation or new action IDs, even
when its terminal action payloads were reclaimed. Exact-ID request lookup
retains its tombstone; ordinary lifecycle/operator listings scan only active
or unresolved custody. Minimal anti-resurrection request history necessarily
grows with distinct released IDs; these live-capacity bounds do not claim a
bounded lifetime tombstone store. Raw pixels remain response-only.

The operator view shows custody phase, controller, epoch, resource state and
reconciliation requirements. Stop fences input cooperatively. Takeover drains
or reconciles accepted work through the driver's priority control path and
suspends agent observations. Handback is explicit, advances the epoch and
requires a new observation before input. Physical release is a distinct
operation; timeout, failure and lease expiry do not prove physical cleanup.

A lost control reply leaves the original control request pending in encrypted
custody. Retry that SAME request ID and input under current authorization.
The existing worker returns its recorded successful receipt without applying
the control twice, after checking the original owner, lease, driver and epoch.
Worker-side reconciliation requires that exact resulting session to remain
current and unexpired. A later transition, changed input or unknown original
close cannot become a successful old worker receipt or restore an older input
epoch. A cached host acknowledgement is an original factual receipt; it does
not establish current readiness or permission for further input.
Safety calls consume no normal action/history slots. Lifecycle withdrawal
fences retained sessions, while periodic maintenance releases expired or idle
original leases. Uncertain cleanup retains its lease/connection for retries.

This milestone's host and persistent-storage tests use bounded SDK driver
doubles and actual server authentication/routing. They do not establish real
Docker, remote-provider, human interactive transport, network-egress, workspace
transfer or full epic readiness. Concrete driver and product composition must
supply those measurements. Offline browsing stays explicitly gated from network
access until active-call egress holds the exact originating authorization lease;
autonomous browser authority is not inferred from attribution.

## Active-call egress and priority safety

A concrete network adapter may accept the trusted `EgressGuard` callback in
`visual_computer.egress`. The runtime activates it only while one original
`driver.act` RPC is running. Each borrowed gateway task presents the exact
session/owner/lease/epoch/action ID, holds the callback context through socket
and reply cleanup, and calls its synchronous check before each network effect.
The callback does not mint an OperatorContext, consult causation, acquire a
replacement privacy lease, or survive the originating RPC. The runtime revokes,
cancels and physically joins accepted gateway tasks before that original lease
exits. Job counts and grant/action deadlines are bounded. These host hooks are
not proof that the concrete gateway's DNS, redirect, transport or destination
policy has passed its separate production tests.

Priority Stop and release retain exact binding, freshness, current grant,
capability and boundary checks, while bypassing fresh privacy/authority read
admission. Otherwise a queued privacy writer could wait for a blocked action
while Stop waited behind that same writer. This exception returns no raw data
and is never registered as a raw-operation lease: nested observations still
acquire the ordinary gates and satisfy the complete privacy policy. Artifact
erasure keeps its existing admission behavior.

Workspace transfer uses the same original host operation boundary but requires
a verified paused session with no controller and the exact current control
epoch. A new control intent invalidates that transfer's synchronous check.
Observation/model operations continue to require a ready agent controller.

`visual_plan(session_id, prompt)` invokes `VisualInference` using the original
bound caller and the agent's existing LLM service. An optional trusted
`visual_model` setting selects an explicit model; model input cannot override
that selection. The authenticated `POST /sessions/{id}/plan` accepts only
`{"prompt":"task text"}` and uses the same operation. An optional trusted
`visual_request_options` mapping carries SDK reasoning or native `raw` options
through the existing adapter, and requires an explicit configured visual model.
It is not accepted from a tool task, page, model output or HTTP caller.
Current processing privacy forces the existing router's local
policy when required. The result contains one host-generated, observation-bound
ActionRequest or a finished answer. It does not automatically submit input;
execution and original-ID status reconciliation use the existing action tools.

`visual_computer_step(session_id, action_id, prompt)` plans one action and submits
it within that same original caller operation. The caller supplies a stable
`action_id`. A native, current admission first claims that ID in the existing
encrypted action table and waits for the original transaction COMMIT and reset
acknowledgments. Only the exact original step owner may plan and promote that
claim into the ordinary prepared-action and one-shot dispatch path. Inference
holds neither a SQL transaction nor the session lock, so priority Stop can
fence it promptly. Both OBSERVE and ACT remain current through model and driver
I/O; ordinary tool security policy still applies. No new grant, queue or store
is introduced, and the current claim writer has no late-result privilege.

The claim binds the original owner/session, prompt and host model/configuration
facts. A repeat checks the same row before inference: a terminal SDK receipt or
host no-action result is returned, while planning and uncertain dispatch remain
UNKNOWN and are never planned or typed again. Changed prompt, configuration or
source facts conflict with that original ID. Real input reconciliation also
requires STATUS; returning an already terminal result does not. A host no-action
result is a separate terminal fact with `receipt=null`, not an SDK input receipt.
Missing or uncertain claim acknowledgment admits no model or input. Absence by
itself is insufficient: a replacement must win a fresh atomic INSERT and native
acknowledgment against every original writer. A committed planning intent stays
charged across restart, Stop and physical release. Terminal payload reclamation
follows the existing verified-release rules; unresolved ownership is not freed
from task completion or inventory absence. The existing 15-second action
proposal, frame, lease and original operation deadlines remain unchanged.

Each proposal starts with a fresh driver screenshot and retains the original
session/grant/privacy operation through model delivery. PNG identity, dimensions,
byte count, epoch and freshness are checked before image transport. Every model
attempt uses the existing router, configured fallback policy, image folding and
finished-generation gate. Concrete model discovery must positively advertise
vision, and the adapter must identify a supported byte-image transport. Cold or
unknown vision metadata is an explicit refusal; refresh the existing model
catalog before using that route. Text readiness alone is insufficient, including
on a private inference lease. No browser/model location pairing is implicitly
authorized by choosing the other location.

The original host caller deadline uses the configured `operation_seconds`,
bounded by `max_lifetime_seconds`; the default remains 30 seconds. Increasing
that trusted budget never extends an existing context, resource lease, accepted
action or the independent 60-second frame freshness check. Grant and control
revocation remain immediate, including during a longer local model call. Local
Ollama model vision evidence comes from `/api/show` capabilities, not model-name
heuristics. Rejected model output is never repaired into an accepted action.

The existing native execution admission carries the retained visual check to
each actual HTTP dispatch, including SDK retries after request observers. It is
also checked before and after non-HTTP provider attempts. A takeover, grant
withdrawal or expired operation stops fallback and withholds the proposal. A
retired callback cannot regain authority in a copied task. The model may propose
only an SDK action or completion: IDs, deadlines, owner and epoch are host-owned.
Generated clicks are checked against the original frame before being offered;
execution revalidates them through the runtime's accepted-action ledger.

Before offering input, the host observes again under the retained original
caller/control operation. Its verified pixels, dimensions and epoch must match
the model's original frame exactly; any change refuses the proposal. The action
uses that fresh observation ID and a short deadline. The response preserves both
`model_frame` and the action's `frame` as distinct correlation facts. This does
not renew the original model-frame freshness, caller deadline or control epoch,
and does not relax a driver's shorter input-frame window. No new model call or
input is submitted automatically when the screen changes.

Visual prompts, images, answers and provider exception chains are withheld from
LLM telemetry. Content-free model/usage/failure classification remains available.
Private artifact/computer HTTP URLs are excluded before optional OTel span
construction; access and inner error paths redact private identifiers as well.
