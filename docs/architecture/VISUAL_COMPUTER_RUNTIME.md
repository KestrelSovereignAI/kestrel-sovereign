# Governed visual computer runtime

`VisualComputerFeature` owns `/api/visual-computer` and its console Computer
panel. It is independent of the legacy `ComputerUseFeature` host file/shell
grants. Default configuration exposes no visual tools or routes.

The host loads environment factories from
`kestrel_sovereign.environment_providers` and computer factories from
`kestrel_sovereign.computer_drivers`. Entry points construct their own SDK
protocol with trusted private settings. No shell/local-host fallback exists.
The built-in `local-docker-browser` adapter is delivered separately; registering
its factory is not evidence that its worker or Docker boundary is ready.

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

The operator view shows custody phase, controller, epoch, resource state and
reconciliation requirements. Stop fences input cooperatively. Takeover drains
or reconciles accepted work through the driver's priority control path and
suspends agent observations. Handback is explicit, advances the epoch and
requires a new observation before input. Physical release is a distinct
operation; timeout, failure and lease expiry do not prove physical cleanup.
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
