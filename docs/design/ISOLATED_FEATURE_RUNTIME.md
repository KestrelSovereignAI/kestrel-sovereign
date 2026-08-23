---
type: Design Note
title: Isolated-Feature Runtime
description: Draft design for running Kestrel features in isolated dependency runtimes.
resource: /docs/design/ISOLATED_FEATURE_RUNTIME.md
tags:
- docs
- design
- design-note
- feature-runtime
timestamp: '2026-06-20T00:00:00Z'
status: proposed
owner: feature-runtime
canonical: false
generated: false
privacy: review-before-public
---

# Isolated-Feature Runtime

> Status: **Proposed (Draft)** — 2026-06-17. Author: UncleSaurus.
> Reference consumer: WhatsApp-Web channel (`kestrel-channel-whatsapp`).

## 1. Problem

Kestrel features load as entry-point classes that are **imported into the
single host process and venv** (`kestrel_sovereign/feature_registry.py` →
`importlib.metadata.entry_points()`; the `Feature` base class is instantiated
in-process). Every in-process feature therefore shares **one** Python
dependency graph.

That breaks down when a feature needs dependencies that conflict with the
host's. The concrete trigger: the WhatsApp-Web transport library **neonize**
(0.3.18, the version with a working bundled native lib) requires
**protobuf 7**, while kestrel's gRPC/Google-AI stack hard-caps protobuf
**below 6** (`grpcio-status <6`, `google-ai-generativelanguage <6.0.0dev`).
These cannot coexist in one venv. neonize 0.3.10 is protobuf-5 compatible but
ships a broken/stub native lib. There is no single venv that satisfies both.

This is not WhatsApp-specific: any feature pulling a heavy/native dependency
(future ML runtimes, vendored SDKs, protocol libs) can collide with the host.

### Today's situation: three bespoke point-solutions, no generic pattern

| System | What it is | Isolation mechanism |
|---|---|---|
| **Talon** (`kestrel-feature-talon` + `kestrel-talon`) | external coding coordinator + standalone runner | Both are independently installed; the feature package owns `TalonCoordinatorFeature` and invokes the standalone CLI without a core implementation fallback. |
| **MCP** (`kestrel-feature-mcp`) | third-party tool servers | external **subprocess/SSE servers** spoken to over the MCP protocol; `manager.py` runs a background session loop with reconnect |
| **FeatureFeature** (`kestrel-feature-features`) | agents *authoring* new features | workflow-driven build/review/**publish** of feature *packages* — **authoring only, no runtime/venv concern** |

There is **no first-class "a feature runs in its own venv" capability**. The
Talon and MCP feature packages each hand-roll discovery + subprocess supervision + IPC differently;
FeatureFeature covers only authoring. Building WhatsApp's isolation as another
one-off would be a fourth snowflake.

## 2. Goals / Non-goals

**Goals**
- A first-class way for a feature to declare it runs in **its own venv/process**.
- The platform handles **discovery, lifecycle (start/supervise/health/stop),
  and IPC** generically, so a conflicting-dep feature composes cleanly.
- The isolated feature presents a **normal `Feature` surface** to the agent
  (tools, inbound routing, config) — callers shouldn't care that it's remote.
- Reuse, not reinvent: align with MCP's transport/supervision lessons; let
  Talon optionally migrate; let FeatureFeature *scaffold* isolated features.

**Non-goals**
- Replacing in-process features (the default stays — most features have no
  dependency conflict and pay no IPC cost).
- A remote/distributed feature mesh — this is **localhost, same-host**
  isolation. (Cross-host is A2A's job; see §6.)
- OS sandboxing/security isolation. The runtime provides dependency, lifecycle,
  namespace-ownership, and accidental-path containment; it does not make
  mutually hostile same-UID feature code safe (the driver is *dependency*
  isolation).

## 3. Proposed design

The capability spans three layers. None own it today; each gets a small,
well-scoped addition.

### 3.1 Declaration (authoring side)

A feature package declares its runtime in a `[tool.kestrel.feature]` table in
its `pyproject.toml` (read from installed metadata, no import required):

```toml
[tool.kestrel.feature]
runtime = "isolated-venv"          # default: "in-process"
service = "kestrel_whatsapp_web.service:main" # Python module:callable
# A bare [project.scripts] name such as "kestrel-whatsapp-service" is also valid.
# Optional standalone-only mutable selection. Hosted declarations must use an
# absolute, already-built immutable operator venv.
venv = "/opt/kestrel/prebuilt/whatsapp-service-venv"
```

- `runtime = "in-process"` (default, omitted) → today's behavior, unchanged.
- `runtime = "isolated-venv"` → the loader does **not** import the heavy
  package; it launches/connects the declared service in its own venv.

`service` accepts two deliberately separate forms:

- A bare portable console executable starts with an ASCII letter or digit and
  may then contain ASCII letters, digits, `.`, `_`, and `-`. Separators, drive
  syntax, leading punctuation, trailing dots, and Windows device names are
  rejected. Core verifies and launches the same exact file below the venv's
  `bin/` (or `Scripts/` on Windows), so metadata cannot make verification
  inspect one wrapper while execution selects another.
- A Python callable uses exactly `module:callable`. Every dotted module
  component and the callable must be a non-keyword ASCII Python identifier;
  separators, traversal, drive syntax, extra colons, and dotted or
  expression-like callable targets are rejected. Core launches this form
  through the current venv's interpreter in Python isolated mode (`-I`,
  available since Python 3.4), keeping the mutable child working directory,
  user site, and Python environment injection out of the import boundary. This
  remains compatible with older operator-prebuilt feature interpreters that do
  not implement Python 3.11's `-P`. Core also disables bytecode writes (`-B`),
  so verification cannot mutate an immutable prebuilt venv. Before any Core
  provisioning manifest is stamped, Core uses that same isolated interpreter
  to import the module, resolve the attribute, and prove it is callable. It has
  no console wrapper, so console-wrapper relocation repair is not applicable.

Core's child distribution and SDK freshness probes use the same `-I` boundary.
Neither service verification nor a freshness decision can import a same-named
module from the host process working directory.

The operator-level `KESTREL_FEATURE_<NAME>_BIN` setting is a complete executable
override and preserves the historical ability to omit `service`. When BIN is
present, Core validates the hosted prebuilt executable but does not parse or
forward unused service metadata. If BIN is absent when launch paths are
resolved, `service` is required and the appropriate grammar above is enforced
before any venv preparation or child start.

FeatureFeature's design/scaffold stage emits this table (and the `service/`
sub-project layout) when a proposed feature declares conflicting deps.

### 3.2 Runtime loading (core)

When the feature loader sees `runtime = "isolated-venv"` for an enabled
feature, instead of importing the class it instantiates a **proxy `Feature`**:

- The proxy implements the `Feature` contract (tools, `get_router`, config) by
  **forwarding over IPC** to the isolated process.
- On `initialize()`, the proxy resolves the service's venv and **launches**
  the service process (a background asyncio task supervises it — the MCP
  `manager.py` session-loop pattern), then queries the service for its tool /
  skill metadata and **mirrors** it so the agent sees ordinary tools.
- On `shutdown()`, the proxy stops the service.

**venv resolution / provisioning** (Talon's discovery order, generalized):
1. `KESTREL_FEATURE_<NAME>_VENV` / `..._BIN` env override.
2. Configured path (host config). In hosted mode, metadata `venv` must be an
   absolute, already-existing immutable prebuilt venv; relative paths are
   rejected because they otherwise resolve through the shared Core CWD.
3. Convention: `<package>/<venv>/.venv` (e.g. `service/.venv`) or sibling repo.
4. On-demand provision: `uv venv` + `uv pip install` of the service project
   (opt-in; gives true "feature brings its own venv" UX).

#### Rolling config authority and operator boundary

An isolated proxy's durable config is DID-scoped for new agents.  Older
proxies used the unscoped `feature_config:<class-name>` row, however, so a
same-agent visible legacy row remains the authority for the duration of a
mixed-version rollout.  A proxy never copies that row to a scoped key: old and
new replicas must contend on the same CAS row.

The graph store cannot atomically compare an old binary's legacy key with the
new scoped key.  The proxy checks for visible legacy authority before every
scoped read and write, then again before an SDK lifecycle hook or traffic can
continue.  If legacy appears after a scoped stage, it fences the transition and
quarantines the proxy rather than promoting, deleting, or live-applying across
two authorities.  This may leave an orphaned scoped *candidate* in the final
cross-key race; a second read narrows that window but does not close it.

After all old replicas have been drained, an operator must inspect the two
same-agent rows, choose the intended active config, and remove or reconcile an
orphan only in an exclusive maintenance window.  Do not delete a legacy row
while an old replica may still write it, and never perform this cleanup against
another tenant's invisible rows.  New proxies continue to converge on legacy
for as long as that row is visible.

#### Hosted runtime namespace contract

A standalone filesystem agent continues to place isolated runtime data beside
its storage database. This includes AgentManager's existing per-agent SQLite
layout: upgrading Core does not move its feature state or WhatsApp credentials.
A pathless/shared-database hosted factory must instead construct `KestrelAgent`
with all three of:

```python
from kestrel_sovereign.features.isolated_runtime import (
    derive_isolated_runtime_namespace,
)

KestrelAgent(
    ...,
    isolated_runtime_root=operator_runtime_root,
    isolated_runtime_namespace=derive_isolated_runtime_namespace(
        authenticated_user_id,
        companion_did,
    ),
    # Managed upgrade input only: the released hosted layout below this
    # exact agent data directory. It is never used as a runtime fallback.
    isolated_runtime_legacy_root=agent_data_dir / "feature_venvs",
    isolated_runtime_hosted=True,
)
```

The root is operator-owned; the namespace is a canonical relative path. The
root's parent must already exist, so an absent volume mount cannot be replaced
silently with ephemeral directories. Core may create the root leaf at `0700`.
If the root already exists, Core verifies service-account ownership and rejects
group/world-writable mode, but does not chmod operator state (a safe `0755`
root remains `0755`).

Core rejects traversal, aliases, reserved/case-colliding path components, and
binds the namespace to the agent DID with a private ownership marker. Marker
publication writes and fsyncs a same-directory temporary file, then installs it
with exclusive link semantics and fsyncs the directory; a crash cannot expose
a partial marker. An interrupted link/unlink tail is recovered only when the
temporary name is a hard link to the complete marker. A missing, short, or
otherwise corrupt visible marker is never guessed or automatically replaced:
the operator must verify custody and remove the corrupt marker before retrying.
Secure cleanup retains the valid ownership marker throughout its recursive
sweep and removes it only as the final entry immediately before removing the
namespace leaf. A partial filesystem failure therefore retains custody proof
and can be retried after the operator corrects the underlying error.
POSIX directory creation is descriptor-relative and no-follow, with inode/path
binding checks before child launch. Multi-component namespaces are supported,
but allocated namespace leaves must be prefix-free: secure cleanup refuses a
tree containing any descendant ownership marker before deleting its contents.

A namespace already bound to a different DID, an unsafe operator root, a
symlink/path swap, or a corrupt marker is a containment/ownership violation and
fails hosted agent discovery closed. Ordinary OS preparation failures such as
ENOSPC or a transient descriptor limit instead make that optional isolated
feature unavailable; they do not suppress the agent's mandatory features.

`isolated_feature_data_dir` remains only a standalone compatibility input. It
is not an adequate hosted containment boundary because it does not identify a
trusted root separately from tenant-controlled namespace material. PostgreSQL
agents constructed without a filesystem storage path are treated as hosted,
so an old factory cannot silently recover the shared `agent_data/default`
fallback.

The default standalone/SQLite WhatsApp venv remains under
`<agent>/feature_venvs/WhatsAppFeature/.venv`. Standalone launches preserve the
historical child environment and cwd: `KESTREL_FEATURE_DATA_DIR` still takes
precedence over `KESTREL_DATA_DIR`, which still takes precedence over the
prebuilt/default venv's `sys.prefix.parent/whatsapp_service` fallback. Core does
not inject its hosted canonical data-dir variable into a standalone child, so
an upgrade neither relocates credentials nor changes Telegram/general feature
HOME, TMPDIR, XDG, or relative-path behavior. Operators remain responsible for
ensuring any process-wide standalone data-directory override is not shared by
agents that require distinct mutable state.

Released managed PostgreSQL agents also used that class-named layout under
their resolved per-agent data directory. The managed factory supplies that
exact old `feature_venvs` root as migration input. On first isolated-feature
startup, Core validates the old root and class directory without following
symlinks, verifies service-account custody, and atomically renames the complete
class directory (venv plus service state/credentials) into the DID-owned stable
namespace. A cross-device move, unsafe owner/mode, path race, or both-old-and-
new collision retains the old tree (and both trees for a collision) and
quarantines the optional feature for operator reconciliation; Core never
copies, merges, overwrites, or deletes ambiguous credentials.
Explicit agent offboarding checks this released per-agent root independently
of feature discovery, so state for a feature that was never loaded/adopted is
not orphaned while the new namespace is reported removed. A service-owned,
non-writable released tree is securely swept without following symlinks;
ambiguous custody or a nested namespace marker retains it and makes the whole
offboarding outcome visibly retained.

Moving a venv changes the absolute interpreter path embedded in console-script
shebangs. Core therefore records the canonical absolute venv path in its
private provisioning manifest. A directory rename observed in the current
startup, a mismatched non-empty path stamp, or a console wrapper containing a
foreign absolute interpreter is relocation evidence. When that evidence finds
a stale wrapper, Core performs one full package reinstall at the adopted
destination before child launch; an ordinary `--upgrade` is not accepted as
repair because already-satisfied packages may retain stale scripts. If repair
cannot complete, Core quarantines the optional feature rather than launch the
unusable wrapper. Import, configuration, and preparation quarantines are also
retained in the agent's rejected-contribution inventory, so detailed health is
degraded and names the unavailable feature instead of reporting a silent
healthy boot. A missing path stamp alone is not relocation evidence:
unchanged pre-upgrade venvs are positively probed and their path is atomically
backfilled without contacting an index. A migrated module-callable runtime (or
an already-repaired console wrapper) is similarly verified and adopted without
an unnecessary reinstall. The manifest is replaced atomically only after
install/adoption and distribution verification succeed, so a crash retries a
still-demonstrably-stale repair on restart. Service credentials and state beside
`.venv` are preserved, and a valid same-path stamp remains idempotent.

Every hosted feature receives a private mutable directory below that namespace
for its working directory, home, temp, XDG config/data/cache, channel artifacts,
provisioning manifest, and default venv. Hosted children inherit only a narrow
execution/locale/CA/proxy environment (`HTTP_PROXY`, `HTTPS_PROXY`, and
`NO_PROXY`, including lowercase spellings); feature configuration and secrets
travel in the per-agent initialize handshake. Process-wide legacy Telegram,
Twilio, WhatsApp, or `KESTREL_FEATURE_<NAME>_*` configuration is not silently
dropped or forwarded across tenants: hosted discovery names the offending keys
and refuses that optional feature until the values are moved into the agent's
persisted feature configuration. `..._BIN` and `..._VENV` remain host-side
artifact-selection inputs and are never exposed to the child. In hosted mode
these process-wide overrides are accepted only when the executable or complete
venv already exists; a venv carrying Core's provisioning manifest is refused.
Core never creates, upgrades, or stamps a process-wide override. The validated
operator-selected artifact may be shared, but every child process and mutable
workspace/cache/state path remains namespace-distinct.

Installed `[tool.kestrel.feature] venv = ...` metadata follows the same hosted
immutable-prebuilt contract as `..._VENV`: the value must be absolute, the venv
and executable interpreter must already exist, and the directory must not carry
a Core provisioning manifest. Core revalidates it immediately before the
mutation boundary and never creates, upgrades, or stamps it. Standalone agents
retain their historical mutable and relative metadata behavior.

The per-feature directory identity is a digest of the normalized distribution
name and declared feature class. Entry-point module paths and service runner
spellings are deployment details: refactoring either does not move tenant state
or credentials. On the first startup with this stable identity, Core atomically
renames a directory produced by the earlier distribution/class/entry-point/
service digest before creating the stable workspace. If both old and stable
trees exist, Core never merges or guesses between credential stores; it retains
both and marks that optional feature unavailable pending operator custody
reconciliation.

Hosted venv SDK/distribution probes use the same narrow execution/locale/CA/
proxy allowlist and receive no package credentials. `uv` provisioning and
feature build backends additionally receive only explicit `PIP_INDEX_*`,
`UV_*INDEX*`, and named `UV_INDEX_<NAME>_{USERNAME,PASSWORD}` package-index
settings. They do not inherit API/channel/tenant secrets, config-file or
keyring paths, SSH-agent authority, or the host process environment. Private
VCS dependencies that require those broader capabilities must be materialized
as an operator-owned prebuilt venv rather than widening the hosted subprocess
boundary. Core resolves `uv` to a trusted absolute host executable after
excluding the mutable feature venv from the search path, and provisioning uses
an explicit private per-agent/per-feature `UV_CACHE_DIR` in a dedicated
`provisioning_cache` directory. That cache is distinct from the child's
`XDG_CACHE_HOME`, is not exported to the child, and never relies on `HOME`,
passwd-database discovery, or an operator uv cache.

Namespace creation and ownership binding happen only during isolated-feature
startup. QR GETs resolve the cached path read-only and perform their filesystem
stat off the event loop; QR events use the already-prepared cached directory and
offload atomic file replacement/unlink work. A read or event therefore never
materializes a missing tenant namespace.

Explicit tenant offboarding is the runtime-state deletion boundary. Normal host
shutdown, restart, lifespan teardown, ordinary `DELETE /api/agents/{name}`, and
ordinary child termination stop/unpublish agents but retain every hosted
namespace. The DELETE compatibility default also leaves `multi_agent.toml`
registration intact, so a later host restart may start the agent again.
`DELETE /api/agents/{name}?offboard_runtime=true` is the explicit destructive
contract. It is available only to config-driven deployments. Core first resolves
the registered local DID without mutating configuration, refuses an absent or
ambiguous identity, then removes the persisted registration before runtime
deletion. A stopped/never-loaded registered agent follows the same DID-scoped
offboarding admission without being constructed. This prevents credential-less
autostart resurrection and avoids guessing a namespace from a routing name;
auto-discovery deployments are refused because safely deprovisioning them would
also require an explicit primary-storage policy.
Approved `terminate_child(..., offboard_runtime=true)` calls and rollback of an
uncommitted spawn carry the same separate destructive intent. After durable
shutdown, AgentManager verifies the ownership
marker against the removed DID and deletes the namespace with
descriptor-relative, no-follow recursion. A foreign/corrupt or nested marker,
unsafe path, or unsupported secure-deletion platform retains the tree and makes
offboarding fail visibly. Filesystem deletion is admitted while removal is
serialized but awaited only after manager, A2A, and scheduler lifecycle locks
are released. The administrative wait is bounded by
`KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S`, whose default is 30 seconds independently
of the ordinary five-second agent-shutdown bound. Timeout/cancellation
unpublishes the stopped agent, retains the exact cleanup worker for terminal
draining, and returns sanitized metadata with `runtime_cleanup_state=pending`:
the cleanup may still finish, so this is not a claim of permanent retention.
A completed cleanup failure instead reports `runtime_cleanup_state=retained`.
Neither response exposes the host path. If shutdown is quarantined, its retained
reaper deletes only when the original caller carried that explicit intent and
durable shutdown later succeeds. The initiating DELETE/tool call immediately
receives a typed pending-custody outcome rather than claiming deletion; the
reaper remains the sole owner of eventual deletion. The `terminate_child` tool
defaults to state retention and is permission-gated `ALWAYS_ASK` because its
explicit `offboard_runtime=true` variant is irreversible and ordinary ASK can
be promoted by auto/demo policy. For that destructive variant,
the tool reports pending/retained cleanup as partial success: the named child is
stopped and must not be retried, while any named child or descendant with
pending/retained runtime custody is identified separately for operator
reconciliation.
Storage-derived standalone/SQLite trees are outside hosted namespace cleanup
and retain their existing storage lifecycle policy. A destructive request for
such an agent therefore reports `runtime_cleanup_state=not_hosted`; it never
claims the storage-backed tree was removed. If the exact owned hosted namespace
is already absent, Core reports `runtime_cleanup_state=already_absent`. Both are
typed no-op custody outcomes (`runtime_offboarded=false`) rather than a false
deprovisioning success; routing withdrawal and persisted-registration removal
remain truthful separate fields.
Core-created standalone feature, workspace, and channel-artifact directories
are private `0700`; a pre-existing storage parent (including process CWD) is
operator-owned and its mode is never changed.

#### Same-service-account trust boundary

The hosted namespace contract is not an operating-system sandbox. Isolated
feature subprocesses normally run as the same service account as Core and each
other. POSIX `0700` prevents access by other UIDs; it cannot prevent a malicious
same-UID child from traversing a sibling namespace once it learns or guesses a
path. Absolute paths for only the child's own workspace are unavoidable launch
inputs (`cwd`, HOME/XDG/temp, and the feature data directory), and Core does not
publish the operator root, sibling paths, or retained host paths through its
HTTP error metadata. Those paths are locations, not capabilities.

Feature packages installed into one Core service must therefore be trusted at
the service-account boundary. A hosted operator that runs mutually distrustful
or tenant-supplied feature code must put each trust domain behind a real OS
boundary—separate UID, container, VM, or an equivalent mandatory sandbox—and
give it only its own mounted namespace. Namespace validation, no-follow file
operations, ownership markers, and private modes remain defense in depth
against traversal bugs, symlink swaps, accidental collision, and other Unix
accounts; they are not claimed to isolate hostile peers sharing a UID.

### 3.3 IPC + lifecycle contract (SDK)

A shared contract in `kestrel-sovereign-sdk` so every isolated feature (and,
over time, Talon/MCP) uses one substrate:

- **Transport**: line-delimited **JSON-RPC over the child's stdio**. The
  service is a supervised child process, so its stdin/stdout pipes are the
  natural channel — **fully cross-platform (macOS/Linux/Windows)**, no port or
  socket file, and inherently scoped to the parent (no other local process can
  connect). This matches MCP's primary local transport. For services that must
  be **shared across parents or outlive a single one** (the per-host scope,
  §6), fall back to **loopback TCP (127.0.0.1, ephemeral port) + a per-launch
  bearer token**. Unix domain sockets are intentionally **not** the baseline:
  Python's asyncio has no UDS support on Windows (`create_unix_server` is
  Unix-only), so a UDS baseline would break Windows hosts.
- **Lifecycle**: `start` → `ready`/`health` → graceful `stop`; host supervises
  with restart-on-crash + backoff; readiness gate before tools are exposed.
- **Surface**: the service advertises its tool/skill metadata; the proxy
  mirrors it. Tool calls are JSON-RPC requests. **Inbound** events (e.g. a
  WhatsApp message) are JSON-RPC notifications service → host, which the proxy
  routes into the normal path (`ChannelFeature.handle_inbound`, signals, etc.).
- **State/secret ownership**: the service owns its own data dir / session DB
  (e.g. WhatsApp linked-device credentials) under the standalone agent data
  directory or the explicit hosted runtime namespace described above.

#### External-ingress transition fence

Some isolated services acknowledge input at an external producer before their
stdio notification reaches Core (for example, a long-poll cursor).  Such a
producer opts into a pre-gate lifecycle by advertising both private
host-ingress names `external-ingress-quiesce` and `external-ingress-resume`.
These are ordinary versioned SDK host-ingress capability names, not channel
operations and never agent tools.

For each config transition, Core generates a fresh opaque 256-bit
`transition_id`, calls `external-ingress-quiesce` **while its traffic gate is
still open**, and closes/drains that gate only after a strict `quiesced`
acknowledgment. The service must stop its producer, cancel/reap a pending
external receive, and wait for an already-started event callback to settle
before returning that acknowledgment. On the shared stdio wire that callback's
notification precedes the quiesce response; the SDK client serially delivers
the notification before resolving the response, so the gate drain also covers
it. This is an ordering handshake, not a sleep or a dropped-event workaround.

If staging/promotion fails or the old child live-applies the change, Core
invokes `external-ingress-resume` with the same id **while its traffic gate is
still closed**. The service must return the strict `resumed` response before it
starts polling or produces another callback; only after Core receives that
response does it reopen the gate. A first callback produced immediately after
the response is held by the closed-gate deferred-ingress path, so it cannot be
lost in the response/reopen handoff. A replacement owns a new child and does
not resume the retired producer. A mismatched id, malformed acknowledgment,
timeout, cross-loop operation, or cancellation-ambiguous lifecycle result
fails closed and leaves the exact child under the established terminal/reaping
ownership rules. Services which do not advertise both names, including legacy
SDK/services, keep the existing safe replacement path unchanged.

### 3.4 Layout (same repo, two venvs)

Isolation is a **venv** boundary, not a **repo** boundary. Default layout keeps
both halves in one repo for atomic versioning and a shared contract:

```
<feature-repo>/
├── pyproject.toml            # thin feature → installs into the HOST venv
├── <pkg>/                    # proxy-friendly Feature + provider clients (no heavy deps)
├── service/
│   ├── pyproject.toml        # the service → installs into ITS OWN venv (heavy/conflicting deps)
│   └── <pkg>_service/        # the actual runtime (e.g. neonize client + IPC server)
└── tests/
```

A separate repo (Talon-style) remains allowed where history/ownership warrants
it; the contract is identical either way.

## 4. How Talon / MCP / FeatureFeature relate

- **MCP** is the closest existing substrate (external process + protocol +
  supervised session loop). The isolated-feature runtime is the *first-party*
  analog; it should **share MCP's supervision/transport code** where practical
  rather than duplicate it.
- **Talon** is an isolated-venv feature *avant la lettre* (separate venv + CLI,
  request/response rather than persistent IPC). It can **stay as-is** or
  migrate onto the contract later — non-blocking.
- **FeatureFeature** gains the ability to **scaffold** isolated features
  (emit the `[tool.kestrel.feature]` table + `service/` layout) in its
  design/scaffold stage. No runtime changes there.

## 5. Reference consumer: WhatsApp-Web

`kestrel-channel-whatsapp` (transport already proven end-to-end: QR link +
send + receive via neonize):

- **Thin feature** (host venv): `WhatsAppFeature` with `provider="web"`; the
  web provider is a proxy client to the service. Twilio provider stays
  in-process (no conflict).
- **Service** (`service/`, own venv): neonize 0.3.18 + protobuf 7 + libmagic;
  holds the WhatsApp-Web socket, owns the session DB, exposes send + emits
  inbound over the IPC contract.
- Inbound → service notification → proxy → `ChannelFeature.handle_inbound` →
  `channel.message` cognition signal (the existing, proven path). Loop guards
  (skip-from-self, dedupe) live in the service.

## 6. Security & open questions

- **Transport**: ~~Unix socket~~ → **resolved: stdio JSON-RPC** for the default
  child-process model (portable incl. Windows, parent-scoped, MCP-aligned);
  **loopback TCP + token** for shared/standalone services. UDS rejected as
  baseline (no asyncio UDS on Windows). A2A (DID-signed) is overkill for
  same-host. **Resolved.**
- **venv provisioning** → **resolved: per-agent, auto-provisioned**. Each agent
  gets its **own copy** of the feature venv, created on demand (`uv venv` +
  `uv pip install` of the service project) under its standalone data root or
  explicit hosted namespace (`<runtime>/feature_venvs/<feature>/`). An explicit
  path override stays as a dev convenience and is never mutated when it is an
  operator-prebuilt environment. **Resolved.**
- **Service scope** → **resolved: per-agent**. One service instance per agent
  (each owns its venv copy and its session DB, e.g. its own WhatsApp link).
  Per-host sharing is explicitly out for now. **Resolved.**
- Socket perms `0600`, service runs as the same user, no network listener.

## 7. Phasing → issues → Talon

Each phase becomes an issue in the named repo; Talon implements in chunks.

1. **SDK — IPC/lifecycle contract** (`kestrel-sovereign-sdk`): JSON-RPC-over-UDS
   transport, `IsolatedFeatureService` base, lifecycle/health, surface-advertise
   protocol. *(foundational; others depend on it)*
2. **Core — loader honors `runtime=isolated-venv`** (`kestrel-sovereign`): read
   `[tool.kestrel.feature]`, proxy `Feature`, venv discovery/provision, MCP-shared
   supervision.
3. **Reference consumer** (`kestrel-channel-whatsapp`): `service/` sub-project
   (neonize) + proxy web provider on the contract; end-to-end test.
4. **Authoring** (`kestrel-feature-features`): scaffold isolated-feature layout
   in the design stage.
5. **(Optional, later)** Migrate/align Talon & MCP onto the shared substrate.

Phases 1→3 deliver WhatsApp on the generic runtime; 4–5 are follow-ups.

## 8. Future direction: toward per-agent venvs

Today all agents on a host share one venv and one process. The per-agent,
auto-provisioned venv model adopted here (§6) points at a larger trajectory:
**each agent eventually getting its own venv** (and potentially its own
process) for full per-agent dependency and runtime isolation.

The isolated-feature runtime is a deliberate **stepping stone** toward that:
the machinery it introduces — per-agent venv provisioning under the agent data
dir, supervised child processes, and a portable stdio IPC/lifecycle contract —
is the same substrate per-agent process isolation would need. This design does
not commit to per-agent processes, but is intentionally compatible with it:
nothing here assumes a single shared host venv, and the provisioning path is
already agent-scoped. Treat full per-agent-venv isolation as a separate future
proposal that can build on this one.
