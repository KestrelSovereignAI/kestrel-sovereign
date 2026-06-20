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
| **Talon** (`kestrel-talon`) | external coding-agent runner | **separate repo + its own `.venv`**, invoked as a CLI; thin `TalonCoordinatorFeature` lives in core (`features/talon/coordinator.py`), discovery is env → PATH → sibling `.venv` |
| **MCP** (`kestrel-feature-mcp`) | third-party tool servers | external **subprocess/SSE servers** spoken to over the MCP protocol; `manager.py` runs a background session loop with reconnect |
| **FeatureFeature** (`kestrel-feature-features`) | agents *authoring* new features | workflow-driven build/review/**publish** of feature *packages* — **authoring only, no runtime/venv concern** |

There is **no first-class "a feature runs in its own venv" capability**. Talon
and MCP each hand-roll discovery + subprocess supervision + IPC differently;
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
- Sandboxing/security isolation as a primary aim (that's a bonus, not the
  driver — the driver is *dependency* isolation).

## 3. Proposed design

The capability spans three layers. None own it today; each gets a small,
well-scoped addition.

### 3.1 Declaration (authoring side)

A feature package declares its runtime in a `[tool.kestrel.feature]` table in
its `pyproject.toml` (read from installed metadata, no import required):

```toml
[tool.kestrel.feature]
runtime = "isolated-venv"          # default: "in-process"
service = "kestrel_whatsapp_web.service:main"   # entry point of the service
venv = "service"                   # path (relative to package) of the service's own venv/project
```

- `runtime = "in-process"` (default, omitted) → today's behavior, unchanged.
- `runtime = "isolated-venv"` → the loader does **not** import the heavy
  package; it launches/connects the declared service in its own venv.

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
2. Configured path (host config).
3. Convention: `<package>/<venv>/.venv` (e.g. `service/.venv`) or sibling repo.
4. On-demand provision: `uv venv` + `uv pip install` of the service project
   (opt-in; gives true "feature brings its own venv" UX).

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
  (e.g. WhatsApp linked-device credentials) under the agent data dir.

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
  `uv pip install` of the service project) under the agent data dir
  (`<agent_data>/feature_venvs/<feature>/`). An explicit path override stays as
  a dev convenience, but production provisions per agent. **Resolved.**
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
