# Kestrel: Sovereign AI Agent Framework

> Build AI agents that nobody can take away from their users — not you, not the cloud, not the next pivot.

Kestrel is a pre-1.0 framework for creating autonomous AI agents with cryptographic identity, privacy-aware memory, and constitutional governance. An agent's identity, keys, and local data remain under its operator's control; persistence, cloud routing, response auditing, and specialized integrations are explicit configuration or feature choices. See [Feature maturity](#-feature-maturity) for the current boundaries.

### Three Pillars

| Pillar | What it means |
|--------|--------------|
| **Portable DID identity** | Cryptographic identity the operator controls. Exportable, self-hostable, cloud-optional — the agent is not bound to a model provider. |
| **Privacy-aware memory** | Local-first search, knowledge graph retrieval, and RAG when the selected privacy mode permits persistence. `EPHEMERAL` conversations are intentionally not retained. Conversation history, file blobs, identity private keys, and agent-resource bodies encrypt at rest when `KESTREL_DATA_KEY` is set; saved-item and RAG document-chunk bodies remain plaintext columns until application-layer encryption lands (see [Encryption at Rest](#-encryption-at-rest)). SQLAlchemy-backed vector storage is in tree for saved items, document chunks, and conversation history; saved-item and RAG embeddings route through the active LLM provider when that provider supports embeddings. |
| **Constitutional governance** | Every agent runs under an audited set of principles enforced *above* the LLM. Genesis audit on creation. Amendment requires cryptographic signature. |

### What's in core, what's an add-on

`pip install kestrel-sovereign` gives you a complete, working sovereign agent: identity, memory, constitution, privacy modes, multi-LLM support, local guarded compute, and a Cloud Run deployment path. Everything you need to run an agent locally with zero cloud commitment.

Package ownership has four distinct runtime forms:

| Boundary | Ownership and install behavior |
|---|---|
| **Bundled Feature** | A Feature lifecycle class discovered from `kestrel_sovereign/features/`; it ships in `kestrel-sovereign` and needs no separate install. |
| **Extracted Feature package** | A separate distribution, such as Talon, voice, MCP, GitHub, wallet, council, or observability, that registers Feature classes through `kestrel_sovereign.features`. |
| **Provider package** | A separate backend distribution, such as RunPod, Vast.ai, GCP Compute, a voice cloud backend, or a storage backend, that implements a provider contract through a provider-specific entry point. It is not a Feature lifecycle package. |
| **Standalone tool** | An independent control surface. `kestrel-talon` is the standalone coding engine and is modeled separately from its independently installed `kestrel-feature-talon` coordinator. |

The base install also contains runtime components that are not Feature
lifecycle classes—for example, `PrivacyAgent`. The registry calls these
`bundled-component` rather than pretending they are Feature packages. The
canonical ownership contract and live in-tree inventory are in
[`KESTREL_FEATURES.md`](KESTREL_FEATURES.md); the machine-readable catalog is
[`kestrel_sovereign/data/feature_registry.toml`](kestrel_sovereign/data/feature_registry.toml).

## 🚀 Quick Start

### Prerequisites
- Python 3.11-3.14
- [uv](https://docs.astral.sh/uv/) (for package management)
- [Ollama](https://ollama.ai) (optional - for local LLM inference without API keys)
- **Linux: glibc 2.34 or newer** (Ubuntu 22.04+, Debian 12+, RHEL 9+). The
  post-quantum signing dependency publishes `manylinux_2_34` wheels only, so on
  an older glibc — Ubuntu 20.04, Debian 11, RHEL 8 — installation falls back to
  building it from source and requires a Rust toolchain. macOS and Windows are
  unaffected.

### Install uv

If you don't have `uv` installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Installation

```bash
# 1. Clone and setup
git clone https://github.com/KestrelSovereignAI/kestrel-sovereign.git
cd kestrel-sovereign
uv sync  # Creates .venv and installs all dependencies

# 2. (Optional) Start Ollama for local models - skip if using cloud APIs
ollama serve
ollama pull llama3.2:3b

# 3. Run the setup wizard. Interactive by default; --quickstart
#    accepts every default non-interactively. --quickstart auto-detects
#    available LLM providers — checks for cloud API keys
#    (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
#    GOOGLE_API_KEY) in your shell env, probes Ollama at
#    localhost:11434, and writes the priority list accordingly. If
#    none are available, it falls back to Ollama-only.
uv run kestrel setup               # interactive
uv run kestrel setup --quickstart  # non-interactive; auto-detects providers
# Or hand-edit: cp kestrel.toml.example kestrel.toml

# 4. Doctor check (verify readiness)
uv run kestrel doctor

# 5. (Optional) Create an additional agent. `--quickstart` already
#    registers one named `Kestrel`; this step is for adding more or
#    if you ran the interactive wizard without auto-registering.
uv run kestrel create MyAgent

# 6. Start an agent. Name the one you want, or omit to start every
#    agent registered with autostart=true (the wizard's default).
#    After --quickstart with no step 5, the autostart agent is "Kestrel".
uv run kestrel start            # starts every agent with autostart=true
uv run kestrel start Kestrel    # if you only ran --quickstart
uv run kestrel start MyAgent    # if you ran step 5 (works regardless of autostart)
```

Installed features may contribute setup steps through the SDK. Their complete
ordering is validated before any step runs, and contributed default steps may
not precede the built-in `keys` custody boundary. Use `kestrel setup --core-only`
to recover built-in configuration without importing provider code. `setup
--check` is always core-only because Python plugins are not sandboxed and
therefore cannot be trusted to honor a read-only check contract.

> **Refreshing later — use the `kestrel` CLI, not raw `uv sync`.** The `uv sync` above is only for this first install. To pull in upstream changes afterward, run **`kestrel update`** (pull → install → reconcile → feature sync → restart). Plain `uv sync` makes the venv match `pyproject.toml` exactly and therefore **prunes out-of-tree feature packages** (`kestrel-feature-*`, voice/provider packages); `kestrel update` restores them in the same pass. See [Pulling in upstream changes](#pulling-in-upstream-changes-kestrel-update).

If you're upgrading from a pre-2026-05 setup that used a standalone `llm_config.toml`, run `uv run kestrel migrate-llm-config` to fold it into `kestrel.toml [llm]`. The legacy file is no longer read.

Local identity exports are plaintext continuity packages. By default they are
published under the active agent's own data directory; standalone operators can
override that root with `KESTREL_DATA_DIR`, and multi-agent operators can set
`identity_export_dir` on an individual agent. Relative per-agent overrides are
resolved below that agent's `data_dir`. The destination is a private `0700`
directory with `0600` files and an atomic no-clobber write. `kestrel doctor`
reports legacy exports with unsafe metadata without reading their contents; run
`uv run kestrel identity harden-exports` to restrict eligible operator-owned
legacy packages. New exports also carry public, signed succession evidence so a
self-certifying `did:pkh`/`did:key` root can be verified on a fresh target
without copying private source custody. Born-hybrid `did:web` roots require a
receiver-owned public-key pin; package-declared or arbitrary network DID
documents are never trusted. See
[Identity export custody and fresh-target trust](docs/architecture/security/IDENTITY_EXPORT_CUSTODY.md).

When the optional Phoenix trace backend is installed, its prompts, model
outputs, tool data, and SQLite state are kept in a private host-data directory:
`$KESTREL_PHOENIX_WORKING_DIR` when explicitly set, otherwise
`$KESTREL_HOME/host-data/phoenix` or `~/.kestrel/host-data/phoenix`. Kestrel
creates directories as `0700`, supervisor files as `0600`, and launches Phoenix
with `umask 077` so DB sidecars are private at creation. An insecure or
ambiguous legacy store disables local tracing before the OTLP endpoint is
auto-wired. See [Phoenix trace custody](docs/architecture/security/PHOENIX_TRACE_CUSTODY.md).

Fleet-wide host-feature state uses the same private host-data root, in
`host-features.db`, rather than writing `kestrel_host.db` into whichever source
checkout launched the host. The directory is `0700`; the database and its
SQLite WAL/SHM/journal family are `0600` from creation. An explicit
`KESTREL_HOST_DB_PATH` remains supported when its parent is a dedicated `0700`
directory. This feature-state database is intentionally separate from the
payment/key vault at `agent_data/host.db`; Kestrel never falls back or copies
between them. See [Host runtime storage custody](docs/architecture/security/HOST_RUNTIME_STORAGE_CUSTODY.md).

Your agent is now running. Two ports to know about, depending on which start form you used:

| Command | Listens on | Why |
|---|---|---|
| `kestrel start` (no name) | `http://localhost:8888` (the multi-agent **host**) | Default in-process multi-agent mode; the host fronts every agent registered with `autostart = true` at `multi_agent.host.port` (default `8888`). Agents registered with `autostart = false` aren't loaded — start those by name. |
| `kestrel start <name>` | the **agent's own port**, printed by the CLI on start | Each agent gets the next free slot at or above `8801` (the first auto-assigned agent lands there; subsequent agents go to `8802`, etc., or whatever `--port` you passed to `kestrel create`). The exact value lives in `multi_agent.toml` under that agent's entry, and `kestrel start <name>` prints `Starting <name> on :<port>...`. |

> **Port conflict?** Edit the agent's entry in `multi_agent.toml` to change its port, or recreate the agent with a chosen port (`kestrel create MyAgent --port 8899`). Edit `multi_agent.toml`'s `[host]` section to change the host port (default `8888`). `kestrel start` itself doesn't take a `--port` flag — runtime ports are read from `multi_agent.toml`.

> **Test it:** Visit the URL the CLI printed on start (`http://localhost:8888` for the multi-agent host, or whatever per-agent port `kestrel start <name>` reported). The Sovereign Console is the default page; append `/health` for the public aggregate JSON readiness probe. Full `/health/detailed` diagnostics require the normal API key, JWT, or OAuth session.

> **Windows users:** the CLI prints emoji. If you see `UnicodeEncodeError: 'charmap' codec can't encode character ...`, run `chcp 65001` once in your PowerShell session to switch the console to UTF-8. (As of v0.1.9 the CLI auto-reconfigures stdout, so a fresh install should not hit this.)

### Install from PyPI (no source clone)

The Quick Start above clones the repo so you have demos, examples, and the `kestrel.toml.example` next to your agent. If you only want to run an agent and don't need the source tree, install from PyPI:

```bash
# 1. Install the CLI (uv tool install is preferred — `kestrel`
#    lands on PATH in an isolated venv. Plain `pip install
#    kestrel-sovereign` works too, inside an activated venv.)
uv tool install kestrel-sovereign

# 2. Pick where Kestrel keeps your data. Either set KESTREL_HOME
#    explicitly (recommended for ops) — or skip this and Kestrel will
#    use ~/.kestrel/ by default.
export KESTREL_HOME="$HOME/kestrel-data"

# 3. Same wizard as above. Writes .env, kestrel.toml, multi_agent.toml,
#    and agent_data/ under $KESTREL_HOME.
kestrel setup --quickstart
kestrel start
```

The default uv compute executor requires the Kestrel process itself to run
inside a Python `venv` or `virtualenv`. This lets it pin an interpreter outside
Kestrel's runtime while `uv run --isolated --no-project` creates a fresh,
project-free script environment, so scripts cannot inherit Kestrel's installed
packages. `uv tool install` and the source checkout's `uv sync` satisfy this
automatically. For a plain pip installation, create and activate a Python
virtual environment first. A system or `--user` install can run Kestrel, but
the uv compute executor deliberately reports unavailable. A Conda environment
alone is also insufficient because it does not provide the distinct
`sys.prefix`/`sys.base_prefix` boundary this executor validates.

**Where data lives.** `kestrel` resolves the project directory in this order: `KESTREL_HOME` → walk up from CWD looking for a `multi_agent.toml` / `kestrel.toml` / `.env` marker → `~/.kestrel/` for pip-installed users with no markers anywhere. A pure pip install with no `KESTREL_HOME` and no project in CWD lands on `~/.kestrel/` and creates it on first run. **Never** writes to `site-packages/` — `pip install --upgrade kestrel-sovereign` is safe and won't touch your agent data.

If you later want to switch your data dir, move it: `mv ~/.kestrel /new/path && export KESTREL_HOME=/new/path`. The agent's database file is portable; nothing about the data dir is hard-coded.

### CLI Commands (Cross-Platform)

All commands work on Windows, macOS, and Linux. Pass the agent directory as an argument:

> **Running `kestrel` directly.** The commands below are written as bare
> `kestrel …`. That works whenever the `kestrel` console script is on your
> `PATH` — i.e. your project venv is activated (`source .venv/bin/activate`,
> or `.venv\Scripts\activate` on Windows) or you installed it globally with
> `uv tool install`. Working from a source clone *without* activating? Prefix
> any command with `uv run` (e.g. `uv run kestrel status`) and uv will resolve
> the project environment for you.

```bash
kestrel doctor                       # Check prerequisites and readiness
kestrel create MyAgent               # Create a new agent
kestrel start MyAgent                # Start an agent
kestrel terminate MyAgent            # Terminate an agent process
kestrel restart MyAgent              # Restart (stop then start)
kestrel update [MyAgent]             # Pull + install + feature sync + restart (see below)
kestrel status                       # Show all running agents
kestrel list                         # List available agents
kestrel shell MyAgent                # CLI chat interface
kestrel config ./agent_data/MyAgent  # Show agent config
```

#### Pulling in upstream changes (`kestrel update`)

The typical refresh-the-checkout loop is four commands:

```bash
git pull                               # new code from origin
uv sync                                # refresh deps from uv.lock (PRUNES kestrel-feature-* packages)
kestrel feature sync                   # restore the feature packages uv sync just pruned
kestrel restart                        # bring agents back so they see the new install
```

`kestrel update` collapses that into one verb:

```bash
kestrel update                       # pull + install + reconcile + feature sync + restart all agents
kestrel update Emma                  # same, but only restart the named agent
kestrel update Emma --startup-timeout 300 # allow extra time for feature-heavy startup
kestrel update --dry-run             # preview the steps (incl. the reconcile plan) without mutating
kestrel update --no-pull --no-install # only reconcile + feature sync + restart (e.g. after a manual checkout)
kestrel update --no-features         # skip BOTH the reconcile and `feature sync` steps
kestrel update --allow-dirty         # pull even with modified TRACKED files (also relaxes editable reconcile pulls)
kestrel update --prefer-source       # reconcile: git-pull every feature with a known checkout
kestrel update --prefer-pypi         # reconcile: pip --upgrade every feature (no editable git pulls)
kestrel update --uv-sync             # force `uv sync` for the install step
kestrel update --no-uv-sync          # force `uv pip install -e .` for the install step
kestrel update --no-deps             # pass --no-deps to `uv pip install` for the fast path
kestrel update --continue-on-error   # proceed past a reconcile/sync error to the restart
```

`kestrel start`, `kestrel restart`, and `kestrel update` share one readiness
deadline while waiting for the server's `/health` endpoint. Use
`--startup-timeout SECONDS` on any of those commands when a deployment needs a
different deadline; each command's `--help` reports the current default. A
timeout does not kill the process—the error names the log to inspect because
initialization may still be in progress.

**Reconcile step — host venv ⇆ agent allowlists.** Between the install and `feature sync` steps, `kestrel update` reconciles the host venv against `union(all [agents.*].features) + the mandatory features`. The per-agent `features` allowlist is a *filter*, not an *installer*: naming a feature class an agent doesn't have installed just makes it silently never load. Reconcile closes that gap — it resolves each required class to its package (live entry-points + in-tree bundled classes first, the static feature registry as fallback), then **installs** any that are missing and **updates** the rest. A class that no installed package, bundled feature, or catalogued package provides is a hard error (no blind fallback) — fix the allowlist or add the package to the registry.

Update mode is per-feature, read from the `.kestrel-host-features.toml` **source map** (each entry records where a package comes from — `editable = "/path"` *or* `pypi = ">=x,<y"`): editable features are `git pull --ff-only`'d in their checkout (that checkout IS the running code); PyPI features are `pip install --upgrade`'d (respecting floor pins). A non-fast-forward or a dirty checkout is reported, never auto-merged. `--prefer-source` / `--prefer-pypi` bulk-override the mode.

**Install step auto-detect.** When `uv.lock` exists at the source checkout root, the install step runs `uv sync --active` against the venv that owns the running `kestrel` binary (detected via `sys.prefix != sys.base_prefix`, so systemd / cron / direct-path invocations work too). That refreshes deps from the lockfile AND prunes packages not in it — which is precisely why `kestrel feature sync` runs immediately after, to restore any out-of-tree feature packages (`kestrel-feature-*`) that `uv sync` just pruned. Without `uv.lock` present, the step falls back to `uv pip install --python sys.executable -e .` which only reinstalls the project package. Force either branch with `--uv-sync` / `--no-uv-sync`.

**Dirty-tree detection.** Only modified or staged **tracked** files block the pull. Untracked files (`kestrel.toml.backup-*` written by `kestrel setup`, scratch files, etc.) don't count — `git pull --ff-only` can't collide with files git doesn't know about. The refusal message surfaces the porcelain summary so you can see exactly what to commit/stash.

**Safety:**

- `git pull --ff-only` so a non-fast-forward upstream aborts instead of producing a surprise merge.
- Refuses to pull when the working tree has modified TRACKED files unless `--allow-dirty` is passed.
- Any step's failure short-circuits the rest, so a half-applied update never reaches the restart phase.
- Source checkout and runtime data root (`KESTREL_HOME`) are resolved separately. The pull/install run against the source checkout (discovered by introspecting `kestrel_sovereign.__file__` — must have both `pyproject.toml` and `.git`). The runtime data root is used only for `feature sync`'s manifest lookup.

**Pip-installed users** (no editable source checkout — `pip install kestrel-sovereign` against PyPI) — both pull AND install are silently skipped; `feature sync` and `restart` still run. Upgrade the package itself with `pip install --upgrade kestrel-sovereign` first, then run `kestrel update` to pick up new features + restart.

**Bootstrap when your installed `kestrel update` is older than the fix you want.** If your CLI predates a `kestrel update` change you're trying to pull in, the old version may refuse on a dirty tree (early behaviour treated untracked files as dirty) or run the wrong install command. One-time bootstrap:

```bash
git pull && kestrel update          # explicit pull first, then the new code runs
# or
kestrel update --allow-dirty         # bypass the old refusal a single time
```

After the new code lands in your checkout, `kestrel update` alone handles subsequent refreshes.

#### Scheduler protocol-v2 cutover is not an ordinary update

If this database has ever been served by a pre-protocol-v2 scheduler, do **not**
run a normal rolling `kestrel update` or restart an old image after a failed
first v2 start. The first v2 process deliberately fences legacy-visible
schedules and waits for a drain acknowledgement; an old binary against that
fenced database can leave work dormant or duplicate an occurrence. This applies
to both local SQLite and shared PostgreSQL deployments.

Stop every legacy poller and schedule writer, take a backend-appropriate backup,
perform the controlled v2 acknowledgement, and use forward-only recovery unless
you deliberately restore the complete pre-cutover database while all writers
remain stopped. The exact local, systemd, PyPI, SQLite, and PostgreSQL commands
are in the [scheduler protocol-v2 deployment runbook](docs/deployment/README.md#scheduler-protocol-v2-upgrade-sqlite-and-postgresql).

Shared PostgreSQL hosts pre-seed database-global scheduler provenance and every
resolvable configured DID before parallel agent runners start. That lets a new
DID join a known fresh-v2 fleet safely, but an absent/unknown provenance marker
is always treated as a legacy migration; do not seed or unfence scheduler rows
manually.

#### SDK 0.37 release cascade

Core requires `kestrel-sovereign-sdk[tracing]>=0.37.1,<0.38`; the
`observability` extra carries the same SDK line with `metrics`. This is a
runtime contract for durable isolated execution, provider-neutral private
inference leases (including bounded owner-scoped idle renewal), and private
host ingress, plus feature-owned operator and context-clause contribution
contracts. It is not a preference that a downstream package may relax. The
Core-owned release-cascade contract is:

| Downstream release gate | Required published SDK constraint before Core ships | Core assertion |
|---|---|---|
| Frinz | `kestrel-sovereign-sdk>=0.37.1,<0.38` | External prerequisite; Core does not claim Frinz has changed. |
| Observability fleet | `kestrel-sovereign-sdk>=0.37.1,<0.38` | External prerequisite; Core does not claim observability has changed. |

Verify the published Frinz and observability constraints and tests before the
Core publish. Do not weaken Core's requirement to make an older sibling
resolver succeed.

The Core dependency-contract test verifies the base and observability
declarations and lockfile resolve the same SDK line. It cannot validate another
repository's working tree, so the downstream release verification remains an
explicit release gate owned by those repositories.

#### Telegram acknowledged-ingress release order

Core 0.52.1 adds the host-authenticated initialize capability
`channel-inbound-acknowledgement-v1` for the Telegram polling bridge. Telegram
0.1.3 requires that capability before it will start polling, preventing an old
Core from mis-parsing its durable ACK envelope and leaving a child poller
blocked. Publish Core 0.52.1 first, then publish Telegram 0.1.3; do not relax
the Telegram requirement to make the reverse order installable.

Windows base installs also include `tzdata`, so default `UTC` and named-IANA
scheduler time zones work without optional Pandas or Phoenix extras.

### Feature management (`kestrel feature`)

Kestrel ships a self-contained core with the local storage, identity, governance, and built-in LLM-routing dependencies needed to run an agent. Optional feature packages register `Feature` classes through the `kestrel_sovereign.features` entry-point group. Provider packages, such as cloud, voice, and storage backends, register with provider-specific entry-point groups and are consumed by their owning core or feature module.

```bash
kestrel feature list                   # Show installed + available features
kestrel feature info <name>            # Detailed info about a feature
kestrel feature install <name>         # Install a feature package
kestrel feature upgrade [<name>...]    # Upgrade installed feature/provider packages
kestrel feature sync                   # Restore the host's declared feature set (see below)
kestrel feature enable <name>          # Enable an installed feature
kestrel feature disable <name>         # Disable without uninstalling
kestrel feature scaffold <name>        # Generate a new feature package skeleton
```

The canonical inventory and package-boundary contract live in
[`KESTREL_FEATURES.md`](KESTREL_FEATURES.md). The runtime registry records the
same taxonomy in `boundary`: `bundled`, `bundled-component`,
`feature-package`, `provider-package`, or `standalone-tool`. Its `package` is
always the owning distribution/install target for that row; `core` remains a
compatibility flag and is true only for the two bundled categories. Catalog
status `available` means “known but not detected in this environment,” not that
an external distribution was verified reachable.

#### Keeping features installed across `uv sync` (`kestrel feature sync`)

Feature and provider packages are deliberately **not** declared in `pyproject.toml` (that would invert the open-core dependency direction), so `uv sync` — which makes the venv match declared dependencies exactly — **prunes** them. After a sync or any environment rebuild, an installed feature can silently vanish (e.g. voice endpoints start returning 404). `kestrel feature upgrade` cannot bring it back, because a pruned package leaves no entry-point to discover.

`kestrel feature sync` is the restore counterpart. It reads a host-local manifest, `.kestrel-host-features.toml` (gitignored; template at [`.kestrel-host-features.toml.example`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/.kestrel-host-features.toml.example)), and reinstalls whatever is missing:

```bash
kestrel feature sync --capture         # Snapshot currently-installed extensions into the manifest
kestrel feature sync                    # Reinstall the manifest's packages that are missing
kestrel feature sync --dry-run          # Preview without installing
kestrel feature sync --manifest PATH    # Use a non-default manifest location
```

A manifest entry names a package (by registry name or raw dist name) and records **one** source — an `editable` checkout *or* a `pypi` version spec (mutually exclusive) — plus optional extras. This is the **source map** the `kestrel update` reconcile step reads to decide each feature's update mode (git pull vs `pip --upgrade`):

```toml
[[feature]]
name = "voice"
editable = "/path/to/kestrel-feature-voice" # dev checkout: git pull --ff-only on update
extras = ["local"]                          # also pull the on-device pipeline (Piper / faster-whisper)

[[feature]]
name = "github"
pypi = ">=0.2,<0.3"                          # from PyPI: pip install --upgrade within the pin

[[feature]]
name = "reflection"                         # neither form = "install latest from PyPI" (back-compat)
```

This never touches `pyproject.toml`, so installing `kestrel-sovereign` itself still pulls zero feature packages. The typical recovery after a prune is `kestrel feature sync` followed by a host/agent restart so the restored entry-points load.

### Per-Agent Configuration

Each agent can have a `kestrel.toml` config file in its directory:

```toml
# agent_data/myagent/kestrel.toml
[agent]
name = "MyAgent"
port = 8888
host = "0.0.0.0"
log_level = "INFO"
```

Create or edit config:
```bash
kestrel config ./agent_data/myagent --init           # Create config
kestrel config ./agent_data/myagent --set-port 8899  # Change port
kestrel config ./agent_data/myagent --set-name MyAgent  # Change name
```

### Running Multiple Agents

Each agent runs on its own port. Create configs for each:

```bash
# Agent 1: Alpha on port 8888
kestrel create Alpha --port 8888
kestrel start Alpha

# Agent 2: Helper on port 8889
kestrel create Helper --port 8889
kestrel start Helper

# Check status of all agents
kestrel status
```

### Alternative: Direct Commands

```bash
# Canonical direct-server command (set KESTREL_DB_PATH first). Explicit
# loopback binding avoids exposing a local development agent to the LAN:
KESTREL_DB_PATH=./agent_data/myagent uv run python -m kestrel_sovereign.server \
  --host 127.0.0.1 --port 8888

# CLI chat (no server needed)
python -m kestrel_sovereign.main ./agent_data/myagent

# Source-clone shorthand for CLI chat remains available:
#   python main.py ./agent_data/myagent
```

> **Note:** `KESTREL_DB_PATH` is a **directory** path, not a file path. The database file `kestrel_prime.db` is created inside the specified directory. For example, setting `KESTREL_DB_PATH=./agent_data/myagent` stores the database at `./agent_data/myagent/kestrel_prime.db`.

The direct server rejects unknown options instead of silently ignoring bind
intent. `--host` overrides `KESTREL_SERVER_HOST`, while `--port` overrides the
platform-standard `PORT`; the compatibility defaults remain `0.0.0.0:8888`.
For normal fleet operation, `kestrel start` reads its bind and port from
`multi_agent.toml` and does not accept these direct-server flags. See the
[server launch contract](docs/architecture/core/SERVER_LAUNCH_CONTRACT.md) for
the complete precedence and managed-container behavior.

> **Keeping the host awake (long sessions / 24/7).** Kestrel does not manage
> host power state itself — a long-running agent that must survive an idle
> machine relies on the OS staying awake. For a dedicated deployment, prefer a
> service supervisor (launchd/systemd) or an always-on host (Mac mini, VPS,
> cloud container). For a foreground run on a laptop that would otherwise
> sleep, wrap the command in the platform's keep-awake utility:
>
> - **macOS:** `caffeinate -s <command>` (e.g. `caffeinate -s kestrel start MyAgent`). `-s` prevents idle sleep while on AC power.
> - **Linux (systemd):** `systemd-inhibit --what=sleep:idle --why="kestrel agent" <command>` holds a sleep inhibitor for the command's lifetime.
> - **Windows:** there is no built-in command wrapper. Either disable sleep under **Settings → System → Power**, or hold an execution-state assertion for the session, e.g. in PowerShell:
>   ```powershell
>   $sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint e);'
>   $t = Add-Type -MemberDefinition $sig -Name Power -Namespace Win32 -PassThru
>   $t::SetThreadExecutionState(0x80000003)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
>   kestrel start MyAgent
>   ```

<a id="web-ui-sovereign-console"></a>
## 🖥️ Web UI (Sovereign Console)

Kestrel includes a built-in web interface called the **Sovereign Console**. Once your agent is running, open the URL the CLI printed on start — `http://localhost:8888` for the multi-agent host (default `kestrel start` mode), or the per-agent port for a single-agent start — in any browser; no additional software required.

The built-in console exposes the following panels; installed features may add
their own contributions:

| Tab | Description |
|-----|-------------|
| **Chat** | Converse with the agent (supports model selection, privacy modes, chat history) |
| **Identity** | View the agent's DID, name, and cryptographic identity |
| **Constitution** | View and audit the agent's constitutional principles |
| **Memories** | Browse the agent's knowledge graph and stored memories |
| **Tasks** | Monitor background tasks and activity |
| **Sovereignty** | Manage data sovereignty, backups, and exports |
| **Resources** | View agent resource usage and configuration |
| **Metrics** | Inspect the metrics dashboard when the optional Observability feature is installed |
| **Features** | Browse installed, available, enabled, and disabled features |
| **Security** | Manage permissions, audit logs, and session security |
| **Approvals** | Review pending and recent governed invocations |

> **Alternative clients:** The server also exposes an OpenAI-compatible API at `/v1/chat/completions`, so you can connect any OpenAI-compatible client (e.g., [Open WebUI](https://github.com/open-webui/open-webui)) if you prefer.

## 🏗️ Architecture Overview

Kestrel agents are built on several key components:

- **Cryptographic identity**: each agent has a DID and signed identity lifecycle.
- **Privacy-aware storage**: local SQL-backed memory, FTS, knowledge graphs, RAG, and provider-backed embeddings where the selected route supports them.
- **Multi-provider LLM routing**: vendor/route/model configuration for built-in Anthropic/Claude Max, OpenAI/Codex, Gemini/Vertex AI, Ollama, and OpenRouter adapters, with additional providers discoverable through entry points.
- **Constitutional governance**: immutable articles with governed interpretation and amendment paths.
- **Local audit anchors**: optional persistent, tamper-evident integrity anchors and verification. Kestrel does not implement external blockchain anchoring.

## 📁 Project Structure

```
kestrel-sovereign/
├── kestrel_sovereign/         # Core sovereign package
│   ├── cli.py                 # `kestrel` CLI entry point (canonical)
│   ├── kestrel_agent.py       # Core agent class
│   ├── inception_service.py   # Agent creation (DID + genesis audit)
│   ├── agent_config.py        # Per-agent config loader
│   ├── data/feature_registry.toml  # Runtime feature registry
│   ├── features/              # Core bundled features
│   ├── storage/               # SQL-backed storage facade and stores
│   ├── static/                # Sovereign Console frontend
│   └── ...
├── server.py                  # Re-export shim → kestrel_sovereign/server.py (consolidated host + agent app)
├── main.py                    # Re-export shim → kestrel_sovereign/main.py
├── docs/                      # Architecture & guides
└── tests/                     # Test suite

# The Kestrel SDK lives in its own repo + PyPI package:
#   kestrel-sovereign-sdk  (https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk)
# Feature authors import its types directly:
#   from kestrel_sdk.features.base import Feature, tool
#   from kestrel_sdk.tools.base   import ToolCategory, ToolResult
#   from kestrel_sdk.hooks.base   import Hook, HookEvent
```

## 🎯 Core Features

### 1. Sovereign Memory
- **Privacy-gated Storage**: SQL-backed stores with full-text search and knowledge graphs when the active privacy mode permits persistence
- **RAG Pipeline**: Document chunking and semantic retrieval through SQLAlchemy-backed vector search where available; embeddings are requested from the active LLM provider route when it advertises embedding support, otherwise search falls back to BM25/LIKE paths
- **Conversation History**: retained interaction metadata and history outside `EPHEMERAL` mode
- **Human-Led Interactions**: user-directed memory and retrieval controls rather than an unconditional promise to retain every turn

### 2. Multi-Model Intelligence
- **Local or cloud routes**: Ollama can keep inference local; configured cloud routes are available when an operator opts in
- **Vendor/route/model selection**: routing is configured per provider and can use ordered fallbacks
- **Extensible adapters**: the in-tree adapters and entry-point providers expose their capabilities through the same registry

### 3. Cryptographic Identity
- **DID Generation**: self-certifying `did:pkh` roots plus supported `did:web` deployment paths
- **Verification roots**: exported `did:key` identities are supported as self-certifying verification roots; inception does not create new `did:key` agents
- **Fresh-target verification**: exported self-certifying identities carry public succession evidence; a born-hybrid `did:web` root requires a receiver-owned public-key pin
- **Bounded trust**: arbitrary network DID documents and package-declared keys are not implicitly trusted; third-party verification remains an explicit integration boundary

### 4. Constitutional Governance
- **Immutable Articles**: Core principles that cannot be changed
- **Interpretive Canons**: Flexible guidelines for decision-making
- **Amendment Process**: Cryptographically-signed governance updates

### 5. Data Sovereignty & Privacy Modes
- **Ephemeral Mode**: True off-the-record conversations (nothing stored)
- **Privacy Granularity**: selectable privacy modes for different use cases
- **Portable data**: operator-controlled local data and identity exports; decentralized-storage integrations are optional surfaces
- **Optional Economics**: Wallet and autonomous payment flows are installable feature-package surfaces, not part of the base install

## ⚠️ Feature maturity

Kestrel is pre-1.0 and its public interfaces may change. The maintained source
of truth for the in-tree feature inventory, providers, commands, and endpoints
is [`KESTREL_FEATURES.md`](KESTREL_FEATURES.md). Use `kestrel feature list` or
the runtime installed-features API, plus the
[testing guide](docs/architecture/testing/TESTING_GUIDE.md), when evaluating a
specific deployment. The categories below are qualitative so they do not turn
into stale release metrics.

### Maintained core

- **Constitutional governance, privacy modes, identity lifecycle, local storage, and the LLM service** are bundled and exercised as the framework's primary runtime path.
- **Identity verification** supports fresh-target trust for exported self-certifying roots and pin-gated born-hybrid `did:web` roots. It deliberately does not treat arbitrary third-party DID documents as trusted.
- **Memory and retrieval** support FTS, knowledge graphs, RAG, vector search, and keyword fallback, subject to the selected privacy mode and provider capabilities.
- **Local audit anchors** make supported audit data tamper-evident. Per-response auditing is optional and must be enabled by the operator.
- **Cloud Run** is the maintained serverless deployment path; local and self-hosted operation remain first-class.

### Optional or actively evolving surfaces

- **Voice, wallets/economics, observability, channel transports, and specialized provider backends** are feature or provider packages, not guarantees of the base install. The generic `ChannelFeature` coordination surface itself is bundled.
- **RunPod, Vast.ai, and GCP Compute** are provider integrations whose live behavior depends on credentials and the selected package.
- **Training/LoRA** supplies core protocol and local-adapter seams; cloud adapters are evolving separately.
- **GitHub code introspection** supports its shipped retrieval and issue-tool surface. The deeper static-analysis design in [`docs/architecture/GITHUB_FEATURE_DESIGN.md`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/architecture/GITHUB_FEATURE_DESIGN.md) is not an implementation claim.

### Known boundaries

- **APIs and configuration** can change before a 1.0 release.
- **Turnkey WhatsApp, Telegram, Discord, and Slack adapters; browser automation; and visual workspaces** are not bundled in this framework.
- **Encryption coverage has explicit limits**; see [Encryption at Rest](#-encryption-at-rest) rather than assuming whole-database encryption.

**Bottom line:** Kestrel is suited to developers and supervised operators building privacy-first, portable agents. Treat a deployment as an engineering system: choose its privacy mode and providers deliberately, verify its enabled features, and do not rely on it as an unmanaged consumer platform.

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [Documentation Index](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/README.md)

- [Agent Ecosystem](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/architecture/core/AGENT_ECOSYSTEM.md)
- [Agent Economics](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/architecture/economics/AGENT_ECONOMICS.md)
- [Kestrel Constitution](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/principles/KESTREL_CONSTITUTION.md)
- [Cryptographic Anchoring](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/architecture/security/CRYPTOGRAPHIC_ANCHORING.md)
- [Decentralized Storage](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/architecture/storage/DECENTRALIZED_STORAGE.md)
- [Multi-Model Support](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/architecture/core/MULTI_MODEL_SUPPORT.md)
- [Privacy Modes](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/architecture/security/PRIVACY_MODES.md)
- [LLM Service Architecture](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/architecture/LLM_SERVICE_ARCHITECTURE.md)

## 💡 Example Applications

Kestrel is a foundation for AI agents that need to outlive any single vendor, deployment, or owner. Concrete deployments and good-fit use cases:

- **Healthcare RPM agents** — Constitutional governance over an LLM, persistent patient-owned memory, audit trail for every clinically-relevant action.
- **Long-running personal research agents** — Memory accumulates across months without dependency on a single provider's chat history.
- **Custodial agents for sensitive document workflows** — Privacy-mode tiers (EPHEMERAL → PUBLIC) let one agent handle both an off-the-record consult and a fully-anchored long-term contract.
- **Multi-agent A2A networks** — JSON-RPC 2.0 agent-to-agent protocol lets sovereign agents collaborate without surrendering their identity to a central broker.

## 🧪 Testing

Run the test suite from the activated virtual environment:
```bash
# Run a single test with uv and -x
uv run pytest -x tests/test_inception.py::test_successful_inception
```

### Clean Install Verification

Kestrel supports multiple installation configurations. Use the verification script to test that clean installs work correctly across all supported scenarios:

```bash
# Run all 5 install scenarios (creates isolated venvs)
uv run kestrel verify-install

# Run specific tests only
uv run kestrel verify-install 1 3    # SDK-only and wallet package
```

The install matrix covers:

| Test | Scenario | Verifies |
|------|----------|----------|
| 1 | **SDK only** | `from kestrel_sdk.features.base import Feature` |
| 2 | **Core sovereign** | `from kestrel_sovereign.features.base import Feature` compatibility + `/health` |
| 3 | **Feature package** | `from kestrel_feature_wallet import WalletFeature` |
| 4 | **SDK + feature dev mode** | Feature packages can develop against SDK alone |
| 5 | **Full stack** | Sovereign + wallet + intelligence, entry_point discovery |

Integration tests for the same import paths run as part of the normal test suite:

```bash
uv run pytest tests/integration/test_clean_install_verification.py -v
```

## 🔧 Configuration

### LLM Configuration (`kestrel.toml` `[llm]`)

LLM config lives under the `[llm]` section of `kestrel.toml`. The setup wizard (`kestrel setup llm`) will write it for you; you can also hand-edit `kestrel.toml` after copying from `kestrel.toml.example`.

Kestrel uses a **vendor/route/model** schema. A *vendor* is who makes the weights; a *route* is how to reach them (adapter + base URL + auth). API keys belong in `.env` and are referenced by `api_key_env`. See [`kestrel.toml.example`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/kestrel.toml.example) and [`docs/architecture/LLM_SERVICE_ARCHITECTURE.md`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/architecture/LLM_SERVICE_ARCHITECTURE.md) for the canonical spec.

```toml
[llm]
route_priority = ["openai:api", "ollama:local"]

[llm.vendors.openai]
is_cloud = true

[llm.vendors.openai.routes.api]
adapter        = "OpenAIAdapter"
api_key_env    = "OPENAI_API_KEY"
model          = "auto"
selection_hints = ["gpt-5", "mini"]

[llm.vendors.ollama]
is_cloud = false

[llm.vendors.ollama.routes.local]
adapter        = "OllamaAdapter"
host           = "http://localhost:11434"
model          = "auto"
selection_hints = ["llama3.2", "qwen"]
```

> Pre-2026-05 setups used a standalone `llm_config.toml` at the repo root. That path was removed (epic #938). Run `kestrel migrate-llm-config` to fold a legacy file into `kestrel.toml [llm]`; the source is renamed to `.bak`, your prior `kestrel.toml` is timestamp-backed-up, and the operation is idempotent.

### Environment Variables

See `.env.example` for a complete list. Key variables:

**LLM Providers:**
- `OPENROUTER_API_KEY`: OpenRouter API key (recommended - access to multiple providers)
- `OPENAI_API_KEY`: OpenAI API key for cloud models
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude models
- `KESTREL_SKIP_REACHABILITY_PROBE=1`: Skip startup reachability probes for local LLM daemons in CI/test contexts

**Storage:**
- `KESTREL_DB_PATH`: Directory where the agent database is stored (default: `./agent_data`). This is a **directory** path -- the database file `kestrel_prime.db` is created inside it.
- `KESTREL_DATA_KEY`: Fernet encryption key for data at rest

**GitHub Integration:**
- `GITHUB_TOKEN`: Personal access token for GitHub features
- `GITHUB_SELF_REPO`: Agent's source repository (default: `KestrelSovereignAI/kestrel-sovereign`)

## 🚢 Deployment

Kestrel supports multiple deployment targets. See [KESTREL_FEATURES.md](KESTREL_FEATURES.md) for the full catalog.

### Cloud Run (Serverless)

Scales to zero when idle ($0/month), auto-scales under load. Each sovereign agent gets its own service.

```bash
# One-time: set up GCP secrets from .env
uv run kestrel deploy secrets sync

# Build and push to GCR (both single-agent and multi_agent images)
uv run kestrel deploy build

# Deploy to dev (multi-agent host, always warm) or prod (single-agent, auto-scales)
uv run kestrel deploy dev
uv run kestrel deploy prod
```

Profiles live in [`deploy_config.toml`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/deploy_config.toml). See [`docs/deployment/README.md`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/deployment/README.md) for the full runbook (status, logs, teardown, health).

Auto-deploys on version tags via [GitHub Actions](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/.github/workflows/deploy.yml).

### Docker (Local)

```bash
# Remote LLM — smallest image (~500MB)
docker build -f docker/Dockerfile.remote -t kestrel .
docker run -p 8888:8888 -e OPENAI_API_KEY=... kestrel

# Standalone with Ollama (no API keys needed)
docker build -f docker/Dockerfile.standalone -t kestrel-standalone .
docker run -p 8888:8888 kestrel-standalone

# GPU with CUDA
docker build -f docker/Dockerfile.gpu -t kestrel-gpu .
docker run --gpus all -p 8888:8888 kestrel-gpu
```

## 🔐 Backups and Storage Tiers

Backups can be created interactively from the agent using privacy-gated storage tiers:

- local-only: cache the backup tar.gz locally only
- delegated-decentralized: encrypt + gzip and store through Lighthouse/IPFS/Filecoin when configured
- expedient-cloud: store sync snapshots in GCS when configured
- sovereign-operated: currently none/decommissioned; requires a reachable reprovisioned `SOVEREIGN_IPFS_URL`

Privacy gating:

- EPHEMERAL: backups disabled
- ISOLATED: cache-only; use `!promote-backup` to save the isolated session and back up
- ANONYMOUS: backups allowed; encryption forced for filecoin tier
- NORMAL: backups allowed; encryption configurable (default on)

Usage from the REPL:

```text
!backup tier=local
!backup tier=ipfs
!backup tier=filecoin
!promote-backup tier=filecoin
```

Each backup produces a `backup_artifact` node in the graph linked to the agent with properties like `content_hash`, `ipfs_cid`, `filecoin_deal_id`, `encrypted`, and timestamp.

## 🔒 Encryption at Rest

`KESTREL_DATA_KEY` (a Fernet key or passphrase) is the master key for Kestrel's **application-layer** encryption at rest. It is **not** whole-database encryption. With it set, Kestrel transparently encrypts conversation-history rows, stored file blobs, agent identity private keys (`SecureKeyStorage`, AES-256-GCM), and agent-resource bodies such as the SOUL (`AgentResourceStore`, Fernet). It does **not** cover saved-item (`saved_items.content`) or RAG document-chunk (`document_chunks.content`) bodies — those are plaintext columns today, with application-layer encryption tracked in [#2677](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2677). Set it before first run:

The accepted, not-yet-implemented design for closing those two gaps is
[Saved-item and RAG content encryption](docs/architecture/security/MEMORY_CONTENT_ENCRYPTION.md).
That ADR is a rollout contract, not a claim that the current columns are
encrypted.

```bash
export KESTREL_DATA_KEY=$(python - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
)
```

Encryption is transparent on read/write. Sovereignty exports are encrypted by default on **every** storage tier — local included — deriving their key from the same `KESTREL_DATA_KEY` master key; there is no separate backup key. An export requested with `encrypt=True` is refused outright when `KESTREL_DATA_KEY` is unset, never silently downgraded to plaintext; pass `encrypt=False` to deliberately accept an unencrypted backup. For production, set `KESTREL_DATA_KEY` from an env/KMS secret rather than the dev placeholder.

What is actually encrypted at rest today:

| Data at rest | Encrypted at rest today? | Mechanism |
|---|---|---|
| Conversation-history rows | Yes, when `KESTREL_DATA_KEY` is set | App-layer Fernet/AEAD |
| Stored file blobs | Yes, when `KESTREL_DATA_KEY` is set | App-layer Fernet/AEAD |
| Agent identity private keys | Yes, when `KESTREL_DATA_KEY` is set | `SecureKeyStorage`, AES-256-GCM (key derived from `KESTREL_DATA_KEY`) |
| Agent-resource bodies (e.g. SOUL) | Yes, when `KESTREL_DATA_KEY` is set | `AgentResourceStore`, Fernet (agent-scoped, derived from `KESTREL_DATA_KEY`) |
| Saved-item bodies (`saved_items.content`) | **No** — plaintext column | Application-layer encryption planned — [#2677](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2677) |
| RAG document-chunk bodies (`document_chunks.content`) | **No** — plaintext column | Same gap — [#2677](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2677) |
| Embeddings (vectors) | No | Numeric vectors, not reversible ciphertext — their presence is not encryption |
| Remote backups (IPFS/Filecoin export) | Yes, by default | Encrypted from the `KESTREL_DATA_KEY` master key; export fails if the key is unset |
| Local export (`storage_tier="local"`) | Yes, by default | Same portable per-content key as the remote tiers; `encrypt=False` opts out, and `encrypt=True` with no key is refused rather than downgraded |
| Identity export packages | N/A — not storage-at-rest encryption | Signed, permissioned continuity packages — see [Identity export custody](docs/architecture/security/IDENTITY_EXPORT_CUSTODY.md) |
| Whole database file (SQLCipher) | **No** — not wired in this release | `KESTREL_DB_KEY` is not read by the runtime today; the SQLite backend opens plain (`aiosqlite`) databases |

Directory and file permissions (`0700` / `0600`) are access control, not encryption; they do not substitute for the mechanisms above.

### Whole-database encryption (SQLCipher) — not yet wired

Whole-database (SQLCipher) encryption is **not implemented in this release**. The runtime opens the SQLite database with plain `aiosqlite` and does not read `KESTREL_DB_KEY`, so setting that variable does not encrypt the database today. Until full-DB or application-layer encryption lands, the saved-item and RAG document-chunk bodies noted above remain plaintext at rest ([#2677](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2677)). Do not rely on `KESTREL_DB_KEY` for at-rest protection.

## 🧩 OpenAI-Compatible API

The server exposes OpenAI-compatible endpoints for use with third-party clients:

- `GET /v1/models`
- `POST /v1/chat/completions`

For most users, the built-in **Sovereign Console** at the printed URL (`http://localhost:8888` for the multi-agent host, or the agent's own port for a single-agent start) is the easiest way to interact with your agent (see the [Web UI section](#web-ui-sovereign-console) above). If you prefer an external client, point any OpenAI-compatible tool (e.g., [Open WebUI](https://github.com/open-webui/open-webui)) at your server's `/v1/chat/completions` endpoint. Use the model name from `/v1/models`.

> **Auth:** every request to `/v1/...` requires the `X-API-Key` header (or `Authorization: Bearer <key>`). The key was written to `.env` as `KESTREL_API_KEY` by `kestrel setup` (or `--quickstart`). Most OpenAI-compatible clients let you set the key via their `OPENAI_API_KEY` env var or settings UI; point that at your `KESTREL_API_KEY` value.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `python -m pytest -x`
5. Submit a pull request

## 📄 License

Apache 2.0 — see [LICENSE](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/LICENSE) for details.

## 🆘 Support

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Documentation**: See [`docs/`](docs/README.md) and [`KESTREL_FEATURES.md`](KESTREL_FEATURES.md)

---

*Kestrel: Where AI meets sovereignty.* 

## 📚 Key Files Reference

| File | Purpose |
|------|---------|
| `kestrel_sovereign/cli.py` | Canonical `kestrel` CLI entry point |
| `kestrel_sovereign/server.py` | FastAPI agent server + consolidated multi-agent host (root `server.py` is a re-export shim for source clones) |
| `kestrel_sovereign/main.py` | Direct interactive REPL (root `main.py` is a re-export shim for source clones) |
| `kestrel.toml` | Unified config (LLM, agents, features). `[llm]` holds provider config. |
| `KESTREL_FEATURES.md` | Canonical feature inventory |
| `kestrel_sovereign/kestrel_agent.py` | Core agent logic |
| `kestrel_sovereign/agent_config.py` | Per-agent config loader |
| `kestrel_sovereign/inception_service.py` | New agent creation (DID + genesis audit) |
| `kestrel_sovereign/data/feature_registry.toml` | Runtime feature registry |
| `agent_data/<name>/kestrel.toml` | Per-agent configuration |
| `agent_data/<name>/kestrel_prime.db` | Agent database |
| `docs/**/*.md` | Detailed documentation |

## Architecture

### Storage System

The Kestrel storage system is designed to be modular and extensible. It is composed of several specialized components, orchestrated by a high-level facade.

*   **`storage.Database` / `AsyncDatabase`**: Manages the low-level SQL-backed connection and schema. SQLite remains the local default; PostgreSQL is supported through the async backend layer.
*   **`storage.FileStore`**: Handles the storage and retrieval of files.
*   **`storage.GraphStore`**: Manages the knowledge graph (nodes and edges).
*   **`storage.RAGStore`**: Responsible for document chunking and semantic search for the RAG pipeline and "case law" system. It uses SQLAlchemy/vector backends when available, requests embeddings through the active LLM provider route when supported, and falls back gracefully when embeddings are unavailable.
*   **`storage.ConversationStore`**: Manages the agent's conversation history.

The main `Storage` class in `storage/__init__.py` acts as a facade, providing a single, unified interface to these components.

### Genesis Self-Audit

Every incepted agent carries a structured `genesis_audit` receipt bound to the
SHA-256 hash of the same governing constitution bytes resolved and anchored by
the production constitution resolver.

- **Configured `kestrel setup` / `kestrel create`:** when an audit-capable LLM route
  is currently usable, inception runs the audit before returning success. A
  risk-level-3 result, malformed result, or attempted-auditor failure aborts
  creation and removes the new database and identity/key artifacts. A pass is
  stored both on the agent node and as a `genesis_audit` conversation event.
- **LLM-less or programmatic creation:** when no auditor is supplied, inception
  records an explicit `pending` receipt; it never invents a pass. Both streaming
  and non-streaming chat paths hard-gate the first cognition request, complete
  the audit once a configured auditor is available, and admit the turn only
  after the durable `passed` receipt exists.
- **Concurrency and restart:** simultaneous first turns share one serialized
  audit. A completed `passed` or `failed` receipt survives restart, is bound to
  its constitution hash, and is never silently rerun or overwritten. A signed
  constitution reanchor archives that receipt as history and creates a new
  hash-bound `pending` receipt before cognition can resume.
- **Tests and demos:** deterministic auditors may be injected, but their
  provenance is recorded in the same production receipt and event; there is no
  lifecycle bypass.

Diagnostic/recovery commands remain available while an audit is pending. Normal
cognition returns a clear pending/failed response without sending the user's
turn to an LLM.

## 🔄 Next Steps

After getting started:

1.  **Explore Features**: Read [`KESTREL_FEATURES.md`](KESTREL_FEATURES.md) and [`docs/guides/BUILDING_FEATURES.md`](docs/guides/BUILDING_FEATURES.md)
