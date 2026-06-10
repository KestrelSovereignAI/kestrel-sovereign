# Kestrel: Sovereign AI Agent Framework

> Build AI agents that nobody can take away from their users — not you, not the cloud, not the next pivot.

Kestrel is a continuously-developing framework for creating autonomous AI agents with cryptographic identity, persistent memory, and constitutional governance. Every agent you deploy is **owned by its user**, governed by **immutable principles**, and able to **remember across every conversation**. The core install is stable enough to run real agents today; the surrounding ecosystem (cloud providers, training adapters, integrations) is actively evolving — see [Feature Stability](#-feature-stability-v018-beta) for the current per-feature picture.

### Three Pillars

| Pillar | What it means |
|--------|--------------|
| **Portable DID identity** | Cryptographic identity the agent's user owns. Exportable, self-hostable, cloud-optional — the agent is not bound to any provider. |
| **Persistent memory you own** | Local-first memory with full-text search, knowledge graph retrieval, and RAG. Conversations, documents, relationships — searchable, portable, and encrypted at rest when configured. SQLAlchemy-backed vector storage is in tree for saved items, document chunks, and conversation history; embedding generation is still being standardized across LLM providers. |
| **Constitutional governance** | Every agent runs under an audited set of principles enforced *above* the LLM. Genesis audit on creation. Amendment requires cryptographic signature. |

### What's in core, what's an add-on

`pip install kestrel-sovereign` gives you a complete, working sovereign agent: identity, memory, constitution, privacy modes, multi-LLM support, local guarded compute, and a Cloud Run deployment path. Everything you need to run an agent locally with zero cloud commitment.

Voice, MCP, GitHub App, wallet, council, observability, and similar specialized capabilities are **installable feature packages**. RunPod, Vast.ai, GCP Compute, voice cloud backends, and storage backends are **provider packages** that register with provider-specific entry points rather than the feature entry-point group. This split is being completed across [#462](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/462) and [#560](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/560); current state is documented in [`KESTREL_FEATURES.md`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/KESTREL_FEATURES.md).

## 🚀 Quick Start

### Prerequisites
- Python 3.11-3.14
- [uv](https://docs.astral.sh/uv/) (for package management)
- [Ollama](https://ollama.ai) (optional - for local LLM inference without API keys)

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

If you're upgrading from a pre-2026-05 setup that used a standalone `llm_config.toml`, run `uv run kestrel migrate-llm-config` to fold it into `kestrel.toml [llm]`. The legacy file is no longer read.

Your agent is now running. Two ports to know about, depending on which start form you used:

| Command | Listens on | Why |
|---|---|---|
| `kestrel start` (no name) | `http://localhost:8888` (the multi-agent **host**) | Default in-process multi-agent mode; the host fronts every agent registered with `autostart = true` at `multi_agent.host.port` (default `8888`). Agents registered with `autostart = false` aren't loaded — start those by name. |
| `kestrel start <name>` | the **agent's own port**, printed by the CLI on start | Each agent gets the next free slot at or above `8801` (the first auto-assigned agent lands there; subsequent agents go to `8802`, etc., or whatever `--port` you passed to `kestrel create`). The exact value lives in `multi_agent.toml` under that agent's entry, and `kestrel start <name>` prints `Starting <name> on :<port>...`. |

> **Port conflict?** Edit the agent's entry in `multi_agent.toml` to change its port, or recreate the agent with a chosen port (`kestrel create MyAgent --port 8899`). Edit `multi_agent.toml`'s `[host]` section to change the host port (default `8888`). `kestrel start` itself doesn't take a `--port` flag — runtime ports are read from `multi_agent.toml`.

> **Test it:** Visit the URL the CLI printed on start (`http://localhost:8888` for the multi-agent host, or whatever per-agent port `kestrel start <name>` reported). The Sovereign Console is the default page; append `/health` for a JSON readiness probe.

> **Windows users:** the CLI prints emoji. If you see `UnicodeEncodeError: 'charmap' codec can't encode character ...`, run `chcp 65001` once in your PowerShell session to switch the console to UTF-8. (As of v0.1.9 the CLI auto-reconfigures stdout, so a fresh install should not hit this.)

### Install from PyPI (no source clone)

The Quick Start above clones the repo so you have demos, examples, and the `kestrel.toml.example` next to your agent. If you only want to run an agent and don't need the source tree, install from PyPI:

```bash
# 1. Install the CLI (uv tool install is preferred — `kestrel`
#    lands on PATH in an isolated venv. Plain `pip install
#    kestrel-sovereign` works too, into whichever venv is active.)
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

**Where data lives.** `kestrel` resolves the project directory in this order: `KESTREL_HOME` → walk up from CWD looking for a `multi_agent.toml` / `kestrel.toml` / `.env` marker → `~/.kestrel/` for pip-installed users with no markers anywhere. A pure pip install with no `KESTREL_HOME` and no project in CWD lands on `~/.kestrel/` and creates it on first run. **Never** writes to `site-packages/` — `pip install --upgrade kestrel-sovereign` is safe and won't touch your agent data.

If you later want to switch your data dir, move it: `mv ~/.kestrel /new/path && export KESTREL_HOME=/new/path`. The agent's encrypted DB is portable; nothing about the data dir is hard-coded.

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
kestrel health                       # Check prerequisites
kestrel create MyAgent               # Create a new agent
kestrel start MyAgent                # Start an agent
kestrel stop MyAgent                 # Stop an agent
kestrel restart MyAgent              # Restart (stop then start)
kestrel update [MyAgent]             # Pull + install + feature sync + restart (see below)
kestrel status                       # Show all running agents
kestrel list                         # List available agents
kestrel shell MyAgent                # CLI chat interface
kestrel config ./agent_data/MyAgent  # Show agent config
```

#### Pulling in upstream changes (`kestrel update`)

Replaces the four-step ritual `git pull && uv sync && kestrel feature sync && kestrel restart` with one verb:

```bash
kestrel update                       # pull + install + feature sync + restart all agents
kestrel update Emma                  # same, but only restart the named agent
kestrel update --dry-run             # preview the steps without mutating anything
kestrel update --no-pull --no-install # only feature sync + restart (e.g. after a manual checkout)
kestrel update --allow-dirty         # pull even with modified tracked files in the working tree
kestrel update --uv-sync             # force `uv sync` for the install step
kestrel update --no-uv-sync          # force `uv pip install -e .` for the install step
kestrel update --no-deps             # pass --no-deps to `uv pip install` for the fast path
kestrel update --continue-on-error   # restart even if `feature sync` reports an error
```

**Install step auto-detect.** When `uv.lock` exists at the source checkout root, the install step runs `uv sync` — that refreshes deps from the lockfile AND prunes packages not in it. That's why `kestrel feature sync` runs immediately after: it restores any out-of-tree feature packages (`kestrel-feature-*`) that `uv sync` just pruned. Without `uv.lock` present, the step falls back to `uv pip install -e .` which only reinstalls the project package. Force either with `--uv-sync` / `--no-uv-sync`.

**Dirty-tree detection.** Only modified or staged **tracked** files block the pull. Untracked files (`kestrel.toml.backup-*`, scratch files, etc.) don't count — `git pull --ff-only` can't collide with files git doesn't know about. The refusal message surfaces the porcelain summary so you can see exactly what to commit/stash.

Safety:

- `git pull --ff-only` so a non-fast-forward upstream aborts instead of producing a surprise merge.
- Refuses to pull when the working tree has modified tracked files unless `--allow-dirty` is passed.
- Any step's failure short-circuits the rest, so a half-applied update never reaches the restart phase.
- If `kestrel-sovereign` was installed from PyPI (no editable source checkout discoverable from the package's `__file__`), both `pull` AND `install` are silently skipped — `feature sync` and `restart` still run. Upgrade the package itself with `pip install --upgrade kestrel-sovereign` and then re-run `kestrel update` to pick up the new feature manifest and restart agents.

### Feature management (`kestrel feature`)

Kestrel ships a lean core. Optional feature packages register `Feature` classes through the `kestrel_sovereign.features` entry-point group. Provider packages, such as cloud, voice, and storage backends, register with provider-specific entry-point groups and are consumed by their owning core or feature module.

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

The canonical inventory of features lives in [`KESTREL_FEATURES.md`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/KESTREL_FEATURES.md); the runtime registry is in [`kestrel_sovereign/data/feature_registry.toml`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/kestrel_sovereign/data/feature_registry.toml).

#### Keeping features installed across `uv sync` (`kestrel feature sync`)

Feature and provider packages are deliberately **not** declared in `pyproject.toml` (that would invert the open-core dependency direction), so `uv sync` — which makes the venv match declared dependencies exactly — **prunes** them. After a sync or any environment rebuild, an installed feature can silently vanish (e.g. voice endpoints start returning 404). `kestrel feature upgrade` cannot bring it back, because a pruned package leaves no entry-point to discover.

`kestrel feature sync` is the restore counterpart. It reads a host-local manifest, `.kestrel-host-features.toml` (gitignored; template at [`.kestrel-host-features.toml.example`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/.kestrel-host-features.toml.example)), and reinstalls whatever is missing:

```bash
kestrel feature sync --capture         # Snapshot currently-installed extensions into the manifest
kestrel feature sync                    # Reinstall the manifest's packages that are missing
kestrel feature sync --dry-run          # Preview without installing
kestrel feature sync --manifest PATH    # Use a non-default manifest location
```

A manifest entry names a package (by registry name or raw dist name) and may pin an editable checkout and extras:

```toml
[[feature]]
name = "voice"
extras = ["local"]                      # also pull the on-device pipeline (Piper / faster-whisper)

[[feature]]
name = "github"
editable = "/path/to/kestrel-feature-github"
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
# Start server directly (set KESTREL_DB_PATH first).
# Pip-installed users invoke the in-package module (the wheel only ships
# `kestrel_sovereign/`, no top-level `server.py`):
KESTREL_DB_PATH=./agent_data/myagent uvicorn kestrel_sovereign.server:app --port 8888

# CLI chat (no server needed)
python -m kestrel_sovereign.main ./agent_data/myagent

# Source-clone shorthand: a thin re-export shim at the repo root keeps
# the historical commands working when you've cloned the repo:
#   uvicorn server:app --port 8888
#   python main.py ./agent_data/myagent
```

> **Note:** `KESTREL_DB_PATH` is a **directory** path, not a file path. The database file `kestrel_prime.db` is created inside the specified directory. For example, setting `KESTREL_DB_PATH=./agent_data/myagent` stores the database at `./agent_data/myagent/kestrel_prime.db`.

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

The console provides 9 tabs:

| Tab | Description |
|-----|-------------|
| **Chat** | Converse with the agent (supports model selection, privacy modes, chat history) |
| **Identity** | View the agent's DID, name, and cryptographic identity |
| **Constitution** | View and audit the agent's constitutional principles |
| **Memories** | Browse the agent's knowledge graph and stored memories |
| **Tasks** | Monitor background tasks and activity |
| **Sovereignty** | Manage data sovereignty, backups, and exports |
| **Resources** | View agent resource usage and configuration |
| **Features** | Browse installed, available, enabled, and disabled features |
| **Security** | Manage permissions, audit logs, and session security |

> **Alternative clients:** The server also exposes an OpenAI-compatible API at `/v1/chat/completions`, so you can connect any OpenAI-compatible client (e.g., [Open WebUI](https://github.com/open-webui/open-webui)) if you prefer.

## 🏗️ Architecture Overview

Kestrel agents are built on several key components:

- **Cryptographic Identity**: Each agent has a unique DID (Decentralized Identifier)
- **Enhanced Storage**: Local-first memory with SQL-backed stores, SQLAlchemy vector mappings, FTS, knowledge graphs, and RAG; provider-standard embedding generation is in progress
- **Multi-Model LLM**: Fallback between local (Ollama) and cloud (OpenAI) models
- **Constitutional Governance**: Immutable principles with interpretive flexibility
- **Blockchain Anchoring**: Optional integrity verification via blockchain

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
├── server.py                  # Re-export shim → kestrel_sovereign/server.py
├── host.py                    # Multi-agent multi_agent host
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
- **Persistent Storage**: SQL-backed stores with full-text search and knowledge graphs
- **RAG Pipeline**: Document chunking and semantic retrieval through SQLAlchemy-backed vector search where available; embedding generation still uses the current embedding service while provider-standard embedding functions are being added
- **Conversation History**: Complete interaction tracking with metadata
- **Human-Led Interactions**: Prioritizes user narratives (e.g., storytelling) for preservation and no-loss continuity.

### 2. Multi-Model Intelligence
- **Local First**: Ollama for privacy and cost efficiency
- **Cloud Fallback**: OpenAI for complex reasoning when needed
- **Configurable**: Easy provider switching via configuration

### 3. Cryptographic Identity
- **DID Generation**: Unique decentralized identifiers
- **Signed Operations**: Cryptographic verification of agent actions
- **Ownership Transfer**: Secure agent handoff between users

### 4. Constitutional Governance
- **Immutable Articles**: Core principles that cannot be changed
- **Interpretive Canons**: Flexible guidelines for decision-making
- **Amendment Process**: Cryptographically-signed governance updates

### 5. Data Sovereignty & Privacy Modes
- **Ephemeral Mode**: True off-the-record conversations (nothing stored)
- **Privacy Granularity**: 5 distinct privacy levels for different use cases
- **Decentralized Storage**: Filecoin/IPFS integration for vendor independence
- **Optional Economics**: Wallet and autonomous payment flows are installable feature-package surfaces, not part of the base install

## ⚠️ Feature Stability (v0.18+ Beta)

Kestrel covers a wide surface; not all of it ships at the same maturity. **Verified 2026-05-31** against the current feature inventory, runtime registry, package-boundary docs, tests, and recent documentation audit:

### ✅ Stable — battle-tested in production by the maintainers

- **Constitutional AI** — Genesis audits, hierarchical permissions, approval queues
- **DID-based Identity** — `did:pkh` format, portable agent identity, export/import
- **5-Level Privacy Modes** — EPHEMERAL → ISOLATED → ANONYMOUS → NORMAL → PUBLIC
- **Memory & Storage** — Local-first memory, FTS, knowledge graph, RAG, and SQLAlchemy-backed vector storage are core; embedding generation is the moving part as LLM providers gain standardized embedding functions
- **LLM service** — Vendor/route/model architecture with Anthropic, OpenAI, Vertex AI, Ollama, OpenRouter, xAI, Groq; retry, structured output, streaming, vision
- **A2A Protocol** — JSON-RPC 2.0 for agent-to-agent communication
- **Cloud Run deploy** — 90 tests, active maintenance; the most-tested cloud feature

### 🧪 Experimental — works on the happy path; gaps to know about

- **Optional voice feature package** — `kestrel-feature-voice` supplies `VoiceFeature`; cloud TTS/STT providers live in `kestrel-voice-*` provider packages.
- **Wallet / agent economics** — installable feature package surface, not part of the base install.
- **RunPod, Vast.ai, and GCP Compute** — cloud provider packages/bridges, not core features; integration tests skip in CI without provider credentials.
- **Azure Container Apps deploy** — provider stub; not the recommended deploy target.
- **GitHub code introspection** — file reading, code search, definition lookup, issue tools all work (48 unit tests). The deeper static-analysis surface promised in [`docs/architecture/GITHUB_FEATURE_DESIGN.md`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/architecture/GITHUB_FEATURE_DESIGN.md) (call graphs, inheritance trees, dependency analysis) is not implemented.
- **Training (LoRA pipeline)** — core ships the protocol + factory; the local-MPS adapter is actively maintained. Cloud-training adapters (RunPod/Vertex/Replicate) work but skip CI without API keys; production-grade adapters are being moved to private packages.

### ⚠️ Work-in-progress

- **DID Verification Layer** — generation works; verification is incomplete
- **E2E Test Stability** — some integration tests are occasionally flaky
- **API Stability** — APIs may change before v1.0; breaking changes will be documented

### ❌ Not implemented in this framework

These are not bundled in the `kestrel-sovereign` base install:

- **Turnkey channel adapters** — WhatsApp, Telegram, Discord, and Slack adapters are not shipped in the base install; the core channels surface is a registry/logging foundation.
- **Bundled voice cloud backends** — ElevenLabs, Deepgram, and OpenAI voice support live in optional `kestrel-voice-*` provider packages.
- **Browser Automation** — Chrome/Chromium control
- **Visual Workspaces** — A2UI canvas, live reload

**Bottom line:** Kestrel is ready for developers building privacy-first, economically-independent AI agents and for the soft-launch preview cohort. Not yet ready for unmanaged production apps or general consumer use. If you find a stability classification above doesn't match your experience, please open an issue — that's the kind of signal we need.

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

**Storage:**
- `KESTREL_DB_PATH`: Directory where the agent database is stored (default: `./agent_data`). This is a **directory** path -- the database file `kestrel_prime.db` is created inside it.
- `KESTREL_DATA_KEY`: Fernet encryption key for data at rest

**GitHub Integration:**
- `GITHUB_TOKEN`: Personal access token for GitHub features
- `GITHUB_SELF_REPO`: Agent's source repository (default: `KestrelSovereignAI/kestrel-sovereign`)

## 🚢 Deployment

Kestrel supports multiple deployment targets. See [KESTREL_FEATURES.md](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/KESTREL_FEATURES.md#11-deployment) for the full catalog.

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

- local: cache the backup tar.gz locally only
- ipfs: encrypt + gzip and store on IPFS; also cache locally
- filecoin: same as IPFS and propose a Filecoin deal via Lotus when available; fallback to local if not

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

- Files and conversation history can be encrypted at rest by setting `KESTREL_DATA_KEY` (Fernet key or passphrase):

```bash
export KESTREL_DATA_KEY=$(python - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
)
```

- With the key set, stored file blobs and conversation entries are encrypted transparently. Backups remain encrypted by default. For production, wire the backup master key to an env/KMS and avoid the dev placeholder.

### Optional: Full-DB Encryption (SQLCipher)

- If you install `pysqlcipher3` and set `KESTREL_DB_KEY`, the SQLite connection will use SQLCipher and encrypt the entire DB:

```bash
export KESTREL_DB_KEY="your-db-passphrase"
uv run python -m kestrel_sovereign.server
```

- Without `pysqlcipher3`, the system falls back to normal SQLite. File blobs and conversations still encrypt with `KESTREL_DATA_KEY` if set.

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
- **Documentation**: See [`docs/`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/README.md) and [`KESTREL_FEATURES.md`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/KESTREL_FEATURES.md)

---

*Kestrel: Where AI meets sovereignty.* 

## 📚 Key Files Reference

| File | Purpose |
|------|---------|
| `kestrel_sovereign/cli.py` | Canonical `kestrel` CLI entry point |
| `kestrel_sovereign/server.py` | FastAPI agent server (root `server.py` is a re-export shim for source clones) |
| `host.py` | Multi-agent multi_agent host (Cloud Run) |
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
*   **`storage.RAGStore`**: Responsible for document chunking and semantic search for the RAG pipeline and "case law" system. It uses SQLAlchemy/vector backends when available and falls back gracefully; embedding generation is being aligned with standardized provider embedding functions.
*   **`storage.ConversationStore`**: Manages the agent's conversation history.

The main `Storage` class in `storage/__init__.py` acts as a facade, providing a single, unified interface to these components.

### Genesis Self-Audit

To ensure the integrity of all new agents, Kestrel implements a "genesis self-audit." When a new agent is created via `inception_service.py`:

1.  The agent's foundational files (keys, database) are created.
2.  The `KESTREL_CONSTITUTION.md` is stored as the agent's first memory.
3.  The agent is instantiated and its very first action is to perform an integrity audit on its own constitution.
4.  If the audit returns a high risk level, the creation process is aborted, and all generated files are cleaned up, preventing the existence of a non-compliant agent.

This process guarantees that every agent in the ecosystem starts from a foundation of verifiable integrity.

## 🔄 Next Steps

After getting started:

1.  **Explore Features**: Read [`KESTREL_FEATURES.md`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/KESTREL_FEATURES.md) and [`docs/guides/BUILDING_FEATURES.md`](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/docs/guides/BUILDING_FEATURES.md)
