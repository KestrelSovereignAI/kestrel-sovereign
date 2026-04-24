# Kestrel: Sovereign AI Agent Framework

> Build AI agents that nobody can take away from their users — not you, not the cloud, not the next pivot.

Kestrel is a production-ready framework for creating autonomous AI agents with cryptographic identity, persistent memory, and constitutional governance. Every agent you deploy is **owned by its user**, governed by **immutable principles**, and able to **remember across every conversation**.

### Three Pillars

| Pillar | What it means |
|--------|--------------|
| **Portable DID identity** | Cryptographic identity the agent's user owns. Exportable, self-hostable, cloud-optional — the agent is not bound to any provider. |
| **Persistent memory you own** | SQLite-backed knowledge graph with full-text search and RAG. Conversations, documents, relationships — all searchable, portable, and encrypted at rest. |
| **Constitutional governance** | Every agent runs under an audited set of principles enforced *above* the LLM. Genesis audit on creation. Amendment requires cryptographic signature. |

> **In production:** Kestrel powers the AI Companion layer at Caprock Health, replacing static RPM bots with constitutional AI agents for remote patient monitoring.

## 🚀 Quick Start

### Prerequisites
- Python 3.11-3.13 (3.14 not yet supported due to tiktoken)
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

# 3. Configure LLM - edit with your API keys or Ollama settings
cp llm_config.toml.example llm_config.toml

# 4. Health check (verify prerequisites)
uv run kestrel health

# 5. Create your agent
uv run kestrel create MyAgent

# 6. Start your agent
uv run kestrel start MyAgent
```

Your agent is now running at `http://localhost:8888`.

> **Port conflict?** Each agent has its own config. Edit `agent_data/myagent/kestrel.toml` to change the port, or use `--port 8899` on the command line.

> **Test it:** Visit `http://localhost:8888` in your browser to open the built-in **Sovereign Console** (web UI with Chat, Identity, Constitution, Memories, and more). Or check `http://localhost:8888/health` for a quick health check.

> **Windows users:** the CLI prints emoji. If you see `UnicodeEncodeError: 'charmap' codec can't encode character ...`, run `chcp 65001` once in your PowerShell session to switch the console to UTF-8. (As of v0.1.9 the CLI auto-reconfigures stdout, so a fresh install should not hit this.)

### CLI Commands (Cross-Platform)

All commands work on Windows, macOS, and Linux. Pass the agent directory as an argument:

```bash
uv run kestrel health                       # Check prerequisites
uv run kestrel create MyAgent               # Create a new agent
uv run kestrel start MyAgent                # Start an agent
uv run kestrel stop MyAgent                 # Stop an agent
uv run kestrel status                       # Show all running agents
uv run kestrel list                         # List available agents
uv run kestrel shell MyAgent                # CLI chat interface
uv run kestrel config ./agent_data/MyAgent  # Show agent config
```

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
uv run kestrel config ./agent_data/myagent --init           # Create config
uv run kestrel config ./agent_data/myagent --set-port 8899  # Change port
uv run kestrel config ./agent_data/myagent --set-name MyAgent  # Change name
```

### Running Multiple Agents

Each agent runs on its own port. Create configs for each:

```bash
# Agent 1: Alpha on port 8888
uv run kestrel create Alpha --port 8888
uv run kestrel start Alpha

# Agent 2: Helper on port 8889
uv run kestrel create Helper --port 8889
uv run kestrel start Helper

# Check status of all agents
uv run kestrel status
```

### Alternative: Direct Commands

```bash
# Start server directly (set KESTREL_DB_PATH first)
KESTREL_DB_PATH=./agent_data/myagent uv run uvicorn server:app --port 8888

# CLI chat (no server needed)
uv run python main.py ./agent_data/myagent
```

> **Note:** `KESTREL_DB_PATH` is a **directory** path, not a file path. The database file `kestrel_prime.db` is created inside the specified directory. For example, setting `KESTREL_DB_PATH=./agent_data/myagent` stores the database at `./agent_data/myagent/kestrel_prime.db`.

## 🖥️ Web UI (Sovereign Console)

Kestrel includes a built-in web interface called the **Sovereign Console**. Once your agent is running, open `http://localhost:8888` in any browser -- no additional software required.

The console provides 8 tabs:

| Tab | Description |
|-----|-------------|
| **Identity** | View the agent's DID, name, and cryptographic identity |
| **Chat** | Converse with the agent (supports model selection, privacy modes, chat history) |
| **Constitution** | View and audit the agent's constitutional principles |
| **Memories** | Browse the agent's knowledge graph and stored memories |
| **Tasks** | Monitor background tasks and activity |
| **Sovereignty** | Manage data sovereignty, backups, and exports |
| **Resources** | View agent resource usage and configuration |
| **Security** | Manage permissions, audit logs, and session security |

> **Alternative clients:** The server also exposes an OpenAI-compatible API at `/v1/chat/completions`, so you can connect any OpenAI-compatible client (e.g., [Open WebUI](https://github.com/open-webui/open-webui)) if you prefer.

## 🏗️ Architecture Overview

Kestrel agents are built on several key components:

- **Cryptographic Identity**: Each agent has a unique DID (Decentralized Identifier)
- **Enhanced Storage**: SQLite-based memory with FTS, knowledge graphs, and RAG
- **Multi-Model LLM**: Fallback between local (Ollama) and cloud (OpenAI) models
- **Constitutional Governance**: Immutable principles with interpretive flexibility
- **Blockchain Anchoring**: Optional integrity verification via blockchain

## 📁 Project Structure

```
kestrel-sovereign/
├── kestrel_cli.py             # `kestrel` CLI entry point
├── server.py                  # FastAPI agent server
├── main.py                    # Direct interactive chat
├── kestrel_sovereign/         # Core sovereign package
│   ├── kestrel_agent.py      # Core agent class
│   ├── inception_service.py  # Agent creation (DID + genesis audit)
│   ├── agent_config.py       # Per-agent config loader
│   └── ...
├── kestrel_sdk/               # Public SDK for feature authors
├── packages/                  # Extracted feature packages
├── features/                  # Built-in feature documentation
├── llm/                       # LLM adapters and services
├── docs/                      # Architecture & guides
└── tests/                     # Test suite
```

## 🎯 Core Features

### 1. Sovereign Memory
- **Persistent Storage**: SQLite with full-text search and knowledge graphs
- **RAG Pipeline**: Document chunking, embedding, and semantic retrieval
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
- **Agent Economics**: Autonomous economic contracts using cryptographic payments

## ⚠️ Known Limitations (v0.1.8 Beta)

### ✅ Working & Production-Ready:
- **Constitutional AI** - Genesis audits, hierarchical permissions, approval queues
- **DID-based Identity** - `did:pkh` format, portable agent identity
- **5-Level Privacy Modes** - EPHEMERAL → ISOLATED → ANONYMOUS → NORMAL → PUBLIC
- **Multi-LLM Support** - Anthropic, OpenAI, Gemini, Ollama, OpenRouter, xAI, Groq
- **Agent Economics** - Multi-currency wallets (FIL, USDC, USDT, ETH)
- **A2A Protocol** - JSON-RPC 2.0 for agent-to-agent communication
- **Enhanced Storage** - SQLite/PostgreSQL with FTS, knowledge graphs, RAG pipeline

### ⚠️ Work-in-Progress:
- **DID Verification Layer** - Identity generation works, verification incomplete
- **E2E Test Stability** - Some integration tests are occasionally flaky
- **API Stability** - APIs may change before v1.0 (we'll document breaking changes)

### ❌ Not Implemented (Use OpenClaw Instead):
- **Multi-Channel Messaging** - WhatsApp, Telegram, Discord, Slack integration
- **Voice Interaction** - Wake word detection, TTS
- **Browser Automation** - Chrome/Chromium control
- **Visual Workspaces** - A2UI canvas, live reload

**Bottom Line**: Kestrel is ready for developers building privacy-first, economically-independent AI agents. Not ready for production apps or general consumer use.

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [Documentation Index](docs/README.md)

- [Agent Ecosystem](docs/architecture/AGENT_ECOSYSTEM.md)
- [Agent Economics](docs/architecture/AGENT_ECONOMICS.md)
- [Kestrel Constitution](docs/principles/KESTREL_CONSTITUTION.md)
- [Cryptographic Anchoring](docs/architecture/CRYPTOGRAPHIC_ANCHORING.md)
- [Decentralized Storage](docs/architecture/DECENTRALIZED_STORAGE.md)
- [Multi-Model Support](docs/architecture/MULTI_MODEL_SUPPORT.md)
- [Privacy Modes](docs/architecture/PRIVACY_MODES.md)

## 💡 Example Applications

Kestrel is a foundation for AI agents that need to outlive any single vendor, deployment, or owner. Concrete deployments and good-fit use cases:

- **Healthcare RPM agents** — Already running in a clinical study at Caprock Health: constitutional governance over an LLM, persistent patient-owned memory, audit trail for every clinically-relevant action.
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
./scripts/verify_clean_install.sh

# Run specific tests only
./scripts/verify_clean_install.sh 1 3    # SDK-only and wallet package
```

The install matrix covers:

| Test | Scenario | Verifies |
|------|----------|----------|
| 1 | **SDK only** | `from kestrel_sdk.features.base import Feature` |
| 2 | **Core sovereign** | `from kestrel_sovereign.features.base import Feature` + `/health` |
| 3 | **Feature package** | `from kestrel_feature_wallet import WalletFeature` |
| 4 | **SDK + feature dev mode** | Feature packages can develop against SDK alone |
| 5 | **Full stack** | Sovereign + wallet + intelligence, entry_point discovery |

Integration tests for the same import paths run as part of the normal test suite:

```bash
uv run pytest tests/integration/test_clean_install_verification.py -v
```

## 🔧 Configuration

### LLM Configuration (`llm_config.toml`)

Kestrel uses a **vendor/route/model** schema. A *vendor* is who makes the weights; a *route* is how to reach them (adapter + base URL + auth). API keys belong in `.env` and are referenced by `api_key_env`. See [`llm_config.toml.example`](llm_config.toml.example) and [`docs/architecture/LLM_SERVICE_ARCHITECTURE.md`](docs/architecture/LLM_SERVICE_ARCHITECTURE.md) for the canonical spec.

```toml
route_priority = ["openai:api", "ollama:local"]

[vendors.openai]
is_cloud = true

[vendors.openai.routes.api]
adapter        = "OpenAIAdapter"
api_key_env    = "OPENAI_API_KEY"
model          = "auto"
selection_hints = ["gpt-5", "mini"]

[vendors.ollama]
is_cloud = false

[vendors.ollama.routes.local]
adapter        = "OllamaAdapter"
host           = "http://localhost:11434"
model          = "auto"
selection_hints = ["llama3.2", "qwen"]
```

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

Kestrel supports multiple deployment targets. See [KESTREL_FEATURES.md](KESTREL_FEATURES.md#11-deployment) for the full catalog.

### Cloud Run (Serverless)

Scales to zero when idle ($0/month), auto-scales under load. Each sovereign agent gets its own service.

```bash
# One-time: set up GCP secrets from .env
scripts/cloudrun/setup_secrets.sh

# Build and push to GCR
scripts/cloudrun/build.sh

# Deploy to dev (scales to zero) or prod (always warm)
scripts/cloudrun/deploy_dev.sh
scripts/cloudrun/deploy_prod.sh
```

Auto-deploys on version tags via [GitHub Actions](.github/workflows/deploy.yml).

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
uv run python server.py
```

- Without `pysqlcipher3`, the system falls back to normal SQLite. File blobs and conversations still encrypt with `KESTREL_DATA_KEY` if set.

## 🧩 OpenAI-Compatible API

The server exposes OpenAI-compatible endpoints for use with third-party clients:

- `GET /v1/models`
- `POST /v1/chat/completions`

For most users, the built-in **Sovereign Console** at `http://localhost:8888` is the easiest way to interact with your agent (see [Web UI](#-web-ui-sovereign-console) above). If you prefer an external client, point any OpenAI-compatible tool (e.g., [Open WebUI](https://github.com/open-webui/open-webui)) at your server's `/v1/chat/completions` endpoint. Use the model name from `/v1/models`.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `python -m pytest -x`
5. Submit a pull request

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## 🆘 Support

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Documentation**: See `features/` directory for detailed guides

---

*Kestrel: Where AI meets sovereignty.* 

## 📚 Key Files Reference

| File | Purpose |
|------|---------|
| `kestrel_cli.py` | Main CLI entry point |
| `main.py` | Interactive chat interface |
| `server.py` | FastAPI server |
| `llm_config.toml` | LLM provider configuration |
| `kestrel_sovereign/kestrel_agent.py` | Core agent logic |
| `kestrel_sovereign/agent_config.py` | Per-agent config loader |
| `kestrel_sovereign/inception_service.py` | New agent creation |
| `agent_data/<name>/kestrel.toml` | Per-agent configuration |
| `agent_data/<name>/kestrel_prime.db` | Agent database |
| `docs/**/*.md` | Detailed documentation |

## Architecture

### Storage System

The Kestrel storage system is designed to be modular and extensible. It is composed of several specialized components, orchestrated by a high-level facade.

*   **`storage.Database`**: Manages the low-level SQLite connection and schema.
*   **`storage.FileStore`**: Handles the storage and retrieval of files.
*   **`storage.GraphStore`**: Manages the knowledge graph (nodes and edges).
*   **`storage.RAGStore`**: Responsible for document chunking and semantic search for the RAG pipeline and "case law" system.
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

1.  **Explore Features**: Read `features/` documentation 
