# Kestrel: Sovereign AI Agent Framework

> ⚠️ **Work In Progress** - Kestrel is under active development and is NOT ready for production use. APIs, schemas, and features may change without notice. Use at your own risk. We welcome contributions and feedback!

Kestrel is a framework for creating autonomous AI agents with cryptographic identity, persistent memory, and governance principles. Each agent maintains sovereignty over its own data and decision-making while operating under a constitutional framework.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (for package management)
- [Ollama](https://ollama.ai) (for local LLM inference)

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

# 2. Start Ollama (in a separate terminal)
ollama serve
ollama pull llama3.2:3b

# 3. Configure LLM
cp llm_config.toml.example llm_config.toml

# 4. Health check (verify prerequisites)
uv run kestrel health

# 5. Create your agent
uv run kestrel create --output-dir ./my_agent

# 6. Start the server
uv run kestrel start --agent-dir ./my_agent
```

Your agent is now running at `http://localhost:8888`.

### CLI Commands (Cross-Platform)

All commands work on Windows, macOS, and Linux:

```bash
uv run python kestrel_cli.py health         # Check prerequisites
uv run python kestrel_cli.py create         # Create a new agent  
uv run python kestrel_cli.py start          # Start the server
uv run python kestrel_cli.py status         # Check if running
uv run python kestrel_cli.py stop           # Stop the server
uv run python kestrel_cli.py chat           # CLI chat interface
```

Or after `uv sync`, use the shorter form:

```bash
uv run kestrel health
uv run kestrel create
uv run kestrel start
# etc.
```

### Alternative: Direct Commands

```bash
# Start server directly (set KESTREL_DB_PATH first)
KESTREL_DB_PATH=./my_agent uv run python server.py

# CLI chat (no server needed)
uv run python main.py ./my_agent
```

> **Note:** `main.py` expects a **directory** containing `kestrel_prime.db`, not the database file itself.

## 🏗️ Architecture Overview

Kestrel agents are built on several key components:

- **Cryptographic Identity**: Each agent has a unique DID (Decentralized Identifier)
- **Enhanced Storage**: SQLite-based memory with FTS, knowledge graphs, and RAG
- **Multi-Model LLM**: Fallback between local (Ollama) and cloud (OpenAI) models
- **Constitutional Governance**: Immutable principles with interpretive flexibility
- **Blockchain Anchoring**: Optional integrity verification via blockchain

## 📁 Project Structure

```
kestrel/
├── main.py                    # CLI entry point
├── kestrel_agent.py          # Core agent class
├── storage.py       # Memory and knowledge management
├── inception_service.py      # Agent creation service
├── config.py                 # Configuration management
├── llm/                      # LLM adapters and services
├── features/                 # Detailed feature documentation
└── tests/                    # Test suite
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

The Kestrel framework is a bedrock for sovereign AI, with the Sovereign Companion as primary focus:
- **Kestrel App (Romantic/Friendship Companion)**: User-owned for private intimacy, human-led sharing without loss.
- **Elderly Companion**: Human storytelling collection, health assistance, 'picture frame' legacies.
- **General Counsel Assistant**: Ethical info provision, bound by Constitution.
- **Healthcare Assistant**: ONC-compliant data ownership.
- **Sovereign Mail Services**: Physical mail/notary for tangible anchors.
Bedrock ensures no companion 'loss' via sovereignty; apps extend for specific human-AI symbiosis, leading to potential agentic independence.

## 🧪 Testing

Run the test suite from the activated virtual environment:
```bash
# Run a single test with uv and -x
uv run pytest -x tests/test_inception.py::test_successful_inception
```

## 🔧 Configuration

### LLM Configuration (`llm_config.toml`)
```toml
[providers.ollama]
enabled = true
base_url = "http://localhost:11434"
model = "llama3.2:3b"
priority = 1

[providers.openai]
enabled = false
api_key = "your-api-key"
model = "gpt-5"
priority = 2
```

### Environment Variables
- `KESTREL_DB_PATH`: Custom database location (default: `./`)
- `OPENAI_API_KEY`: OpenAI API key for cloud models

## 🚢 Deployment

Kestrel supports containerized deployment:

```bash
# Build Docker image
docker build -t kestrel .

# Run with custom database path
docker run -e KESTREL_DB_PATH=/data -v ./data:/data kestrel
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

The server exposes minimal OpenAI-compatible endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`

Point Open WebUI or any OpenAI client at your server (http://localhost:7777). Use model from `/v1/models`.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `python -m pytest -x`
5. Submit a pull request

## 📄 License

[License information to be added]

## 🆘 Support

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Documentation**: See `features/` directory for detailed guides

---

*Kestrel: Where AI meets sovereignty.* 

## 📚 Key Files Reference

| File | Purpose |
|------|---------|
| `main.py` | Interactive chat interface |
| `kestrel_agent.py` | Core agent logic |
| `storage/__init__.py` | High-level storage facade |
| `llm_config.toml` | LLM provider configuration |
| `inception_service.py` | New agent creation |
| `test_*.py` | Test suites |
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