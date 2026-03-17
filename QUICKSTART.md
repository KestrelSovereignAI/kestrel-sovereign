# Kestrel Quickstart — Sovereign Agent in 30 Minutes

By the end of this guide you will have a running Kestrel agent with all three sovereignty pillars active:
- A cryptographic DID identity (yours, portable, no platform owns it)
- An encrypted memory store that persists across sessions
- A constitutional governance layer enforced above the LLM

**Time:** ~30 minutes (most of that is downloading Ollama + the model)
**Prerequisites:** Python 3.11–3.13, internet connection for initial setup

---

## Step 1 — Install uv (fast Python package manager)

Kestrel uses [uv](https://github.com/astral-sh/uv) for dependency management. If you already have it, skip this step.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installing.

---

## Step 2 — Clone and install Kestrel

```bash
git clone https://github.com/KestrelSovereignAI/kestrel-sovereign.git
cd kestrel-sovereign
uv sync   # creates .venv and installs all dependencies
```

---

## Step 3 — Install Ollama (local LLM — no API key needed)

Download and install from [ollama.ai](https://ollama.ai), then:

```bash
ollama serve              # start Ollama (keep this terminal open)
ollama pull llama3.2:3b   # ~2GB download
```

> **Want to use OpenAI instead?** Skip to Step 4b. You can always switch later.

---

## Step 4a — Configure LLM (Ollama — recommended for privacy)

```bash
cp llm_config.toml.example llm_config.toml
```

The default config works with Ollama out of the box. No edits needed to start.

> The config file sets provider priority: Ollama first, OpenAI as fallback. Add your OpenAI key later if you want cloud fallback for complex reasoning.

---

## Step 4b — Configure LLM (OpenAI — if you prefer cloud)

```bash
cp llm_config.toml.example llm_config.toml
```

Edit `llm_config.toml` and set:
```toml
provider_priority = ["openai"]

[openai]
api_key = "sk-your-key-here"   # or use OPENAI_API_KEY env variable
model = "gpt-4o-mini"
```

---

## Step 5 — Create your first sovereign agent

```bash
uv run kestrel create --name MyAgent
```

This runs the **Inception Service**, which:
1. Generates a secp256k1 keypair and derives a cryptographic DID
2. Creates an encrypted SQLite memory store at `./agent_data/myagent/`
3. Anchors the Kestrel Constitution as the agent's first memory
4. Runs a genesis integrity audit — if it fails, the agent is not created

You'll see output like:
```
Agent created: did:kestrel:0x8955b8...
Constitution anchored
Genesis audit passed
Database: ./agent_data/myagent/kestrel_prime.db
```

> **That DID is yours.** It's cryptographically unique, not stored on any platform, and travels with the `kestrel_prime.db` file wherever you take it.

---

## Step 6 — Start your agent

```bash
uv run kestrel start ./agent_data/myagent
```

Your agent is now running at `http://localhost:8888`.

Open it in a browser to access the **Sovereign Console** — a full web UI with Chat, Identity, Constitution, Memories, Security, and more.

Or use the CLI chat interface:

```bash
uv run kestrel chat ./agent_data/myagent
```

Type anything to start:

```
You: Hello — who are you?
Agent: I'm MyAgent, a sovereign AI agent. I have a persistent memory, a cryptographic
       identity, and I operate under a constitutional governance framework...
```

Stop and restart anytime:

```bash
uv run kestrel stop ./agent_data/myagent
uv run kestrel start ./agent_data/myagent
```

Your agent remembers every previous conversation. It always will, until you tell it not to.

---

## Step 7 — Explore what your agent knows about itself

From the Sovereign Console at `http://localhost:8888`:

| Tab | What you'll find |
|-----|-----------------|
| **Identity** | DID, cryptographic keys, agent metadata |
| **Constitution** | The governing rules — what the agent can and cannot do |
| **Memories** | Knowledge graph, conversation history, stored documents |
| **Security** | Permissions, audit log, session management |

Check agent status from the CLI:

```bash
uv run kestrel status
```

---

## Privacy Modes

Kestrel has five privacy levels that control what your agent stores. The default is `NORMAL` (full encrypted persistence).

| Mode | What it means |
|------|--------------|
| `NORMAL` (default) | Full memory, full history, encrypted at rest |
| `ANONYMOUS` | Stored but stripped of identifying information |
| `ISOLATED` | Remembers within this session only — not across sessions |
| `EPHEMERAL` | Nothing stored — true off-the-record mode |
| `PUBLIC` | Stored with sharing enabled |

See [docs/architecture/PRIVACY_MODES.md](docs/architecture/PRIVACY_MODES.md) for the full reference.

---

## Optional: Connect an external client

Kestrel exposes an OpenAI-compatible REST API at `/v1/chat/completions`. Connect any OpenAI-compatible tool — [Open WebUI](https://github.com/open-webui/open-webui), Cursor, etc. — by pointing it at `http://localhost:8888`.

Verify it's running:
```bash
curl http://localhost:8888/health
# → {"status": "ok"}
```

---

## CLI reference

```bash
uv run kestrel health                         # check prerequisites
uv run kestrel create --name MyAgent          # create a new agent
uv run kestrel start ./agent_data/myagent     # start an agent
uv run kestrel stop ./agent_data/myagent      # stop an agent
uv run kestrel chat ./agent_data/myagent      # CLI chat
uv run kestrel status                         # show all running agents
uv run kestrel list                           # list available agents
uv run kestrel config ./agent_data/myagent    # show agent config
```

---

## What's next

| Guide | What it covers |
|-------|---------------|
| [docs/architecture/PRIVACY_MODES.md](docs/architecture/PRIVACY_MODES.md) | Privacy mode internals |
| [docs/architecture/CRYPTOGRAPHIC_ANCHORING.md](docs/architecture/CRYPTOGRAPHIC_ANCHORING.md) | DID identity and integrity anchoring |
| [docs/principles/KESTREL_CONSTITUTION.md](docs/principles/KESTREL_CONSTITUTION.md) | The default constitutional framework — and how to write your own |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute to Kestrel |

---

## Troubleshooting

**`ollama: command not found`**
Ollama isn't in your PATH after install. Restart your terminal or follow the [Ollama install guide](https://ollama.ai).

**`No agent found`**
You're pointing at a directory that hasn't been initialized yet. Run `kestrel create` first.

**Ollama is slow on first response**
The first response loads the model into memory — takes 10–30 seconds depending on hardware. Subsequent responses are fast.

**Windows: `Activate.ps1 cannot be loaded`**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Port already in use**
Edit `agent_data/myagent/kestrel.toml` and change the port, or pass `--port 8899` to `kestrel start`.

---

*Kestrel Sovereign AI — [back to README](README.md)*
