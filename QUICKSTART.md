# Kestrel Quickstart — a Sovereign Agent in 30 Minutes

This guide takes a fresh source checkout to one running agent named `Kestrel`.
The setup creates a cryptographic identity, privacy-aware local storage,
constitutional governance, and an LLM route. What Kestrel retains depends on
the active privacy mode, and application-layer encryption does not cover every
database column; the exact boundaries are summarized below.

**Prerequisites:** Python 3.11–3.14, an internet connection for installation,
and either Ollama or a supported cloud-provider API key.

## Step 1 — Install uv

Kestrel uses [uv](https://docs.astral.sh/uv/) for Python and dependency
management. Skip this step if `uv` is already installed.

macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the terminal if the installer asks you to update `PATH`.

## Step 2 — Clone and install Kestrel

```bash
git clone https://github.com/KestrelSovereignAI/kestrel-sovereign.git
cd kestrel-sovereign
uv sync
```

`uv sync` creates `.venv` and installs the source checkout. It is the
first-install command; use `kestrel update` for later refreshes so separately
installed feature packages are reconciled too. See the
[update guide](README.md#pulling-in-upstream-changes-kestrel-update).

## Step 3 — Make an LLM available

The simplest key-free path is Ollama. Install it from
[ollama.ai](https://ollama.ai), ensure the Ollama app or `ollama serve` is
running, and pull the same model tag used by the main README:

```bash
ollama pull llama3.2:3b
```

Semantic-memory embeddings are optional. To make the usual local embedding
model available too:

```bash
ollama pull nomic-embed-text
```

If you prefer a cloud route, export one or more supported credentials before
Step 4: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or
`GOOGLE_API_KEY`. Quickstart detects non-empty cloud credentials and a reachable
Ollama service, then writes their ordered routes to `kestrel.toml`. Models stay
on `auto`; Kestrel discovers an available model using the route's selection
hints. If none are detected, setup still writes an Ollama route so the local
configuration is complete; Ollama must be running with a suitable model before
the agent can answer.

## Step 4 — Run non-interactive setup

```bash
uv run kestrel setup --quickstart
```

On a fresh checkout, quickstart:

1. Generates `KESTREL_DATA_KEY` and `KESTREL_API_KEY` in the gitignored
   `.env` file.
2. Writes the detected LLM routes to `kestrel.toml`.
3. Creates the first agent as `Kestrel` under `agent_data/Kestrel/`, including
   its DID, database, identity keys, constitution, and genesis audit state.
4. Registers `Kestrel` for autostart in `multi_agent.toml`. The first agent is
   assigned port `8801`; if that port is already allocated, setup uses the next
   available agent port and reports it.

Setup is idempotent. Keep `.env` private and backed up: losing or changing
`KESTREL_DATA_KEY` can make encrypted data unreadable, and
`KESTREL_API_KEY` authenticates clients to the server.

## Step 5 — Check readiness

```bash
uv run kestrel doctor
```

Doctor is read-only. It validates persisted configuration and local integrity;
it deliberately does not contact LLM providers. Resolve any reported blocker
before starting the agent. For the Ollama path, separately make sure Ollama is
reachable and `llama3.2:3b` is installed.

## Step 6 — Start the multi-agent host

The recommended first run starts the host because quickstart registered
`Kestrel` with autostart enabled:

```bash
uv run kestrel start
```

With the default fresh configuration, the CLI prints:

```text
URL:      http://localhost:8888
MultiAgent ready: http://localhost:8888
```

Open <http://localhost:8888> for the Sovereign Console. In this mode, port
`8888` belongs to the in-process host. The configured agent port (`8801` for
the first fresh agent) is not a second listener; it is used when that agent is
started by name instead.

The host's public readiness URL is top-level, while agent APIs are namespaced:

```text
Console and public health: http://localhost:8888
Kestrel API base:          http://localhost:8888/api/agents/Kestrel
```

## Step 7 — Verify health and call the API

For the following macOS/Linux examples, set the two non-secret URLs and load
the generated keys into the current shell without printing them:

```bash
export KESTREL_URL="http://localhost:8888"
export KESTREL_AGENT_API="$KESTREL_URL/api/agents/Kestrel"
set -a
. ./.env
set +a
```

The minimal readiness probe is intentionally public:

```bash
curl -sS "$KESTREL_URL/health"
# {"status":"ok","agent_initialized":true}
```

Detailed health is for operators and requires authentication:

```bash
curl -sS \
  -H "X-API-Key: $KESTREL_API_KEY" \
  "$KESTREL_URL/health/detailed"
```

OpenAI-compatible endpoints require the same authentication. In host mode,
use the agent-prefixed API base:

```bash
curl -sS \
  -H "X-API-Key: $KESTREL_API_KEY" \
  "$KESTREL_AGENT_API/v1/models"

curl -sS \
  -H "X-API-Key: $KESTREL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"kestrel-local","messages":[{"role":"user","content":"Hello — who are you?"}]}' \
  "$KESTREL_AGENT_API/v1/chat/completions"
```

`Authorization: Bearer $KESTREL_API_KEY` is also accepted. Never put the
generated key in a URL, committed file, screenshot, or command example with a
literal value.

Stop the host when you are finished:

```bash
uv run kestrel terminate
```

## Alternative — start only the named agent

Use this form instead of the host when you want one standalone agent process:

```bash
uv run kestrel start Kestrel
```

The CLI prints `Starting Kestrel on :<port>...`; use that reported port. In a
fresh quickstart it is normally `8801`, but `multi_agent.toml` is authoritative
if another allocation was needed. The console, `/health`, `/health/detailed`,
and `/v1/...` endpoints are all top-level on the named agent's URL—there is no
`/api/agents/Kestrel` prefix in standalone mode.

Do not run the host and a standalone process for the same agent at once. If the
host is already running, terminate it first with `uv run kestrel terminate`.

## Optional — create an additional agent

`Kestrel` already exists after quickstart. Creating another agent is optional:

```bash
uv run kestrel create MyAgent
uv run kestrel list
```

`create` prints the new agent's assigned port and registers it for autostart.
If the host is running, restart it to load the new agent:

```bash
uv run kestrel restart
```

The new host API is namespaced as
`http://localhost:8888/api/agents/MyAgent`. Alternatively, stop the host and
run `uv run kestrel start MyAgent`, then use the standalone port printed by the
CLI.

## Privacy and memory boundaries

`NORMAL` is the default. It permits persistent storage; it does not promise
that every interaction is remembered forever. Retention depends on the active
mode and on the storage path used by the feature handling the data.

| Mode | Storage and processing contract |
|---|---|
| `NORMAL` | Full persistent storage; cloud processing allowed; private sharing |
| `ANONYMOUS` | PII-redacted persistent storage; local processing; private sharing |
| `DEIDENTIFIED` | Deidentified persistent storage; trusted processing; research sharing; audit required |
| `ISOLATED` | Temporary session storage only; local processing; private sharing |
| `EPHEMERAL` | No storage; local processing; private sharing |
| `PUBLIC` | Full persistent storage; cloud processing allowed; public sharing |

See the [privacy modes reference](docs/architecture/security/PRIVACY_MODES.md)
for enforcement details and mode transitions.

## Encryption boundaries

Quickstart generates `KESTREL_DATA_KEY`, which enables application-layer
encryption for conversation-history rows, stored file blobs, identity private
keys, and agent-resource bodies such as the SOUL. It is not whole-database
encryption.

Saved-item bodies (`saved_items.content`) and RAG document-chunk bodies
(`document_chunks.content`) remain plaintext columns today. Embeddings are
unencrypted numeric vectors. SQLCipher is not wired, and setting
`KESTREL_DB_KEY` does not encrypt the database. Directory permissions are
access control, not encryption.

See the [README encryption coverage matrix](README.md#-encryption-at-rest) for
the current per-data-type contract.

## What’s next

Keep the 30-minute path small and use the focused runbooks for advanced
operations:

| Guide | Topic |
|---|---|
| [Identity export custody](docs/architecture/security/IDENTITY_EXPORT_CUSTODY.md) | Export permissions and fresh-target trust |
| [Host runtime storage custody](docs/architecture/security/HOST_RUNTIME_STORAGE_CUSTODY.md) | Host-owned feature-state storage |
| [Phoenix trace custody](docs/architecture/security/PHOENIX_TRACE_CUSTODY.md) | Optional tracing data and supervisor state |
| [Cryptographic anchoring](docs/architecture/security/CRYPTOGRAPHIC_ANCHORING.md) | Integrity anchors and verification boundaries |
| [Update guide](README.md#pulling-in-upstream-changes-kestrel-update) | Pull, install, feature reconciliation, and restart |
| [Deployment runbook](docs/deployment/README.md) | Cloud Run build and deployment |
| [Kestrel Constitution](docs/principles/KESTREL_CONSTITUTION.md) | Constitutional framework |

## Troubleshooting

**Doctor reports Ollama unavailable**

Start the Ollama app or run `ollama serve` in another terminal, then confirm
`ollama list` includes `llama3.2:3b`.

**`Agent 'Kestrel' not found`**

Run `uv run kestrel list`. On a clean checkout, rerun
`uv run kestrel setup --quickstart` to create and register `Kestrel`.

**Port already in use**

Run `uv run kestrel status` and stop an existing Kestrel process. Host port
configuration lives under `[host]` in `multi_agent.toml`; agent ports live in
their `[agents.<name>]` entries. `kestrel start` does not accept a `--port`
override.

**The first response is slow**

Ollama loads the model into memory on first use. Later responses are normally
faster.

---

*Kestrel Sovereign AI — [back to README](README.md)*
