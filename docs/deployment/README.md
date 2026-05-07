# Deployment Operations

How to build, deploy, and update Kestrel Sovereign on Cloud Run.

## Commands

All deployment operations run through the `kestrel deploy` CLI — a Python
entry point that works on Linux, macOS, and Windows (no bash required).
Configuration lives in [`deploy_config.toml`](../../deploy_config.toml);
the CLI is a thin wrapper over `DeployManager` and the build/secrets ports.

| Command | Purpose |
|---|---|
| `uv run kestrel deploy build` | Build + push single-agent and multi_agent images to GCR |
| `uv run kestrel deploy build --target kestrel-multi_agent` | Build + push the multi_agent image only |
| `uv run kestrel deploy dev` | Deploy `[profiles.dev]` (multi-agent host) to `kestrel-dev` |
| `uv run kestrel deploy prod` | Deploy `[profiles.prod]` to `kestrel-prod` |
| `uv run kestrel deploy multi-agent-dev` | Deploy the standalone multi_agent profile |
| `uv run kestrel deploy secrets sync` | One-time / on-rotate: push `.env` values into GCP Secret Manager |
| `uv run kestrel deploy status` | List active deployments |
| `uv run kestrel deploy logs <profile>` | Tail Cloud Run logs |
| `uv run kestrel deploy teardown <profile>` | Delete a deployed service |
| `uv run kestrel deploy health <profile>` | Probe the service's `/health` endpoint |

Run `uv run kestrel deploy --help` for the full flag set.

## Environment variables

The CLI reads from your shell env (or `.env` if you `set -a && source .env`).
Required:

- `GCP_PROJECT_ID` — e.g. `kestel-469222` (also reads `[manager].gcp_project_id` in `deploy_config.toml`)
- `KESTREL_ALLOWED_EMAILS` — comma-separated list of authorized Google accounts; expanded into `[profiles.*.env_vars]` via `${KESTREL_ALLOWED_EMAILS}` placeholders.
- `GITHUB_TOKEN` (build only) — env-first, falls back to `gh auth token`. Needed for Dockerfiles that install private deps.

## Typical dev deploy flow

```bash
# 1. Load env
set -a && source .env && set +a

# 2. Build + push images (takes ~5 min, multi-arch via docker buildx)
uv run kestrel deploy build

# 3. Deploy to dev
uv run kestrel deploy dev
```

Dev service URL (stable): `https://kestrel-dev-7jpbsywhdq-uc.a.run.app`

## GitHub Actions

| Workflow | Trigger | Purpose |
|---|---|---|
| [`deploy.yml`](../../.github/workflows/deploy.yml) | `v*` tag push (via `Publish to PyPI`) or manual | Calls `uv run kestrel deploy build` then `kestrel deploy dev` / `prod` |
| [`ci.yml`](../../.github/workflows/ci.yml) | Every PR/push | Unit + integration tests |
| [`clean-install.yml`](../../.github/workflows/clean-install.yml) | Scheduled | Verify `pip install` works from clean venv |
| [`weekly-analysis.yml`](../../.github/workflows/weekly-analysis.yml) | Weekly cron | Codebase analysis reports |

## Secrets

Stored in GCP Secret Manager. `kestrel deploy secrets sync` reads `.env` and
creates / updates secret versions per the `[profiles.*.secrets]` map in
`deploy_config.toml`. The deploy command mounts them as env vars on Cloud Run.

| Secret | Purpose |
|---|---|
| `kestrel-openai-key` | OpenAI API key |
| `kestrel-anthropic-key` | Anthropic API key |
| `kestrel-api-key` | Internal Kestrel API key |
| `kestrel-data-key` | Encryption key for agent data |
| `kestrel-session-secret` | Session cookie signing |
| `kestrel-google-client-id` / `-secret` | Google OAuth |
| `kestrel-lighthouse-key` | Lighthouse pricing/oversight feed |
| `github-read-kestrel` | GitHub PAT for private package installs |
| `github-app-id` | Kestrel GitHub App ID |
| `github-app-private-key` | Kestrel GitHub App PEM |
| `github-app-webhook-secret` | Webhook HMAC secret |

Rotate by adding a new secret value to `.env` and re-running:

```bash
uv run kestrel deploy secrets sync
```

Or directly via `gcloud`:

```bash
gcloud secrets versions add <secret-name> --data-file=<path>
```

## Kestrel GitHub Agent — update flow

The GitHub Agent ([`kestrel_sovereign/features/github_app/`](../../kestrel_sovereign/features/github_app/)) runs on the same Cloud Run multi_agent as the main Kestrel agents. Its source code (and the codebase it reads when answering questions) is baked into the Docker image at build time.

**When you change source code, it won't reflect on the agent until you redeploy:**

```bash
set -a && source .env && set +a
uv run kestrel deploy build
uv run kestrel deploy dev
```

The agent uses `min-instances=1` to stay warm (LLM calls exceed GitHub's 10s webhook timeout, so responses are async — instance must persist after returning 200).

**Long-term:** Auto-deploy on merge to main via GitHub Actions (see #TBD).

## MultiAgent vs single-agent

The dev deployment runs **multi_agent mode** — a host process routes traffic to per-agent subprocesses. See [`host.py`](../../host.py) for the host, [`server.py`](../../server.py) for individual agents.

Two agents ship in the default multi_agent image:
- `Kestrel` (port 8801) — main agent with all features including GitHubApp
- `kestrel-demo` (port 8802) — demo agent for UI examples

The `/webhooks/github-app` endpoint is a proxy on the host that forwards to the first agent (Kestrel).

## Troubleshooting

### Webhook returns 401 with `{"detail": "Invalid or missing API Key"}`
The host's auth middleware is rejecting the request. Check [`host.py`](../../host.py) — `/webhooks/` paths must bypass API key auth.

### Webhook returns 200 but agent doesn't respond
Check the webhook delivery page on GitHub. Response body includes diagnostic info in dev mode. Common issues:
- Installation ID missing from payload → App not installed on the repo
- `no_mention` in diag → comment missing `@kestrel` trigger
- LLM error → check Cloud Run logs for the actual exception

### Logs not appearing in Cloud Run
Python `logging.info()` calls from request handlers don't always reach Cloud Run's structured log pipeline. Use `print(json.dumps({"severity": "WARNING", "message": "..."}), flush=True)` for debugging.
