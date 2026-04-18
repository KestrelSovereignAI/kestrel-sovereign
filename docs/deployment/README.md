# Deployment Operations

How to build, deploy, and update Kestrel Sovereign on Cloud Run.

## Scripts

All deployment scripts live in [`scripts/cloudrun/`](../../scripts/cloudrun/).

| Script | Purpose |
|---|---|
| [`build.sh`](../../scripts/cloudrun/build.sh) | Build + push single-agent and rookery images to GCR |
| [`build_rookery.sh`](../../scripts/cloudrun/build_rookery.sh) | Build + push rookery image only |
| [`deploy_dev.sh`](../../scripts/cloudrun/deploy_dev.sh) | Deploy rookery to `kestrel-dev` service |
| [`deploy_prod.sh`](../../scripts/cloudrun/deploy_prod.sh) | Deploy to `kestrel-prod` service |
| [`deploy_rookery_dev.sh`](../../scripts/cloudrun/deploy_rookery_dev.sh) | Deploy rookery image to dev |
| [`setup_secrets.sh`](../../scripts/cloudrun/setup_secrets.sh) | One-time: create GCP Secret Manager entries |

## Environment variables

Scripts read from `.env` at repo root. Required:

- `GCP_PROJECT_ID` — e.g. `kestel-469222`
- `KESTREL_ALLOWED_EMAILS` — comma-separated list of authorized Google accounts

## Typical dev deploy flow

```bash
# 1. Load env
set -a && source .env && set +a

# 2. Build + push images (takes ~5 min)
GITHUB_TOKEN=$(gh auth token --user UncleSaurus) bash scripts/cloudrun/build.sh

# 3. Deploy to dev
bash scripts/cloudrun/deploy_dev.sh
```

Dev service URL (stable): `https://kestrel-dev-7jpbsywhdq-uc.a.run.app`

## GitHub Actions

| Workflow | Trigger | Purpose |
|---|---|---|
| [`deploy.yml`](../../.github/workflows/deploy.yml) | `v*` tag push or manual | Build + deploy to Cloud Run |
| [`ci.yml`](../../.github/workflows/ci.yml) | Every PR/push | Unit + integration tests |
| [`clean-install.yml`](../../.github/workflows/clean-install.yml) | Scheduled | Verify `pip install` works from clean venv |
| [`weekly-analysis.yml`](../../.github/workflows/weekly-analysis.yml) | Weekly cron | Codebase analysis reports |

## Secrets

Stored in GCP Secret Manager. The deploy scripts mount them as env vars.

| Secret | Purpose |
|---|---|
| `kestrel-openai-key` | OpenAI API key |
| `kestrel-anthropic-key` | Anthropic API key |
| `kestrel-api-key` | Internal Kestrel API key |
| `kestrel-data-key` | Encryption key for agent data |
| `kestrel-session-secret` | Session cookie signing |
| `kestrel-google-client-id` / `-secret` | Google OAuth |
| `github-read-kestrel` | GitHub PAT for private package installs |
| `github-app-id` | Kestrel GitHub App ID |
| `github-app-private-key` | Kestrel GitHub App PEM |
| `github-app-webhook-secret` | Webhook HMAC secret |

Rotate by creating a new version:
```bash
gcloud secrets versions add <secret-name> --data-file=<path>
```

## Kestrel GitHub Agent — update flow

The GitHub Agent ([`kestrel_sovereign/features/github_app/`](../../kestrel_sovereign/features/github_app/)) runs on the same Cloud Run rookery as the main Kestrel agents. Its source code (and the codebase it reads when answering questions) is baked into the Docker image at build time.

**When you change source code, it won't reflect on the agent until you redeploy:**

```bash
set -a && source .env && set +a
GITHUB_TOKEN=$(gh auth token --user UncleSaurus) bash scripts/cloudrun/build.sh
bash scripts/cloudrun/deploy_dev.sh
```

The agent uses `min-instances=1` to stay warm (LLM calls exceed GitHub's 10s webhook timeout, so responses are async — instance must persist after returning 200).

**Long-term:** Auto-deploy on merge to main via GitHub Actions (see #TBD).

## Rookery vs single-agent

The dev deployment runs **rookery mode** — a host process routes traffic to per-agent subprocesses. See [`host.py`](../../host.py) for the host, [`server.py`](../../server.py) for individual agents.

Two agents ship in the default rookery image:
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
