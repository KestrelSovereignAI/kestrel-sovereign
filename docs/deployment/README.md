---
type: Runbook
title: Deployment Operations
description: How to build, deploy, and update Kestrel Sovereign on Cloud Run.
resource: /docs/deployment/README.md
tags:
- docs
- deployment
- runbook
timestamp: '2026-06-18T00:00:00Z'
status: active
owner: documentation
canonical: false
generated: false
privacy: public
---

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

Production also requires `KESTREL_PROD_EXPECTED_DID`. It is the exact active
signing DID provisioned during the custody ceremony below; the deploy manager
rejects an unset, unresolved, or non-DID value.

## Cloud Run identity and state lifetime

Cloud Run's writable filesystem is in-memory and disposable: files written by
one instance do not survive that instance and are not shared with another
instance. Kestrel profiles therefore declare one of two explicit contracts:

| Mode | Allowed use | Identity and state |
|---|---|---|
| `ephemeral_demo` | Development/test only, exactly one maximum instance | May mint a new test/demo DID after a cold start; SQLite and memory are intentionally disposable |
| `durable_sovereign` | Single-agent production | Restores one encrypted, pinned identity bundle; PostgreSQL is the authoritative store for DID, constitution anchor, audit records, and memory |

The checked-in `dev` and `multi-agent-dev` profiles are explicit ephemeral
demos and cap `max_instances` at 1. They are not sovereign production
deployments. `prod` is durable and may scale horizontally because every
instance restores the same cryptographically verified signing identity and
uses the same transactional PostgreSQL database. `multi-agent-prod` is
intentionally refused until Kestrel can bind a separate custody bundle and
database identity to every hosted agent.

Do not put SQLite on a Cloud Storage mount. Object storage does not provide the
filesystem locking/transaction semantics SQLite requires. See Google's
[Cloud Run runtime contract](https://docs.cloud.google.com/run/docs/container-contract),
[Cloud Run limits](https://docs.cloud.google.com/run/quotas), and
[Secret Manager integration](https://docs.cloud.google.com/run/docs/configuring/services/secrets).

### Production custody ceremony

Perform this from an isolated operator host against a new, dedicated
PostgreSQL database. Keep the output directory and bundle outside the source
tree. The same `KESTREL_DATA_KEY` must protect the identity at ceremony and at
runtime.

```bash
export KESTREL_DATABASE_URL='postgresql://...'
export KESTREL_DATA_KEY='...'
export KESTREL_DID_WEB_DOMAIN='agents.kestrelsovereign.com'
export KESTREL_CEREMONY_DIR="$(mktemp -d)"

uv run python - <<'PY'
import asyncio
import os

from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.storage.async_database import AsyncDatabase

async def provision():
    db = await AsyncDatabase.postgres(os.environ["KESTREL_DATABASE_URL"])
    try:
        credentials = await create_kestrel_identity_async(
            output_dir=os.environ["KESTREL_CEREMONY_DIR"],
            database=db,
            agent_name="Kestrel Production",
            did_web_slug="kestrel",
        )
        print(credentials.agent_did)
    finally:
        await db.close()

asyncio.run(provision())
PY

# Set this to the DID printed above. The bundle command verifies the encrypted
# private keys against that DID before exporting anything.
export KESTREL_PROD_EXPECTED_DID='did:web:agents.kestrelsovereign.com:kestrel'
uv run python -m kestrel_sovereign.identity.custody_bundle create \
  --agent-dir "$KESTREL_CEREMONY_DIR" \
  --expected-did "$KESTREL_PROD_EXPECTED_DID" \
  --output "$KESTREL_CEREMONY_DIR/custody.json"
```

Upload the database URL, data key, and `custody.json` as separate Secret
Manager secrets. Grant the Cloud Run runtime service account
`roles/secretmanager.secretAccessor` only on those required secrets. Secret
Manager access is visible in Cloud Audit Logs; never print the bundle/data key
or bake either into an image. The three custody references in
`deploy_config.toml` must use immutable numeric versions such as `:7`, never
`:latest`: two instances in one revision must not resolve different keys or
bundles. Cloud Run environment values have a 32 KiB limit, which the bundle
export enforces.

After adding a new secret version, update all three numeric references and
deploy a new immutable image tag. A revision whose database, data key, bundle,
or expected DID is missing/mismatched fails startup and never re-incepts.

### Continuity and recovery check

Before shifting traffic, record the DID, both active verification methods,
the PostgreSQL agent node's `constitution_hash`, and a non-sensitive sentinel
memory. Then:

1. Start more than one instance and send concurrent requests.
2. Roll a new revision and allow the old instances to terminate.
3. Confirm every response uses the recorded DID and verification methods.
4. Confirm the constitution hash and sentinel memory are unchanged.
5. Remove custody access from a canary revision and confirm that revision
   fails readiness instead of minting a replacement DID.

Recovery means restoring the exact pinned database, identity bundle, and data
key versions, then redeploying with the recorded expected DID. If any one of
those artifacts is unavailable, stop: creating a new identity is replacement,
not recovery.

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

## Kestrel GitHub Bot — update flow

The Kestrel GitHub bot is the hosted GitHub App/webhook service that answers
project issues and discussions. It is an application surface, not a
`kestrel-feature-*` package for arbitrary agents. Its source code (and the
codebase it reads when answering questions) is baked into the Docker image at
build time.

**When you change source code, it won't reflect on the agent until you redeploy:**

```bash
set -a && source .env && set +a
uv run kestrel deploy build
uv run kestrel deploy dev
```

The agent uses `min-instances=1` to stay warm (LLM calls exceed GitHub's 10s webhook timeout, so responses are async — instance must persist after returning 200).

**Long-term:** Auto-deploy on merge to main via GitHub Actions (see #TBD).

## MultiAgent vs single-agent

The dev deployment runs **multi_agent mode** — a single `kestrel_sovereign.server:app` process co-hosts every agent in-process via `AgentManager`. See [`server.py`](../../kestrel_sovereign/server.py) for the consolidated host+agent application. (The legacy proxy host `host.py` and the `kestrel start --subprocess` launch mode were retired in #2382.)

Two agents ship in the default multi_agent image:
- `Kestrel` (port 8801) — main agent and GitHub bot webhook target
- `kestrel-demo` (port 8802) — demo agent for UI examples

The `/webhooks/github-app` endpoint is served by `server:app` and dispatched to the target agent in-process.

## Troubleshooting

### Webhook returns 401 with `{"detail": "Invalid or missing API Key"}`
The server's auth middleware is rejecting the request. Check [`server.py`](../../kestrel_sovereign/server.py) — `/webhooks/` paths must bypass API key auth.

### Webhook returns 200 but agent doesn't respond
Check the webhook delivery page on GitHub. Response body includes diagnostic info in dev mode. Common issues:
- Installation ID missing from payload → App not installed on the repo
- `no_mention` in diag → comment missing `@kestrel` trigger
- LLM error → check Cloud Run logs for the actual exception

### Logs not appearing in Cloud Run
Python `logging.info()` calls from request handlers don't always reach Cloud Run's structured log pipeline. Use `print(json.dumps({"severity": "WARNING", "message": "..."}), flush=True)` for debugging.
