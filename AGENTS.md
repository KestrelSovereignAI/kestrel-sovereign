# Kestrel Sovereign — Agent Instructions

> **🐢 See [docs/TORTOISE_DOCTRINE.md](docs/TORTOISE_DOCTRINE.md) for the Tortoise Philosophy and coding standards.**

---

## Project Overview

Kestrel Sovereign is a Constitutional AI Agent Framework with cryptographic identity (DIDs). It provides:
- FastAPI server with agent endpoints
- Multi-LLM support (Anthropic, OpenAI, Gemini, Ollama)
- Privacy-preserving agent memory
- Constitutional protections
- GPU cloud deployment (RunPod, Vast.ai, GCP)
- Cloud Run serverless deployment (scales to zero)

## Key Directories

- `kestrel_sovereign/` - Main package
- `endpoints/` - FastAPI route handlers
- `tests/` - Test suite (pytest)
- `docker/` - Docker configurations (10 Dockerfile variants)
- `scripts/` - Utility scripts (Cloud Run build/deploy is now the `kestrel deploy` CLI; see `docs/deployment/README.md`)

## Running Tests

See the full testing guide: [`docs/architecture/testing/TESTING_GUIDE.md`](docs/architecture/testing/TESTING_GUIDE.md)

### Quick Start

```bash
# Unit tests (fast, no dependencies)
./run_tests.py --unit --skip-check

# Integration tests
./run_tests.py --integration --skip-check

# Re-run only failed tests
./run_tests.py --unit --failed

# E2E tests (requires running server)
uv run python -m kestrel_sovereign.server &
cd tests/e2e && npx playwright test
```

### Test Pyramid Strategy

Run tests in order: Unit → Integration → E2E. Fix failures before moving up.

## Kestrel Talon (GitHub Issue Processor)

Autonomous GitHub issue processing is handled by the standalone [`kestrel-talon`](https://github.com/KestrelSovereignAI/kestrel-talon) package. Installed as a dependency.

### Quick Start

```bash
# Process a specific issue
kestrel-talon claim --repo owner/repo --issue 42 --backend codex --codex-model gpt-5.5

# Multi-iteration for complex issues
kestrel-talon claim --repo owner/repo --issue 42 --backend codex --codex-model gpt-5.5 --max-iterations 5

# PRD batch mode
kestrel-talon batch --prd prd.json
```

Kestrel's in-agent Talon feature has its own runtime control surface. Use
`talon_get_config` / `talon_set_config` to manage mutable Talon preferences
such as default backend/model, while operator policy remains in
`[talon.policy]`. Talon runtime is intentionally separate from Kestrel chat
LLM routing: `backend="codex"` uses ChatGPT/Codex OAuth, `backend="claude"`
uses Claude OAuth or API billing according to `auth_lane`, and
`backend="opencode"` delegates provider selection to OpenCode config. Omit
the Codex model unless you need to pin one; Talon/Codex should otherwise use
its current default model.

### Test evidence gates (review loop)

Running tests is a **first-class evidence gate**, not just a habit. The
loop has three gates owned by three layers:

- **Implementation** — Talon runs targeted tests and reports evidence
  (commands, exit codes, CI status) before PR handoff; it rides back on
  the `talon.job_complete` signal's `test_evidence` / `ci_status` fields.
- **Review** — the Sovereign reviewer runs independent verification via
  the `talon_verify` tool (allowlisted test commands run without
  prompting; everything else is approval-gated). Result states:
  `passed` / `failed` / `blocked_by_policy` / `blocked_by_user` /
  `blocked_by_sandbox` / `tooling_error`. A sandbox/policy block is
  **never** reported as a user denial unless the approval record says so.
- **Merge** — CI is the repository merge gate.

Restart/update (RestartCoordinator) is **not** part of this — it is only
the deployment primitive. Full runbook:
[`docs/architecture/testing/TEST_EVIDENCE_GATES.md`](docs/architecture/testing/TEST_EVIDENCE_GATES.md).
Verification layer: `kestrel_sovereign/features/talon/verification.py`.

### Environment Variables

```bash
GITHUB_TOKEN=ghp_...              # GitHub PAT with repo access
ANTHROPIC_API_KEY=sk-ant-...      # Claude API key (only for auth_lane=api_key)
GITHUB_HUMAN_REVIEWER=username    # Human for blocked issues (optional)
```

## Common Tasks

### Adding a new endpoint
1. Create handler in `endpoints/`
2. Register in `server.py`
3. Add tests in `tests/`

### Modifying agent behavior
1. Check `kestrel_sovereign/agent.py`
2. Review constitutional protections in `kestrel_sovereign/constitution.py`
3. Run constitution-verifier tests

### Working with signals (anything that wakes the bird)

The bird wakes via **signals** dispatched through
`SignalDispatcher`. Heartbeat ticks, scheduled cron tasks, A2A peer
task completions, and external webhooks (Stripe deposits, etc.)
are all signals.

- **Design spec**: [`docs/architecture/SIGNAL_DISPATCHER.md`](docs/architecture/SIGNAL_DISPATCHER.md)
- **Adding a new source**: [`docs/architecture/SIGNAL_SOURCES_GUIDE.md`](docs/architecture/SIGNAL_SOURCES_GUIDE.md) (walkthrough + cycle-detection worked examples)
- **All current sources**: [`kestrel_sovereign/signals/sources/`](kestrel_sovereign/signals/sources/) — grep here to see exactly what wakes the bird
- **Hooks ≠ signals**: hooks intercept work in flight, signals originate work. The dispatcher module docstring covers the distinction loudly.

### Working with LLM providers
1. Config in the `[llm]` section of `kestrel.toml`. Legacy standalone `llm_config.toml` was retired in epic #938 — run `kestrel migrate-llm-config` to fold a legacy file in.
2. Provider implementations in `kestrel_sovereign/llm/`

### Deploying to Cloud Run
1. One-time: `uv run kestrel deploy secrets sync` (creates GCP Secret Manager entries from `.env`)
2. Build: `uv run kestrel deploy build` (builds + pushes both images to GCR)
3. Deploy: `uv run kestrel deploy dev` or `uv run kestrel deploy prod`
4. Or push a `v*` tag to trigger `.github/workflows/deploy.yml` automatically (which calls the same Python entry points)

Profiles, secrets, and env vars are configured in `deploy_config.toml`. See `docs/deployment/README.md` for the operator runbook.

---

## Authoring multi-agent Workflow scripts

When using the Workflow tool with multiple stages (typically `observation → diagnosis → synthesis`), the synthesis stage tends to drift more confident than the upstream evidence — diagnosing a bug correctly and then asserting *state* (merged, shipped, closed) that observation never reported. That's narrative smoothing, and it caused [#1484](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1484).

Rules for synthesis prompts in any workflow script:

- **Quote upstream observation fields verbatim** before asserting state. Format: ``<observation-agent> reported `<field>: <value>`. Therefore <claim>.``
- **No state claims beyond what observation returned.** If observation didn't return `merged_sha`, don't claim a merge.
- **Distinguish diagnosis from state.** Diagnosis = *what the bug is* (be confident if upstream supports it). State = *what has happened in the world* (strictly observational).
- **Don't infer state from diagnosis.** A correct-looking fix does not imply the fix has shipped.

See [`docs/development/WORKFLOW_AUTHORING.md`](docs/development/WORKFLOW_AUTHORING.md) for the full convention, the reference synthesis prompt template, and the `#1479` incident the rule was filed from.

---

## 📚 Lessons Learned (Case Studies)

### Model Selection System - What NOT to Do

The model selection system became a maintenance nightmare because quick fixes accumulated:

**What happened:**
1. Someone needed to set a model → added `set_default_model()`
2. Later, mandate system needed → added `set_model_preference()` in mixin
3. Later, simpler API needed → added ANOTHER `set_model_preference()` that shadows the first
4. Each endpoint/feature reimplemented "get current model" logic slightly differently
5. Nobody consolidated → 3 ways to set, 4 places with duplicate fallback logic

**The result:**
- User sets model via UI → calls one method
- User sets model via command → calls different method
- API endpoint checks → uses yet another path
- Actual LLM routing → checks a fourth place
- Nothing stays in sync!

**The fix (Feb 2026):**
- Consolidated to ONE source of truth: `llm_service.get_model_preference()`
- All callers now use the same method
- Removed duplicate fallback logic from 4 places
- Deleted legacy methods that used wrong APIs

**The lesson:**
Before adding a new method, SEARCH for existing ones. If one exists, EXTEND it.
If you find yourself copying logic, EXTRACT it to a shared function first.
