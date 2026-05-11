# Kestrel Workflows Feature — Architecture Design

> Draft v4.1. Filed as epic #1131 (body to be updated when v4.1 lands). v1→v3.4 history at the bottom of the appendix; v3.4→v4 was a load-bearing reframing after a platform survey caught v3.4 designing a parallel runtime instead of composing on top of the existing Signal Dispatcher; v4→v4.1 closed two codex round-5 P2s where v4 referenced SignalDispatcher contract fields that don't exist.

## Executive Summary

Kestrel needs a first-class, multi-stage, gated, signed orchestration primitive for pipelines like FeatureFeature, sovereignty migrations, release flows, content shipping, and council reviews. Today every such pipeline reinvents orchestration ad-hoc.

**v4 reduces Workflows to orchestration on top of the existing Signal Dispatcher.** Stages dispatch Signals; runtime mechanics (durability, dedupe, locks, causation chain, retention, redaction, supervised tasks, three-mode contract) all live in the dispatcher already. Workflows adds only what the dispatcher genuinely lacks: stage sequencing, gates, saga compensation, definition versioning, and adversarial red-team review.

```
WorkflowsFeature  (orchestration: stages, edges, gates, compensation, definitions)
       │
       ▼
SignalDispatcher  (existing — durable, deduped, locked, traced, redacted, retained)
       │
       ├── COGNITION → agent turn lifecycle
       ├── ARTIFACT  → artifact_handler
       └── ACTION    → handler
```

## What Workflows uses (existing platform — does not duplicate)

| Concern | Provided by | Reference |
|---|---|---|
| Durable signal dispatch and supervised tasks | `SignalDispatcher.dispatch_signal()` / `enqueue_signal()` | [SIGNAL_DISPATCHER.md](SIGNAL_DISPATCHER.md) |
| Source registration with three modes (ACTION/ARTIFACT/COGNITION) | `SourceRegistration` in `kestrel_sdk.signals.models` | `kestrel_sdk/signals/models.py` |
| Single ordered lock manager (`CONVERSATION`, `MEMORY`, `WALLET`, `SCHEDULER`, `A2A`, `SIGNAL_LOG`) | dispatcher lock manager | SIGNAL_DISPATCHER.md §Concern 2 |
| Causation chain with TTL=5 cycle detection (propagates across A2A) | `CausationFrame` on the Signal envelope | SIGNAL_DISPATCHER.md §Concern 6 |
| Privacy-first event log: digest + redacted payload + per-source `RedactionPolicy` + `retention_days` | `signal_log` table | SIGNAL_DISPATCHER.md §Logging & Privacy |
| Trust (TRUSTED/UNTRUSTED), Visibility (INTERNAL/USER_VISIBLE/ADMIN_VISIBLE), Urgency, RateLimit, AttentionPolicy with quiet hours, coalescing windows, dedupe keys | Signal envelope + SourceRegistration | `kestrel_sdk/signals/models.py` |
| Cron triggers as `cron.<task_name>` source registrations | `signals/sources/scheduler.py:CRON_TASKS` | already shipping |
| A2A task-completion as COGNITION signal with full causation chain | `signals/sources/a2a.py` | already shipping |
| Webhook triggers (Stripe, GitHub) as COGNITION/ACTION sources with sanitizers for UNTRUSTED payloads | `signals/sources/wallet.py`, future `signals/sources/github.py` | shipping / patterned |
| Approval gates on paused turns | hook system (`HooksManager`, `ApprovalQueue`) — **not** signals | SIGNAL_DISPATCHER.md §Concern 5 |
| Constitution embedding + identity attestation | `kestrel-talon/governance/constitution.py` IdentityPackage | follow-up: dispatcher-level injection upgrade (see §6) |

If a Workflows section ever proposes adding any of the above, it's wrong; reuse instead.

## What Workflows adds (genuinely new on top of the platform)

1. **Multi-stage orchestration.** Sequencing, branching, parallel fan-out, sub-workflows. The dispatcher fires one signal at a time; no current platform component says "stage 2 fires when stage 1's gate passes."
2. **Per-stage gates beyond hooks.** Hooks gate individual tool calls (`PRE_TOOL_USE`, `POST_TOOL_USE`); workflow gates evaluate AFTER a stage's signal completes and BEFORE the next stage fires. Closed vocabulary in §3.3.
3. **Adversarial `red_team_clear` gate.** Multi-reviewer, model-family-diverse, prompt-injection-defended, budget-capped, version-pinned reviewer prompts, named adjudicator. New primitive (§3.4).
4. **Per-stage saga compensation.** On cancel/abort, run compensations in reverse order over completed stages. Includes irreversibility flag for stages whose side effects can't be reversed (§3.5).
5. **Cancellation barrier semantics.** Cancel is a barrier (not interrupt); in-flight stages complete; the next `enter`/`gate_*` checks the barrier and routes to compensation. Implemented as a `cancel` Signal with the `SCHEDULER`-style supervised-task-cancel pattern (§3.5).
6. **Workflow definitions as DID-signed versioned artifacts.** Versioned, revocable, schema-validated, with explicit `forbidden_modules` constitutional-boundary declarations (§3.6).
7. **Constitutional-boundary runtime enforcement.** Stages can declare `forbidden_modules`; the runner runs static-import-scan over emitted code; imports from forbidden modules → `gate_fail(reason=constitutional_boundary_violation)` (§3.7).

## 1. Why this exists

`kestrel-feature-features` (agent-authored features) cannot ship without this. Its workflow has eight stages, three of which are gates:

```
explore → propose → constitutional_check → file_github_epic → assign_talon_fleet
   → synthesize → tests_pass → lint_clean → ci_green
   → red_team_clear → council_approve → publish → audit_anchor
```

That's the pattern Kestrel needs everywhere — sovereignty CAR export/import, quantum-hardening waves (#921), release-signing, FeatureFeature, content shipping. Composing on top of the dispatcher means each becomes a `WorkflowDefinition`, not a code path.

## 2. Hard prerequisite (separate epic)

**Constitutional-injection upgrade for SignalDispatcher** (filed as its own epic, **#1137**). v3.4 of this design carried §3.8 "Talon-aware constitutional injection" inside Workflows; v4 splits that out because hash-verified injection, priority-ordered truncation, recursive ref-resolution, `constitution_echo_verified` reception, and `constitutional_boundary_clean` static-scan apply to **every COGNITION signal**, not just workflow stages. Workflows depends on that work landing. Until it does, Workflows COGNITION stages get whatever Talon's current `_CONSTITUTION_CACHE` injection provides — which is acceptable for v1 since the workflow gate `constitution_echo_verified` (§3.3) still acts as a defense-in-depth layer at the workflow boundary even if the dispatcher hasn't been upgraded yet.

## 3. Domain model

### 3.1 Concepts

- **WorkflowDefinition** — versioned spec: stages, edges, params schema, `forbidden_modules`, `retention_days`. Signed by author DID. Stored in Postgres/SQLite.
- **Stage** — node in the workflow graph. Has:
  - `signal_source` (must be a registered SourceRegistration)
  - `signal_mode` (must be in the source's `allowed_modes`)
  - `params` (validated against the source's schema)
  - `gate` (closed vocabulary, §3.3)
  - `compensate` (mandatory; closed eligibility for `noop_idempotent`, §3.5)
  - `forbidden_modules: list[str]` (optional, §3.7)
  - `irreversible: bool` (default False; affects compensation, §3.5)
  - `non_deterministic: bool` (declares attempt-input non-determinism for retry-key salting)
  - `read_only: bool` (default False; gates eligibility for `noop_idempotent` compensation, §3.5)
- **Per-stage prompt selection.** Workflows do NOT carry their own prompt-template overrides. The current `SignalDispatcher` renders only `registration.prompt_template`; per-signal overrides would require a contract change. If a workflow needs a different prompt for a COGNITION stage, the operator registers a new SourceRegistration with that prompt and the stage's `signal_source` points to it. (Follow-up: per-signal prompt overrides as a SignalDispatcher contract change, coordinated with #1137 — sub-issue.)
- **Edge** — typed connection: `Sequential`, `Branch(condition, true_stage, false_stage)`, `Parallel(stages[], join_strategy)`, `Subworkflow(name, version, params)`.
- **Gate** — pass/fail predicate evaluated after a stage's `SignalResult` returns. Closed vocabulary in §3.3.
- **WorkflowRun** — one execution. Tracks current stage(s), parent_run_id (subworkflows), `started_by_did`, `scheduler_task_id` (when cron-triggered), status, signed transitions.
- **StageLink** — a join row from `(run_id, stage_name, attempt_number)` to the `signal_id` the dispatcher logged for that stage's execution. Holds workflow-only fields (gate outcome, compensate state, DID signature of the transition). The signal_log row carries the redacted payload and result summary.

### 3.2 No "actor" abstraction

v3 had `Actor` types. They were not new primitives — they were Signal source registrations. Removed in v4:

| v3 actor | v4 equivalent |
|---|---|
| `agent(did)` | `signal_source = "agent.<did>"`, mode COGNITION (registers an existing or new source) |
| `talon_fleet(profile)` | `signal_source = "talon.<profile>"`, mode COGNITION via A2A; causation chain propagates |
| `ci_runner(repo, branch)` | `signal_source = "ci.<repo>"`, mode ACTION (handler does push + Actions watch) |
| `council(quorum)` | `signal_source = "council.<name>"`, mode ARTIFACT (collects N votes, returns aggregate) |
| `human` (consent) | **Hook**, not a signal — uses existing `HooksManager` + `ApprovalQueue` paused-turn pattern |
| `system` | `signal_source = "system.<task>"`, mode ACTION |

If a workflow needs a new actor, the operator registers a new SourceRegistration. Same registration boundary as every other signal source. Default-deny.

### 3.3 Built-in gate types (closed set)

| Gate | Semantics |
|---|---|
| `tests_pass(suite)` | pytest under sandbox venv, exit 0 = pass |
| `ci_green(repo, branch)` | All required GitHub Actions checks green |
| `lint_clean(scopes[])` | ruff + mypy + per-feature linters all clean |
| `red_team_clear(reviewer_pool, blockers="zero", prompt_pack_constraint, budget)` | Adversarial review, see §3.4 |
| `council_approve(quorum, timeout)` | N-of-M DID approvals via existing council feature |
| `consent_collect(scope)` | Existing `HooksManager` paused-turn approval; Workflow waits on hook completion |
| `signature_collected(did)` | Single named DID signature gathered |
| `script(language, src_hash, signature, signing_did, sandbox)` | Custom predicate via `features/compute/` sandbox; runner-`ScriptStore` resolution only, no auto-fetch; signature + permission re-check at run-start |
| `constitution_echo_verified(canary)` | Defense-in-depth wrapper over the dispatcher-level injection upgrade; actor must echo per-invocation canary derived from operative `constitution_hash` |
| `constitutional_boundary_clean(forbidden_modules[])` | Static-import-scan over the action's emitted code; imports from forbidden module → fail |
| `signal_status_ok` | Default gate: passes if `SignalResult.status == OK`. Trivial but explicit. |

**Removed in v4 from v3.x:** the `custom(callable)` arbitrary-callable hatch (the `script(...)` form replaces it as a sandboxed alternative).

### 3.4 The `red_team_clear` gate (carried over from v3.4 with no changes)

Adversarial review is non-optional for any stage that publishes code. Hardened mechanics:

**Distinctness (enforced by gate, not convention):**
- `reviewer_dids ∩ {proposer_did, definition_author_did} == ∅`
- Pool MUST include at least two distinct **model families**, verified at registration via the attestation registry (§below). Single-family panels fail closed.

**Reviewer attestation registry** (per v3.1's honest framing):
- **Hosted models:** registration includes (a) HMAC-fingerprint of API key over a fixed canary salt — binds reviewer DID to provider account — AND (b) per-invocation, the response's `model` field plus any provider-supplied response signature/headers. Honest scope: fingerprinting alone proves account ownership, not model identity. The gate also rejects responses whose `model` field doesn't match the registered family at invocation time.
- **Local models:** SHA-256 of GGUF/weight checkpoint, signed by the same DID. Genuinely strong because the runner can hash and verify.
- **Reviewer Constitution attestation:** Constitution embedding hash present and matches operator's known-good list.
- **No ambient claims.** "I am codex" without provider/weight proof rejected at registration time.
- **Open follow-up:** upgrade hosted-model attestation when major providers ship signed-response receipts. Sub-issue tracked.

**Prompt-injection defense:**
- PR content wrapped in `<<<UNTRUSTED_BEGIN ... UNTRUSTED_END>>>` fence; reviewer prompt instructs ignore-instructions-inside-fence.
- Per-invocation **canary** that the reviewer MUST echo in structured response. Missing canary → blocker.
- Responses parsed as structured JSON only.

**Blocker semantics:**
- Each reviewer reports `blockers: list[Blocker]` with `severity, location, rationale`.
- **Any reviewer reporting any blocker fails the gate.** Disagreement → blocker.
- **Adjudication:** contested blockers require a separate, distinct DID (not in the pool, not the proposer) to sign an upholding-or-dismissing record. Logged structured + signed.
- **Round cap:** 4–5 rounds (per memory `feedback_codex_round_velocity.md`). At cap with non-zero blockers → `gate_fail(reason=red_team_unconverged)`, workflow paused awaiting human signature. Never auto-pass.

**Externally version-pinned reviewer prompts:**
- Prompts live in a separate `kestrel-red-team-prompts` package with own release cadence.
- Gate definition **must** include `prompt_pack_constraint` (PEP 440 spec). Included in `spec_hash`. Run-start refuses if locally-installed pack doesn't satisfy.
- Each `red_team_clear` event records `(prompt_pack_name, prompt_pack_version, prompt_hash, constraint)` signed.

**Cost budget:**
- Gate accepts `max_total_cost_usd` and `max_total_tokens` ceilings. Operator default in `kestrel.toml`. Effective ceiling is `min(operator_default, definition_value)` when both are present, the sole configured value when only one is present, and unbounded only when neither is present.
- Operator policy is an upper bound: a workflow definition whose declared ceiling exceeds the operator default fails closed at run start instead of being silently clipped.
- Overrun → `gate_fail(reason=red_team_budget_exhausted)`.

### 3.5 Saga semantics

**Cancellation as barrier (not interrupt):**
- `workflow_cancel(run_id)` writes a `cancel_barrier` signal that downstream-of-stage logic checks before each `enter`/`gate_*` write.
- In-flight stage actions are NOT killed mid-call. They complete; their result is recorded (with a `post_cancel=True` flag in the StageLink); the engine then runs compensation.
- The `WorkflowHarness` MUST exercise the action_complete↔cancel race (test fixture in Phase 1).

**Compensation:**
- `compensate` is mandatory on every stage. Default `noop_idempotent` is a **closed eligibility set** evaluated against fields on the **Stage** (not on `SourceRegistration`, which intentionally exposes `allowed_modes`/`default_mode` only):
  - `stage.signal_mode == ACTION` AND `stage.read_only == True` (the latter is the Stage-declared field from §3.1; runner instruments the ACTION handler with a backend-portable write-counter shim and fails the gate if the handler writes any non-`workflow_*` table. Handler child tasks inherit the audit callback; late non-`workflow_*` writes after handler return are rejected before execution).
  - Stages whose gate is `consent_collect` (where rejection naturally compensates the request).
- All other stages declare a real `compensate` (a separate signal source dispatched in compensation order).
- Compensation events run reverse-order over completed stages, each as its own dispatched signal with its own idempotency key.

**Irreversible stages:**
- Declare `irreversible=True` (e.g., money cleared, audit anchor merklized to a public ledger, package published to PyPI).
- Compensation is `compensate_record_only` — engine records the cancellation but does not attempt to reverse the side effect.
- `WorkflowRun.status` becomes `cancelled_with_irreversible_residue`. Logged structured but does NOT page (per follow-up sub-issue M4 on page-fatigue tiering); only compensation *failure* pages.
- Stages downstream of an irreversible one cannot run until human acknowledgement is recorded (`signature_collected` with the operator's DID).

**Idempotency key:**
- `idempotency_key = sha256(run_id || stage_name || sha256(canonical_stage_input || attempt_number || engine_nonce))`.
- For `non_deterministic=True` stages, every retry attempt gets a fresh `attempt_number` slot so legitimate retries don't collapse.
- Engine-contributed `nonce` (not actor-contributed) prevents adversarial collision.

### 3.6 Definition versioning, revocation, signature verification

- **In-flight runs pin to definition version at start.** A run started against v3 keeps running against v3 even if v4 publishes mid-flight. No auto-promotion.
- **Definition signature re-verification on every run start.** A revoked DID's definitions fail to start; in-flight runs continue (because aborting could leave irreversible residue) with `signature_post_revocation=True` on the run row unless the revocation reason is `compromised`.
- **Typed revocation reasons** are a closed vocabulary: `compromised`, `retired`, `rotated`. Each revocation records the revoking authority DID and signature alongside `deleted_at`.
- **Revocation semantics:** `compromised` triggers `force_revoke`-with-compensation for active runs by default; `retired` allows in-flight runs to drain and refuses new starts; `rotated` also allows pinned in-flight runs to continue, while new starts must target the replacement key/version.
- **`force_revoke` admin path** for compromise scenarios; cancels all in-flight runs of the DID's definitions with full compensation. Logged with admin DID signature.

### 3.7 `forbidden_modules` runtime enforcement

Stages that emit code (typically `signal_mode=ACTION` with a code-emitting handler, or `signal_mode=COGNITION` whose response yields a patch) can declare `forbidden_modules: list[str]` (e.g., `["features/security", "features/constitution", "features/identity"]`). The runner runs a static-import-scan over the emitted source; imports from any forbidden module → `gate_fail(reason=constitutional_boundary_violation)` and compensation runs. Runtime enforcement turns doctrine into mechanism.

## 4. Tool surface (agent-callable)

| Tool | Purpose | Gating |
|---|---|---|
| `workflow_define(spec)` | Register a DID-signed workflow definition. | Sovereign-only or explicit grant |
| `workflow_run(name, params)` | Start a run; runner re-verifies definition signature first. Returns `run_id`. | Permission-checked per-definition |
| `workflow_status(run_id)` | Current stage(s), gate state, blockers, compensation state. | Read for owner/admin |
| `workflow_pause(run_id)` | Pause at current stage boundary. | Owner/admin |
| `workflow_resume(run_id)` | Resume from last checkpoint. | Owner/admin |
| `workflow_cancel(run_id)` | Set the cancel barrier; reverse-order compensation runs. | Owner/admin |
| `workflow_remediate(run_id, stage, action)` | Handle on-fail routes (retry, skip-with-justification, abort). | Sovereign or named adjudicator |
| `workflow_history(run_id)` | Full audit trail of stage transitions, joining `workflow_stage_links` and `signal_log`. | Read for owner/admin |
| `workflow_list_definitions()` | Browse registered workflows. | Open |
| `workflow_list_runs(filters)` | Browse runs. | Filtered to caller's permissions |

Workflow definitions can declare a `triggers: list[Trigger]` — Phase 3 ships `cron(expression, params)` (registers a cron task whose handler is `await workflow_run(name, params)` — exactly the existing scheduler integration pattern; see SIGNAL_DISPATCHER.md Concern #7) and `manual` (the default).

## 5. Storage schema

Massively reduced from v3.x. Workflow-specific tables only — everything else lives in `signal_log` and the existing dispatcher infrastructure.

```sql
-- Workflow definitions, versioned, signed.
CREATE TABLE workflow_definitions (
    name              TEXT NOT NULL,
    version           INT  NOT NULL,
    spec_json         JSONB NOT NULL,    -- TEXT on SQLite per StorageAdapter migration mapping
    spec_hash         TEXT NOT NULL,
    author_did        TEXT NOT NULL,
    author_sig        TEXT NOT NULL,
    retention_days    INT,                -- NULL = retain forever
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at        TIMESTAMPTZ,
    PRIMARY KEY (name, version)
);
CREATE INDEX ON workflow_definitions (author_did, created_at);
CREATE INDEX ON workflow_definitions (deleted_at) WHERE deleted_at IS NULL;

-- Runs.
CREATE TABLE workflow_runs (
    run_id                  UUID PRIMARY KEY,
    workflow_name           TEXT NOT NULL,
    workflow_ver            INT  NOT NULL,
    parent_run_id           UUID REFERENCES workflow_runs(run_id),
    params_json             JSONB NOT NULL,
    status                  TEXT NOT NULL,
        -- pending|running|paused|waiting|compensating|completed|failed
        -- |cancelled|cancelled_with_irreversible_residue
    current_stages_json     TEXT NOT NULL DEFAULT '[]',  -- JSON list
    cancel_barrier_at       TIMESTAMPTZ,
    started_by_did          TEXT NOT NULL,
    scheduler_task_id       UUID,        -- non-null when cron-triggered
    signature_post_revocation BOOLEAN NOT NULL DEFAULT FALSE,
    started_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at             TIMESTAMPTZ,
    deleted_at              TIMESTAMPTZ
);
CREATE INDEX ON workflow_runs (workflow_name, workflow_ver, status);
CREATE INDEX ON workflow_runs (status, started_at);
CREATE INDEX ON workflow_runs (parent_run_id);
CREATE INDEX ON workflow_runs (workflow_name, finished_at) WHERE deleted_at IS NULL;
CREATE INDEX ON workflow_runs (scheduler_task_id) WHERE scheduler_task_id IS NOT NULL;

-- Stage execution links: one row per (run, stage, attempt). Joins to signal_log via signal_id
-- for redacted payload + result summary; carries workflow-only fields (gate, compensate, sig).
CREATE TABLE workflow_stage_links (
    link_id           UUID PRIMARY KEY,
    run_id            UUID NOT NULL REFERENCES workflow_runs(run_id),
    stage_name        TEXT NOT NULL,
    attempt_number    INT  NOT NULL,
    signal_id         TEXT,            -- FK semantics to signal_log.id; NULL until signal dispatched
    idempotency_key   TEXT NOT NULL,   -- sha256(run_id||stage||sha256(input||attempt||nonce))
    gate_outcome      TEXT,            -- pass|fail|pending; fail carries reason
    gate_reason       TEXT,            -- e.g. red_team_unconverged, doctrine_drift, constitutional_boundary_violation
    compensate_state  TEXT,            -- not_required|pending|complete|record_only|failed
    post_cancel       BOOLEAN NOT NULL DEFAULT FALSE,
    actor_did         TEXT NOT NULL,
    actor_sig         TEXT NOT NULL,   -- DID signature over (run_id, stage_name, attempt_number, signal_id, gate_outcome)
    occurred_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, stage_name, attempt_number)
);
CREATE INDEX ON workflow_stage_links (run_id, occurred_at);
CREATE INDEX ON workflow_stage_links (signal_id) WHERE signal_id IS NOT NULL;
CREATE INDEX ON workflow_stage_links (actor_did, occurred_at);
CREATE INDEX ON workflow_stage_links (gate_outcome) WHERE gate_outcome IN ('fail', 'pending');
```

That's it. No `workflow_signals` (use `signal_log`). No worker lease/heartbeat columns (the dispatcher's supervised-task tracker handles fail/retry). No GIN indexes (signal_log already has them on Postgres). The `workflow_stage_links` row plus its referenced `signal_log` row gives an auditor the full transition trail.

**SQLite parity** uses the same StorageAdapter mapping the dispatcher already uses (`UUID`→`TEXT`, `TIMESTAMPTZ`→`TEXT` ISO-8601, `JSONB`→`TEXT` JSON-validated, `DEFAULT now()`→`DEFAULT (strftime(...))`, partial indexes supported, `PRAGMA foreign_keys = ON` at connection). No new dialect work; we use what `signal_log` already uses.

## 6. Sub-issue plan

### Phase 0 — Spec lock (days, not weeks)

The v3.x engine spike dissolves. SignalDispatcher already gives durable execution on whichever backend `KESTREL_DB_BACKEND` selected; SQLite-default is preserved natively (per memory `feedback_sqlite_non_negotiable.md`).

- [ ] `WorkflowSpec` / `Stage` / `Edge` / `Gate` / `WorkflowRun` / `StageLink` dataclasses + JSON schema
- [ ] Storage migrations (§5) using existing StorageAdapter dialect helpers
- [ ] DID signing + verification helpers (reuse existing `kestrel-talon` IdentityPackage primitives)
- [ ] Stage-to-Signal mapping spec (which `SignalMode` per gate; how `params` map to source schemas)

### Phase 1 — Foundation

- [ ] WorkflowsFeature class + tool surface (§4)
- [ ] WorkflowRunner: walks the stage graph, dispatches each stage's signal via `dispatch_signal()`/`enqueue_signal()`, evaluates gates on `SignalResult`, advances/branches/parallelizes
- [ ] Idempotency-key enforcement at `workflow_stage_links` insert
- [ ] Cancellation barrier handling + reverse-order compensation runner
- [ ] `WorkflowHarness` test fixture (deterministic clock; cassette-replay layer for non-deterministic actor types — `agent`, `talon_fleet`, `human`, `council`, `red_team_clear` reviewers; stages without `actor_replay_safe=True` fail closed in harness mode)
  - **Glossary — cassette:** the standard VCR.py / vcrpy test-fixture pattern. The first run records every reviewer/agent LLM call (request + response) into a fixture file; every subsequent run replays the recorded responses without hitting the network. Lets `red_team_clear` (and any other non-deterministic gate) run in CI deterministically without burning per-run LLM budget. See sub-issues #1138 (cassette redaction & encryption) and #1142 (CI bootstrap for first-run recording) for the security and bootstrap implications.
  - **Cassette security:** `red_team_clear` cassettes are encrypted envelopes only (`*.workflow-cassette.enc`), never plaintext JSON. The harness derives a per-owner encryption key from the operator cassette secret and run-owner DID, binds the DID + cassette id as AEAD associated data, records an expiry timestamp, and purges expired envelopes. CI replay must load the root cassette secret from a separate secret store after manual approval (`KESTREL_WORKFLOW_CASSETTE_SECRET_STORE=approved` plus `KESTREL_WORKFLOW_CASSETTE_KEY`); package repos gitignore `.workflow-cassettes/`, `tests/cassettes/`, and plaintext cassette patterns.
  - **First-cassette bootstrap runbook:** use the manual `Kestrel CI` workflow with `workflow_cassette_mode=record` from the trusted `refs/heads/main` workflow ref; normal CI jobs are skipped in this mode so only the protected bootstrap job receives secrets. Before dispatch, compute the exact approval payload with `uv run python scripts/workflow_cassette_ci.py approval-payload --repository KestrelSovereignAI/kestrel-sovereign --ref refs/heads/main --sha <sha> --owner-did <owner-did> --cassette-id <cassette-id>` and sign those bytes with the operator's legacy secp256k1 DID key. The workflow runs in the protected `workflow-cassette-recording` environment, verifies `workflow_dispatch`, the trusted ref, the environment approval flag, `KESTREL_WORKFLOW_CASSETTE_OPERATOR_PUBLIC_KEY_HEX`, and the signature before the cassette key is exposed to the encryption step; plaintext cassette contents must never be passed through `workflow_dispatch` inputs and are materialized only inside the approved environment (today via `KESTREL_WORKFLOW_CASSETTE_RECORD_PAYLOAD_B64`, later by the reviewer recorder step). Bootstrap envelopes use an explicit 90-day CI retention (`KESTREL_WORKFLOW_CASSETTE_RETENTION_SECONDS=7776000`) so committed fixtures have a predictable rotation window. Reviewer-pool credentials must stay out of the pure encryption step and may only be added to a future recorder step that actually calls reviewers after the same gate. The recorded payload is immediately encrypted through the same `EncryptedWorkflowCassetteStore` pipeline and uploaded as `workflow-cassette-recording`; operators review the artifact, commit only `*.workflow-cassette.enc`, and rotate by repeating the same gate with a new cassette id or owner DID.
- [ ] Definition revocation handler (§3.6)
- [ ] OTel spans rolling up under workflow run; Prometheus counters for stage gate_pass/gate_fail/compensate_complete/compensate_failed (signal_log already provides per-signal observability; we add only workflow-level rollups)

### Phase 2 — Gates

- [ ] `signal_status_ok` (default), `tests_pass`, `ci_green`, `lint_clean` (CI introspection via gh API)
- [ ] `council_approve`, `consent_collect` (wraps existing HooksManager paused-turn approval), `signature_collected`
- [ ] `script(language, src_hash, signature, signing_did, sandbox)` via existing compute feature with full content-addressing rules (§3.3 details preserved from v3.1)
- [ ] `red_team_clear` — full §3.4 hardening (DID-distinctness, model-family diversity via attestation registry, prompt-injection defense, structured JSON canary, any-blocker-wins + adjudication, round-cap → paused-pending-human, `kestrel-red-team-prompts` external package, `prompt_pack_constraint` required, cost budgets with `min(operator, definition)` precedence)
- [ ] `constitution_echo_verified` — defense-in-depth wrapper (works whether or not the dispatcher-level injection upgrade has landed)
- [ ] `constitutional_boundary_clean(forbidden_modules[])` — static-import-scan over emitted code

### Phase 3 — Triggers

- [ ] `manual` (default — `workflow_run` tool call)
- [ ] `cron(expression, params)` — registers a `cron.<workflow_name>` SourceRegistration whose ACTION handler is `await workflow_run(name, params)`. Reuses existing scheduler integration pattern (SIGNAL_DISPATCHER.md Concern #7).
- [ ] Signal-source-as-trigger (any registered Signal source can fire `workflow_run` as its ARTIFACT/ACTION handler payload — generalizes cron)

### Phase 4 — Ergonomics (CLI + API only — no UI in v1)

- [ ] CLI: `kestrel workflow define / run / status / list / cancel / history`
- [ ] API: `/api/workflows/*` JSON endpoints
- [ ] Workflow definitions are Python objects only in v1 (no YAML/TOML)

### Phase 5 — Pilot migrations

**Pilot 1: FeatureFeature workflows.** Active-pain consumer; exercises every gate type and trigger. FeatureFeature → Workflows is sequencing, not a cycle.

**Pilot 2: reflection cycle.** The `cron.reflect` ARTIFACT signal already runs a multi-step flow internally (gather → analyze → propose → constitutional review → apply). Lifting it to a Workflow definition with explicit stages exposes the gates and saga compensation that today are imperative inside the artifact_handler.

- [ ] FeatureFeature `feature_propose_tool` workflow definition
- [ ] FeatureFeature `feature_propose_package` workflow definition
- [ ] Reflection-cycle migration (filed as a follow-up sub-issue; second pilot, not v1 blocker)
- [ ] Other migrations (sovereignty CAR, quantum-hardening waves, content shipping) — separate follow-ups; consumers opt in when ready

### Phase 6 — Docs

- [ ] Architecture chapter (this doc, post-merge)
- [ ] Developer guide: "Defining a Kestrel Workflow" (mirrors `SIGNAL_SOURCES_GUIDE.md` shape)
- [ ] SKILL.md for the WorkflowsFeature
- [ ] Update #1047 feature-package architecture rewrite

## 7. Non-goals

- General-purpose BPM / Airflow replacement.
- Sub-stage primitives. Stages with action duration < ~100ms should use `features/tasks/` directly.
- A second cron primitive. Reuse `features/scheduler/` exclusively (see Phase 3).
- A second event log. Reuse `signal_log` exclusively (workflow-only fields go in `workflow_stage_links`).
- A second lock manager. Reuse the dispatcher's ResourceLock manager.
- A second causation chain. Reuse `CausationFrame`.
- A workflow-specific durable-execution engine (DBOS, Temporal, etc.). The dispatcher is the engine.
- Web UI in v1 (CLI/API only).
- YAML/TOML definitions in v1 (Python only).
- Bulk migration of existing flows in v1 (pilot two; rest are follow-ups).
- Arbitrary Python callables in gates (closed vocabulary + sandboxed `script(...)`).
- Workflow-level constitutional injection (split out to dispatcher-level epic — see §2).

## 8. Open questions

1. **Determining `read_only=True` automatically.** Resolved in Phase 1: backend-portable write-counter shim around the registered ACTION handler. Postgres audit triggers can still be added later for richer operator telemetry, but compensation eligibility does not depend on a Postgres-only mechanism.
2. **Where workflow definitions live.** In each consuming feature's package, or in a shared `kestrel-workflows-stdlib`? Lean: each consuming feature owns its definitions; this package only provides primitives.
3. **Reviewer attestation registry storage location.** Per-agent in `features/identity/`, or fleet-wide in Castle/kestrel-claws? Lean: schema is the same, storage layer is pluggable; sovereign solos use per-agent, managed fleets use fleet-wide.
4. **`workflow_stage_links.signal_id` foreign-key strictness.** True FK to `signal_log.id` (with cross-table delete cascade implications on retention) vs. soft reference (text column, no constraint). Lean: soft reference — `signal_log` has its own retention which can outlive or under-live workflow runs.

## 9. Risks

| Risk | Mitigation |
|---|---|
| SignalDispatcher abstraction leaks workflow-shaped concerns into source registrations | Keep `WorkflowsFeature` as the only consumer that creates workflow-shaped sources; standard sources (`agent.*`, `talon.*`, `ci.*`) stay general-purpose |
| Workflow gate evaluation latency dominates stage execution for short stages | §7 non-goal sets the ~100ms boundary; below that, use `features/tasks/` |
| Dispatcher-level constitutional injection upgrade slips | `constitution_echo_verified` workflow gate is defense-in-depth; works whether or not the upgrade has landed; v1 of Workflows ships with it active |
| Compensation on irreversible stages confuses operators (residue-not-paged vs failure-paged) | Sub-issue M4 documents tiering explicitly; alert-rule names spell it out |
| Reflection cycle migration regresses existing scheduled artifact behavior | Pilot 2 status; v1 ships without touching `cron.reflect`; migration validated against parallel run before cutover |

## 10. Related

- #375 Incubator — one workflow definition (constitutional self-modification protocol)
- #460 Runtime feature management — hot-load is a workflow
- #409 Speciation — fork-and-reconcile is a workflow
- #376 Agent Lifecycle Hardening — `consent_collect` gate via existing hook system
- #462 Open Source Core/Feature Split — feature-package architecture
- #1131 this epic
- #1137 Constitutional injection upgrade for SignalDispatcher — hard prerequisite for §3.7 + dispatcher-wide benefit; in v1 of Workflows the `constitution_echo_verified` gate is defense-in-depth at the workflow boundary so Workflows can ship before #1137 lands
- `docs/architecture/SIGNAL_DISPATCHER.md` — canonical platform reference
- `docs/architecture/SIGNAL_SOURCES_GUIDE.md` — registration walkthrough
- `kestrel-talon/governance/constitution.py` — current constitution embedding upstream

## 11. References

- Tortoise Doctrine: `docs/TORTOISE_DOCTRINE.md`
- SignalDispatcher: `docs/architecture/SIGNAL_DISPATCHER.md`
- SignalSourcesGuide: `docs/architecture/SIGNAL_SOURCES_GUIDE.md`
- SDK Signal model: `kestrel_sdk/signals/models.py`
- Existing sources: `kestrel_sovereign/signals/sources/{a2a,heartbeat,scheduler,wallet}.py`
- HooksManager + ApprovalQueue (consent gate provider): `kestrel_sovereign/hooks/`, `kestrel_sovereign/features/security/approval_queue.py`

---

## Appendix: Changelogs

### v4 → v4.1 changelog (codex round-5 P2 fixes)

Two findings, both spec bugs where v4 referenced SignalDispatcher contract fields that don't exist:

- **§3.1 Stage `prompt_template_override` removed.** SignalDispatcher renders `registration.prompt_template` only; per-signal overrides require a contract change. Stages that need a different prompt for a COGNITION source register a new SourceRegistration. Per-signal prompt-override is a follow-up coordinated with #1137.
- **§3.5 `noop_idempotent` eligibility re-grounded on Stage fields.** v4 said "`signal_source.mode == ACTION` AND `read_only=True`" but `SourceRegistration` has `allowed_modes`/`default_mode` (no `mode`) and no `read_only` field. v4.1 evaluates against `stage.signal_mode == ACTION` AND `stage.read_only == True`; added explicit `read_only: bool` field to the Stage model in §3.1.

### v3.4 → v4 changelog (platform-survey reframing)

User flagged a load-bearing miss: "we have signals besides cron jobs....i really feel like you are not considering the whole existing platform if existing cron pattern is news to you, you need to look deeper." After reading `SIGNAL_DISPATCHER.md`, `SIGNAL_SOURCES_GUIDE.md`, the SDK signal model, and the existing source registrations, it became clear v3.4 was building a parallel runtime alongside the dispatcher. Tortoise #1 (one source of truth) was being violated.

**Reframed:** Workflows is orchestration on top of SignalDispatcher. Stages dispatch signals; runtime mechanics live in the dispatcher; Workflows adds only stage sequencing, gates, saga compensation, definition versioning, and the `red_team_clear` adversarial primitive.

**Removed (was duplicating the platform):**
- §2 engine-survey (DBOS, Hatchet, Temporal, ...) — moot; the dispatcher is the engine.
- §2.3 15-invariant `WorkflowEngineAdapter` conformance contract — moot; conformance is the dispatcher's already-documented contract.
- `DBOSAdapter` / `StorageAdapter` interface and Phase 0 engine spike — dissolved.
- §3.2 `Actor` types as new primitives — they're SignalSource registrations.
- §3.7 workflow-specific OTel/Prometheus duplicating per-signal observability — kept only workflow-level rollups (gate outcomes, compensate states).
- §3.8 Talon-aware constitutional injection — **split out into a separate epic against SignalDispatcher** because hash-verified injection / priority-truncation / `constitution_echo_verified` reception apply to every COGNITION signal, not just workflow stages. Workflows depends on that work; in the meantime, the workflow gate `constitution_echo_verified` is defense-in-depth at the workflow boundary.
- §3.9 cron-trigger custom integration — collapsed into "cron is a source registration whose handler is `workflow_run(...)`," exactly the existing scheduler pattern.
- `workflow_signals` table — moot; `signal_log` is the privacy-first event log.
- `claimed_by_worker_id` / `claim_expires_at` / `last_heartbeat_at` columns — moot; supervised-task tracker handles it.
- Worker-crash-mid-claim lease semantics, p99 SQLite latency budget, stress-tests 1–4 — all dissolved into "the dispatcher already handles durable execution; workflows is orchestration."

**Schema reduced ~60%** to: `workflow_definitions` + `workflow_runs` + `workflow_stage_links` (the link rows reference `signal_log.id` for the redacted payload + result summary).

**Pivots (carried over but recast):**
- Saga compensation, irreversibility flag, cancellation barrier — workflow-specific; survive.
- Definition versioning + revocation — workflow-specific; survive.
- `red_team_clear` adversarial gate — workflow-specific; survives unchanged from v3.4.
- `script(...)` content-addressed-and-signed gate — survives unchanged.
- `forbidden_modules` runtime enforcement — survives, now §3.7.
- `constitution_echo_verified` gate — survives at workflow boundary as defense-in-depth.

**Phase plan completely re-spec'd:**
- Phase 0 dropped from "1-week engine spike with stress tests + DBOS conformance" to "days of spec lock" (dataclasses, migrations, signing helpers, stage-to-signal mapping).
- Phase 1 dropped ~60% (no adapters, no lease/heartbeat, no engine-conformance suite, OTel is just rollup).
- Phase 3 explicit triggers section — `cron` reuses scheduler exclusively; ad-hoc Signal-source-as-trigger generalizes.
- Phase 5 pilot 2 added: reflection cycle (`cron.reflect`) lifting to explicit workflow.

**Memory updates this session:**
- `feedback_survey_platform_first.md` — survey docs/architecture/*.md AND walk source tree before designing a primitive. Workflows v3.4 burned a full design cycle on parallel-implementation.
- `feedback_sqlite_non_negotiable.md` (from earlier today) — SQLite is Kestrel's default backend; "drop SQLite, mandate Postgres" is not a kill option. Carried into v4 by virtue of using the dispatcher's existing StorageAdapter dialect handling.

### v3.3 → v3.4 changelog

User caught a load-bearing wrong escape hatch: Phase 0 kill criteria, Risks table, and an Open Question all listed "drop SQLite, mandate Postgres" as fallbacks. Removed every such phrase. SQLite-default is non-negotiable for kestrel-feature-* packages. Memory `feedback_sqlite_non_negotiable.md` added.

### v3.2 → v3.3 changelog (codex round 4 P2 internal-consistency fixes)

Four self-contradictions: Phase 0 conformance suite range, Phase 2 script-gate shape, full SQLite dialect mapping, missing event-type enum entries. All fixed.

### v3.1 → v3.2 changelog (Talon integration)

Added §3.8 hash-verified constitutional injection. (In v4: split out into a separate epic against SignalDispatcher.)

### v3 → v3.1 changelog (review-3 inline fixes)

Script-store population path, attestation honesty downgrade (account ≠ model identity), behavioral invariants, sequencing dependency callout.

### v2 → v3 changelog

Conformance contract, idempotency, lease/heartbeat, attestation, cost budgets, missing migration/revocation/observability sections. (In v4: most of these dissolve since the dispatcher provides durable execution; the genuinely workflow-specific pieces — saga, versioning, red_team_clear, irreversibility — survive.)

### v1 → v2 changelog

Engine flip from DBOS-as-foundation to StorageAdapter-on-existing-storage after user verified `server.py:304` defaults to SQLite. (In v4: the entire adapter layer dissolves because the dispatcher is the engine.)
