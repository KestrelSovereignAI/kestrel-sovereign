# Kestrel Workflows Feature — Architecture Design

> Draft v2. Pre-filing. Revised after adversarial review (review-1 findings recorded in the v1→v2 diff). This is the source the GitHub epic body is derived from.

## Executive Summary

Kestrel has every primitive *except* the composition layer that ties them into a named, versioned, gated, auditable pipeline. Tasks, Scheduler, Spawn, Talon, Council, Consent, AuditAnchor — they all exist. What's missing is **Workflow**: a first-class, durable, multi-stage, multi-actor, DID-signed pipeline with built-in gates including a non-optional **adversarial red-team review**.

Today every multi-stage agent epic (FeatureFeature, sovereignty migrations, release flows, content shipping, council reviews) reinvents orchestration ad-hoc. This is a Tortoise Doctrine #2 root-cause moment: the symptom is "every epic reinvents orchestration"; the disease is "no Workflow primitive."

**This v2 design proposes:**

1. A `kestrel-feature-workflows` feature package (open-core feature, not in core sovereign — core stays lean).
2. **Build minimal durable-execution primitives on the existing Kestrel storage abstraction** (SQLite + Postgres), not on a Postgres-only third-party engine. Rationale below.
3. A `WorkflowEngineAdapter` interface, with two adapters from day one: `StorageAdapter` (default, works on every Kestrel deployment) and `DBOSAdapter` (optional Phase-2 acceleration for Postgres-backed deployments at scale).
4. A Kestrel domain layer: `Workflow`, `Stage`, `Gate`, `Actor`, `Branch`, `Parallel`, `Subworkflow`, with **mandatory per-stage compensation** (saga semantics) and DID-signed transitions.
5. A `red_team_clear` gate that is genuinely adversarial — DID-distinctness enforced, model-family diversity required, prompt-injection-defended, any-blocker-wins with named adjudicator.
6. Closed gate vocabulary — no arbitrary-callable hatch. Custom predicates run as signed scripts in the existing compute sandbox.

---

## 1. Why This Is Load-Bearing

`kestrel-feature-features` (agent-authored features) cannot ship without this. Its workflow has eight stages, three of which are gates:

```
explore → propose → constitutional_check → file_github_epic → assign_talon_fleet
   → synthesize → tests_pass → lint_clean → ci_green
   → red_team_clear → council_approve → publish → audit_anchor
```

That's not a bespoke pipeline. It's *the* pipeline pattern Kestrel needs everywhere. The Quantum Hardening epic (#921), the sovereignty CAR export/import flow, the release-signing flow — all of these are workflows in disguise. Building Workflow as a primitive means each of those becomes a definition, not a code path.

---

## 2. Engine Choice

### 2.1 Reality Check (corrected from v1)

**v1 of this design recommended DBOS on the premise that "Postgres is already a Kestrel dependency." That premise is false.** Verified:

- `kestrel_sovereign/server.py:304` — `KESTREL_DB_BACKEND = os.environ.get("KESTREL_DB_BACKEND", "sqlite")`. **SQLite is the default**; Postgres is opt-in.
- `kestrel_sovereign/features/audit_anchor/feature.py`, `features/security/permissions.py`, `features/scheduler/`, `features/tasks/` — all use `aiosqlite` directly. No engine layer abstracts them.
- `pyproject.toml` lists both `aiosqlite>=0.21.0` and `asyncpg>=0.30.0` / `psycopg2-binary>=2.9.10`, confirming Postgres is a real codepath, but not the default.

DBOS, Hatchet, Temporal, and Restate all require Postgres-or-equivalent. Adopting any of them as the foundation forces every Kestrel deployment that wants Workflows to stand up Postgres — that is a major new operational surface, not zero.

### 2.2 Survey (revised)

| Engine | Works on Kestrel default (SQLite) | Python-first | Durable exec | Ops surface | License | Verdict |
|---|---|---|---|---|---|---|
| **Build on Kestrel storage abstraction** | **Yes** | Yes | Yes (we own it) | None — uses existing DB | Apache 2.0 (ours) | **Top pick for v1** |
| DBOS | No (Postgres only) | Yes | Yes | Existing Postgres + lib | Apache 2.0 | Phase-2 optional adapter |
| Hatchet | No (Postgres only) | Yes | Yes | Separate Hatchet engine service | MIT | Rejected |
| Temporal | No | Yes (`temporalio`) | Best in class | Cluster (4+ services) | MIT | Rejected (overkill) |
| Prefect 3 | Partial | Yes | Partial | Prefect Server | Apache 2.0 | Rejected |
| Restate | No | Python SDK | Yes | Separate Rust service | BSL→Apache | Rejected (young + ops) |
| Inngest | No | Python SDK | Yes | SaaS-leaning | Apache 2.0 | Wrong shape |
| Airflow | No | Yes | DAG retries only | Webserver+scheduler+workers | Apache 2.0 | Wrong shape |
| Dagster | Postgres metadata | Yes | Partial | Dagit + daemon | Apache 2.0 | Asset-oriented |
| Windmill | Postgres-native | Polyglot | Yes | App server | AGPLv3 | License-blocked |
| Celery | Broker not engine | Yes | No | Broker + workers | BSD | Not a workflow engine |

### 2.3 Recommendation: Build on Kestrel storage; DBOS as optional Phase-2 adapter

**The minimal durable-execution primitives we need are not large:**
1. State machine: stage state in a row, transition by writing a new row.
2. Step replay on crash: at startup, find runs in `running`/`waiting` status and resume from the last completed-step row.
3. Wait-for-signal: a row another process writes; waiter wakes via Postgres `LISTEN/NOTIFY` or SQLite polling.
4. Timeout: deadline column; reaper task (existing Scheduler feature) emits timeout signals.

These primitives sit naturally on the storage abstraction Kestrel already has. Both backends supported via the existing pattern (aiosqlite for default deployments; asyncpg when `KESTREL_DB_BACKEND=postgres`).

**Why this is Tortoise-correct, not anti-Tortoise:**
- **#1 One source of truth:** existing storage layer.
- **#2 Root cause:** we don't need a separate engine to get durability — we already have a database.
- **#4 Interfaces over implementations:** `WorkflowEngineAdapter` remains. `StorageAdapter` is one impl; `DBOSAdapter` is another.
- **#5 Tech debt is debt:** zero new ops surface beats "ops surface that nine of ten deployments don't need."

**DBOS-abandonment exit plan:** the `WorkflowEngineAdapter` interface is honest because we ship two adapters from Phase 1. `DBOSAdapter` is implemented and runs the same conformance test suite as `StorageAdapter`. It is not aspirational — if it doesn't pass conformance, the interface is wrong. No "second adapter as proof of independence" theatre.

```
WorkflowsFeature  (Kestrel domain — Stage, Gate, Actor, DID signatures, audit_anchor)
       │
       ▼
WorkflowEngineAdapter  (protocol — start, signal, recv, status, list, cancel, compensate)
       ├──────────────────┐
       ▼                  ▼
StorageAdapter       DBOSAdapter   (conformance-tested twin)
   (default)          (optional)
       │                  │
       ▼                  ▼
  SQLite OR        Postgres only
  Postgres
```

### 2.4 Postgres-Lock Contention (acknowledged)

Even on Postgres, durable-execution engines that take per-step row locks degrade under fan-out. We pre-empt this in `StorageAdapter` by partitioning the events table by `(workflow_name, started_at::date)` on Postgres (no-op on SQLite) and using `SELECT FOR UPDATE SKIP LOCKED` for the run-claim path.

---

## 3. Domain Model

### 3.1 Concepts

- **WorkflowDefinition** — versioned spec: stages, edges, actors, on-fail routes, params schema. **Signed by author DID; runner refuses to execute unsigned or unrecognized-DID definitions.** Stored in Postgres/SQLite.
- **Stage** — node in the workflow graph. Has actor, input contract, action, output contract, gate, on-fail, **`compensate` (mandatory)**, `idempotency_key_template`.
- **Edge** — typed connection between stages: `Sequential`, `Branch(condition, true→stage, false→stage)`, `Parallel(fan_out=[...], join_strategy)`, `Subworkflow(name, version, params)`.
- **Gate** — pass/fail predicate evaluated after a stage's action completes. Closed vocabulary in §3.3.
- **Actor** — who executes the stage. Types in §3.2.
- **WorkflowRun** — one execution. Tracks current stage(s), parent_run_id (for subworkflows), signed transition history.
- **StageEvent** — every transition (`enter`, `action_complete`, `gate_pass`, `gate_fail`, `compensate_start`, `compensate_complete`, `retry`, `abort`). Each is signed by the actor's DID and anchored via `features/audit_anchor/`.
- **Idempotency Key** — `sha256(run_id || stage_name || attempt_input_hash)`. Required on every stage attempt; engine rejects duplicate completions for the same key.

### 3.2 Actor Types

| Actor | Meaning |
|---|---|
| `human` | Pause for human action via consent UI |
| `agent(did)` | Dispatch a tool call to a specific agent |
| `talon_fleet(profile)` | Assign to a Talon agent fleet (kestrel-claws integration) |
| `ci_runner(repo, branch)` | Push branch, watch GitHub Actions |
| `council(quorum)` | Multi-DID async vote via existing council feature |
| `system` | Run in the host agent's process (cheap stages: a script, a query) |

### 3.3 Built-in Gate Types (closed set)

| Gate | Semantics |
|---|---|
| `tests_pass(suite)` | pytest under sandbox venv, exit 0 = pass |
| `ci_green(repo, branch)` | All required GitHub Actions checks green |
| `lint_clean(scopes[])` | ruff + mypy + per-feature linters all clean |
| `red_team_clear(reviewer_pool, blockers="zero")` | Adversarial review, see §3.4 |
| `council_approve(quorum, timeout)` | N-of-M DID approvals via existing council feature |
| `consent_collect(scope)` | Human consent via existing consent feature |
| `signature_collected(did)` | Single named DID signature gathered |
| `script(language, src_hash, sandbox)` | Custom predicate executed in the existing **compute feature** sandbox; src content-addressed by hash; runs under the proposing DID's compute permissions, not the runner's |

**Removed in v2:** `custom(callable)`. An arbitrary Python callable in a workflow definition runs in the host agent's process — that's a privilege-escalation hatch (governed agent authors definition → sovereign agent runs it → arbitrary code runs in sovereign context). Replaced by `script(...)` which goes through the existing compute sandbox and inherits the *proposing* agent's compute permissions.

### 3.4 The `red_team_clear` Gate (hardened)

Adversarial review is not optional for any stage that publishes code. v1's design was rubber-stamp theatre. v2 hardens it.

**Distinctness requirements (enforced by gate, not convention):**
- `reviewer_dids ∩ {proposer_did, definition_author_did} == ∅` — gate fails if a reviewer DID overlaps the proposer or definition author.
- Reviewer pool MUST include **at least two distinct model families**. Default policy: at least one Anthropic-lineage reviewer + one non-Anthropic-lineage reviewer (OpenAI codex, local Kimi, etc.). Single-family panels fail closed.
- Reviewer DIDs are checked against an attestation registry that records the reviewer's underlying model family at registration.

**Prompt-injection defense:**
- PR content fed to reviewer LLMs is wrapped in a fixed quote-fence (`<<<UNTRUSTED_BEGIN ... UNTRUSTED_END>>>`) and the reviewer prompt instructs ignoring instructions inside that fence.
- Each reviewer prompt contains a **canary** (a fresh per-invocation token) that the reviewer MUST echo in its structured response. Missing canary → blocker, regardless of what else the response says.
- Reviewer responses are parsed as structured JSON only; free-form prose is stripped before adjudication.

**Blocker semantics:**
- Each reviewer reports `blockers: list[Blocker]` with `severity, location, rationale`.
- **Any reviewer reporting any blocker fails the gate** (not "disagreement = blocker"; that's noise-wins).
- Contested blockers (one reviewer flags, others don't) require an **adjudication** step: a separate, distinct DID (not in the reviewer pool, not the proposer) signs an adjudication record either upholding or dismissing the blocker. Adjudication is logged structured + signed.
- Round cap: per `feedback_codex_round_velocity.md`, 4–5 rounds. **At cap with non-zero blockers, gate emits `gate_fail(reason=red_team_unconverged)` and the workflow lands in `paused` awaiting human signature.** Never auto-pass.

**Reviewer prompts are versioned externally:**
- Prompts live in a separate `kestrel-red-team-prompts` package with its own release cadence (so library upgrades don't silently change adversarial-review behavior).
- Each `red_team_clear` event records `(prompt_pack_name, prompt_pack_version, prompt_hash)` and is signed.

### 3.5 Cancellation, Compensation, Saga Semantics

- **Cancellation propagates downward.** Cancelling a parent run cancels active child subworkflow runs.
- **Cancellation triggers compensation.** Every stage declares a `compensate` action. On `cancel` or `abort`, the engine runs compensations in **reverse order** of completed stages.
- **`compensate` is mandatory.** Default may be `noop_idempotent`, but only if the stage author signs an attestation that the action has no externally-visible side effects. Stages that push code, sign artifacts, write audit anchors, or move money must implement real compensation.

---

## 4. Tool Surface (Agent-Callable)

| Tool | Purpose | Gating |
|---|---|---|
| `workflow_define(spec)` | Register a (DID-signed) workflow definition. | Sovereign-only or explicit grant |
| `workflow_run(name, params)` | Start a run; runner verifies definition signature first. Returns `run_id`. | Permission-checked per-definition |
| `workflow_status(run_id)` | Current stage(s), gate state, blockers, compensation state. | Read for owner/admin |
| `workflow_pause(run_id)` | Pause at current stage boundary. | Owner/admin |
| `workflow_resume(run_id)` | Resume from last checkpoint. | Owner/admin |
| `workflow_cancel(run_id)` | Abort with reverse-order compensation. | Owner/admin |
| `workflow_remediate(run_id, stage, action)` | Handle on-fail routes (retry, skip-with-justification, abort). | Sovereign or named adjudicator |
| `workflow_history(run_id)` | Full audit trail of stage transitions (signed). | Read for owner/admin |
| `workflow_list_definitions()` | Browse registered workflows. | Open |
| `workflow_list_runs(filters)` | Browse runs. | Filtered to caller's permissions |

**Definition-signature verification:** before *every* run start, the runner re-verifies the definition's DID signature against the current trust registry. A revoked DID's definitions stop running.

---

## 5. Storage Schema

```sql
-- Workflow definitions, versioned, signed
CREATE TABLE workflow_definitions (
    name              TEXT NOT NULL,
    version           INT  NOT NULL,
    spec_json         JSONB NOT NULL,
    spec_hash         TEXT NOT NULL,
    author_did        TEXT NOT NULL,
    author_sig        TEXT NOT NULL,
    retention_days    INT,                -- NULL = retain forever
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at        TIMESTAMPTZ,        -- soft-delete (matches #763-#767 pattern)
    PRIMARY KEY (name, version)
);
CREATE INDEX ON workflow_definitions (author_did, created_at);
CREATE INDEX ON workflow_definitions (deleted_at) WHERE deleted_at IS NULL;

-- Runs
CREATE TABLE workflow_runs (
    run_id            UUID PRIMARY KEY,
    workflow_name     TEXT NOT NULL,
    workflow_ver      INT  NOT NULL,
    parent_run_id     UUID REFERENCES workflow_runs(run_id),  -- subworkflows
    params_json       JSONB NOT NULL,
    status            TEXT NOT NULL,  -- pending|running|paused|waiting|compensating|completed|failed|cancelled
    current_stages    TEXT[] NOT NULL DEFAULT '{}',  -- multiple under Parallel
    started_by_did    TEXT NOT NULL,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at       TIMESTAMPTZ,
    deleted_at        TIMESTAMPTZ
);
CREATE INDEX ON workflow_runs (workflow_name, workflow_ver, status);
CREATE INDEX ON workflow_runs (status, started_at);
CREATE INDEX ON workflow_runs (parent_run_id);
CREATE INDEX ON workflow_runs (deleted_at) WHERE deleted_at IS NULL;

-- Stage transitions (signed, audit-anchored, idempotent)
CREATE TABLE workflow_stage_events (
    event_id          UUID PRIMARY KEY,
    run_id            UUID NOT NULL REFERENCES workflow_runs(run_id),
    stage             TEXT NOT NULL,
    event_type        TEXT NOT NULL,  -- enter|action_complete|gate_pass|gate_fail|compensate_start|compensate_complete|retry|abort
    idempotency_key   TEXT NOT NULL,  -- sha256(run_id||stage||attempt_input_hash)
    payload_json      JSONB NOT NULL,
    actor_did         TEXT NOT NULL,
    actor_sig         TEXT NOT NULL,
    anchor_id         TEXT,           -- audit_anchor reference
    occurred_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, stage, idempotency_key, event_type)
);
CREATE INDEX ON workflow_stage_events (run_id, occurred_at);
CREATE INDEX ON workflow_stage_events (actor_did, occurred_at);
CREATE INDEX ON workflow_stage_events (event_type, occurred_at);
-- Postgres only:
-- CREATE INDEX ON workflow_stage_events USING GIN (payload_json);

-- Long-lived signal store (decoupled from any engine's recv internals)
CREATE TABLE workflow_signals (
    signal_id         UUID PRIMARY KEY,
    run_id            UUID NOT NULL REFERENCES workflow_runs(run_id),
    stage             TEXT NOT NULL,
    signal_name       TEXT NOT NULL,
    payload_json      JSONB NOT NULL,
    sender_did        TEXT NOT NULL,
    sender_sig        TEXT NOT NULL,
    received_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    consumed_at       TIMESTAMPTZ
);
CREATE INDEX ON workflow_signals (run_id, stage, signal_name) WHERE consumed_at IS NULL;
```

**Retention:**
- A retention cron (built on `features/scheduler/`) honors `workflow_definitions.retention_days`, soft-deleting runs and events past their TTL.
- Soft-delete pattern matches the existing project pattern (`project_soft_delete_shipped.md`, PRs #763–#767). A separate hard-delete cron purges soft-deleted rows after a grace window.

**SQLite parity:** `JSONB` becomes `TEXT` on SQLite; GIN index is Postgres-only and the StorageAdapter detects backend at startup.

---

## 6. Sub-Issue Plan

### Phase 0 — Engine Spike (strict 1-week box)

**Scope:** prove durable pause-for-days on both `StorageAdapter` (SQLite + Postgres) and `DBOSAdapter`. Nothing else.

- [ ] Build a 3-stage pause/resume prototype against `StorageAdapter` (SQLite)
- [ ] Build the same against `StorageAdapter` (Postgres)
- [ ] Build the same against `DBOSAdapter`
- [ ] Run conformance: pause for 24h, simulated process restart mid-pause, resume cleanly with signal payload intact

**Kill criteria (decided at end of week):**
- If `StorageAdapter` cannot achieve clean pause-for-24h on SQLite, we re-scope to Postgres-required and DBOS becomes primary.
- If `DBOSAdapter` fails conformance, we ship `StorageAdapter` only and drop DBOS from the roadmap.
- If both pass, both ship in Phase 1.

### Phase 1 — Foundation

- [ ] Lock `WorkflowEngineAdapter` protocol (`start`, `signal`, `recv`, `status`, `list`, `cancel`, `compensate`)
- [ ] `WorkflowSpec` / `Stage` / `Edge` / `Gate` / `Actor` / `WorkflowRun` dataclasses + JSON schema
- [ ] Storage migrations (§5)
- [ ] DID-signed transition format
- [ ] `StorageAdapter` (production-grade, both backends)
- [ ] `DBOSAdapter` (or removed per Phase-0 kill criteria)
- [ ] **Adapter conformance test suite** — both adapters run identical tests
- [ ] Idempotency-key enforcement
- [ ] Cancellation propagation + compensation runner
- [ ] OTel spans per stage event; Prometheus counters on gate_pass/gate_fail/abort/compensate
- [ ] `WorkflowHarness` test fixture (deterministic clock, mock actors, replay from event log)
- [ ] `docker-compose.workflows.yml` for local Postgres dev (SQLite needs no compose)
- [ ] Retention cron skeleton

### Phase 2 — Built-in Gates

- [ ] `tests_pass`, `ci_green`, `lint_clean` (CI introspection via gh API)
- [ ] `council_approve`, `consent_collect`, `signature_collected` (wrap existing features)
- [ ] `script(language, src_hash, sandbox)` via the existing compute feature
- [ ] **`red_team_clear`** — full §3.4 hardening
  - [ ] Reviewer DID-distinctness enforcement
  - [ ] Model-family diversity check against attestation registry
  - [ ] Quote-fenced PR content + canary token in reviewer prompts
  - [ ] Structured JSON response schema; canary echo required
  - [ ] Any-blocker-wins + adjudication step (separate signing DID)
  - [ ] Round-cap → paused-pending-human at non-zero blockers
  - [ ] External `kestrel-red-team-prompts` package with its own release cadence
  - [ ] Codex CLI reviewer / Claude CLI reviewer / Talon-agent reviewer implementations

### Phase 3 — Built-in Actors

- [ ] `agent(did)` — dispatch tool call
- [ ] `talon_fleet(profile)` — kestrel-claws integration
- [ ] `ci_runner(repo, branch)` — push + GitHub Actions watch
- [ ] `council(quorum)` — multi-DID async vote
- [ ] `human` — consent UI pause

### Phase 4 — Ergonomics (CLI + API only — no UI in v1)

- [ ] CLI: `kestrel workflow define / run / status / list / cancel`
- [ ] API: `/api/workflows/*` JSON endpoints for future UI consumption
- [ ] Workflow definitions are Python objects only in v1 (no YAML/TOML — see Open Q)

### Phase 5 — One Pilot Migration

- [ ] **Release-signing flow** as a workflow (smallest, lowest risk, exercises gates + actors fully)
- [ ] Other migrations (sovereignty CAR, quantum-hardening waves, FeatureFeature workflows) become *separate follow-up issues* so consumers opt in when ready

### Phase 6 — Docs

- [ ] Architecture chapter in `docs/architecture/` (this doc, post-merge)
- [ ] Developer guide: "Defining a Kestrel Workflow"
- [ ] SKILL.md for the WorkflowsFeature itself
- [ ] Update #1047 feature-package architecture rewrite to reference workflows

---

## 7. Non-Goals

- General-purpose BPM / Airflow replacement.
- Sub-second workflow tasks (use `features/tasks/`).
- Cron-style triggers (use `features/scheduler/` — workflows can be *invoked* by Scheduler).
- Cross-tenant workflow sharing (Castle/Falconer concern).
- Web UI in v1 (CLI/API only).
- YAML/TOML definitions in v1 (Python only).
- Bulk migration of existing flows (pilot only; rest are follow-ups).
- Arbitrary Python callables in gates (closed vocabulary + sandboxed scripts).

---

## 8. Open Questions

1. **`StorageAdapter` SQLite pause-for-days mechanics.** SQLite has no LISTEN/NOTIFY; we'll fall back to short-poll on `workflow_signals`. Acceptable for sovereign workloads (≤ a few hundred runs); revisit if scale changes.
2. **Where do workflow definitions live?** In each consuming feature's package, or in a shared `kestrel-workflows-stdlib`? Lean: each consuming feature owns its definitions; this package only provides primitives.
3. **`red_team_clear` reviewer-pool default location.** Per-workflow vs. global default. Lean: global default in `kestrel.toml`, per-workflow override. Reviewer attestation registry lives where?
4. **Compensation policy for stages with external side effects we can't fully reverse** (e.g., a money transfer that has already cleared). Lean: declared `irreversible=True` flag on the stage; cancellation past such a stage requires explicit human acknowledgement, never silently aborts.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| `StorageAdapter` durability semantics weaker than DBOS in subtle ways | Conformance test suite; DBOSAdapter as the second implementation forces the abstraction honest |
| Pause-for-days on SQLite via polling has unacceptable latency | Spike-confirmed in Phase 0; if untenable, kill SQLite support and require Postgres |
| `red_team_clear` reviewer attestation registry becomes the new single point of trust | Registry entries are themselves DID-signed; rotation policy defined at registry creation |
| Prompt-injection defense bypassed by future LLM behaviors | Canary-echo requirement is structural, not pattern-based; quote-fence + JSON-only response further narrows the attack surface |
| Adapter interface leaks engine-isms | Conformance test suite catches divergence; second adapter from day one |
| Compensation runner has bugs that leave partial state | Compensation events are themselves stage events with their own idempotency keys; replayable |

---

## 10. Related

- #375 Incubator — the constitutional self-modification protocol; one workflow definition
- #460 Runtime feature management — hot-load is a workflow
- #409 Speciation — fork-and-reconcile is a workflow
- (Verify before filing) consent feature issue number
- `features/audit_anchor/` — transition signer
- `features/council/` — multi-DID gate provider
- `features/compute/` — sandboxed script gate runner
- `feedback_pr_review_tooling.md` (memory) — codex primary, claude fallback for `red_team_clear`
- `feedback_codex_round_velocity.md` (memory) — round-cap policy for adversarial gates
- `project_soft_delete_shipped.md` (memory) — soft-delete pattern adopted in §5

---

## 11. References

- Tortoise Doctrine: `docs/TORTOISE_DOCTRINE.md`
- DBOS: https://docs.dbos.dev/
- Hatchet: https://docs.hatchet.run/
- Temporal Python SDK: https://docs.temporal.io/dev-guide/python
- Postgres `LISTEN/NOTIFY` + `SELECT FOR UPDATE SKIP LOCKED` patterns

---

## Appendix: v1 → v2 changelog

- **§2 (engine):** flipped recommendation. v1's premise that Postgres is a Kestrel default is false (`server.py:304` defaults to SQLite). v2 builds on the storage abstraction and adds DBOS as an optional Phase-2 adapter. `DBOSAdapter` ships in Phase 1 alongside `StorageAdapter` so the adapter interface is honest.
- **§3.1, §3.5:** added compensation/saga semantics. Every stage now has a mandatory `compensate`; cancellation runs compensations in reverse. Added Branch / Parallel / Subworkflow edge types and idempotency keys.
- **§3.3:** removed `custom(callable)` (privilege-escalation hatch). Replaced with `script(language, src_hash, sandbox)` running in the existing compute sandbox under proposing-DID permissions.
- **§3.4:** hardened `red_team_clear`. DID-distinctness enforcement, model-family diversity requirement, quote-fenced + canary-token prompt-injection defense, any-blocker-wins with named adjudicator, round-cap → human pause (never auto-pass), externalized reviewer-prompt versioning.
- **§4:** added definition-signature re-verification on every run start.
- **§5:** added `deleted_at` (soft-delete pattern), `retention_days`, `parent_run_id` (subworkflows), `current_stages[]` (parallel), idempotency key uniqueness, additional indexes, separate `workflow_signals` table for long-lived human-gate signals.
- **§6:** Phase 0 strictly time-boxed to engine spike with kill criteria. Phase 1 expanded to include conformance tests, OTel/Prometheus, WorkflowHarness, docker-compose. Phase 4 cut to CLI/API only (no UI). Phase 5 cut to one pilot migration. Phase 2 includes external `kestrel-red-team-prompts` package.
- **§7:** explicit non-goals for UI, YAML/TOML, bulk migrations, arbitrary callables.
