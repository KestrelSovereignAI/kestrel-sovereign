# Kestrel Workflows Feature — Architecture Design

> Draft. Pre-filing. This document is the source the GitHub epic body is derived from.

## Executive Summary

Kestrel has every primitive *except* the composition layer that ties them into a named, versioned, gated, auditable pipeline. Tasks, Scheduler, Spawn, Talon, Council, Consent, AuditAnchor — they all exist. What's missing is **Workflow**: a first-class, durable, multi-stage, multi-actor, DID-signed pipeline with built-in gates including a non-optional **adversarial red-team review**.

Today every multi-stage agent epic (FeatureFeature, sovereignty migrations, release flows, content shipping, council reviews) reinvents orchestration ad-hoc. This is a Tortoise Doctrine #2 root-cause moment: the symptom is "every epic reinvents orchestration"; the disease is "no Workflow primitive."

**This design proposes:**

1. A `kestrel-feature-workflows` feature package (open-core feature, not in core sovereign — core stays lean).
2. **DBOS** as the durable-execution engine underneath. Postgres is already a Kestrel dependency; DBOS uses it directly as the orchestration store. No new operational surface.
3. A Kestrel domain layer on top: `Workflow`, `Stage`, `Gate`, `Actor`, with DID-signed transitions and audit-anchor integration.
4. A `red_team_clear` gate as a built-in primitive — every code-publishing stage must traverse it.
5. A workflow-engine adapter interface so DBOS can be swapped for Hatchet or Temporal without rewriting Kestrel domain code.

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

### 2.1 Survey

| Engine | Postgres-native | Python-first | Durable exec | Ops surface | License | Verdict |
|---|---|---|---|---|---|---|
| **DBOS** | **Yes — Postgres IS the engine** | Yes (decorators) | Yes (per-step checkpoints) | Existing Postgres + python lib | Apache 2.0 | **Top pick** |
| Hatchet | Yes (Postgres backend) | Yes (SDK) | Yes | Separate Hatchet engine service | MIT | Strong second |
| Temporal | Cassandra/Postgres | Yes (`temporalio`) | Best in class | Separate cluster (4+ services) | MIT | Overkill |
| Prefect 3 | Yes (Prefect Server on Postgres) | Yes | Partial — task retries, not full step replay | Prefect Server process | Apache 2.0 | Mid |
| Restate | Postgres optional | Python SDK | Yes | Separate Rust service | BSL→Apache | Promising but young |
| Inngest | Their store | Python SDK | Yes | SaaS-leaning or self-host | Apache 2.0 | Wrong shape |
| Airflow | Postgres metadata | Yes | DAG retries only | Webserver + scheduler + workers | Apache 2.0 | Wrong shape (batch/cron) |
| Dagster | Postgres metadata | Yes | Partial | Dagit + daemon | Apache 2.0 | Asset-oriented |
| Windmill | Postgres-native | Polyglot | Yes | Separate app server | AGPLv3 | License-blocked for embeddable |
| Celery | Broker not engine | Yes | No | Broker + workers | BSD | Not a workflow engine |

### 2.2 Recommendation: DBOS as engine, Kestrel domain layer on top

**Why DBOS wins:**

1. **No new operational surface.** Postgres is already there. DBOS uses it as the orchestration store directly — no separate cluster, scheduler, broker, or daemon. This is the Tortoise answer.
2. **Per-step checkpoints in Postgres.** Pause-for-days human gates are natural. Crash recovery resumes from the last completed step.
3. **Python decorators.** `@DBOS.workflow` and `@DBOS.step` map cleanly to "stages defined in code with gates between them."
4. **Apache 2.0**, compatible with Kestrel's open-core direction.
5. **Small surface to wrap.** DBOS doesn't impose a domain model — we get to define `Stage`/`Gate`/`Actor` ourselves and call DBOS only for durability primitives.

**Honest caveats:**

- DBOS is younger than Temporal. Track record is shorter.
- DBOS's primitives are workflow + step + queue. "Gate" is not a DBOS primitive; we build it as a step that evaluates a predicate and either continues or parks the workflow waiting for an external signal (`DBOS.recv` / notification).
- Long-lived human gates use DBOS's `recv`/notification primitives. Pattern is well-trodden but warrants a spike before locking in.

**Mitigation:** Workflow-engine adapter interface (Tortoise Doctrine #4 — interfaces over implementations). The Kestrel domain layer talks to a `WorkflowEngineAdapter` protocol; `DBOSAdapter` is the first implementation. If DBOS doesn't pan out, `HatchetAdapter` or `TemporalAdapter` plugs in without touching domain code.

```
WorkflowsFeature  (Kestrel domain — Stage, Gate, Actor, DID signatures, audit_anchor)
       │
       ▼
WorkflowEngineAdapter  (protocol — start, pause, resume, cancel, signal, status)
       │
       ▼
DBOSAdapter  (first impl — wraps DBOS)
       │
       ▼
Postgres  (already there)
```

---

## 3. Domain Model

### 3.1 Concepts

- **WorkflowDefinition** — a versioned spec: stages, gates, actors, on-fail routes, params schema. Signed by the author DID. Stored in Postgres.
- **Stage** — a node in the workflow graph. Has actor type, input contract, action, output contract, gate, on-fail.
- **Gate** — pass/fail predicate evaluated after a stage's action completes. Built-in types listed in §3.3.
- **Actor** — who executes the stage. Types in §3.2.
- **WorkflowRun** — one execution. Has params, current stage, signed transition history.
- **StageEvent** — every transition (start, complete, gate-pass, gate-fail, retry, abort). Each is signed by the actor's DID and anchored via `features/audit_anchor/`.

### 3.2 Actor Types

| Actor | Meaning |
|---|---|
| `human` | Pause for human action via consent UI |
| `agent(did)` | Dispatch a tool call to a specific agent |
| `talon_fleet(profile)` | Assign to a Talon agent fleet (kestrel-claws integration) |
| `ci_runner(repo, branch)` | Push branch, watch GitHub Actions |
| `council(quorum)` | Multi-DID async vote via existing council feature |
| `system` | Run in the host agent's process (for cheap stages: a script, a query) |

### 3.3 Built-in Gate Types

| Gate | Semantics |
|---|---|
| `tests_pass(suite)` | pytest under sandbox venv, exit 0 = pass |
| `ci_green(repo, branch)` | All required GitHub Actions checks green |
| `lint_clean(scopes[])` | ruff + mypy + per-feature linters all clean |
| `red_team_clear(reviewers, blockers="zero")` | **Adversarial review**, see §3.4 |
| `council_approve(quorum, timeout)` | N-of-M DID approvals via existing council feature |
| `consent_collect(scope)` | Human consent via existing consent feature |
| `signature_collected(did)` | Single named DID signature gathered |
| `custom(callable)` | Escape hatch — predicate function with declared inputs |

### 3.4 The `red_team_clear` Gate (load-bearing)

Adversarial review is **not optional** for any stage that publishes code. It is its own gate type, distinct from normal review.

**Mechanics:**

- Configurable `reviewers: list[ReviewerSpec]`. Default reviewer pool: codex CLI, claude CLI in adversarial mode, optionally a Talon agent primed with security/constitutional/privilege-escalation prompts.
- Each reviewer runs against the PR branch in the worktree.
- Reviewers report blockers. Gate passes only when **all reviewers report zero blockers**. Disagreement → blocker.
- Reviewer prompts are versioned and stored in `kestrel_feature_workflows/red_team_prompts/` so the gate's behavior is reproducible.
- Per existing memory `feedback_codex_round_velocity.md`: cap rounds at 4–5 for low-traffic surfaces; if no convergence, file a tracking ticket and escalate to human.

**Why a dedicated gate type and not `custom`:** the agent ecosystem needs to be able to *verify* that a stage was red-teamed. `red_team_clear` events carry structured metadata (reviewer DIDs, prompt versions, blocker counts, round count) that downstream auditors can query.

---

## 4. Tool Surface (Agent-Callable)

| Tool | Purpose |
|---|---|
| `workflow_define(spec)` | Register a workflow definition (version-bumped). Returns `(workflow_name, version)`. |
| `workflow_run(name, params)` | Start a run. Returns `run_id`. |
| `workflow_status(run_id)` | Current stage, gate state, blockers, history summary. |
| `workflow_pause(run_id)` | Pause at current stage boundary. |
| `workflow_resume(run_id)` | Resume from last checkpoint. |
| `workflow_cancel(run_id)` | Abort with rollback. |
| `workflow_remediate(run_id, stage, action)` | Handle on-fail routes (retry, skip-with-justification, abort). |
| `workflow_history(run_id)` | Full audit trail of stage transitions (signed). |
| `workflow_list_definitions()` | Browse registered workflows. |
| `workflow_list_runs(filters)` | Browse runs by status/actor/age. |

All tools are governed by the existing security feature's tool permissions. Workflow-mutation tools (`define`, `cancel`, `remediate`) require sovereign or explicit grant.

---

## 5. Storage Schema (Postgres)

```sql
-- Workflow definitions, versioned
CREATE TABLE workflow_definitions (
    name           TEXT NOT NULL,
    version        INT  NOT NULL,
    spec_json      JSONB NOT NULL,
    spec_hash      TEXT NOT NULL,
    author_did     TEXT NOT NULL,
    author_sig     TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (name, version)
);

-- Runs
CREATE TABLE workflow_runs (
    run_id         UUID PRIMARY KEY,
    workflow_name  TEXT NOT NULL,
    workflow_ver   INT  NOT NULL,
    params_json    JSONB NOT NULL,
    status         TEXT NOT NULL,  -- pending|running|paused|completed|failed|cancelled
    current_stage  TEXT,
    started_by_did TEXT NOT NULL,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ
);

-- Stage transitions (signed, audit-anchored)
CREATE TABLE workflow_stage_events (
    event_id       UUID PRIMARY KEY,
    run_id         UUID NOT NULL REFERENCES workflow_runs(run_id),
    stage          TEXT NOT NULL,
    event_type     TEXT NOT NULL,  -- enter|action_complete|gate_pass|gate_fail|retry|abort
    payload_json   JSONB NOT NULL,
    actor_did      TEXT NOT NULL,
    actor_sig      TEXT NOT NULL,
    anchor_id      TEXT,           -- audit_anchor reference
    occurred_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON workflow_stage_events (run_id, occurred_at);
CREATE INDEX ON workflow_runs (status, started_at);
```

DBOS owns its own internal tables (`dbos.workflow_status` etc.). The Kestrel tables above are the **domain** layer — what an auditor or operator would query. A `run_id` in our table maps 1:1 to a DBOS workflow UUID.

---

## 6. Sub-Issue Plan

### Phase 0 — Spec & Engine Spike

- [ ] DBOS-vs-Hatchet spike: prototype a 3-stage workflow with one human-gate pause/resume cycle in each. Confirm pause-for-days primitive works as expected.
- [ ] Lock in `WorkflowEngineAdapter` protocol (start, signal, recv, status, list, cancel)
- [ ] `WorkflowSpec` / `Stage` / `Gate` / `Actor` / `WorkflowRun` dataclasses + JSON schema
- [ ] Storage migrations (§5)
- [ ] DID-signed transition format

### Phase 1 — Core Engine Wrap

- [ ] `DBOSAdapter` implementation
- [ ] Workflow runner: sequential stages first, then parallel
- [ ] Gate evaluation pipeline (calls into stage's gate function after action)
- [ ] Pause / resume / cancel
- [ ] Audit anchoring per transition (wraps `features/audit_anchor/`)
- [ ] On-fail routing (retry, remediation, abort)

### Phase 2 — Built-in Gates

- [ ] `tests_pass`, `ci_green`, `lint_clean` (CI introspection via gh API)
- [ ] `council_approve`, `consent_collect`, `signature_collected` (wrap existing features)
- [ ] **`red_team_clear`** — multi-reviewer adversarial gate (§3.4)
  - [ ] Codex CLI reviewer
  - [ ] Claude CLI reviewer (fallback per `feedback_pr_review_tooling.md`)
  - [ ] Talon-agent reviewer w/ adversarial prompt
  - [ ] Round-cap enforcement per `feedback_codex_round_velocity.md`
  - [ ] Blocker schema + reviewer disagreement = blocker rule
  - [ ] Versioned reviewer prompts in repo

### Phase 3 — Built-in Actors

- [ ] `agent(did)` — dispatch tool call
- [ ] `talon_fleet(profile)` — kestrel-claws integration
- [ ] `ci_runner(repo, branch)` — push + GitHub Actions watch
- [ ] `council(quorum)` — multi-DID async vote
- [ ] `human` — consent UI pause

### Phase 4 — Ergonomics

- [ ] CLI: `kestrel workflow define / run / status / list / cancel`
- [ ] API: `/api/workflows/*` for UI consumption
- [ ] UI: workflow-run dashboard (DAG visualization, stage status, audit trail)
- [ ] Workflow definitions as YAML/TOML in `~/.kestrel/workflows/`

### Phase 5 — Migration of Existing Ad-Hoc Pipelines

- [ ] Sovereignty CAR export/import as a workflow
- [ ] Quantum-hardening waves as workflows (#921)
- [ ] Release signing flow as a workflow
- [ ] FeatureFeature's `feature_propose_*` workflows (this is the *consumer* side of the dependency)

### Phase 6 — Docs

- [ ] Architecture chapter in `docs/architecture/`
- [ ] Developer guide: "Defining a Kestrel Workflow"
- [ ] SKILL.md for the WorkflowsFeature itself
- [ ] Update #1047 feature-package architecture rewrite to reference workflows

---

## 7. Non-Goals

- General-purpose BPM/Airflow replacement.
- Sub-second workflow tasks (use `features/tasks/`).
- Cron-style triggers (use `features/scheduler/` — workflows can be *invoked* by Scheduler though).
- Cross-tenant workflow sharing (Castle/Falconer concern).
- A second durable-execution engine in core; the adapter exists so we *can* swap, not so we ship two.

---

## 8. Open Questions

1. **DBOS vs. Hatchet — final.** The Phase 0 spike answers this. Proposing to ship the spike as a 1-week box and pick the winner.
2. **Workflow definitions in Python vs. YAML/TOML.** Python is more expressive (real types, real callables), YAML/TOML is more declarative (auditable, diffable, static-analyzable). Lean: both — Python is canonical, YAML is the export/import format.
3. **Where do workflow definitions live?** In each consuming feature's package, or in a shared `kestrel-workflows-stdlib` package? Lean: each consuming feature owns its definitions; the workflows feature only provides primitives.
4. **`red_team_clear` reviewer pool.** Configurable per-workflow vs. global default. Lean: global default + per-workflow override.
5. **Pause-for-days mechanics.** DBOS `recv` works for hours; for days/weeks, do we want a separate `human_signal_store` table that survives DBOS state cleanups? Lean: yes, decouple long-lived signals from DBOS internals.
6. **DBOS version pinning.** Pin a minor at a time (Tortoise — controlled upgrades, not "latest").

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| DBOS limitations surface mid-implementation | Phase 0 spike; adapter interface |
| Pause-for-days human gates lose state | Decouple signals from DBOS `recv` (Q5) |
| Red-team gate becomes theatre (rubber-stamp reviewers) | Versioned prompts, blocker schema, reviewer-disagreement-as-blocker, round cap |
| Workflow definitions become a god-object DSL | Keep stage actions small; favor composing existing tools over new DSL primitives |
| Adapter interface leaks DBOS-isms | Code review on `WorkflowEngineAdapter` shape; second adapter (even if unused) as proof of independence |

---

## 10. Related

- #375 Incubator — the constitutional self-modification protocol; one workflow definition
- #460 Runtime feature management — hot-load is a workflow
- #409 Speciation — fork-and-reconcile is a workflow
- #1043 / consent feature — gate provider
- `features/audit_anchor/` — transition signer
- `features/council/` — multi-DID gate provider
- `feedback_pr_review_tooling.md` (memory) — codex primary, claude fallback for `red_team_clear`
- `feedback_codex_round_velocity.md` (memory) — round-cap policy for adversarial gates

## 11. References

- DBOS: https://docs.dbos.dev/
- Hatchet: https://docs.hatchet.run/
- Temporal Python SDK: https://docs.temporal.io/dev-guide/python
- Tortoise Doctrine: `docs/TORTOISE_DOCTRINE.md`
