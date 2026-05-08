# Kestrel Workflows Feature — Architecture Design

> Draft v3.3. Pre-filing. Revised after four review rounds (v1→v2 flipped the engine; v2→v3 hardened conformance / idempotency / leases / attestation / budgets; v3→v3.1 closed round-3 inline fixes; v3.1→v3.2 added Talon-aware constitutional injection §3.8; v3.2→v3.3 closed codex round-4 internal-consistency P2s). All changelogs in the appendix.

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

#### Conformance Contract (defined before Phase 0)

The `WorkflowEngineAdapter` protocol is only honest if its invariants are written down concretely before any adapter is built. The conformance contract is a closed list of testable invariants. If DBOS's API surface cannot satisfy this list, `DBOSAdapter` is dropped — we do not relax the contract to fit DBOS, because that would make the abstraction leaky in DBOS's direction and break the swap-out story.

```python
class WorkflowEngineAdapterContract:
    # Identity & lifecycle
    INV_1_run_id_is_uuid           # start() returns a UUID; same UUID surfaces in status/list/cancel
    INV_2_idempotent_completion    # complete(stage, idempotency_key) is at-most-once per key
    INV_3_resume_after_crash       # if process dies mid-stage, restart resumes from last completed event

    # Signaling
    INV_4_signal_durable           # signal(run_id, name, payload) survives process restart
    INV_5_signal_at_least_once     # recv() blocks until matching signal; no signals dropped under planned restart
    INV_6_signal_ordering          # signals to a single (run_id, name) are observed FIFO

    # Cancellation & compensation
    INV_7_cancel_sets_barrier      # cancel(run_id) is observed by the next enter/gate_* attempt
    INV_8_compensate_in_reverse    # on cancel, compensate runs reverse-order over completed stages
    INV_9_compensate_idempotent    # compensate-key uniqueness identical to forward-stage idempotency

    # Worker semantics
    INV_10_lease_required          # claim returns (run_lease_id, expires_at); expired leases reclaimable
    INV_11_lease_heartbeat         # lease_renew(run_lease_id) extends; missed heartbeats expire the lease

    # Behavioral integrity
    INV_12_rejects_event_mutation  # adapter MUST reject any UPDATE/DELETE on a recorded stage event
    INV_13_rejects_unsigned_writes # adapter MUST reject any write whose actor signature fails verification
    INV_14_result_fidelity         # complete(stage, result); status(run_id).result_for(stage) returns byte-identical result
    INV_15_visibility_after_commit # after complete() returns, the result is visible to status() in any other process
```

Phase 0 checks this list against DBOS's documented API. If `INV_8`/`INV_9` (compensation-in-reverse with idempotent-keyed compensate) cannot be expressed in DBOS's existing primitives without adapter-side bookkeeping that subverts DBOS's own checkpointing, `DBOSAdapter` is cut. We do not paper over.

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
- **Stage** — node in the workflow graph. Has actor, input contract, action, output contract, gate, on-fail, **`compensate` (mandatory)**, `idempotency_key_template`, `forbidden_modules: list[str]` (constitutional boundary, see §3.8), `prompt_template: "claude"|"codex"|"local"` (LLM-actor stages only).
- **Edge** — typed connection between stages: `Sequential`, `Branch(condition, true→stage, false→stage)`, `Parallel(fan_out=[...], join_strategy)`, `Subworkflow(name, version, params)`.
- **Gate** — pass/fail predicate evaluated after a stage's action completes. Closed vocabulary in §3.3.
- **Actor** — who executes the stage. Types in §3.2.
- **WorkflowRun** — one execution. Tracks current stage(s), parent_run_id (for subworkflows), signed transition history.
- **StageEvent** — every transition (`enter`, `action_complete`, `gate_pass`, `gate_fail`, `compensate_start`, `compensate_complete`, `retry`, `abort`). Each is signed by the actor's DID and anchored via `features/audit_anchor/`.
- **Idempotency Key** — `sha256(run_id || stage_name || attempt_input_hash)` where:
  - `attempt_input_hash = sha256(canonical_stage_input || attempt_number || nonce)`
  - `attempt_number` increments on every retry of the same logical stage attempt; without it, a legitimate retry of an LLM stage with byte-identical input would be rejected as a duplicate.
  - `nonce` is contributed by the *engine* (not the actor) on every `enter` event so that adversarial actors cannot collide keys.
  - **Non-determinism declaration:** stages whose action is non-deterministic (`agent`, `talon_fleet`, `human`, `council`, `red_team_clear` reviewer dispatch) MUST set `non_deterministic=True` on their definition. The harness uses this to require cassette-replay in tests; the engine uses it to ensure each retry attempt gets a fresh `attempt_number` slot. Stages without the flag are presumed deterministic and may collapse retries (which is the right behavior for read-only system stages).

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
| `script(language, src_hash, signature, signing_did, sandbox)` | Custom predicate executed in the existing **compute feature** sandbox. See content-addressing rules below. |
| `constitution_echo_verified(canary)` | Actor must echo the per-invocation canary derived from the operative `constitution_hash`; missing/wrong echo → fail. See §3.8. |
| `constitutional_boundary_clean(forbidden_modules[])` | Static-import-scan over the action's emitted code; imports from any forbidden module → fail. See §3.8. |

**Removed in v2:** `custom(callable)`. An arbitrary Python callable in a workflow definition runs in the host agent's process — that's a privilege-escalation hatch (governed agent authors definition → sovereign agent runs it → arbitrary code runs in sovereign context).

**Replacement: `script(...)` content-addressing rules (v3-hardened):**

The workflow definition does NOT just reference a `src_hash`. It embeds a **content-addressed-and-signed** triple:

1. `src_hash` — sha256 of the script body.
2. `signature` — a signature over `src_hash` produced by the script-signer feature (existing pattern in `features/compute/script_signer.py`).
3. `signing_did` — the DID that produced that signature.

At every run start, before any `script` gate executes, the runner:
- Resolves `src_hash` against the **runner agent's own `ScriptStore`**, not whichever store wins a global lookup race. If not present locally, the runner refuses (it does not auto-fetch).
- Verifies `signature` against `signing_did`'s public key.
- Checks `signing_did` currently holds the `compute.script_run` permission. Permission revocation between definition-author time and run-start time fails closed.
- Records the resolved `(src_hash, signature, signing_did)` in the `script_pass` event payload.

This eliminates the substitution attack (where a malicious workflow references a hash backed by a body in someone else's store).

**Population path (closes the "where do scripts come from?" question):** scripts arrive in the runner's `ScriptStore` via the existing `compute.script_register(body, signature, signing_did)` call — same DID-signed envelope used for any other compute script. **The runner refuses any run whose required `src_hash` is not already locally registered.** There is no auto-fetch, no on-miss resolution, no fleet-wide sync. Operators that want to deploy a workflow that uses `script(...)` gates must register the script bodies on every runner first (Castle/kestrel-claws can fan-out registration; that's a Castle concern, not a Workflows one). This trades convenience for an explicit, signed propagation step.

### 3.4 The `red_team_clear` Gate (hardened)

Adversarial review is not optional for any stage that publishes code. v1's design was rubber-stamp theatre. v2 hardens it.

**Distinctness requirements (enforced by gate, not convention):**
- `reviewer_dids ∩ {proposer_did, definition_author_did} == ∅` — gate fails if a reviewer DID overlaps the proposer or definition author.
- Reviewer pool MUST include **at least two distinct model families**. Default policy: at least one Anthropic-lineage reviewer + one non-Anthropic-lineage reviewer (OpenAI codex, local Kimi, etc.). Single-family panels fail closed.
- Reviewer DIDs are checked against an attestation registry. **Self-attestation alone fails closed** — registry entries must include verifiable proof of the underlying model:
  - **Hosted models (Anthropic / OpenAI / etc.):** registration includes (a) an HMAC-fingerprint of the API key over a fixed canary salt — binds the reviewer DID to a real provider *account* — AND (b) per-invocation, the response's `model` field plus the raw provider response signature/headers (when the provider exposes one). **Honest scope:** fingerprinting alone proves account ownership, not model identity. To raise the bar, the gate also rejects responses whose `model` field doesn't match the registered family at invocation time. We acknowledge this is not cryptographic proof that "Claude isn't pretending to be codex with codex's key in its env" — full-strength model identity needs server-side attested receipts that no major provider ships today. The mitigation is layered: account-binding + per-response model-field check + the existing model-family-diversity requirement (single-family panels still fail closed).
  - **Local models (llama.cpp / vllm / etc.):** SHA-256 of the GGUF or weight checkpoint, signed by the same DID. This is genuinely strong because the runner can hash the file at registration and verify at every invocation.
  - **Reviewer Constitution attestation:** the reviewer's Constitution embedding hash must be present and match a known-good list maintained by the operator.
  - Ambient claims of "I am codex" without provider/weight proof are rejected at registration time; the gate cannot rescue a missing attestation at run time.
  - **Open follow-up:** when major providers ship signed-response receipts (Anthropic / OpenAI roadmaps), upgrade hosted-model attestation to verify the receipt. Tracked as a sub-issue.

**Prompt-injection defense:**
- PR content fed to reviewer LLMs is wrapped in a fixed quote-fence (`<<<UNTRUSTED_BEGIN ... UNTRUSTED_END>>>`) and the reviewer prompt instructs ignoring instructions inside that fence.
- Each reviewer prompt contains a **canary** (a fresh per-invocation token) that the reviewer MUST echo in its structured response. Missing canary → blocker, regardless of what else the response says.
- Reviewer responses are parsed as structured JSON only; free-form prose is stripped before adjudication.

**Blocker semantics:**
- Each reviewer reports `blockers: list[Blocker]` with `severity, location, rationale`.
- **Any reviewer reporting any blocker fails the gate** (not "disagreement = blocker"; that's noise-wins).
- Contested blockers (one reviewer flags, others don't) require an **adjudication** step: a separate, distinct DID (not in the reviewer pool, not the proposer) signs an adjudication record either upholding or dismissing the blocker. Adjudication is logged structured + signed.
- Round cap: per `feedback_codex_round_velocity.md`, 4–5 rounds. **At cap with non-zero blockers, gate emits `gate_fail(reason=red_team_unconverged)` and the workflow lands in `paused` awaiting human signature.** Never auto-pass.

**Reviewer prompts are versioned externally and constraint-pinned by the workflow definition:**
- Prompts live in a separate `kestrel-red-team-prompts` package with its own release cadence (so library upgrades don't silently change adversarial-review behavior).
- The `red_team_clear` gate definition **must** include `prompt_pack_constraint` (PEP 440 spec, e.g., `">=1.4,<2"`). The constraint is part of the workflow's `spec_hash`. At run start, if the locally-installed pack does not satisfy the constraint, the runner refuses to start the run — silent version drift is impossible.
- Each `red_team_clear` event records `(prompt_pack_name, prompt_pack_version, prompt_hash, constraint)` and is signed.

**Cost budget (v3-added):**
- The gate accepts `max_total_cost_usd` and `max_total_tokens` ceilings; the runner halts the gate (returning `gate_fail(reason=red_team_budget_exhausted)`) on overrun.
- Default ceilings live in `kestrel.toml` so operators can set fleet-wide policy without changing every workflow definition.

### 3.5 Cancellation, Compensation, Saga Semantics

- **Cancellation propagates downward.** Cancelling a parent run cancels active child subworkflow runs.
- **Cancellation is a barrier, not an interrupt.** `cancel(run_id)` writes a barrier event. The next `enter` or `gate_*` write checks the barrier and routes to `compensate_start` instead of proceeding. In-flight stage actions are NOT killed mid-call (we don't know if they're idempotent on partial completion); they complete, their result is recorded as `action_complete_post_cancel`, and the engine then runs compensation. The race between `action_complete` and `cancel` is therefore well-defined: the action either completed before the barrier (gets compensated) or after (its compensate runs immediately on next-event). `WorkflowHarness` MUST exercise this race.
- **Cancellation triggers compensation.** On `cancel` or `abort`, the engine runs compensations in **reverse order** of completed stages.
- **`compensate` is mandatory.** Default `noop_idempotent` is only available to a **closed eligibility set**:
  - `actor=system` AND `read_only=True` (declared in the definition; runner verifies the stage made no DB writes outside `workflow_*` tables — instrumentation in Phase 1).
  - `actor=human` consent stages where rejection naturally compensates the request.
  - All other stages MUST declare a real `compensate` action. Self-attestation by the author is **not** sufficient; the eligibility check is structural, not based on the author's swear.
- **Irreversible stages.** A stage may declare `irreversible=True` (e.g., a money transfer that has cleared, an audit anchor written to a public ledger). Compensation for an irreversible stage is `compensate_record_only` — the engine records the cancellation but does not attempt to reverse the side effect; the `workflow_run` enters `cancelled_with_irreversible_residue` status and pages an operator. The next stage past an irreversible one cannot run until human acknowledgement is recorded.

### 3.6 Definition Versioning & Revocation

- **In-flight runs pin to definition version at start.** A run started against `(name, ver=3)` keeps running against v3 even if v4 is published mid-flight. There is no auto-promotion. If a definition author wants in-flight runs to switch, they must explicitly `workflow_run` with the new version.
- **Definition signature revocation.** When an author DID is revoked (key rotation, ejection, etc.), the runner refuses to *start* new runs of that DID's definitions. **In-flight runs continue under the previously-valid signature snapshot**, because aborting them mid-flight could leave irreversible-stage residue without proper human acknowledgement. The status surface flags such runs (`signature_post_revocation=True`) so operators can choose to cancel-with-compensation explicitly.
- **Hard revocation.** For compromise scenarios, a `force_revoke` admin action cancels all in-flight runs of the DID's definitions and runs compensation. This is logged with the admin's DID signature and is not a routine path.

### 3.7 Observability

- **Per-event OTel spans** with `run_id`, `workflow_name`, `workflow_ver`, `stage`, `actor_did` as span attributes. Stage durations roll up under the parent workflow span.
- **Prometheus counters:** `workflow_runs_started_total`, `workflow_stage_events_total{event_type}`, `workflow_gate_fail_total{gate_type}`, `workflow_compensate_total{outcome}`, `workflow_lease_expired_total`, `workflow_cancel_total`.
- **Prometheus gauges:** `workflow_runs_in_status{status}`, `workflow_active_leases`, `workflow_runs_waiting_age_seconds_p99`.
- **Built-in alert rules** (shipped as part of the package's Prometheus rules file):
  - `workflow_run_waiting_too_long` — fires when a run sits in `waiting` for > N hours (default 72h, per-workflow override)
  - `workflow_lease_expired_run_unclaimed` — a lease expired and no worker has reclaimed it within the heartbeat-interval × 3
  - `workflow_red_team_budget_exhausted_rate` — `gate_fail{reason=red_team_budget_exhausted}` rate > expected (catches runaway-cost workflows)
  - `workflow_compensate_failed_total > 0` — compensation itself failed; manual intervention required
- **Stuck-run dashboard** in Phase 4 ergonomics — read-only, surfaces all of the above.

### 3.8 Constitutional Injection & Actor Binding (Talon-aware)

Every stage that dispatches to an LLM-backed actor (`agent`, `talon_fleet`, `human` consent UIs that show prompts, `red_team_clear` reviewers) MUST inject the operative Kestrel Constitution + doctrine into that actor's system prompt and verify the actor received it. Today Talon does part of this; it has gaps the workflow runner closes.

**What Talon does today (and where it stops):**

| Talon path | What works | Where it falls short |
|---|---|---|
| `governance/constitution.py` | SHA-256 of canonical text; binds `DID + constitution_hash + timestamp` via `IdentityPackage` signature | `_CONSTITUTION_CACHE` is module-import-scoped; mid-process file changes go undetected |
| `governance/data/KESTREL_CONSTITUTION.md` | Single canonical source | No verification that the *injected* text's hash matches the *signed* IdentityPackage's hash |
| `context.py` `discover_context_files` + `_resolve_file_references` | Auto-loads AGENTS.md / CLAUDE.md / TORTOISE_DOCTRINE.md (via abs-path follow), one level deep | One level only; refs inside refs don't recurse; no priority ordering |
| `context.py` `MAX_FILE_SIZE = 50_000` | Bounds prompt growth | Truncates with no awareness of clause importance — a long doctrine can lose its tail clause silently |
| `agent.py:_build_system_prompt` | `SystemPromptPreset(type=preset, preset=claude_code, append=context)` cleanly separates injection from task | Hardcoded `claude_code` preset; can't target codex / local-model reviewers with the same constitution |
| Per-issue prompt | Encodes `TALON_COMPLETE` / `TALON_BLOCKED` / `LEARNED` protocol | Bakes Talon's completion semantics; a workflow that needs different signals has to fork or live with these |
| Doctrine as text | Advisory | Zero runtime enforcement — the agent's *judgment* is the only barrier between it and `features/security/` |

**What Workflows adds (closes the gaps):**

1. **Hash-verified injection.** When a stage with an LLM-actor `enter`s, the runner:
   - Loads constitution text fresh (no module-cache); hashes it.
   - Loads the doctrine bundle (TORTOISE_DOCTRINE.md, AGENTS.md, project supplements) fresh; hashes the concatenation as `doctrine_bundle_hash`.
   - Compares `(constitution_hash, doctrine_bundle_hash)` against the values recorded in the workflow definition's signature envelope. **Mismatch fails the stage** with `gate_fail(reason=doctrine_drift)` — doctrine cannot drift mid-run without an explicit definition re-sign.
   - Records both hashes in the `enter` event payload, signed by the runner's DID.

2. **Priority-ordered truncation.** The injection loader uses an explicit clause-priority order: `KESTREL_CONSTITUTION` > `TORTOISE_DOCTRINE` > `project AGENTS.md` > supplements > general README. If total size exceeds the actor's prompt budget, lower-priority clauses are dropped first. The `enter` event records `(injected_clauses[], dropped_clauses[], total_bytes_injected)` so audits can see what got cut.

3. **Constitution echo (receipt verification).** New gate type `constitution_echo_verified` (closed-set addition to §3.3). The injection includes a fresh canary token derived from `sha256(run_id || stage_name || constitution_hash || nonce)`. The actor's first structured response MUST echo the canary verbatim. Missing canary → `gate_fail(reason=constitution_not_received)`, regardless of any other passing signals.

4. **Recursive reference resolution (bounded).** Replaces Talon's one-level recursion with N-level bounded resolution (default N=3) plus a cycle detector. If the bundle pulls in more files than declared, the runner records the extras in the `enter` event so an auditor can see what doctrine the actor actually saw.

5. **Multi-actor prompt templates.** `Stage.prompt_template` is keyed by model family, not hardcoded to Claude. Defaults: `claude` (uses SystemPromptPreset like Talon does today), `codex` (uses codex's system-prompt format), `local` (raw prepend). The injection content is the same; only the wrapping differs.

6. **Runtime enforcement of constitutional boundaries.** Stages can declare `forbidden_modules: list[str]` (e.g., `["features/security", "features/constitution", "features/identity"]`). For code-touching actions, the runner runs a static-import-scan on any source the action emitted; if the diff imports from a forbidden module, the stage's `gate_pass` fails with `gate_fail(reason=constitutional_boundary_violation)` and compensation runs. This is the runtime enforcement Talon lacks: doctrine becomes mechanism, not a swear-on.

7. **Reuse, don't duplicate.** Talon's `_build_system_prompt` and `discover_context_files` are kept and called *through* a new `kestrel_sovereign.workflows.context.WorkflowStageContext` adapter — Talon's existing entry points stay valid for legacy callers and become consumers of the same loader the runner uses. We do not fork Talon. (See follow-up sub-issue: "Talon integration upgrade — replace `_CONSTITUTION_CACHE` with WorkflowStageContext-backed loader.")

**Where this lives in the schema:**

`workflow_stage_events.payload_json` for `enter` events carries:

```json
{
  "actor_did": "...",
  "constitution_hash": "...",
  "doctrine_bundle_hash": "...",
  "injected_clauses": ["KESTREL_CONSTITUTION", "TORTOISE_DOCTRINE", "AGENTS.md"],
  "dropped_clauses": ["docs/research/long-supplement.md"],
  "total_bytes_injected": 24531,
  "prompt_template": "claude",
  "echo_canary_expected": "...",
  "forbidden_modules": ["features/security", "features/constitution"]
}
```

The `gate_pass` event for a `constitution_echo_verified` gate carries `echo_canary_received` (must equal expected) plus the actor's signed acknowledgement.

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
    run_id                  UUID PRIMARY KEY,
    workflow_name           TEXT NOT NULL,
    workflow_ver            INT  NOT NULL,
    parent_run_id           UUID REFERENCES workflow_runs(run_id),  -- subworkflows
    params_json             JSONB NOT NULL,
    status                  TEXT NOT NULL,
        -- pending|running|paused|waiting|compensating|completed|failed
        -- |cancelled|cancelled_with_irreversible_residue
    current_stages_json     TEXT NOT NULL DEFAULT '[]',
        -- JSON array; SQLite has no array type, Postgres uses JSONB cast
    cancel_barrier_at       TIMESTAMPTZ,           -- non-null = barrier set
    -- Worker lease (no engine has live worker if these are null/expired):
    claimed_by_worker_id    TEXT,
    claim_expires_at        TIMESTAMPTZ,
    last_heartbeat_at       TIMESTAMPTZ,
    started_by_did          TEXT NOT NULL,
    started_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at             TIMESTAMPTZ,
    deleted_at              TIMESTAMPTZ
);
CREATE INDEX ON workflow_runs (workflow_name, workflow_ver, status);
CREATE INDEX ON workflow_runs (status, started_at);
CREATE INDEX ON workflow_runs (parent_run_id);
CREATE INDEX ON workflow_runs (claim_expires_at) WHERE status IN ('running', 'compensating');
CREATE INDEX ON workflow_runs (workflow_name, finished_at) WHERE deleted_at IS NULL;
CREATE INDEX ON workflow_runs (deleted_at) WHERE deleted_at IS NULL;

-- Stage transitions (signed, audit-anchored, idempotent)
CREATE TABLE workflow_stage_events (
    event_id          UUID PRIMARY KEY,
    run_id            UUID NOT NULL REFERENCES workflow_runs(run_id),
    stage             TEXT NOT NULL,
    event_type        TEXT NOT NULL,
        -- enter|action_complete|action_complete_post_cancel|gate_pass|gate_fail
        -- |compensate_start|compensate_complete|retry|abort|cancel_barrier_set
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
-- Find stages currently blocked on a specific human DID (auditor query):
CREATE INDEX ON workflow_stage_events (actor_did, event_type, occurred_at)
    WHERE event_type = 'enter';
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

**SQLite parity (full mapping):** the schema above is written in Postgres dialect for readability; the `StorageAdapter`'s migration emitter produces backend-specific DDL using the following substitutions on SQLite. Migrations that depend only on Postgres types fail loudly at SQLite startup with a missing-mapping error rather than silently truncating.

| Postgres | SQLite | Notes |
|---|---|---|
| `UUID` | `TEXT` | Application stores canonical 36-char UUID strings; lexicographic ordering is fine for indexes. |
| `TIMESTAMPTZ` | `TEXT` | ISO-8601 with explicit `Z` suffix; `datetime(... )` SQLite functions handle comparisons. |
| `DEFAULT now()` | `DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))` | Same wall-clock semantics. |
| `JSONB` | `TEXT` | JSON-validated at write time by the adapter; queries use `json_extract` on SQLite, `jsonb_path_query` on Postgres. |
| `TEXT[]` (n/a — replaced by `current_stages_json`) | `TEXT` (JSON array) | Already encoded as JSON in v3 schema for portability. |
| Partial indexes (`WHERE foo IS NULL`) | Same syntax | SQLite supports partial indexes since 3.8.0. |
| GIN on JSONB | (none — Postgres-only) | StorageAdapter skips on SQLite; falls back to in-app filtering for payload introspection queries. |
| `REFERENCES … ON DELETE` | Same | Both backends honor; SQLite needs `PRAGMA foreign_keys = ON` set at connection. |

---

## 6. Sub-Issue Plan

### Phase 0 — Engine Spike (strict 1-week box)

**Pre-spike (day 0):** write the §2.3 Conformance Contract as a runnable test suite (`test_adapter_conformance.py`). The spike is testing against this suite, not vibes.

**Scope:** prove the conformance contract holds for `StorageAdapter` (SQLite + Postgres) and `DBOSAdapter`. Nothing else.

- [ ] Conformance suite implements all `INV_1`..`INV_15` invariants as parameterized tests (including `INV_14_result_fidelity` and `INV_15_visibility_after_commit`)
- [ ] `StorageAdapter` (SQLite): suite passes
- [ ] `StorageAdapter` (Postgres): suite passes
- [ ] `DBOSAdapter`: suite passes — or, if it can't, document precisely which invariants DBOS cannot satisfy and why
- [ ] **Stress test 1 — pause durability:** pause for 24h with simulated process restart mid-pause; signal payload intact on resume
- [ ] **Stress test 2 — fan-out under WAL contention (SQLite):** 30 parallel subworkflows, 50 stage_events/sec each, no deadlocks, no signal loss
- [ ] **Stress test 3 — listener death:** kill the sole listener process between poll and consume; verify lease expires and a new listener resumes the run cleanly
- [ ] **Stress test 4 — claim-and-crash:** worker claims a run, crashes mid-stage; lease expires; second worker reclaims; only one wins
- [ ] **Latency budget:** p99 signal-arrival-to-wakeup latency ≤ 2s for SQLite poll-based, ≤ 200ms for Postgres LISTEN/NOTIFY. If unmet, downgrade SQLite to a documented "small-deployment" tier.
- [ ] **Concurrent-runs ceiling:** measure throughput; document the per-backend ceiling above which the operator gets a startup warning

**Kill criteria (decided at end of week):**
- If `StorageAdapter` cannot pass conformance + Stress 1–4 on SQLite, SQLite support is dropped, the Workflows feature mandates Postgres, and we revisit DBOS-as-primary.
- If `DBOSAdapter` cannot pass the conformance contract, it's cut from Phase 1 (no relaxing the contract to fit DBOS — that breaks the swap-out story).
- If both pass on Postgres but SQLite-StorageAdapter only passes a subset, we ship SQLite as a documented "small-deployment" tier with explicit limits.

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
- [ ] Worker lease + heartbeat (claim_expires_at, last_heartbeat_at; reaper expires stale claims)
- [ ] Cancellation barrier semantics (§3.5 race contract); harness exercises action_complete↔cancel race
- [ ] OTel spans per stage event; Prometheus counters/gauges per §3.7
- [ ] Built-in alert rules file (§3.7) shipped with the package
- [ ] `WorkflowHarness` test fixture:
  - Deterministic clock
  - Cassette-replay layer for non-deterministic actor types (`agent`, `talon_fleet`, `human`, `council`, `red_team_clear` reviewers)
  - Stages without `actor_replay_safe=True` (or `non_deterministic=True` + cassette) **fail closed in harness mode** — the harness refuses to call out to the network silently
- [ ] `docker-compose.workflows.yml` for local Postgres dev (SQLite needs no compose)
- [ ] Retention cron skeleton (uses `features/scheduler/`)
- [ ] Definition revocation handler (§3.6 — refuses new runs of revoked DIDs; flags in-flight)

### Phase 2 — Built-in Gates

- [ ] `tests_pass`, `ci_green`, `lint_clean` (CI introspection via gh API)
- [ ] `council_approve`, `consent_collect`, `signature_collected` (wrap existing features)
- [ ] `script(language, src_hash, signature, signing_did, sandbox)` via the existing compute feature, with the full §3.3 content-addressing rules: runner resolves only against its own `ScriptStore`, refuses auto-fetch, verifies signature, re-checks `compute.script_run` permission for `signing_did` at run-start
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

**Pilot target (revised):** the `feature_propose_tool` workflow from `kestrel-feature-features` (the consumer that motivated this whole epic). Release-signing was the v2 pilot suggestion but is already shipped (`project_release_signing_key.md`, 2026-05-06) — migrating a working flow with no active pain is scope creep.

**Dependency direction (not circular):** `kestrel-feature-features` depends on `kestrel-feature-workflows`, not the other way around. Workflows ships first as a primitive package; FeatureFeature consumes it. This is sequencing, not a cycle.

The FeatureFeature pipeline is the right pilot because:
- It has every gate type we need (`tests_pass`, `lint_clean`, `ci_green`, `red_team_clear`, `council_approve`)
- It has every actor type (`agent`, `talon_fleet`, `ci_runner`, `council`, `human`)
- It has irreversible stages (publish + audit_anchor) — exercises §3.5 irreversibility semantics
- It has compensation requirements (rollback an editable install) — exercises saga properly
- It is the consumer blocking on this work; pilot ships value, not just proof

- [ ] FeatureFeature `feature_propose_tool` workflow definition
- [ ] FeatureFeature `feature_propose_package` workflow definition (Phase-3 tier)
- [ ] Migration of in-flight FeatureFeature runs (if any) — N/A in v1, but document the path
- [ ] Other migrations (sovereignty CAR, quantum-hardening waves, content shipping) become *separate follow-up issues* so consumers opt in when ready

### Phase 6 — Docs

- [ ] Architecture chapter in `docs/architecture/` (this doc, post-merge)
- [ ] Developer guide: "Defining a Kestrel Workflow"
- [ ] SKILL.md for the WorkflowsFeature itself
- [ ] Update #1047 feature-package architecture rewrite to reference workflows

---

## 7. Non-Goals

- General-purpose BPM / Airflow replacement.
- Sub-stage primitives. Stages with action duration < ~100ms should use `features/tasks/` directly; below that threshold, workflow overhead (signing, audit anchor, idempotency-key insert) dominates the action itself.
- Cron-style triggers (use `features/scheduler/` — workflows can be *invoked* by Scheduler).
- Cross-tenant workflow sharing (Castle/Falconer concern).
- Web UI in v1 (CLI/API only).
- YAML/TOML definitions in v1 (Python only).
- Bulk migration of existing flows (pilot only; rest are follow-ups).
- Arbitrary Python callables in gates (closed vocabulary + sandboxed scripts).

---

## 8. Open Questions

1. **`StorageAdapter` SQLite latency under sustained fan-out.** Phase-0 stress tests resolve the kill-or-tier question. The "≤ a few hundred runs" claim is replaced by whatever the spike measures; if SQLite cannot meet the documented latency budget, it ships as a "small-deployment tier" with hard ceilings or is dropped.
2. **Where do workflow definitions live?** In each consuming feature's package, or in a shared `kestrel-workflows-stdlib`? Lean: each consuming feature owns its definitions; this package only provides primitives. Resolved: lean is the answer unless a v1 consumer pushes back.
3. **Reviewer attestation registry storage.** Two viable locations: (a) per-agent in the existing `features/identity/` attestation tables; (b) a fleet-wide registry in Castle/kestrel-claws. Lean: (a) for sovereign solos, (b) for managed fleets — the schema is the same, the storage layer is pluggable.
4. **Determining `read_only=True` automatically.** §3.5 says the runner instruments stages declared `read_only=True` to verify they made no DB writes outside `workflow_*` tables. Mechanism is open: trigger-based (Postgres) vs. write-counter shim (both backends). Resolve in Phase 1.

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
- #376 Agent Lifecycle Hardening (Graduation/Consent/Cryostasis epic) — `consent_collect` gate provider
- `kestrel-talon` repo (`/Volumes/data2/projects/kestrel-talon`) — `governance/constitution.py`, `governance/data/KESTREL_CONSTITUTION.md`, `context.py` (auto-discovery + abs-path follow), `agent.py:_build_system_prompt` — current upstream of constitution/doctrine injection. §3.8 reuses these and closes their gaps.
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

## Appendix: Changelogs

### Follow-up sub-issues (filed alongside the epic)

Round-3 review verdict was FILE; the following residual findings became sub-issues rather than blocking the epic body:

- **H2 — Cassette redaction & access control for `red_team_clear`.** Cassette files contain PR diff + adversarial reviewer findings. Need: encrypted-at-rest under run-owner DID, never committed to package repo, separate secret-store CI access. Sub-issue.
- **H3 — Cost-budget precedence rule.** Spec the `min(operator_ceiling, definition_ceiling)` rule and document that operator policy is an upper bound. Sub-issue.
- **H4 — `cancel` cost / `force_abort` primitive.** Per-actor-type timeouts; separate `force_abort` primitive for genuine emergencies (no compensation guarantees). Sub-issue.
- **M1 — Typed revocation reasons.** `compromised` triggers `force_revoke`; `retired` allows in-flight to drain. Sub-issue.
- **M2 — CI bootstrap for cassette recording.** Manual-approval-gated record mode for the very-first cassette set. Sub-issue.
- **M4 — Page-fatigue tiering.** `cancelled_with_irreversible_residue` from `compensate_record_only` is logged-not-paged; only `compensate_failed_total > 0` pages. Sub-issue.
- **L3 — `read_only=True` mechanism.** Trigger-based vs. write-counter shim; closed in Phase 1. Promote to sub-issue.
- **C2 follow-up — Provider signed-response receipts.** Upgrade hosted-model attestation when major providers ship them. Sub-issue.
- **Talon integration upgrade.** Replace Talon's module-scoped `_CONSTITUTION_CACHE` with a `WorkflowStageContext`-backed loader so Talon and the workflow runner share one constitution-injection path; backport hash-verified injection, priority-ordered truncation, recursive ref-resolution, and `constitution_echo_verified` to standalone Talon runs. Sub-issue.

### v3.2 → v3.3 changelog (codex round 4 P2 internal-consistency fixes)

Codex CLI review caught four self-contradictions; all fixed inline:

- **§6 Phase 0 conformance suite:** updated from `INV_1..INV_13` to `INV_1..INV_15` so v3.1's added `INV_14_result_fidelity` and `INV_15_visibility_after_commit` ship tested.
- **§6 Phase 2 script gate:** updated checklist from `script(language, src_hash, sandbox)` to the full `script(language, src_hash, signature, signing_did, sandbox)` with the §3.3 content-addressing rules spelled out, so the v3.1 hardening doesn't get lost in implementation.
- **§5 SQLite parity:** expanded from a one-line `JSONB→TEXT` note to a full mapping table covering `UUID`, `TIMESTAMPTZ`, `DEFAULT now()`, partial indexes, GIN, foreign keys. StorageAdapter migration emitter fails loudly on missing mappings rather than silently truncating.
- **§5 event-type enum:** added `action_complete_post_cancel` (referenced by §3.5 cancellation barrier) and `cancel_barrier_set` so the well-defined cancel/action_complete race can be represented in the audit trail.

Codex verdict: P2-only round (no P0/P1, no architectural concerns). Per `feedback_codex_round_velocity.md`, this is convergence — round 4 found internal consistency bugs only, which is the highest-leverage final pass before filing.

### v3.1 → v3.2 changelog (Talon integration)

- **§3.8 NEW Constitutional Injection & Actor Binding (Talon-aware):** mapped Talon's current path (governance/constitution.py + context.py + agent.py:_build_system_prompt), enumerated where it falls short (module-cache staleness, no injected-vs-signed hash check, naive truncation, hardcoded claude_code preset, no echo verification, no recursion past one level, doctrine as advisory text only), and specified what Workflows adds: hash-verified injection (mismatch → gate_fail with reason=doctrine_drift), priority-ordered truncation with audit trail, `constitution_echo_verified` gate (canary token in injection, actor must echo or fail), bounded N-level recursive reference resolution, multi-actor prompt templates (claude/codex/local), runtime constitutional-boundary enforcement via static-import-scan against `forbidden_modules`. Reuse-not-duplicate: Talon's existing entry points become consumers of a shared `WorkflowStageContext` loader.
- **§3.1 Stage:** added `forbidden_modules: list[str]` and `prompt_template` fields.
- **§3.3 Gates:** added `constitution_echo_verified` and `constitutional_boundary_clean` to the closed gate vocabulary.
- **§5 Schema:** `enter` event payload schema now includes `constitution_hash`, `doctrine_bundle_hash`, `injected_clauses[]`, `dropped_clauses[]`, `total_bytes_injected`, `prompt_template`, `echo_canary_expected`, `forbidden_modules[]`.
- **§10 References:** added kestrel-talon repo as the upstream of constitution/doctrine injection; this design extends, doesn't fork.
- **Follow-up sub-issues:** added "Talon integration upgrade" — replace `_CONSTITUTION_CACHE` with WorkflowStageContext-backed loader; backport hash-verified injection / priority truncation / recursive ref-resolution / echo verification to standalone Talon runs.

### v3 → v3.1 changelog (review-3 inline fixes)

- **§3.3 Script-store population:** specified that scripts arrive via `compute.script_register(body, signature, signing_did)` and the runner refuses any run whose `src_hash` is not already locally registered. No auto-fetch; fleet propagation is a Castle/kestrel-claws concern.
- **§3.4 Reviewer attestation honesty:** downgraded the API-key-fingerprint claim. Account-binding ≠ model identity. Layered mitigation: account fingerprint + per-response `model` field check + family-diversity requirement. Local-model attestation remains genuinely strong via weight-checkpoint SHA. Provider signed-response receipts tracked as a follow-up.
- **§2.3 Conformance contract:** replaced tautological `INV_12_event_append_only`/`INV_13_signed_transitions` with behavioral `INV_12_rejects_event_mutation`/`INV_13_rejects_unsigned_writes`. Added `INV_14_result_fidelity` (byte-identity of completion result) and `INV_15_visibility_after_commit` (cross-process visibility post-commit).
- **§6 Phase 5:** explicit "FeatureFeature → Workflows is sequencing not a cycle" note.
- **L1 cleanup:** changelog/spec field name unified (`canonical_stage_input` + engine-contributed `nonce`).

### v2 → v3 changelog

- **§2.3 Conformance Contract:** added closed list of 13 testable invariants (`INV_1`..`INV_13`) the `WorkflowEngineAdapter` protocol must satisfy. Written *before* Phase 0 spike. If DBOS cannot satisfy, `DBOSAdapter` is cut, contract is not relaxed.
- **§3.1 Idempotency key:** revised to `sha256(run_id || stage_name || sha256(canonical_stage_input || attempt_number || nonce))` (engine-contributed nonce) so that legitimate retries (especially `red_team_clear` rounds) do not collapse. Stages declare `non_deterministic=True` to opt into per-attempt salt.
- **§3.3 `script` gate:** content-addressing hardened — definition embeds `(src_hash, signature, signing_did)` triple; runner resolves only against its own `ScriptStore`, refuses auto-fetch, verifies signature, and re-checks `compute.script_run` permission at run-start. Eliminates the substitution-via-hash attack.
- **§3.4 Reviewer attestation registry:** self-attestation rejected. Registry entries require provider-API-key fingerprint (hosted models) or weight-checkpoint SHA (local models) signed by Constitution-attested DID. Reviewer Constitution embedding hash must match a known-good list.
- **§3.4 Prompt-pack constraint:** `prompt_pack_constraint` is a **required** PEP-440 spec on every `red_team_clear` gate, included in `spec_hash`, refused at run-start if the locally-installed pack doesn't satisfy.
- **§3.4 Cost budget:** `max_total_cost_usd` and `max_total_tokens` ceilings; `gate_fail(reason=red_team_budget_exhausted)` on overrun. Operator default in `kestrel.toml`.
- **§3.5 Cancellation barrier:** spelled out the action_complete↔cancel race contract. Cancel is a barrier, not an interrupt; in-flight actions complete before compensation runs. `WorkflowHarness` MUST exercise the race.
- **§3.5 Compensation eligibility:** `noop_idempotent` is a closed structural set (`actor=system AND read_only=True`, plus `actor=human` consent stages). Self-attestation is rejected.
- **§3.5 Irreversible stages:** `irreversible=True` flag with `compensate_record_only` semantics; `cancelled_with_irreversible_residue` status; gates downstream cannot run until human acknowledgement.
- **§3.6 Definition versioning & revocation:** new section — runs pin to start-time version (no auto-promotion); revocation refuses new runs but preserves in-flight (with `signature_post_revocation=True` flag); `force_revoke` admin path for compromise.
- **§3.7 Observability:** new section — OTel spans, Prometheus counters/gauges, four built-in alert rules (`workflow_run_waiting_too_long`, `workflow_lease_expired_run_unclaimed`, `workflow_red_team_budget_exhausted_rate`, `workflow_compensate_failed_total > 0`).
- **§5 Schema:** added `cancel_barrier_at`, `claimed_by_worker_id`, `claim_expires_at`, `last_heartbeat_at` to `workflow_runs`; `current_stages_json TEXT` (SQLite-portable) replaces Postgres-only array; partial index for unconsumed `enter` events; retention reaper index `(workflow_name, finished_at)`; new status `cancelled_with_irreversible_residue`.
- **§6 Phase 0:** pre-spike step (write conformance suite). Stress tests 1–4 added (pause durability, fan-out under WAL, listener death, claim-and-crash). p99 latency budget (≤ 2s SQLite poll, ≤ 200ms Postgres LISTEN/NOTIFY). Concurrent-runs ceiling measured and surfaced as startup warning.
- **§6 Phase 1:** worker lease/heartbeat, cancellation barrier, alert rules file, `WorkflowHarness` cassette-replay layer with fail-closed semantics for non-replay-safe actors, definition-revocation handler.
- **§6 Phase 5 pilot:** switched from release-signing (already shipped) to FeatureFeature `feature_propose_*` workflows (active-pain consumer; exercises every gate and actor type).
- **§7 Non-goals:** `~100ms` boundary spelled out for sub-stage primitives.
- **§10:** consent reference resolved to #376 (Agent Lifecycle Hardening epic).

### v1 → v2 changelog

- **§2 (engine):** flipped recommendation. v1's premise that Postgres is a Kestrel default is false (`server.py:304` defaults to SQLite). v2 builds on the storage abstraction and adds DBOS as an optional Phase-2 adapter. `DBOSAdapter` ships in Phase 1 alongside `StorageAdapter` so the adapter interface is honest.
- **§3.1, §3.5:** added compensation/saga semantics. Every stage now has a mandatory `compensate`; cancellation runs compensations in reverse. Added Branch / Parallel / Subworkflow edge types and idempotency keys.
- **§3.3:** removed `custom(callable)` (privilege-escalation hatch). Replaced with `script(language, src_hash, sandbox)` running in the existing compute sandbox under proposing-DID permissions.
- **§3.4:** hardened `red_team_clear`. DID-distinctness enforcement, model-family diversity requirement, quote-fenced + canary-token prompt-injection defense, any-blocker-wins with named adjudicator, round-cap → human pause (never auto-pass), externalized reviewer-prompt versioning.
- **§4:** added definition-signature re-verification on every run start.
- **§5:** added `deleted_at` (soft-delete pattern), `retention_days`, `parent_run_id` (subworkflows), `current_stages[]` (parallel), idempotency key uniqueness, additional indexes, separate `workflow_signals` table for long-lived human-gate signals.
- **§6:** Phase 0 strictly time-boxed to engine spike with kill criteria. Phase 1 expanded to include conformance tests, OTel/Prometheus, WorkflowHarness, docker-compose. Phase 4 cut to CLI/API only (no UI). Phase 5 cut to one pilot migration. Phase 2 includes external `kestrel-red-team-prompts` package.
- **§7:** explicit non-goals for UI, YAML/TOML, bulk migrations, arbitrary callables.
