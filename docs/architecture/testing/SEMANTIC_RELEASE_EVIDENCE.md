---
type: Architecture Spec
title: Semantic Knowledge Release Evidence
description: Machine-checkable release gates for semantic knowledge conformance, parity, erasure, performance, and live-agent verification.
resource: /docs/architecture/testing/SEMANTIC_RELEASE_EVIDENCE.md
tags:
- docs
- architecture
- testing
- semantic-knowledge
timestamp: '2026-07-31T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Semantic Knowledge Release Evidence

The semantic-KB release gate is an evidence artifact, not a feature switch.
It records what was actually run and leaves missing work visibly blocked. It
does not enable a capability, relax privacy, make a test timeout pass, or let a
feature consumer access a database directly.

The runner is
[`kestrel_sovereign.knowledge.release_evidence`](../../../kestrel_sovereign/knowledge/release_evidence.py).
It emits canonical JSON with `ready: false` until every required gate has a
command, zero exit status, and an artifact reference. This is deliberately
stricter than a narrative release note.

## Create and assemble evidence

Start each release with a fresh template. It records exact offline registry
pins, Python/RDFLib versions, the stable-only check, and every required
unobserved gate.

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence template \
  --output /tmp/semantic-release-template.json
```

Record a command result without copying test stdout, assertion values, tenant
IDs, source locators, or database credentials into the JSON artifact. The
referenced log belongs in the CI system or approved evidence store.

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence record \
  --gate rdf11_projection_fixture \
  --artifact-ref ci://semantic-rdf11-projection \
  --output /tmp/rdf11.json -- \
  uv run pytest tests/unit/test_knowledge_rdf_codec.py -q

uv run python -m kestrel_sovereign.knowledge.release_evidence assemble \
  --record /tmp/rdf11.json \
  --output /tmp/semantic-release-evidence.json
```

`assemble` starts from the current conservative template and accepts only
predeclared gate IDs. An external consumer cannot add a lookalike success gate
to conceal a required core or adapter check. A nonzero command exit becomes
`failed`; an absent service remains `blocked` or `not_run`, never `passed`.

## Standards and stable-only conformance

The artifact's standards matrix is built from the offline
[Semantic Knowledge Registry](../storage/SEMANTIC_KNOWLEDGE_REGISTRY.md), so it
records exact RDF/RDFS/OWL/SHACL/SPARQL resources, versions, dates, hashes,
maturity, and profiles plus the runtime library versions. It is not derived
from mutable `latest` URLs.

The advertised stable path requires evidence for RDF 1.1 projection, RDFS 1.1
inference, OWL 2 RL inference, SHACL 2017 Core, and read-only SPARQL 1.1
fixtures. A missing official fixture stays `not_run`; it cannot be written as
`skipped`. RDF 1.2, SHACL 1.2, and SPARQL 1.2 fixtures are marked `skipped`
only because they are outside the advertised stable-only profile. The runner
instantiates the default RDF codec and proves every experimental registry
capability rejects selection without `allow_experimental=True`; this uses the
same canonical data and does not perform a migration.

At minimum run:

```bash
uv run pytest tests/unit/test_knowledge_registry.py \
  tests/unit/test_knowledge_rdf_codec.py tests/unit/test_shacl_validation.py \
  tests/unit/test_semantic_inference.py tests/unit/test_semantic_assertion_recall.py -q
```

Do not describe a stable-only run as a draft fallback. It is the production
RDF 1.1 / SHACL 2017 route, with all 1.2 capabilities disabled.

## SQLite and PostgreSQL parity

The artifact requires independently recorded results for SQLite and PostgreSQL
for assertion, ownership/privacy, validation, inference/retraction, migration,
governed-corpus export, and retrieval. The existing dual-backend suite is the
core executable contract:

```bash
export TEST_POSTGRES_URL='postgresql://user:pass@localhost:5433/kestrel_test'
uv run pytest tests/integration/test_storage_backend_parity.py -m dual_backend -q
```

Without `TEST_POSTGRES_URL` (or `KESTREL_DATABASE_URL` / `DATABASE_URL`) the
fixture skips PostgreSQL. Record that as `blocked` with
`postgres_service_unavailable`; do not turn a SQLite pass into a parity pass.
The semantic recall path also requires a PostgreSQL image with the `vector`
extension (for example `pgvector/pgvector:pg16`, or an operator-managed
equivalent). A plain `postgres:16` image is useful to prove that missing
dependency is reported as `postgres_vector_extension_unavailable`; it is not
semantic parity evidence. The report stores no DSN.

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence block \
  --gate postgres_assertion --reason-code postgres_service_unavailable \
  --artifact-ref ci://semantic-parity-environment \
  --output /tmp/postgres-assertion-blocked.json
```

## Performance budgets

`SemanticBenchmarkHarness` repeats a named, seeded workload at least three
times and records its fixture digest, runtime environment, and samples.
`PerformanceBudget.from_observed()` derives p95 plus explicit headroom. It does
not set a process timeout, because a timeout can hide a regression rather than
measure it.

The release report requires observed budgets for all of these workloads:

- startup;
- assertion write plus validation;
- bounded inference;
- hybrid recall;
- changed-work sleep and unchanged sleep;
- storage growth; and
- a representative legacy-fact migration.

Create a separate, content-free budget payload for each completed workload;
the command record and budget are both required before a metric can pass:

```bash
printf '%s\n' '{"backend":"sqlite","seed":"p4-v1","workload":"hybrid_recall"}' \
  > /tmp/hybrid-recall-fixture.json
uv run python -m kestrel_sovereign.knowledge.release_evidence budget \
  --metric hybrid_recall --samples-ms 12.1 12.6 13.0 \
  --headroom-fraction 0.20 --fixture /tmp/hybrid-recall-fixture.json \
  --output /tmp/hybrid-recall-budget.json
uv run python -m kestrel_sovereign.knowledge.release_evidence assemble \
  --record /tmp/hybrid-recall-command.json \
  --budget /tmp/hybrid-recall-budget.json \
  --output /tmp/semantic-release-evidence.json
```

Store the exact fixture description (backend, seeded corpus shape, profile,
and workload) with the benchmark log. Re-run and compare against the recorded
budget after an implementation or review change affecting the workload.

## Erasure drill and feature-owned adapters

An erasure report is complete only when it proves absence from active
assertions, derivations, the vector index, semantic-recall candidates, export
snapshots, governed corpus, future corpus output, projection candidates, and
serving eligibility. The final two stages are intentionally separate: an
adapter must both remove a candidate and refuse to serve an already-prepared
candidate after erasure.

Core owns the canonical/corpus boundaries. The
`kestrel-feature-parametric-self` adapter is an external optional consumer and
is **not** imported by this repository. It supplies three independently
recorded results to the existing artifact schema:

- `external_corpus_consumed`;
- `external_candidate_invalidated`; and
- `external_served_eligibility_rejected`.

Record its repository and tested commit (currently the P4 hand-off records
`KestrelSovereignAI/kestrel-feature-parametric-self` at `bcbfbb2`) alongside
the evidence references. Core accepts that report structurally; it neither
installs the feature nor treats a missing report as a pass.

For core regression coverage, run the governed corpus, inference, recall, and
backend parity tests before an erasure drill. The isolated Kite test remains
the live bar below.

## Content-free operational diagnostics

`SleepReport.semantic_maintenance_diagnostics()` is the live structured
diagnostic contract. It exposes only:

- verified active semantic capability/version labels;
- source and checkpoint generations;
- assertion/report backlog and whether the run is partial; and
- a fixed repair-guidance code.

The matching `!sleep` text is bounded and has the same data. It never renders
assertion terms, IDs, tenant identifiers, source provenance, raw errors, or
arbitrary capability-map values. Use `rerun_bounded_maintenance`,
`check_semantic_profile_pins`, or `wait_for_active_maintenance_lease` as the
runbook action codes; trusted operators may use the public governed maintenance
API for an explicit repair. Raw diagnostic text is not a repair instruction.

## Kite live-agent verification

Follow the isolation and HTTP invocation procedure in
[Live-Agent Dogfooding](LIVE_AGENT_DOGFOODING.md). The release artifact does
not replace this test; it has a required `kite_http_invoke_release_drill` gate.
On a dedicated worktree, venv, `KESTREL_HOME`, and port, prove through
`POST /api/agents/<name>/api/agent/invoke`:

1. paraphrased recall includes provenance through the live context path;
2. contradiction/supersession does not choose a vector-ranked winner;
3. an invalid import is quarantined rather than recalled;
4. changed and unchanged sleep report bounded maintenance state;
5. a restart preserves only valid governed state; and
6. post-delete input cannot recall the erased fact, corpus candidate, or served
   adapter output.

A freshly generated template intentionally remains non-ready until this live
result is attached. A local direct service test is not a substitute for HTTP
invoke.

## Compatibility retirement and rollback

No compatibility path may be removed based only on code inspection. The
artifact's retirement decision stays `retain` until it carries all of:

- a bounded observed telemetry-window reference;
- a complete inventory;
- zero unmigrated eligible rows;
- zero required consumers; and
- a passing migration-equivalence record.

For the legacy graph-fact migration this complements the migration service's
own `complete_inventory` / `removal_safe` result and operator review of each
rejection class. The migration keeps legacy rows for audit and rollback never
resurrects erased facts. If any required evidence is absent, keep the
compatibility path and document the blocker instead of guessing that it is
safe to remove.

## Evidence pyramid

The release artifact is implementation-side evidence. It does not replace
independent review via `talon_verify` or CI, described in
[Test Evidence Gates](TEST_EVIDENCE_GATES.md). Run the semantic targeted unit
tests, then the dual-backend integration suite, then Kite. If a review changes
live wiring, re-run the affected artifact records and dogfood the live path
again.
