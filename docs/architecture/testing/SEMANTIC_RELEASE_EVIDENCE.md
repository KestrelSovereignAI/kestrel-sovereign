---
type: Architecture Spec
title: Semantic Knowledge Release Evidence
description: Immutable, content-free evidence contracts for semantic knowledge conformance, parity, erasure, performance, diagnostics, and live-agent verification.
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

The semantic-KB release artifact is an immutable evidence contract, not a
feature switch or a release claim. Its catalog lives in
[`kestrel_sovereign.knowledge.release_evidence`](../../../kestrel_sovereign/knowledge/release_evidence.py)
and supplies every gate's exact runner, command-pattern digest, execution
environment, fixture ID/digest, observation schema, and (where needed) drill
correlation. A record is accepted only when it binds to that current `GateSpec`.

**Current status:** the catalog and assembly tooling exist, but no release
evidence has been executed for this rollout. A newly generated artifact is
always `ready: false`; [#2753](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2753)
remains open. Do not infer readiness, a completed migration, or a shipped
semantic capability from this document or from a passing unit test.

The artifact stores only content-free structure: declared IDs, digests,
aggregate counts, booleans, and safe opaque references. It never stores test
stdout, assertions, prompts, tenant IDs, source locators, DSNs, credentials, or
arbitrary command lines.

## Catalog-bound evidence, not command execution

The CLI does **not** execute a supplied command and has no `record -- <argv>`
escape hatch. A declared runner executes the review-approved workload outside
the artifact, then its result is entered using the catalog gate and its exact
content-free observation schema. `record` derives the spec, runner, command
digest, environment, fixture, and drill binding from the catalog; callers
cannot override them.

Start with a fresh conservative template:

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence template \
  --output /tmp/semantic-release-template.json
```

For example, an RDF 1.1 fixture runner can attach a measured aggregate result
and an approved artifact digest. Replace `SHA256_OF_APPROVED_ARTIFACT` with the
actual lowercase SHA-256 of the referenced CI/evidence artifact.

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence record \
  --gate rdf11_projection_fixture \
  --artifact-ref ci://semantic-release/rdf11-projection \
  --artifact-digest SHA256_OF_APPROVED_ARTIFACT \
  --observation-json '{"case_count":42,"assertion_count":97}' \
  --output /tmp/rdf11-projection.json

uv run python -m kestrel_sovereign.knowledge.release_evidence assemble \
  --record /tmp/rdf11-projection.json \
  --output /tmp/semantic-release-evidence.json
```

Safe references are deliberately narrow (`ci://`, `artifact://`, or
`evidence://` with a content-free path) and always travel with a SHA-256
digest. URLs with query strings, identity-bearing parts, secret-like labels, or
connection strings are rejected. A blocked gate has no fake artifact payload:

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence block \
  --gate postgres_assertion \
  --reason-code postgres_service_unavailable \
  --output /tmp/postgres-assertion-blocked.json
```

`assemble` starts from the current catalog, rejects unknown or duplicate gate
records, recomputes `ready`, and preserves missing work as `not_run` or
`blocked`. JSON fields named `ready` are output only; they cannot make an
incomplete report ready.

## Standards matrix and stable capability boundary

The template records exact offline registry pins and exact runtime library
versions. The matrix carries the official fixture ID/digest and predeclared
runner contract for the advertised RDF 1.1, RDFS 1.1, OWL 2 RL, SHACL 2017,
and read-only SPARQL 1.1 checks. A required library resolving as `unavailable`
blocks artifact construction rather than becoming a pass.

RDF 1.2, SHACL 1.2, and SPARQL 1.2 remain explicitly unadvertised draft gates;
only those gates may be `skipped` with `outside_advertised_capability`. The
stable registry capability-rejection gate is separate from live behavior:

- `stable_only_capability_selection` proves the registry rejects an
  experimental capability without opt-in.
- `kite_http_stable_only_release_drill` proves the isolated Kite HTTP path
  works under the stable-only profile.
- `kite_http_experimental_enabled_release_drill` independently proves the
  explicit experimental-enabled HTTP path.
- `stable_persisted_data_no_canonical_migration_drill` proves persisted stable
  data stays usable without a canonical-data migration.

All four are required gates. A registry unit check is not a substitute for the
two HTTP invoke drills, and an experimental-enabled result is not evidence for
stable-only operation.

## Backend parity and measured budgets

SQLite and PostgreSQL each require distinct results for assertion,
ownership/privacy, validation, inference/retraction, migration, corpus export,
and retrieval. A skipped PostgreSQL service stays blocked; a SQLite result
cannot satisfy its PostgreSQL counterpart. The artifact never stores a DSN.

Every performance workload is also separate by backend and execution mode:
startup; assertion write/validation; bounded inference; hybrid recall;
changed-work and unchanged-work sleep; storage growth; and representative
migration. Duration metrics use `ms`; storage growth uses `bytes`. No timeout
is a performance budget.

Record the performance gate's catalog observation and derive its budget from
at least three measured samples:

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence record \
  --gate performance_hybrid_recall_sqlite_integration \
  --artifact-ref ci://semantic-release/hybrid-recall-sqlite \
  --artifact-digest SHA256_OF_BENCHMARK_RESULT \
  --observation-json '{"sample_count":3,"p95_ms":13}' \
  --output /tmp/hybrid-recall-sqlite-record.json

uv run python -m kestrel_sovereign.knowledge.release_evidence budget \
  --gate performance_hybrid_recall_sqlite_integration \
  --samples 12.1 12.6 13.0 \
  --headroom-fraction 0.20 \
  --artifact-ref ci://semantic-release/hybrid-recall-sqlite-budget \
  --artifact-digest SHA256_OF_BENCHMARK_BUDGET_ARTIFACT \
  --output /tmp/hybrid-recall-sqlite-budget.json

uv run python -m kestrel_sovereign.knowledge.release_evidence assemble \
  --record /tmp/hybrid-recall-sqlite-record.json \
  --budget /tmp/hybrid-recall-sqlite-budget.json \
  --output /tmp/semantic-release-evidence.json
```

The record and budget must bind to the same immutable performance gate,
backend, mode, environment, fixture, and successful evidence digest.

## Correlated erasure and external adapter evidence

One release erasure drill has the catalog-defined content-free ID/digest
`semantic_erasure_release_drill_v1`. Every required core stage and every
external adapter stage must carry that same binding; a mismatched or omitted
drill is rejected. The stages cover active assertions, derivations, vector
index, recall candidates, exports, governed/future corpus output, projection
candidates, and served eligibility.

`kestrel-feature-parametric-self` is an external optional consumer, not a
dependency imported by this repository. Its report is absent from a new
template and therefore remains a readiness blocker until it contains all of:

- the exact repository and source revision required by the catalog;
- an attestation for `external_corpus_consumed`,
  `external_candidate_invalidated`, and
  `external_served_eligibility_rejected`;
- each stage's exact gate-spec digest, evidence-result digest, artifact
  reference/digest, and common drill binding; and
- a report digest over those fields.

An external report is supplied as safe structured JSON—never as pre-populated
metadata—and is checked against the locally assembled stage records:

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence assemble \
  --record /tmp/external-corpus-consumed.json \
  --record /tmp/external-candidate-invalidated.json \
  --record /tmp/external-served-eligibility-rejected.json \
  --external-report /tmp/parametric-self-erasure-report.json \
  --output /tmp/semantic-release-evidence.json
```

Missing stages, another repository/revision, a different result, a different
artifact, or a different drill remains rejected rather than being narrated as
evidence.

## Content-free diagnostics and Kite HTTP proof

`semantic_maintenance_diagnostics_contract` is a required gate. The live
`!sleep --consolidate-only` HTTP invoke surface must yield a positive
content-free diagnostic count and zero redaction violations. Its structured
diagnostic exposes bounded checkpoint/backlog state, fixed repair guidance,
and only exact registry-verified capability labels.

The diagnostic renderer does not accept a version merely because it resembles
one: `v999` and `999.999.999` are omitted. An inference-profile hash is shown
only when it reconstructs from exact local ontology/rule pins; an ontology is
shown as its registry resource ID rather than its raw namespace. Missing or
unverifiable producer profile/ontology fields are explicitly `omitted` or
`unavailable`. No assertion terms, IDs, tenant data, source provenance, raw
errors, or arbitrary capability-map values reach the HTTP response.

Drive Kite through the isolated HTTP path described in
[Live-Agent Dogfooding](LIVE_AGENT_DOGFOODING.md), not by calling a storage or
service method directly. The required live gates cover paraphrased recall with
provenance, contradiction/supersession, invalid-import quarantine, sleep,
restart persistence, and post-delete non-recall. The release artifact records
only the catalog-bound aggregate observation and approved artifact reference;
the transcript remains in the isolated evidence environment.

## Compatibility retirement is an observed decision

`legacy_fact_migration_equivalence` is a dedicated non-readiness gate used only
for compatibility retirement. A removal decision is bound to that exact gate
ID, its current spec digest, and its passing run digest; a result from any
other gate or an unrelated result digest is rejected.

Telemetry is a separate content-free attestation with UTC start/end timestamps,
an inventory digest, inventory-complete flag, unmigrated-eligible-row count,
required-consumer count, artifact reference/digest, and a digest over the
whole envelope. Create it with the safe CLI input, then attach it only during
assembly:

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence telemetry \
  --window-started-at 2026-07-30T00:00:00Z \
  --window-ended-at 2026-07-31T00:00:00Z \
  --inventory-digest SHA256_OF_TELEMETRY_INVENTORY \
  --inventory-complete \
  --unmigrated-eligible-rows 0 \
  --required-consumer-count 0 \
  --artifact-ref ci://semantic-release/legacy-fact-retirement-window \
  --artifact-digest SHA256_OF_TELEMETRY_ARTIFACT \
  --output /tmp/legacy-fact-retirement-telemetry.json

uv run python -m kestrel_sovereign.knowledge.release_evidence assemble \
  --record /tmp/legacy-fact-migration-equivalence.json \
  --retirement-telemetry /tmp/legacy-fact-retirement-telemetry.json \
  --output /tmp/semantic-release-evidence.json
```

Only a complete inventory with zero eligible rows, zero required consumers,
and the exact passing equivalence result yields `eligible_for_review`. It is
not an automatic deletion; operator review and the migration/rollback policy
remain required. Any absent, mismatched, or incomplete component leaves the
path at `retain`.

## Verification order

The artifact is implementation-side evidence, not a substitute for independent
review or CI. Run the focused contract tests first, then actual backend and
live-agent evidence when those environments are available:

```bash
uv run pytest tests/unit/test_semantic_release_evidence.py \
  tests/unit/test_sleep_observability.py -q
```

Then follow the test pyramid in [Testing Guide](TESTING_GUIDE.md): targeted
backend integration, isolated Kite HTTP dogfooding, independent
`talon_verify`, and CI. If a review changes live wiring, rerun the affected
catalog records and re-dogfood before describing the release as observed.
