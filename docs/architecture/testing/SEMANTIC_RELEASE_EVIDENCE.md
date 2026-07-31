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

## Catalog-bound execution and unverified structural assembly

The public CLI has no `record -- <argv>` escape hatch and does **not** accept
an observation JSON object or benchmark samples to create a passed result.
`record` and `budget` are retained only as fail-closed migration errors.
`run` resolves a pre-registered, immutable `(runner_id, command_id)` workload
from the current catalog; it has no caller argv, artifact reference, or
observation fields. After the workload returns a schema-valid content-free
measurement, the execution authority creates an opaque artifact reference and
signs the exact run digest with Ed25519.

The public `assemble` command is deliberately **not** a signature-verification
or release-verdict surface. It reads no trust policy from CLI inputs or process
environment. It validates catalog, schema, digest, artifact, budget, drill,
and correlation bindings, then emits the separate structural contract
`semantic-kb-v1-release-evidence-structural-v1` with fixed
`"ready": false` and `"trust_status": "unverified"`. A
`"structurally_complete": true` result means only that every submitted field
fits the catalog; it is never proof that the signers are trusted or that a
release is ready.

Signature verification remains an explicit in-process verifier API:
`apply_evidence_records` and `apply_performance_budgets` require a
`TrustedExecutionPolicy`. That policy and any signed verification receipt must
be supplied by an independent Talon/CI verifier, not by a report author or the
public CLI. Until such a verifier issues a receipt, #2753 remains open and no
assembled artifact is a release claim. Within that verifier policy,
`external_ci` keys may attest only the catalog's `external_ci` runner;
`catalog_runner` keys cannot attest that runner, and neither source can
substitute for the other.

At present, the built-in executable workload is
`registry/stable_only_capability_selection_v1`. Other named pytest, benchmark,
and Kite HTTP workloads remain explicitly `blocked` with
`catalog_workload_unavailable` until their production harnesses are registered.
`run` writes that content-free block but exits nonzero so automation cannot
mistake an unavailable workload for successful execution; use the explicit
`block` command to record an independently observed block. They cannot become
passes through artifact JSON.

Start with a fresh conservative template:

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence template \
  --output /tmp/semantic-release-template.json
```

For example, a protected CI signing key can run the currently registered
stable-only registry workload. The key file contains raw 32-byte
lowercase-hex Ed25519 private-key material and must stay in the CI/host secret
store rather than source control or a command-line value. It must be a regular
file owned by the effective CI user, with no group or other permission bits;
symlinks are rejected.

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence run \
  --gate stable_only_capability_selection \
  --signing-key-file /secure/semantic-release.ed25519 \
  --issuer-id kestrel_ci \
  --key-id release_runner \
  --output /tmp/stable-only-registry.json

uv run python -m kestrel_sovereign.knowledge.release_evidence assemble \
  --record /tmp/stable-only-registry.json \
  --output /tmp/semantic-release-evidence.json
```

An independent verifier's trust policy stores public information only. Each
key fixes an issuer/key ID, `catalog_runner` or `external_ci` source, raw
Ed25519 public key, and allowed runner IDs. It is verifier-owned configuration,
not an attachment supplied by a record or an input to `assemble`:

```json
{
  "keys": [{
    "issuer_id": "kestrel_ci",
    "key_id": "release_runner",
    "source": "catalog_runner",
    "public_key": "<64-lowercase-hex-ed25519-public-key>",
    "runner_ids": ["registry", "pytest", "semantic_benchmark", "kite_http"]
  }]
}
```

Safe references are deliberately narrow: exactly
`ci://sha256/<64-lowercase-hex>`, `artifact://sha256/<64-lowercase-hex>`, or
`evidence://sha256/<64-lowercase-hex>`, alongside the artifact's SHA-256
digest; the SHA-256 component embedded in the locator must exactly equal
`artifact_digest`. The locator is opaque—not a path, label, subject, user, or
secret—and both the exact locator and digest are bound into the run digest.
URLs with query strings, identity-bearing parts, secret-like labels, traversal
segments, or connection strings are rejected. A blocked gate has no fake
artifact payload:

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence block \
  --gate postgres_assertion \
  --reason-code postgres_service_unavailable \
  --output /tmp/postgres-assertion-blocked.json
```

`assemble` starts from the current catalog, rejects unknown or duplicate gate
records, preserves missing work as `not_run` or `blocked`, and reports
structural completeness for inspection. It always emits `ready: false` and
includes `trust_verification_required` as a blocker. JSON fields named
`ready`, supplied keys, and environment variables cannot make the structural
artifact ready.

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

### Per-agent capability selection

An agent without a `[semantic_capabilities]` table is stable-only. An explicit
stable selection may contain only `mode = "stable"`. Draft mode is all or
nothing: it must select exact registry capability/version pairs for RDF 1.2,
SPARQL 1.2, and SHACL 1.2, plus an exact SHACL 1.2-compatible shape-set pin.
For example, the current snapshot-pinned experiment is:

```toml
[semantic_capabilities]
mode = "experimental"

[semantic_capabilities.rdf12]
capability = "rdf-profile:rdf12-cr-20260407-experimental"
version = "0.1.0"

[semantic_capabilities.sparql12]
capability = "query-profile:sparql12-20260605-experimental"
version = "0.1.0"

[semantic_capabilities.shacl12]
capability = "validation-profile:shacl12-core-20260602-experimental"
version = "0.1.0"

[semantic_capabilities.shape_set]
identifier = "kestrel-assertion-shapes-shacl12-experimental"
version = "0.1.0"
```

The host parses and verifies every selected local resource before publishing
the agent. Missing, partial, unknown, mismatched, or incompatible selections
fail closed. The selected capability map is then passed to the bounded sleep
maintenance coordinator and appears only as content-free `!sleep` diagnostics.
It changes no canonical assertion identity, storage schema, or migration path;
disabling the experiment returns to the stable profile without a migration.

## Backend parity and measured budgets

SQLite and PostgreSQL each require distinct results for assertion,
ownership/privacy, validation, inference/retraction, migration, corpus export,
and retrieval. A skipped PostgreSQL service stays blocked; a SQLite result
cannot satisfy its PostgreSQL counterpart. The artifact never stores a DSN.

Every performance workload is also separate by backend and execution mode:
startup; assertion write/validation; bounded inference; hybrid recall;
changed-work and unchanged-work sleep; storage growth; and representative
migration. Duration metrics use `ms`; storage growth uses the measured
post-operation minus pre-operation backend footprint in `bytes` (a SQLite file
size or Postgres relation-size reader), never elapsed time relabeled as bytes.
No timeout is a performance budget.

An allowlisted production benchmark workload must measure at least three
samples, emit the matching catalog observation, and write the signed record
and signed budget together. The public CLI intentionally has no
`--observation-json` or `--samples` route for this. Until a named benchmark
workload is registered, `run` records a block and the target remains a release
blocker. A registered workload is invoked in this shape:

```bash
uv run python -m kestrel_sovereign.knowledge.release_evidence run \
  --gate performance_hybrid_recall_sqlite_integration \
  --signing-key-file /secure/semantic-release.ed25519 \
  --issuer-id kestrel_ci \
  --key-id benchmark_runner \
  --output /tmp/hybrid-recall-sqlite-record.json \
  --budget-output /tmp/hybrid-recall-sqlite-budget.json

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

For this contract the pinned source is
`KestrelSovereignAI/kestrel-feature-parametric-self` at
`260ba985bcfdfab3dab1ea58da5b259057f3749f`; another revision remains
non-evidence even if its result fields look similar.

An external report is supplied as safe structured JSON—never as pre-populated
metadata—and is structurally checked against the locally assembled stage
records:

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
  --artifact-ref ci://sha256/SHA256_OF_TELEMETRY_ARTIFACT \
  --artifact-digest SHA256_OF_TELEMETRY_ARTIFACT \
  --output /tmp/legacy-fact-retirement-telemetry.json

uv run python -m kestrel_sovereign.knowledge.release_evidence assemble \
  --record /tmp/legacy-fact-migration-equivalence.json \
  --retirement-telemetry /tmp/legacy-fact-retirement-telemetry.json \
  --output /tmp/semantic-release-evidence.json
```

Only a complete inventory with zero eligible rows, zero required consumers,
and the exact passing equivalence result can make the **structural** decision
read `eligible_for_review`. That is not a trusted removal verdict: an
independent verifier must validate the signed evidence before operator review
and the migration/rollback policy may proceed. Any absent, mismatched, or
incomplete component leaves the path at `retain`.

## Verification order

The structural artifact is implementation-side evidence, not a substitute for
independent review or CI. Run the focused contract tests first, then actual
backend and live-agent evidence when those environments are available:

```bash
uv run pytest tests/unit/test_semantic_release_evidence.py \
  tests/unit/test_sleep_observability.py -q
```

Then follow the test pyramid in [Testing Guide](TESTING_GUIDE.md): targeted
backend integration, isolated Kite HTTP dogfooding, independent
`talon_verify`, and CI. The independent verifier—not `assemble`—must emit any
signed verification receipt and own its trust policy. If a review changes live
wiring, rerun the affected catalog records and re-dogfood before describing the
release as observed.
