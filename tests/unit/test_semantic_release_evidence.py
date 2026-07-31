"""Adversarial contracts for spec-bound semantic release evidence."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kestrel_sovereign.knowledge.release_evidence import (
    ArtifactReference,
    CompatibilityRetirementDecision,
    EvidenceRecord,
    EvidenceState,
    ExecutionEnvironment,
    ExternalCapabilityReport,
    ExternalGateAttestation,
    GateResult,
    PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
    PARAMETRIC_SELF_EVIDENCE_REVISION,
    PerformanceBudget,
    PerformanceMetric,
    PerformanceTarget,
    ReleaseEvidenceError,
    TelemetryAttestation,
    TrustedExecutionPolicy,
    attach_external_capability_report,
    attach_retirement_telemetry,
    apply_evidence_records,
    apply_performance_budgets,
    build_standards_matrix,
    evidence_record_from_mapping,
    inspect_stable_only_capabilities,
    performance_targets,
    release_evidence_template,
    release_gate_specs,
    telemetry_attestation_from_mapping,
)
from kestrel_sovereign.knowledge.release_evidence_execution import CatalogSigningIdentity
from kestrel_sovereign.knowledge.release_evidence_models import (
    ExecutionSource,
    SemanticBenchmarkHarness,
)


def _gate(gate_id: str):
    return next(spec for spec in release_gate_specs() if spec.gate_id == gate_id)


def _observation(spec) -> dict[str, object]:
    """Produce a schema-valid content-free result for structural tests only."""
    values: dict[str, object] = {}
    for field in spec.observation_schema.fields:
        values[field.field_id] = (
            3
            if field.kind == "sample_count"
            else 0
            if field.kind == "zero_count"
            else True
            if field.kind == "boolean"
            else "a" * 64
            if field.kind == "digest"
            else 1
        )
    return values


def _artifact(name: str = "result") -> ArtifactReference:
    digest = _opaque_artifact_digest(name)
    return ArtifactReference(f"ci://sha256/{digest}", digest)


def _opaque_artifact_digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _opaque_artifact_ref(name: str) -> str:
    return f"ci://sha256/{_opaque_artifact_digest(name)}"


_CATALOG_TEST_IDENTITY = CatalogSigningIdentity(
    issuer_id="test_ci",
    key_id="release_test_key",
    private_key=Ed25519PrivateKey.from_private_bytes(b"\x01" * 32),
)
_EXTERNAL_TEST_IDENTITY = CatalogSigningIdentity(
    issuer_id="parametric_ci",
    key_id="external_release_key",
    private_key=Ed25519PrivateKey.from_private_bytes(b"\x05" * 32),
    source=ExecutionSource.EXTERNAL_CI,
)


def _test_policy() -> TrustedExecutionPolicy:
    return TrustedExecutionPolicy(
        (
            _CATALOG_TEST_IDENTITY.trusted_key(
                tuple(
                    sorted(
                        {
                            spec.runner.runner_id
                            for spec in release_gate_specs()
                            if spec.runner.runner_id != "external_ci"
                        }
                    )
                )
            ),
            _EXTERNAL_TEST_IDENTITY.trusted_key(("external_ci",)),
        )
    )


def _record(
    spec,
    *,
    observation: dict[str, object] | None = None,
    identity: CatalogSigningIdentity | None = None,
):
    observed = observation or _observation(spec)
    artifact = _artifact(spec.gate_id)
    selected_identity = identity or (
        _EXTERNAL_TEST_IDENTITY
        if spec.runner.runner_id == "external_ci"
        else _CATALOG_TEST_IDENTITY
    )
    _, run_digest = EvidenceRecord._bound_run_digest(
        spec,
        observed,
        artifact,
        state=EvidenceState.PASSED,
    )
    return EvidenceRecord._from_trusted_execution(
        spec,
        observed,
        artifact,
        state=EvidenceState.PASSED,
        execution_attestation=selected_identity.sign(
            kind="evidence_record", spec=spec, run_digest=run_digest
        ),
    )


def _budget(
    spec,
    samples: tuple[float | int, ...] = (1.0, 2.0, 3.0),
    *,
    identity: CatalogSigningIdentity | None = None,
) -> PerformanceBudget:
    artifact = _artifact(f"{spec.gate_id}-budget")
    _, _, _, run_digest = PerformanceBudget._bound_run_digest(
        spec,
        samples,
        headroom_fraction=0.2,
        artifact=artifact,
    )
    return PerformanceBudget._from_trusted_execution(
        spec,
        samples,
        headroom_fraction=0.2,
        artifact=artifact,
        execution_attestation=(identity or _CATALOG_TEST_IDENTITY).sign(
            kind="performance_budget", spec=spec, run_digest=run_digest
        ),
    )


def _gate_result(spec, record):
    return GateResult(spec, record, _test_policy())


def _apply_records(template, records):
    return apply_evidence_records(template, records, trust_policy=_test_policy())


def _apply_budgets(template, budgets):
    return apply_performance_budgets(template, budgets, trust_policy=_test_policy())


def _retirement_telemetry() -> TelemetryAttestation:
    return TelemetryAttestation.attest(
        window_started_at="2026-07-30T00:00:00Z",
        window_ended_at="2026-07-31T00:00:00Z",
        inventory_digest="d" * 64,
        inventory_complete=True,
        unmigrated_eligible_rows=0,
        required_consumer_count=0,
        artifact=_artifact("retirement-telemetry"),
    )


def _external_report(evidence) -> ExternalCapabilityReport:
    external_gates = [gate for gate in evidence.gates if gate.spec.category == "external_adapter"]
    attestations: list[ExternalGateAttestation] = []
    for gate in external_gates:
        record = gate.evidence
        assert record.passed and record.run_digest is not None and record.artifact is not None
        assert record.drill is not None
        attestations.append(
            ExternalGateAttestation(
                gate_id=gate.spec.gate_id,
                gate_spec_digest=gate.spec.digest,
                result_digest=record.run_digest,
                artifact=record.artifact,
                drill=record.drill,
            )
        )
    return ExternalCapabilityReport.attest(
        capability_id="parametric_self_governed_corpus",
        repository=PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
        source_revision=PARAMETRIC_SELF_EVIDENCE_REVISION,
        attestations=tuple(attestations),
    )


def test_catalog_has_single_immutable_contract_for_every_release_gate() -> None:
    specs = release_gate_specs()

    assert len({spec.gate_id for spec in specs}) == len(specs)
    assert all(len(spec.digest) == 64 for spec in specs)
    assert all(spec.runner.runner_id and len(spec.runner.command_digest) == 64 for spec in specs)
    assert all(spec.environment.backend and spec.environment.mode for spec in specs)
    assert all(spec.fixture.binding.fixture_id and len(spec.fixture.binding.fixture_digest) == 64 for spec in specs)
    assert all(spec.observation_schema.fields for spec in specs)


def test_standards_matrix_records_official_fixture_digest_and_harness() -> None:
    matrix = {entry.identifier: entry for entry in build_standards_matrix()}
    gate_to_standard = {
        "rdf11_projection_fixture": "rdf11-concepts-20140225",
        "rdfs11_inference_fixture": "rdfs-20140225",
        "owl2rl_inference_fixture": "owl2-profiles-20121211",
        "shacl2017_core_fixture": "shacl-core-20170720",
        "sparql11_readonly_fixture": "sparql11-readonly",
    }

    rdf = matrix["rdf11-concepts-20140225"]
    assert rdf.fixture is not None
    assert rdf.fixture.official is True
    assert rdf.fixture.binding.fixture_id == "official.rdf11_projection.v1"
    assert rdf.fixture.harness_id == "rdf11_projection_harness_v1"
    assert len(rdf.fixture.binding.fixture_digest) == 64
    assert rdf.runner == _gate("rdf11_projection_fixture").runner
    assert rdf.runner is not None and len(rdf.runner.command_digest) == 64
    for gate_id, standard_id in gate_to_standard.items():
        entry = matrix[standard_id]
        spec = _gate(gate_id)
        assert entry.fixture == spec.fixture
        assert entry.fixture is not None and entry.fixture.official is True
        assert entry.runner == spec.runner


def test_catalog_keeps_registry_and_live_kite_release_gates_separate_and_required() -> None:
    specs = {spec.gate_id: spec for spec in release_gate_specs()}
    live_gate_ids = {
        "kite_http_stable_only_release_drill",
        "kite_http_experimental_enabled_release_drill",
        "stable_persisted_data_no_canonical_migration_drill",
    }

    assert specs["stable_only_capability_selection"].runner.runner_id == "registry"
    assert specs["stable_only_capability_selection"].category == "capability_selection"
    assert all(specs[gate_id].category == "live_agent" for gate_id in live_gate_ids)
    assert all(specs[gate_id].runner.runner_id == "kite_http" for gate_id in live_gate_ids)
    assert all(specs[gate_id].required_for_ready for gate_id in live_gate_ids)
    assert specs["stable_persisted_data_no_canonical_migration_drill"].environment.profile == "stable_only"
    assert specs["kite_http_experimental_enabled_release_drill"].environment.profile == "experimental_enabled"
    assert {
        field.field_id: field.kind
        for field in specs["stable_persisted_data_no_canonical_migration_drill"].observation_schema.fields
    }["canonical_migration_count"] == "zero_count"
    assert specs["semantic_maintenance_diagnostics_contract"].required_for_ready is True


def test_template_never_auto_passes_registry_selection_or_missing_evidence() -> None:
    template = release_evidence_template()

    assert template.ready is False
    assert "stable_only_capability_selection" in template.blocking_gate_ids()
    assert "kite_http_stable_only_release_drill" in template.blocking_gate_ids()
    assert "kite_http_experimental_enabled_release_drill" in template.blocking_gate_ids()
    assert "stable_persisted_data_no_canonical_migration_drill" in template.blocking_gate_ids()
    assert "semantic_maintenance_diagnostics_contract" in template.blocking_gate_ids()
    assert "postgres_assertion" in template.blocking_gate_ids()
    assert "performance_hybrid_recall_postgres_integration" in template.blocking_gate_ids()
    assert "external_adapter_attestation" in template.blocking_gate_ids()
    assert template.external_capabilities == ()
    assert all(gate.evidence.state is not EvidenceState.PASSED for gate in template.gates)
    assert inspect_stable_only_capabilities()["rejected_capability_count"] > 0


def test_reviewer_adversarial_erasure_drill_rejects_mismatched_or_missing_correlation() -> None:
    correlated = [
        spec
        for spec in release_gate_specs()
        if spec.category in {"erasure", "external_adapter"}
    ]

    assert correlated
    assert all(spec.correlation is not None for spec in correlated)
    assert len({spec.correlation for spec in correlated}) == 1
    with pytest.raises(ReleaseEvidenceError, match="require drill correlation"):
        replace(correlated[0], correlation=None)
    for spec in correlated:
        record = _record(spec)
        assert record.drill == spec.correlation
        assert record.drill is not None
        with pytest.raises(ReleaseEvidenceError, match="drill correlation"):
            _gate_result(spec, replace(record, drill=replace(record.drill, drill_digest="b" * 64)))
        with pytest.raises(ReleaseEvidenceError, match="drill correlation"):
            _gate_result(spec, replace(record, drill=None))


def test_reviewer_adversarial_retirement_rejects_unrelated_gate_or_result_reference() -> None:
    template = release_evidence_template()
    migration = _gate("legacy_fact_migration_equivalence")
    unrelated = _record(_gate("rdf11_projection_fixture"))
    with_unrelated = _apply_records(template, (unrelated,))
    telemetry = _retirement_telemetry()

    wrong_gate = CompatibilityRetirementDecision(
        path_id="legacy_fact_migration_compatibility",
        migration_gate_id=unrelated.gate_id,
        migration_spec_digest=unrelated.gate_spec_digest,
        migration_run_digest=unrelated.run_digest,
        telemetry=telemetry,
    )
    with pytest.raises(ReleaseEvidenceError, match="not bound to legacy migration equivalence spec"):
        replace(with_unrelated, compatibility_retirement=(wrong_gate,))

    wrong_result = CompatibilityRetirementDecision(
        path_id="legacy_fact_migration_compatibility",
        migration_gate_id=migration.gate_id,
        migration_spec_digest=migration.digest,
        migration_run_digest=unrelated.run_digest,
        telemetry=telemetry,
    )
    with pytest.raises(ReleaseEvidenceError, match="unrelated migration result"):
        replace(with_unrelated, compatibility_retirement=(wrong_result,))

    retained = attach_retirement_telemetry(with_unrelated, telemetry)
    assert retained.compatibility_retirement[0].decision == "retain"
    assert retained.compatibility_retirement[0].reason_code == "migration_equivalence_not_observed"


def test_retirement_telemetry_requires_its_digest_and_bound_equivalence_record() -> None:
    template = release_evidence_template()
    migration = _gate("legacy_fact_migration_equivalence")
    migration_record = _record(migration)
    telemetry = _retirement_telemetry()

    payload = telemetry.to_mapping()
    payload["required_consumer_count"] = 1
    with pytest.raises(ReleaseEvidenceError, match="telemetry digest"):
        telemetry_attestation_from_mapping(payload)

    updated = attach_retirement_telemetry(
        _apply_records(template, (migration_record,)),
        telemetry,
    )
    decision = updated.compatibility_retirement[0]
    assert decision.migration_gate_id == migration.gate_id
    assert decision.migration_spec_digest == migration.digest
    assert decision.migration_run_digest == migration_record.run_digest
    assert decision.removal_safe is True
    assert decision.decision == "eligible_for_review"


def test_reviewer_adversarial_external_report_requires_exact_stages_repo_revision_and_artifacts() -> None:
    template = release_evidence_template()
    external_specs = [spec for spec in release_gate_specs() if spec.category == "external_adapter"]
    evidence = _apply_records(template, tuple(_record(spec) for spec in external_specs))
    report = _external_report(evidence)

    assert set(report.gate_ids) == {
        "external_corpus_consumed",
        "external_candidate_invalidated",
        "external_served_eligibility_rejected",
    }
    attached = attach_external_capability_report(evidence, report)
    assert attached.external_capabilities == (report,)
    assert "external_adapter_attestation" not in attached.blocking_gate_ids()

    incomplete = ExternalCapabilityReport.attest(
        capability_id=report.capability_id,
        repository=report.repository,
        source_revision=report.source_revision,
        attestations=report.attestations[:-1],
    )
    with pytest.raises(ReleaseEvidenceError, match="cover corpus, candidate, and served stages"):
        attach_external_capability_report(evidence, incomplete)

    wrong_repository = ExternalCapabilityReport.attest(
        capability_id=report.capability_id,
        repository="example/other-adapter",
        source_revision=report.source_revision,
        attestations=report.attestations,
    )
    with pytest.raises(ReleaseEvidenceError, match="repository or revision"):
        attach_external_capability_report(evidence, wrong_repository)

    mismatched_artifact = ExternalCapabilityReport.attest(
        capability_id=report.capability_id,
        repository=report.repository,
        source_revision=report.source_revision,
        attestations=(
            replace(report.attestations[0], artifact=_artifact("wrong-external-artifact")),
            *report.attestations[1:],
        ),
    )
    with pytest.raises(ReleaseEvidenceError, match="correlated gate result/artifact"):
        attach_external_capability_report(evidence, mismatched_artifact)


def test_reviewer_adversarial_record_cannot_spoof_a_gate_with_exit_zero() -> None:
    """A raw generic command is neither persisted nor accepted as a record."""
    spec = _gate("rdf11_projection_fixture")
    record = _record(spec)
    payload = record.to_mapping()
    payload["command"] = "python -c pass"

    with pytest.raises(ReleaseEvidenceError, match="unknown or missing"):
        evidence_record_from_mapping(payload)


def test_reviewer_adversarial_record_rejects_spec_environment_and_fixture_mismatch() -> None:
    spec = _gate("rdf11_projection_fixture")
    record = _record(spec)
    bad_environment = ExecutionEnvironment("postgres", "integration", "stable_only")

    with pytest.raises(ReleaseEvidenceError, match="environment"):
        _gate_result(
            spec,
            replace(record, environment=bad_environment, environment_digest=bad_environment.digest),
        )
    with pytest.raises(ReleaseEvidenceError, match="fixture"):
        _gate_result(
            spec,
            replace(record, fixture=replace(record.fixture, fixture_id="other.fixture.v1")),
        )
    with pytest.raises(ReleaseEvidenceError, match="spec digest"):
        _gate_result(spec, replace(record, gate_spec_digest="b" * 64))


def test_reviewer_adversarial_record_rejects_wrong_observation_schema() -> None:
    spec = _gate("rdf11_projection_fixture")
    record = _record(spec)

    with pytest.raises(ReleaseEvidenceError, match="observation"):
        _gate_result(spec, replace(record, observation={"case_count": 1}))


@pytest.mark.parametrize(
    "reference",
    (
        "postgresql://user:password@db.example/kestrel",
        "ci://semantic-release/123?token=secret",
        "artifact://tenant-42/semantic-result",
        "evidence://user@host/release",
        "artifact://semantic-release/credential-proof",
        "ci://sha256/patient-alice-hiv",
        "ci://sha256/sk-proj-credential",
        "ci://sha256/0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd..",
    ),
)
def test_reviewer_adversarial_artifact_references_reject_secrets_and_identity(reference: str) -> None:
    with pytest.raises(ReleaseEvidenceError, match="artifact"):
        ArtifactReference(reference, "a" * 64)


def test_reviewer_adversarial_artifact_reference_digest_must_match_its_locator() -> None:
    with pytest.raises(ReleaseEvidenceError, match="must match artifact_digest"):
        ArtifactReference(_opaque_artifact_ref("record"), _opaque_artifact_digest("other"))


def test_reviewer_adversarial_run_digest_binds_the_complete_opaque_artifact_reference() -> None:
    spec = _gate("rdf11_projection_fixture")
    record = _record(spec)
    assert record.artifact is not None

    with pytest.raises(ReleaseEvidenceError, match="run digest"):
        _gate_result(
            spec,
            replace(
                record,
                artifact=ArtifactReference(
                    _opaque_artifact_ref("other"), _opaque_artifact_digest("other")
                ),
            ),
        )

    benchmark = _performance_spec(PerformanceMetric.HYBRID_RECALL, "sqlite")
    budget = _budget(benchmark)
    with pytest.raises(ReleaseEvidenceError, match="run digest"):
        replace(
            budget,
            artifact=ArtifactReference(
                _opaque_artifact_ref("other-benchmark"),
                _opaque_artifact_digest("other-benchmark"),
            ),
        )


def _performance_spec(metric: PerformanceMetric, backend: str):
    return next(
        spec
        for spec in release_gate_specs()
        if spec.performance_target
        and spec.performance_target.metric is metric
        and spec.performance_target.backend == backend
    )


def test_reviewer_adversarial_performance_rejects_short_zero_invalid_and_unit_mismatch() -> None:
    postgres = _performance_spec(PerformanceMetric.HYBRID_RECALL, "postgres")
    artifact = _artifact("benchmark")

    with pytest.raises(ReleaseEvidenceError, match="cannot mint"):
        PerformanceBudget.from_observed(postgres, (1.0, 2.0), headroom_fraction=0.2, artifact=artifact)
    with pytest.raises(ReleaseEvidenceError, match="positive"):
        PerformanceBudget._bound_run_digest(
            postgres, (0.0, 1.0, 2.0), headroom_fraction=0.2, artifact=artifact
        )
    with pytest.raises(ReleaseEvidenceError, match="duration metrics"):
        PerformanceTarget(PerformanceMetric.HYBRID_RECALL, "postgres", "integration", "bytes")
    with pytest.raises(ReleaseEvidenceError, match="storage growth"):
        PerformanceTarget(PerformanceMetric.STORAGE_GROWTH, "postgres", "integration", "ms")


def test_sqlite_only_budget_cannot_satisfy_postgres_or_release_readiness() -> None:
    template = release_evidence_template()
    sqlite = _performance_spec(PerformanceMetric.HYBRID_RECALL, "sqlite")
    sqlite_record = _record(sqlite)
    with_record = _apply_records(template, (sqlite_record,))
    sqlite_budget = _budget(sqlite)
    updated = _apply_budgets(with_record, (sqlite_budget,))

    assert updated.performance_budgets[sqlite.performance_target] == sqlite_budget
    assert any(
        target.metric is PerformanceMetric.HYBRID_RECALL
        and target.backend == "postgres"
        and budget is None
        for target, budget in updated.performance_budgets.items()
    )
    assert updated.ready is False


def test_missing_runtime_library_version_is_a_release_block_not_a_pass() -> None:
    template = release_evidence_template()

    with pytest.raises(ReleaseEvidenceError, match="missing library version"):
        replace(template, libraries={**template.libraries, "rdflib": "unavailable"})


def test_performance_targets_cover_each_metric_on_sqlite_and_postgres() -> None:
    targets = performance_targets()

    assert len(targets) == len(PerformanceMetric) * 2
    for metric in PerformanceMetric:
        assert {target.backend for target in targets if target.metric is metric} == {"sqlite", "postgres"}


def test_storage_growth_benchmark_measures_byte_delta_not_elapsed_time() -> None:
    spec = _performance_spec(PerformanceMetric.STORAGE_GROWTH, "sqlite")
    footprint = {"bytes": 100}

    async def grow_storage() -> None:
        footprint["bytes"] += 37

    measured = asyncio.run(
        SemanticBenchmarkHarness(iterations=3).run(
            spec,
            grow_storage,
            storage_bytes=lambda: footprint["bytes"],
        )
    )

    assert measured.target.unit == "bytes"
    assert measured.samples == (37, 37, 37)
    assert all(type(sample) is int for sample in measured.samples)


def test_storage_growth_benchmark_requires_a_real_byte_reader() -> None:
    spec = _performance_spec(PerformanceMetric.STORAGE_GROWTH, "postgres")

    with pytest.raises(ReleaseEvidenceError, match="storage_bytes"):
        asyncio.run(SemanticBenchmarkHarness(iterations=3).run(spec, lambda: None))


def _cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "kestrel_sovereign.knowledge.release_evidence",
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _telemetry_command(output: Path) -> list[str]:
    return [
        "telemetry",
        "--window-started-at",
        "2026-07-30T00:00:00Z",
        "--window-ended-at",
        "2026-07-31T00:00:00Z",
        "--inventory-digest",
        "d" * 64,
        "--inventory-complete",
        "--unmigrated-eligible-rows",
        "0",
        "--required-consumer-count",
        "0",
        "--artifact-ref",
        _opaque_artifact_ref("retirement-telemetry"),
        "--artifact-digest",
        _opaque_artifact_digest("retirement-telemetry"),
        "--output",
        str(output),
    ]


def _write_record(path: Path, record: EvidenceRecord) -> None:
    path.write_text(json.dumps(record.to_mapping()), encoding="utf-8")


def _write_private_key(path: Path, key_bytes: bytes) -> None:
    path.write_text(key_bytes.hex(), encoding="utf-8")
    path.chmod(0o600)


def _write_structurally_complete_submission(
    tmp_path: Path,
    *,
    core_identity: CatalogSigningIdentity,
    external_identity: CatalogSigningIdentity,
) -> tuple[list[Path], list[Path], Path, Path]:
    records: dict[str, EvidenceRecord] = {}
    record_paths: list[Path] = []
    for spec in release_gate_specs():
        if not spec.advertised:
            continue
        identity = (
            external_identity
            if spec.runner.runner_id == "external_ci"
            else core_identity
        )
        record = _record(spec, identity=identity)
        records[spec.gate_id] = record
        path = tmp_path / f"submitted-{spec.gate_id}.json"
        _write_record(path, record)
        record_paths.append(path)

    budget_paths: list[Path] = []
    for spec in release_gate_specs():
        if spec.performance_target is None:
            continue
        samples: tuple[float | int, ...] = (
            (1, 2, 3)
            if spec.performance_target.unit == "bytes"
            else (1.0, 2.0, 3.0)
        )
        budget = _budget(spec, samples, identity=core_identity)
        path = tmp_path / f"submitted-{spec.gate_id}-budget.json"
        path.write_text(json.dumps(budget.to_mapping()), encoding="utf-8")
        budget_paths.append(path)

    telemetry_path = tmp_path / "submitted-telemetry.json"
    telemetry_path.write_text(
        json.dumps(_retirement_telemetry().to_mapping()), encoding="utf-8"
    )
    external_specs = [
        spec for spec in release_gate_specs() if spec.category == "external_adapter"
    ]
    external_attestations: list[ExternalGateAttestation] = []
    for spec in external_specs:
        record = records[spec.gate_id]
        assert record.run_digest is not None
        assert record.artifact is not None
        assert record.drill is not None
        external_attestations.append(
            ExternalGateAttestation(
                gate_id=spec.gate_id,
                gate_spec_digest=spec.digest,
                result_digest=record.run_digest,
                artifact=record.artifact,
                drill=record.drill,
            )
        )
    external_report = ExternalCapabilityReport.attest(
        capability_id="parametric_self_governed_corpus",
        repository=PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
        source_revision=PARAMETRIC_SELF_EVIDENCE_REVISION,
        attestations=tuple(external_attestations),
    )
    external_report_path = tmp_path / "submitted-external-report.json"
    external_report_path.write_text(
        json.dumps(external_report.to_mapping()), encoding="utf-8"
    )
    return record_paths, budget_paths, telemetry_path, external_report_path


def test_direct_attest_and_public_record_refuse_fabricated_observations(tmp_path: Path) -> None:
    spec = _gate("rdf11_projection_fixture")
    output = tmp_path / "record.json"

    with pytest.raises(ReleaseEvidenceError, match="cannot mint"):
        EvidenceRecord.attest(spec, _observation(spec), _artifact("fabricated"))

    result = _cli(
        "record",
        "--gate",
        spec.gate_id,
        "--observation-json",
        json.dumps(_observation(spec)),
        "--output",
        str(output),
    )

    assert result.returncode != 0
    assert not output.exists()
    assert "disabled" in result.stderr


def test_verified_execution_attestation_rejects_signature_tampering() -> None:
    spec = _gate("rdf11_projection_fixture")
    record = _record(spec)
    assert record.execution_attestation is not None

    with pytest.raises(ReleaseEvidenceError, match="signature verification failed"):
        _apply_records(
            release_evidence_template(),
            (
                replace(
                    record,
                    execution_attestation=replace(record.execution_attestation, signature="0" * 128),
                ),
            ),
        )


def test_trusted_execution_policy_keeps_core_and_external_runner_scopes_separate() -> None:
    core_record = _record(
        _gate("rdf11_projection_fixture"), identity=_EXTERNAL_TEST_IDENTITY
    )
    external_record = _record(
        _gate("external_corpus_consumed"), identity=_CATALOG_TEST_IDENTITY
    )

    with pytest.raises(ReleaseEvidenceError, match="runner is not allowed"):
        _apply_records(release_evidence_template(), (core_record,))
    with pytest.raises(ReleaseEvidenceError, match="runner is not allowed"):
        _apply_records(release_evidence_template(), (external_record,))


def test_catalog_signing_key_loader_requires_a_regular_owner_only_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_bytes = b"\x02" * 32
    key_file = tmp_path / "signing.key"
    _write_private_key(key_file, key_bytes)

    identity = CatalogSigningIdentity.from_private_key_file(
        key_file,
        issuer_id="local_ci",
        key_id="registry_runner",
    )
    assert identity.public_key

    key_file.chmod(0o644)
    with pytest.raises(ReleaseEvidenceError, match="group or other permissions"):
        CatalogSigningIdentity.from_private_key_file(
            key_file,
            issuer_id="local_ci",
            key_id="registry_runner",
        )

    key_file.chmod(0o600)
    symlink = tmp_path / "signing-link.key"
    symlink.symlink_to(key_file)
    with pytest.raises(ReleaseEvidenceError, match="must not be a symlink"):
        CatalogSigningIdentity.from_private_key_file(
            symlink,
            issuer_id="local_ci",
            key_id="registry_runner",
        )

    monkeypatch.setattr(os, "geteuid", lambda: key_file.stat().st_uid + 1)
    with pytest.raises(ReleaseEvidenceError, match="owned by the effective user"):
        CatalogSigningIdentity.from_private_key_file(
            key_file,
            issuer_id="local_ci",
            key_id="registry_runner",
        )


def test_cli_run_executes_an_allowlisted_registry_workload_and_assemble_marks_it_unverified(
    tmp_path: Path,
) -> None:
    spec = _gate("stable_only_capability_selection")
    key_bytes = b"\x02" * 32
    key_file = tmp_path / "signing.key"
    _write_private_key(key_file, key_bytes)
    identity = CatalogSigningIdentity(
        issuer_id="local_ci",
        key_id="registry_runner",
        private_key=Ed25519PrivateKey.from_private_bytes(key_bytes),
    )
    record = tmp_path / "record.json"
    report = tmp_path / "report.json"

    recorded = _cli(
        "run",
        "--gate",
        spec.gate_id,
        "--signing-key-file",
        str(key_file),
        "--issuer-id",
        identity.issuer_id,
        "--key-id",
        identity.key_id,
        "--output",
        str(record),
    )
    assembled = _cli(
        "assemble",
        "--record",
        str(record),
        "--output",
        str(report),
    )

    assert recorded.returncode == 0, recorded.stderr
    assert assembled.returncode == 0, assembled.stderr
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["state"] == "passed"
    assert payload["execution_attestation"]["source"] == "catalog_runner"
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert report_payload["ready"] is False
    assert report_payload["trust_status"] == "unverified"
    assert report_payload["structurally_complete"] is False


def test_cli_run_blocks_a_kite_workload_until_its_dedicated_http_harness_exists(tmp_path: Path) -> None:
    spec = _gate("kite_http_stable_only_release_drill")
    key_file = tmp_path / "signing.key"
    _write_private_key(key_file, b"\x03" * 32)
    record = tmp_path / "record.json"

    result = _cli(
        "run",
        "--gate",
        spec.gate_id,
        "--signing-key-file",
        str(key_file),
        "--issuer-id",
        "local_ci",
        "--key-id",
        "pytest_runner",
        "--output",
        str(record),
    )

    assert result.returncode == 2, result.stderr
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["state"] == "blocked"
    assert payload["reason_code"] == "catalog_workload_unavailable"
    assert payload["execution_attestation"] is None


def test_default_catalog_registers_real_core_pytest_contracts_but_not_kite_or_benchmarks() -> None:
    """Only immutable core test nodes are executable from this runner."""
    from kestrel_sovereign.knowledge.release_evidence_execution import (
        default_catalog_workloads,
    )

    workloads = default_catalog_workloads()
    for gate_id in (
        "rdf11_projection_fixture",
        "rdfs11_inference_fixture",
        "owl2rl_inference_fixture",
        "shacl2017_core_fixture",
        "sparql11_readonly_fixture",
        "sqlite_assertion",
        "postgres_assertion",
        "semantic_maintenance_diagnostics_contract",
        "legacy_fact_migration_equivalence",
    ):
        spec = _gate(gate_id)
        assert (spec.runner.runner_id, spec.runner.command_id) in workloads
    for gate_id in (
        "performance_hybrid_recall_sqlite_integration",
        "kite_http_stable_only_release_drill",
        "stable_persisted_data_no_canonical_migration_drill",
        "erasure_active_assertions",
    ):
        spec = _gate(gate_id)
        assert (spec.runner.runner_id, spec.runner.command_id) not in workloads


def test_cli_run_executes_a_real_rdf_fixture_workload_without_recording_test_arguments(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "signing.key"
    _write_private_key(key_file, b"\x08" * 32)
    record = tmp_path / "rdf-record.json"

    result = _cli(
        "run",
        "--gate",
        "rdf11_projection_fixture",
        "--signing-key-file",
        str(key_file),
        "--issuer-id",
        "local_ci",
        "--key-id",
        "rdf_fixture_runner",
        "--output",
        str(record),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["state"] == "passed"
    assert payload["observation"] == {"assertion_count": 6, "case_count": 6}
    encoded = json.dumps(payload)
    assert "test_rdf11" not in encoded
    assert "::" not in encoded


def test_cli_block_explicitly_records_an_observed_block_with_success_status(tmp_path: Path) -> None:
    output = tmp_path / "blocked.json"

    result = _cli(
        "block",
        "--gate",
        "rdf11_projection_fixture",
        "--reason-code",
        "fixture_service_unavailable",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["state"] == "blocked"


def test_verifier_api_requires_an_explicit_trusted_execution_policy() -> None:
    spec = _gate("rdf11_projection_fixture")
    with pytest.raises(ReleaseEvidenceError, match="explicit TrustedExecutionPolicy"):
        apply_evidence_records(
            release_evidence_template(),
            (_record(spec),),
            trust_policy=None,  # type: ignore[arg-type]
        )


def test_cli_assemble_cannot_self_authorize_a_structurally_complete_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forged_core_identity = CatalogSigningIdentity(
        issuer_id="forged_ci",
        key_id="forged_core_key",
        private_key=Ed25519PrivateKey.from_private_bytes(b"\x06" * 32),
    )
    forged_external_identity = CatalogSigningIdentity(
        issuer_id="forged_external_ci",
        key_id="forged_external_key",
        private_key=Ed25519PrivateKey.from_private_bytes(b"\x07" * 32),
        source=ExecutionSource.EXTERNAL_CI,
    )
    forged_policy = tmp_path / "forged-policy.json"
    core_runner_ids = tuple(
        sorted(
            {
                spec.runner.runner_id
                for spec in release_gate_specs()
                if spec.runner.runner_id != "external_ci"
            }
        )
    )
    forged_policy.write_text(
        json.dumps(
            {
                "keys": [
                    forged_core_identity.trusted_key(core_runner_ids).to_mapping(),
                    forged_external_identity.trusted_key(("external_ci",)).to_mapping(),
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KESTREL_SEMANTIC_RELEASE_TRUST_POLICY_PATH", str(forged_policy))
    monkeypatch.setenv(
        "KESTREL_SEMANTIC_RELEASE_TRUST_POLICY_SHA256",
        hashlib.sha256(forged_policy.read_bytes()).hexdigest(),
    )
    record_paths, budget_paths, telemetry_path, external_report_path = (
        _write_structurally_complete_submission(
            tmp_path,
            core_identity=forged_core_identity,
            external_identity=forged_external_identity,
        )
    )
    output = tmp_path / "report.json"

    arguments = ["assemble"]
    for path in record_paths:
        arguments.extend(("--record", str(path)))
    for path in budget_paths:
        arguments.extend(("--budget", str(path)))
    arguments.extend(
        (
            "--retirement-telemetry",
            str(telemetry_path),
            "--external-report",
            str(external_report_path),
            "--output",
            str(output),
        )
    )
    result = _cli(*arguments)

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["structurally_complete"] is True
    assert payload["trust_status"] == "unverified"
    assert payload["ready"] is False
    assert "trust_verification_required" in payload["blocking_gate_ids"]


def test_cli_assemble_rejects_tampered_record_environment(tmp_path: Path) -> None:
    spec = _gate("rdf11_projection_fixture")
    record = tmp_path / "record.json"
    tampered = tmp_path / "tampered.json"
    output = tmp_path / "report.json"
    _write_record(record, _record(spec))
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["environment"]["backend"] = "postgres"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    result = _cli(
        "assemble",
        "--record",
        str(tampered),
        "--output",
        str(output),
    )

    assert result.returncode == 1
    assert not output.exists()
    assert "environment" in result.stderr


def test_cli_assemble_safely_binds_retirement_and_external_adapter_attestations(
    tmp_path: Path,
) -> None:
    migration = _gate("legacy_fact_migration_equivalence")
    external_specs = [spec for spec in release_gate_specs() if spec.category == "external_adapter"]
    records = {spec.gate_id: _record(spec) for spec in (migration, *external_specs)}
    record_paths: list[Path] = []
    for gate_id, record in records.items():
        path = tmp_path / f"{gate_id}.json"
        _write_record(path, record)
        record_paths.append(path)
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_result = _cli(*_telemetry_command(telemetry_path))
    assert telemetry_result.returncode == 0, telemetry_result.stderr

    report = ExternalCapabilityReport.attest(
        capability_id="parametric_self_governed_corpus",
        repository=PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
        source_revision=PARAMETRIC_SELF_EVIDENCE_REVISION,
        attestations=tuple(
            ExternalGateAttestation(
                gate_id=spec.gate_id,
                gate_spec_digest=spec.digest,
                result_digest=records[spec.gate_id].run_digest,
                artifact=records[spec.gate_id].artifact,
                drill=records[spec.gate_id].drill,
            )
            for spec in external_specs
        ),
    )
    report_path = tmp_path / "external-report.json"
    report_path.write_text(json.dumps(report.to_mapping()), encoding="utf-8")

    output = tmp_path / "report.json"
    arguments = ["assemble"]
    for path in record_paths:
        arguments.extend(("--record", str(path)))
    arguments.extend(
        (
            "--retirement-telemetry",
            str(telemetry_path),
            "--external-report",
            str(report_path),
            "--output",
            str(output),
        )
    )
    assembled = _cli(*arguments)

    assert assembled.returncode == 0, assembled.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["compatibility_retirement"][0]["decision"] == "eligible_for_review"
    assert payload["external_capabilities"][0]["repository"] == PARAMETRIC_SELF_EVIDENCE_REPOSITORY
    assert "external_adapter_attestation" not in payload["blocking_gate_ids"]
    assert payload["ready"] is False
    assert payload["trust_status"] == "unverified"


def test_cli_assemble_rejects_tampered_retirement_telemetry(tmp_path: Path) -> None:
    migration = _gate("legacy_fact_migration_equivalence")
    record = tmp_path / "migration.json"
    telemetry = tmp_path / "telemetry.json"
    output = tmp_path / "report.json"
    _write_record(record, _record(migration))
    assert _cli(*_telemetry_command(telemetry)).returncode == 0
    payload = json.loads(telemetry.read_text(encoding="utf-8"))
    payload["unmigrated_eligible_rows"] = 1
    telemetry.write_text(json.dumps(payload), encoding="utf-8")

    result = _cli(
        "assemble",
        "--record",
        str(record),
        "--retirement-telemetry",
        str(telemetry),
        "--output",
        str(output),
    )

    assert result.returncode == 1
    assert not output.exists()
    assert "telemetry digest" in result.stderr


def test_cli_budget_refuses_caller_supplied_samples(tmp_path: Path) -> None:
    output = tmp_path / "budget.json"

    result = _cli("budget", "--samples", "1", "2", "3", "--output", str(output))

    assert result.returncode != 0
    assert not output.exists()
    assert "disabled" in result.stderr
