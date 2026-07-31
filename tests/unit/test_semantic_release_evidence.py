"""Adversarial contracts for spec-bound semantic release evidence."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

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
from kestrel_sovereign.knowledge.release_evidence_models import SemanticBenchmarkHarness


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
    return ArtifactReference(_opaque_artifact_ref(name), "a" * 64)


def _opaque_artifact_ref(name: str) -> str:
    return f"ci://sha256/{hashlib.sha256(name.encode('utf-8')).hexdigest()}"


def _record(spec):
    return EvidenceRecord.attest(spec, _observation(spec), _artifact(spec.gate_id))


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
            GateResult(spec, replace(record, drill=replace(record.drill, drill_digest="b" * 64)))
        with pytest.raises(ReleaseEvidenceError, match="drill correlation"):
            GateResult(spec, replace(record, drill=None))


def test_reviewer_adversarial_retirement_rejects_unrelated_gate_or_result_reference() -> None:
    template = release_evidence_template()
    migration = _gate("legacy_fact_migration_equivalence")
    unrelated = _record(_gate("rdf11_projection_fixture"))
    with_unrelated = apply_evidence_records(template, (unrelated,))
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
        apply_evidence_records(template, (migration_record,)),
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
    evidence = apply_evidence_records(template, tuple(_record(spec) for spec in external_specs))
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
        GateResult(
            spec,
            replace(record, environment=bad_environment, environment_digest=bad_environment.digest),
        )
    with pytest.raises(ReleaseEvidenceError, match="fixture"):
        GateResult(
            spec,
            replace(record, fixture=replace(record.fixture, fixture_id="other.fixture.v1")),
        )
    with pytest.raises(ReleaseEvidenceError, match="spec digest"):
        GateResult(spec, replace(record, gate_spec_digest="b" * 64))


def test_reviewer_adversarial_record_rejects_wrong_observation_schema() -> None:
    spec = _gate("rdf11_projection_fixture")
    record = _record(spec)

    with pytest.raises(ReleaseEvidenceError, match="observation"):
        GateResult(spec, replace(record, observation={"case_count": 1}))


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


def test_reviewer_adversarial_run_digest_binds_the_complete_opaque_artifact_reference() -> None:
    spec = _gate("rdf11_projection_fixture")
    record = _record(spec)
    assert record.artifact is not None

    with pytest.raises(ReleaseEvidenceError, match="run digest"):
        GateResult(
            spec,
            replace(record, artifact=ArtifactReference(_opaque_artifact_ref("other"), record.artifact.artifact_digest)),
        )

    benchmark = _performance_spec(PerformanceMetric.HYBRID_RECALL, "sqlite")
    budget = PerformanceBudget.from_observed(
        benchmark,
        (1.0, 2.0, 3.0),
        headroom_fraction=0.2,
        artifact=_artifact("benchmark-artifact"),
    )
    with pytest.raises(ReleaseEvidenceError, match="run digest"):
        replace(budget, artifact=ArtifactReference(_opaque_artifact_ref("other-benchmark"), "a" * 64))


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

    with pytest.raises(ReleaseEvidenceError, match="sample_count"):
        PerformanceBudget.from_observed(postgres, (1.0, 2.0), headroom_fraction=0.2, artifact=artifact)
    with pytest.raises(ReleaseEvidenceError, match="positive"):
        PerformanceBudget.from_observed(postgres, (0.0, 1.0, 2.0), headroom_fraction=0.2, artifact=artifact)
    with pytest.raises(ReleaseEvidenceError, match="duration metrics"):
        PerformanceTarget(PerformanceMetric.HYBRID_RECALL, "postgres", "integration", "bytes")
    with pytest.raises(ReleaseEvidenceError, match="storage growth"):
        PerformanceTarget(PerformanceMetric.STORAGE_GROWTH, "postgres", "integration", "ms")


def test_sqlite_only_budget_cannot_satisfy_postgres_or_release_readiness() -> None:
    template = release_evidence_template()
    sqlite = _performance_spec(PerformanceMetric.HYBRID_RECALL, "sqlite")
    sqlite_record = _record(sqlite)
    with_record = apply_evidence_records(template, (sqlite_record,))
    sqlite_budget = PerformanceBudget.from_observed(
        sqlite,
        (1.0, 2.0, 3.0),
        headroom_fraction=0.2,
        artifact=_artifact("sqlite-benchmark"),
    )
    updated = apply_performance_budgets(with_record, (sqlite_budget,))

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


def _record_command(spec, output: Path, artifact_ref: str | None = None) -> list[str]:
    artifact_ref = artifact_ref or _opaque_artifact_ref("record")
    return [
        "record",
        "--gate",
        spec.gate_id,
        "--artifact-ref",
        artifact_ref,
        "--artifact-digest",
        "b" * 64,
        "--observation-json",
        json.dumps(_observation(spec)),
        "--output",
        str(output),
    ]


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
        "e" * 64,
        "--output",
        str(output),
    ]


def test_cli_record_and_assemble_use_the_catalog_not_raw_argv(tmp_path: Path) -> None:
    spec = _gate("rdf11_projection_fixture")
    record = tmp_path / "record.json"
    report = tmp_path / "report.json"

    recorded = _cli(*_record_command(spec, record))
    assembled = _cli("assemble", "--record", str(record), "--output", str(report))

    assert recorded.returncode == 0, recorded.stderr
    assert assembled.returncode == 0, assembled.stderr
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert "command" not in payload
    assert payload["runner_id"] == spec.runner.runner_id
    assert payload["command_digest"] == spec.runner.command_digest
    assert payload["environment"] == spec.environment.to_mapping()
    assert json.loads(report.read_text(encoding="utf-8"))["ready"] is False


def test_cli_record_refuses_arbitrary_true_argv(tmp_path: Path) -> None:
    spec = _gate("rdf11_projection_fixture")
    output = tmp_path / "record.json"

    result = _cli(*_record_command(spec, output), "--", "true")

    assert result.returncode != 0
    assert not output.exists()
    assert "unrecognized arguments" in result.stderr


@pytest.mark.parametrize(
    "reference",
    (
        "postgresql://user:password@db.example/kestrel",
        "ci://semantic-release/42?token=secret",
        "artifact://tenant-42/result",
        "evidence://user@host/result",
    ),
)
def test_cli_record_refuses_sensitive_artifact_reference(tmp_path: Path, reference: str) -> None:
    spec = _gate("rdf11_projection_fixture")
    output = tmp_path / "record.json"

    result = _cli(*_record_command(spec, output, reference))

    assert result.returncode == 1
    assert not output.exists()
    assert "artifact" in result.stderr


def test_cli_record_refuses_environment_backend_mode_overrides(tmp_path: Path) -> None:
    spec = _gate("rdf11_projection_fixture")
    output = tmp_path / "record.json"

    result = _cli(*_record_command(spec, output), "--backend", "postgres", "--mode", "integration")

    assert result.returncode != 0
    assert not output.exists()
    assert "unrecognized arguments" in result.stderr


def test_cli_assemble_rejects_tampered_record_environment(tmp_path: Path) -> None:
    spec = _gate("rdf11_projection_fixture")
    record = tmp_path / "record.json"
    tampered = tmp_path / "tampered.json"
    output = tmp_path / "report.json"
    assert _cli(*_record_command(spec, record)).returncode == 0
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["environment"]["backend"] = "postgres"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    result = _cli("assemble", "--record", str(tampered), "--output", str(output))

    assert result.returncode == 1
    assert not output.exists()
    assert "environment" in result.stderr


def test_cli_assemble_safely_binds_retirement_and_external_adapter_attestations(tmp_path: Path) -> None:
    migration = _gate("legacy_fact_migration_equivalence")
    external_specs = [spec for spec in release_gate_specs() if spec.category == "external_adapter"]
    record_paths: list[Path] = []
    for spec in (migration, *external_specs):
        path = tmp_path / f"{spec.gate_id}.json"
        result = _cli(
            *_record_command(
                spec,
                path,
                artifact_ref=_opaque_artifact_ref(spec.gate_id),
            )
        )
        assert result.returncode == 0, result.stderr
        record_paths.append(path)

    telemetry_path = tmp_path / "telemetry.json"
    telemetry_result = _cli(*_telemetry_command(telemetry_path))
    assert telemetry_result.returncode == 0, telemetry_result.stderr

    records = {
        record.gate_id: record
        for record in (
            evidence_record_from_mapping(json.loads(path.read_text(encoding="utf-8")))
            for path in record_paths
        )
    }
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


def test_cli_assemble_rejects_tampered_retirement_telemetry(tmp_path: Path) -> None:
    migration = _gate("legacy_fact_migration_equivalence")
    record = tmp_path / "migration.json"
    telemetry = tmp_path / "telemetry.json"
    output = tmp_path / "report.json"
    assert _cli(*_record_command(migration, record)).returncode == 0
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


def test_cli_budget_uses_performance_gate_backend_and_mode(tmp_path: Path) -> None:
    spec = _performance_spec(PerformanceMetric.HYBRID_RECALL, "postgres")
    output = tmp_path / "budget.json"

    result = _cli(
        "budget",
        "--gate",
        spec.gate_id,
        "--samples",
        "1",
        "2",
        "3",
        "--headroom-fraction",
        "0.2",
        "--artifact-ref",
        _opaque_artifact_ref("postgres-benchmark"),
        "--artifact-digest",
        "c" * 64,
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["target"]["backend"] == "postgres"
    assert payload["target"]["mode"] == "integration"
    assert payload["target"]["unit"] == "ms"
