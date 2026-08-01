"""Adversarial contracts for spec-bound semantic release evidence."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from kestrel_sovereign.knowledge import release_evidence_verifier as verifier_module
from kestrel_sovereign.knowledge.release_evidence import (
    ArtifactReference,
    CompatibilityRetirementDecision,
    EvidenceRecord,
    EvidenceState,
    ExecutionEnvironment,
    ExternalCapabilityReport,
    ExternalGateAttestation,
    GateResult,
    CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
    PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
    PARAMETRIC_SELF_EVIDENCE_REVISION,
    PerformanceBudget,
    PerformanceMetric,
    PerformanceTarget,
    ReleaseEvidenceError,
    TelemetryAttestation,
    TrustedExecutionPolicy,
    attach_external_capability_report,
    attach_structural_external_capability_report,
    attach_retirement_telemetry,
    apply_evidence_records,
    apply_performance_budgets,
    apply_structural_evidence_records,
    build_standards_matrix,
    evidence_record_from_mapping,
    external_capability_report_from_mapping,
    inspect_stable_only_capabilities,
    performance_targets,
    release_evidence_template,
    release_gate_specs,
    structural_release_evidence_template,
    telemetry_attestation_from_mapping,
)
from kestrel_sovereign.knowledge.release_evidence_freshness import ExternalFreshnessLedger
from kestrel_sovereign.knowledge.release_evidence_execution import CatalogSigningIdentity
from kestrel_sovereign.knowledge.release_evidence_verifier import (
    VerifierReceiptIdentity,
    finalize_verified_artifacts,
    issue_verification_receipt,
    load_external_envelope,
    load_budgets,
    load_external_report,
    load_records,
    combine_external_envelope_submission,
    prepare_trusted_evidence,
    read_verifier_configuration,
    verification_receipt_from_mapping,
    verify_verification_receipt,
)
from kestrel_sovereign.knowledge.release_evidence_models import (
    ExecutionSource,
    SemanticBenchmarkHarness,
    _canonical_json,
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


def test_trusted_execution_policy_rejects_public_key_aliases_across_sources_and_labels() -> None:
    shared_private_key = Ed25519PrivateKey.from_private_bytes(b"\x09" * 32)
    catalog = CatalogSigningIdentity(
        issuer_id="catalog_alias", key_id="catalog_key", private_key=shared_private_key,
    )
    external = CatalogSigningIdentity(
        issuer_id="external_alias", key_id="external_key", private_key=shared_private_key,
        source=ExecutionSource.EXTERNAL_CI,
    )
    with pytest.raises(ReleaseEvidenceError, match="repeats an Ed25519 public key"):
        TrustedExecutionPolicy((
            catalog.trusted_key(("pytest",)),
            external.trusted_key(("external_ci",)),
        ))

    relabeled_catalog = CatalogSigningIdentity(
        issuer_id="catalog_alias_two", key_id="catalog_key_two", private_key=shared_private_key,
    )
    with pytest.raises(ReleaseEvidenceError, match="repeats an Ed25519 public key"):
        TrustedExecutionPolicy((
            catalog.trusted_key(("pytest",)),
            relabeled_catalog.trusted_key(("semantic_benchmark",)),
        ))


def _record(
    spec,
    *,
    observation: dict[str, object] | None = None,
    identity: CatalogSigningIdentity | None = None,
    external_run_nonce: str | None = None,
    external_evidence_runner_revision: str | None = None,
):
    observed = observation or _observation(spec)
    artifact = _artifact(spec.gate_id)
    selected_identity = identity or (
        _EXTERNAL_TEST_IDENTITY
        if spec.runner.runner_id == "external_ci"
        else _CATALOG_TEST_IDENTITY
    )
    external_run_nonce = (
        external_run_nonce or "a" * 64
        if spec.runner.runner_id == "external_ci"
        else None
    )
    external_evidence_runner_revision = (
        external_evidence_runner_revision or "b" * 40
        if spec.runner.runner_id == "external_ci"
        else None
    )
    _, run_digest = EvidenceRecord._bound_run_digest(
        spec,
        observed,
        artifact,
        state=EvidenceState.PASSED,
        external_run_nonce=external_run_nonce,
        external_evidence_runner_revision=external_evidence_runner_revision,
    )
    return EvidenceRecord._from_trusted_execution(
        spec,
        observed,
        artifact,
        state=EvidenceState.PASSED,
        execution_attestation=selected_identity.sign(
            kind="evidence_record", spec=spec, run_digest=run_digest
        ),
        external_run_nonce=external_run_nonce,
        external_evidence_runner_revision=external_evidence_runner_revision,
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


def _external_report(
    evidence,
    *,
    run_nonce: str = "a" * 64,
    evidence_runner_revision: str = "b" * 40,
) -> ExternalCapabilityReport:
    external_gates = [
        gate
        for gate in evidence.gates
        if gate.spec.gate_id.startswith("external_")
    ]
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
        capability_source_revision=PARAMETRIC_SELF_EVIDENCE_REVISION,
        evidence_runner_revision=evidence_runner_revision,
        core_release_evidence_contract_digest=CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
        run_nonce=run_nonce,
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


def test_served_adapter_erasure_gate_is_external_ci_evidence_not_a_core_kite_workload() -> None:
    served = _gate("erasure_served_adapter_eligibility")
    external_gate_ids = tuple(
        spec.gate_id
        for spec in release_gate_specs()
        if spec.category == "external_adapter"
    )

    assert served.category == "external_adapter"
    assert served.owner == "parametric_self"
    assert served.runner.runner_id == "external_ci"
    assert served.environment.mode == "external_adapter"
    assert served.correlation is not None
    assert external_gate_ids == (
        "erasure_served_adapter_eligibility",
        "external_corpus_consumed",
        "external_candidate_invalidated",
        "external_served_eligibility_rejected",
    )
    assert len(CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST) == 64
    assert all(
        spec.gate_id != served.gate_id
        for spec in release_gate_specs()
        if spec.category == "erasure"
    )


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


def test_reviewer_adversarial_external_report_requires_exact_stages_repo_revision_and_artifacts(
    tmp_path: Path,
) -> None:
    template = release_evidence_template()
    external_specs = [spec for spec in release_gate_specs() if spec.category == "external_adapter"]
    ledger = ExternalFreshnessLedger(
        tmp_path / "verifier-freshness.sqlite", trusted_root=tmp_path
    )
    run_nonce = ledger.issue_challenge()
    evidence = _apply_records(
        template,
        tuple(_record(spec, external_run_nonce=run_nonce) for spec in external_specs),
    )
    report = _external_report(evidence, run_nonce=run_nonce)

    assert set(report.gate_ids) == {
        "external_corpus_consumed",
        "external_candidate_invalidated",
        "external_served_eligibility_rejected",
    }
    attached = attach_external_capability_report(
        evidence,
        report,
        freshness_ledger=ledger,
        expected_evidence_runner_revision="b" * 40,
    )
    assert attached.external_capabilities == (report,)
    assert "external_adapter_attestation" not in attached.blocking_gate_ids()

    incomplete = ExternalCapabilityReport.attest(
        capability_id=report.capability_id,
        repository=report.repository,
        capability_source_revision=report.capability_source_revision,
        evidence_runner_revision=report.evidence_runner_revision,
        core_release_evidence_contract_digest=report.core_release_evidence_contract_digest,
        run_nonce=run_nonce,
        attestations=report.attestations[:-1],
    )
    with pytest.raises(ReleaseEvidenceError, match="exactly the three external capability stages"):
        attach_external_capability_report(
            evidence, incomplete, freshness_ledger=ledger, expected_evidence_runner_revision="b" * 40
        )

    wrong_repository = ExternalCapabilityReport.attest(
        capability_id=report.capability_id,
        repository="example/other-adapter",
        capability_source_revision=report.capability_source_revision,
        evidence_runner_revision=report.evidence_runner_revision,
        core_release_evidence_contract_digest=report.core_release_evidence_contract_digest,
        run_nonce="c" * 64,
        attestations=report.attestations,
    )
    with pytest.raises(ReleaseEvidenceError, match="repository or revision"):
        attach_external_capability_report(
            evidence, wrong_repository, freshness_ledger=ledger, expected_evidence_runner_revision="b" * 40
        )

    wrong_core_contract = ExternalCapabilityReport.attest(
        capability_id=report.capability_id,
        repository=report.repository,
        capability_source_revision=report.capability_source_revision,
        evidence_runner_revision=report.evidence_runner_revision,
        core_release_evidence_contract_digest="0" * 64,
        run_nonce="d" * 64,
        attestations=report.attestations,
    )
    with pytest.raises(ReleaseEvidenceError, match="core catalog contract"):
        attach_external_capability_report(
            evidence, wrong_core_contract, freshness_ledger=ledger, expected_evidence_runner_revision="b" * 40
        )

    mismatched_artifact = ExternalCapabilityReport.attest(
        capability_id=report.capability_id,
        repository=report.repository,
        capability_source_revision=report.capability_source_revision,
        evidence_runner_revision=report.evidence_runner_revision,
        core_release_evidence_contract_digest=report.core_release_evidence_contract_digest,
        run_nonce=run_nonce,
        attestations=(
            replace(report.attestations[0], artifact=_artifact("wrong-external-artifact")),
            *report.attestations[1:],
        ),
    )
    with pytest.raises(ReleaseEvidenceError, match="correlated gate result/artifact"):
        attach_external_capability_report(
            evidence, mismatched_artifact, freshness_ledger=ledger, expected_evidence_runner_revision="b" * 40
        )


def test_verifier_external_envelope_requires_each_ordered_served_evidence_record_once(
    tmp_path: Path,
) -> None:
    external_specs = tuple(
        spec for spec in release_gate_specs() if spec.category == "external_adapter"
    )
    records = tuple(
        _record(spec, external_run_nonce="a" * 64) for spec in external_specs
    )
    evidence = _apply_records(release_evidence_template(), records)
    report = _external_report(evidence)
    envelope_mapping = _external_envelope_mapping(records, report)
    envelope_path = tmp_path / "external-envelope.json"
    envelope_path.write_text(json.dumps(envelope_mapping), encoding="utf-8")

    envelope = load_external_envelope(envelope_path)
    core_record = _record(_gate("rdf11_projection_fixture"))
    assembled_records, assembled_report = combine_external_envelope_submission(
        records=(core_record,), envelope=envelope
    )
    assert assembled_records == (core_record, *records)
    assert assembled_report == report

    missing_served = json.loads(json.dumps(envelope_mapping))
    missing_served["records"] = missing_served["records"][:-1]
    envelope_path.write_text(json.dumps(missing_served), encoding="utf-8")
    with pytest.raises(ReleaseEvidenceError, match="declared external gate order"):
        load_external_envelope(envelope_path)

    duplicated_served = json.loads(json.dumps(envelope_mapping))
    duplicated_served["records"].append(duplicated_served["records"][-1])
    envelope_path.write_text(json.dumps(duplicated_served), encoding="utf-8")
    with pytest.raises(ReleaseEvidenceError, match="declared external gate order"):
        load_external_envelope(envelope_path)

    substituted_served = json.loads(json.dumps(envelope_mapping))
    substituted_served["records"][-1] = {
        **substituted_served["records"][0],
        "gate_id": "external_served_eligibility_rejected",
    }
    envelope_path.write_text(json.dumps(substituted_served), encoding="utf-8")
    with pytest.raises(ReleaseEvidenceError, match="spec digest"):
        load_external_envelope(envelope_path)

    mixed_nonce = json.loads(json.dumps(envelope_mapping))
    mixed_nonce["records"][0] = _record(
        external_specs[0], external_run_nonce="b" * 64
    ).to_mapping()
    envelope_path.write_text(json.dumps(mixed_nonce), encoding="utf-8")
    with pytest.raises(ReleaseEvidenceError, match="nonce and runner revision"):
        load_external_envelope(envelope_path)

    mixed_revision = json.loads(json.dumps(envelope_mapping))
    mixed_revision["records"][0] = _record(
        external_specs[0], external_evidence_runner_revision="c" * 40
    ).to_mapping()
    envelope_path.write_text(json.dumps(mixed_revision), encoding="utf-8")
    with pytest.raises(ReleaseEvidenceError, match="nonce and runner revision"):
        load_external_envelope(envelope_path)

    wrong_source = json.loads(json.dumps(envelope_mapping))
    wrong_source["records"][0] = _record(
        external_specs[0], identity=_CATALOG_TEST_IDENTITY
    ).to_mapping()
    envelope_path.write_text(json.dumps(wrong_source), encoding="utf-8")
    with pytest.raises(ReleaseEvidenceError, match="externally signed passes"):
        load_external_envelope(envelope_path)

    served = records[0]
    assert served.run_digest is not None and served.artifact is not None and served.drill
    report_with_served = ExternalCapabilityReport.attest(
        capability_id=report.capability_id,
        repository=report.repository,
        capability_source_revision=report.capability_source_revision,
        evidence_runner_revision=report.evidence_runner_revision,
        core_release_evidence_contract_digest=report.core_release_evidence_contract_digest,
        run_nonce=report.run_nonce,
        attestations=(
            ExternalGateAttestation(
                gate_id=served.gate_id,
                gate_spec_digest=served.gate_spec_digest,
                result_digest=served.run_digest,
                artifact=served.artifact,
                drill=served.drill,
            ),
            *report.attestations,
        ),
    )
    envelope_path.write_text(
        json.dumps(_external_envelope_mapping(records, report_with_served)),
        encoding="utf-8",
    )
    with pytest.raises(ReleaseEvidenceError, match="capability gate order"):
        load_external_envelope(envelope_path)

    unexpected_field = json.loads(json.dumps(envelope_mapping))
    unexpected_field["extra"] = "forbidden"
    envelope_path.write_text(json.dumps(unexpected_field), encoding="utf-8")
    with pytest.raises(ReleaseEvidenceError, match="unknown or missing"):
        load_external_envelope(envelope_path)

    with pytest.raises(ReleaseEvidenceError, match="standalone external records"):
        combine_external_envelope_submission(
            records=(records[-1],), envelope=envelope
        )
    with pytest.raises(ReleaseEvidenceError, match="requires exactly one --external-envelope"):
        combine_external_envelope_submission(
            records=(), envelope=None
        )

    forged_signature = json.loads(json.dumps(envelope_mapping))
    forged_signature["records"][0]["execution_attestation"]["signature"] = "0" * 128
    envelope_path.write_text(json.dumps(forged_signature), encoding="utf-8")
    parsed_forgery = load_external_envelope(envelope_path)
    with pytest.raises(ReleaseEvidenceError, match="signature verification failed"):
        _apply_records(release_evidence_template(), parsed_forgery.records)


def test_external_report_freshness_is_hash_bound_and_replay_protected_across_verifiers(
    tmp_path: Path,
) -> None:
    template = release_evidence_template()
    external_specs = [spec for spec in release_gate_specs() if spec.category == "external_adapter"]
    ledger_path = tmp_path / "independent-verifier.sqlite"
    ledger = ExternalFreshnessLedger(ledger_path, trusted_root=tmp_path)
    run_nonce = ledger.issue_challenge()
    evidence = _apply_records(
        template,
        tuple(_record(spec, external_run_nonce=run_nonce) for spec in external_specs),
    )
    report = _external_report(evidence, run_nonce=run_nonce)

    tampered = report.to_mapping()
    tampered["run_nonce"] = "b" * 64
    with pytest.raises(ReleaseEvidenceError, match="attestation digest"):
        external_capability_report_from_mapping(tampered)
    caller_receipt = report.to_mapping()
    caller_receipt["freshness_receipt"] = "0" * 64
    with pytest.raises(ReleaseEvidenceError, match="unknown or missing"):
        external_capability_report_from_mapping(caller_receipt)
    missing_runner_revision = report.to_mapping()
    del missing_runner_revision["evidence_runner_revision"]
    with pytest.raises(ReleaseEvidenceError, match="unknown or missing"):
        external_capability_report_from_mapping(missing_runner_revision)

    masqueraded_runner = ExternalCapabilityReport.attest(
        capability_id=report.capability_id,
        repository=report.repository,
        capability_source_revision=report.capability_source_revision,
        evidence_runner_revision="c" * 40,
        core_release_evidence_contract_digest=report.core_release_evidence_contract_digest,
        run_nonce=run_nonce,
        attestations=report.attestations,
    )
    with pytest.raises(ReleaseEvidenceError, match="runner revision"):
        attach_external_capability_report(
            evidence, masqueraded_runner, freshness_ledger=ledger, expected_evidence_runner_revision="b" * 40
        )
    with pytest.raises(ReleaseEvidenceError, match="runner revision does not match verifier policy"):
        attach_external_capability_report(
            evidence,
            report,
            freshness_ledger=ledger,
            expected_evidence_runner_revision="c" * 40,
        )

    rewrap_nonce = ledger.issue_challenge()
    rewrapped = ExternalCapabilityReport.attest(
        capability_id=report.capability_id,
        repository=report.repository,
        capability_source_revision=report.capability_source_revision,
        evidence_runner_revision=report.evidence_runner_revision,
        core_release_evidence_contract_digest=report.core_release_evidence_contract_digest,
        run_nonce=rewrap_nonce,
        attestations=report.attestations,
    )
    with pytest.raises(ReleaseEvidenceError, match="served adapter evidence.*nonce"):
        attach_external_capability_report(
            evidence, rewrapped, freshness_ledger=ledger, expected_evidence_runner_revision="b" * 40
        )

    caller_nonce = "f" * 64
    caller_evidence = _apply_records(
        template,
        tuple(_record(spec, external_run_nonce=caller_nonce) for spec in external_specs),
    )
    caller_report = _external_report(caller_evidence, run_nonce=caller_nonce)
    with pytest.raises(ReleaseEvidenceError, match="not an issued pending"):
        attach_external_capability_report(
            caller_evidence, caller_report, freshness_ledger=ledger, expected_evidence_runner_revision="b" * 40
        )

    attached = attach_external_capability_report(
        evidence,
        report,
        freshness_ledger=ledger,
        expected_evidence_runner_revision="b" * 40,
    )
    assert attached.external_capabilities == (report,)

    # A fresh verifier object represents a later process opening the same
    # verifier-owned ledger; SQLite persists the receipt claim across both.
    with pytest.raises(ReleaseEvidenceError, match="already consumed"):
        attach_external_capability_report(
            evidence,
            report,
            freshness_ledger=ExternalFreshnessLedger(ledger_path, trusted_root=tmp_path),
            expected_evidence_runner_revision="b" * 40,
        )


def test_structural_external_attachment_does_not_consume_verifier_freshness(
    tmp_path: Path,
) -> None:
    external_specs = [spec for spec in release_gate_specs() if spec.category == "external_adapter"]
    ledger = ExternalFreshnessLedger(
        tmp_path / "verifier.sqlite", trusted_root=tmp_path
    )
    run_nonce = ledger.issue_challenge()
    records = tuple(
        _record(spec, external_run_nonce=run_nonce) for spec in external_specs
    )
    structural = apply_structural_evidence_records(structural_release_evidence_template(), records)
    verified = _apply_records(release_evidence_template(), records)
    report = _external_report(verified, run_nonce=run_nonce)

    structural_attached = attach_structural_external_capability_report(structural, report)
    assert structural_attached.trust_status == "unverified"
    trusted_attached = attach_external_capability_report(
        verified,
        report,
        freshness_ledger=ledger,
        expected_evidence_runner_revision="b" * 40,
    )
    assert trusted_attached.external_capabilities == (report,)


def test_external_freshness_ledger_rejects_insecure_parent_file_and_symlink_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel_sovereign.knowledge import release_evidence_freshness

    insecure_parent = tmp_path / "insecure"
    insecure_parent.mkdir(mode=0o700)
    insecure_parent.chmod(0o755)
    with pytest.raises(ReleaseEvidenceError, match="no group/other access"):
        ExternalFreshnessLedger(insecure_parent / "ledger.sqlite", trusted_root=insecure_parent)
    with pytest.raises(ReleaseEvidenceError):
        ExternalFreshnessLedger(
            Path("/tmp/kestrel-semantic-release-ledger.sqlite"), trusted_root=Path("/tmp")
        )

    secure_parent = tmp_path / "secure"
    secure_parent.mkdir(mode=0o700)
    nested_secure_parent = secure_parent / "nested"
    nested_secure_parent.mkdir(mode=0o700)
    assert ExternalFreshnessLedger(
        nested_secure_parent / "ledger.sqlite", trusted_root=secure_parent
    ).trusted_root == secure_parent
    outside_parent = tmp_path / "outside"
    outside_parent.mkdir(mode=0o700)
    with pytest.raises(ReleaseEvidenceError, match="cannot contain '..'"):
        ExternalFreshnessLedger(
            secure_parent / ".." / "outside" / "ledger.sqlite",
            trusted_root=secure_parent,
        )

    shared_ancestor = tmp_path / "shared-ancestor"
    shared_ancestor.mkdir(mode=0o700)
    shared_ancestor.chmod(0o777)
    private_root = shared_ancestor / "private-root"
    private_root.mkdir(mode=0o700)
    assert ExternalFreshnessLedger(
        private_root / "ledger.sqlite", trusted_root=private_root
    ).path == private_root / "ledger.sqlite"

    insecure_file = secure_parent / "insecure.sqlite"
    insecure_file.touch()
    insecure_file.chmod(0o644)
    with pytest.raises(ReleaseEvidenceError, match="owner-only access"):
        ExternalFreshnessLedger(insecure_file, trusted_root=secure_parent)

    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(outside_parent, target_is_directory=True)
    with pytest.raises(ReleaseEvidenceError, match="symlink|escapes"):
        ExternalFreshnessLedger(symlink_parent / "ledger.sqlite", trusted_root=tmp_path)

    symlink_file = secure_parent / "symlink.sqlite"
    symlink_file.symlink_to(insecure_file)
    with pytest.raises(ReleaseEvidenceError, match="non-symlink"):
        ExternalFreshnessLedger(symlink_file, trusted_root=secure_parent)

    owner_only_file = secure_parent / "owner.sqlite"
    owner_only_file.touch(mode=0o600)
    current_euid = os.geteuid()
    monkeypatch.setattr(release_evidence_freshness.os, "geteuid", lambda: current_euid + 1)
    with pytest.raises(ReleaseEvidenceError, match="verifier-owned"):
        ExternalFreshnessLedger(owner_only_file, trusted_root=secure_parent)


def test_external_freshness_ledger_fails_closed_if_path_replaced_while_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel_sovereign.knowledge import release_evidence_freshness

    external_specs = [spec for spec in release_gate_specs() if spec.category == "external_adapter"]
    evidence = _apply_records(
        release_evidence_template(),
        tuple(_record(spec) for spec in external_specs),
    )
    report = _external_report(evidence)
    secure_parent = tmp_path / "secure"
    secure_parent.mkdir(mode=0o700)
    ledger_path = secure_parent / "ledger.sqlite"
    replacement = secure_parent / "replacement.sqlite"
    replacement.touch(mode=0o600)
    ledger = ExternalFreshnessLedger(ledger_path, trusted_root=secure_parent)
    original_connect = release_evidence_freshness.sqlite3.connect

    def replace_then_connect(*args: object, **kwargs: object):
        ledger_path.replace(secure_parent / "original.sqlite")
        replacement.replace(ledger_path)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(release_evidence_freshness.sqlite3, "connect", replace_then_connect)
    with pytest.raises(ReleaseEvidenceError, match="changed while opening"):
        ledger.consume(report)


def test_external_freshness_ledger_fails_closed_if_trusted_root_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel_sovereign.knowledge import release_evidence_freshness

    trusted_root = tmp_path / "trusted-root"
    trusted_root.mkdir(mode=0o700)
    ledger_path = trusted_root / "ledger.sqlite"
    ledger = ExternalFreshnessLedger(ledger_path, trusted_root=trusted_root)
    original_connect = release_evidence_freshness.sqlite3.connect

    def replace_root_then_connect(*args: object, **kwargs: object):
        trusted_root.replace(tmp_path / "original-trusted-root")
        replacement_root = tmp_path / "replacement-root"
        replacement_root.mkdir(mode=0o700)
        (replacement_root / "ledger.sqlite").touch(mode=0o600)
        replacement_root.replace(trusted_root)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(release_evidence_freshness.sqlite3, "connect", replace_root_then_connect)
    with pytest.raises(ReleaseEvidenceError, match="changed while opening"):
        ledger.issue_challenge()


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


def test_duration_benchmark_excludes_prepared_sample_setup_from_the_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-startup benchmark callers may prepare an isolated sample untimed."""
    from kestrel_sovereign.knowledge import release_evidence_models

    spec = _performance_spec(PerformanceMetric.HYBRID_RECALL, "sqlite")
    events: list[str] = []
    timestamps = iter((10.0, 10.002, 20.0, 20.002, 30.0, 30.002))

    def perf_counter() -> float:
        events.append("timer")
        return next(timestamps)

    async def prepare() -> None:
        events.append("setup")

    async def operation() -> None:
        events.append("operation")

    async def close() -> None:
        events.append("teardown")

    monkeypatch.setattr(release_evidence_models.time, "perf_counter", perf_counter)
    measured = asyncio.run(
        SemanticBenchmarkHarness(iterations=3).run(
            spec,
            operation,
            before_sample=prepare,
            after_sample=close,
        )
    )

    assert measured.samples == pytest.approx((2.0, 2.0, 2.0))
    assert events == [
        "setup", "timer", "operation", "timer", "teardown",
        "setup", "timer", "operation", "timer", "teardown",
        "setup", "timer", "operation", "timer", "teardown",
    ]


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


def _verifier_cli(config: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Exercise the fixed-locator executable with a test-process patch only.

    Production's ``python -m`` entry point has no config option or environment
    override.  Tests patch its module constant before calling ``main`` so they
    never need to create or modify the real host administrator's ``/etc``
    authority file.
    """
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                "from kestrel_sovereign.knowledge import release_evidence_verifier_cli as cli; "
                "cli.HOST_VERIFIER_CONFIGURATION = Path(sys.argv[1]); "
                "raise SystemExit(cli.main(sys.argv[2:]))"
            ),
            str(config),
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


def _write_protected_verifier_configuration(
    root: Path,
    *,
    declared_root: Path | None = None,
    config_relative: Path = Path("verifier.json"),
) -> Path:
    """Create one complete private verifier configuration for locator tests."""
    root.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    config_parent = root / config_relative.parent
    config_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    config_parent.chmod(0o700)
    receipt_bytes = b"\x07" * 32
    receipt_key = root / "receipt.key"
    _write_private_key(receipt_key, receipt_bytes)
    receipt_public = Ed25519PrivateKey.from_private_bytes(receipt_bytes).public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    locator_root = declared_root or root
    policy = {
        "keys": [
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
            ).to_mapping(),
        ]
    }
    config = locator_root / config_relative
    config.write_text(
        json.dumps(
            {
                "trusted_root": str(locator_root),
                "ledger_path": str(locator_root / "ledger.sqlite"),
                "trust_policy": policy,
                "expected_external_runner_revision": "b" * 40,
                "receipt_key_file": str(locator_root / "receipt.key"),
                "receipt_issuer_id": "verifier_ci",
                "receipt_key_id": "semantic_release",
                "receipt_public_key": receipt_public,
                "verifier_role": "semantic_release_verifier",
            }
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    return config


def _external_envelope_mapping(
    records: tuple[EvidenceRecord, ...], report: ExternalCapabilityReport,
) -> dict[str, object]:
    by_gate = {record.gate_id: record for record in records}
    external_records = [
        by_gate[spec.gate_id].to_mapping()
        for spec in release_gate_specs()
        if spec.category == "external_adapter"
    ]
    return {
        "core_release_evidence_contract_digest": report.core_release_evidence_contract_digest,
        "repository": report.repository,
        "capability_source_revision": report.capability_source_revision,
        "evidence_runner_revision": report.evidence_runner_revision,
        "run_nonce": report.run_nonce,
        "trust_status": "external_signature_requires_core_policy_verification",
        "records": external_records,
        "report": report.to_mapping(),
    }


def _write_structurally_complete_submission(
    tmp_path: Path,
    *,
    core_identity: CatalogSigningIdentity,
    external_identity: CatalogSigningIdentity,
    external_run_nonce: str = "a" * 64,
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
        record = _record(
            spec,
            identity=identity,
            external_run_nonce=external_run_nonce,
        )
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
        spec for spec in release_gate_specs() if spec.gate_id.startswith("external_")
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
        capability_source_revision=PARAMETRIC_SELF_EVIDENCE_REVISION,
        evidence_runner_revision="b" * 40,
        core_release_evidence_contract_digest=CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
        run_nonce=external_run_nonce,
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


def test_verifier_configuration_canonicalizes_an_alias_above_its_declared_root(
    tmp_path: Path,
) -> None:
    """A fixed ``/etc``-style alias must not defeat rooted containment."""
    physical_parent = tmp_path / "private"
    physical_root = physical_parent / "etc" / "kestrel"
    physical_parent.mkdir(mode=0o700)
    system_alias = tmp_path / "etc"
    system_alias.symlink_to(physical_parent / "etc", target_is_directory=True)
    declared_root = system_alias / "kestrel"
    config = _write_protected_verifier_configuration(
        physical_root, declared_root=declared_root,
    )

    configuration = read_verifier_configuration(config)

    assert configuration.trusted_root == physical_root
    assert configuration.ledger_path == physical_root / "ledger.sqlite"
    assert configuration.receipt_key_file == physical_root / "receipt.key"


def test_verifier_configuration_rejects_a_symlink_below_its_declared_root(
    tmp_path: Path,
) -> None:
    """Canonicalizing a host alias must never hide a child-directory symlink."""
    root = tmp_path / "verifier"
    nested_config = _write_protected_verifier_configuration(
        root, config_relative=Path("nested/verifier.json"),
    )
    config_alias = root / "config-link"
    config_alias.symlink_to(root / "nested", target_is_directory=True)

    with pytest.raises(ReleaseEvidenceError, match="private non-symlink"):
        read_verifier_configuration(config_alias / nested_config.name)


def test_verifier_configuration_rechecks_a_replaced_child_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second config read must retain its before/after parent inode check."""
    root = tmp_path / "verifier"
    config = _write_protected_verifier_configuration(
        root, config_relative=Path("config/verifier.json"),
    )
    replacement = root / "replacement"
    replacement.mkdir(mode=0o700)
    replacement.chmod(0o700)
    replacement_config = replacement / config.name
    replacement_config.write_bytes(config.read_bytes())
    replacement_config.chmod(0o600)
    original_parent = config.parent
    original_reader = verifier_module._read_owner_only_file
    reads = 0

    def replace_after_rooted_read(path: Path, *, kind: str) -> bytes:
        nonlocal reads
        result = original_reader(path, kind=kind)
        if kind == "verifier configuration":
            reads += 1
            if reads == 2:
                original_parent.rename(root / "config-original")
                replacement.rename(original_parent)
        return result

    monkeypatch.setattr(verifier_module, "_read_owner_only_file", replace_after_rooted_read)

    with pytest.raises(ReleaseEvidenceError, match="parent changed while being used"):
        read_verifier_configuration(config)


def test_verifier_cli_requires_protected_config_and_consumes_one_external_challenge(
    tmp_path: Path,
) -> None:
    root = tmp_path / "verifier"
    root.mkdir(mode=0o700)
    receipt_key = root / "receipt.key"
    receipt_bytes = b"\x07" * 32
    _write_private_key(receipt_key, receipt_bytes)
    receipt_public = Ed25519PrivateKey.from_private_bytes(receipt_bytes).public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    policy = {
        "keys": [
            _CATALOG_TEST_IDENTITY.trusted_key(
                tuple(sorted({spec.runner.runner_id for spec in release_gate_specs() if spec.runner.runner_id != "external_ci"}))
            ).to_mapping(),
            _EXTERNAL_TEST_IDENTITY.trusted_key(("external_ci",)).to_mapping(),
        ]
    }
    config = root / "verifier.json"
    config_mapping = {
        "trusted_root": str(root), "ledger_path": str(root / "ledger.sqlite"),
        "trust_policy": policy, "expected_external_runner_revision": "b" * 40,
        "receipt_key_file": str(receipt_key), "receipt_issuer_id": "verifier_ci",
        "receipt_key_id": "semantic_release", "receipt_public_key": receipt_public,
        "verifier_role": "semantic_release_verifier",
    }
    config.write_text(json.dumps(config_mapping), encoding="utf-8")
    config.chmod(0o600)
    challenge = _verifier_cli(config, "issue-challenge")
    assert challenge.returncode == 0, challenge.stderr
    nonce = challenge.stdout.strip()
    records, budgets, _, report = _write_structurally_complete_submission(
        tmp_path, core_identity=_CATALOG_TEST_IDENTITY, external_identity=_EXTERNAL_TEST_IDENTITY,
        external_run_nonce=nonce,
    )
    submitted_records = load_records(records)
    external_report = load_external_report(report)
    envelope_path = tmp_path / "parametric-self-envelope.json"
    envelope_path.write_text(
        json.dumps(_external_envelope_mapping(submitted_records, external_report)),
        encoding="utf-8",
    )
    output, receipt = root / "verified.json", root / "receipt.json"
    args = ["assemble", "--external-envelope", str(envelope_path), "--output", str(output), "--receipt-output", str(receipt)]
    external_record_path: Path | None = None
    for record_path, submitted_record in zip(records, submitted_records, strict=True):
        if submitted_record.runner_id == "external_ci":
            external_record_path = record_path
            continue
        args.extend(("--record", str(record_path)))
    for budget in budgets:
        args.extend(("--budget", str(budget)))
    assert external_record_path is not None

    configuration = read_verifier_configuration(config)
    ledger = ExternalFreshnessLedger(configuration.ledger_path, trusted_root=configuration.trusted_root)
    prepared = prepare_trusted_evidence(
        records=submitted_records, budgets=load_budgets(budgets), report=external_report,
        trust_policy=configuration.trust_policy,
        expected_evidence_runner_revision=configuration.expected_external_runner_revision,
    )
    receipt_identity = VerifierReceiptIdentity.from_configuration(configuration)
    valid_receipt = issue_verification_receipt(
        prepared, policy_digest=configuration.policy_digest, identity=receipt_identity,
    )
    other_evidence = attach_retirement_telemetry(prepared, _retirement_telemetry())
    receipt_for_other_evidence = issue_verification_receipt(
        other_evidence, policy_digest=configuration.policy_digest, identity=receipt_identity,
    )
    with pytest.raises(ReleaseEvidenceError, match="not bound to the exact evidence"):
        finalize_verified_artifacts(
            prepared, receipt_for_other_evidence,
            evidence_output=root / "wrong-evidence.json",
            receipt_output=root / "wrong-evidence-receipt.json",
            configuration=configuration, freshness_ledger=ledger,
        )

    tampered_report_receipt = replace(
        valid_receipt, capability_source_revision="c" * 40, signature="0" * 128,
    )
    tampered_report_receipt = replace(
        tampered_report_receipt,
        signature=receipt_identity.private_key.sign(
            _canonical_json(tampered_report_receipt.signed_payload()).encode("utf-8")
        ).hex(),
    )
    with pytest.raises(ReleaseEvidenceError, match="not bound to the exact evidence"):
        finalize_verified_artifacts(
            prepared, tampered_report_receipt,
            evidence_output=root / "tampered-report.json",
            receipt_output=root / "tampered-report-receipt.json",
            configuration=configuration, freshness_ledger=ledger,
        )

    with pytest.raises(ReleaseEvidenceError, match="signature verification failed"):
        finalize_verified_artifacts(
            prepared, replace(valid_receipt, signature="0" * 128),
            evidence_output=root / "forged.json", receipt_output=root / "forged-receipt.json",
            configuration=configuration, freshness_ledger=ledger,
        )
    assert not any(root.glob("wrong-evidence*"))
    assert not any(root.glob("tampered-report*"))
    assert not any(root.glob("forged*"))

    for duplicate_path in (envelope_path, tmp_path / "different-envelope.json"):
        duplicate_envelope = _verifier_cli(
            config,
            *args,
            "--external-envelope",
            str(duplicate_path),
        )
        assert duplicate_envelope.returncode == 2
        assert "--external-envelope may be supplied only once" in duplicate_envelope.stderr
    assert not output.exists()
    assert not receipt.exists()
    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute(
            "SELECT state FROM external_freshness_challenges WHERE run_nonce = ?",
            (nonce,),
        ).fetchone() == ("pending",)

    duplicated_external_record = _verifier_cli(
        config, *args, "--record", str(external_record_path)
    )
    assert duplicated_external_record.returncode == 1
    assert "standalone external records" in duplicated_external_record.stderr

    split_external_report = _verifier_cli(
        config, *args, "--external-report", str(report)
    )
    assert split_external_report.returncode == 2
    assert "unrecognized arguments" in split_external_report.stderr

    # Output failure happens after evidence preparation but before ledger
    # finalization.  The same verifier-issued challenge must remain usable.
    receipt.write_text("impostor\n", encoding="utf-8")
    receipt.chmod(0o600)
    failed_output = _verifier_cli(config, *args)
    assert failed_output.returncode == 1
    assert not output.exists()
    assert (root / ".verified.json.semantic-release-pending").exists()
    receipt.unlink()
    assembled = _verifier_cli(config, *args)
    assert assembled.returncode == 0, assembled.stderr
    assert json.loads(output.read_text())["ready"] is True
    receipt_mapping = json.loads(receipt.read_text())
    assert receipt_mapping["evidence_runner_revision"] == "b" * 40
    assert receipt.stat().st_mode & 0o777 == 0o600
    issued_receipt = verification_receipt_from_mapping(receipt_mapping)
    verify_verification_receipt(issued_receipt, read_verifier_configuration(config))
    tampered_contract = {**receipt_mapping, "core_release_evidence_contract_digest": "0" * 64}
    with pytest.raises(ReleaseEvidenceError, match="unknown or missing"):
        verification_receipt_from_mapping(tampered_contract)
    replay_args = [*args]
    replay_args[replay_args.index(str(output))] = str(root / "replay.json")
    replay_args[replay_args.index(str(receipt))] = str(root / "replay-receipt.json")
    replay = _verifier_cli(config, *replay_args)
    assert replay.returncode == 1
    assert "different verifier evidence" in replay.stderr

    dangling_output = root / "dangling.json"
    dangling_output.symlink_to(root / "missing.json")
    dangling_args = [*args]
    dangling_args[dangling_args.index(str(output))] = str(dangling_output)
    dangling = _verifier_cli(config, *dangling_args)
    assert dangling.returncode == 1
    assert "owner-only regular non-symlink" in dangling.stderr

    # The public executable itself has no caller-selectable configuration
    # boundary; the test-only helper above patches its module constant only in
    # the disposable child process.
    public_config_override = subprocess.run(
        [
            sys.executable, "-m",
            "kestrel_sovereign.knowledge.release_evidence_verifier_cli",
            "--config", str(config), "issue-challenge",
        ],
        check=False, capture_output=True, text=True,
    )
    assert public_config_override.returncode == 2
    assert "invalid choice" in public_config_override.stderr

    wrong_role_mapping = json.loads(config.read_text(encoding="utf-8"))
    wrong_role_mapping["verifier_role"] = "catalog_runner"
    wrong_role = root / "wrong-role.json"
    wrong_role.write_text(json.dumps(wrong_role_mapping), encoding="utf-8")
    wrong_role.chmod(0o600)
    rejected_role = _verifier_cli(wrong_role, "issue-challenge")
    assert rejected_role.returncode == 1
    assert "semantic_release_verifier role" in rejected_role.stderr

    aliased_key_mapping = {**config_mapping, "receipt_public_key": policy["keys"][0]["public_key"]}
    aliased_key = root / "aliased-key.json"
    aliased_key.write_text(json.dumps(aliased_key_mapping), encoding="utf-8")
    aliased_key.chmod(0o600)
    rejected_alias = _verifier_cli(aliased_key, "issue-challenge")
    assert rejected_alias.returncode == 1
    assert "public key must be distinct" in rejected_alias.stderr

    nested = root / "nested"
    nested.mkdir(mode=0o700)
    nested_config = nested / "verifier.json"
    config.replace(nested_config)
    symlinked_config_parent = root / "config-link"
    symlinked_config_parent.symlink_to(nested, target_is_directory=True)
    escaped = _verifier_cli(symlinked_config_parent / "verifier.json", "issue-challenge")
    assert escaped.returncode == 1
    assert "private non-symlink" in escaped.stderr

    nested_config.chmod(0o644)
    denied = _verifier_cli(nested_config, "issue-challenge")
    assert denied.returncode == 1
    assert "owner-only" in denied.stderr


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


def test_default_catalog_registers_real_core_contracts_but_not_kite_http_drills() -> None:
    """Core test/benchmark runners remain distinct from real Kite HTTP work."""
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
        "performance_hybrid_recall_sqlite_integration",
        "performance_hybrid_recall_postgres_integration",
    ):
        spec = _gate(gate_id)
        assert (spec.runner.runner_id, spec.runner.command_id) in workloads
    for gate_id in (
        "kite_http_stable_only_release_drill",
        "stable_persisted_data_no_canonical_migration_drill",
    ):
        spec = _gate(gate_id)
        assert (spec.runner.runner_id, spec.runner.command_id) not in workloads
    for gate_id in (
        "erasure_active_assertions",
        "erasure_vector_index",
        "erasure_governed_corpus",
        "erasure_future_corpus",
    ):
        spec = _gate(gate_id)
        assert (spec.runner.runner_id, spec.runner.command_id) in workloads


def test_catalog_benchmark_runs_three_real_isolated_sqlite_startup_samples() -> None:
    from kestrel_sovereign.knowledge.release_evidence_execution import (
        CatalogExecutionAuthority,
        default_catalog_workloads,
    )

    execution = asyncio.run(
        CatalogExecutionAuthority(_CATALOG_TEST_IDENTITY, default_catalog_workloads()).execute(
            _gate("performance_startup_sqlite_startup")
        )
    )

    assert execution.record.passed
    assert execution.budget is not None
    assert len(execution.budget.samples) == 3
    assert all(sample > 0 for sample in execution.budget.samples)


@pytest.mark.parametrize(
    "gate_id",
    (
        "performance_assertion_write_validation_sqlite_integration",
        "performance_bounded_inference_sqlite_integration",
        "performance_hybrid_recall_sqlite_integration",
        "performance_storage_growth_sqlite_integration",
        "performance_representative_migration_sqlite_integration",
    ),
)
def test_catalog_benchmark_runs_real_isolated_sqlite_semantic_workload(gate_id: str) -> None:
    from kestrel_sovereign.knowledge.release_evidence_execution import (
        CatalogExecutionAuthority,
        default_catalog_workloads,
    )

    execution = asyncio.run(
        CatalogExecutionAuthority(_CATALOG_TEST_IDENTITY, default_catalog_workloads()).execute(
            _gate(gate_id)
        )
    )

    assert execution.record.passed
    assert execution.budget is not None
    assert len(execution.budget.samples) == 3
    if _gate(gate_id).performance_target.unit == "bytes":
        assert all(type(sample) is int and sample > 0 for sample in execution.budget.samples)
    else:
        assert all(sample > 0 for sample in execution.budget.samples)


def test_catalog_benchmark_blocks_postgres_without_explicit_isolated_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel_sovereign.knowledge.release_evidence_execution import (
        CatalogExecutionAuthority,
        default_catalog_workloads,
    )

    monkeypatch.delenv("KESTREL_SEMANTIC_RELEASE_ISOLATED", raising=False)
    monkeypatch.delenv("KESTREL_SEMANTIC_RELEASE_ISOLATED_POSTGRES_ADMIN_DSN", raising=False)
    execution = asyncio.run(
        CatalogExecutionAuthority(_CATALOG_TEST_IDENTITY, default_catalog_workloads()).execute(
            _gate("performance_startup_postgres_startup")
        )
    )

    assert execution.record.state is EvidenceState.BLOCKED
    assert execution.record.reason_code == "isolated_postgres_ack_required"


def test_catalog_benchmark_blocks_postgres_without_an_isolated_admin_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel_sovereign.knowledge.release_evidence_execution import (
        CatalogExecutionAuthority,
        default_catalog_workloads,
    )

    monkeypatch.setenv("KESTREL_SEMANTIC_RELEASE_ISOLATED", "1")
    monkeypatch.delenv("KESTREL_SEMANTIC_RELEASE_ISOLATED_POSTGRES_ADMIN_DSN", raising=False)
    execution = asyncio.run(
        CatalogExecutionAuthority(_CATALOG_TEST_IDENTITY, default_catalog_workloads()).execute(
            _gate("performance_startup_postgres_startup")
        )
    )

    assert execution.record.state is EvidenceState.BLOCKED
    assert execution.record.reason_code == "isolated_postgres_admin_unavailable"


@pytest.mark.parametrize("failure_point", ("fetchval", "create_database"))
def test_disposable_postgres_closes_admin_connection_when_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    from types import SimpleNamespace

    from kestrel_sovereign.knowledge.release_evidence_execution import CatalogWorkloadUnavailable
    from kestrel_sovereign.knowledge.release_evidence_postgres import DisposablePostgresDatabase

    class FailingConnection:
        closed = False

        async def fetchval(self, _query: str):
            if failure_point == "fetchval":
                raise RuntimeError("admin fetch failed")
            return "postgres"

        async def execute(self, _query: str):
            if failure_point == "create_database":
                raise RuntimeError("create failed")

        async def close(self) -> None:
            self.closed = True

    connection = FailingConnection()

    async def connect(_dsn: str) -> FailingConnection:
        return connection

    monkeypatch.setenv("KESTREL_SEMANTIC_RELEASE_ISOLATED", "1")
    monkeypatch.setenv(
        "KESTREL_SEMANTIC_RELEASE_ISOLATED_POSTGRES_ADMIN_DSN",
        "postgresql://admin@localhost/postgres",
    )
    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(connect=connect))

    with pytest.raises(CatalogWorkloadUnavailable, match="isolated_postgres_database_create_failed"):
        asyncio.run(DisposablePostgresDatabase.create())
    assert connection.closed is True


@pytest.mark.parametrize("failure_point", ("factory", "temporary_directory"))
def test_postgres_benchmark_closes_disposable_database_when_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    """A setup failure after DB creation must still drop the isolated DB."""
    from kestrel_sovereign.knowledge import release_evidence_benchmarks as benchmarks

    class DisposableDatabase:
        dsn = "postgresql://isolated/kestrel_semantic_release_0123456789abcdef0123456789abcdef"
        closed = False

        async def close(self) -> None:
            self.closed = True

    database = DisposableDatabase()

    async def create() -> DisposableDatabase:
        return database

    monkeypatch.setattr(
        benchmarks.DisposablePostgresDatabase,
        "create",
        staticmethod(create),
    )
    if failure_point == "factory":
        def fail_factory(*_args, **_kwargs):
            raise RuntimeError("factory failed")

        monkeypatch.setattr(benchmarks, "_IsolatedStorageFactory", fail_factory)
    else:
        def fail_temporary_directory(*_args, **_kwargs):
            raise OSError("temporary directory failed")

        monkeypatch.setattr(benchmarks, "TemporaryDirectory", fail_temporary_directory)

    with pytest.raises((RuntimeError, OSError), match="failed"):
        asyncio.run(benchmarks._run_benchmark(_gate("performance_startup_postgres_startup")))
    assert database.closed is True


@pytest.mark.parametrize(
    "shared_name",
    ("postgres", "template0", "template1", "kestrel", "kestrel_test", "test"),
)
def test_disposable_postgres_identity_rejects_common_or_shared_database_names(
    shared_name: str,
) -> None:
    from kestrel_sovereign.knowledge.release_evidence_postgres import _quoted_identifier

    with pytest.raises(ReleaseEvidenceError, match="generated disposable"):
        _quoted_identifier(shared_name)


def test_parity_child_environment_never_inherits_an_ambient_postgres_dsn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kestrel_sovereign.knowledge.release_evidence_workloads import _isolated_test_environment

    monkeypatch.setenv("TEST_POSTGRES_URL", "postgresql://ambient-shared-db/kestrel")
    ambient_free = _isolated_test_environment(str(tmp_path))
    generated = _isolated_test_environment(
        str(tmp_path),
        postgres_dsn="postgresql://isolated-runner/kestrel_semantic_release_0123456789abcdef0123456789abcdef",
    )

    assert "TEST_POSTGRES_URL" not in ambient_free
    assert generated["TEST_POSTGRES_URL"] != "postgresql://ambient-shared-db/kestrel"
    assert "kestrel_semantic_release_" in generated["TEST_POSTGRES_URL"]


@pytest.mark.parametrize(
    "gate_id",
    (
        "performance_changed_work_sleep_sqlite_kite_http",
        "performance_changed_work_sleep_postgres_kite_http",
        "performance_unchanged_sleep_sqlite_kite_http",
        "performance_unchanged_sleep_postgres_kite_http",
    ),
)
def test_catalog_benchmark_refuses_to_relabel_inprocess_sleep_as_kite_http(gate_id: str) -> None:
    from kestrel_sovereign.knowledge.release_evidence_execution import (
        CatalogExecutionAuthority,
        default_catalog_workloads,
    )

    execution = asyncio.run(
        CatalogExecutionAuthority(_CATALOG_TEST_IDENTITY, default_catalog_workloads()).execute(
            _gate(gate_id)
        )
    )

    assert execution.record.state is EvidenceState.BLOCKED
    assert execution.record.reason_code == "kite_http_benchmark_runner_required"


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
        capability_source_revision=PARAMETRIC_SELF_EVIDENCE_REVISION,
        evidence_runner_revision="b" * 40,
        core_release_evidence_contract_digest=CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
        run_nonce="a" * 64,
        attestations=tuple(
            ExternalGateAttestation(
                gate_id=spec.gate_id,
                gate_spec_digest=spec.digest,
                result_digest=records[spec.gate_id].run_digest,
                artifact=records[spec.gate_id].artifact,
                drill=records[spec.gate_id].drill,
            )
            for spec in external_specs
            if spec.gate_id.startswith("external_")
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
