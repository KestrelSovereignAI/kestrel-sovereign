"""Adversarial contracts for spec-bound semantic release evidence."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from kestrel_sovereign.knowledge.release_evidence import (
    ArtifactReference,
    EvidenceRecord,
    EvidenceState,
    ExecutionEnvironment,
    GateResult,
    PerformanceBudget,
    PerformanceMetric,
    PerformanceTarget,
    ReleaseEvidenceError,
    apply_evidence_records,
    apply_performance_budgets,
    build_standards_matrix,
    evidence_record_from_mapping,
    inspect_stable_only_capabilities,
    performance_targets,
    release_evidence_template,
    release_gate_specs,
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
    return ArtifactReference(f"ci://semantic-release/{name}", "a" * 64)


def _record(spec):
    return EvidenceRecord.attest(spec, _observation(spec), _artifact(spec.gate_id))


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

    rdf = matrix["rdf11-concepts-20140225"]
    assert rdf.fixture is not None
    assert rdf.fixture.official is True
    assert rdf.fixture.harness_id == "rdf11_projection_harness_v1"
    assert len(rdf.fixture.binding.fixture_digest) == 64


def test_template_never_auto_passes_registry_selection_or_missing_evidence() -> None:
    template = release_evidence_template()

    assert template.ready is False
    assert "stable_only_capability_selection" in template.blocking_gate_ids()
    assert "postgres_assertion" in template.blocking_gate_ids()
    assert "performance_hybrid_recall_postgres_integration" in template.blocking_gate_ids()
    assert all(gate.evidence.state is not EvidenceState.PASSED for gate in template.gates)
    assert inspect_stable_only_capabilities()["rejected_capability_count"] > 0


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
    ),
)
def test_reviewer_adversarial_artifact_references_reject_secrets_and_identity(reference: str) -> None:
    with pytest.raises(ReleaseEvidenceError, match="artifact"):
        ArtifactReference(reference, "a" * 64)


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


def _record_command(spec, output: Path, artifact_ref: str = "ci://semantic-release/record") -> list[str]:
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
        "ci://semantic-release/postgres-benchmark",
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
