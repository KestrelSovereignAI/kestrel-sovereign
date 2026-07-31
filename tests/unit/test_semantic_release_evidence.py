"""Contracts for the semantic release-evidence runner (#2753)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys

import pytest

from kestrel_sovereign.knowledge.release_evidence import (
    EvidenceRecord,
    EvidenceState,
    GateResult,
    GateSpec,
    PerformanceBudget,
    PerformanceMetric,
    ReleaseEvidenceError,
    apply_evidence_records,
    apply_performance_budgets,
    build_standards_matrix,
    evidence_record_from_mapping,
    inspect_stable_only_capabilities,
    release_evidence_template,
    release_gate_specs,
)
from kestrel_sovereign.knowledge.release_evidence_models import SemanticBenchmarkHarness


def _passed(gate_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        gate_id=gate_id,
        state=EvidenceState.PASSED,
        command="uv run pytest tests/unit/test_semantic_release_evidence.py -q",
        exit_code=0,
        artifact_ref="docs/architecture/testing/SEMANTIC_RELEASE_EVIDENCE.md",
    )


def _gate(template, gate_id: str) -> GateResult:
    return next(item for item in template.gates if item.spec.gate_id == gate_id)


def test_matrix_records_exact_verified_registry_pins_and_runtime_versions() -> None:
    evidence = release_evidence_template()

    matrix = {entry.identifier: entry for entry in build_standards_matrix()}
    assert matrix["rdf11-concepts-20140225"].version == "1.0.0"
    assert matrix["rdf11-concepts-20140225"].maturity == "stable"
    assert matrix["shacl-core-20170720"].kind == "validation-profile"
    assert matrix["rdf12-cr-20260407"].maturity == "experimental"
    assert evidence.libraries["rdflib"] != ""
    assert evidence.libraries["python"] != ""
    assert "rdf-profile:rdf11" in evidence.profile[
        "advertised_stable_capabilities"
    ]


def test_catalog_declares_every_required_surface_once() -> None:
    specs = release_gate_specs()
    assert {
        spec.scope for spec in specs if spec.category == "conformance" and spec.advertised
    } == {
        "rdf11_projection",
        "rdfs11_inference",
        "owl2rl_inference",
        "shacl2017_core",
        "sparql11_readonly",
    }
    backend_scopes = {
        spec.scope for spec in specs if spec.category == "backend_parity"
    }
    assert backend_scopes == {
        f"{backend}:{surface}"
        for backend in ("sqlite", "postgres")
        for surface in (
            "assertion",
            "ownership_privacy",
            "validation",
            "inference_retraction",
            "migration",
            "corpus_export",
            "retrieval",
        )
    }
    assert {spec.scope for spec in specs if spec.category == "erasure"} == {
        "active_assertions",
        "derivations",
        "vector_index",
        "recall_candidates",
        "export_snapshots",
        "governed_corpus",
        "future_corpus",
        "projection_candidates",
        "served_adapter_eligibility",
    }
    external = [spec.gate_id for spec in specs if spec.category == "external_adapter"]
    assert external == [
        "external_corpus_consumed",
        "external_candidate_invalidated",
        "external_served_eligibility_rejected",
    ]


def test_template_is_conservative_about_unobserved_release_gates() -> None:
    evidence = release_evidence_template()

    assert evidence.ready is False
    assert "rdf11_projection_fixture" in evidence.blocking_gate_ids()
    assert "postgres_assertion" in evidence.blocking_gate_ids()
    assert "performance_hybrid_recall" in evidence.blocking_gate_ids()
    assert "erasure_served_adapter_eligibility" in evidence.blocking_gate_ids()
    assert "external_served_eligibility_rejected" in evidence.blocking_gate_ids()
    assert "kite_http_invoke_release_drill" in evidence.blocking_gate_ids()
    assert "legacy_fact_migration_equivalence" not in evidence.blocking_gate_ids()
    retirement = evidence.compatibility_retirement[0]
    assert retirement.decision == "retain"
    assert retirement.reason_code == "telemetry_not_observed"
    assert _gate(evidence, "rdf12_triple_term_fixture").evidence.state is EvidenceState.SKIPPED


def test_only_declared_gates_can_be_updated_from_local_or_external_evidence() -> None:
    template = release_evidence_template()
    updated = apply_evidence_records(template, (_passed("rdf11_projection_fixture"),))

    assert _gate(updated, "rdf11_projection_fixture").evidence.passed
    assert updated.ready is False
    with pytest.raises(ReleaseEvidenceError, match="do not match"):
        apply_evidence_records(template, (_passed("made_up_release_gate"),))


def test_advertised_gate_cannot_be_skipped() -> None:
    with pytest.raises(ReleaseEvidenceError, match="advertised release gate"):
        GateResult(
            GateSpec("rdf11_fixture", "conformance", "rdf11"),
            EvidenceRecord(
                gate_id="rdf11_fixture",
                state=EvidenceState.SKIPPED,
                reason_code="outside_advertised_capability",
                outside_advertised_capability=True,
            ),
        )


def test_evidence_record_parser_rejects_unstructured_or_overstated_records() -> None:
    parsed = evidence_record_from_mapping(_passed("rdf11_projection_fixture").to_mapping())
    assert parsed.passed
    with pytest.raises(ReleaseEvidenceError, match="passed evidence requires"):
        evidence_record_from_mapping(
            {
                "gate_id": "rdf11_projection_fixture",
                "state": "passed",
                "command": "pytest",
                "exit_code": 0,
            }
        )


def test_stable_only_check_rejects_every_draft_capability_without_migration() -> None:
    records = inspect_stable_only_capabilities()

    assert [record.gate_id for record in records] == [
        "stable_only_query_profile_sparql12_20260605_experimental",
        "stable_only_rdf_profile_rdf12_cr_20260407_experimental",
        "stable_only_serialization_rdf12_ntriples_wd_20260515_experimental",
        "stable_only_validation_profile_shacl12_core_20260602_experimental",
        "stable_only_validation_profile_shacl12_sparql_20260130_experimental",
    ]
    assert all(record.state is EvidenceState.PASSED for record in records)
    assert all(record.exit_code == 0 for record in records)


@pytest.mark.asyncio
async def test_benchmark_harness_is_reproducible_and_budget_is_observed_not_timeout() -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)

    fixture = {
        "backend": "sqlite",
        "seed": "semantic-release-evidence-v1",
        "workload": "hybrid_recall",
    }
    run = await SemanticBenchmarkHarness(iterations=3).run(
        PerformanceMetric.HYBRID_RECALL,
        operation,
        fixture_description=fixture,
    )
    budget = PerformanceBudget.from_observed(
        run.metric,
        run.samples_ms,
        headroom_fraction=0.25,
        fixture_description=fixture,
    )

    assert calls == 3
    assert run.fixture_digest == budget.fixture_digest
    assert budget.budget_ms >= budget.p95_ms
    assert budget.evaluate(budget.budget_ms) is EvidenceState.PASSED
    assert budget.evaluate(budget.budget_ms + 0.001) is EvidenceState.FAILED


def test_performance_pass_needs_measured_budget_in_addition_to_command_evidence() -> None:
    template = release_evidence_template()
    metric = PerformanceMetric.HYBRID_RECALL
    budget = PerformanceBudget.from_observed(
        metric,
        (1.0, 2.0, 3.0),
        headroom_fraction=0.2,
        fixture_description={"backend": "sqlite", "workload": metric.value},
    )
    with pytest.raises(ReleaseEvidenceError, match="requires passing command evidence"):
        apply_performance_budgets(template, (budget,))

    observed = apply_evidence_records(template, (_passed(f"performance_{metric.value}"),))
    updated = apply_performance_budgets(observed, (budget,))
    assert updated.performance_budgets[metric] == budget
    assert updated.ready is False


def test_cli_writes_a_schema_valid_non_ready_template(tmp_path: Path) -> None:
    output = tmp_path / "semantic-release-evidence.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kestrel_sovereign.knowledge.release_evidence",
            "template",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["ready"] is False
    assert any(
        gate["scope"] == "served_adapter_eligibility" for gate in payload["gates"]
    )


def test_cli_records_budget_and_assembles_an_observed_result(tmp_path: Path) -> None:
    record = tmp_path / "rdf11.json"
    performance_record = tmp_path / "hybrid-recall.json"
    fixture = tmp_path / "benchmark-fixture.json"
    budget = tmp_path / "benchmark-budget.json"
    artifact = tmp_path / "release.json"
    fixture.write_text(
        json.dumps({"backend": "sqlite", "workload": "hybrid_recall"}),
        encoding="utf-8",
    )
    recorded = subprocess.run(
        [
            sys.executable,
            "-m",
            "kestrel_sovereign.knowledge.release_evidence",
            "record",
            "--gate",
            "rdf11_projection_fixture",
            "--artifact-ref",
            "local-test-log",
            "--output",
            str(record),
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    budget_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kestrel_sovereign.knowledge.release_evidence",
            "budget",
            "--metric",
            "hybrid_recall",
            "--samples-ms",
            "1",
            "2",
            "3",
            "--headroom-fraction",
            "0.2",
            "--fixture",
            str(fixture),
            "--output",
            str(budget),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    performance_recorded = subprocess.run(
        [
            sys.executable,
            "-m",
            "kestrel_sovereign.knowledge.release_evidence",
            "record",
            "--gate",
            "performance_hybrid_recall",
            "--artifact-ref",
            "local-benchmark-log",
            "--output",
            str(performance_record),
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assembled = subprocess.run(
        [
            sys.executable,
            "-m",
            "kestrel_sovereign.knowledge.release_evidence",
            "assemble",
            "--record",
            str(record),
            "--record",
            str(performance_record),
            "--budget",
            str(budget),
            "--output",
            str(artifact),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert recorded.returncode == 0, recorded.stderr
    assert budget_result.returncode == 0, budget_result.stderr
    assert performance_recorded.returncode == 0, performance_recorded.stderr
    assert assembled.returncode == 0, assembled.stderr
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert next(
        item["budget"]
        for item in payload["performance_budgets"]
        if item["metric"] == "hybrid_recall"
    ) is not None


def test_cli_can_record_an_explicit_block_without_claiming_success(tmp_path: Path) -> None:
    output = tmp_path / "postgres-blocked.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kestrel_sovereign.knowledge.release_evidence",
            "block",
            "--gate",
            "postgres_assertion",
            "--reason-code",
            "postgres_service_unavailable",
            "--artifact-ref",
            "local-environment-check",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["state"] == "blocked"
    assert payload["reason_code"] == "postgres_service_unavailable"
