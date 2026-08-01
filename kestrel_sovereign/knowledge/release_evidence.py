"""Spec-bound semantic release evidence catalog and report assembly.

The catalog below is the one source of truth for release gates.  It does not
record arbitrary argv, test stdout, DSNs, tenant identifiers, or unbound exit
codes.  A ``passed`` record is accepted only when its immutable spec digest,
runner/command digest, environment, fixture, and measured observation all
bind to the current catalog.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import metadata
import json
from pathlib import Path
import platform
from typing import cast

from .registry import ExperimentalCapabilityError, SemanticKnowledgeRegistry, StandardsMaturity, get_knowledge_registry
from .release_evidence_models import (
    ArtifactReference,
    CompatibilityRetirementDecision,
    DrillBinding,
    EvidenceRecord,
    EvidenceState,
    ErasureStage,
    ExecutionAttestation,
    ExecutionEnvironment,
    ExecutionSource,
    ExternalCapabilityReport,
    ExternalGateAttestation,
    FixtureBinding,
    FixtureContract,
    GateResult,
    GateSpec,
    ObservationField,
    ObservationSchema,
    PerformanceBudget,
    PerformanceMetric,
    PerformanceTarget,
    ReleaseEvidenceError,
    RunnerContract,
    StandardsMatrixEntry,
    StructuralGateResult,
    TelemetryAttestation,
    TrustedExecutionKey,
    TrustedExecutionPolicy,
    _canonical_json,
    _sha256,
)
from .release_evidence_freshness import ExternalFreshnessLedger


RELEASE_EVIDENCE_SCHEMA_VERSION = 3
SEMANTIC_RELEASE_CONTRACT = "semantic-kb-v1-release-evidence-v3"
STRUCTURAL_RELEASE_EVIDENCE_SCHEMA_VERSION = 1
STRUCTURAL_RELEASE_CONTRACT = "semantic-kb-v1-release-evidence-structural-v1"
PARAMETRIC_SELF_EVIDENCE_REPOSITORY = "KestrelSovereignAI/kestrel-feature-parametric-self"
PARAMETRIC_SELF_CAPABILITY_SOURCE_REVISION = "260ba985bcfdfab3dab1ea58da5b259057f3749f"
# Compatibility name for callers that only need the immutable capability
# baseline. Evidence producers must also provide a separate runner revision.
PARAMETRIC_SELF_EVIDENCE_REVISION = PARAMETRIC_SELF_CAPABILITY_SOURCE_REVISION
_ERASURE_DRILL = DrillBinding(
    "semantic_erasure_release_drill_v1",
    _sha256("semantic-release-evidence-v3:drill:semantic_erasure_release_drill_v1"),
)
_EXTERNAL_ADAPTER_GATE_IDS = (
    "external_corpus_consumed",
    "external_candidate_invalidated",
    "external_served_eligibility_rejected",
)


def _fixture_binding(fixture_id: str) -> FixtureBinding:
    return FixtureBinding(
        fixture_id,
        _sha256(f"semantic-release-evidence-v3:fixture:{fixture_id}"),
    )


def _fixture(fixture_id: str, harness_id: str, *, official: bool = False) -> FixtureContract:
    return FixtureContract(_fixture_binding(fixture_id), harness_id, official=official)


def _runner(runner_id: str, command_id: str) -> RunnerContract:
    """Pin a predeclared command pattern by digest without persisting argv."""
    return RunnerContract(
        runner_id,
        command_id,
        _sha256(f"semantic-release-evidence-v3:runner:{runner_id}:{command_id}"),
    )


def _schema(schema_id: str, *fields: tuple[str, str]) -> ObservationSchema:
    return ObservationSchema(schema_id, tuple(ObservationField(*field) for field in fields))


def _environment(backend: str, mode: str, profile: str = "stable_only") -> ExecutionEnvironment:
    return ExecutionEnvironment(backend, mode, profile)


_CONFORMANCE_STANDARD_IDS = {
    "rdf11_projection_fixture": "rdf11-concepts-20140225",
    "rdfs11_inference_fixture": "rdfs-20140225",
    "owl2rl_inference_fixture": "owl2-profiles-20121211",
    "shacl2017_core_fixture": "shacl-core-20170720",
    "sparql11_readonly_fixture": "sparql11-readonly",
}


def performance_targets() -> tuple[PerformanceTarget, ...]:
    """Every budget target: both databases and the relevant execution mode."""
    targets: list[PerformanceTarget] = []
    for metric in PerformanceMetric:
        mode = (
            "kite_http"
            if metric in {PerformanceMetric.CHANGED_WORK_SLEEP, PerformanceMetric.UNCHANGED_SLEEP}
            else "startup"
            if metric is PerformanceMetric.STARTUP
            else "integration"
        )
        unit = "bytes" if metric is PerformanceMetric.STORAGE_GROWTH else "ms"
        for backend in ("sqlite", "postgres"):
            targets.append(PerformanceTarget(metric, backend, mode, unit))
    return tuple(targets)


def _gate(
    gate_id: str,
    category: str,
    scope: str,
    *,
    runner_id: str,
    command_id: str,
    environment: ExecutionEnvironment,
    fixture: FixtureContract,
    schema: ObservationSchema,
    owner: str = "kestrel_core",
    advertised: bool = True,
    required_for_ready: bool = True,
    performance_target: PerformanceTarget | None = None,
    correlation: DrillBinding | None = None,
) -> GateSpec:
    return GateSpec(
        gate_id,
        category,
        scope,
        _runner(runner_id, command_id),
        environment,
        fixture,
        schema,
        owner=owner,
        advertised=advertised,
        required_for_ready=required_for_ready,
        performance_target=performance_target,
        correlation=correlation,
    )


def release_gate_specs(
    registry: SemanticKnowledgeRegistry | None = None,
) -> tuple[GateSpec, ...]:
    """Return the immutable release catalog; no caller may add a lookalike gate."""
    selected = registry or get_knowledge_registry()
    conformance = (
        _gate(
            "rdf11_projection_fixture", "conformance", "rdf11_projection",
            runner_id="pytest", command_id="rdf11_projection_v1",
            environment=_environment("sqlite", "unit"),
            fixture=_fixture("official.rdf11_projection.v1", "rdf11_projection_harness_v1", official=True),
            schema=_schema("rdf11_projection_result_v1", ("case_count", "positive_count"), ("assertion_count", "positive_count")),
        ),
        _gate(
            "rdfs11_inference_fixture", "conformance", "rdfs11_inference",
            runner_id="pytest", command_id="rdfs11_inference_v1",
            environment=_environment("sqlite", "unit"),
            fixture=_fixture("official.rdfs11_inference.v1", "rdfs11_inference_harness_v1", official=True),
            schema=_schema("rdfs11_inference_result_v1", ("case_count", "positive_count"), ("derivation_count", "positive_count")),
        ),
        _gate(
            "owl2rl_inference_fixture", "conformance", "owl2rl_inference",
            runner_id="pytest", command_id="owl2rl_inference_v1",
            environment=_environment("sqlite", "unit"),
            fixture=_fixture("official.owl2rl_inference.v1", "owl2rl_inference_harness_v1", official=True),
            schema=_schema("owl2rl_inference_result_v1", ("case_count", "positive_count"), ("derivation_count", "positive_count")),
        ),
        _gate(
            "shacl2017_core_fixture", "conformance", "shacl2017_core",
            runner_id="pytest", command_id="shacl2017_core_v1",
            environment=_environment("sqlite", "unit"),
            fixture=_fixture("official.shacl2017_core.v1", "shacl2017_core_harness_v1", official=True),
            schema=_schema("shacl2017_core_result_v1", ("case_count", "positive_count"), ("report_count", "positive_count")),
        ),
        _gate(
            "sparql11_readonly_fixture", "conformance", "sparql11_readonly",
            runner_id="pytest", command_id="sparql11_readonly_v1",
            environment=_environment("sqlite", "unit"),
            fixture=_fixture("official.sparql11_readonly.v1", "sparql11_readonly_harness_v1", official=True),
            schema=_schema("sparql11_readonly_result_v1", ("case_count", "positive_count"), ("query_count", "positive_count")),
        ),
        _gate(
            "rdf12_triple_term_fixture", "conformance", "rdf12_triple_terms",
            runner_id="pytest", command_id="rdf12_triple_terms_v1",
            environment=_environment("sqlite", "unit", "experimental_enabled"),
            fixture=_fixture("draft.rdf12_triple_terms.v1", "rdf12_harness_v1"),
            schema=_schema("rdf12_result_v1", ("case_count", "positive_count")),
            advertised=False, required_for_ready=False,
        ),
        _gate(
            "shacl12_fixture", "conformance", "shacl12",
            runner_id="pytest", command_id="shacl12_v1",
            environment=_environment("sqlite", "unit", "experimental_enabled"),
            fixture=_fixture("draft.shacl12.v1", "shacl12_harness_v1"),
            schema=_schema("shacl12_result_v1", ("case_count", "positive_count")),
            advertised=False, required_for_ready=False,
        ),
        _gate(
            "sparql12_fixture", "conformance", "sparql12",
            runner_id="pytest", command_id="sparql12_v1",
            environment=_environment("sqlite", "unit", "experimental_enabled"),
            fixture=_fixture("draft.sparql12.v1", "sparql12_harness_v1"),
            schema=_schema("sparql12_result_v1", ("case_count", "positive_count")),
            advertised=False, required_for_ready=False,
        ),
    )
    stable_selection = (
        _gate(
            "stable_only_capability_selection", "capability_selection", "experimental_selection_rejected",
            runner_id="registry", command_id="stable_only_capability_selection_v1",
            environment=_environment("none", "registry_contract"),
            fixture=_fixture("stable_only_selection.v1", "registry_selection_harness_v1"),
            schema=_schema("stable_only_selection_result_v1", ("rejected_capability_count", "positive_count")),
        ),
    )
    parity = tuple(
        _gate(
            f"{backend}_{surface}", "backend_parity", f"{backend}:{surface}",
            runner_id="pytest", command_id=f"backend_parity_{surface}_v1",
            environment=_environment(backend, "integration"),
            fixture=_fixture(f"backend_parity.{surface}.v1", "storage_parity_harness_v1"),
            schema=_schema(f"backend_parity_{surface}_result_v1", ("scenario_count", "positive_count"), ("assertion_count", "positive_count")),
        )
        for backend in ("sqlite", "postgres")
        for surface in (
            "assertion", "ownership_privacy", "validation", "inference_retraction",
            "migration", "corpus_export", "retrieval",
        )
    )
    performance = tuple(
        _gate(
            f"performance_{target.gate_suffix}", "performance", target.gate_suffix,
            runner_id="semantic_benchmark", command_id=f"benchmark_{target.gate_suffix}_v1",
            environment=_environment(target.backend, target.mode),
            fixture=_fixture(f"benchmark.{target.gate_suffix}.v1", "semantic_benchmark_harness_v1"),
            schema=_schema(
                f"benchmark_{target.gate_suffix}_result_v1",
                ("sample_count", "sample_count"),
                (("p95_bytes" if target.unit == "bytes" else "p95_ms"), ("positive_bytes" if target.unit == "bytes" else "positive_duration_ms")),
            ),
            performance_target=target,
        )
        for target in performance_targets()
    )
    erasure = tuple(
        _gate(
            f"erasure_{stage.value}", "erasure", stage.value,
            runner_id="kite_http", command_id=f"erasure_{stage.value}_v1",
            environment=_environment("dual_backend", "kite_http"),
            fixture=_fixture("erasure_drill.v1", "erasure_drill_harness_v1"),
            schema=_schema("erasure_result_v1", ("erased_count", "positive_count"), ("remaining_count", "zero_count")),
            owner="parametric_self" if stage is ErasureStage.SERVED_ADAPTER_ELIGIBILITY else "kestrel_core",
            correlation=_ERASURE_DRILL,
        )
        for stage in ErasureStage
    )
    external = tuple(
        _gate(
            gate_id, "external_adapter", gate_id,
            runner_id="external_ci", command_id=f"{gate_id}_v1",
            environment=_environment("dual_backend", "external_adapter"),
            fixture=_fixture("parametric_self_erasure.v1", "external_adapter_harness_v1"),
            schema=_schema("external_adapter_result_v1", ("erased_count", "positive_count"), ("remaining_count", "zero_count")),
            owner="parametric_self",
            correlation=_ERASURE_DRILL,
        )
        for gate_id in _EXTERNAL_ADAPTER_GATE_IDS
    )
    diagnostics = (
        _gate(
            "semantic_maintenance_diagnostics_contract", "diagnostics", "content_free_maintenance_diagnostics",
            runner_id="pytest", command_id="semantic_maintenance_diagnostics_v1",
            environment=_environment("sqlite", "unit"),
            fixture=_fixture("semantic_maintenance_diagnostics.v1", "diagnostics_harness_v1"),
            schema=_schema("diagnostics_result_v1", ("diagnostic_count", "positive_count"), ("redaction_violation_count", "zero_count")),
        ),
    )
    live_agent = (
        _gate(
            "kite_http_stable_only_release_drill", "live_agent", "stable_only_http_invoke",
            runner_id="kite_http", command_id="kite_http_stable_only_release_v1",
            environment=_environment("dual_backend", "kite_http_stable", "stable_only"),
            fixture=_fixture("kite_http_stable_only.v1", "kite_http_release_harness_v1"),
            schema=_schema("kite_http_stable_only_result_v1", ("invoke_count", "positive_count"), ("scenario_count", "positive_count"), ("provenance_check_count", "positive_count")),
        ),
        _gate(
            "kite_http_experimental_enabled_release_drill", "live_agent", "experimental_enabled_http_invoke",
            runner_id="kite_http", command_id="kite_http_experimental_enabled_release_v1",
            environment=_environment("dual_backend", "kite_http_experimental", "experimental_enabled"),
            fixture=_fixture("kite_http_experimental_enabled.v1", "kite_http_release_harness_v1"),
            schema=_schema("kite_http_experimental_result_v1", ("invoke_count", "positive_count"), ("scenario_count", "positive_count"), ("experimental_selection_count", "positive_count")),
        ),
        _gate(
            "stable_persisted_data_no_canonical_migration_drill", "live_agent", "stable_persisted_data_no_migration",
            runner_id="kite_http", command_id="kite_http_persisted_stable_release_v1",
            environment=_environment("dual_backend", "kite_http_persisted", "stable_only"),
            fixture=_fixture("kite_http_persisted_stable.v1", "kite_http_release_harness_v1"),
            schema=_schema("kite_http_persisted_result_v1", ("persisted_assertion_count", "positive_count"), ("canonical_migration_count", "zero_count")),
        ),
    )
    compatibility = (
        _gate(
            "legacy_fact_migration_equivalence", "compatibility_retirement", "legacy_fact_migration_compatibility",
            runner_id="pytest", command_id="legacy_fact_migration_equivalence_v1",
            environment=_environment("dual_backend", "integration"),
            fixture=_fixture("legacy_fact_migration.v1", "legacy_migration_harness_v1"),
            schema=_schema("legacy_migration_result_v1", ("scenario_count", "positive_count"), ("mismatch_count", "zero_count")),
            required_for_ready=False,
        ),
    )
    # Read the registry to retain its fail-closed initialization check.  The
    # resulting capability list is intentionally *not* a passing record.
    if not _experimental_capabilities(selected):
        raise ReleaseEvidenceError("stable-only release catalog requires experimental capabilities")
    return conformance + stable_selection + parity + performance + erasure + external + diagnostics + live_agent + compatibility


def _external_adapter_contract_digest() -> str:
    """Hash the immutable external-facing catalog contract, not a runtime SHA."""
    external_specs = tuple(
        spec for spec in release_gate_specs() if spec.category == "external_adapter"
    )
    if tuple(spec.gate_id for spec in external_specs) != _EXTERNAL_ADAPTER_GATE_IDS:
        raise ReleaseEvidenceError("external adapter catalog does not preserve its declared gate order")
    return _sha256(
        _canonical_json(
            {
                "release_evidence_schema_version": RELEASE_EVIDENCE_SCHEMA_VERSION,
                "semantic_release_contract": SEMANTIC_RELEASE_CONTRACT,
                "external_adapter_gates": [
                    {"gate_id": spec.gate_id, "gate_spec_digest": spec.digest}
                    for spec in external_specs
                ],
            }
        )
    )


@dataclass(frozen=True, slots=True)
class SemanticReleaseEvidence:
    """One report whose readiness is recomputed from the immutable catalog."""

    standards: tuple[StandardsMatrixEntry, ...]
    libraries: Mapping[str, str]
    profile: Mapping[str, object]
    gates: tuple[GateResult, ...]
    performance_budgets: Mapping[PerformanceTarget, PerformanceBudget | None]
    external_capabilities: tuple[ExternalCapabilityReport, ...]
    compatibility_retirement: tuple[CompatibilityRetirementDecision, ...]
    schema_version: int = RELEASE_EVIDENCE_SCHEMA_VERSION
    contract: str = SEMANTIC_RELEASE_CONTRACT

    def __post_init__(self) -> None:
        if self.schema_version != RELEASE_EVIDENCE_SCHEMA_VERSION or self.contract != SEMANTIC_RELEASE_CONTRACT:
            raise ReleaseEvidenceError("unsupported semantic release evidence contract")
        if any(not isinstance(gate, GateResult) for gate in self.gates):
            raise ReleaseEvidenceError("verified release evidence requires verified gate results")
        _validate_release_evidence_envelope(
            standards=self.standards,
            libraries=self.libraries,
            profile=self.profile,
            gates=self.gates,
            performance_budgets=self.performance_budgets,
            external_capabilities=self.external_capabilities,
            compatibility_retirement=self.compatibility_retirement,
        )

    @property
    def ready(self) -> bool:
        return (
            all(gate.ready for gate in self.gates)
            and all(budget is not None for budget in self.performance_budgets.values())
            and _external_capabilities_ready(self.external_capabilities, self.gates)
        )

    def blocking_gate_ids(self) -> tuple[str, ...]:
        return _structural_blocking_gate_ids(
            self.gates,
            self.performance_budgets,
            self.external_capabilities,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "ready": self.ready,
            "blocking_gate_ids": list(self.blocking_gate_ids()),
            "standards": [entry.to_mapping() for entry in self.standards],
            "libraries": dict(sorted(self.libraries.items())),
            "profile": dict(self.profile),
            "gates": [gate.to_mapping() for gate in sorted(self.gates, key=lambda item: item.spec.gate_id)],
            "performance_budgets": [
                {"target": target.to_mapping(), "budget": budget.to_mapping() if budget else None}
                for target, budget in sorted(self.performance_budgets.items(), key=lambda item: item[0].gate_suffix)
            ],
            "external_capabilities": [item.to_mapping() for item in self.external_capabilities],
            "compatibility_retirement": [item.to_mapping() for item in self.compatibility_retirement],
        }


@dataclass(frozen=True, slots=True)
class StructuralReleaseEvidence:
    """Inspectable catalog evidence without a verifier-issued trust verdict.

    This is the only report shape emitted by the public CLI.  Its fixed
    ``ready: false`` and ``trust_status: unverified`` prevent a report author
    from converting arbitrary records, keys, or process environment into a
    release claim.  An independent verifier may later consume the same records
    through the explicit :func:`apply_evidence_records` API.
    """

    standards: tuple[StandardsMatrixEntry, ...]
    libraries: Mapping[str, str]
    profile: Mapping[str, object]
    gates: tuple[StructuralGateResult, ...]
    performance_budgets: Mapping[PerformanceTarget, PerformanceBudget | None]
    external_capabilities: tuple[ExternalCapabilityReport, ...]
    compatibility_retirement: tuple[CompatibilityRetirementDecision, ...]
    schema_version: int = STRUCTURAL_RELEASE_EVIDENCE_SCHEMA_VERSION
    contract: str = STRUCTURAL_RELEASE_CONTRACT

    def __post_init__(self) -> None:
        if (
            self.schema_version != STRUCTURAL_RELEASE_EVIDENCE_SCHEMA_VERSION
            or self.contract != STRUCTURAL_RELEASE_CONTRACT
        ):
            raise ReleaseEvidenceError("unsupported structural release evidence contract")
        if any(not isinstance(gate, StructuralGateResult) for gate in self.gates):
            raise ReleaseEvidenceError("structural release evidence requires structural gate results")
        _validate_release_evidence_envelope(
            standards=self.standards,
            libraries=self.libraries,
            profile=self.profile,
            gates=self.gates,
            performance_budgets=self.performance_budgets,
            external_capabilities=self.external_capabilities,
            compatibility_retirement=self.compatibility_retirement,
        )

    @property
    def trust_status(self) -> str:
        return "unverified"

    @property
    def ready(self) -> bool:
        return False

    @property
    def structurally_complete(self) -> bool:
        return (
            all(gate.structurally_ready for gate in self.gates)
            and all(budget is not None for budget in self.performance_budgets.values())
            and _external_capabilities_ready(self.external_capabilities, self.gates)
        )

    def blocking_gate_ids(self) -> tuple[str, ...]:
        blocking = set(
            _structural_blocking_gate_ids(
                self.gates,
                self.performance_budgets,
                self.external_capabilities,
            )
        )
        blocking.add("trust_verification_required")
        return tuple(sorted(blocking))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "trust_status": self.trust_status,
            "ready": False,
            "structurally_complete": self.structurally_complete,
            "blocking_gate_ids": list(self.blocking_gate_ids()),
            "standards": [entry.to_mapping() for entry in self.standards],
            "libraries": dict(sorted(self.libraries.items())),
            "profile": dict(self.profile),
            "gates": [gate.to_mapping() for gate in sorted(self.gates, key=lambda item: item.spec.gate_id)],
            "performance_budgets": [
                {"target": target.to_mapping(), "budget": budget.to_mapping() if budget else None}
                for target, budget in sorted(
                    self.performance_budgets.items(), key=lambda item: item[0].gate_suffix
                )
            ],
            "external_capabilities": [item.to_mapping() for item in self.external_capabilities],
            "compatibility_retirement": [item.to_mapping() for item in self.compatibility_retirement],
        }


def _validate_release_evidence_envelope(
    *,
    standards: tuple[StandardsMatrixEntry, ...],
    libraries: Mapping[str, str],
    profile: Mapping[str, object],
    gates: tuple[GateResult | StructuralGateResult, ...],
    performance_budgets: Mapping[PerformanceTarget, PerformanceBudget | None],
    external_capabilities: tuple[ExternalCapabilityReport, ...],
    compatibility_retirement: tuple[CompatibilityRetirementDecision, ...],
) -> None:
    if not standards or not profile or not gates:
        raise ReleaseEvidenceError("release evidence requires matrix, profile, and gates")
    required_libraries = {"python", "rdflib", "kestrel_sovereign"}
    if set(libraries) != required_libraries:
        raise ReleaseEvidenceError("release evidence requires the exact runtime library set")
    if any(not isinstance(value, str) or not value or value == "unavailable" for value in libraries.values()):
        raise ReleaseEvidenceError("missing library version blocks release evidence")
    expected_specs = {spec.gate_id: spec for spec in release_gate_specs()}
    supplied_specs = {gate.spec.gate_id: gate.spec for gate in gates}
    if set(supplied_specs) != set(expected_specs) or any(
        supplied_specs[key].digest != expected_specs[key].digest for key in expected_specs
    ):
        raise ReleaseEvidenceError("report gates do not match the immutable release catalog")
    expected_targets = set(performance_targets())
    if set(performance_budgets) != expected_targets:
        raise ReleaseEvidenceError("release evidence requires every backend/mode performance target")
    spec_by_target = {
        gate.spec.performance_target: gate.spec
        for gate in gates
        if gate.spec.performance_target is not None
    }
    for target, budget in performance_budgets.items():
        if budget is not None:
            budget.validate_against(spec_by_target[target])
    if len(compatibility_retirement) != 1:
        raise ReleaseEvidenceError("release evidence requires one compatibility decision")
    _validate_retirement_decision(compatibility_retirement[0], gates)
    if external_capabilities:
        _validate_external_capability_reports(external_capabilities, gates)


def _structural_blocking_gate_ids(
    gates: tuple[GateResult | StructuralGateResult, ...],
    performance_budgets: Mapping[PerformanceTarget, PerformanceBudget | None],
    external_capabilities: tuple[ExternalCapabilityReport, ...],
) -> tuple[str, ...]:
    blocking = {
        gate.spec.gate_id
        for gate in gates
        if gate.spec.required_for_ready and not gate.evidence.passed
    }
    blocking.update(
        f"performance_{target.gate_suffix}"
        for target, budget in performance_budgets.items()
        if budget is None
    )
    if not _external_capabilities_ready(external_capabilities, gates):
        blocking.add("external_adapter_attestation")
    return tuple(sorted(blocking))


def _legacy_migration_gate(
    gates: tuple[GateResult | StructuralGateResult, ...],
) -> GateResult | StructuralGateResult:
    try:
        return next(gate for gate in gates if gate.spec.gate_id == "legacy_fact_migration_equivalence")
    except StopIteration as error:  # pragma: no cover - catalog check catches this first.
        raise ReleaseEvidenceError("immutable catalog lacks legacy migration equivalence gate") from error


def _validate_retirement_decision(
    decision: CompatibilityRetirementDecision,
    gates: tuple[GateResult | StructuralGateResult, ...],
) -> None:
    """Bind retirement only to the exact declared migration-equivalence result."""
    if decision.path_id != "legacy_fact_migration_compatibility":
        raise ReleaseEvidenceError("unknown compatibility retirement path")
    migration = _legacy_migration_gate(gates)
    if (
        decision.migration_gate_id != migration.spec.gate_id
        or decision.migration_spec_digest != migration.spec.digest
    ):
        raise ReleaseEvidenceError("retirement decision is not bound to legacy migration equivalence spec")
    if decision.migration_run_digest is not None:
        if not migration.evidence.passed or migration.evidence.run_digest != decision.migration_run_digest:
            raise ReleaseEvidenceError("retirement decision references an unrelated migration result")
    if decision.removal_safe and decision.migration_run_digest is None:
        raise ReleaseEvidenceError("retirement cannot be safe without bound migration equivalence")


def _external_gate_results(
    gates: tuple[GateResult | StructuralGateResult, ...],
) -> dict[str, GateResult | StructuralGateResult]:
    return {
        gate.spec.gate_id: gate
        for gate in gates
        if gate.spec.category == "external_adapter"
    }


def _validate_external_capability_reports(
    reports: tuple[ExternalCapabilityReport, ...],
    gates: tuple[GateResult | StructuralGateResult, ...],
    *,
    expected_evidence_runner_revision: str | None = None,
) -> None:
    """Require exact repo/revision and hashed result/artifact bindings from Pself."""
    if len(reports) != 1:
        raise ReleaseEvidenceError("release evidence requires exactly one external adapter attestation")
    report = reports[0]
    if (
        report.capability_id != "parametric_self_governed_corpus"
        or report.repository != PARAMETRIC_SELF_EVIDENCE_REPOSITORY
        or report.capability_source_revision != PARAMETRIC_SELF_CAPABILITY_SOURCE_REVISION
    ):
        raise ReleaseEvidenceError("external adapter report repository or revision does not match contract")
    if expected_evidence_runner_revision is not None:
        if (
            not isinstance(expected_evidence_runner_revision, str)
            or len(expected_evidence_runner_revision) != 40
            or any(character not in "0123456789abcdef" for character in expected_evidence_runner_revision)
        ):
            raise ReleaseEvidenceError("expected external evidence runner revision must be a full lowercase commit SHA")
        if report.evidence_runner_revision != expected_evidence_runner_revision:
            raise ReleaseEvidenceError("external adapter report runner revision does not match verifier policy")
    if report.core_release_evidence_contract_digest != CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST:
        raise ReleaseEvidenceError("external adapter report core catalog contract does not match")
    expected = _external_gate_results(gates)
    supplied = {item.gate_id: item for item in report.attestations}
    if set(supplied) != set(expected) or set(supplied) != set(_EXTERNAL_ADAPTER_GATE_IDS):
        raise ReleaseEvidenceError("external adapter report must cover corpus, candidate, and served stages")
    if report.gate_ids != _EXTERNAL_ADAPTER_GATE_IDS or tuple(expected) != _EXTERNAL_ADAPTER_GATE_IDS:
        raise ReleaseEvidenceError("external adapter report must preserve declared external gate order")
    for gate_id, gate in expected.items():
        attestation = supplied[gate_id]
        evidence = gate.evidence
        if (
            not evidence.passed
            or evidence.run_digest is None
            or evidence.artifact is None
            or attestation.gate_spec_digest != gate.spec.digest
            or attestation.result_digest != evidence.run_digest
            or attestation.artifact != evidence.artifact
            or attestation.drill != gate.spec.correlation
            or evidence.external_run_nonce != report.run_nonce
            or evidence.external_evidence_runner_revision != report.evidence_runner_revision
        ):
            raise ReleaseEvidenceError(
                "external adapter attestation is not bound to its runner revision, external run_nonce, or correlated gate result/artifact"
            )


def _external_capabilities_ready(
    reports: tuple[ExternalCapabilityReport, ...],
    gates: tuple[GateResult | StructuralGateResult, ...],
) -> bool:
    if not reports:
        return False
    try:
        _validate_external_capability_reports(reports, gates)
    except ReleaseEvidenceError:
        return False
    return True


def build_standards_matrix(registry: SemanticKnowledgeRegistry | None = None) -> tuple[StandardsMatrixEntry, ...]:
    """Return exact registry pins plus official fixture/harness metadata."""
    selected = registry or get_knowledge_registry()
    contract_by_standard = {
        _CONFORMANCE_STANDARD_IDS[spec.gate_id]: spec
        for spec in release_gate_specs(selected)
        if spec.gate_id in _CONFORMANCE_STANDARD_IDS
    }
    return tuple(
        StandardsMatrixEntry(
            identifier=resource.identifier,
            version=str(resource.version),
            maturity=resource.maturity.value,
            kind=resource.kind.value,
            uri=resource.uri,
            published_date=resource.published_date,
            sha256=resource.sha256,
            capabilities=tuple(sorted(resource.capabilities)),
            fixture=(
                contract_by_standard[resource.identifier].fixture
                if resource.identifier in contract_by_standard
                else None
            ),
            runner=(
                contract_by_standard[resource.identifier].runner
                if resource.identifier in contract_by_standard
                else None
            ),
        )
        for resource in selected.resources
    )


def implementation_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "rdflib": _distribution_version("rdflib"),
        "kestrel_sovereign": _distribution_version("kestrel-sovereign"),
    }


def inspect_stable_only_capabilities(registry: SemanticKnowledgeRegistry | None = None) -> Mapping[str, int]:
    """Run the registry check but never silently convert it into a passed gate."""
    selected = registry or get_knowledge_registry()
    rejected = 0
    for capability in _experimental_capabilities(selected):
        try:
            selected.select_capability(capability)
        except ExperimentalCapabilityError:
            rejected += 1
        else:
            raise ReleaseEvidenceError("stable-only registry selected an experimental capability")
    if rejected == 0:
        raise ReleaseEvidenceError("stable-only registry check found no experimental capability")
    return {"rejected_capability_count": rejected}


def release_evidence_template(registry: SemanticKnowledgeRegistry | None = None) -> SemanticReleaseEvidence:
    selected = registry or get_knowledge_registry()
    specs = release_gate_specs(selected)
    gates = tuple(GateResult(spec, spec.initial_evidence()) for spec in specs)
    migration = _legacy_migration_gate(gates)
    return SemanticReleaseEvidence(
        standards=build_standards_matrix(selected),
        libraries=implementation_versions(),
        profile={
            "canonical_contract": "semantic-kb-v1",
            "advertised_stable_capabilities": list(_stable_capabilities(selected)),
            "disabled_experimental_capabilities": list(_experimental_capabilities(selected)),
        },
        gates=gates,
        performance_budgets={target: None for target in performance_targets()},
        external_capabilities=(),
        compatibility_retirement=(CompatibilityRetirementDecision(
            path_id="legacy_fact_migration_compatibility",
            migration_gate_id=migration.spec.gate_id,
            migration_spec_digest=migration.spec.digest,
            migration_run_digest=None,
            telemetry=None,
        ),),
    )


def structural_release_evidence_template(
    registry: SemanticKnowledgeRegistry | None = None,
) -> StructuralReleaseEvidence:
    """Create the public CLI's unverified structural evidence template."""
    verified_template = release_evidence_template(registry)
    return StructuralReleaseEvidence(
        standards=verified_template.standards,
        libraries=verified_template.libraries,
        profile=verified_template.profile,
        gates=tuple(
            StructuralGateResult(gate.spec, gate.evidence)
            for gate in verified_template.gates
        ),
        performance_budgets=verified_template.performance_budgets,
        external_capabilities=verified_template.external_capabilities,
        compatibility_retirement=verified_template.compatibility_retirement,
    )


def apply_evidence_records(
    evidence: SemanticReleaseEvidence,
    records: Iterable[EvidenceRecord],
    *,
    trust_policy: TrustedExecutionPolicy,
) -> SemanticReleaseEvidence:
    """Verify catalog-bound records against an independently supplied policy."""
    if not isinstance(trust_policy, TrustedExecutionPolicy):
        raise ReleaseEvidenceError("evidence verification requires an explicit TrustedExecutionPolicy")
    updates = _record_updates_for_declared_gates(evidence.gates, records)
    gates = tuple(
        GateResult(
            gate.spec,
            updates.get(gate.spec.gate_id, gate.evidence),
            trust_policy,
        )
        for gate in evidence.gates
    )
    return replace(evidence, gates=gates)


def apply_structural_evidence_records(
    evidence: StructuralReleaseEvidence,
    records: Iterable[EvidenceRecord],
) -> StructuralReleaseEvidence:
    """Attach structurally valid records without claiming their signers are trusted."""
    updates = _record_updates_for_declared_gates(evidence.gates, records)
    gates = tuple(
        StructuralGateResult(
            gate.spec,
            updates.get(gate.spec.gate_id, gate.evidence),
        )
        for gate in evidence.gates
    )
    return replace(evidence, gates=gates)


def apply_performance_budgets(
    evidence: SemanticReleaseEvidence,
    budgets: Iterable[PerformanceBudget],
    *,
    trust_policy: TrustedExecutionPolicy,
) -> SemanticReleaseEvidence:
    """Verify benchmark budgets against an independently supplied policy."""
    if not isinstance(trust_policy, TrustedExecutionPolicy):
        raise ReleaseEvidenceError("budget verification requires an explicit TrustedExecutionPolicy")
    updates = _performance_budget_updates_for_declared_gates(evidence.gates, budgets)
    gate_by_id = {gate.spec.gate_id: gate for gate in evidence.gates}
    for budget in updates.values():
        trust_policy.verify_budget(gate_by_id[budget.gate_id].spec, budget)
    return replace(evidence, performance_budgets={**evidence.performance_budgets, **updates})


def apply_structural_performance_budgets(
    evidence: StructuralReleaseEvidence,
    budgets: Iterable[PerformanceBudget],
) -> StructuralReleaseEvidence:
    """Attach structurally valid budgets without a signer trust verdict."""
    updates = _performance_budget_updates_for_declared_gates(evidence.gates, budgets)
    return replace(evidence, performance_budgets={**evidence.performance_budgets, **updates})


def _record_updates_for_declared_gates(
    gates: tuple[GateResult | StructuralGateResult, ...],
    records: Iterable[EvidenceRecord],
) -> dict[str, EvidenceRecord]:
    supplied = tuple(records)
    updates = {record.gate_id: record for record in supplied}
    if len(updates) != len(supplied):
        raise ReleaseEvidenceError("evidence records must not repeat a gate_id")
    known = {gate.spec.gate_id for gate in gates}
    unknown = sorted(set(updates) - known)
    if unknown:
        raise ReleaseEvidenceError(
            "evidence records do not match declared template gates: " + ", ".join(unknown)
        )
    return updates


def _performance_budget_updates_for_declared_gates(
    gates: tuple[GateResult | StructuralGateResult, ...],
    budgets: Iterable[PerformanceBudget],
) -> dict[PerformanceTarget, PerformanceBudget]:
    supplied = tuple(budgets)
    updates = {budget.target: budget for budget in supplied}
    if len(updates) != len(supplied):
        raise ReleaseEvidenceError("performance budgets must not repeat a backend/mode target")
    gate_by_id = {gate.spec.gate_id: gate for gate in gates}
    for budget in updates.values():
        gate = gate_by_id.get(budget.gate_id)
        if gate is None or not gate.evidence.passed:
            raise ReleaseEvidenceError("performance budget requires its spec-bound passing gate")
        budget.validate_against(gate.spec)
    return updates


def attach_retirement_telemetry(
    evidence: SemanticReleaseEvidence,
    telemetry: TelemetryAttestation,
) -> SemanticReleaseEvidence:
    """Safely attach a telemetry window to the one declared migration gate."""
    if not isinstance(telemetry, TelemetryAttestation):
        raise ReleaseEvidenceError("retirement telemetry must be TelemetryAttestation")
    migration = _legacy_migration_gate(evidence.gates)
    decision = CompatibilityRetirementDecision(
        path_id="legacy_fact_migration_compatibility",
        migration_gate_id=migration.spec.gate_id,
        migration_spec_digest=migration.spec.digest,
        migration_run_digest=migration.evidence.run_digest if migration.evidence.passed else None,
        telemetry=telemetry,
    )
    return replace(evidence, compatibility_retirement=(decision,))


def attach_structural_retirement_telemetry(
    evidence: StructuralReleaseEvidence,
    telemetry: TelemetryAttestation,
) -> StructuralReleaseEvidence:
    """Attach a structurally valid telemetry window without a trust verdict."""
    if not isinstance(telemetry, TelemetryAttestation):
        raise ReleaseEvidenceError("retirement telemetry must be TelemetryAttestation")
    migration = _legacy_migration_gate(evidence.gates)
    decision = CompatibilityRetirementDecision(
        path_id="legacy_fact_migration_compatibility",
        migration_gate_id=migration.spec.gate_id,
        migration_spec_digest=migration.spec.digest,
        migration_run_digest=migration.evidence.run_digest if migration.evidence.passed else None,
        telemetry=telemetry,
    )
    return replace(evidence, compatibility_retirement=(decision,))


def attach_external_capability_report(
    evidence: SemanticReleaseEvidence,
    report: ExternalCapabilityReport,
    *,
    freshness_ledger: ExternalFreshnessLedger,
    expected_evidence_runner_revision: str,
) -> SemanticReleaseEvidence:
    """Attach and durably consume a fully bound report from external CI.

    This is verifier-only ingestion.  It deliberately requires a ledger owned
    by that verifier, so the public structural assembler cannot mark an
    external bundle fresh or reuse a report author's transient state.
    """
    if not isinstance(report, ExternalCapabilityReport):
        raise ReleaseEvidenceError("external adapter report must be ExternalCapabilityReport")
    if not isinstance(freshness_ledger, ExternalFreshnessLedger):
        raise ReleaseEvidenceError("trusted external adapter ingestion requires verifier-owned freshness ledger")
    attached = replace(evidence, external_capabilities=(report,))
    _validate_external_capability_reports(
        attached.external_capabilities,
        attached.gates,
        expected_evidence_runner_revision=expected_evidence_runner_revision,
    )
    freshness_ledger.consume(report)
    return attached


def attach_structural_external_capability_report(
    evidence: StructuralReleaseEvidence,
    report: ExternalCapabilityReport,
) -> StructuralReleaseEvidence:
    """Attach a structurally bound external report without a trust verdict."""
    if not isinstance(report, ExternalCapabilityReport):
        raise ReleaseEvidenceError("external adapter report must be ExternalCapabilityReport")
    return replace(evidence, external_capabilities=(report,))


def _expect_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReleaseEvidenceError(f"{field} must be a mapping")
    return cast(Mapping[str, object], value)


def _strict_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise ReleaseEvidenceError(f"{field} has unknown or missing fields")


def _environment_from_mapping(value: object) -> ExecutionEnvironment | None:
    if value is None:
        return None
    mapping = _expect_mapping(value, "environment")
    _strict_keys(mapping, {"backend", "mode", "profile"}, "environment")
    return ExecutionEnvironment(cast(str, mapping["backend"]), cast(str, mapping["mode"]), cast(str, mapping["profile"]))


def _fixture_from_mapping(value: object) -> FixtureBinding | None:
    if value is None:
        return None
    mapping = _expect_mapping(value, "fixture")
    _strict_keys(mapping, {"fixture_id", "fixture_digest"}, "fixture")
    return FixtureBinding(cast(str, mapping["fixture_id"]), cast(str, mapping["fixture_digest"]))


def _artifact_from_mapping(value: object) -> ArtifactReference | None:
    if value is None:
        return None
    mapping = _expect_mapping(value, "artifact")
    _strict_keys(mapping, {"artifact_ref", "artifact_digest"}, "artifact")
    return ArtifactReference(cast(str, mapping["artifact_ref"]), cast(str, mapping["artifact_digest"]))


def _drill_from_mapping(value: object) -> DrillBinding | None:
    if value is None:
        return None
    mapping = _expect_mapping(value, "drill")
    _strict_keys(mapping, {"drill_id", "drill_digest"}, "drill")
    return DrillBinding(cast(str, mapping["drill_id"]), cast(str, mapping["drill_digest"]))


def _execution_attestation_from_mapping(value: object) -> ExecutionAttestation | None:
    if value is None:
        return None
    mapping = _expect_mapping(value, "execution_attestation")
    _strict_keys(mapping, {"issuer_id", "key_id", "source", "signature"}, "execution_attestation")
    try:
        source = ExecutionSource(cast(str, mapping["source"]))
    except ValueError as error:
        raise ReleaseEvidenceError("execution_attestation has an invalid source") from error
    return ExecutionAttestation(
        issuer_id=cast(str, mapping["issuer_id"]),
        key_id=cast(str, mapping["key_id"]),
        source=source,
        signature=cast(str, mapping["signature"]),
    )


def trusted_execution_policy_from_mapping(value: Mapping[str, object]) -> TrustedExecutionPolicy:
    """Parse an operator-owned public-key policy, never a private key."""
    mapping = _expect_mapping(value, "trusted execution policy")
    _strict_keys(mapping, {"keys"}, "trusted execution policy")
    raw_keys = mapping["keys"]
    if not isinstance(raw_keys, list):
        raise ReleaseEvidenceError("trusted execution policy keys must be a list")
    keys: list[TrustedExecutionKey] = []
    for raw_key in raw_keys:
        key = _expect_mapping(raw_key, "trusted execution key")
        _strict_keys(
            key,
            {"issuer_id", "key_id", "source", "public_key", "runner_ids"},
            "trusted execution key",
        )
        raw_runner_ids = key["runner_ids"]
        if not isinstance(raw_runner_ids, list) or not all(isinstance(value, str) for value in raw_runner_ids):
            raise ReleaseEvidenceError("trusted execution key runner_ids must be a string list")
        try:
            source = ExecutionSource(cast(str, key["source"]))
        except ValueError as error:
            raise ReleaseEvidenceError("trusted execution key has an invalid source") from error
        keys.append(
            TrustedExecutionKey(
                issuer_id=cast(str, key["issuer_id"]),
                key_id=cast(str, key["key_id"]),
                source=source,
                public_key=cast(str, key["public_key"]),
                runner_ids=tuple(cast(list[str], raw_runner_ids)),
            )
        )
    return TrustedExecutionPolicy(tuple(keys))


def evidence_record_from_mapping(value: Mapping[str, object]) -> EvidenceRecord:
    mapping = _expect_mapping(value, "evidence record")
    expected = {
        "gate_id", "state", "gate_spec_digest", "runner_id", "command_id", "command_digest",
        "environment", "environment_digest", "fixture", "observation", "artifact", "run_digest",
        "external_run_nonce", "external_evidence_runner_revision",
        "execution_attestation", "drill",
        "reason_code", "outside_advertised_capability",
    }
    _strict_keys(mapping, expected, "evidence record")
    try:
        state = EvidenceState(cast(str, mapping["state"]))
    except ValueError as error:
        raise ReleaseEvidenceError("evidence record has an invalid state") from error
    observation = mapping["observation"]
    return EvidenceRecord(
        gate_id=cast(str, mapping["gate_id"]), state=state,
        gate_spec_digest=cast(str | None, mapping["gate_spec_digest"]), runner_id=cast(str | None, mapping["runner_id"]),
        command_id=cast(str | None, mapping["command_id"]), command_digest=cast(str | None, mapping["command_digest"]),
        environment=_environment_from_mapping(mapping["environment"]), environment_digest=cast(str | None, mapping["environment_digest"]),
        fixture=_fixture_from_mapping(mapping["fixture"]), observation=_expect_mapping(observation, "observation") if observation is not None else None,
        artifact=_artifact_from_mapping(mapping["artifact"]), run_digest=cast(str | None, mapping["run_digest"]),
        external_run_nonce=cast(str | None, mapping["external_run_nonce"]),
        external_evidence_runner_revision=cast(str | None, mapping["external_evidence_runner_revision"]),
        execution_attestation=_execution_attestation_from_mapping(mapping["execution_attestation"]),
        drill=_drill_from_mapping(mapping["drill"]),
        reason_code=cast(str | None, mapping["reason_code"]), outside_advertised_capability=cast(bool, mapping["outside_advertised_capability"]),
    )


def performance_budget_from_mapping(value: Mapping[str, object]) -> PerformanceBudget:
    mapping = _expect_mapping(value, "performance budget")
    expected = {
        "target", "gate_id", "gate_spec_digest", "samples", "p95", "budget", "headroom_fraction",
        "fixture", "environment", "artifact", "run_digest", "execution_attestation",
    }
    _strict_keys(mapping, expected, "performance budget")
    target_value = _expect_mapping(mapping["target"], "performance target")
    _strict_keys(target_value, {"metric", "backend", "mode", "unit"}, "performance target")
    target = PerformanceTarget(PerformanceMetric(cast(str, target_value["metric"])), cast(str, target_value["backend"]), cast(str, target_value["mode"]), cast(str, target_value["unit"]))
    samples = mapping["samples"]
    if not isinstance(samples, list):
        raise ReleaseEvidenceError("performance samples must be a list")
    fixture = _fixture_from_mapping(mapping["fixture"])
    environment = _environment_from_mapping(mapping["environment"])
    artifact = _artifact_from_mapping(mapping["artifact"])
    if fixture is None or environment is None or artifact is None:
        raise ReleaseEvidenceError("performance budget requires fixture, environment, and artifact")
    execution_attestation = _execution_attestation_from_mapping(mapping["execution_attestation"])
    if execution_attestation is None:
        raise ReleaseEvidenceError("performance budget requires an execution_attestation")
    return PerformanceBudget(
        target,
        cast(str, mapping["gate_id"]),
        cast(str, mapping["gate_spec_digest"]),
        tuple(cast(list[float | int], samples)),
        cast(float | int, mapping["p95"]),
        cast(float | int, mapping["budget"]),
        cast(float, mapping["headroom_fraction"]),
        fixture,
        environment,
        artifact,
        cast(str, mapping["run_digest"]),
        execution_attestation,
    )


def telemetry_attestation_from_mapping(value: Mapping[str, object]) -> TelemetryAttestation:
    mapping = _expect_mapping(value, "telemetry attestation")
    _strict_keys(
        mapping,
        {
            "window_started_at", "window_ended_at", "inventory_digest", "inventory_complete",
            "unmigrated_eligible_rows", "required_consumer_count", "artifact", "telemetry_digest",
        },
        "telemetry attestation",
    )
    artifact = _artifact_from_mapping(mapping["artifact"])
    if artifact is None:
        raise ReleaseEvidenceError("telemetry attestation requires artifact")
    return TelemetryAttestation(
        window_started_at=cast(str, mapping["window_started_at"]),
        window_ended_at=cast(str, mapping["window_ended_at"]),
        inventory_digest=cast(str, mapping["inventory_digest"]),
        inventory_complete=cast(bool, mapping["inventory_complete"]),
        unmigrated_eligible_rows=cast(int, mapping["unmigrated_eligible_rows"]),
        required_consumer_count=cast(int, mapping["required_consumer_count"]),
        artifact=artifact,
        telemetry_digest=cast(str, mapping["telemetry_digest"]),
    )


def external_capability_report_from_mapping(value: Mapping[str, object]) -> ExternalCapabilityReport:
    mapping = _expect_mapping(value, "external adapter report")
    _strict_keys(
        mapping,
        {
            "capability_id",
            "repository",
            "capability_source_revision",
            "evidence_runner_revision",
            "core_release_evidence_contract_digest",
            "run_nonce",
            "attestations",
            "attestation_digest",
        },
        "external adapter report",
    )
    raw_attestations = mapping["attestations"]
    if not isinstance(raw_attestations, list):
        raise ReleaseEvidenceError("external adapter attestations must be a list")
    attestations: list[ExternalGateAttestation] = []
    for raw in raw_attestations:
        item = _expect_mapping(raw, "external gate attestation")
        _strict_keys(
            item,
            {"gate_id", "gate_spec_digest", "result_digest", "artifact", "drill"},
            "external gate attestation",
        )
        artifact = _artifact_from_mapping(item["artifact"])
        if artifact is None:
            raise ReleaseEvidenceError("external gate attestation requires artifact")
        drill = _drill_from_mapping(item["drill"])
        if drill is None:
            raise ReleaseEvidenceError("external gate attestation requires drill correlation")
        attestations.append(
            ExternalGateAttestation(
                gate_id=cast(str, item["gate_id"]),
                gate_spec_digest=cast(str, item["gate_spec_digest"]),
                result_digest=cast(str, item["result_digest"]),
                artifact=artifact,
                drill=drill,
            )
        )
    return ExternalCapabilityReport(
        capability_id=cast(str, mapping["capability_id"]),
        repository=cast(str, mapping["repository"]),
        capability_source_revision=cast(str, mapping["capability_source_revision"]),
        evidence_runner_revision=cast(str, mapping["evidence_runner_revision"]),
        core_release_evidence_contract_digest=cast(str, mapping["core_release_evidence_contract_digest"]),
        run_nonce=cast(str, mapping["run_nonce"]),
        attestations=tuple(attestations),
        attestation_digest=cast(str, mapping["attestation_digest"]),
    )


def run_command_evidence(*_args: object, **_kwargs: object) -> EvidenceRecord:
    """Refuse arbitrary argv; the CLI is migrated to predeclared runners next."""
    raise ReleaseEvidenceError("generic argv execution cannot create release-ready evidence")


def write_release_evidence(
    evidence: SemanticReleaseEvidence | StructuralReleaseEvidence,
    output: Path,
    *,
    overwrite: bool = False,
) -> None:
    _write_json(evidence.to_mapping(), output, overwrite=overwrite, kind="evidence artifact")


def write_evidence_record(record: EvidenceRecord, output: Path, *, overwrite: bool = False) -> None:
    _write_json(record.to_mapping(), output, overwrite=overwrite, kind="evidence record")


def write_performance_budget(budget: PerformanceBudget, output: Path, *, overwrite: bool = False) -> None:
    _write_json(budget.to_mapping(), output, overwrite=overwrite, kind="performance budget")


def write_telemetry_attestation(telemetry: TelemetryAttestation, output: Path, *, overwrite: bool = False) -> None:
    _write_json(telemetry.to_mapping(), output, overwrite=overwrite, kind="telemetry attestation")


def _experimental_capabilities(registry: SemanticKnowledgeRegistry) -> tuple[str, ...]:
    return tuple(sorted(capability for resource in registry.resources if resource.maturity is StandardsMaturity.EXPERIMENTAL for capability in resource.capabilities))


def _stable_capabilities(registry: SemanticKnowledgeRegistry) -> tuple[str, ...]:
    return tuple(sorted(capability for resource in registry.resources if resource.maturity is StandardsMaturity.STABLE for capability in resource.capabilities))


# This pin can be produced by a feature using the new report API: it changes
# only when the immutable evidence schema, contract, or external gate specs do.
CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST = _external_adapter_contract_digest()


def _write_json(value: Mapping[str, object], output: Path, *, overwrite: bool, kind: str) -> None:
    if output.exists() and not overwrite:
        raise ReleaseEvidenceError(f"refusing to replace existing {kind} {output}; pass --overwrite")
    if not output.parent.exists():
        raise ReleaseEvidenceError(f"{kind} output parent does not exist: {output.parent}")
    output.write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _distribution_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _main(argv: Sequence[str] | None = None) -> int:
    from .release_evidence_cli import main
    return main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
