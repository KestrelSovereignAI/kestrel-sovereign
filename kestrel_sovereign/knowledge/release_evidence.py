"""Release-evidence catalog and report assembly for semantic knowledge.

This module owns the one declarative gate catalog and report orchestration.
Content-free record validation and benchmark primitives live in
``release_evidence_models``; the CLI lives in ``release_evidence_cli``.  No
function here changes semantic feature behavior or imports an optional feature.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import metadata
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

from .registry import (
    ExperimentalCapabilityError,
    SemanticKnowledgeRegistry,
    StandardsMaturity,
    get_knowledge_registry,
)
from .release_evidence_models import (
    CompatibilityRetirementDecision,
    EvidenceRecord,
    EvidenceState,
    ErasureStage,
    ExternalCapabilityReport,
    GateResult,
    GateSpec,
    PerformanceBudget,
    PerformanceMetric,
    ReleaseEvidenceError,
    StandardsMatrixEntry,
    _canonical_json,
    _require_identifier,
    _require_reference,
    _safe_gate_name,
)


RELEASE_EVIDENCE_SCHEMA_VERSION = 1
SEMANTIC_RELEASE_CONTRACT = "semantic-kb-v1-release-evidence-v1"
PARAMETRIC_SELF_EVIDENCE_REPOSITORY = (
    "KestrelSovereignAI/kestrel-feature-parametric-self"
)
PARAMETRIC_SELF_EVIDENCE_REVISION = "bcbfbb2"


@dataclass(frozen=True, slots=True)
class SemanticReleaseEvidence:
    """One report assembled strictly from the declared release gates."""

    standards: tuple[StandardsMatrixEntry, ...]
    libraries: Mapping[str, str]
    profile: Mapping[str, object]
    gates: tuple[GateResult, ...]
    performance_budgets: Mapping[PerformanceMetric, PerformanceBudget | None]
    external_capabilities: tuple[ExternalCapabilityReport, ...]
    compatibility_retirement: tuple[CompatibilityRetirementDecision, ...]
    schema_version: int = RELEASE_EVIDENCE_SCHEMA_VERSION
    contract: str = SEMANTIC_RELEASE_CONTRACT

    def __post_init__(self) -> None:
        if self.schema_version != RELEASE_EVIDENCE_SCHEMA_VERSION:
            raise ReleaseEvidenceError("unsupported semantic release evidence schema")
        if self.contract != SEMANTIC_RELEASE_CONTRACT:
            raise ReleaseEvidenceError("unsupported semantic release evidence contract")
        if not self.standards or not self.libraries or not self.profile or not self.gates:
            raise ReleaseEvidenceError("release evidence requires matrix, libraries, profile, and gates")
        if any(not isinstance(item, GateResult) for item in self.gates):
            raise ReleaseEvidenceError("release evidence gates must be GateResult values")
        gate_ids = [item.spec.gate_id for item in self.gates]
        if len(set(gate_ids)) != len(gate_ids):
            raise ReleaseEvidenceError("release evidence cannot repeat a gate_id")
        if set(self.performance_budgets) != set(PerformanceMetric):
            raise ReleaseEvidenceError("release evidence requires every performance budget slot")
        for metric, budget in self.performance_budgets.items():
            if not isinstance(metric, PerformanceMetric):
                raise ReleaseEvidenceError("performance budget keys must be PerformanceMetric")
            if budget is not None and (
                not isinstance(budget, PerformanceBudget) or budget.metric is not metric
            ):
                raise ReleaseEvidenceError("performance budget must match its workload metric")
        known = set(gate_ids)
        declared_external = {
            gate.spec.gate_id
            for gate in self.gates
            if gate.spec.category == "external_adapter"
        }
        reported_external: set[str] = set()
        for external in self.external_capabilities:
            if not isinstance(external, ExternalCapabilityReport) or not set(
                external.gate_ids
            ).issubset(known):
                raise ReleaseEvidenceError("external capability report references an unknown gate")
            reported_external.update(external.gate_ids)
        if reported_external != declared_external:
            raise ReleaseEvidenceError(
                "external capability reports must cover every declared adapter gate"
            )

    @property
    def ready(self) -> bool:
        return all(gate.ready for gate in self.gates) and all(
            self.performance_budgets[metric] is not None for metric in PerformanceMetric
        )

    def blocking_gate_ids(self) -> tuple[str, ...]:
        blocking = {
            gate.spec.gate_id
            for gate in self.gates
            if gate.spec.required_for_ready and not gate.evidence.passed
        }
        for metric in PerformanceMetric:
            if self.performance_budgets[metric] is None:
                blocking.add(f"performance_{metric.value}")
        return tuple(sorted(blocking))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "ready": self.ready,
            "blocking_gate_ids": list(self.blocking_gate_ids()),
            "standards": [entry.to_mapping() for entry in self.standards],
            "libraries": dict(sorted(self.libraries.items())),
            "profile": dict(self.profile),
            "gates": [
                gate.to_mapping()
                for gate in sorted(self.gates, key=lambda item: item.spec.gate_id)
            ],
            "performance_budgets": [
                {
                    "metric": metric.value,
                    "budget": (
                        self.performance_budgets[metric].to_mapping()
                        if self.performance_budgets[metric] is not None
                        else None
                    ),
                }
                for metric in PerformanceMetric
            ],
            "external_capabilities": [
                item.to_mapping()
                for item in sorted(
                    self.external_capabilities, key=lambda item: item.capability_id
                )
            ],
            "compatibility_retirement": [
                item.to_mapping()
                for item in sorted(
                    self.compatibility_retirement, key=lambda item: item.path_id
                )
            ],
        }


def build_standards_matrix(
    registry: SemanticKnowledgeRegistry | None = None,
) -> tuple[StandardsMatrixEntry, ...]:
    """Return every exact offline-verified semantic registry resource pin."""
    selected = registry or get_knowledge_registry()
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
        )
        for resource in selected.resources
    )


def implementation_versions() -> dict[str, str]:
    """Capture the runtime/library versions influencing a conformance run."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "rdflib": _distribution_version("rdflib"),
        "kestrel_sovereign": _distribution_version("kestrel-sovereign"),
    }


def release_gate_specs(
    registry: SemanticKnowledgeRegistry | None = None,
) -> tuple[GateSpec, ...]:
    """Build every release gate from one declarative catalog.

    Draft registry capabilities are discovered instead of duplicated, so adding
    a 1.2 capability necessarily adds a stable-only rejection check.
    """
    selected = registry or get_knowledge_registry()
    fixtures = (
        GateSpec("rdf11_projection_fixture", "conformance", "rdf11_projection"),
        GateSpec("rdfs11_inference_fixture", "conformance", "rdfs11_inference"),
        GateSpec("owl2rl_inference_fixture", "conformance", "owl2rl_inference"),
        GateSpec("shacl2017_core_fixture", "conformance", "shacl2017_core"),
        GateSpec("sparql11_readonly_fixture", "conformance", "sparql11_readonly"),
        GateSpec(
            "rdf12_triple_term_fixture",
            "conformance",
            "rdf12_triple_terms",
            advertised=False,
            required_for_ready=False,
        ),
        GateSpec(
            "shacl12_fixture",
            "conformance",
            "shacl12",
            advertised=False,
            required_for_ready=False,
        ),
        GateSpec(
            "sparql12_fixture",
            "conformance",
            "sparql12",
            advertised=False,
            required_for_ready=False,
        ),
    )
    stable_only = tuple(
        GateSpec(
            f"stable_only_{_safe_gate_name(capability)}", "stable_only", capability
        )
        for capability in _experimental_capabilities(selected)
    )
    parity = tuple(
        GateSpec(f"{backend}_{surface}", "backend_parity", f"{backend}:{surface}")
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
    )
    performance = tuple(
        GateSpec(f"performance_{metric.value}", "performance", metric.value)
        for metric in PerformanceMetric
    )
    erasure = tuple(
        GateSpec(
            f"erasure_{stage.value}",
            "erasure",
            stage.value,
            owner=(
                "parametric_self"
                if stage is ErasureStage.SERVED_ADAPTER_ELIGIBILITY
                else "kestrel_core"
            ),
        )
        for stage in ErasureStage
    )
    external = tuple(
        GateSpec(gate_id, "external_adapter", gate_id, owner="parametric_self")
        for gate_id in (
            "external_corpus_consumed",
            "external_candidate_invalidated",
            "external_served_eligibility_rejected",
        )
    )
    live_agent = (
        GateSpec(
            "kite_http_invoke_release_drill",
            "live_agent",
            "recall_provenance:contradiction:quarantine:sleep:restart:erasure",
        ),
    )
    compatibility = (
        GateSpec(
            "legacy_fact_migration_equivalence",
            "compatibility_retirement",
            "legacy_fact_migration_compatibility",
            required_for_ready=False,
        ),
    )
    return (
        fixtures
        + stable_only
        + parity
        + performance
        + erasure
        + external
        + live_agent
        + compatibility
    )


def inspect_stable_only_capabilities(
    registry: SemanticKnowledgeRegistry | None = None,
) -> tuple[EvidenceRecord, ...]:
    """Prove the default codec and registry reject every draft capability."""
    selected = registry or get_knowledge_registry()
    # Reading a prior report should not load the RDF implementation.
    from .rdf_codec import RdfAssertionCodec, RdfCodecConfiguration

    codec = RdfAssertionCodec(registry=selected, configuration=RdfCodecConfiguration())
    if codec.capability_report.rdf12 is not None or codec.capability_report.sparql12 is not None:
        raise ReleaseEvidenceError("stable-only codec unexpectedly selected a draft capability")
    records: list[EvidenceRecord] = []
    for capability in _experimental_capabilities(selected):
        try:
            selected.select_capability(capability)
        except ExperimentalCapabilityError:
            records.append(
                EvidenceRecord(
                    gate_id=f"stable_only_{_safe_gate_name(capability)}",
                    state=EvidenceState.PASSED,
                    command=f"registry.select_capability({capability})",
                    exit_code=0,
                    artifact_ref="in-process stable-only registry selection",
                )
            )
        else:
            raise ReleaseEvidenceError(
                f"stable-only registry unexpectedly selected experimental {capability!r}"
            )
    if not records:
        raise ReleaseEvidenceError("stable-only evidence requires experimental capabilities")
    return tuple(records)


def release_evidence_template(
    registry: SemanticKnowledgeRegistry | None = None,
) -> SemanticReleaseEvidence:
    """Generate a current report whose unobserved work remains explicitly blocked."""
    selected = registry or get_knowledge_registry()
    stable_only = {record.gate_id: record for record in inspect_stable_only_capabilities(selected)}
    gates = tuple(
        GateResult(spec, stable_only.get(spec.gate_id, spec.initial_evidence()))
        for spec in release_gate_specs(selected)
    )
    retirement = CompatibilityRetirementDecision(
        path_id="legacy_fact_migration_compatibility",
        observed_window_ref=None,
        inventory_complete=False,
        unmigrated_eligible_rows=None,
        required_consumer_count=None,
        migration_equivalence=next(
            item.evidence
            for item in gates
            if item.spec.gate_id == "legacy_fact_migration_equivalence"
        ),
    )
    return SemanticReleaseEvidence(
        standards=build_standards_matrix(selected),
        libraries=implementation_versions(),
        profile={
            "canonical_contract": "semantic-kb-v1",
            "advertised_stable_capabilities": list(_stable_capabilities(selected)),
            "disabled_experimental_capabilities": list(_experimental_capabilities(selected)),
        },
        gates=gates,
        performance_budgets={metric: None for metric in PerformanceMetric},
        external_capabilities=(
            ExternalCapabilityReport(
                capability_id="parametric_self_governed_corpus",
                repository=PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
                source_revision=PARAMETRIC_SELF_EVIDENCE_REVISION,
                gate_ids=(
                    "external_corpus_consumed",
                    "external_candidate_invalidated",
                    "external_served_eligibility_rejected",
                ),
            ),
        ),
        compatibility_retirement=(retirement,),
    )


def apply_evidence_records(
    evidence: SemanticReleaseEvidence,
    records: Iterable[EvidenceRecord],
) -> SemanticReleaseEvidence:
    """Apply only declared evidence records; unknown or duplicate gates fail closed."""
    supplied = tuple(records)
    updates = {record.gate_id: record for record in supplied}
    if len(updates) != len(supplied):
        raise ReleaseEvidenceError("evidence records must not repeat a gate_id")
    known = {gate.spec.gate_id for gate in evidence.gates}
    unknown = sorted(set(updates) - known)
    if unknown:
        raise ReleaseEvidenceError(
            "evidence records do not match declared template gates: " + ", ".join(unknown)
        )
    gates = tuple(
        replace(gate, evidence=updates.get(gate.spec.gate_id, gate.evidence))
        for gate in evidence.gates
    )
    retirement = tuple(
        replace(
            decision,
            migration_equivalence=updates.get(
                decision.migration_equivalence.gate_id, decision.migration_equivalence
            ),
        )
        for decision in evidence.compatibility_retirement
    )
    return replace(evidence, gates=gates, compatibility_retirement=retirement)


def apply_performance_budgets(
    evidence: SemanticReleaseEvidence,
    budgets: Iterable[PerformanceBudget],
) -> SemanticReleaseEvidence:
    """Attach measured budgets only after their matching workload command passed."""
    supplied = tuple(budgets)
    updates = {budget.metric: budget for budget in supplied}
    if len(updates) != len(supplied):
        raise ReleaseEvidenceError("performance budgets must not repeat a metric")
    gate_evidence = {item.spec.gate_id: item.evidence for item in evidence.gates}
    for metric in updates:
        if not gate_evidence[f"performance_{metric.value}"].passed:
            raise ReleaseEvidenceError(
                f"performance budget {metric.value} requires passing command evidence first"
            )
    return replace(evidence, performance_budgets={**evidence.performance_budgets, **updates})


def evidence_record_from_mapping(value: Mapping[str, object]) -> EvidenceRecord:
    """Parse one externally produced content-free evidence record."""
    if not isinstance(value, Mapping):
        raise ReleaseEvidenceError("evidence record must be a mapping")
    try:
        state = EvidenceState(value["state"])
    except (KeyError, ValueError) as error:
        raise ReleaseEvidenceError("evidence record has an invalid state") from error
    return EvidenceRecord(
        gate_id=value.get("gate_id"),
        state=state,
        command=value.get("command"),
        exit_code=value.get("exit_code"),
        artifact_ref=value.get("artifact_ref"),
        reason_code=value.get("reason_code"),
        outside_advertised_capability=value.get("outside_advertised_capability", False),
    )


def performance_budget_from_mapping(value: Mapping[str, object]) -> PerformanceBudget:
    """Parse a measured workload budget from a separate benchmark artifact."""
    if not isinstance(value, Mapping):
        raise ReleaseEvidenceError("performance budget must be a mapping")
    try:
        metric = PerformanceMetric(value["metric"])
        samples = tuple(float(sample) for sample in value["samples_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseEvidenceError("performance budget has invalid metric or samples") from error
    return PerformanceBudget(
        metric=metric,
        samples_ms=samples,
        p95_ms=value.get("p95_ms"),
        budget_ms=value.get("budget_ms"),
        headroom_fraction=value.get("headroom_fraction"),
        fixture_digest=value.get("fixture_digest"),
    )


def run_command_evidence(
    gate_id: str,
    command: Sequence[str],
    *,
    artifact_ref: str,
    cwd: Path | None = None,
) -> EvidenceRecord:
    """Run one command and retain only its content-free exit result."""
    _require_identifier(gate_id, "gate_id")
    if not command or any(not isinstance(token, str) or not token for token in command):
        raise ReleaseEvidenceError("evidence command must be a non-empty string sequence")
    _require_reference(artifact_ref, "artifact_ref")
    result = subprocess.run(tuple(command), cwd=cwd, check=False)
    rendered = shlex.join(command)
    if result.returncode == 0:
        return EvidenceRecord(
            gate_id=gate_id,
            state=EvidenceState.PASSED,
            command=rendered,
            exit_code=0,
            artifact_ref=artifact_ref,
        )
    return EvidenceRecord(
        gate_id=gate_id,
        state=EvidenceState.FAILED,
        command=rendered,
        exit_code=result.returncode,
        artifact_ref=artifact_ref,
        reason_code="command_failed",
    )


def write_release_evidence(
    evidence: SemanticReleaseEvidence,
    output: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write canonical JSON without silently replacing a report."""
    _write_json(evidence.to_mapping(), output, overwrite=overwrite, kind="evidence artifact")


def write_evidence_record(
    record: EvidenceRecord,
    output: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write one standalone record for local or cross-repository exchange."""
    _write_json(record.to_mapping(), output, overwrite=overwrite, kind="evidence record")


def write_performance_budget(
    budget: PerformanceBudget,
    output: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write a measured budget separately from command-result evidence."""
    _write_json(budget.to_mapping(), output, overwrite=overwrite, kind="performance budget")


def _experimental_capabilities(registry: SemanticKnowledgeRegistry) -> tuple[str, ...]:
    return tuple(
        sorted(
            capability
            for resource in registry.resources
            if resource.maturity is StandardsMaturity.EXPERIMENTAL
            for capability in resource.capabilities
        )
    )


def _stable_capabilities(registry: SemanticKnowledgeRegistry) -> tuple[str, ...]:
    return tuple(
        sorted(
            capability
            for resource in registry.resources
            if resource.maturity is StandardsMaturity.STABLE
            for capability in resource.capabilities
        )
    )


def _write_json(value: Mapping[str, object], output: Path, *, overwrite: bool, kind: str) -> None:
    if output.exists() and not overwrite:
        raise ReleaseEvidenceError(
            f"refusing to replace existing {kind} {output}; pass --overwrite"
        )
    if not output.parent.exists():
        raise ReleaseEvidenceError(f"{kind} output parent does not exist: {output.parent}")
    output.write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _distribution_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _main(argv: Sequence[str] | None = None) -> int:
    """Compatibility wrapper for ``python -m ...release_evidence``."""
    from .release_evidence_cli import main

    return main(argv)


if __name__ == "__main__":  # pragma: no cover - subprocess tests cover the CLI.
    raise SystemExit(_main())
