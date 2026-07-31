"""CLI for spec-bound, content-free semantic release attestations.

There is intentionally no ``record -- <arbitrary argv>`` escape hatch.  The
gate catalog supplies the runner identity, command-pattern digest, execution
environment, fixture binding, and observation schema.  This command accepts
only an opaque safe artifact reference/digest and the schema's content-free
numeric/boolean/digest observation values.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys

from .release_evidence import (
    ArtifactReference,
    EvidenceRecord,
    EvidenceState,
    PerformanceBudget,
    ReleaseEvidenceError,
    TelemetryAttestation,
    attach_external_capability_report,
    attach_retirement_telemetry,
    apply_evidence_records,
    apply_performance_budgets,
    evidence_record_from_mapping,
    external_capability_report_from_mapping,
    performance_budget_from_mapping,
    release_evidence_template,
    release_gate_specs,
    telemetry_attestation_from_mapping,
    write_evidence_record,
    write_performance_budget,
    write_release_evidence,
    write_telemetry_attestation,
)


def _gate_spec(gate_id: str):
    for spec in release_gate_specs():
        if spec.gate_id == gate_id:
            return spec
    raise ReleaseEvidenceError(f"unknown release gate: {gate_id}")


def _observation(value: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ReleaseEvidenceError("observation_json must be valid JSON") from error
    if not isinstance(parsed, dict):
        raise ReleaseEvidenceError("observation_json must be a JSON object")
    return parsed


def _artifact(args: argparse.Namespace) -> ArtifactReference:
    return ArtifactReference(args.artifact_ref, args.artifact_digest)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate conservative, catalog-bound evidence artifacts."""
    parser = argparse.ArgumentParser(
        description="Generate spec-bound semantic release-evidence artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template", help="write a non-ready release template")
    template.add_argument("--output", type=Path, required=True)
    template.add_argument("--overwrite", action="store_true")

    record = subparsers.add_parser(
        "record",
        help="write a result bound to the declared runner/spec; arbitrary argv is not accepted",
    )
    record.add_argument("--gate", required=True)
    record.add_argument("--artifact-ref", required=True)
    record.add_argument("--artifact-digest", required=True)
    record.add_argument("--observation-json", required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--overwrite", action="store_true")

    block = subparsers.add_parser("block", help="write a content-free blocked result")
    block.add_argument("--gate", required=True)
    block.add_argument("--reason-code", required=True)
    block.add_argument("--output", type=Path, required=True)
    block.add_argument("--overwrite", action="store_true")

    budget = subparsers.add_parser(
        "budget", help="derive a backend/mode-specific measured budget from a performance gate"
    )
    budget.add_argument("--gate", required=True)
    budget.add_argument("--samples", type=float, nargs="+", required=True)
    budget.add_argument("--headroom-fraction", type=float, required=True)
    budget.add_argument("--artifact-ref", required=True)
    budget.add_argument("--artifact-digest", required=True)
    budget.add_argument("--output", type=Path, required=True)
    budget.add_argument("--overwrite", action="store_true")

    telemetry = subparsers.add_parser(
        "telemetry",
        help="write a content-free, digest-bound compatibility telemetry attestation",
    )
    telemetry.add_argument("--window-started-at", required=True)
    telemetry.add_argument("--window-ended-at", required=True)
    telemetry.add_argument("--inventory-digest", required=True)
    telemetry.add_argument("--inventory-complete", action="store_true")
    telemetry.add_argument("--unmigrated-eligible-rows", type=int, required=True)
    telemetry.add_argument("--required-consumer-count", type=int, required=True)
    telemetry.add_argument("--artifact-ref", required=True)
    telemetry.add_argument("--artifact-digest", required=True)
    telemetry.add_argument("--output", type=Path, required=True)
    telemetry.add_argument("--overwrite", action="store_true")

    assemble = subparsers.add_parser("assemble", help="apply only catalog-bound records")
    assemble.add_argument("--record", type=Path, action="append", default=[])
    assemble.add_argument("--budget", type=Path, action="append", default=[])
    assemble.add_argument(
        "--retirement-telemetry",
        type=Path,
        help="one digest-bound telemetry artifact; it is bound only to the catalog migration gate",
    )
    assemble.add_argument(
        "--external-report",
        type=Path,
        help="one Pself report with exact repository, revision, correlated results, and artifacts",
    )
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            evidence = release_evidence_template()
            write_release_evidence(evidence, args.output, overwrite=args.overwrite)
            print(f"semantic release evidence template written: {args.output} (ready=false)")
            return 0
        if args.command == "record":
            spec = _gate_spec(args.gate)
            attestation = EvidenceRecord.attest(
                spec,
                _observation(args.observation_json),
                _artifact(args),
            )
            # GateResult exercises the same binding validation used by assemble
            # before writing a standalone record.
            from .release_evidence_models import GateResult

            GateResult(spec, attestation)
            write_evidence_record(attestation, args.output, overwrite=args.overwrite)
            print(
                "semantic release evidence record written: "
                f"{args.output} ({attestation.gate_id}={attestation.state.value})"
            )
            return 0
        if args.command == "block":
            # Look up the gate so a typo cannot become a disconnected block file.
            _gate_spec(args.gate)
            blocked = EvidenceRecord(
                gate_id=args.gate,
                state=EvidenceState.BLOCKED,
                reason_code=args.reason_code,
            )
            write_evidence_record(blocked, args.output, overwrite=args.overwrite)
            print(
                "semantic release evidence record written: "
                f"{args.output} ({blocked.gate_id}={blocked.state.value})"
            )
            return 0
        if args.command == "budget":
            spec = _gate_spec(args.gate)
            measured = PerformanceBudget.from_observed(
                spec,
                args.samples,
                headroom_fraction=args.headroom_fraction,
                artifact=_artifact(args),
            )
            write_performance_budget(measured, args.output, overwrite=args.overwrite)
            print(
                "semantic performance budget written: "
                f"{args.output} ({spec.gate_id})"
            )
            return 0
        if args.command == "telemetry":
            attestation = TelemetryAttestation.attest(
                window_started_at=args.window_started_at,
                window_ended_at=args.window_ended_at,
                inventory_digest=args.inventory_digest,
                inventory_complete=args.inventory_complete,
                unmigrated_eligible_rows=args.unmigrated_eligible_rows,
                required_consumer_count=args.required_consumer_count,
                artifact=_artifact(args),
            )
            write_telemetry_attestation(attestation, args.output, overwrite=args.overwrite)
            print(f"semantic retirement telemetry written: {args.output}")
            return 0
        records = tuple(
            evidence_record_from_mapping(json.loads(path.read_text(encoding="utf-8")))
            for path in args.record
        )
        budgets = tuple(
            performance_budget_from_mapping(json.loads(path.read_text(encoding="utf-8")))
            for path in args.budget
        )
        evidence = apply_evidence_records(release_evidence_template(), records)
        evidence = apply_performance_budgets(evidence, budgets)
        if args.retirement_telemetry is not None:
            telemetry_mapping = json.loads(args.retirement_telemetry.read_text(encoding="utf-8"))
            evidence = attach_retirement_telemetry(
                evidence,
                telemetry_attestation_from_mapping(telemetry_mapping),
            )
        if args.external_report is not None:
            report_mapping = json.loads(args.external_report.read_text(encoding="utf-8"))
            evidence = attach_external_capability_report(
                evidence,
                external_capability_report_from_mapping(report_mapping),
            )
        write_release_evidence(evidence, args.output, overwrite=args.overwrite)
        print(
            "semantic release evidence assembled: "
            f"{args.output} (ready={str(evidence.ready).lower()})"
        )
        return 0
    except (ReleaseEvidenceError, OSError, ValueError) as error:
        print(f"semantic release evidence failed: {error}", file=sys.stderr)
        return 1
