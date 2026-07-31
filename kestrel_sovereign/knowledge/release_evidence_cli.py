"""CLI for content-free semantic release evidence.

The public CLI never accepts an observation JSON object or benchmark samples
to create passed evidence. ``run`` invokes only an immutable catalog workload
and signs the emitted result; ``assemble`` preserves structurally valid
submissions without making a signer-trust or release-readiness claim.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from .release_evidence import (
    ArtifactReference,
    EvidenceRecord,
    EvidenceState,
    ReleaseEvidenceError,
    TelemetryAttestation,
    attach_structural_external_capability_report,
    attach_structural_retirement_telemetry,
    apply_structural_evidence_records,
    apply_structural_performance_budgets,
    evidence_record_from_mapping,
    external_capability_report_from_mapping,
    performance_budget_from_mapping,
    release_gate_specs,
    structural_release_evidence_template,
    telemetry_attestation_from_mapping,
    write_evidence_record,
    write_performance_budget,
    write_release_evidence,
    write_telemetry_attestation,
)
from .release_evidence_execution import (
    CatalogExecutionAuthority,
    CatalogSigningIdentity,
    default_catalog_workloads,
)


def _gate_spec(gate_id: str):
    for spec in release_gate_specs():
        if spec.gate_id == gate_id:
            return spec
    raise ReleaseEvidenceError(f"unknown release gate: {gate_id}")


def _artifact(args: argparse.Namespace) -> ArtifactReference:
    return ArtifactReference(args.artifact_ref, args.artifact_digest)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate conservative, catalog-bound evidence artifacts."""
    parser = argparse.ArgumentParser(
        description="Generate spec-bound semantic release-evidence artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template", help="write an unverified non-ready release template")
    template.add_argument("--output", type=Path, required=True)
    template.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser(
        "run",
        help="execute one allowlisted catalog workload and write its signed result",
    )
    run.add_argument("--gate", required=True)
    run.add_argument("--signing-key-file", type=Path, required=True)
    run.add_argument("--issuer-id", required=True)
    run.add_argument("--key-id", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--budget-output", type=Path)
    run.add_argument("--overwrite", action="store_true")

    # Keep the former spelling as an explicit fail-closed migration error;
    # accepting its old observation/sample inputs would reintroduce forgery.
    record = subparsers.add_parser(
        "record",
        help="disabled: use run for an allowlisted workload or assemble verified CI evidence",
    )
    record.add_argument("--gate")
    record.add_argument("--artifact-ref")
    record.add_argument("--artifact-digest")
    record.add_argument("--observation-json")
    record.add_argument("--output", type=Path)
    record.add_argument("--overwrite", action="store_true")

    block = subparsers.add_parser("block", help="write a content-free blocked result")
    block.add_argument("--gate", required=True)
    block.add_argument("--reason-code", required=True)
    block.add_argument("--output", type=Path, required=True)
    block.add_argument("--overwrite", action="store_true")

    budget = subparsers.add_parser(
        "budget",
        help="disabled: a budget is emitted only by an allowlisted benchmark workload",
    )
    budget.add_argument("--gate")
    budget.add_argument("--samples", type=float, nargs="+")
    budget.add_argument("--headroom-fraction", type=float)
    budget.add_argument("--artifact-ref")
    budget.add_argument("--artifact-digest")
    budget.add_argument("--output", type=Path)
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

    assemble = subparsers.add_parser(
        "assemble",
        help="inspect structurally valid catalog-bound records without a trust verdict",
    )
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
            evidence = structural_release_evidence_template()
            write_release_evidence(evidence, args.output, overwrite=args.overwrite)
            print(
                "semantic release evidence template written: "
                f"{args.output} (ready=false, trust_status=unverified)"
            )
            return 0
        if args.command == "record":
            raise ReleaseEvidenceError(
                "record is disabled because caller-supplied observations cannot create passed evidence; "
                "use run or assemble a verified external CI record"
            )
        if args.command == "run":
            spec = _gate_spec(args.gate)
            if spec.performance_target is not None and args.budget_output is None:
                raise ReleaseEvidenceError("performance workload run requires --budget-output")
            if spec.performance_target is None and args.budget_output is not None:
                raise ReleaseEvidenceError("--budget-output is valid only for a performance workload")
            identity = CatalogSigningIdentity.from_private_key_file(
                args.signing_key_file,
                issuer_id=args.issuer_id,
                key_id=args.key_id,
            )
            execution = asyncio.run(
                CatalogExecutionAuthority(identity, default_catalog_workloads()).execute(spec)
            )
            write_evidence_record(execution.record, args.output, overwrite=args.overwrite)
            if execution.budget is not None:
                assert args.budget_output is not None
                write_performance_budget(execution.budget, args.budget_output, overwrite=args.overwrite)
            print(
                "semantic catalog workload recorded: "
                f"{args.output} ({execution.record.gate_id}={execution.record.state.value})"
            )
            if execution.record.state is EvidenceState.BLOCKED:
                # The explicit ``block`` command remains available when an
                # operator intentionally records an observed block.  ``run``
                # instead reports an unavailable/failed catalog workload to
                # automation through a nonzero status after preserving its
                # content-free blocked artifact.
                return 2 if execution.record.reason_code == "catalog_workload_unavailable" else 1
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
            raise ReleaseEvidenceError(
                "budget is disabled because caller-supplied samples cannot create a release budget; "
                "use run for an allowlisted benchmark workload"
            )
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
        evidence = apply_structural_evidence_records(
            structural_release_evidence_template(), records
        )
        evidence = apply_structural_performance_budgets(evidence, budgets)
        if args.retirement_telemetry is not None:
            telemetry_mapping = json.loads(args.retirement_telemetry.read_text(encoding="utf-8"))
            evidence = attach_structural_retirement_telemetry(
                evidence,
                telemetry_attestation_from_mapping(telemetry_mapping),
            )
        if args.external_report is not None:
            report_mapping = json.loads(args.external_report.read_text(encoding="utf-8"))
            evidence = attach_structural_external_capability_report(
                evidence,
                external_capability_report_from_mapping(report_mapping),
            )
        write_release_evidence(evidence, args.output, overwrite=args.overwrite)
        print(
            "semantic release evidence assembled: "
            f"{args.output} (ready=false, trust_status=unverified, "
            f"structurally_complete={str(evidence.structurally_complete).lower()})"
        )
        return 0
    except (ReleaseEvidenceError, OSError, ValueError) as error:
        print(f"semantic release evidence failed: {error}", file=sys.stderr)
        return 1
