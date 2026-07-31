"""CLI adapter for :mod:`kestrel_sovereign.knowledge.release_evidence`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from .release_evidence import (
    EvidenceRecord,
    EvidenceState,
    PerformanceMetric,
    PerformanceBudget,
    ReleaseEvidenceError,
    apply_evidence_records,
    apply_performance_budgets,
    evidence_record_from_mapping,
    performance_budget_from_mapping,
    release_evidence_template,
    run_command_evidence,
    write_evidence_record,
    write_performance_budget,
    write_release_evidence,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the content-free template, record, budget, and assemble commands."""
    parser = argparse.ArgumentParser(
        description="Generate conservative semantic release-evidence artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    template = subparsers.add_parser(
        "template", help="write a non-ready release template"
    )
    template.add_argument("--output", type=Path, required=True)
    template.add_argument("--overwrite", action="store_true")
    record = subparsers.add_parser(
        "record", help="run one command and write its outcome"
    )
    record.add_argument("--gate", required=True)
    record.add_argument("--artifact-ref", required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--cwd", type=Path)
    record.add_argument("--overwrite", action="store_true")
    record.add_argument("run_command", nargs=argparse.REMAINDER)
    block = subparsers.add_parser(
        "block", help="write a content-free blocked result without claiming a pass"
    )
    block.add_argument("--gate", required=True)
    block.add_argument("--reason-code", required=True)
    block.add_argument("--artifact-ref")
    block.add_argument("--output", type=Path, required=True)
    block.add_argument("--overwrite", action="store_true")
    budget = subparsers.add_parser("budget", help="derive a measured workload budget")
    budget.add_argument(
        "--metric", choices=[item.value for item in PerformanceMetric], required=True
    )
    budget.add_argument("--samples-ms", type=float, nargs="+", required=True)
    budget.add_argument("--headroom-fraction", type=float, required=True)
    budget.add_argument("--fixture", type=Path, required=True)
    budget.add_argument("--output", type=Path, required=True)
    budget.add_argument("--overwrite", action="store_true")
    assemble = subparsers.add_parser(
        "assemble", help="apply declared results to a template"
    )
    assemble.add_argument("--record", type=Path, action="append", default=[])
    assemble.add_argument("--budget", type=Path, action="append", default=[])
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
            command = tuple(args.run_command)
            if command[:1] == ("--",):
                command = command[1:]
            record = run_command_evidence(
                args.gate, command, artifact_ref=args.artifact_ref, cwd=args.cwd
            )
            write_evidence_record(record, args.output, overwrite=args.overwrite)
            print(
                "semantic release evidence record written: "
                f"{args.output} ({record.gate_id}={record.state.value})"
            )
            return 0 if record.passed else record.exit_code or 1
        if args.command == "block":
            blocked = EvidenceRecord(
                gate_id=args.gate,
                state=EvidenceState.BLOCKED,
                artifact_ref=args.artifact_ref,
                reason_code=args.reason_code,
            )
            write_evidence_record(blocked, args.output, overwrite=args.overwrite)
            print(
                "semantic release evidence record written: "
                f"{args.output} ({blocked.gate_id}={blocked.state.value})"
            )
            return 0
        if args.command == "budget":
            fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
            if not isinstance(fixture, dict):
                raise ReleaseEvidenceError("benchmark fixture must be a JSON object")
            measured = PerformanceBudget.from_observed(
                PerformanceMetric(args.metric),
                args.samples_ms,
                headroom_fraction=args.headroom_fraction,
                fixture_description=fixture,
            )
            write_performance_budget(measured, args.output, overwrite=args.overwrite)
            print(
                "semantic performance budget written: "
                f"{args.output} ({measured.metric.value})"
            )
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
        write_release_evidence(evidence, args.output, overwrite=args.overwrite)
        print(
            "semantic release evidence assembled: "
            f"{args.output} (ready={str(evidence.ready).lower()})"
        )
        return 0
    except (ReleaseEvidenceError, OSError, ValueError) as error:
        print(f"semantic release evidence failed: {error}", file=sys.stderr)
        return 1
