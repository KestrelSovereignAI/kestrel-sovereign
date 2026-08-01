"""Separate verifier-only CLI; public release-evidence CLI remains structural."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .release_evidence import ReleaseEvidenceError
from .release_evidence_freshness import ExternalFreshnessLedger
from .release_evidence_verifier import (
    VerifierReceiptIdentity,
    issue_verification_receipt,
    load_budgets,
    load_external_report,
    load_records,
    read_verifier_configuration,
    trusted_assemble,
    write_verification_receipt,
    write_verified_evidence,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verifier-owned semantic release evidence operations.")
    parser.add_argument("--config", type=Path, required=True, help="protected verifier configuration")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("issue-challenge", help="issue a one-time external CI challenge")
    assemble = commands.add_parser("assemble", help="verify evidence and issue a verifier receipt")
    assemble.add_argument("--record", type=Path, action="append", default=[])
    assemble.add_argument("--budget", type=Path, action="append", default=[])
    assemble.add_argument("--external-report", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = read_verifier_configuration(args.config)
        ledger = ExternalFreshnessLedger(config.ledger_path, trusted_root=config.trusted_root)
        if args.command == "issue-challenge":
            print(ledger.issue_challenge())
            return 0
        report = load_external_report(args.external_report)
        identity = VerifierReceiptIdentity.from_configuration(config)
        evidence = trusted_assemble(
            records=load_records(args.record), budgets=load_budgets(args.budget), report=report,
            trust_policy=config.trust_policy, freshness_ledger=ledger,
            expected_evidence_runner_revision=config.expected_external_runner_revision,
        )
        receipt = issue_verification_receipt(evidence, policy_digest=config.policy_digest, identity=identity)
        ledger.record_verified_receipt(
            report, evidence_digest=receipt.evidence_digest, policy_digest=receipt.policy_digest,
            receipt_digest=receipt.digest,
        )
        write_verified_evidence(evidence, args.output)
        write_verification_receipt(receipt, args.receipt_output)
        print("semantic release evidence verified and receipt issued")
        return 0
    except (ReleaseEvidenceError, OSError, ValueError) as error:
        print(f"semantic verifier failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
