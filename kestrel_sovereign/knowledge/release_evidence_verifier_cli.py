"""Separate verifier-only CLI; public release-evidence CLI remains structural."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .release_evidence import ReleaseEvidenceError
from .release_evidence_freshness import ExternalFreshnessLedger
from .release_evidence_verifier import (
    VerifierReceiptIdentity,
    combine_external_envelope_submission,
    issue_verification_receipt,
    finalize_verified_artifacts,
    load_budgets,
    load_external_envelope,
    load_records,
    read_verifier_configuration,
    prepare_trusted_evidence,
)


# Provisioned by the host administrator before any verifier job runs.  This
# path is intentionally not an argument or environment variable: callers of
# this CLI must not choose their own trust policy, ledger, or receipt key.
HOST_VERIFIER_CONFIGURATION = Path("/etc/kestrel/semantic-release-verifier.json")


class _StoreOnce(argparse.Action):
    """Reject duplicate singleton evidence inputs before verifier state opens."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may be supplied only once")
        setattr(namespace, self.dest, values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verifier-owned semantic release evidence operations.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("issue-challenge", help="issue a one-time external CI challenge")
    assemble = commands.add_parser("assemble", help="verify evidence and issue a verifier receipt")
    assemble.add_argument("--record", type=Path, action="append", default=[])
    assemble.add_argument("--budget", type=Path, action="append", default=[])
    assemble.add_argument(
        "--external-envelope",
        type=Path,
        action=_StoreOnce,
        help="the exact signed parametric-self external-CI envelope; mutually exclusive with split external inputs",
    )
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = read_verifier_configuration(HOST_VERIFIER_CONFIGURATION)
        ledger = ExternalFreshnessLedger(config.ledger_path, trusted_root=config.trusted_root)
        if args.command == "issue-challenge":
            print(ledger.issue_challenge())
            return 0
        envelope = (
            load_external_envelope(args.external_envelope)
            if args.external_envelope is not None
            else None
        )
        records, report = combine_external_envelope_submission(
            records=load_records(args.record), envelope=envelope
        )
        identity = VerifierReceiptIdentity.from_configuration(config)
        evidence = prepare_trusted_evidence(
            records=records, budgets=load_budgets(args.budget), report=report,
            trust_policy=config.trust_policy,
            expected_evidence_runner_revision=config.expected_external_runner_revision,
        )
        receipt = issue_verification_receipt(evidence, policy_digest=config.policy_digest, identity=identity)
        finalize_verified_artifacts(
            evidence, receipt, evidence_output=args.output, receipt_output=args.receipt_output,
            configuration=config, freshness_ledger=ledger,
        )
        print("semantic release evidence verified and receipt issued")
        return 0
    except (ReleaseEvidenceError, OSError, ValueError) as error:
        print(f"semantic verifier failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
