"""CI helpers for encrypted workflow cassette bootstrap.

The normal workflow harness path replays encrypted cassettes. The first
red-team cassette needs a manual record gate so reviewer credentials are
only exposed after an operator has approved the exact cassette id, owner
DID, ref, and commit being recorded.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    CryptoSuiteError,
    get_suite,
)
from tests.fixtures.workflow_harness import (
    CASSETTE_KEY_ENV,
    CASSETTE_SECRET_STORE_ENV,
    EncryptedWorkflowCassetteStore,
    WorkflowCassetteError,
    assert_no_plaintext_workflow_cassettes,
)

OPERATOR_PUBLIC_KEY_ENV = "KESTREL_WORKFLOW_CASSETTE_OPERATOR_PUBLIC_KEY_HEX"
TRUSTED_REF_ENV = "KESTREL_WORKFLOW_CASSETTE_TRUSTED_REF"
RETENTION_SECONDS_ENV = "KESTREL_WORKFLOW_CASSETTE_RETENTION_SECONDS"
DEFAULT_TRUSTED_REF = "refs/heads/main"
DEFAULT_CI_RETENTION_SECONDS = 90 * 24 * 60 * 60


def canonical_record_approval_payload(
    *,
    repository: str,
    ref: str,
    sha: str,
    owner_did: str,
    cassette_id: str,
    mode: str = "record",
) -> bytes:
    """Canonical bytes an operator signs before CI unlocks record mode."""

    payload = {
        "cassette_id": cassette_id,
        "mode": mode,
        "owner_did": owner_did,
        "ref": ref,
        "repository": repository,
        "sha": sha,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _require_nonempty(name: str, value: str | None) -> str:
    if not value:
        raise WorkflowCassetteError(f"{name} is required")
    return value


def _retention_seconds_from_env(env: Mapping[str, str]) -> int:
    raw = env.get(RETENTION_SECONDS_ENV)
    if raw is None:
        return DEFAULT_CI_RETENTION_SECONDS
    try:
        retention_seconds = int(raw)
    except ValueError as exc:
        raise WorkflowCassetteError(
            f"{RETENTION_SECONDS_ENV} must be a positive integer"
        ) from exc
    if retention_seconds <= 0:
        raise WorkflowCassetteError(
            f"{RETENTION_SECONDS_ENV} must be a positive integer"
        )
    return retention_seconds


def verify_operator_record_signature(
    *,
    operator_did: str,
    operator_signature_hex: str,
    operator_public_key_hex: str,
    repository: str,
    ref: str,
    sha: str,
    owner_did: str,
    cassette_id: str,
) -> bytes:
    """Verify the operator signature and return the signed payload bytes."""

    for label, value in {
        "operator_did": operator_did,
        "operator_signature_hex": operator_signature_hex,
        "operator_public_key_hex": operator_public_key_hex,
        "repository": repository,
        "ref": ref,
        "sha": sha,
        "owner_did": owner_did,
        "cassette_id": cassette_id,
    }.items():
        _require_nonempty(label, value)
    if not operator_did.startswith("did:"):
        raise WorkflowCassetteError("operator_did must be a DID")
    if not owner_did.startswith("did:"):
        raise WorkflowCassetteError("owner_did must be a DID")

    try:
        public_key_bytes = bytes.fromhex(operator_public_key_hex)
        signature = bytes.fromhex(operator_signature_hex)
    except ValueError as exc:
        raise WorkflowCassetteError(
            "operator public key and signature must be hex"
        ) from exc

    payload = canonical_record_approval_payload(
        repository=repository,
        ref=ref,
        sha=sha,
        owner_did=owner_did,
        cassette_id=cassette_id,
    )
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    try:
        public_key = suite.deserialize_public_key(public_key_bytes)
        verified = suite.verify(payload, signature, public_key)
    except CryptoSuiteError as exc:
        raise WorkflowCassetteError(
            "operator record approval signature could not be verified"
        ) from exc
    if not verified:
        raise WorkflowCassetteError(
            "operator record approval signature did not verify"
        )
    return payload


def validate_record_gate_from_env(env: Mapping[str, str] = os.environ) -> bytes:
    """Fail closed unless the manual record gate is explicitly satisfied."""

    if env.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise WorkflowCassetteError(
            "workflow cassette record mode only runs from workflow_dispatch"
        )
    if env.get(CASSETTE_SECRET_STORE_ENV) != "approved":
        raise WorkflowCassetteError(
            "workflow cassette record mode requires manual approval via "
            f"{CASSETTE_SECRET_STORE_ENV}=approved"
        )
    trusted_ref = env.get(TRUSTED_REF_ENV, DEFAULT_TRUSTED_REF)
    github_ref = _require_nonempty("GITHUB_REF", env.get("GITHUB_REF"))
    if github_ref != trusted_ref:
        raise WorkflowCassetteError(
            "workflow cassette record mode must run from trusted ref "
            f"{trusted_ref}"
        )
    return verify_operator_record_signature(
        operator_did=_require_nonempty(
            "WORKFLOW_CASSETTE_OPERATOR_DID",
            env.get("WORKFLOW_CASSETTE_OPERATOR_DID"),
        ),
        operator_signature_hex=_require_nonempty(
            "WORKFLOW_CASSETTE_OPERATOR_SIGNATURE",
            env.get("WORKFLOW_CASSETTE_OPERATOR_SIGNATURE"),
        ),
        operator_public_key_hex=_require_nonempty(
            OPERATOR_PUBLIC_KEY_ENV,
            env.get(OPERATOR_PUBLIC_KEY_ENV),
        ),
        repository=_require_nonempty("GITHUB_REPOSITORY", env.get("GITHUB_REPOSITORY")),
        ref=github_ref,
        sha=_require_nonempty("GITHUB_SHA", env.get("GITHUB_SHA")),
        owner_did=_require_nonempty(
            "WORKFLOW_CASSETTE_OWNER_DID",
            env.get("WORKFLOW_CASSETTE_OWNER_DID"),
        ),
        cassette_id=_require_nonempty(
            "WORKFLOW_CASSETTE_ID",
            env.get("WORKFLOW_CASSETTE_ID"),
        ),
    )


def encrypt_payload_file_from_env(
    *,
    payload_file: Path,
    output_dir: Path,
    env: Mapping[str, str] = os.environ,
) -> Path:
    """Encrypt one recorded payload into the workflow cassette envelope format."""

    validate_record_gate_from_env(env)
    cassette_key = _require_nonempty(CASSETTE_KEY_ENV, env.get(CASSETTE_KEY_ENV))
    try:
        payload = json.loads(payload_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowCassetteError("recorded cassette payload must be JSON") from exc
    if not isinstance(payload, dict):
        raise WorkflowCassetteError("recorded cassette payload must be a JSON object")

    store = EncryptedWorkflowCassetteStore(
        output_dir,
        key=cassette_key,
        retention_seconds=_retention_seconds_from_env(env),
    )
    path = store.record(
        owner_did=env["WORKFLOW_CASSETTE_OWNER_DID"],
        cassette_id=env["WORKFLOW_CASSETTE_ID"],
        payload=payload,
    )
    assert_no_plaintext_workflow_cassettes(output_dir)
    return path


def _write_approval_payload(args: argparse.Namespace) -> int:
    payload = canonical_record_approval_payload(
        repository=args.repository,
        ref=args.ref,
        sha=args.sha,
        owner_did=args.owner_did,
        cassette_id=args.cassette_id,
    )
    if args.base64:
        print(base64.b64encode(payload).decode("ascii"))
    else:
        print(payload.decode("utf-8"))
    return 0


def _validate_gate(_args: argparse.Namespace) -> int:
    payload = validate_record_gate_from_env()
    print(payload.decode("utf-8"))
    return 0


def _encrypt(args: argparse.Namespace) -> int:
    path = encrypt_payload_file_from_env(
        payload_file=Path(args.payload_file),
        output_dir=Path(args.output_dir),
    )
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Workflow cassette CI bootstrap helper"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    approval = sub.add_parser(
        "approval-payload",
        help="Print the canonical payload an operator must sign.",
    )
    approval.add_argument("--repository", required=True)
    approval.add_argument("--ref", required=True)
    approval.add_argument("--sha", required=True)
    approval.add_argument("--owner-did", required=True)
    approval.add_argument("--cassette-id", required=True)
    approval.add_argument("--base64", action="store_true")
    approval.set_defaults(func=_write_approval_payload)

    validate = sub.add_parser(
        "validate-record-gate",
        help="Verify workflow_dispatch, manual approval, secret presence, and signature.",
    )
    validate.set_defaults(func=_validate_gate)

    encrypt = sub.add_parser(
        "encrypt",
        help="Encrypt a recorded cassette JSON payload into tests/cassettes.",
    )
    encrypt.add_argument("--payload-file", required=True)
    encrypt.add_argument("--output-dir", required=True)
    encrypt.set_defaults(func=_encrypt)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WorkflowCassetteError as exc:
        parser.exit(2, f"workflow cassette bootstrap failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
