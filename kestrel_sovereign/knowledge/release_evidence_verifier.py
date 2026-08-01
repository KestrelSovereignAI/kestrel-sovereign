"""Verifier-owned assembly and signed receipts for semantic release evidence.

This module is deliberately separate from the public structural assembler.
It accepts only independently produced, signed records, a protected public-key
policy, and a verifier-owned freshness ledger; it never accepts observations,
samples, signer keys, or a producer-supplied replay receipt.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

from .release_evidence import (
    CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
    SemanticReleaseEvidence,
    apply_evidence_records,
    apply_performance_budgets,
    attach_external_capability_report,
    evidence_record_from_mapping,
    external_capability_report_from_mapping,
    performance_budget_from_mapping,
    release_evidence_template,
    trusted_execution_policy_from_mapping,
)
from .release_evidence_freshness import ExternalFreshnessLedger
from .release_evidence_models import (
    EvidenceRecord,
    ExternalCapabilityReport,
    PerformanceBudget,
    ReleaseEvidenceError,
    TrustedExecutionPolicy,
    _canonical_json,
    _sha256,
)


VERIFICATION_RECEIPT_VERSION = "semantic-release-verification-receipt-v1"
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,127}$")


def _read_owner_only_file(path: Path, *, kind: str) -> bytes:
    """Read a verifier policy only from a stable owner-only regular file."""
    if not isinstance(path, Path) or not path.is_absolute():
        raise ReleaseEvidenceError(f"{kind} path must be absolute")
    try:
        before = path.lstat()
    except OSError as error:
        raise ReleaseEvidenceError(f"{kind} cannot be inspected") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) & 0o077
    ):
        raise ReleaseEvidenceError(f"{kind} must be an owner-only regular non-symlink file")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) & 0o077
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ReleaseEvidenceError(f"{kind} changed while being opened")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            return source.read()
    except OSError as error:
        raise ReleaseEvidenceError(f"{kind} cannot be securely read") from error
    finally:
        if descriptor != -1:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class VerifierConfiguration:
    """One protected authority configuration for the separate verifier CLI."""

    trusted_root: Path
    ledger_path: Path
    trust_policy: TrustedExecutionPolicy
    policy_digest: str
    expected_external_runner_revision: str
    receipt_key_file: Path
    receipt_issuer_id: str
    receipt_key_id: str
    receipt_public_key: str


def read_verifier_configuration(path: Path) -> VerifierConfiguration:
    """Load one protected, root-pinned verifier configuration."""
    try:
        raw = _read_owner_only_file(path, kind="verifier configuration")
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseEvidenceError("verifier configuration is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ReleaseEvidenceError("verifier configuration must be a mapping")
    expected = {"trusted_root", "ledger_path", "trust_policy", "expected_external_runner_revision", "receipt_key_file", "receipt_issuer_id", "receipt_key_id", "receipt_public_key"}
    if set(value) != expected:
        raise ReleaseEvidenceError("verifier configuration has unknown or missing fields")
    root = Path(str(value["trusted_root"]))
    ledger_path = Path(str(value["ledger_path"]))
    receipt_key_file = Path(str(value["receipt_key_file"]))
    try:
        resolved_root = root.resolve(strict=True)
        relative_config = path.relative_to(resolved_root)
        ledger_path.relative_to(resolved_root)
        receipt_key_file.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise ReleaseEvidenceError("verifier configuration must be rooted in its private trusted_root") from error
    if root != resolved_root or relative_config == Path("."):
        raise ReleaseEvidenceError("verifier configuration trusted_root must be resolved and contain its config")
    policy_mapping = value["trust_policy"]
    if not isinstance(policy_mapping, Mapping):
        raise ReleaseEvidenceError("verifier configuration trust_policy must be a mapping")
    revision = value["expected_external_runner_revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ReleaseEvidenceError("verifier configuration runner revision must be a full lowercase commit SHA")
    issuer_id, key_id, public_key = (value[field] for field in ("receipt_issuer_id", "receipt_key_id", "receipt_public_key"))
    if not all(isinstance(item, str) for item in (issuer_id, key_id, public_key)) or not _IDENTIFIER_RE.fullmatch(issuer_id) or not _IDENTIFIER_RE.fullmatch(key_id) or not re.fullmatch(r"[0-9a-f]{64}", public_key):
        raise ReleaseEvidenceError("verifier receipt identity is invalid")
    return VerifierConfiguration(resolved_root, ledger_path, trusted_execution_policy_from_mapping(policy_mapping), _sha256(_canonical_json(policy_mapping)), revision, receipt_key_file, issuer_id, key_id, public_key)


def _read_submission(path: Path, *, kind: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseEvidenceError(f"{kind} cannot be read") from error
    if not isinstance(value, Mapping):
        raise ReleaseEvidenceError(f"{kind} must be a mapping")
    return value


def load_records(paths: Iterable[Path]) -> tuple[EvidenceRecord, ...]:
    return tuple(
        evidence_record_from_mapping(_read_submission(path, kind="evidence record"))
        for path in paths
    )


def load_budgets(paths: Iterable[Path]) -> tuple[PerformanceBudget, ...]:
    return tuple(
        performance_budget_from_mapping(_read_submission(path, kind="performance budget"))
        for path in paths
    )


def load_external_report(path: Path) -> ExternalCapabilityReport:
    return external_capability_report_from_mapping(_read_submission(path, kind="external capability report"))


@dataclass(frozen=True, slots=True)
class VerificationReceipt:
    """Content-free verifier signature over one ready release artifact."""

    evidence_digest: str
    policy_digest: str
    external_freshness_receipt: str
    run_nonce: str
    capability_source_revision: str
    evidence_runner_revision: str
    issuer_id: str
    key_id: str
    signature: str
    version: str = VERIFICATION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.version != VERIFICATION_RECEIPT_VERSION:
            raise ReleaseEvidenceError("unsupported verification receipt version")
        if not all(isinstance(value, str) and len(value) == 64 for value in (
            self.evidence_digest, self.policy_digest, self.external_freshness_receipt, self.run_nonce,
        )):
            raise ReleaseEvidenceError("verification receipt requires content-free digests")
        if any(character not in "0123456789abcdef" for value in (
            self.evidence_digest, self.policy_digest, self.external_freshness_receipt, self.run_nonce,
        ) for character in value):
            raise ReleaseEvidenceError("verification receipt requires lowercase digest values")
        if not self.issuer_id or not self.key_id or len(self.signature) != 128:
            raise ReleaseEvidenceError("verification receipt identity or signature is invalid")

    def signed_payload(self) -> dict[str, str]:
        return {
            "version": self.version,
            "evidence_digest": self.evidence_digest,
            "policy_digest": self.policy_digest,
            "external_freshness_receipt": self.external_freshness_receipt,
            "core_release_evidence_contract_digest": CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
            "run_nonce": self.run_nonce,
            "capability_source_revision": self.capability_source_revision,
            "evidence_runner_revision": self.evidence_runner_revision,
            "issuer_id": self.issuer_id,
            "key_id": self.key_id,
        }

    def to_mapping(self) -> dict[str, str]:
        return {**self.signed_payload(), "signature": self.signature}

    @property
    def digest(self) -> str:
        return _sha256(_canonical_json(self.to_mapping()))


def verification_receipt_from_mapping(value: Mapping[str, object]) -> VerificationReceipt:
    expected = {"version", "evidence_digest", "policy_digest", "external_freshness_receipt", "core_release_evidence_contract_digest", "run_nonce", "capability_source_revision", "evidence_runner_revision", "issuer_id", "key_id", "signature"}
    if set(value) != expected or value.get("core_release_evidence_contract_digest") != CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST:
        raise ReleaseEvidenceError("verification receipt has unknown or missing fields")
    return VerificationReceipt(
        evidence_digest=str(value["evidence_digest"]), policy_digest=str(value["policy_digest"]),
        external_freshness_receipt=str(value["external_freshness_receipt"]), run_nonce=str(value["run_nonce"]),
        capability_source_revision=str(value["capability_source_revision"]), evidence_runner_revision=str(value["evidence_runner_revision"]),
        issuer_id=str(value["issuer_id"]), key_id=str(value["key_id"]), signature=str(value["signature"]), version=str(value["version"]),
    )


def verify_verification_receipt(receipt: VerificationReceipt, identity: VerifierConfiguration) -> None:
    if receipt.issuer_id != identity.receipt_issuer_id or receipt.key_id != identity.receipt_key_id:
        raise ReleaseEvidenceError("verification receipt signer does not match verifier configuration")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(identity.receipt_public_key)).verify(
            bytes.fromhex(receipt.signature), _canonical_json(receipt.signed_payload()).encode("utf-8")
        )
    except (InvalidSignature, ValueError) as error:
        raise ReleaseEvidenceError("verification receipt signature verification failed") from error


def trusted_assemble(
    *,
    records: Iterable[EvidenceRecord],
    budgets: Iterable[PerformanceBudget],
    report: ExternalCapabilityReport,
    trust_policy: TrustedExecutionPolicy,
    freshness_ledger: ExternalFreshnessLedger,
    expected_evidence_runner_revision: str,
) -> SemanticReleaseEvidence:
    """Verify signed submissions and consume the verifier-issued challenge.

    No challenge is consumed until every non-external readiness requirement is
    already satisfied, preventing an incomplete core submission from burning a
    valid external run.
    """
    evidence = apply_evidence_records(
        release_evidence_template(), records, trust_policy=trust_policy
    )
    evidence = apply_performance_budgets(evidence, budgets, trust_policy=trust_policy)
    if set(evidence.blocking_gate_ids()) != {"external_adapter_attestation"}:
        raise ReleaseEvidenceError(
            "trusted assembly requires all signed core records and performance budgets before external ingestion"
        )
    evidence = attach_external_capability_report(
        evidence,
        report,
        freshness_ledger=freshness_ledger,
        expected_evidence_runner_revision=expected_evidence_runner_revision,
    )
    if not evidence.ready:
        raise ReleaseEvidenceError("trusted assembly cannot issue a receipt for non-ready evidence")
    return evidence


def issue_verification_receipt(
    evidence: SemanticReleaseEvidence,
    *,
    policy_digest: str,
    identity: "VerifierReceiptIdentity",
) -> VerificationReceipt:
    """Sign a receipt only after all readiness checks are true."""
    if not evidence.ready or len(evidence.external_capabilities) != 1:
        raise ReleaseEvidenceError("verification receipt requires ready trusted release evidence")
    if not isinstance(policy_digest, str) or len(policy_digest) != 64:
        raise ReleaseEvidenceError("verification receipt requires a trusted policy digest")
    evidence_digest = _sha256(_canonical_json(evidence.to_mapping()))
    report = evidence.external_capabilities[0]
    unsigned = {
        "version": VERIFICATION_RECEIPT_VERSION,
        "evidence_digest": evidence_digest,
        "policy_digest": policy_digest,
        "external_freshness_receipt": report.freshness_receipt,
        "run_nonce": report.run_nonce,
        "capability_source_revision": report.capability_source_revision,
        "evidence_runner_revision": report.evidence_runner_revision,
        "issuer_id": identity.issuer_id,
        "key_id": identity.key_id,
    }
    signature = identity.private_key.sign(_canonical_json(unsigned).encode("utf-8")).hex()
    return VerificationReceipt(**unsigned, signature=signature)


@dataclass(frozen=True, slots=True)
class VerifierReceiptIdentity:
    """Distinct verifier-only signing role, never an execution identity."""
    issuer_id: str
    key_id: str
    private_key: Ed25519PrivateKey
    public_key: str

    @classmethod
    def from_configuration(cls, config: VerifierConfiguration) -> "VerifierReceiptIdentity":
        raw = _read_owner_only_file(config.receipt_key_file, kind="verifier receipt key").decode("ascii").strip()
        try:
            private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(raw))
        except ValueError as error:
            raise ReleaseEvidenceError("verifier receipt key is invalid") from error
        public = private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ).hex()
        if len(raw) != 64 or public != config.receipt_public_key:
            raise ReleaseEvidenceError("verifier receipt key does not match configured public key")
        return cls(config.receipt_issuer_id, config.receipt_key_id, private, config.receipt_public_key)


def write_verification_receipt(receipt: VerificationReceipt, output: Path) -> None:
    if not output.is_absolute() or not output.parent.is_dir():
        raise ReleaseEvidenceError("verification receipt output must have an existing absolute parent")
    descriptor = -1
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            descriptor = -1
            destination.write(_canonical_json(receipt.to_mapping()) + "\n")
    except OSError as error:
        raise ReleaseEvidenceError("verification receipt output cannot be securely created") from error
    finally:
        if descriptor != -1:
            os.close(descriptor)


def write_verified_evidence(evidence: SemanticReleaseEvidence, output: Path) -> None:
    """Write ready verifier evidence once, without following/replacing a path."""
    if not output.is_absolute() or not output.parent.is_dir():
        raise ReleaseEvidenceError("verified evidence output must have an existing absolute parent")
    descriptor = -1
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            descriptor = -1
            destination.write(_canonical_json(evidence.to_mapping()) + "\n")
    except OSError as error:
        raise ReleaseEvidenceError("verified evidence output cannot be securely created") from error
    finally:
        if descriptor != -1:
            os.close(descriptor)


__all__ = [
    "VERIFICATION_RECEIPT_VERSION",
    "VerificationReceipt",
    "issue_verification_receipt",
    "load_budgets",
    "load_external_report",
    "load_records",
    "read_trusted_execution_policy",
    "trusted_assemble",
    "write_verification_receipt",
    "write_verified_evidence",
    "verification_receipt_from_mapping",
    "verify_verification_receipt",
]
