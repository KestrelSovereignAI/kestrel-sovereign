"""Verifier-owned assembly and signed receipts for semantic release evidence.

This module is deliberately separate from the public structural assembler.
It accepts only independently produced, signed records, a protected public-key
policy, and a verifier-owned freshness ledger; it never accepts observations,
samples, signer keys, or a producer-supplied replay receipt.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
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
    _EXTERNAL_CAPABILITY_GATE_IDS,
    SemanticReleaseEvidence,
    apply_evidence_records,
    apply_performance_budgets,
    evidence_record_from_mapping,
    external_capability_report_from_mapping,
    performance_budget_from_mapping,
    release_gate_specs,
    release_evidence_template,
    trusted_execution_policy_from_mapping,
    validate_external_capability_attachment,
)
from .release_evidence_freshness import ExternalFreshnessLedger
from .release_evidence_models import (
    EvidenceRecord,
    EvidenceState,
    ExecutionSource,
    ExternalCapabilityReport,
    GateSpec,
    PerformanceBudget,
    ReleaseEvidenceError,
    TrustedExecutionPolicy,
    _canonical_json,
    _sha256,
)


VERIFICATION_RECEIPT_VERSION = "semantic-release-verification-receipt-v1"
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,127}$")
_EXTERNAL_ENVELOPE_TRUST_STATUS = "external_signature_requires_core_policy_verification"


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


def _validate_private_components(root: Path, path: Path, *, kind: str) -> tuple[tuple[int, int], ...]:
    """Return a stable private component snapshot for a rooted verifier path."""
    if not root.is_absolute() or not path.is_absolute() or ".." in path.parts:
        raise ReleaseEvidenceError(f"{kind} must be an absolute non-traversing verifier path")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ReleaseEvidenceError(f"{kind} escapes verifier trusted_root") from error
    try:
        if root.resolve(strict=True) != root:
            raise ReleaseEvidenceError(f"{kind} trusted_root must be a resolved non-symlink directory")
    except OSError as error:
        raise ReleaseEvidenceError(f"{kind} trusted_root cannot be inspected") from error
    current = root
    snapshots: list[tuple[int, int]] = []
    for component in (Path("."), *relative.parent.parts):
        if component != Path("."):
            current = current / component
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ReleaseEvidenceError(f"{kind} parent cannot be inspected") from error
        if (not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077):
            raise ReleaseEvidenceError(f"{kind} parent must be a private non-symlink directory")
        snapshots.append((metadata.st_dev, metadata.st_ino))
    return tuple(snapshots)


def _recheck_private_components(
    root: Path, path: Path, snapshot: tuple[tuple[int, int], ...], *, kind: str,
) -> None:
    """Reject a replacement or intermediate symlink after a sensitive action."""
    if _validate_private_components(root, path, kind=kind) != snapshot:
        raise ReleaseEvidenceError(f"{kind} parent changed while being used")


def _read_rooted_owner_only_file(root: Path, path: Path, *, kind: str) -> bytes:
    """Read a private rooted file while detecting ancestor replacement races."""
    snapshot = _validate_private_components(root, path, kind=kind)
    result = _read_owner_only_file(path, kind=kind)
    _recheck_private_components(root, path, snapshot, kind=kind)
    return result


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
    verifier_role: str


def read_verifier_configuration(path: Path) -> VerifierConfiguration:
    """Load one protected, root-pinned verifier configuration."""
    try:
        raw = _read_owner_only_file(path, kind="verifier configuration")
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseEvidenceError("verifier configuration is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ReleaseEvidenceError("verifier configuration must be a mapping")
    expected = {"trusted_root", "ledger_path", "trust_policy", "expected_external_runner_revision", "receipt_key_file", "receipt_issuer_id", "receipt_key_id", "receipt_public_key", "verifier_role"}
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
    # Re-read only after the configuration's declared root has itself passed
    # the private-component checks.  This prevents a safe leaf file beneath a
    # replaced/symlinked intermediate directory from selecting its own root.
    raw = _read_rooted_owner_only_file(resolved_root, path, kind="verifier configuration")
    try:
        if json.loads(raw.decode("utf-8")) != value:
            raise ReleaseEvidenceError("verifier configuration changed while being read")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseEvidenceError("verifier configuration is not valid JSON") from error
    _validate_private_components(resolved_root, receipt_key_file, kind="verifier receipt key")
    policy_mapping = value["trust_policy"]
    if not isinstance(policy_mapping, Mapping):
        raise ReleaseEvidenceError("verifier configuration trust_policy must be a mapping")
    revision = value["expected_external_runner_revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ReleaseEvidenceError("verifier configuration runner revision must be a full lowercase commit SHA")
    issuer_id, key_id, public_key = (value[field] for field in ("receipt_issuer_id", "receipt_key_id", "receipt_public_key"))
    if not all(isinstance(item, str) for item in (issuer_id, key_id, public_key)) or not _IDENTIFIER_RE.fullmatch(issuer_id) or not _IDENTIFIER_RE.fullmatch(key_id) or not re.fullmatch(r"[0-9a-f]{64}", public_key):
        raise ReleaseEvidenceError("verifier receipt identity is invalid")
    if value["verifier_role"] != "semantic_release_verifier":
        raise ReleaseEvidenceError("verifier configuration requires the semantic_release_verifier role")
    policy = trusted_execution_policy_from_mapping(policy_mapping)
    if (
        (issuer_id, key_id) in {(key.issuer_id, key.key_id) for key in policy.keys}
        or public_key in {key.public_key for key in policy.keys}
    ):
        raise ReleaseEvidenceError(
            "verifier receipt identity and public key must be distinct from execution identities"
        )
    return VerifierConfiguration(resolved_root, ledger_path, policy, _sha256(_canonical_json(policy_mapping)), revision, receipt_key_file, issuer_id, key_id, public_key, value["verifier_role"])


def _read_submission(path: Path, *, kind: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseEvidenceError(f"{kind} cannot be read") from error
    if not isinstance(value, Mapping):
        raise ReleaseEvidenceError(f"{kind} must be a mapping")
    return value


def _external_adapter_specs() -> tuple[GateSpec, ...]:
    """Return the catalog-declared external gates in their signed order."""
    specs = tuple(
        spec for spec in release_gate_specs() if spec.category == "external_adapter"
    )
    if not specs:
        raise ReleaseEvidenceError("semantic release catalog has no external adapter gates")
    return specs


@dataclass(frozen=True, slots=True)
class ExternalEvidenceEnvelope:
    """Exact parametric-self external-CI envelope accepted by the verifier.

    The producer owns this transport shape, but core repeats its strict
    validation at the trust boundary rather than accepting a producer library
    or a loosely related collection of JSON files.  Signature verification
    remains policy-owned and occurs when these records are applied.
    """

    core_release_evidence_contract_digest: str
    repository: str
    capability_source_revision: str
    evidence_runner_revision: str
    run_nonce: str
    trust_status: str
    records: tuple[EvidenceRecord, ...]
    report: ExternalCapabilityReport

    def __post_init__(self) -> None:
        if (
            self.core_release_evidence_contract_digest
            != CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST
        ):
            raise ReleaseEvidenceError(
                "external envelope does not bind the current core release contract"
            )
        if self.trust_status != _EXTERNAL_ENVELOPE_TRUST_STATUS:
            raise ReleaseEvidenceError("external envelope has an invalid trust_status")
        specs = _external_adapter_specs()
        expected_gate_ids = tuple(spec.gate_id for spec in specs)
        if tuple(record.gate_id for record in self.records) != expected_gate_ids:
            raise ReleaseEvidenceError(
                "external envelope records must preserve the declared external gate order"
            )
        for spec, record in zip(specs, self.records, strict=True):
            spec.validate_attestation(record)
            if (
                record.state is not EvidenceState.PASSED
                or record.execution_attestation is None
                or record.execution_attestation.source is not ExecutionSource.EXTERNAL_CI
                or record.external_run_nonce != self.run_nonce
                or record.external_evidence_runner_revision != self.evidence_runner_revision
            ):
                raise ReleaseEvidenceError(
                    "external envelope records must be externally signed passes bound to its nonce and runner revision"
                )
        if (
            self.report.capability_id != "parametric_self_governed_corpus"
            or self.report.repository != self.repository
            or self.report.capability_source_revision != self.capability_source_revision
            or self.report.evidence_runner_revision != self.evidence_runner_revision
            or self.report.core_release_evidence_contract_digest
            != self.core_release_evidence_contract_digest
            or self.report.run_nonce != self.run_nonce
            or self.report.gate_ids != _EXTERNAL_CAPABILITY_GATE_IDS
        ):
            raise ReleaseEvidenceError(
                "external envelope report identity or capability gate order does not match its envelope"
            )
        capability_specs = tuple(
            spec for spec in specs if spec.gate_id in _EXTERNAL_CAPABILITY_GATE_IDS
        )
        capability_records = tuple(
            record
            for record in self.records
            if record.gate_id in _EXTERNAL_CAPABILITY_GATE_IDS
        )
        for spec, record, attestation in zip(
            capability_specs,
            capability_records,
            self.report.attestations,
            strict=True,
        ):
            if (
                attestation.gate_spec_digest != spec.digest
                or attestation.result_digest != record.run_digest
                or attestation.artifact != record.artifact
                or attestation.drill != spec.correlation
            ):
                raise ReleaseEvidenceError(
                    "external envelope report is not bound to each signed external record"
                )


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


def load_external_envelope(path: Path) -> ExternalEvidenceEnvelope:
    """Parse only the exact producer envelope schema at the verifier boundary."""
    value = _read_submission(path, kind="external evidence envelope")
    expected = {
        "core_release_evidence_contract_digest",
        "repository",
        "capability_source_revision",
        "evidence_runner_revision",
        "run_nonce",
        "trust_status",
        "records",
        "report",
    }
    if set(value) != expected:
        raise ReleaseEvidenceError("external evidence envelope has unknown or missing fields")
    raw_records = value["records"]
    if not isinstance(raw_records, list):
        raise ReleaseEvidenceError("external evidence envelope records must be a list")
    if not isinstance(value["report"], Mapping):
        raise ReleaseEvidenceError("external evidence envelope report must be a mapping")
    records: list[EvidenceRecord] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ReleaseEvidenceError("external evidence envelope record must be a mapping")
        records.append(evidence_record_from_mapping(raw_record))
    return ExternalEvidenceEnvelope(
        core_release_evidence_contract_digest=value["core_release_evidence_contract_digest"],
        repository=value["repository"],
        capability_source_revision=value["capability_source_revision"],
        evidence_runner_revision=value["evidence_runner_revision"],
        run_nonce=value["run_nonce"],
        trust_status=value["trust_status"],
        records=tuple(records),
        report=external_capability_report_from_mapping(value["report"]),
    )


def combine_external_envelope_submission(
    *,
    records: Iterable[EvidenceRecord],
    envelope: ExternalEvidenceEnvelope | None,
) -> tuple[tuple[EvidenceRecord, ...], ExternalCapabilityReport]:
    """Require an atomic external envelope beside individually supplied core work.

    Core records and budgets remain individual verifier inputs. The external
    adapter portion is atomic: accepting standalone external records or a
    split report would make duplicate or substituted served-adapter evidence
    ambiguous before the verifier's freshness preflight.
    """
    standalone_records = tuple(records)
    if envelope is None:
        raise ReleaseEvidenceError(
            "verifier assembly requires exactly one --external-envelope"
        )
    external_gate_ids = {spec.gate_id for spec in _external_adapter_specs()}
    duplicate_gate_ids = sorted(
        {record.gate_id for record in standalone_records} & external_gate_ids
    )
    if duplicate_gate_ids:
        raise ReleaseEvidenceError(
            "--external-envelope cannot be combined with standalone external records: "
            + ", ".join(duplicate_gate_ids)
        )
    return (*standalone_records, *envelope.records), envelope.report


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
        if (not _IDENTIFIER_RE.fullmatch(self.issuer_id)
                or not _IDENTIFIER_RE.fullmatch(self.key_id)
                or not re.fullmatch(r"[0-9a-f]{128}", self.signature)
                or not re.fullmatch(r"[0-9a-f]{40}", self.capability_source_revision)
                or not re.fullmatch(r"[0-9a-f]{40}", self.evidence_runner_revision)):
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
    if not all(isinstance(value[field], str) for field in expected):
        raise ReleaseEvidenceError("verification receipt fields must be strings")
    return VerificationReceipt(**{field: value[field] for field in expected if field != "core_release_evidence_contract_digest"})


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
    evidence = prepare_trusted_evidence(
        records=records, budgets=budgets, report=report, trust_policy=trust_policy,
        expected_evidence_runner_revision=expected_evidence_runner_revision,
    )
    freshness_ledger.consume(report)
    return evidence


def prepare_trusted_evidence(
    *,
    records: Iterable[EvidenceRecord],
    budgets: Iterable[PerformanceBudget],
    report: ExternalCapabilityReport,
    trust_policy: TrustedExecutionPolicy,
    expected_evidence_runner_revision: str,
) -> SemanticReleaseEvidence:
    """Prepare fully validated evidence without consuming a freshness nonce.

    This narrow verifier-only split supports crash-safe output staging.  It is
    not exported by the public structural assembly CLI and does not confer
    readiness on an untrusted caller: nonce finalization remains ledger-owned.
    """
    evidence = apply_evidence_records(
        release_evidence_template(), records, trust_policy=trust_policy
    )
    evidence = apply_performance_budgets(evidence, budgets, trust_policy=trust_policy)
    if set(evidence.blocking_gate_ids()) != {"external_adapter_attestation"}:
        raise ReleaseEvidenceError(
            "trusted assembly requires all signed core records and performance budgets before external ingestion"
        )
    evidence = validate_external_capability_attachment(
        evidence, report,
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
        "core_release_evidence_contract_digest": CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
        "run_nonce": report.run_nonce,
        "capability_source_revision": report.capability_source_revision,
        "evidence_runner_revision": report.evidence_runner_revision,
        "issuer_id": identity.issuer_id,
        "key_id": identity.key_id,
    }
    signature = identity.private_key.sign(_canonical_json(unsigned).encode("utf-8")).hex()
    # ``VerificationReceipt`` keeps the contract digest in the canonical
    # signed payload rather than as a caller-controlled constructor field.
    unsigned.pop("core_release_evidence_contract_digest")
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
        raw = _read_rooted_owner_only_file(
            config.trusted_root, config.receipt_key_file, kind="verifier receipt key"
        ).decode("ascii").strip()
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


def _expected_output(path: Path, payload: bytes, *, trusted_root: Path, kind: str) -> Path:
    if not path.is_absolute() or not path.parent.is_dir() or path == trusted_root:
        raise ReleaseEvidenceError(f"{kind} output must have an existing absolute private parent")
    _validate_private_components(trusted_root, path, kind=f"{kind} output")
    if path.exists() or path.is_symlink():
        existing = _read_rooted_owner_only_file(trusted_root, path, kind=f"{kind} output")
        if existing != payload:
            raise ReleaseEvidenceError(f"{kind} output already exists with different content")
    return path.with_name(f".{path.name}.semantic-release-pending")


def _stage_output(path: Path, payload: bytes, *, trusted_root: Path, kind: str) -> Path:
    staged = _expected_output(path, payload, trusted_root=trusted_root, kind=kind)
    snapshot = _validate_private_components(trusted_root, staged, kind=f"{kind} staged output")
    if staged.exists() or staged.is_symlink():
        if _read_rooted_owner_only_file(trusted_root, staged, kind=f"{kind} staged output") != payload:
            raise ReleaseEvidenceError(f"{kind} staged output already exists with different content")
        return staged
    descriptor = -1
    try:
        descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    except OSError as error:
        raise ReleaseEvidenceError(f"{kind} output cannot be securely staged") from error
    finally:
        if descriptor != -1:
            os.close(descriptor)
    _recheck_private_components(trusted_root, staged, snapshot, kind=f"{kind} staged output")
    if _read_rooted_owner_only_file(trusted_root, staged, kind=f"{kind} staged output") != payload:
        raise ReleaseEvidenceError(f"{kind} staged output changed while being written")
    return staged


def _promote_staged_output(path: Path, staged: Path, payload: bytes, *, trusted_root: Path, kind: str) -> None:
    if path.exists() or path.is_symlink():
        if _read_rooted_owner_only_file(trusted_root, path, kind=f"{kind} output") != payload:
            raise ReleaseEvidenceError(f"{kind} output already exists with different content")
        return
    snapshot = _validate_private_components(trusted_root, path, kind=f"{kind} output")
    if _read_rooted_owner_only_file(trusted_root, staged, kind=f"{kind} staged output") != payload:
        raise ReleaseEvidenceError(f"{kind} staged output changed before finalization")
    try:
        os.replace(staged, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise ReleaseEvidenceError(f"{kind} staged output cannot be finalized; retry is safe") from error
    _recheck_private_components(trusted_root, path, snapshot, kind=f"{kind} output")
    if _read_rooted_owner_only_file(trusted_root, path, kind=f"{kind} output") != payload:
        raise ReleaseEvidenceError(f"{kind} output changed while being finalized")


def finalize_verified_artifacts(
    evidence: SemanticReleaseEvidence,
    receipt: VerificationReceipt,
    *,
    evidence_output: Path,
    receipt_output: Path,
    configuration: VerifierConfiguration,
    freshness_ledger: ExternalFreshnessLedger,
) -> None:
    """Stage artifacts, atomically bind the nonce, then recoverably promote them."""
    if evidence_output == receipt_output:
        raise ReleaseEvidenceError("verified evidence and receipt outputs must be distinct")
    if not evidence.ready or len(evidence.external_capabilities) != 1:
        raise ReleaseEvidenceError("verified artifact finalization requires ready trusted evidence")
    if (
        freshness_ledger.trusted_root != configuration.trusted_root
        or freshness_ledger.path != configuration.ledger_path
    ):
        raise ReleaseEvidenceError("verified artifact finalization requires the configured freshness ledger")
    verify_verification_receipt(receipt, configuration)
    report = evidence.external_capabilities[0]
    evidence_digest = _sha256(_canonical_json(evidence.to_mapping()))
    if (
        receipt.evidence_digest != evidence_digest
        or receipt.policy_digest != configuration.policy_digest
        or receipt.external_freshness_receipt != report.freshness_receipt
        or receipt.run_nonce != report.run_nonce
        or receipt.capability_source_revision != report.capability_source_revision
        or receipt.evidence_runner_revision != report.evidence_runner_revision
        or receipt.evidence_runner_revision != configuration.expected_external_runner_revision
    ):
        raise ReleaseEvidenceError(
            "verification receipt is not bound to the exact evidence, policy, and external report"
        )
    trusted_root = configuration.trusted_root
    evidence_payload = (_canonical_json(evidence.to_mapping()) + "\n").encode("utf-8")
    receipt_payload = (_canonical_json(receipt.to_mapping()) + "\n").encode("utf-8")
    staged_evidence = _stage_output(evidence_output, evidence_payload, trusted_root=trusted_root, kind="verified evidence")
    staged_receipt = _stage_output(receipt_output, receipt_payload, trusted_root=trusted_root, kind="verification receipt")
    finalization_digest = _sha256(_canonical_json({
        "evidence_payload_digest": hashlib.sha256(evidence_payload).hexdigest(),
        "evidence_output": str(evidence_output.relative_to(trusted_root)),
        "receipt_payload_digest": hashlib.sha256(receipt_payload).hexdigest(),
        "receipt_output": str(receipt_output.relative_to(trusted_root)),
    }))
    freshness_ledger.finalize_verified_receipt(
        report, evidence_digest=receipt.evidence_digest, policy_digest=receipt.policy_digest,
        receipt_digest=receipt.digest, finalization_digest=finalization_digest,
    )
    _promote_staged_output(evidence_output, staged_evidence, evidence_payload, trusted_root=trusted_root, kind="verified evidence")
    _promote_staged_output(receipt_output, staged_receipt, receipt_payload, trusted_root=trusted_root, kind="verification receipt")


__all__ = [
    "VERIFICATION_RECEIPT_VERSION",
    "ExternalEvidenceEnvelope",
    "VerificationReceipt",
    "combine_external_envelope_submission",
    "finalize_verified_artifacts",
    "issue_verification_receipt",
    "load_budgets",
    "load_external_report",
    "load_external_envelope",
    "load_records",
    "prepare_trusted_evidence",
    "read_verifier_configuration",
    "trusted_assemble",
    "verification_receipt_from_mapping",
    "verify_verification_receipt",
]
