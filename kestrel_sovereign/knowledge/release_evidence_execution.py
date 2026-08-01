"""Trusted execution boundary for semantic release evidence.

The release-evidence artifact is intentionally not a generic command-result
format.  This module is the one place that can turn a catalog workload into a
passed or failed record: it resolves an immutable ``(runner_id, command_id)``
pair, invokes the registered workload without caller argv or observations,
derives an opaque artifact locator, and signs the resulting digest with an
Ed25519 execution identity.

An unavailable workload is represented as a content-free blocked record.  It
cannot be promoted to a pass by supplying JSON to the CLI.  External consumer
workloads remain an import path: their independently signed records are
verified by :class:`TrustedExecutionPolicy` during assembly instead.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import stat

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .release_evidence_models import (
    ArtifactReference,
    EvidenceRecord,
    EvidenceState,
    ExecutionAttestation,
    ExecutionSource,
    GateResult,
    GateSpec,
    PerformanceBudget,
    ReleaseEvidenceError,
    TrustedExecutionKey,
    TrustedExecutionPolicy,
    _canonical_json,
    _sha256,
    execution_attestation_payload,
)


CatalogWorkload = Callable[[GateSpec], "CatalogWorkloadResult | Awaitable[CatalogWorkloadResult]"]


class CatalogWorkloadUnavailable(ReleaseEvidenceError):
    """A required immutable workload cannot run in this isolated environment."""

    def __init__(self, reason_code: str) -> None:
        if not isinstance(reason_code, str) or not reason_code:
            raise ReleaseEvidenceError("catalog workload block requires a reason code")
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class CatalogWorkloadResult:
    """Content-free measurements returned by one actual catalog workload."""

    observation: Mapping[str, object]
    state: EvidenceState = EvidenceState.PASSED
    reason_code: str | None = None
    artifact_digest: str | None = None
    samples: tuple[float | int, ...] | None = None
    headroom_fraction: float = 0.20

    def __post_init__(self) -> None:
        if self.state not in {EvidenceState.PASSED, EvidenceState.FAILED}:
            raise ReleaseEvidenceError("catalog workload results must be passed or failed")
        if self.state is EvidenceState.PASSED and self.reason_code is not None:
            raise ReleaseEvidenceError("passed catalog workload result cannot have a reason_code")
        if self.state is EvidenceState.FAILED and self.reason_code is None:
            raise ReleaseEvidenceError("failed catalog workload result requires a reason_code")
        if self.artifact_digest is not None:
            # Reuse the public artifact validator so the only supplied value
            # is still a digest, not a location or arbitrary artifact text.
            ArtifactReference(f"ci://sha256/{self.artifact_digest}", self.artifact_digest)


@dataclass(frozen=True, slots=True)
class CatalogExecution:
    """One signed record and, for benchmarks, its signed measured budget."""

    record: EvidenceRecord
    budget: PerformanceBudget | None = None


@dataclass(frozen=True, slots=True)
class CatalogSigningIdentity:
    """Private execution identity held by CI/host, never stored in evidence."""

    issuer_id: str
    key_id: str
    private_key: Ed25519PrivateKey
    source: ExecutionSource = ExecutionSource.CATALOG_RUNNER

    def __post_init__(self) -> None:
        if not isinstance(self.private_key, Ed25519PrivateKey):
            raise ReleaseEvidenceError("catalog signing identity requires an Ed25519 private key")
        # Validate identifier/source shapes through the public-key model while
        # deliberately avoiding serializing the private key.
        TrustedExecutionKey(
            self.issuer_id,
            self.key_id,
            self.source,
            self.public_key,
            ("external_ci",)
            if self.source is ExecutionSource.EXTERNAL_CI
            else ("catalog_runner",),
        )

    @classmethod
    def from_private_key_file(
        cls,
        path: Path,
        *,
        issuer_id: str,
        key_id: str,
        source: ExecutionSource = ExecutionSource.CATALOG_RUNNER,
    ) -> "CatalogSigningIdentity":
        """Load a raw 32-byte lowercase-hex Ed25519 key from a protected file."""
        try:
            material = _read_protected_private_key_file(path).decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ReleaseEvidenceError("signing key file must contain raw lowercase-hex Ed25519 bytes") from error
        try:
            private_bytes = bytes.fromhex(material)
        except ValueError as error:
            raise ReleaseEvidenceError("signing key file must contain raw lowercase-hex Ed25519 bytes") from error
        if len(private_bytes) != 32 or material != private_bytes.hex():
            raise ReleaseEvidenceError("signing key file must contain 32 lowercase-hex Ed25519 bytes")
        return cls(
            issuer_id=issuer_id,
            key_id=key_id,
            private_key=Ed25519PrivateKey.from_private_bytes(private_bytes),
            source=source,
        )
    @property
    def public_key(self) -> str:
        return self.private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()

    def trusted_key(self, runner_ids: tuple[str, ...]) -> TrustedExecutionKey:
        """Return the public half for an independently configured verifier."""
        return TrustedExecutionKey(
            self.issuer_id,
            self.key_id,
            self.source,
            self.public_key,
            runner_ids,
        )

    def sign(
        self,
        *,
        kind: str,
        spec: GateSpec,
        run_digest: str,
    ) -> ExecutionAttestation:
        payload = execution_attestation_payload(
            kind=kind,
            issuer_id=self.issuer_id,
            key_id=self.key_id,
            source=self.source,
            gate_id=spec.gate_id,
            gate_spec_digest=spec.digest,
            runner_id=spec.runner.runner_id,
            run_digest=run_digest,
        )
        return ExecutionAttestation(
            issuer_id=self.issuer_id,
            key_id=self.key_id,
            source=self.source,
            signature=self.private_key.sign(payload).hex(),
        )


def _read_protected_private_key_file(path: Path) -> bytes:
    """Read a private key only from a regular, owner-only file.

    The private key is a CI/host secret.  Checking the path with ``lstat`` and
    the opened descriptor with ``fstat`` rejects symlinks and a replacement
    race between validation and read.
    """
    try:
        initial = path.lstat()
    except OSError as error:
        raise ReleaseEvidenceError("signing key file cannot be inspected") from error
    _validate_private_key_file_stat(initial)

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReleaseEvidenceError("signing key file cannot be securely opened") from error
    try:
        opened = os.fstat(descriptor)
        _validate_private_key_file_stat(opened)
        if (opened.st_dev, opened.st_ino) != (initial.st_dev, initial.st_ino):
            raise ReleaseEvidenceError("signing key file changed while being opened")
        with os.fdopen(descriptor, "rb") as key_file:
            descriptor = -1
            return key_file.read()
    except OSError as error:
        raise ReleaseEvidenceError("signing key file cannot be securely read") from error
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _validate_private_key_file_stat(file_stat: os.stat_result) -> None:
    if stat.S_ISLNK(file_stat.st_mode):
        raise ReleaseEvidenceError("signing key file must not be a symlink")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ReleaseEvidenceError("signing key file must be a regular file")
    if file_stat.st_uid != os.geteuid():
        raise ReleaseEvidenceError("signing key file must be owned by the effective user")
    if file_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ReleaseEvidenceError(
            "signing key file must not grant group or other permissions"
        )


class CatalogExecutionAuthority:
    """Execute only pre-registered immutable catalog workloads.

    ``workloads`` is keyed by the exact runner and command IDs already pinned
    in the catalog.  The public CLI supplies only a gate ID; no runtime argv,
    observation JSON, artifact reference, or sample sequence crosses this
    boundary.
    """

    def __init__(
        self,
        identity: CatalogSigningIdentity,
        workloads: Mapping[tuple[str, str], CatalogWorkload],
    ) -> None:
        if identity.source is not ExecutionSource.CATALOG_RUNNER:
            raise ReleaseEvidenceError("catalog execution authority requires a catalog_runner signing identity")
        invalid = [
            key
            for key, workload in workloads.items()
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not all(isinstance(value, str) and value for value in key)
                or not callable(workload)
            )
        ]
        if invalid:
            raise ReleaseEvidenceError("catalog workloads require immutable (runner_id, command_id) callables")
        self._identity = identity
        self._workloads = dict(workloads)

    async def execute(self, spec: GateSpec) -> CatalogExecution:
        """Run one allowlisted workload or return a fail-closed block."""
        if not isinstance(spec, GateSpec):
            raise ReleaseEvidenceError("catalog execution requires a GateSpec")
        key = (spec.runner.runner_id, spec.runner.command_id)
        workload = self._workloads.get(key)
        if workload is None:
            return self._blocked(spec, "catalog_workload_unavailable")
        try:
            result = workload(spec)
            if isinstance(result, Awaitable):
                result = await result
            if not isinstance(result, CatalogWorkloadResult):
                raise ReleaseEvidenceError("catalog workload returned an invalid result")
            spec.observation_schema.validate(result.observation)
            return self._emit(spec, result)
        except CatalogWorkloadUnavailable as error:
            return self._blocked(spec, error.reason_code)
        except ReleaseEvidenceError:
            # This is an evidence failure, not an opportunity to construct a
            # placeholder pass.  Retain only a fixed reason code.
            return self._blocked(spec, "catalog_execution_failed")
        except Exception:
            return self._blocked(spec, "catalog_execution_failed")

    def _emit(self, spec: GateSpec, result: CatalogWorkloadResult) -> CatalogExecution:
        artifact = self._artifact_for(spec, result)
        _, run_digest = EvidenceRecord._bound_run_digest(
            spec,
            result.observation,
            artifact,
            state=result.state,
            reason_code=result.reason_code,
        )
        record = EvidenceRecord._from_trusted_execution(
            spec,
            result.observation,
            artifact,
            state=result.state,
            reason_code=result.reason_code,
            execution_attestation=self._identity.sign(
                kind="evidence_record", spec=spec, run_digest=run_digest
            ),
        )
        # Prove locally that the record can only be accepted via the public
        # key; report assembly must still receive its own policy.
        GateResult(
            spec,
            record,
            TrustedExecutionPolicy((self._identity.trusted_key((spec.runner.runner_id,)),)),
        )
        if spec.performance_target is None:
            return CatalogExecution(record=record)
        if result.state is not EvidenceState.PASSED or result.samples is None:
            raise ReleaseEvidenceError("performance workload requires successful measured samples")
        self._validate_performance_observation(spec, result)
        _, _, _, budget_digest = PerformanceBudget._bound_run_digest(
            spec,
            result.samples,
            headroom_fraction=result.headroom_fraction,
            artifact=artifact,
        )
        budget = PerformanceBudget._from_trusted_execution(
            spec,
            result.samples,
            headroom_fraction=result.headroom_fraction,
            artifact=artifact,
            execution_attestation=self._identity.sign(
                kind="performance_budget", spec=spec, run_digest=budget_digest
            ),
        )
        return CatalogExecution(record=record, budget=budget)

    def _artifact_for(self, spec: GateSpec, result: CatalogWorkloadResult) -> ArtifactReference:
        digest = result.artifact_digest
        if digest is None:
            # This is a digest of the executed, content-free measurement
            # envelope; it cannot encode test stdout or subject data.
            digest = _sha256(
                _canonical_json(
                    {
                        "gate_id": spec.gate_id,
                        "gate_spec_digest": spec.digest,
                        "state": result.state.value,
                        "observation": dict(result.observation),
                        "samples": list(result.samples) if result.samples is not None else None,
                    }
                )
            )
        return ArtifactReference(f"ci://sha256/{digest}", digest)

    @staticmethod
    def _validate_performance_observation(spec: GateSpec, result: CatalogWorkloadResult) -> None:
        assert spec.performance_target is not None
        assert result.samples is not None
        values, p95, _, _ = PerformanceBudget._bound_run_digest(
            spec,
            result.samples,
            headroom_fraction=result.headroom_fraction,
            artifact=ArtifactReference("ci://sha256/" + "0" * 64, "0" * 64),
        )
        p95_field = "p95_bytes" if spec.performance_target.unit == "bytes" else "p95_ms"
        if result.observation.get("sample_count") != len(values) or result.observation.get(p95_field) != p95:
            raise ReleaseEvidenceError("performance observation must match measured sample_count and p95")

    @staticmethod
    def _blocked(spec: GateSpec, reason_code: str) -> CatalogExecution:
        return CatalogExecution(
            record=EvidenceRecord(
                gate_id=spec.gate_id,
                state=EvidenceState.BLOCKED,
                reason_code=reason_code,
            )
        )


def default_catalog_workloads() -> dict[tuple[str, str], CatalogWorkload]:
    """Return built-in workloads that are executable in this package today.

    Other command IDs are deliberately absent until their feature-specific
    pytest, benchmark, or Kite HTTP workload is registered in production.
    ``CatalogExecutionAuthority`` then emits a blocked record instead of
    pretending that a catalog label executed work.
    """

    def stable_only_capability_selection(_spec: GateSpec) -> CatalogWorkloadResult:
        # Import lazily to avoid a module cycle with the catalog definition.
        from .release_evidence import inspect_stable_only_capabilities

        return CatalogWorkloadResult(observation=inspect_stable_only_capabilities())

    workloads: dict[tuple[str, str], CatalogWorkload] = {
        ("registry", "stable_only_capability_selection_v1"): stable_only_capability_selection,
    }
    # Keep the resolver's public boundary here: the test runner owns no caller
    # argv and exposes only reviewed, immutable catalog command IDs.
    from .release_evidence_workloads import pytest_catalog_workloads
    from .release_evidence_benchmarks import semantic_benchmark_workloads
    from .kite_release_evidence_workloads import owned_kite_http_workloads

    workloads.update(pytest_catalog_workloads())
    workloads.update(semantic_benchmark_workloads())
    workloads.update(owned_kite_http_workloads())
    return workloads
