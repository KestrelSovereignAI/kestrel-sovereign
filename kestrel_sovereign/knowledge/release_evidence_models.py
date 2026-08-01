"""Immutable, content-free attestations for semantic release evidence.

Release evidence is deliberately *not* a generic command-result log.  A
successful process with an arbitrary artifact URL says nothing about a
semantic contract.  The models in this module bind every passing attestation
to one immutable :class:`GateSpec`: runner identity, command-pattern digest,
execution environment, fixture identity/digest, and an exact observation
schema.  Only opaque, content-free artifact references are retained.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import inspect
import json
import math
import re
import time
from types import MappingProxyType

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .registry import StandardsMaturity


_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
# Artifact locations are identifiers, not URLs.  A name such as
# ``patient-alice-hiv`` may be safe as URI syntax while still disclosing
# sensitive semantics, so retain only an opaque digest locator.
_SAFE_ARTIFACT_RE = re.compile(r"^(?:ci|artifact|evidence)://sha256/[0-9a-f]{64}$")
_ED25519_PUBLIC_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_ED25519_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")
_EXECUTION_ATTESTATION_VERSION = "semantic-release-execution-attestation-v1"
_SAFE_OBSERVATION_KINDS = frozenset(
    {
        "positive_count",
        "nonnegative_count",
        "zero_count",
        "sample_count",
        "positive_duration_ms",
        "positive_bytes",
        "boolean",
        "digest",
    }
)
_ARTIFACT_FORBIDDEN_TOKENS = frozenset(
    {
        "apikey",
        "credential",
        "dsn",
        "password",
        "secret",
        "tenant",
        "token",
        "user",
    }
)


class EvidenceState(str, Enum):
    """One observed state for a declared release gate."""

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_RUN = "not_run"
    SKIPPED = "skipped"


class ExecutionSource(str, Enum):
    """The independently verifiable source that produced a result."""

    CATALOG_RUNNER = "catalog_runner"
    EXTERNAL_CI = "external_ci"


class PerformanceMetric(str, Enum):
    """Required semantic workload classes; these are not timeouts."""

    STARTUP = "startup"
    ASSERTION_WRITE_VALIDATION = "assertion_write_validation"
    BOUNDED_INFERENCE = "bounded_inference"
    HYBRID_RECALL = "hybrid_recall"
    CHANGED_WORK_SLEEP = "changed_work_sleep"
    UNCHANGED_SLEEP = "unchanged_sleep"
    STORAGE_GROWTH = "storage_growth"
    REPRESENTATIVE_MIGRATION = "representative_migration"


class ErasureStage(str, Enum):
    """Every canonical, projected, corpus, and serving erasure surface."""

    ACTIVE_ASSERTIONS = "active_assertions"
    DERIVATIONS = "derivations"
    VECTOR_INDEX = "vector_index"
    RECALL_CANDIDATES = "recall_candidates"
    EXPORT_SNAPSHOTS = "export_snapshots"
    GOVERNED_CORPUS = "governed_corpus"
    FUTURE_CORPUS = "future_corpus"
    PROJECTION_CANDIDATES = "projection_candidates"
    SERVED_ADAPTER_ELIGIBILITY = "served_adapter_eligibility"


class ReleaseEvidenceError(ValueError):
    """Raised when an artifact would overstate unobserved release evidence."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_identifier(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ReleaseEvidenceError(
            f"{field_name} must be a lowercase content-free identifier, got {value!r}"
        )


def _require_digest(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ReleaseEvidenceError(f"{field_name} must be a lowercase sha256 digest")


def _require_commit(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise ReleaseEvidenceError(f"{field_name} must be a commit SHA")


def _safe_gate_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _safe_observation_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Copy only content-free scalar observations into immutable storage."""
    if not isinstance(value, Mapping) or not value:
        raise ReleaseEvidenceError("measured observation must be a non-empty mapping")
    copied: dict[str, object] = {}
    for key, item in value.items():
        _require_identifier(key, "observation field")
        if type(item) is bool:
            copied[key] = item
        elif isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            copied[key] = item
        elif isinstance(item, float) and math.isfinite(item) and item >= 0:
            copied[key] = item
        elif isinstance(item, str) and _SHA256_RE.fullmatch(item):
            copied[key] = item
        else:
            raise ReleaseEvidenceError(
                "measured observation values must be non-negative numbers, booleans, or digests"
            )
    return MappingProxyType(dict(sorted(copied.items())))


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Opaque content-free evidence location plus its content digest.

    A release artifact must not persist a raw command line, DSN, tenant ID,
    query string, userinfo, or bearer token.  The reference is therefore a
    narrowly shaped opaque locator, not a general URL.
    """

    artifact_ref: str
    artifact_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_ref, str) or not _SAFE_ARTIFACT_RE.fullmatch(
            self.artifact_ref
        ):
            raise ReleaseEvidenceError("artifact reference must use a safe content-free scheme")
        # The strict opaque form above already excludes semantic path labels,
        # but retain an explicit traversal check so future format changes do
        # not accidentally reintroduce it.
        if ".." in self.artifact_ref:
            raise ReleaseEvidenceError("artifact reference cannot contain traversal segments")
        lowered = self.artifact_ref.lower()
        if any(token in lowered for token in _ARTIFACT_FORBIDDEN_TOKENS):
            raise ReleaseEvidenceError("artifact reference contains a forbidden secret or identity token")
        if any(character in self.artifact_ref for character in ("?", "#", "@", "%")):
            raise ReleaseEvidenceError("artifact reference cannot contain query, fragment, userinfo, or escapes")
        _require_digest(self.artifact_digest, "artifact_digest")
        if self.artifact_ref.rsplit("/", 1)[1] != self.artifact_digest:
            raise ReleaseEvidenceError(
                "artifact reference SHA-256 component must match artifact_digest"
            )

    def to_mapping(self) -> dict[str, str]:
        return {
            "artifact_ref": self.artifact_ref,
            "artifact_digest": self.artifact_digest,
        }


@dataclass(frozen=True, slots=True)
class ExecutionAttestation:
    """An Ed25519 proof that a trusted runner emitted a bound result.

    The signature covers the record/budget run digest plus its immutable gate
    and runner contract.  A report author cannot turn it into evidence: a
    separate operator-owned :class:`TrustedExecutionPolicy` must verify it.
    """

    issuer_id: str
    key_id: str
    source: ExecutionSource
    signature: str

    def __post_init__(self) -> None:
        _require_identifier(self.issuer_id, "execution attestation issuer_id")
        _require_identifier(self.key_id, "execution attestation key_id")
        if not isinstance(self.source, ExecutionSource):
            raise ReleaseEvidenceError("execution attestation source is unknown")
        if not isinstance(self.signature, str) or not _ED25519_SIGNATURE_RE.fullmatch(self.signature):
            raise ReleaseEvidenceError("execution attestation must contain an Ed25519 signature")

    def to_mapping(self) -> dict[str, str]:
        return {
            "issuer_id": self.issuer_id,
            "key_id": self.key_id,
            "source": self.source.value,
            "signature": self.signature,
        }


@dataclass(frozen=True, slots=True)
class TrustedExecutionKey:
    """One operator-configured public key and the runners it may attest."""

    issuer_id: str
    key_id: str
    source: ExecutionSource
    public_key: str
    runner_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.issuer_id, "trusted execution issuer_id")
        _require_identifier(self.key_id, "trusted execution key_id")
        if not isinstance(self.source, ExecutionSource):
            raise ReleaseEvidenceError("trusted execution source is unknown")
        if not isinstance(self.public_key, str) or not _ED25519_PUBLIC_KEY_RE.fullmatch(self.public_key):
            raise ReleaseEvidenceError("trusted execution public_key must be an Ed25519 public key")
        if not self.runner_ids or len(set(self.runner_ids)) != len(self.runner_ids):
            raise ReleaseEvidenceError("trusted execution runner_ids must be non-empty and unique")
        for runner_id in self.runner_ids:
            _require_identifier(runner_id, "trusted execution runner_id")
        if self.source is ExecutionSource.EXTERNAL_CI and set(self.runner_ids) != {"external_ci"}:
            raise ReleaseEvidenceError(
                "external_ci keys may attest only the declared external_ci catalog runner"
            )
        if self.source is ExecutionSource.CATALOG_RUNNER and "external_ci" in self.runner_ids:
            raise ReleaseEvidenceError(
                "catalog_runner keys cannot attest the independently external_ci runner"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "issuer_id": self.issuer_id,
            "key_id": self.key_id,
            "source": self.source.value,
            "public_key": self.public_key,
            "runner_ids": list(self.runner_ids),
        }


def execution_attestation_payload(
    *,
    kind: str,
    issuer_id: str,
    key_id: str,
    source: ExecutionSource,
    gate_id: str,
    gate_spec_digest: str,
    runner_id: str,
    run_digest: str,
) -> bytes:
    """Return the exact content-free bytes signed by an execution authority."""
    _require_identifier(kind, "execution attestation kind")
    _require_identifier(issuer_id, "execution attestation issuer_id")
    _require_identifier(key_id, "execution attestation key_id")
    if not isinstance(source, ExecutionSource):
        raise ReleaseEvidenceError("execution attestation source is unknown")
    _require_identifier(gate_id, "execution attestation gate_id")
    _require_digest(gate_spec_digest, "execution attestation gate_spec_digest")
    _require_identifier(runner_id, "execution attestation runner_id")
    _require_digest(run_digest, "execution attestation run_digest")
    return _canonical_json(
        {
            "version": _EXECUTION_ATTESTATION_VERSION,
            "kind": kind,
            "issuer_id": issuer_id,
            "key_id": key_id,
            "source": source.value,
            "gate_id": gate_id,
            "gate_spec_digest": gate_spec_digest,
            "runner_id": runner_id,
            "run_digest": run_digest,
        }
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class TrustedExecutionPolicy:
    """Fail-closed public-key allowlist for release evidence ingestion."""

    keys: tuple[TrustedExecutionKey, ...]

    def __post_init__(self) -> None:
        if not self.keys:
            raise ReleaseEvidenceError("trusted execution policy requires at least one public key")
        if any(not isinstance(key, TrustedExecutionKey) for key in self.keys):
            raise ReleaseEvidenceError("trusted execution policy contains an invalid public key")
        identities = {(key.issuer_id, key.key_id) for key in self.keys}
        if len(identities) != len(self.keys):
            raise ReleaseEvidenceError("trusted execution policy repeats an issuer/key identity")

    def _key_for(self, attestation: ExecutionAttestation, runner_id: str) -> TrustedExecutionKey:
        if not isinstance(attestation, ExecutionAttestation):
            raise ReleaseEvidenceError("release evidence requires an execution attestation")
        for key in self.keys:
            if key.issuer_id == attestation.issuer_id and key.key_id == attestation.key_id:
                if key.source is not attestation.source:
                    raise ReleaseEvidenceError("execution attestation source does not match trusted key")
                if runner_id not in key.runner_ids:
                    raise ReleaseEvidenceError("execution attestation runner is not allowed for trusted key")
                return key
        raise ReleaseEvidenceError("execution attestation issuer/key is not trusted")

    def _verify(
        self,
        *,
        kind: str,
        attestation: ExecutionAttestation,
        gate_id: str,
        gate_spec_digest: str,
        runner_id: str,
        run_digest: str,
    ) -> None:
        key = self._key_for(attestation, runner_id)
        payload = execution_attestation_payload(
            kind=kind,
            issuer_id=attestation.issuer_id,
            key_id=attestation.key_id,
            source=attestation.source,
            gate_id=gate_id,
            gate_spec_digest=gate_spec_digest,
            runner_id=runner_id,
            run_digest=run_digest,
        )
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(key.public_key)).verify(
                bytes.fromhex(attestation.signature), payload
            )
        except (InvalidSignature, ValueError) as error:
            raise ReleaseEvidenceError("execution attestation signature verification failed") from error

    def verify_evidence(self, spec: "GateSpec", evidence: "EvidenceRecord") -> None:
        if (
            evidence.execution_attestation is None
            or evidence.gate_spec_digest is None
            or evidence.run_digest is None
        ):
            raise ReleaseEvidenceError("release evidence requires a verified execution attestation")
        self._verify(
            kind="evidence_record",
            attestation=evidence.execution_attestation,
            gate_id=evidence.gate_id,
            gate_spec_digest=evidence.gate_spec_digest,
            runner_id=spec.runner.runner_id,
            run_digest=evidence.run_digest,
        )

    def verify_budget(self, spec: "GateSpec", budget: "PerformanceBudget") -> None:
        if budget.execution_attestation is None:
            raise ReleaseEvidenceError("performance budget requires a verified execution attestation")
        self._verify(
            kind="performance_budget",
            attestation=budget.execution_attestation,
            gate_id=budget.gate_id,
            gate_spec_digest=budget.gate_spec_digest,
            runner_id=spec.runner.runner_id,
            run_digest=budget.run_digest,
        )


@dataclass(frozen=True, slots=True)
class ExecutionEnvironment:
    """Exact backend/mode/profile contract for a release runner."""

    backend: str
    mode: str
    profile: str

    def __post_init__(self) -> None:
        _require_identifier(self.backend, "environment backend")
        _require_identifier(self.mode, "environment mode")
        _require_identifier(self.profile, "environment profile")

    @property
    def digest(self) -> str:
        return _sha256(_canonical_json(self.to_mapping()))

    def to_mapping(self) -> dict[str, str]:
        return {"backend": self.backend, "mode": self.mode, "profile": self.profile}


@dataclass(frozen=True, slots=True)
class FixtureBinding:
    """Exact fixture identifier and digest, without fixture content."""

    fixture_id: str
    fixture_digest: str

    def __post_init__(self) -> None:
        _require_identifier(self.fixture_id, "fixture_id")
        _require_digest(self.fixture_digest, "fixture_digest")

    def to_mapping(self) -> dict[str, str]:
        return {"fixture_id": self.fixture_id, "fixture_digest": self.fixture_digest}


@dataclass(frozen=True, slots=True)
class DrillBinding:
    """Opaque correlation identity shared by every stage of one erasure drill."""

    drill_id: str
    drill_digest: str

    def __post_init__(self) -> None:
        _require_identifier(self.drill_id, "drill_id")
        _require_digest(self.drill_digest, "drill_digest")

    def to_mapping(self) -> dict[str, str]:
        return {"drill_id": self.drill_id, "drill_digest": self.drill_digest}


@dataclass(frozen=True, slots=True)
class FixtureContract:
    """Fixture binding plus the expected harness identity."""

    binding: FixtureBinding
    harness_id: str
    official: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.binding, FixtureBinding):
            raise ReleaseEvidenceError("fixture contract requires FixtureBinding")
        _require_identifier(self.harness_id, "fixture harness_id")
        if type(self.official) is not bool:
            raise ReleaseEvidenceError("fixture official must be a boolean")

    def to_mapping(self) -> dict[str, object]:
        return {
            **self.binding.to_mapping(),
            "harness_id": self.harness_id,
            "official": self.official,
        }


@dataclass(frozen=True, slots=True)
class RunnerContract:
    """Runner identity and a digest of its predeclared command pattern."""

    runner_id: str
    command_id: str
    command_digest: str

    def __post_init__(self) -> None:
        _require_identifier(self.runner_id, "runner_id")
        _require_identifier(self.command_id, "command_id")
        _require_digest(self.command_digest, "command_digest")

    def to_mapping(self) -> dict[str, str]:
        return {
            "runner_id": self.runner_id,
            "command_id": self.command_id,
            "command_digest": self.command_digest,
        }


@dataclass(frozen=True, slots=True)
class ObservationField:
    """One safe scalar expected from a release runner."""

    field_id: str
    kind: str

    def __post_init__(self) -> None:
        _require_identifier(self.field_id, "observation field_id")
        if self.kind not in _SAFE_OBSERVATION_KINDS:
            raise ReleaseEvidenceError("unknown observation field kind")

    def validate(self, value: object) -> None:
        if self.kind == "boolean":
            if type(value) is not bool:
                raise ReleaseEvidenceError(f"{self.field_id} must be a boolean")
            return
        if self.kind == "digest":
            _require_digest(value, self.field_id)
            return
        if type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ReleaseEvidenceError(f"{self.field_id} must be a finite numeric metric")
        if self.kind in {"positive_count", "nonnegative_count", "zero_count", "sample_count", "positive_bytes"} and type(value) is not int:
            raise ReleaseEvidenceError(f"{self.field_id} must be an integer metric")
        if self.kind in {"positive_count", "positive_duration_ms", "positive_bytes"} and value <= 0:
            raise ReleaseEvidenceError(f"{self.field_id} must be positive")
        if self.kind == "nonnegative_count" and value < 0:
            raise ReleaseEvidenceError(f"{self.field_id} must be non-negative")
        if self.kind == "zero_count" and value != 0:
            raise ReleaseEvidenceError(f"{self.field_id} must be zero")
        if self.kind == "sample_count" and value < 3:
            raise ReleaseEvidenceError(f"{self.field_id} must be at least 3")

    def to_mapping(self) -> dict[str, str]:
        return {"field_id": self.field_id, "kind": self.kind}


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    """Exact content-free observation shape expected by a gate."""

    schema_id: str
    fields: tuple[ObservationField, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.schema_id, "observation schema_id")
        if not self.fields or any(not isinstance(field, ObservationField) for field in self.fields):
            raise ReleaseEvidenceError("observation schema requires fields")
        field_ids = [field.field_id for field in self.fields]
        if len(set(field_ids)) != len(field_ids):
            raise ReleaseEvidenceError("observation schema cannot repeat a field")

    def validate(self, observation: Mapping[str, object]) -> None:
        expected = {field.field_id for field in self.fields}
        if set(observation) != expected:
            raise ReleaseEvidenceError("measured observation does not match the gate schema")
        for field in self.fields:
            field.validate(observation[field.field_id])

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "fields": [field.to_mapping() for field in self.fields],
        }


@dataclass(frozen=True, slots=True)
class PerformanceTarget:
    """One backend/mode-specific performance workload."""

    metric: PerformanceMetric
    backend: str
    mode: str
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.metric, PerformanceMetric):
            raise ReleaseEvidenceError("performance target metric must be PerformanceMetric")
        _require_identifier(self.backend, "performance backend")
        _require_identifier(self.mode, "performance mode")
        if self.unit not in {"ms", "bytes"}:
            raise ReleaseEvidenceError("performance target unit must be ms or bytes")
        if self.metric is PerformanceMetric.STORAGE_GROWTH and self.unit != "bytes":
            raise ReleaseEvidenceError("storage growth must be measured in bytes")
        if self.metric is not PerformanceMetric.STORAGE_GROWTH and self.unit != "ms":
            raise ReleaseEvidenceError("duration metrics must be measured in ms")

    @property
    def gate_suffix(self) -> str:
        return f"{self.metric.value}_{self.backend}_{self.mode}"

    def to_mapping(self) -> dict[str, str]:
        return {
            "metric": self.metric.value,
            "backend": self.backend,
            "mode": self.mode,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class GateSpec:
    """Immutable source of truth for one release-evidence gate."""

    gate_id: str
    category: str
    scope: str
    runner: RunnerContract
    environment: ExecutionEnvironment
    fixture: FixtureContract
    observation_schema: ObservationSchema
    owner: str = "kestrel_core"
    advertised: bool = True
    required_for_ready: bool = True
    performance_target: PerformanceTarget | None = None
    correlation: DrillBinding | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.gate_id, "gate_id")
        _require_identifier(self.category, "gate category")
        _require_identifier(self.scope, "gate scope")
        _require_identifier(self.owner, "gate owner")
        if not isinstance(self.runner, RunnerContract):
            raise ReleaseEvidenceError("gate requires RunnerContract")
        if not isinstance(self.environment, ExecutionEnvironment):
            raise ReleaseEvidenceError("gate requires ExecutionEnvironment")
        if not isinstance(self.fixture, FixtureContract):
            raise ReleaseEvidenceError("gate requires FixtureContract")
        if not isinstance(self.observation_schema, ObservationSchema):
            raise ReleaseEvidenceError("gate requires ObservationSchema")
        if type(self.advertised) is not bool or type(self.required_for_ready) is not bool:
            raise ReleaseEvidenceError("gate advertised and required values must be booleans")
        if self.required_for_ready and not self.advertised:
            raise ReleaseEvidenceError("an unadvertised gate cannot be required for ready")
        if self.category == "performance" and self.performance_target is None:
            raise ReleaseEvidenceError("performance gate requires a PerformanceTarget")
        if self.category != "performance" and self.performance_target is not None:
            raise ReleaseEvidenceError("only performance gates may carry a PerformanceTarget")
        if self.correlation is not None and not isinstance(self.correlation, DrillBinding):
            raise ReleaseEvidenceError("gate correlation must be DrillBinding or null")
        if self.category in {"erasure", "external_adapter"} and self.correlation is None:
            raise ReleaseEvidenceError("erasure and external adapter gates require drill correlation")
        if self.category not in {"erasure", "external_adapter"} and self.correlation is not None:
            raise ReleaseEvidenceError("only erasure and external adapter gates may carry drill correlation")

    @property
    def digest(self) -> str:
        return _sha256(_canonical_json(self.contract_mapping()))

    def contract_mapping(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "category": self.category,
            "scope": self.scope,
            "runner": self.runner.to_mapping(),
            "environment": self.environment.to_mapping(),
            "fixture": self.fixture.to_mapping(),
            "observation_schema": self.observation_schema.to_mapping(),
            "owner": self.owner,
            "advertised": self.advertised,
            "required_for_ready": self.required_for_ready,
            "performance_target": (
                self.performance_target.to_mapping() if self.performance_target else None
            ),
            "correlation": self.correlation.to_mapping() if self.correlation else None,
        }

    def initial_evidence(self) -> "EvidenceRecord":
        if not self.advertised:
            return EvidenceRecord(
                gate_id=self.gate_id,
                state=EvidenceState.SKIPPED,
                reason_code="outside_advertised_capability",
                outside_advertised_capability=True,
            )
        if self.owner != "kestrel_core":
            return EvidenceRecord(
                gate_id=self.gate_id,
                state=EvidenceState.BLOCKED,
                reason_code="external_evidence_required",
            )
        return EvidenceRecord(
            gate_id=self.gate_id,
            state=EvidenceState.NOT_RUN,
            reason_code="not_executed",
        )

    def validate_attestation(self, evidence: "EvidenceRecord") -> None:
        """Reject an attestation that is not cryptographically bound to this spec."""
        if evidence.gate_id != self.gate_id:
            raise ReleaseEvidenceError("evidence gate_id does not match its gate spec")
        if evidence.state not in {EvidenceState.PASSED, EvidenceState.FAILED}:
            return
        if evidence.gate_spec_digest != self.digest:
            raise ReleaseEvidenceError("evidence gate spec digest does not match immutable catalog")
        if evidence.runner_id != self.runner.runner_id:
            raise ReleaseEvidenceError("evidence runner does not match immutable catalog")
        if evidence.command_id != self.runner.command_id or evidence.command_digest != self.runner.command_digest:
            raise ReleaseEvidenceError("evidence command digest does not match immutable catalog")
        if evidence.environment != self.environment or evidence.environment_digest != self.environment.digest:
            raise ReleaseEvidenceError("evidence environment does not match immutable catalog")
        if evidence.fixture != self.fixture.binding:
            raise ReleaseEvidenceError("evidence fixture does not match immutable catalog")
        if evidence.drill != self.correlation:
            raise ReleaseEvidenceError("evidence drill correlation does not match immutable catalog")
        if evidence.observation is None:
            raise ReleaseEvidenceError("release-ready evidence requires a measured observation")
        self.observation_schema.validate(evidence.observation)
        if evidence.run_digest != evidence.calculated_run_digest():
            raise ReleaseEvidenceError("evidence run digest does not bind its structured result")

    def to_mapping(self) -> dict[str, object]:
        return {**self.contract_mapping(), "spec_digest": self.digest}


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """A spec-bound result; raw argv and arbitrary artifact URLs are forbidden."""

    gate_id: str
    state: EvidenceState
    gate_spec_digest: str | None = None
    runner_id: str | None = None
    command_id: str | None = None
    command_digest: str | None = None
    environment: ExecutionEnvironment | None = None
    environment_digest: str | None = None
    fixture: FixtureBinding | None = None
    observation: Mapping[str, object] | None = None
    artifact: ArtifactReference | None = None
    run_digest: str | None = None
    execution_attestation: ExecutionAttestation | None = None
    drill: DrillBinding | None = None
    reason_code: str | None = None
    outside_advertised_capability: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.gate_id, "gate_id")
        if not isinstance(self.state, EvidenceState):
            raise ReleaseEvidenceError("evidence state must be EvidenceState")
        if self.reason_code is not None:
            _require_identifier(self.reason_code, "reason_code")
        if type(self.outside_advertised_capability) is not bool:
            raise ReleaseEvidenceError("outside_advertised_capability must be a boolean")
        if self.drill is not None and not isinstance(self.drill, DrillBinding):
            raise ReleaseEvidenceError("evidence drill must be DrillBinding or null")
        technical = (
            self.gate_spec_digest,
            self.runner_id,
            self.command_id,
            self.command_digest,
            self.environment,
            self.environment_digest,
            self.fixture,
            self.observation,
            self.artifact,
            self.run_digest,
            self.execution_attestation,
        )
        if self.state in {EvidenceState.PASSED, EvidenceState.FAILED}:
            if any(value is None for value in technical):
                raise ReleaseEvidenceError(
                    "passing or failed evidence requires spec, runner, environment, fixture, observation, artifact, and run digests"
                )
            _require_digest(self.gate_spec_digest, "gate_spec_digest")
            _require_identifier(self.runner_id, "runner_id")
            _require_identifier(self.command_id, "command_id")
            _require_digest(self.command_digest, "command_digest")
            if not isinstance(self.environment, ExecutionEnvironment):
                raise ReleaseEvidenceError("evidence environment must be ExecutionEnvironment")
            _require_digest(self.environment_digest, "environment_digest")
            if not isinstance(self.fixture, FixtureBinding):
                raise ReleaseEvidenceError("evidence fixture must be FixtureBinding")
            if not isinstance(self.artifact, ArtifactReference):
                raise ReleaseEvidenceError("evidence artifact must be ArtifactReference")
            if not isinstance(self.observation, Mapping):
                raise ReleaseEvidenceError("evidence observation must be a mapping")
            object.__setattr__(self, "observation", _safe_observation_mapping(self.observation))
            _require_digest(self.run_digest, "run_digest")
            if not isinstance(self.execution_attestation, ExecutionAttestation):
                raise ReleaseEvidenceError("release-ready evidence requires an execution attestation")
            if self.state is EvidenceState.PASSED and self.reason_code is not None:
                raise ReleaseEvidenceError("passed evidence cannot have a reason_code")
            if self.state is EvidenceState.FAILED and self.reason_code is None:
                raise ReleaseEvidenceError("failed evidence requires a reason_code")
        elif self.state in {EvidenceState.BLOCKED, EvidenceState.NOT_RUN}:
            if self.reason_code is None:
                raise ReleaseEvidenceError(f"{self.state.value} evidence requires a reason_code")
            if any(value is not None for value in technical):
                raise ReleaseEvidenceError("blocked/not_run evidence cannot carry an unbound result")
        elif self.state is EvidenceState.SKIPPED:
            if not self.outside_advertised_capability or self.reason_code is None:
                raise ReleaseEvidenceError("skipped evidence is permitted only outside advertised capability")
            if any(value is not None for value in technical):
                raise ReleaseEvidenceError("skipped evidence cannot carry an unbound result")
        if self.outside_advertised_capability and self.state is not EvidenceState.SKIPPED:
            raise ReleaseEvidenceError("outside_advertised_capability is valid only for skipped evidence")

    @classmethod
    def attest(
        cls,
        spec: GateSpec,
        observation: Mapping[str, object],
        artifact: ArtifactReference,
        *,
        state: EvidenceState = EvidenceState.PASSED,
        reason_code: str | None = None,
    ) -> "EvidenceRecord":
        """Refuse direct record minting outside the trusted execution boundary."""
        raise ReleaseEvidenceError(
            "EvidenceRecord.attest cannot mint release-ready evidence; "
            "use an allowlisted CatalogExecutionAuthority or verified external CI"
        )

    @classmethod
    def _bound_run_digest(
        cls,
        spec: GateSpec,
        observation: Mapping[str, object],
        artifact: ArtifactReference,
        *,
        state: EvidenceState,
        reason_code: str | None = None,
    ) -> tuple[Mapping[str, object], str]:
        """Build content-free bound fields before an authority signs them."""
        if state not in {EvidenceState.PASSED, EvidenceState.FAILED}:
            raise ReleaseEvidenceError("trusted execution supports only passed or failed release results")
        safe_observation = _safe_observation_mapping(observation)
        payload = {
            "gate_id": spec.gate_id,
            "state": state.value,
            "gate_spec_digest": spec.digest,
            "runner_id": spec.runner.runner_id,
            "command_id": spec.runner.command_id,
            "command_digest": spec.runner.command_digest,
            "environment": spec.environment.to_mapping(),
            "environment_digest": spec.environment.digest,
            "fixture": spec.fixture.binding.to_mapping(),
            "observation": dict(safe_observation),
            "artifact": artifact.to_mapping(),
            "drill": spec.correlation.to_mapping() if spec.correlation else None,
            "reason_code": reason_code,
        }
        return safe_observation, _sha256(_canonical_json(payload))

    @classmethod
    def _from_trusted_execution(
        cls,
        spec: GateSpec,
        observation: Mapping[str, object],
        artifact: ArtifactReference,
        *,
        state: EvidenceState,
        execution_attestation: ExecutionAttestation,
        reason_code: str | None = None,
    ) -> "EvidenceRecord":
        """Construct a signed record for an execution authority only.

        This underscore API intentionally has no public CLI call path.  The
        caller must first obtain a valid signature over the deterministic run
        digest from an authority that actually executed an allowlisted
        workload; :class:`GateResult` later verifies it against an independent
        public-key policy.
        """
        safe_observation, run_digest = cls._bound_run_digest(
            spec,
            observation,
            artifact,
            state=state,
            reason_code=reason_code,
        )
        return cls(
            gate_id=spec.gate_id,
            state=state,
            gate_spec_digest=spec.digest,
            runner_id=spec.runner.runner_id,
            command_id=spec.runner.command_id,
            command_digest=spec.runner.command_digest,
            environment=spec.environment,
            environment_digest=spec.environment.digest,
            fixture=spec.fixture.binding,
            observation=safe_observation,
            artifact=artifact,
            run_digest=run_digest,
            execution_attestation=execution_attestation,
            drill=spec.correlation,
            reason_code=reason_code,
        )

    def calculated_run_digest(self) -> str:
        if self.state not in {EvidenceState.PASSED, EvidenceState.FAILED}:
            raise ReleaseEvidenceError("unbound evidence does not have a run digest")
        assert self.environment is not None
        assert self.fixture is not None
        assert self.observation is not None
        assert self.artifact is not None
        return _sha256(
            _canonical_json(
                {
                    "gate_id": self.gate_id,
                    "state": self.state.value,
                    "gate_spec_digest": self.gate_spec_digest,
                    "runner_id": self.runner_id,
                    "command_id": self.command_id,
                    "command_digest": self.command_digest,
                    "environment": self.environment.to_mapping(),
                    "environment_digest": self.environment_digest,
                    "fixture": self.fixture.to_mapping(),
                    "observation": dict(self.observation),
                    "artifact": self.artifact.to_mapping(),
                    "drill": self.drill.to_mapping() if self.drill else None,
                    "reason_code": self.reason_code,
                }
            )
        )

    @property
    def passed(self) -> bool:
        return self.state is EvidenceState.PASSED

    def to_mapping(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "state": self.state.value,
            "gate_spec_digest": self.gate_spec_digest,
            "runner_id": self.runner_id,
            "command_id": self.command_id,
            "command_digest": self.command_digest,
            "environment": self.environment.to_mapping() if self.environment else None,
            "environment_digest": self.environment_digest,
            "fixture": self.fixture.to_mapping() if self.fixture else None,
            "observation": dict(self.observation) if self.observation else None,
            "artifact": self.artifact.to_mapping() if self.artifact else None,
            "run_digest": self.run_digest,
            "execution_attestation": (
                self.execution_attestation.to_mapping() if self.execution_attestation else None
            ),
            "drill": self.drill.to_mapping() if self.drill else None,
            "reason_code": self.reason_code,
            "outside_advertised_capability": self.outside_advertised_capability,
        }


def _validate_gate_evidence_binding(spec: GateSpec, evidence: EvidenceRecord) -> None:
    """Validate catalog binding without deciding whether a signer is trusted."""
    if not isinstance(spec, GateSpec) or not isinstance(evidence, EvidenceRecord):
        raise ReleaseEvidenceError("gate result requires a GateSpec and EvidenceRecord")
    if spec.gate_id != evidence.gate_id:
        raise ReleaseEvidenceError("gate result evidence must match its declared gate_id")
    if spec.advertised and evidence.state is EvidenceState.SKIPPED:
        raise ReleaseEvidenceError("an advertised release gate cannot be skipped")
    if not spec.advertised and (
        evidence.state is not EvidenceState.SKIPPED
        or not evidence.outside_advertised_capability
    ):
        raise ReleaseEvidenceError("an unadvertised gate must remain explicitly skipped")
    spec.validate_attestation(evidence)


@dataclass(frozen=True, slots=True)
class StructuralGateResult:
    """Catalog-bound evidence submitted for inspection without a trust verdict.

    This preserves the exact content and digest binding of a submitted result,
    while deliberately making no claim that the listed signing key is trusted.
    """

    spec: GateSpec
    evidence: EvidenceRecord

    def __post_init__(self) -> None:
        _validate_gate_evidence_binding(self.spec, self.evidence)

    @property
    def structurally_ready(self) -> bool:
        return not self.spec.required_for_ready or self.evidence.passed

    def to_mapping(self) -> dict[str, object]:
        return {**self.spec.to_mapping(), "evidence": self.evidence.to_mapping()}


@dataclass(frozen=True, slots=True)
class GateResult:
    """A declared gate whose signer was verified by an explicit trust policy."""

    spec: GateSpec
    evidence: EvidenceRecord
    trust_policy: TrustedExecutionPolicy | None = None

    def __post_init__(self) -> None:
        _validate_gate_evidence_binding(self.spec, self.evidence)
        if self.evidence.state in {EvidenceState.PASSED, EvidenceState.FAILED}:
            if not isinstance(self.trust_policy, TrustedExecutionPolicy):
                raise ReleaseEvidenceError(
                    "release-ready evidence requires an operator trusted execution policy"
                )
            self.trust_policy.verify_evidence(self.spec, self.evidence)
        elif self.trust_policy is not None and not isinstance(self.trust_policy, TrustedExecutionPolicy):
            raise ReleaseEvidenceError("gate trust_policy must be TrustedExecutionPolicy or null")

    @property
    def ready(self) -> bool:
        return not self.spec.required_for_ready or self.evidence.passed

    def to_mapping(self) -> dict[str, object]:
        return {**self.spec.to_mapping(), "evidence": self.evidence.to_mapping()}


@dataclass(frozen=True, slots=True)
class StandardsMatrixEntry:
    """One exact offline registry resource pin and official fixture contract."""

    identifier: str
    version: str
    maturity: str
    kind: str
    uri: str
    published_date: str
    sha256: str
    capabilities: tuple[str, ...]
    fixture: FixtureContract | None = None
    runner: RunnerContract | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.identifier, "standards identifier")
        if not isinstance(self.version, str) or not self.version:
            raise ReleaseEvidenceError("standards version must be present")
        if self.maturity not in {member.value for member in StandardsMaturity}:
            raise ReleaseEvidenceError("standards maturity is unknown")
        if not all(isinstance(value, str) and value for value in (self.kind, self.uri, self.published_date)):
            raise ReleaseEvidenceError("standards matrix metadata must be present")
        _require_digest(self.sha256, "standards sha256")
        if any(not isinstance(capability, str) or not capability for capability in self.capabilities):
            raise ReleaseEvidenceError("standards capabilities must be non-empty strings")
        if self.fixture is not None and not isinstance(self.fixture, FixtureContract):
            raise ReleaseEvidenceError("standards fixture must be FixtureContract")
        if self.runner is not None and not isinstance(self.runner, RunnerContract):
            raise ReleaseEvidenceError("standards runner must be RunnerContract")

    def to_mapping(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "version": self.version,
            "maturity": self.maturity,
            "kind": self.kind,
            "uri": self.uri,
            "published_date": self.published_date,
            "sha256": self.sha256,
            "capabilities": list(self.capabilities),
            "fixture": self.fixture.to_mapping() if self.fixture else None,
            "runner": self.runner.to_mapping() if self.runner else None,
        }


@dataclass(frozen=True, slots=True)
class PerformanceBudget:
    """Measured p95 budget bound to one backend/mode gate, never a timeout."""

    target: PerformanceTarget
    gate_id: str
    gate_spec_digest: str
    samples: tuple[float | int, ...]
    p95: float | int
    budget: float | int
    headroom_fraction: float
    fixture: FixtureBinding
    environment: ExecutionEnvironment
    artifact: ArtifactReference
    run_digest: str
    execution_attestation: ExecutionAttestation

    def __post_init__(self) -> None:
        if not isinstance(self.target, PerformanceTarget):
            raise ReleaseEvidenceError("performance budget requires PerformanceTarget")
        _require_identifier(self.gate_id, "performance gate_id")
        _require_digest(self.gate_spec_digest, "performance gate_spec_digest")
        if len(self.samples) < 3:
            raise ReleaseEvidenceError("performance budgets require sample_count >= 3")
        if any(
            type(sample) is bool
            or not isinstance(sample, (int, float))
            or not math.isfinite(sample)
            or sample <= 0
            for sample in self.samples
        ):
            raise ReleaseEvidenceError("performance samples must be finite and positive")
        if self.target.unit == "bytes" and any(type(sample) is not int for sample in self.samples):
            raise ReleaseEvidenceError("storage growth performance samples must be integer byte deltas")
        for name, value in (("p95", self.p95), ("budget", self.budget), ("headroom_fraction", self.headroom_fraction)):
            if type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ReleaseEvidenceError(f"performance {name} must be finite")
        if self.p95 <= 0 or self.budget <= 0 or self.budget < self.p95:
            raise ReleaseEvidenceError("performance p95 and budget must be positive")
        if not 0 < self.headroom_fraction <= 10:
            raise ReleaseEvidenceError("performance headroom_fraction must be positive")
        expected_p95 = sorted(self.samples)[math.ceil(len(self.samples) * 0.95) - 1]
        expected_budget = expected_p95 * (1 + self.headroom_fraction)
        if not math.isclose(self.p95, expected_p95, rel_tol=0.0, abs_tol=1e-9):
            raise ReleaseEvidenceError("performance p95 must match observed samples")
        if not math.isclose(self.budget, expected_budget, rel_tol=0.0, abs_tol=1e-9):
            raise ReleaseEvidenceError("performance budget must equal p95 plus declared headroom")
        if not isinstance(self.fixture, FixtureBinding) or not isinstance(self.environment, ExecutionEnvironment):
            raise ReleaseEvidenceError("performance budget requires fixture and environment bindings")
        if self.environment.backend != self.target.backend or self.environment.mode != self.target.mode:
            raise ReleaseEvidenceError("performance environment does not match target backend/mode")
        if not isinstance(self.artifact, ArtifactReference):
            raise ReleaseEvidenceError("performance budget requires an artifact")
        _require_digest(self.run_digest, "performance run_digest")
        if not isinstance(self.execution_attestation, ExecutionAttestation):
            raise ReleaseEvidenceError("performance budget requires an execution attestation")
        if self.run_digest != self.calculated_run_digest():
            raise ReleaseEvidenceError("performance run digest does not bind samples and artifact")

    @classmethod
    def from_observed(
        cls,
        spec: GateSpec,
        samples: Iterable[float | int],
        *,
        headroom_fraction: float,
        artifact: ArtifactReference,
    ) -> "PerformanceBudget":
        """Refuse caller-provided samples outside trusted benchmark execution."""
        raise ReleaseEvidenceError(
            "PerformanceBudget.from_observed cannot mint a release budget; "
            "use an allowlisted CatalogExecutionAuthority"
        )

    @classmethod
    def _bound_run_digest(
        cls,
        spec: GateSpec,
        samples: Iterable[float | int],
        *,
        headroom_fraction: float,
        artifact: ArtifactReference,
    ) -> tuple[tuple[float | int, ...], float | int, float | int, str]:
        if spec.performance_target is None:
            raise ReleaseEvidenceError("performance budget requires a performance gate spec")
        values = tuple(samples)
        if len(values) < 3 or any(
            type(sample) is bool
            or not isinstance(sample, (int, float))
            or not math.isfinite(sample)
            or sample <= 0
            for sample in values
        ):
            raise ReleaseEvidenceError("performance samples must be finite and positive with sample_count >= 3")
        if spec.performance_target.unit == "bytes" and any(type(sample) is not int for sample in values):
            raise ReleaseEvidenceError("storage growth performance samples must be integer byte deltas")
        if (
            type(headroom_fraction) is bool
            or not isinstance(headroom_fraction, (int, float))
            or not math.isfinite(headroom_fraction)
            or not 0 < headroom_fraction <= 10
        ):
            raise ReleaseEvidenceError("performance headroom_fraction must be positive")
        if not isinstance(artifact, ArtifactReference):
            raise ReleaseEvidenceError("performance budget requires an artifact")
        p95 = sorted(values)[math.ceil(len(values) * 0.95) - 1] if values else 0
        budget = p95 * (1 + headroom_fraction)
        payload = {
            "target": spec.performance_target.to_mapping(),
            "gate_id": spec.gate_id,
            "gate_spec_digest": spec.digest,
            "samples": list(values),
            "p95": p95,
            "budget": budget,
            "headroom_fraction": headroom_fraction,
            "fixture": spec.fixture.binding.to_mapping(),
            "environment": spec.environment.to_mapping(),
            "artifact": artifact.to_mapping(),
        }
        return values, p95, budget, _sha256(_canonical_json(payload))

    @classmethod
    def _from_trusted_execution(
        cls,
        spec: GateSpec,
        samples: Iterable[float | int],
        *,
        headroom_fraction: float,
        artifact: ArtifactReference,
        execution_attestation: ExecutionAttestation,
    ) -> "PerformanceBudget":
        values, p95, budget, run_digest = cls._bound_run_digest(
            spec,
            samples,
            headroom_fraction=headroom_fraction,
            artifact=artifact,
        )
        return cls(
            target=spec.performance_target,
            gate_id=spec.gate_id,
            gate_spec_digest=spec.digest,
            samples=values,
            p95=p95,
            budget=budget,
            headroom_fraction=headroom_fraction,
            fixture=spec.fixture.binding,
            environment=spec.environment,
            artifact=artifact,
            run_digest=run_digest,
            execution_attestation=execution_attestation,
        )

    def calculated_run_digest(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "target": self.target.to_mapping(),
                    "gate_id": self.gate_id,
                    "gate_spec_digest": self.gate_spec_digest,
                    "samples": list(self.samples),
                    "p95": self.p95,
                    "budget": self.budget,
                    "headroom_fraction": self.headroom_fraction,
                    "fixture": self.fixture.to_mapping(),
                    "environment": self.environment.to_mapping(),
                    "artifact": self.artifact.to_mapping(),
                }
            )
        )

    def validate_against(self, spec: GateSpec) -> None:
        if spec.performance_target != self.target:
            raise ReleaseEvidenceError("performance target does not match immutable catalog")
        if self.gate_id != spec.gate_id or self.gate_spec_digest != spec.digest:
            raise ReleaseEvidenceError("performance budget does not match its gate spec")
        if self.fixture != spec.fixture.binding or self.environment != spec.environment:
            raise ReleaseEvidenceError("performance budget fixture/environment does not match catalog")

    def to_mapping(self) -> dict[str, object]:
        return {
            "target": self.target.to_mapping(),
            "gate_id": self.gate_id,
            "gate_spec_digest": self.gate_spec_digest,
            "samples": list(self.samples),
            "p95": self.p95,
            "budget": self.budget,
            "headroom_fraction": self.headroom_fraction,
            "fixture": self.fixture.to_mapping(),
            "environment": self.environment.to_mapping(),
            "artifact": self.artifact.to_mapping(),
            "run_digest": self.run_digest,
            "execution_attestation": self.execution_attestation.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """Reproducible samples for one declared performance target."""

    target: PerformanceTarget
    samples: tuple[float | int, ...]
    fixture: FixtureBinding
    environment: ExecutionEnvironment

    def __post_init__(self) -> None:
        if len(self.samples) < 3 or any(
            type(value) is bool
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in self.samples
        ):
            raise ReleaseEvidenceError("benchmark runs require at least three positive samples")
        if self.target.unit == "bytes" and any(type(value) is not int for value in self.samples):
            raise ReleaseEvidenceError("storage growth benchmark samples must be integer byte deltas")
        if not isinstance(self.fixture, FixtureBinding) or not isinstance(self.environment, ExecutionEnvironment):
            raise ReleaseEvidenceError("benchmark run requires fixture and environment")

    def to_mapping(self) -> dict[str, object]:
        return {
            "target": self.target.to_mapping(),
            "samples": list(self.samples),
            "fixture": self.fixture.to_mapping(),
            "environment": self.environment.to_mapping(),
        }


class SemanticBenchmarkHarness:
    """Repeat a declared workload using the metric's actual unit.

    Duration gates use elapsed milliseconds.  ``storage_growth`` is different:
    each sample is the measured post-operation storage footprint minus the
    pre-operation footprint in bytes.  Callers must supply a backend-specific
    byte reader (for example a SQLite file-size reader or a Postgres relation
    size query); wall-clock time is never relabeled as bytes.
    """

    def __init__(self, *, iterations: int = 5) -> None:
        if type(iterations) is not int or iterations < 3:
            raise ReleaseEvidenceError("benchmark iterations must be an integer >= 3")
        self.iterations = iterations

    async def run(
        self,
        spec: GateSpec,
        operation: Callable[[], object | Awaitable[object]],
        *,
        storage_bytes: Callable[[], int | Awaitable[int]] | None = None,
        before_sample: Callable[[], object | Awaitable[object]] | None = None,
        after_sample: Callable[[], object | Awaitable[object]] | None = None,
    ) -> BenchmarkRun:
        if spec.performance_target is None or not callable(operation):
            raise ReleaseEvidenceError("benchmark requires a performance gate and callable operation")
        is_storage_growth = spec.performance_target.metric is PerformanceMetric.STORAGE_GROWTH
        if is_storage_growth and not callable(storage_bytes):
            raise ReleaseEvidenceError("storage growth benchmark requires a backend-specific storage_bytes reader")
        if not is_storage_growth and storage_bytes is not None:
            raise ReleaseEvidenceError("storage_bytes is valid only for a storage growth benchmark")
        samples: list[float | int] = []
        for _ in range(self.iterations):
            if before_sample is not None:
                prepared = before_sample()
                if inspect.isawaitable(prepared):
                    await prepared
            before_bytes = await self._storage_bytes(storage_bytes) if is_storage_growth else None
            started = time.perf_counter() if not is_storage_growth else None
            result = operation()
            if inspect.isawaitable(result):
                await result
            if is_storage_growth:
                assert before_bytes is not None
                after_bytes = await self._storage_bytes(storage_bytes)
                delta_bytes = after_bytes - before_bytes
                if delta_bytes <= 0:
                    raise ReleaseEvidenceError("storage growth benchmark must observe a positive byte delta")
                samples.append(delta_bytes)
            else:
                assert started is not None
                elapsed = (time.perf_counter() - started) * 1_000
                if elapsed <= 0:
                    elapsed = float.fromhex("0x1.0p-52")
                samples.append(elapsed)
            if after_sample is not None:
                finalized = after_sample()
                if inspect.isawaitable(finalized):
                    await finalized
        return BenchmarkRun(
            target=spec.performance_target,
            samples=tuple(samples),
            fixture=spec.fixture.binding,
            environment=spec.environment,
        )

    @staticmethod
    async def _storage_bytes(
        reader: Callable[[], int | Awaitable[int]] | None,
    ) -> int:
        assert reader is not None
        value = reader()
        if inspect.isawaitable(value):
            value = await value
        if type(value) is not int or value < 0:
            raise ReleaseEvidenceError("storage_bytes reader must return a non-negative integer byte count")
        return value


def _utc_timestamp(value: object, field_name: str) -> datetime:
    """Validate an explicit UTC timestamp without accepting locale-dependent text."""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ReleaseEvidenceError(f"{field_name} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ReleaseEvidenceError(f"{field_name} must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None:
        raise ReleaseEvidenceError(f"{field_name} must include UTC timezone")
    return parsed


@dataclass(frozen=True, slots=True)
class TelemetryAttestation:
    """Content-free, digest-bound compatibility telemetry observation window."""

    window_started_at: str
    window_ended_at: str
    inventory_digest: str
    inventory_complete: bool
    unmigrated_eligible_rows: int
    required_consumer_count: int
    artifact: ArtifactReference
    telemetry_digest: str

    def __post_init__(self) -> None:
        started = _utc_timestamp(self.window_started_at, "window_started_at")
        ended = _utc_timestamp(self.window_ended_at, "window_ended_at")
        if ended <= started:
            raise ReleaseEvidenceError("telemetry window must end after it starts")
        _require_digest(self.inventory_digest, "inventory_digest")
        if type(self.inventory_complete) is not bool:
            raise ReleaseEvidenceError("inventory_complete must be a boolean")
        for value, name in (
            (self.unmigrated_eligible_rows, "unmigrated_eligible_rows"),
            (self.required_consumer_count, "required_consumer_count"),
        ):
            if type(value) is not int or value < 0:
                raise ReleaseEvidenceError(f"{name} must be a non-negative integer")
        if not isinstance(self.artifact, ArtifactReference):
            raise ReleaseEvidenceError("telemetry attestation requires an artifact")
        _require_digest(self.telemetry_digest, "telemetry_digest")
        if self.telemetry_digest != self.calculated_digest():
            raise ReleaseEvidenceError("telemetry digest does not bind its window and counts")

    @classmethod
    def attest(
        cls,
        *,
        window_started_at: str,
        window_ended_at: str,
        inventory_digest: str,
        inventory_complete: bool,
        unmigrated_eligible_rows: int,
        required_consumer_count: int,
        artifact: ArtifactReference,
    ) -> "TelemetryAttestation":
        payload = {
            "window_started_at": window_started_at,
            "window_ended_at": window_ended_at,
            "inventory_digest": inventory_digest,
            "inventory_complete": inventory_complete,
            "unmigrated_eligible_rows": unmigrated_eligible_rows,
            "required_consumer_count": required_consumer_count,
            "artifact": artifact.to_mapping(),
        }
        return cls(
            window_started_at=window_started_at,
            window_ended_at=window_ended_at,
            inventory_digest=inventory_digest,
            inventory_complete=inventory_complete,
            unmigrated_eligible_rows=unmigrated_eligible_rows,
            required_consumer_count=required_consumer_count,
            artifact=artifact,
            telemetry_digest=_sha256(_canonical_json(payload)),
        )

    def calculated_digest(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "window_started_at": self.window_started_at,
                    "window_ended_at": self.window_ended_at,
                    "inventory_digest": self.inventory_digest,
                    "inventory_complete": self.inventory_complete,
                    "unmigrated_eligible_rows": self.unmigrated_eligible_rows,
                    "required_consumer_count": self.required_consumer_count,
                    "artifact": self.artifact.to_mapping(),
                }
            )
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "window_started_at": self.window_started_at,
            "window_ended_at": self.window_ended_at,
            "inventory_digest": self.inventory_digest,
            "inventory_complete": self.inventory_complete,
            "unmigrated_eligible_rows": self.unmigrated_eligible_rows,
            "required_consumer_count": self.required_consumer_count,
            "artifact": self.artifact.to_mapping(),
            "telemetry_digest": self.telemetry_digest,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityRetirementDecision:
    """A removal decision bound to the exact legacy-migration gate and telemetry."""

    path_id: str
    migration_gate_id: str
    migration_spec_digest: str
    migration_run_digest: str | None
    telemetry: TelemetryAttestation | None

    def __post_init__(self) -> None:
        _require_identifier(self.path_id, "compatibility path_id")
        _require_identifier(self.migration_gate_id, "migration_gate_id")
        _require_digest(self.migration_spec_digest, "migration_spec_digest")
        if self.migration_run_digest is not None:
            _require_digest(self.migration_run_digest, "migration_run_digest")
        if self.telemetry is not None and not isinstance(self.telemetry, TelemetryAttestation):
            raise ReleaseEvidenceError("telemetry must be TelemetryAttestation or null")

    @property
    def removal_safe(self) -> bool:
        return bool(
            self.migration_run_digest
            and self.telemetry
            and self.telemetry.inventory_complete
            and self.telemetry.unmigrated_eligible_rows == 0
            and self.telemetry.required_consumer_count == 0
        )

    @property
    def decision(self) -> str:
        return "eligible_for_review" if self.removal_safe else "retain"

    @property
    def reason_code(self) -> str:
        if self.telemetry is None:
            return "telemetry_not_observed"
        if not self.telemetry.inventory_complete:
            return "inventory_incomplete"
        if self.telemetry.unmigrated_eligible_rows:
            return "eligible_rows_remain"
        if self.telemetry.required_consumer_count:
            return "required_consumers_remain"
        if self.migration_run_digest is None:
            return "migration_equivalence_not_observed"
        return "telemetry_and_equivalence_observed"

    def to_mapping(self) -> dict[str, object]:
        return {
            "path_id": self.path_id,
            "migration_gate_id": self.migration_gate_id,
            "migration_spec_digest": self.migration_spec_digest,
            "migration_run_digest": self.migration_run_digest,
            "telemetry": self.telemetry.to_mapping() if self.telemetry else None,
            "decision": self.decision,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class ExternalGateAttestation:
    """External CI binding for one correlated adapter erasure result/artifact."""

    gate_id: str
    gate_spec_digest: str
    result_digest: str
    artifact: ArtifactReference
    drill: DrillBinding

    def __post_init__(self) -> None:
        _require_identifier(self.gate_id, "external gate_id")
        _require_digest(self.gate_spec_digest, "external gate_spec_digest")
        _require_digest(self.result_digest, "external result_digest")
        if not isinstance(self.artifact, ArtifactReference):
            raise ReleaseEvidenceError("external gate attestation requires artifact")
        if not isinstance(self.drill, DrillBinding):
            raise ReleaseEvidenceError("external gate attestation requires drill correlation")

    def to_mapping(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "gate_spec_digest": self.gate_spec_digest,
            "result_digest": self.result_digest,
            "artifact": self.artifact.to_mapping(),
            "drill": self.drill.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class ExternalCapabilityReport:
    """Hash-bound external adapter evidence; metadata alone is never sufficient."""

    capability_id: str
    repository: str
    source_revision: str
    core_release_evidence_contract_digest: str
    run_nonce: str
    attestations: tuple[ExternalGateAttestation, ...]
    attestation_digest: str

    def __post_init__(self) -> None:
        _require_identifier(self.capability_id, "external capability_id")
        if not isinstance(self.repository, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository):
            raise ReleaseEvidenceError("external repository must be owner/name")
        _require_commit(self.source_revision, "external source_revision")
        _require_digest(self.core_release_evidence_contract_digest, "external core_release_evidence_contract_digest")
        _require_digest(self.run_nonce, "external run_nonce")
        if (
            not isinstance(self.attestations, tuple)
            or not self.attestations
            or any(not isinstance(item, ExternalGateAttestation) for item in self.attestations)
        ):
            raise ReleaseEvidenceError("external capability report requires gate attestations")
        gate_ids = [item.gate_id for item in self.attestations]
        if len(set(gate_ids)) != len(gate_ids):
            raise ReleaseEvidenceError("external capability report cannot repeat a gate")
        _require_digest(self.attestation_digest, "external attestation_digest")
        if self.attestation_digest != self.calculated_digest():
            raise ReleaseEvidenceError("external attestation digest does not bind nonce/result/artifact references")

    @classmethod
    def attest(
        cls,
        *,
        capability_id: str,
        repository: str,
        source_revision: str,
        core_release_evidence_contract_digest: str,
        run_nonce: str,
        attestations: tuple[ExternalGateAttestation, ...],
    ) -> "ExternalCapabilityReport":
        payload = {
            "capability_id": capability_id,
            "repository": repository,
            "source_revision": source_revision,
            "core_release_evidence_contract_digest": core_release_evidence_contract_digest,
            "run_nonce": run_nonce,
            "attestations": [item.to_mapping() for item in attestations],
        }
        return cls(
            capability_id=capability_id,
            repository=repository,
            source_revision=source_revision,
            core_release_evidence_contract_digest=core_release_evidence_contract_digest,
            run_nonce=run_nonce,
            attestations=attestations,
            attestation_digest=_sha256(_canonical_json(payload)),
        )

    @property
    def gate_ids(self) -> tuple[str, ...]:
        return tuple(item.gate_id for item in self.attestations)

    @property
    def freshness_receipt(self) -> str:
        """Derive the content-free replay key for the independent verifier.

        The result-digest sequence is intentionally ordered.  The core
        catalog validates that order against its immutable external gates
        before a verifier consumes this core-derived receipt.
        """
        return calculate_external_freshness_receipt(
            core_release_evidence_contract_digest=self.core_release_evidence_contract_digest,
            repository=self.repository,
            source_revision=self.source_revision,
            run_nonce=self.run_nonce,
            record_digests=tuple(item.result_digest for item in self.attestations),
        )

    def calculated_digest(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "capability_id": self.capability_id,
                    "repository": self.repository,
                    "source_revision": self.source_revision,
                    "core_release_evidence_contract_digest": self.core_release_evidence_contract_digest,
                    "run_nonce": self.run_nonce,
                    "attestations": [item.to_mapping() for item in self.attestations],
                }
            )
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "capability_id": self.capability_id,
            "repository": self.repository,
            "source_revision": self.source_revision,
            "core_release_evidence_contract_digest": self.core_release_evidence_contract_digest,
            "run_nonce": self.run_nonce,
            "attestations": [item.to_mapping() for item in self.attestations],
            "attestation_digest": self.attestation_digest,
        }


def calculate_external_freshness_receipt(
    *,
    core_release_evidence_contract_digest: str,
    repository: str,
    source_revision: str,
    run_nonce: str,
    record_digests: tuple[str, ...],
) -> str:
    """Calculate the portable, content-free receipt an independent verifier consumes."""
    return _sha256(
        _canonical_json(
            {
                "core_release_evidence_contract_digest": core_release_evidence_contract_digest,
                "repository": repository,
                "source_revision": source_revision,
                "run_nonce": run_nonce,
                "record_digests": list(record_digests),
            }
        )
    )
