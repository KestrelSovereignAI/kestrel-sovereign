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
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
import hashlib
import inspect
import json
import math
import platform
import re
import time
from types import MappingProxyType

from .registry import StandardsMaturity


_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SAFE_ARTIFACT_RE = re.compile(
    r"^(?:ci|artifact|evidence)://[a-z0-9][a-z0-9._/-]{0,239}$"
)
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
        lowered = self.artifact_ref.lower()
        if any(token in lowered for token in _ARTIFACT_FORBIDDEN_TOKENS):
            raise ReleaseEvidenceError("artifact reference contains a forbidden secret or identity token")
        if any(character in self.artifact_ref for character in ("?", "#", "@", "%")):
            raise ReleaseEvidenceError("artifact reference cannot contain query, fragment, userinfo, or escapes")
        _require_digest(self.artifact_digest, "artifact_digest")

    def to_mapping(self) -> dict[str, str]:
        return {
            "artifact_ref": self.artifact_ref,
            "artifact_digest": self.artifact_digest,
        }


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
        """Create one bound record from the immutable catalog, never raw argv."""
        if state not in {EvidenceState.PASSED, EvidenceState.FAILED}:
            raise ReleaseEvidenceError("attest supports only passed or failed release results")
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
            "artifact_digest": artifact.artifact_digest,
        }
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
            run_digest=_sha256(_canonical_json(payload)),
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
                    "artifact_digest": self.artifact.artifact_digest,
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
            "reason_code": self.reason_code,
            "outside_advertised_capability": self.outside_advertised_capability,
        }


@dataclass(frozen=True, slots=True)
class GateResult:
    """A declared gate and the only evidence that may satisfy it."""

    spec: GateSpec
    evidence: EvidenceRecord

    def __post_init__(self) -> None:
        if not isinstance(self.spec, GateSpec) or not isinstance(self.evidence, EvidenceRecord):
            raise ReleaseEvidenceError("gate result requires a GateSpec and EvidenceRecord")
        if self.spec.gate_id != self.evidence.gate_id:
            raise ReleaseEvidenceError("gate result evidence must match its declared gate_id")
        if self.spec.advertised and self.evidence.state is EvidenceState.SKIPPED:
            raise ReleaseEvidenceError("an advertised release gate cannot be skipped")
        if not self.spec.advertised and (
            self.evidence.state is not EvidenceState.SKIPPED
            or not self.evidence.outside_advertised_capability
        ):
            raise ReleaseEvidenceError("an unadvertised gate must remain explicitly skipped")
        self.spec.validate_attestation(self.evidence)

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
        if spec.performance_target is None:
            raise ReleaseEvidenceError("performance budget requires a performance gate spec")
        values = tuple(samples)
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
            "artifact_digest": artifact.artifact_digest,
        }
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
            run_digest=_sha256(_canonical_json(payload)),
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
                    "artifact_digest": self.artifact.artifact_digest,
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
        }


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """Reproducible samples for one declared performance target."""

    target: PerformanceTarget
    samples: tuple[float, ...]
    fixture: FixtureBinding
    environment: ExecutionEnvironment

    def __post_init__(self) -> None:
        if len(self.samples) < 3 or any(not math.isfinite(value) or value <= 0 for value in self.samples):
            raise ReleaseEvidenceError("benchmark runs require at least three positive samples")
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
    """Repeat a declared workload; no timeout is ever treated as a benchmark."""

    def __init__(self, *, iterations: int = 5) -> None:
        if type(iterations) is not int or iterations < 3:
            raise ReleaseEvidenceError("benchmark iterations must be an integer >= 3")
        self.iterations = iterations

    async def run(
        self,
        spec: GateSpec,
        operation: Callable[[], object | Awaitable[object]],
    ) -> BenchmarkRun:
        if spec.performance_target is None or not callable(operation):
            raise ReleaseEvidenceError("benchmark requires a performance gate and callable operation")
        samples: list[float] = []
        for _ in range(self.iterations):
            started = time.perf_counter()
            result = operation()
            if inspect.isawaitable(result):
                await result
            elapsed = (time.perf_counter() - started) * 1_000
            if elapsed <= 0:
                elapsed = float.fromhex("0x1.0p-52")
            samples.append(elapsed)
        return BenchmarkRun(
            target=spec.performance_target,
            samples=tuple(samples),
            fixture=spec.fixture.binding,
            environment=spec.environment,
        )


@dataclass(frozen=True, slots=True)
class CompatibilityRetirementDecision:
    """Retain legacy compatibility unless separately bound telemetry proves removal."""

    path_id: str
    observed_window_ref: str | None
    inventory_complete: bool
    unmigrated_eligible_rows: int | None
    required_consumer_count: int | None
    migration_equivalence: EvidenceRecord

    def __post_init__(self) -> None:
        _require_identifier(self.path_id, "compatibility path_id")
        if self.observed_window_ref is not None:
            ArtifactReference(self.observed_window_ref, "0" * 64)
        if type(self.inventory_complete) is not bool:
            raise ReleaseEvidenceError("inventory_complete must be a boolean")
        for value, field_name in ((self.unmigrated_eligible_rows, "unmigrated eligible rows"), (self.required_consumer_count, "required consumer count")):
            if value is not None and (type(value) is not int or value < 0):
                raise ReleaseEvidenceError(f"{field_name} must be a non-negative integer or null")
        if not isinstance(self.migration_equivalence, EvidenceRecord):
            raise ReleaseEvidenceError("migration_equivalence must be EvidenceRecord")

    @property
    def removal_safe(self) -> bool:
        return False

    @property
    def decision(self) -> str:
        return "retain"

    @property
    def reason_code(self) -> str:
        return "telemetry_binding_required"

    def to_mapping(self) -> dict[str, object]:
        return {
            "path_id": self.path_id,
            "observed_window_ref": self.observed_window_ref,
            "inventory_complete": self.inventory_complete,
            "unmigrated_eligible_rows": self.unmigrated_eligible_rows,
            "required_consumer_count": self.required_consumer_count,
            "migration_equivalence": self.migration_equivalence.to_mapping(),
            "decision": self.decision,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class ExternalCapabilityReport:
    """External consumer identity; result binding is added in the next slice."""

    capability_id: str
    repository: str
    source_revision: str
    gate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.capability_id, "external capability_id")
        if not isinstance(self.repository, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository):
            raise ReleaseEvidenceError("external repository must be owner/name")
        _require_commit(self.source_revision, "external source_revision")
        if not self.gate_ids:
            raise ReleaseEvidenceError("external capability report requires gate IDs")
        for gate_id in self.gate_ids:
            _require_identifier(gate_id, "external gate_id")

    def to_mapping(self) -> dict[str, object]:
        return {
            "capability_id": self.capability_id,
            "repository": self.repository,
            "source_revision": self.source_revision,
            "gate_ids": list(self.gate_ids),
        }
