"""Content-free models and benchmark primitives for semantic release evidence."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
import inspect
import json
import math
import platform
import re
import time

from .registry import StandardsMaturity


_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")


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


def _require_identifier(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ReleaseEvidenceError(
            f"{field_name} must be a lowercase content-free identifier, got {value!r}"
        )


def _require_reference(value: object, field_name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value or len(value) > 512 or "\n" in value:
        raise ReleaseEvidenceError(
            f"{field_name} must be a bounded, single-line artifact reference"
        )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_gate_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One command or external outcome, never raw test output or user content."""

    gate_id: str
    state: EvidenceState
    command: str | None = None
    exit_code: int | None = None
    artifact_ref: str | None = None
    reason_code: str | None = None
    outside_advertised_capability: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.gate_id, "gate_id")
        if not isinstance(self.state, EvidenceState):
            raise ReleaseEvidenceError("evidence state must be EvidenceState")
        _require_reference(self.command, "command", optional=True)
        _require_reference(self.artifact_ref, "artifact_ref", optional=True)
        if self.reason_code is not None:
            _require_identifier(self.reason_code, "reason_code")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ReleaseEvidenceError("exit_code must be an integer or null")
        if type(self.outside_advertised_capability) is not bool:
            raise ReleaseEvidenceError("outside_advertised_capability must be a boolean")
        if self.state is EvidenceState.PASSED:
            if self.command is None or self.exit_code != 0 or self.artifact_ref is None:
                raise ReleaseEvidenceError(
                    "passed evidence requires a command, exit_code=0, and artifact_ref"
                )
            if self.reason_code is not None:
                raise ReleaseEvidenceError("passed evidence cannot have a reason_code")
        elif self.state is EvidenceState.FAILED:
            if (
                self.command is None
                or self.exit_code in (None, 0)
                or self.artifact_ref is None
                or self.reason_code is None
            ):
                raise ReleaseEvidenceError(
                    "failed evidence requires a command, non-zero exit_code, artifact_ref, and reason_code"
                )
        elif self.state in {EvidenceState.BLOCKED, EvidenceState.NOT_RUN}:
            if self.reason_code is None:
                raise ReleaseEvidenceError(
                    f"{self.state.value} evidence requires a reason_code"
                )
        elif self.state is EvidenceState.SKIPPED:
            if not self.outside_advertised_capability or self.reason_code is None:
                raise ReleaseEvidenceError(
                    "skipped evidence is permitted only outside the advertised capability"
                )
        if self.outside_advertised_capability and self.state is not EvidenceState.SKIPPED:
            raise ReleaseEvidenceError(
                "outside_advertised_capability is valid only for skipped evidence"
            )

    @property
    def passed(self) -> bool:
        return self.state is EvidenceState.PASSED

    def to_mapping(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "state": self.state.value,
            "command": self.command,
            "exit_code": self.exit_code,
            "artifact_ref": self.artifact_ref,
            "reason_code": self.reason_code,
            "outside_advertised_capability": self.outside_advertised_capability,
        }


@dataclass(frozen=True, slots=True)
class GateSpec:
    """Declarative source of truth for one release-evidence gate."""

    gate_id: str
    category: str
    scope: str
    owner: str = "kestrel_core"
    advertised: bool = True
    required_for_ready: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.gate_id, "gate_id")
        _require_identifier(self.category, "gate category")
        _require_reference(self.scope, "gate scope")
        _require_identifier(self.owner, "gate owner")
        if type(self.advertised) is not bool or type(self.required_for_ready) is not bool:
            raise ReleaseEvidenceError("gate advertised and required values must be booleans")
        if self.required_for_ready and not self.advertised:
            raise ReleaseEvidenceError("an unadvertised gate cannot be required for ready")

    def initial_evidence(self) -> EvidenceRecord:
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

    def to_mapping(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "category": self.category,
            "scope": self.scope,
            "owner": self.owner,
            "advertised": self.advertised,
            "required_for_ready": self.required_for_ready,
        }


@dataclass(frozen=True, slots=True)
class GateResult:
    """A declared gate and the constrained evidence attached to it."""

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
            raise ReleaseEvidenceError(
                "an unadvertised release gate must remain an explicit skipped capability"
            )

    @property
    def ready(self) -> bool:
        return not self.spec.required_for_ready or self.evidence.passed

    def to_mapping(self) -> dict[str, object]:
        return {**self.spec.to_mapping(), "evidence": self.evidence.to_mapping()}


@dataclass(frozen=True, slots=True)
class StandardsMatrixEntry:
    """One exact offline registry resource pin in the release matrix."""

    identifier: str
    version: str
    maturity: str
    kind: str
    uri: str
    published_date: str
    sha256: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.identifier, "standards identifier")
        _require_reference(self.version, "standards version")
        if self.maturity not in {member.value for member in StandardsMaturity}:
            raise ReleaseEvidenceError("standards maturity is unknown")
        _require_reference(self.kind, "standards kind")
        _require_reference(self.uri, "standards uri")
        _require_reference(self.published_date, "published_date")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ReleaseEvidenceError("standards sha256 must be lowercase sha256")
        if any(not isinstance(capability, str) or not capability for capability in self.capabilities):
            raise ReleaseEvidenceError("standards capabilities must be non-empty strings")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PerformanceBudget:
    """An explicit observed p95 budget; never a process timeout."""

    metric: PerformanceMetric
    samples_ms: tuple[float, ...]
    p95_ms: float
    budget_ms: float
    headroom_fraction: float
    fixture_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.metric, PerformanceMetric):
            raise ReleaseEvidenceError("performance metric must be PerformanceMetric")
        if len(self.samples_ms) < 3 or any(
            not isinstance(sample, (int, float))
            or isinstance(sample, bool)
            or not math.isfinite(sample)
            or sample < 0
            for sample in self.samples_ms
        ):
            raise ReleaseEvidenceError(
                "performance budgets require at least three non-negative samples"
            )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in (self.p95_ms, self.budget_ms, self.headroom_fraction)
        ):
            raise ReleaseEvidenceError("performance values must be numeric")
        if self.p95_ms < 0 or self.budget_ms < self.p95_ms:
            raise ReleaseEvidenceError("performance budget must be at least the observed p95")
        if not 0 <= self.headroom_fraction <= 10:
            raise ReleaseEvidenceError("performance headroom_fraction must be in [0, 10]")
        expected_p95 = sorted(self.samples_ms)[math.ceil(len(self.samples_ms) * 0.95) - 1]
        expected_budget = expected_p95 * (1 + self.headroom_fraction)
        if not math.isclose(self.p95_ms, expected_p95, rel_tol=0.0, abs_tol=1e-9):
            raise ReleaseEvidenceError("performance p95 must match the observed samples")
        if not math.isclose(self.budget_ms, expected_budget, rel_tol=0.0, abs_tol=1e-9):
            raise ReleaseEvidenceError(
                "performance budget must equal p95 plus the declared headroom"
            )
        if not isinstance(self.fixture_digest, str) or not _SHA256_RE.fullmatch(self.fixture_digest):
            raise ReleaseEvidenceError("performance fixture_digest must be sha256")

    @classmethod
    def from_observed(
        cls,
        metric: PerformanceMetric,
        samples_ms: Iterable[float],
        *,
        headroom_fraction: float,
        fixture_description: Mapping[str, object],
    ) -> "PerformanceBudget":
        """Derive p95 plus declared headroom from observed workload samples."""
        if not isinstance(fixture_description, Mapping):
            raise ReleaseEvidenceError("benchmark fixture_description must be a mapping")
        values = tuple(float(value) for value in samples_ms)
        if len(values) < 3:
            raise ReleaseEvidenceError("derive_budget requires at least three samples")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ReleaseEvidenceError("benchmark samples must be finite and non-negative")
        if not 0 <= headroom_fraction <= 10:
            raise ReleaseEvidenceError("headroom_fraction must be in [0, 10]")
        ordered = sorted(values)
        p95 = ordered[math.ceil(len(ordered) * 0.95) - 1]
        return cls(
            metric=metric,
            samples_ms=values,
            p95_ms=p95,
            budget_ms=p95 * (1 + headroom_fraction),
            headroom_fraction=headroom_fraction,
            fixture_digest=_sha256(_canonical_json(dict(fixture_description))),
        )

    def evaluate(self, observed_ms: float) -> EvidenceState:
        """Compare a completed observation to this budget without timing it out."""
        if (
            not isinstance(observed_ms, (int, float))
            or isinstance(observed_ms, bool)
            or not math.isfinite(observed_ms)
            or observed_ms < 0
        ):
            raise ReleaseEvidenceError("observed benchmark result must be non-negative")
        return EvidenceState.PASSED if observed_ms <= self.budget_ms else EvidenceState.FAILED

    def to_mapping(self) -> dict[str, object]:
        return {
            "metric": self.metric.value,
            "samples_ms": list(self.samples_ms),
            "p95_ms": self.p95_ms,
            "budget_ms": self.budget_ms,
            "headroom_fraction": self.headroom_fraction,
            "fixture_digest": self.fixture_digest,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """Reproducible measured samples for one seeded workload fixture."""

    metric: PerformanceMetric
    samples_ms: tuple[float, ...]
    fixture_digest: str
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        if len(self.samples_ms) < 3 or any(
            not isinstance(sample, (int, float))
            or isinstance(sample, bool)
            or not math.isfinite(sample)
            or sample < 0
            for sample in self.samples_ms
        ):
            raise ReleaseEvidenceError(
                "benchmark runs require at least three non-negative samples"
            )
        if not isinstance(self.fixture_digest, str) or not _SHA256_RE.fullmatch(self.fixture_digest):
            raise ReleaseEvidenceError("benchmark fixture_digest must be sha256")
        if not self.environment or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.environment.items()
        ):
            raise ReleaseEvidenceError("benchmark environment must be a non-empty string mapping")

    def to_mapping(self) -> dict[str, object]:
        return {
            "metric": self.metric.value,
            "samples_ms": list(self.samples_ms),
            "fixture_digest": self.fixture_digest,
            "environment": dict(sorted(self.environment.items())),
        }


class SemanticBenchmarkHarness:
    """Repeat a supplied semantic workload; setup and teardown stay with caller."""

    def __init__(self, *, iterations: int = 5) -> None:
        if type(iterations) is not int or iterations < 3:
            raise ReleaseEvidenceError("benchmark iterations must be an integer >= 3")
        self.iterations = iterations

    async def run(
        self,
        metric: PerformanceMetric,
        operation: Callable[[], object | Awaitable[object]],
        *,
        fixture_description: Mapping[str, object],
    ) -> BenchmarkRun:
        if not isinstance(metric, PerformanceMetric):
            raise ReleaseEvidenceError("benchmark metric must be PerformanceMetric")
        if not callable(operation) or not isinstance(fixture_description, Mapping):
            raise ReleaseEvidenceError(
                "benchmark operation must be callable and fixture_description a mapping"
            )
        samples: list[float] = []
        for _ in range(self.iterations):
            started = time.perf_counter()
            result = operation()
            if inspect.isawaitable(result):
                await result
            samples.append((time.perf_counter() - started) * 1_000)
        return BenchmarkRun(
            metric=metric,
            samples_ms=tuple(samples),
            fixture_digest=_sha256(_canonical_json(dict(fixture_description))),
            environment={
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
            },
        )


@dataclass(frozen=True, slots=True)
class CompatibilityRetirementDecision:
    """Telemetry-gated compatibility deletion decision, separate from release pass."""

    path_id: str
    observed_window_ref: str | None
    inventory_complete: bool
    unmigrated_eligible_rows: int | None
    required_consumer_count: int | None
    migration_equivalence: EvidenceRecord

    def __post_init__(self) -> None:
        _require_identifier(self.path_id, "compatibility path_id")
        _require_reference(self.observed_window_ref, "observed_window_ref", optional=True)
        if type(self.inventory_complete) is not bool:
            raise ReleaseEvidenceError("inventory_complete must be a boolean")
        for value, field_name in (
            (self.unmigrated_eligible_rows, "unmigrated eligible rows"),
            (self.required_consumer_count, "required consumer count"),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ReleaseEvidenceError(f"{field_name} must be a non-negative integer or null")
        if not isinstance(self.migration_equivalence, EvidenceRecord):
            raise ReleaseEvidenceError("migration_equivalence must be EvidenceRecord")

    @property
    def removal_safe(self) -> bool:
        return (
            self.observed_window_ref is not None
            and self.inventory_complete
            and self.unmigrated_eligible_rows == 0
            and self.required_consumer_count == 0
            and self.migration_equivalence.passed
        )

    @property
    def decision(self) -> str:
        return "eligible_for_review" if self.removal_safe else "retain"

    @property
    def reason_code(self) -> str:
        if self.removal_safe:
            return "telemetry_and_equivalence_observed"
        if self.observed_window_ref is None:
            return "telemetry_not_observed"
        if not self.inventory_complete:
            return "inventory_incomplete"
        if self.unmigrated_eligible_rows != 0:
            return "eligible_rows_remain"
        if self.required_consumer_count != 0:
            return "required_consumers_remain"
        return "migration_equivalence_not_observed"

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
    """Cross-repository report identity, with no feature import into core."""

    capability_id: str
    repository: str
    source_revision: str
    gate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.capability_id, "external capability_id")
        _require_reference(self.repository, "external repository")
        if not isinstance(self.source_revision, str) or not _COMMIT_RE.fullmatch(
            self.source_revision
        ):
            raise ReleaseEvidenceError("external source_revision must be a commit SHA")
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
