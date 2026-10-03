"""De-identification evidence artifacts.

Every de-identification run produces one evidence artifact. It is the audit
record ``audit=required`` asks for, and the only thing that authorizes a
de-identified save or export. It records:

* the method (Safe Harbor or Expert Determination) and the matching assurance
  value from the privacy-config vocabulary;
* keyed digests of each source record and its identifier — never the content;
* a plain SHA-256 of each de-identified output record, which binds the artifact
  to exactly the bytes it describes;
* per record and per identifier category, which fields were removed,
  generalized, or transformed, and by which detector;
* the pipeline name, version, and configuration digest, the policy version,
  the run timestamp, and the operator/request context;
* for Safe Harbor, the operator's actual-knowledge attestation; for Expert
  Determination, the reference and digest of the expert's report.

The artifact carries its own SHA-256 (``artifact_digest``) over its canonical
form. That is an integrity digest, not a signature: it exposes corruption and
an edit that did not also recompute it, nothing more. Validation is strict and
fails closed: an artifact that is incomplete, internally inconsistent, from an
unknown pipeline, or presented with records it does not describe is rejected.

What validation cannot do is tell a genuine artifact from one assembled by
code running in the same process, which can call the same constructors. As
everywhere in the privacy layer, the guarantee rests on content — the artifact
must describe exactly the records being saved, and those records are scanned
for residual identifiers — not on who built the object.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from kestrel_sovereign.audit_time import utc_now_iso
from kestrel_sovereign.deidentification.errors import EvidenceValidationError
from kestrel_sovereign.deidentification.identifiers import (
    SafeHarborIdentifier,
    TransformationAction,
)

EVIDENCE_SCHEMA = "kestrel.deidentification.evidence/v1"
PIPELINE_NAME = "kestrel.deidentification"
PIPELINE_VERSION = "1.0.0"
SUPPORTED_PIPELINE_VERSIONS = frozenset({PIPELINE_VERSION})

#: How far an attestation may be dated after the run it covers, to absorb
#: clock skew between the operator's host and the pipeline's.
ATTESTATION_CLOCK_SKEW = timedelta(minutes=5)

#: The rule set the pipeline implements: the Safe Harbor identifier list of
#: 45 CFR 164.514(b)(2)(i) with the (ii) actual-knowledge condition.
SAFE_HARBOR_POLICY_VERSION = "hipaa-safe-harbor-45cfr164.514(b)(2)/v1"
SUPPORTED_POLICY_VERSIONS = frozenset({SAFE_HARBOR_POLICY_VERSION})

#: The exact statement a Safe Harbor attestation affirms. The pipeline never
#: makes this statement itself; an operator supplies the attestation.
ACTUAL_KNOWLEDGE_STATEMENT = (
    "The attesting operator has no actual knowledge that the information "
    "remaining after de-identification could be used, alone or in combination "
    "with other information, to identify an individual who is a subject of "
    "the information."
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_ID = re.compile(r"^[0-9a-f]{32}$")
_KEY_ID = re.compile(r"^[0-9a-f]{16,64}$")


class DeidentificationMethod(Enum):
    """The two HIPAA de-identification methods.

    Values equal the ``PrivacyConfig.assurance`` vocabulary, so a feature can
    compare ``evidence.assurance`` with a required assurance generically.
    """

    SAFE_HARBOR = "safe_harbor"
    EXPERT_DETERMINATION = "expert_determination"


#: Assurance values that only an evidence artifact can back.
DEIDENTIFICATION_ASSURANCES = frozenset(m.value for m in DeidentificationMethod)


def canonical_json_bytes(value: Any) -> bytes:
    """The deterministic encoding every digest in this module is taken over."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError(
            f"value is not canonically encodable: {type(exc).__name__}"
        ) from None


def output_record_digest(record: Mapping[str, Any]) -> str:
    """SHA-256 of a de-identified output record's canonical encoding."""
    return hashlib.sha256(canonical_json_bytes(dict(record))).hexdigest()


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceValidationError(f"{name} must be a non-empty string")
    return value


def _require_hex64(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        raise EvidenceValidationError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def parse_timestamp(value: Any, name: str) -> datetime:
    """Parse an artifact timestamp, which must be ISO-8601 with an offset."""
    _require_text(value, name)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise EvidenceValidationError(f"{name} must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None:
        raise EvidenceValidationError(f"{name} must carry a UTC offset")
    return parsed


def _require_count(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _enum(enum_type, value: Any, name: str):
    try:
        return enum_type(value)
    except ValueError:
        raise EvidenceValidationError(f"{name} {value!r} is not recognized") from None


@dataclass(frozen=True)
class OperatorContext:
    """Who requested the run, and under which request."""

    operator_id: str
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"operator_id": self.operator_id, "request_id": self.request_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatorContext":
        return cls(operator_id=data["operator_id"], request_id=data.get("request_id"))

    def validate(self) -> None:
        _require_text(self.operator_id, "operator.operator_id")
        if self.request_id is not None:
            _require_text(self.request_id, "operator.request_id")


@dataclass(frozen=True)
class ActualKnowledgeAttestation:
    """The operator's Safe Harbor (b)(2)(ii) attestation.

    The pipeline cannot know what the operator knows, so it never fills this in:
    a Safe Harbor run without one is refused. ``no_actual_knowledge`` must be
    passed explicitly as ``True``; the attestation affirms
    :data:`ACTUAL_KNOWLEDGE_STATEMENT` verbatim.
    """

    attested_by: str
    no_actual_knowledge: bool
    attested_at: str = field(default_factory=utc_now_iso)
    statement: str = ACTUAL_KNOWLEDGE_STATEMENT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attested_by": self.attested_by,
            "attested_at": self.attested_at,
            "no_actual_knowledge": self.no_actual_knowledge,
            "statement": self.statement,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActualKnowledgeAttestation":
        return cls(
            attested_by=data["attested_by"],
            no_actual_knowledge=data["no_actual_knowledge"],
            attested_at=data["attested_at"],
            statement=data["statement"],
        )

    def validate(self) -> None:
        _require_text(self.attested_by, "attestation.attested_by")
        parse_timestamp(self.attested_at, "attestation.attested_at")
        if self.no_actual_knowledge is not True:
            raise EvidenceValidationError(
                "attestation.no_actual_knowledge must be explicitly true"
            )
        if self.statement != ACTUAL_KNOWLEDGE_STATEMENT:
            raise EvidenceValidationError(
                "attestation.statement must be the Safe Harbor actual-knowledge statement"
            )


@dataclass(frozen=True)
class ExpertDeterminationReference:
    """The qualified expert's documented determination an Expert Determination
    run rests on. The pipeline never claims Expert Determination without it."""

    report_reference: str
    report_digest: str
    expert_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_reference": self.report_reference,
            "report_digest": self.report_digest,
            "expert_id": self.expert_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpertDeterminationReference":
        return cls(
            report_reference=data["report_reference"],
            report_digest=data["report_digest"],
            expert_id=data["expert_id"],
        )

    def validate(self) -> None:
        _require_text(self.report_reference, "expert_determination.report_reference")
        _require_hex64(self.report_digest, "expert_determination.report_digest")
        _require_text(self.expert_id, "expert_determination.expert_id")


@dataclass(frozen=True)
class FieldTransformation:
    """Identifier occurrences in one field handled one way by one detector."""

    field: str
    category: SafeHarborIdentifier
    action: TransformationAction
    detector: str
    occurrences: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "category": self.category.value,
            "action": self.action.value,
            "detector": self.detector,
            "occurrences": self.occurrences,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FieldTransformation":
        return cls(
            field=data["field"],
            category=_enum(SafeHarborIdentifier, data["category"], "category"),
            action=_enum(TransformationAction, data["action"], "action"),
            detector=data["detector"],
            occurrences=data["occurrences"],
        )

    def validate(self) -> None:
        _require_text(self.field, "transformation.field")
        _require_text(self.detector, "transformation.detector")
        _require_count(self.occurrences, "transformation.occurrences", minimum=1)


@dataclass(frozen=True)
class RecordEvidence:
    """What happened to one source record."""

    source_id_digest: str
    source_digest: str
    output_digest: str
    transformations: Tuple[FieldTransformation, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id_digest": self.source_id_digest,
            "source_digest": self.source_digest,
            "output_digest": self.output_digest,
            "transformations": [t.to_dict() for t in self.transformations],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecordEvidence":
        return cls(
            source_id_digest=data["source_id_digest"],
            source_digest=data["source_digest"],
            output_digest=data["output_digest"],
            transformations=tuple(
                FieldTransformation.from_dict(t) for t in data["transformations"]
            ),
        )

    def validate(self) -> None:
        _require_hex64(self.source_id_digest, "record.source_id_digest")
        _require_hex64(self.source_digest, "record.source_digest")
        _require_hex64(self.output_digest, "record.output_digest")
        for transformation in self.transformations:
            transformation.validate()


@dataclass(frozen=True)
class CategoryDisposition:
    """The run's handling of one Safe Harbor category across every record.

    ``fields_classified`` lists the schema fields declared as this category, so
    a reviewer can see a category was considered even when no value of it was
    present to remove.
    """

    category: SafeHarborIdentifier
    fields_classified: Tuple[str, ...]
    removed: int
    generalized: int
    transformed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "fields_classified": list(self.fields_classified),
            "removed": self.removed,
            "generalized": self.generalized,
            "transformed": self.transformed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CategoryDisposition":
        return cls(
            category=_enum(SafeHarborIdentifier, data["category"], "category"),
            fields_classified=tuple(data["fields_classified"]),
            removed=data["removed"],
            generalized=data["generalized"],
            transformed=data["transformed"],
        )

    def count(self, action: TransformationAction) -> int:
        return {
            TransformationAction.REMOVED: self.removed,
            TransformationAction.GENERALIZED: self.generalized,
            TransformationAction.TRANSFORMED: self.transformed,
        }[action]


def summarize_categories(
    records: Sequence[RecordEvidence],
    fields_by_category: Mapping[SafeHarborIdentifier, Sequence[str]],
) -> Tuple[CategoryDisposition, ...]:
    """One disposition per Safe Harbor category, in the regulation's order."""
    totals: Dict[Tuple[SafeHarborIdentifier, TransformationAction], int] = {}
    for record in records:
        for t in record.transformations:
            key = (t.category, t.action)
            totals[key] = totals.get(key, 0) + t.occurrences
    return tuple(
        CategoryDisposition(
            category=category,
            fields_classified=tuple(sorted(fields_by_category.get(category, ()))),
            removed=totals.get((category, TransformationAction.REMOVED), 0),
            generalized=totals.get((category, TransformationAction.GENERALIZED), 0),
            transformed=totals.get((category, TransformationAction.TRANSFORMED), 0),
        )
        for category in SafeHarborIdentifier
    )


@dataclass(frozen=True)
class DeidentificationEvidence:
    """The evidence artifact for one de-identification run."""

    evidence_id: str
    method: DeidentificationMethod
    created_at: str
    reference_date: str
    policy_version: str
    pipeline_name: str
    pipeline_version: str
    config_digest: str
    detectors: Tuple[str, ...]
    source_digest_key_id: str
    operator: OperatorContext
    records: Tuple[RecordEvidence, ...]
    categories: Tuple[CategoryDisposition, ...]
    attestation: Optional[ActualKnowledgeAttestation]
    expert_determination: Optional[ExpertDeterminationReference]
    artifact_digest: str

    @property
    def assurance(self) -> str:
        """The ``PrivacyConfig.assurance`` value this artifact backs."""
        return self.method.value

    def _body(self) -> Dict[str, Any]:
        return {
            "schema": EVIDENCE_SCHEMA,
            "evidence_id": self.evidence_id,
            "method": self.method.value,
            "assurance": self.assurance,
            "created_at": self.created_at,
            "reference_date": self.reference_date,
            "policy_version": self.policy_version,
            "pipeline": {
                "name": self.pipeline_name,
                "version": self.pipeline_version,
                "config_digest": self.config_digest,
                "detectors": list(self.detectors),
            },
            "source_digest": {
                "algorithm": "hmac-sha256",
                "key_id": self.source_digest_key_id,
            },
            "operator": self.operator.to_dict(),
            "records": [r.to_dict() for r in self.records],
            "identifier_categories": [c.to_dict() for c in self.categories],
            "attestation": self.attestation.to_dict() if self.attestation else None,
            "expert_determination": (
                self.expert_determination.to_dict()
                if self.expert_determination
                else None
            ),
        }

    def compute_digest(self) -> str:
        """SHA-256 over the canonical artifact, excluding ``artifact_digest``."""
        return hashlib.sha256(canonical_json_bytes(self._body())).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        body = self._body()
        body["artifact_digest"] = self.artifact_digest
        return body

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeidentificationEvidence":
        """Rebuild a persisted artifact. Call :func:`validate_evidence` on it."""
        try:
            if data["schema"] != EVIDENCE_SCHEMA:
                raise EvidenceValidationError(
                    f"unsupported evidence schema {data['schema']!r}"
                )
            method = _enum(DeidentificationMethod, data["method"], "method")
            if data["assurance"] != method.value:
                raise EvidenceValidationError("assurance does not match method")
            if data["source_digest"]["algorithm"] != "hmac-sha256":
                raise EvidenceValidationError("unsupported source digest algorithm")
            pipeline = data["pipeline"]
            attestation = data["attestation"]
            expert = data["expert_determination"]
            return cls(
                evidence_id=data["evidence_id"],
                method=method,
                created_at=data["created_at"],
                reference_date=data["reference_date"],
                policy_version=data["policy_version"],
                pipeline_name=pipeline["name"],
                pipeline_version=pipeline["version"],
                config_digest=pipeline["config_digest"],
                detectors=tuple(pipeline["detectors"]),
                source_digest_key_id=data["source_digest"]["key_id"],
                operator=OperatorContext.from_dict(data["operator"]),
                records=tuple(RecordEvidence.from_dict(r) for r in data["records"]),
                categories=tuple(
                    CategoryDisposition.from_dict(c)
                    for c in data["identifier_categories"]
                ),
                attestation=(
                    ActualKnowledgeAttestation.from_dict(attestation)
                    if attestation is not None
                    else None
                ),
                expert_determination=(
                    ExpertDeterminationReference.from_dict(expert)
                    if expert is not None
                    else None
                ),
                artifact_digest=data["artifact_digest"],
            )
        except (KeyError, TypeError) as exc:
            raise EvidenceValidationError(
                f"evidence artifact is malformed: missing or mistyped {exc}"
            ) from None

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "DeidentificationEvidence":
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise EvidenceValidationError("evidence artifact is not valid JSON") from None
        if not isinstance(data, dict):
            raise EvidenceValidationError("evidence artifact must be a JSON object")
        return cls.from_dict(data)


def _validate_method(evidence: DeidentificationEvidence) -> None:
    if evidence.method is DeidentificationMethod.SAFE_HARBOR:
        if evidence.attestation is None:
            raise EvidenceValidationError(
                "a Safe Harbor artifact requires the operator's actual-knowledge "
                "attestation"
            )
        if evidence.expert_determination is not None:
            raise EvidenceValidationError(
                "a Safe Harbor artifact must not cite an Expert Determination report"
            )
        evidence.attestation.validate()
        _validate_attestation_timing(evidence)
    else:
        if evidence.expert_determination is None:
            raise EvidenceValidationError(
                "an Expert Determination artifact requires the expert report reference"
            )
        evidence.expert_determination.validate()
        if evidence.attestation is not None:
            evidence.attestation.validate()
            _validate_attestation_timing(evidence)


def _validate_attestation_timing(evidence: DeidentificationEvidence) -> None:
    attested_at = parse_timestamp(evidence.attestation.attested_at, "attestation.attested_at")
    created_at = parse_timestamp(evidence.created_at, "created_at")
    if attested_at > created_at + ATTESTATION_CLOCK_SKEW:
        raise EvidenceValidationError("attestation is dated after the run it covers")


def _validate_categories(evidence: DeidentificationEvidence) -> None:
    if tuple(c.category for c in evidence.categories) != tuple(SafeHarborIdentifier):
        raise EvidenceValidationError(
            "identifier_categories must list all eighteen Safe Harbor categories "
            "exactly once, in order"
        )
    totals: Dict[Tuple[SafeHarborIdentifier, TransformationAction], int] = {}
    for record in evidence.records:
        for t in record.transformations:
            key = (t.category, t.action)
            totals[key] = totals.get(key, 0) + t.occurrences
    for disposition in evidence.categories:
        for name in disposition.fields_classified:
            _require_text(name, "identifier_categories.fields_classified")
        for action in TransformationAction:
            count = _require_count(
                disposition.count(action),
                f"identifier_categories[{disposition.category.value}].{action.value}",
            )
            if count != totals.get((disposition.category, action), 0):
                raise EvidenceValidationError(
                    f"identifier_categories[{disposition.category.value}] "
                    f"{action.value} count disagrees with the record transformations"
                )


def validate_evidence(
    evidence: DeidentificationEvidence,
    records: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    required_assurance: Optional[str] = None,
) -> None:
    """Fail closed unless ``evidence`` is a complete, self-consistent artifact.

    When ``records`` is given, the artifact must describe exactly those output
    records, in order. When ``required_assurance`` is given (``"safe_harbor"``
    or ``"expert_determination"``), the artifact must back that assurance; this
    is the generic check a feature uses to require a de-identification method
    without naming a domain-specific privacy mode.
    """
    if not isinstance(evidence, DeidentificationEvidence):
        raise EvidenceValidationError("evidence must be a DeidentificationEvidence")
    if required_assurance is not None:
        if required_assurance not in DEIDENTIFICATION_ASSURANCES:
            raise EvidenceValidationError(
                f"{required_assurance!r} is not an assurance an evidence artifact "
                f"can back; expected one of {sorted(DEIDENTIFICATION_ASSURANCES)}"
            )
        if evidence.assurance != required_assurance:
            raise EvidenceValidationError(
                f"evidence backs {evidence.assurance!r}, not the required "
                f"{required_assurance!r}"
            )
    if not isinstance(evidence.evidence_id, str) or not _EVIDENCE_ID.match(evidence.evidence_id):
        raise EvidenceValidationError("evidence_id must be a 32-character hex identifier")
    parse_timestamp(evidence.created_at, "created_at")
    _require_text(evidence.reference_date, "reference_date")
    try:
        date.fromisoformat(evidence.reference_date)
    except ValueError:
        raise EvidenceValidationError("reference_date must be an ISO date") from None
    if evidence.policy_version not in SUPPORTED_POLICY_VERSIONS:
        raise EvidenceValidationError(
            f"policy_version {evidence.policy_version!r} is not supported"
        )
    if evidence.pipeline_name != PIPELINE_NAME:
        raise EvidenceValidationError(
            f"pipeline.name {evidence.pipeline_name!r} is not the Kestrel "
            "de-identification pipeline"
        )
    if evidence.pipeline_version not in SUPPORTED_PIPELINE_VERSIONS:
        raise EvidenceValidationError(
            f"pipeline.version {evidence.pipeline_version!r} is not supported"
        )
    _require_hex64(evidence.config_digest, "pipeline.config_digest")
    if not evidence.detectors:
        raise EvidenceValidationError("pipeline.detectors must not be empty")
    for name in evidence.detectors:
        _require_text(name, "pipeline.detectors")
    if not isinstance(evidence.source_digest_key_id, str) or not _KEY_ID.match(
        evidence.source_digest_key_id
    ):
        raise EvidenceValidationError("source_digest.key_id must be a hex key id")
    evidence.operator.validate()
    _validate_method(evidence)
    if not evidence.records:
        raise EvidenceValidationError("an evidence artifact must cover at least one record")
    for record in evidence.records:
        record.validate()
        for t in record.transformations:
            if t.detector not in evidence.detectors:
                raise EvidenceValidationError(
                    f"transformation detector {t.detector!r} is not a recorded "
                    "pipeline detector"
                )
    _validate_categories(evidence)
    if evidence.artifact_digest != evidence.compute_digest():
        raise EvidenceValidationError(
            "artifact_digest does not match the artifact; it was altered after "
            "the run"
        )
    if records is not None:
        if len(records) != len(evidence.records):
            raise EvidenceValidationError(
                f"evidence covers {len(evidence.records)} record(s), "
                f"{len(records)} presented"
            )
        for index, (record, record_evidence) in enumerate(zip(records, evidence.records, strict=True)):
            if output_record_digest(record) != record_evidence.output_digest:
                raise EvidenceValidationError(
                    f"record {index} does not match the output digest its "
                    "evidence recorded"
                )
