"""The de-identification pipeline.

A :class:`DeidentificationPipeline` is built from a field schema that
classifies every field a record may carry (see
:class:`~kestrel_sovereign.deidentification.identifiers.FieldSpec`). A run
transforms a batch of records and returns a :class:`DeidentificationResult`:
the de-identified records together with the evidence artifact that describes
them. There is no way to obtain de-identified output without its artifact.

What a run does, per field:

* **identifier** fields are removed, or generalized where Safe Harbor permits
  it (ZIP code to its three-digit prefix or ``000``; dates to their year; ages
  over 89, and dates 90 or more years old, to ``90+``);
* **free-text** fields have every detectable identifier span removed or
  generalized — the record's own identifier values, every pattern-detectable
  category, and names/places via a named-entity detector;
* **non-identifying** fields are kept, but a value that matches a precise
  identifier pattern (or repeats one of the record's own identifier values,
  or gives a birth date Safe Harbor does not permit) refutes the
  classification and the run is refused.

This is distinct from PII redaction (``storage="pii_redacted"``): redaction is
best-effort masking of obvious identifiers, with no evidence and no claim. A
Safe Harbor run also needs a statement the pipeline cannot make, the operator's
attestation of no actual knowledge; without it the run is refused and no
artifact exists to authorize a save.
"""

import dataclasses
import hashlib
import hmac
import math
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from kestrel_sovereign.deidentification.detectors import (
    DETECTOR_NAMES,
    SCHEMA_DETECTOR,
    EntityDetector,
    default_entity_detector,
    find_identifier_patterns,
    parse_date_year,
    scrub_free_text,
)
from kestrel_sovereign.deidentification.errors import (
    DeidentificationConfigError,
    DeidentificationError,
    DeidentificationRefused,
    EvidenceValidationError,
)
from kestrel_sovereign.deidentification.evidence import (
    ATTESTATION_CLOCK_SKEW,
    PIPELINE_NAME,
    PIPELINE_VERSION,
    SAFE_HARBOR_POLICY_VERSION,
    ActualKnowledgeAttestation,
    DeidentificationEvidence,
    DeidentificationMethod,
    ExpertDeterminationReference,
    FieldTransformation,
    OperatorContext,
    RecordEvidence,
    canonical_json_bytes,
    load_json_object,
    output_record_digest,
    parse_timestamp,
    summarize_categories,
    validate_evidence,
)
from kestrel_sovereign.deidentification.identifiers import (
    AGE_90_OR_OLDER,
    HHS_RESTRICTED_ZIP3_PREFIXES,
    FieldKind,
    FieldRole,
    FieldSpec,
    SafeHarborIdentifier,
    TransformationAction,
)

EXPORT_SCHEMA = "kestrel.deidentification.export/v1"

#: How long before a run an operator's attestation may have been made. The
#: actual-knowledge condition concerns the data being released, so a standing
#: attestation reused across batches is refused once it is this old.
DEFAULT_ATTESTATION_MAX_AGE = timedelta(hours=24)

#: Minimum length of the key that digests source records. Source records hold
#: low-entropy identifiers (an SSN, an MRN); an unkeyed hash of one can be
#: reversed by enumeration, so the artifact carries keyed digests only.
MIN_SOURCE_DIGEST_KEY_BYTES = 32

_SOURCE_DOMAIN = b"kestrel.deidentification.source-record/v1"
_SOURCE_ID_DOMAIN = b"kestrel.deidentification.source-record-id/v1"
_KEY_ID_DOMAIN = b"kestrel.deidentification.key-id/v1"
_USE_DEFAULT = object()
_DROP = object()
_ZIP_RE = re.compile(r"^\s*(\d{5})(?:-?\d{4})?\s*$")
_YEAR_RE = re.compile(r"^\s*(\d{4})\s*$")
# Hyphenated parts stay whole ("Jane-Marie"); the closing known-value pass
# matches on word boundaries, so it also finds them within longer text.
_NAME_TOKEN_SPLIT = re.compile(r"[^\w'-]+")
_SCALAR_TYPES = (str, int, float, bool, type(None))
_CATEGORY_ORDER = {category: index for index, category in enumerate(SafeHarborIdentifier)}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclasses.dataclass(frozen=True)
class SourceRecord:
    """One record to de-identify.

    ``record_id`` is the caller's reference to the source (it may itself be an
    identifier, such as a record number); the artifact stores only its keyed
    digest.
    """

    record_id: str
    fields: Mapping[str, Any]


def _scannable_text(value: Any) -> Optional[str]:
    """The text form of a value the identifier patterns should see."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _artifact_field_names(evidence: DeidentificationEvidence) -> List[str]:
    """Every field name an artifact carries, in a stable order."""
    names = {t.field for record in evidence.records for t in record.transformations}
    names.update(name for c in evidence.categories for name in c.fields_classified)
    return sorted(names)


def _identifier_bearing_name(
    names: Iterable[str], reference_year: int
) -> Optional[Tuple[int, SafeHarborIdentifier]]:
    """(position, category) of the first name carrying an identifier pattern.

    A field name is written into the evidence artifact and the output records,
    so one such as ``patient.jane@example.com`` would export the identifier
    with them. Callers report the position, never the name itself.
    """
    for position, name in enumerate(names):
        findings = find_identifier_patterns(name, reference_year=reference_year)
        if findings:
            return position, findings[0][0]
    return None


def verify_deidentified_records(
    evidence: DeidentificationEvidence,
    records: Sequence[Mapping[str, Any]],
    *,
    required_assurance: Optional[str] = None,
) -> None:
    """Fail closed unless ``evidence`` authorizes exactly ``records``.

    Validates the artifact against the records, then scans the records for
    residual identifiers with the same precise patterns that refute a
    non-identifying classification, measuring ages against the artifact's
    reference date. The pipeline scrubs text until those patterns no longer
    match, so a hit means the records are not the output the artifact
    describes. Raises :class:`EvidenceValidationError`.
    """
    validate_evidence(evidence, records, required_assurance=required_assurance)
    reference_year = date.fromisoformat(evidence.reference_date).year
    # The artifact names every classified field, including the ones the output
    # dropped, and is exported and stored with the records.
    if _identifier_bearing_name(_artifact_field_names(evidence), reference_year) is not None:
        raise EvidenceValidationError(
            "the evidence artifact names a field carrying an identifier pattern"
        )
    # Messages name a field by position: the records may come from an imported
    # bundle, whose field names are as untrusted as its values.
    for index, record in enumerate(records):
        for position, (field_name, value) in enumerate(record.items()):
            # Keys are serialized with the values, so they are scanned too.
            if not isinstance(field_name, str):
                raise EvidenceValidationError(
                    f"record {index} field {position} has a non-string name"
                )
            if find_identifier_patterns(field_name, reference_year=reference_year):
                raise EvidenceValidationError(
                    f"record {index} field {position} has a field name carrying "
                    "an identifier pattern"
                )
            if not isinstance(value, _SCALAR_TYPES):
                raise EvidenceValidationError(
                    f"record {index} field {position} is not a JSON scalar"
                )
            text = _scannable_text(value)
            if text is not None:
                findings = find_identifier_patterns(text, reference_year=reference_year)
                if findings:
                    category, detector = findings[0]
                    raise EvidenceValidationError(
                        f"record {index} field {position} still carries a "
                        f"{category.value} identifier ({detector})"
                    )


@dataclasses.dataclass(frozen=True)
class DeidentificationResult:
    """De-identified records and the evidence artifact that authorizes them.

    Construction verifies the artifact against the records, so a result whose
    evidence does not describe its records cannot exist. The records are
    read-only mappings of JSON scalars.
    """

    records: Tuple[Mapping[str, Any], ...]
    evidence: DeidentificationEvidence

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "records",
            tuple(MappingProxyType(dict(record)) for record in self.records),
        )
        self.verify()

    def verify(self, *, required_assurance: Optional[str] = None) -> None:
        """Re-check this result; see :func:`verify_deidentified_records`."""
        verify_deidentified_records(
            self.evidence, self.records, required_assurance=required_assurance
        )

    def records_as_dicts(self) -> List[Dict[str, Any]]:
        return [dict(record) for record in self.records]

    def export_bundle(self) -> bytes:
        """The canonical export of these records with their evidence artifact.

        An export never leaves without its artifact: the bundle is produced only
        after :meth:`verify` passes, and it embeds the artifact verbatim. The
        bytes themselves are then read back with :meth:`from_export_bundle`, so
        what leaves is verified as the bytes a recipient will parse, not only as
        the in-process objects that produced them.
        """
        self.verify()
        payload = canonical_json_bytes({
            "schema": EXPORT_SCHEMA,
            "evidence": self.evidence.to_dict(),
            "records": self.records_as_dicts(),
        })
        DeidentificationResult.from_export_bundle(payload)
        return payload

    @classmethod
    def from_export_bundle(
        cls, payload: bytes, *, required_assurance: Optional[str] = None
    ) -> "DeidentificationResult":
        """Rebuild a result from :meth:`export_bundle` bytes, failing closed.

        The bytes must be exactly what :meth:`export_bundle` writes for the
        result they parse to: canonical, and carrying no field the bundle or
        its artifact does not define. The embedded artifact must validate and
        describe exactly the embedded records, which must carry no residual
        identifier pattern. Raises :class:`EvidenceValidationError`.
        """
        data = load_json_object(payload, "export bundle")
        if set(data) != {"schema", "evidence", "records"}:
            raise EvidenceValidationError(
                "an export bundle holds exactly a schema, evidence, and records"
            )
        if data["schema"] != EXPORT_SCHEMA:
            raise EvidenceValidationError("unsupported export bundle schema")
        records, evidence = data["records"], data["evidence"]
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise EvidenceValidationError("export bundle records must be a list of objects")
        if not isinstance(evidence, dict):
            raise EvidenceValidationError("export bundle evidence must be an object")
        result = cls(tuple(records), DeidentificationEvidence.from_dict(evidence))
        # Parsing drops what the artifact does not define; re-encoding what was
        # parsed and validated must give back every byte that was presented.
        if canonical_json_bytes({
            "schema": EXPORT_SCHEMA,
            "evidence": result.evidence.to_dict(),
            "records": result.records_as_dicts(),
        }) != payload:
            raise EvidenceValidationError(
                "export bundle carries content its schema does not define"
            )
        if required_assurance is not None:
            result.verify(required_assurance=required_assurance)
        return result


def _encode_source_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    if isinstance(value, bytes):
        return {"$bytes_sha256": hashlib.sha256(value).hexdigest()}
    return value


def _parse_year(value: Any) -> Optional[int]:
    if isinstance(value, (datetime, date)):
        return value.year
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 1000 <= value <= 9999 else None
    if isinstance(value, str):
        match = _YEAR_RE.match(value)
        if match:
            return int(match.group(1))
        text = value.strip()
        for parse in (date.fromisoformat, datetime.fromisoformat):
            try:
                return parse(text).year
            except ValueError:
                continue
        return parse_date_year(text)
    return None


def _parse_zip(value: Any) -> Optional[str]:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:05d}" if 0 <= value <= 99999 else None
    if isinstance(value, str):
        match = _ZIP_RE.match(value)
        if match:
            return match.group(1)
    return None


def _parse_age(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


class DeidentificationPipeline:
    """Schema-driven de-identification with a per-run evidence artifact."""

    def __init__(
        self,
        schema: Mapping[str, FieldSpec],
        *,
        source_digest_key: bytes,
        entity_detector: Union[EntityDetector, None, object] = _USE_DEFAULT,
        restricted_zip3_prefixes: Iterable[str] = HHS_RESTRICTED_ZIP3_PREFIXES,
        attestation_max_age: timedelta = DEFAULT_ATTESTATION_MAX_AGE,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """
        Args:
            schema: Every field a record may carry, mapped to its classification.
            source_digest_key: Secret (at least 32 bytes) keying the source
                digests in the artifact. Keep it to re-derive a digest from a
                source record; it is never written to the artifact.
            entity_detector: Finds names and places in free text. Defaults to
                the spaCy model when installed. A schema with free-text fields
                is refused without one, because Safe Harbor category (A) cannot
                be covered by patterns alone.
            restricted_zip3_prefixes: Three-digit ZIP prefixes reported as
                ``000``. Defaults to the HHS guidance list.
            attestation_max_age: How long before a run its attestation may
                have been made; an older (reused) attestation is refused.
            clock: Source of the run timestamp (timezone-aware).
        """
        if not isinstance(schema, Mapping) or not schema:
            raise DeidentificationConfigError("the field schema must be a non-empty mapping")
        for name, spec in schema.items():
            if not isinstance(name, str) or not name:
                raise DeidentificationConfigError("schema field names must be non-empty strings")
            if not isinstance(spec, FieldSpec):
                raise DeidentificationConfigError(
                    f"schema field {name!r} must map to a FieldSpec"
                )
        if not isinstance(source_digest_key, bytes) or len(source_digest_key) < MIN_SOURCE_DIGEST_KEY_BYTES:
            raise DeidentificationConfigError(
                f"source_digest_key must be at least {MIN_SOURCE_DIGEST_KEY_BYTES} bytes"
            )
        prefixes = frozenset(restricted_zip3_prefixes)
        if any(not (isinstance(p, str) and len(p) == 3 and p.isdigit()) for p in prefixes):
            raise DeidentificationConfigError(
                "restricted ZIP prefixes must be three-digit strings"
            )
        if not isinstance(attestation_max_age, timedelta) or attestation_max_age <= timedelta(0):
            raise DeidentificationConfigError("attestation_max_age must be a positive timedelta")
        unsafe = _identifier_bearing_name(schema, clock().year)
        if unsafe is not None:
            raise DeidentificationConfigError(
                f"schema field {unsafe[0]} has a name carrying a "
                f"{unsafe[1].value} identifier; field names are written into "
                "the evidence artifact"
            )
        has_free_text = any(spec.kind is FieldKind.FREE_TEXT for spec in schema.values())
        if entity_detector is _USE_DEFAULT:
            entity_detector = default_entity_detector() if has_free_text else None
        if has_free_text and entity_detector is None:
            raise DeidentificationConfigError(
                "free-text fields need a named-entity detector: patterns alone "
                "cannot find names (Safe Harbor category A). Install the spaCy "
                "model or pass entity_detector=..."
            )
        detectors = DETECTOR_NAMES
        if entity_detector is not None:
            name = getattr(entity_detector, "name", None)
            if not isinstance(name, str) or not name.strip() or name in DETECTOR_NAMES:
                raise DeidentificationConfigError(
                    "the entity detector needs a distinct, non-empty name"
                )
            detectors = detectors + (name,)

        self._schema: Dict[str, FieldSpec] = dict(schema)
        self._key = source_digest_key
        self._entity_detector = entity_detector
        self._restricted_zip3 = prefixes
        self._clock = clock
        self._attestation_max_age = attestation_max_age
        self._detectors: Tuple[str, ...] = detectors
        self._key_id = hmac.new(source_digest_key, _KEY_ID_DOMAIN, hashlib.sha256).hexdigest()[:32]
        self._config_digest = hashlib.sha256(canonical_json_bytes({
            "pipeline": PIPELINE_NAME,
            "version": PIPELINE_VERSION,
            "policy_version": SAFE_HARBOR_POLICY_VERSION,
            "schema": {name: self._schema[name].to_dict() for name in sorted(self._schema)},
            "restricted_zip3_prefixes": sorted(prefixes),
            "attestation_max_age_seconds": attestation_max_age.total_seconds(),
            "detectors": list(detectors),
        })).hexdigest()
        self._fields_by_category: Dict[SafeHarborIdentifier, List[str]] = {}
        for name, spec in self._schema.items():
            if spec.kind is FieldKind.IDENTIFIER:
                self._fields_by_category.setdefault(spec.identifier, []).append(name)

    @property
    def config_digest(self) -> str:
        """SHA-256 of the pipeline version, policy version, schema, and detectors."""
        return self._config_digest

    @property
    def detectors(self) -> Tuple[str, ...]:
        return self._detectors

    def run(
        self,
        records: Sequence[SourceRecord],
        *,
        operator: OperatorContext,
        method: Union[DeidentificationMethod, str] = DeidentificationMethod.SAFE_HARBOR,
        attestation: Optional[ActualKnowledgeAttestation] = None,
        expert_determination: Optional[ExpertDeterminationReference] = None,
        reference_date: Optional[date] = None,
    ) -> DeidentificationResult:
        """De-identify ``records`` and produce the run's evidence artifact.

        Preconditions are checked before any record is read, and a refused run
        produces nothing:

        * Safe Harbor requires ``attestation`` — the operator's statement of no
          actual knowledge. The pipeline never supplies it.
        * Expert Determination requires ``expert_determination`` — the expert
          report it rests on. The pipeline never claims it otherwise.

        ``reference_date`` is the date ages are measured against when deciding
        whether a date is 90 or more years old (and so aggregated to 90+); it
        defaults to the run date and may not fall in an earlier year, which
        would understate every age. The attestation must be dated within
        ``attestation_max_age`` before the run and not after it.
        """
        try:
            method = DeidentificationMethod(method)
        except ValueError:
            raise DeidentificationRefused(f"unknown de-identification method {method!r}") from None
        if not isinstance(operator, OperatorContext):
            raise DeidentificationRefused("a run requires an OperatorContext")
        try:
            operator.validate()
            if method is DeidentificationMethod.SAFE_HARBOR:
                if attestation is None:
                    raise DeidentificationRefused(
                        "Safe Harbor requires the operator's actual-knowledge "
                        "attestation; no artifact is produced without it"
                    )
                if expert_determination is not None:
                    raise DeidentificationRefused(
                        "a Safe Harbor run cannot cite an expert report; run with "
                        "method=expert_determination instead"
                    )
            elif expert_determination is None:
                raise DeidentificationRefused(
                    "Expert Determination requires a reference to the qualified "
                    "expert's report; it is never claimed without one"
                )
            if attestation is not None:
                if not isinstance(attestation, ActualKnowledgeAttestation):
                    raise DeidentificationRefused(
                        "attestation must be an ActualKnowledgeAttestation"
                    )
                attestation.validate()
            if expert_determination is not None:
                if not isinstance(expert_determination, ExpertDeterminationReference):
                    raise DeidentificationRefused(
                        "expert_determination must be an ExpertDeterminationReference"
                    )
                expert_determination.validate()
            now = self._clock()
            if now.tzinfo is None:
                raise DeidentificationConfigError("the pipeline clock must be timezone-aware")
            if attestation is not None:
                attested_at = parse_timestamp(attestation.attested_at, "attestation.attested_at")
                if attested_at > now + ATTESTATION_CLOCK_SKEW:
                    raise DeidentificationRefused("the attestation is dated after the run")
                if now - attested_at > self._attestation_max_age:
                    raise DeidentificationRefused(
                        "the attestation is older than the pipeline accepts; the "
                        "operator must attest for this release"
                    )
        except EvidenceValidationError as exc:
            raise DeidentificationRefused(str(exc)) from None
        reference = now.date() if reference_date is None else reference_date
        if not isinstance(reference, date) or isinstance(reference, datetime):
            raise DeidentificationRefused("reference_date must be a date")
        # Ages are measured in years, so only an earlier year can understate one.
        if reference.year < now.year:
            raise DeidentificationRefused(
                "reference_date precedes the run's year: ages measured against "
                "it are understated, keeping birth years Safe Harbor aggregates"
            )
        # A later reference year makes more cued birth years identifying than
        # the construction-time check saw.
        unsafe = _identifier_bearing_name(self._schema, reference.year)
        if unsafe is not None:
            raise DeidentificationRefused(
                f"schema field {unsafe[0]} has a name carrying a "
                f"{unsafe[1].value} identifier against this run's reference date"
            )

        batch = list(records)
        if not batch:
            raise DeidentificationError("a run needs at least one record")
        record_ids = [getattr(record, "record_id", None) for record in batch]
        if len(set(record_ids)) != len(record_ids):
            raise DeidentificationError("record_id values must be unique within a run")

        outputs: List[Dict[str, Any]] = []
        record_evidence: List[RecordEvidence] = []
        for record in batch:
            output, evidence = self._deidentify(record, reference.year)
            outputs.append(output)
            record_evidence.append(evidence)

        draft = DeidentificationEvidence(
            evidence_id=uuid.uuid4().hex,
            method=method,
            created_at=now.isoformat(),
            reference_date=reference.isoformat(),
            policy_version=SAFE_HARBOR_POLICY_VERSION,
            pipeline_name=PIPELINE_NAME,
            pipeline_version=PIPELINE_VERSION,
            config_digest=self._config_digest,
            detectors=self._detectors,
            source_digest_key_id=self._key_id,
            operator=operator,
            records=tuple(record_evidence),
            categories=summarize_categories(record_evidence, self._fields_by_category),
            attestation=attestation,
            expert_determination=expert_determination,
            artifact_digest="",
        )
        evidence = dataclasses.replace(draft, artifact_digest=draft.compute_digest())
        return DeidentificationResult(tuple(outputs), evidence)

    # ── per-record transformation ────────────────────────────────────────────

    def _digest(self, domain: bytes, payload: bytes) -> str:
        return hmac.new(self._key, domain + b"\x00" + payload, hashlib.sha256).hexdigest()

    def _known_values(
        self, fields: Mapping[str, Any], reference_year: int
    ) -> Tuple[List[Tuple[str, SafeHarborIdentifier]], List[Tuple[str, SafeHarborIdentifier]]]:
        """The record's own identifier values as literals to find in its text.

        Returns (whole values, whole values plus scrub-only literals). The
        scrub-only literals are removed from free text but are too loose to
        refute a non-identifying value: name tokens (a note says "Ms. Doe",
        not "Jane Q. Doe", but a surname such as White is also a race), and an
        age over 89 or a date implying one ("120/92" is a blood pressure, not
        an age). Ages and dates below that are permitted and kept.
        """
        # Keyed case-insensitively; matching is case-insensitive too.
        whole: Dict[str, Tuple[str, SafeHarborIdentifier]] = {}
        scrub_only: Dict[str, Tuple[str, SafeHarborIdentifier]] = {}
        for name, value in fields.items():
            spec = self._schema[name]
            if spec.kind is not FieldKind.IDENTIFIER:
                continue
            if spec.role is FieldRole.AGE:
                age = _parse_age(value)
                if age is not None and age > 89:
                    # "95.0" from a float column is written "95" in a note.
                    for literal in {str(value).strip(), str(int(age))}:
                        scrub_only.setdefault(literal.lower(), (literal, spec.identifier))
                continue
            if spec.role in (FieldRole.DATE, FieldRole.BIRTH_DATE):
                year = _parse_year(value)
                if year is not None and reference_year - year >= 90:
                    scrub_only.setdefault(str(year), (str(year), spec.identifier))
            literals: List[str] = []
            if isinstance(value, str):
                literals.append(value.strip())
            elif isinstance(value, int) and not isinstance(value, bool):
                literals.append(str(value))
            elif isinstance(value, float):
                # Finite: _deidentify refuses the rest. An MRN read from a
                # float column as 4321.0 is written "4321" elsewhere.
                literals.append(str(value))
                if value.is_integer():
                    literals.append(str(int(value)))
            elif isinstance(value, datetime):
                literals.extend((value.isoformat(), value.date().isoformat()))
            elif isinstance(value, date):
                literals.append(value.isoformat())
            if spec.role is FieldRole.ZIP_CODE:
                zip_code = _parse_zip(value)
                if zip_code is not None:
                    literals.append(zip_code)
            for literal in literals:
                if len(literal) >= 2:
                    whole.setdefault(literal.lower(), (literal, spec.identifier))
            if spec.identifier is SafeHarborIdentifier.NAMES and isinstance(value, str):
                for token in _NAME_TOKEN_SPLIT.split(value):
                    if len(token) >= 2:
                        scrub_only.setdefault(token.lower(), (token, spec.identifier))

        def ordered(literals: Dict[str, Tuple[str, SafeHarborIdentifier]]):
            return sorted(literals.values(), key=lambda item: (-len(item[0]), item[0]))

        return ordered(whole), ordered({**scrub_only, **whole})

    def _transform_identifier(
        self, spec: FieldSpec, value: Any, reference_year: int
    ) -> Tuple[Any, Optional[TransformationAction]]:
        """(output value or ``_DROP``, the action taken or ``None`` if kept)."""
        if spec.role is FieldRole.ZIP_CODE:
            zip_code = _parse_zip(value)
            if zip_code is None:
                return _DROP, TransformationAction.REMOVED
            prefix = zip_code[:3]
            return ("000" if prefix in self._restricted_zip3 else prefix), TransformationAction.GENERALIZED
        if spec.role in (FieldRole.DATE, FieldRole.BIRTH_DATE):
            year = _parse_year(value)
            if year is None or year > reference_year:
                return _DROP, TransformationAction.REMOVED
            # A birth year that old implies an age over 89; so does any other
            # date the person was alive for (an admission, a procedure).
            if reference_year - year >= 90:
                return AGE_90_OR_OLDER, TransformationAction.GENERALIZED
            return year, TransformationAction.GENERALIZED
        if spec.role is FieldRole.AGE:
            age = _parse_age(value)
            if age is None:
                return _DROP, TransformationAction.REMOVED
            if age > 89:
                return AGE_90_OR_OLDER, TransformationAction.GENERALIZED
            return value, None
        return _DROP, TransformationAction.REMOVED

    def _deidentify(
        self, record: SourceRecord, reference_year: int
    ) -> Tuple[Dict[str, Any], RecordEvidence]:
        if not isinstance(record, SourceRecord):
            raise DeidentificationError("records must be SourceRecord instances")
        if not isinstance(record.record_id, str) or not record.record_id.strip():
            raise DeidentificationError("record_id must be a non-empty string")
        if not isinstance(record.fields, Mapping):
            raise DeidentificationError("record fields must be a mapping")
        fields = dict(record.fields)
        unknown = sorted(str(name) for name in fields if name not in self._schema)
        if unknown:
            raise DeidentificationError(
                f"record carries field(s) the schema does not classify: {unknown}"
            )
        for name, value in fields.items():
            if not isinstance(value, _SCALAR_TYPES + (date, bytes)):
                raise DeidentificationError(
                    f"field {name!r} holds an unsupported {type(value).__name__} value"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise DeidentificationError(f"field {name!r} holds a non-finite number")

        whole_values, scrub_values = self._known_values(fields, reference_year)
        output: Dict[str, Any] = {}
        counts: Dict[Tuple[str, SafeHarborIdentifier, TransformationAction, str], int] = {}
        for name in sorted(fields):
            spec, value = self._schema[name], fields[name]
            if spec.kind is FieldKind.IDENTIFIER:
                if _is_empty(value):
                    continue
                transformed, action = self._transform_identifier(spec, value, reference_year)
                if transformed is not _DROP:
                    output[name] = transformed
                if action is not None:
                    key = (name, spec.identifier, action, SCHEMA_DETECTOR)
                    counts[key] = counts.get(key, 0) + 1
            elif spec.kind is FieldKind.FREE_TEXT:
                if value is None:
                    output[name] = None
                    continue
                if not isinstance(value, str):
                    raise DeidentificationError(f"free-text field {name!r} must hold a string")
                text, spans = scrub_free_text(
                    value,
                    known_values=scrub_values,
                    reference_year=reference_year,
                    entity_detector=self._entity_detector,
                )
                output[name] = text
                for (category, action, detector), occurrences in spans.items():
                    key = (name, category, action, detector)
                    counts[key] = counts.get(key, 0) + occurrences
            else:
                if isinstance(value, (date, bytes)):
                    raise DeidentificationError(
                        f"field {name!r} is classified non-identifying but holds a "
                        f"{type(value).__name__} value; classify it as an identifier"
                    )
                text = _scannable_text(value)
                if text is not None:
                    findings = find_identifier_patterns(
                        text, reference_year=reference_year, known_values=whole_values
                    )
                    if findings:
                        category, detector = findings[0]
                        raise DeidentificationError(
                            f"field {name!r} is classified non-identifying but contains "
                            f"a {category.value} identifier ({detector}); classify it "
                            "as free text or as an identifier"
                        )
                output[name] = value

        transformations = tuple(
            FieldTransformation(field_name, category, action, detector, occurrences)
            for (field_name, category, action, detector), occurrences in sorted(
                counts.items(),
                key=lambda item: (
                    item[0][0], _CATEGORY_ORDER[item[0][1]], item[0][2].value, item[0][3]
                ),
            )
        )
        source_payload = canonical_json_bytes(
            {name: _encode_source_value(fields[name]) for name in sorted(fields)}
        )
        return output, RecordEvidence(
            source_id_digest=self._digest(_SOURCE_ID_DOMAIN, record.record_id.encode("utf-8")),
            source_digest=self._digest(_SOURCE_DOMAIN, source_payload),
            output_digest=output_record_digest(output),
            transformations=transformations,
        )
