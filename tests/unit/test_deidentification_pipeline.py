"""De-identification pipeline and evidence artifact (#1761).

All records here are synthetic. The pipeline must cover every Safe Harbor
identifier category, stay distinct from PII redaction, and never produce an
artifact (and so never authorize a save) without the inputs it cannot supply
itself: the operator's attestation, or the expert's report reference.
"""

import dataclasses
import hashlib
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from kestrel_sovereign.deidentification import (
    ACTUAL_KNOWLEDGE_STATEMENT,
    AGE_90_OR_OLDER,
    DEIDENTIFICATION_ASSURANCES,
    EXPORT_SCHEMA,
    PIPELINE_VERSION,
    SAFE_HARBOR_POLICY_VERSION,
    ActualKnowledgeAttestation,
    DeidentificationConfigError,
    DeidentificationError,
    DeidentificationEvidence,
    DeidentificationMethod,
    DeidentificationPipeline,
    DeidentificationRefused,
    DeidentificationResult,
    EntitySpan,
    EvidenceValidationError,
    ExpertDeterminationReference,
    FieldSpec,
    OperatorContext,
    SafeHarborIdentifier,
    SourceRecord,
    TransformationAction,
    redaction_placeholder,
    validate_evidence,
)
from kestrel_sovereign.deidentification import NerEntityDetector
from kestrel_sovereign.deidentification import detectors as detectors_module
from kestrel_sovereign.deidentification.evidence import output_record_digest
from kestrel_sovereign.features.privacy.pii_detector import PIIDetector, PIIMatch, PIIType
from kestrel_sovereign.privacy import get_privacy_preset

KEY = b"synthetic-source-digest-key-0001"
FIXED_NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
OPERATOR = OperatorContext("operator:synthetic", "request-0001")


class ListedNames:
    """An entity detector that finds a fixed list of synthetic names."""

    name = "entity:synthetic-names"

    def __init__(self, *names, category=SafeHarborIdentifier.NAMES):
        self._names = names
        self._category = category

    def detect(self, text):
        for name in self._names:
            start = text.find(name)
            while start != -1:
                yield EntitySpan(start, start + len(name), self._category)
                start = text.find(name, start + 1)


def _attestation():
    return ActualKnowledgeAttestation(
        attested_by="operator:synthetic",
        no_actual_knowledge=True,
        attested_at=FIXED_NOW.isoformat(),
    )


def _pipeline(schema, **kwargs):
    kwargs.setdefault("entity_detector", ListedNames("Rowan Example"))
    return DeidentificationPipeline(
        schema, source_digest_key=KEY, clock=lambda: FIXED_NOW, **kwargs
    )


def _run(pipeline, *records, **kwargs):
    kwargs.setdefault("attestation", _attestation())
    batch = [
        r if isinstance(r, SourceRecord) else SourceRecord(f"rec-{i}", r)
        for i, r in enumerate(records)
    ]
    return pipeline.run(batch, operator=OPERATOR, **kwargs)


def _transformations(result, field):
    return [t for t in result.evidence.records[0].transformations if t.field == field]


def _disposition(result, category):
    return next(c for c in result.evidence.categories if c.category is category)


# ── Preset and taxonomy ──────────────────────────────────────────────────────


def test_deidentified_preset_dimensions_stay_distinct_from_anonymous():
    deidentified = get_privacy_preset("deidentified")
    anonymous = get_privacy_preset("anonymous")
    assert (
        deidentified.storage,
        deidentified.processing,
        deidentified.sharing,
        deidentified.assurance,
        deidentified.audit,
    ) == ("deidentified", "trusted", "research", "safe_harbor", "required")
    assert anonymous.storage == "pii_redacted"
    assert anonymous.assurance == "pii_redacted"
    # Only an evidence artifact can back a de-identification assurance; PII
    # redaction is not one of them.
    assert deidentified.assurance in DEIDENTIFICATION_ASSURANCES
    assert anonymous.assurance not in DEIDENTIFICATION_ASSURANCES
    assert DEIDENTIFICATION_ASSURANCES == {"safe_harbor", "expert_determination"}


def test_safe_harbor_identifier_list_has_all_eighteen_categories():
    assert len(SafeHarborIdentifier) == 18


def test_pii_redaction_is_not_deidentification():
    """The PII redactor leaves identifier categories the pipeline removes."""
    text = (
        "MRN: A1029384, portal https://portal.example.test/u/7, "
        "host 10.20.30.40, plate ABC1234, seen March 5, 2026"
    )
    redacted = PIIDetector.__new__(PIIDetector)
    redacted.nlp = None
    pii_output = redacted.anonymize(text)
    assert "A1029384" in pii_output
    assert "portal.example.test" in pii_output
    assert "10.20.30.40" in pii_output

    result = _run(_pipeline({"note": FieldSpec.free_text()}), {"note": text})
    note = result.records[0]["note"]
    for identifier in ("A1029384", "portal.example.test", "10.20.30.40", "ABC1234", "March"):
        assert identifier not in note
    assert "2026" in note  # the year of a date is permitted


# ── Structured fields ────────────────────────────────────────────────────────


@pytest.mark.parametrize("category", list(SafeHarborIdentifier))
def test_every_category_can_be_removed_by_schema(category):
    result = _run(
        _pipeline({"value": FieldSpec.remove(category), "kept": FieldSpec.non_identifying()}),
        {"value": "synthetic-value", "kept": "kept"},
    )
    assert result.records[0] == {"kept": "kept"}
    [transformation] = _transformations(result, "value")
    assert transformation.category is category
    assert transformation.action is TransformationAction.REMOVED
    assert transformation.detector == "schema"
    disposition = _disposition(result, category)
    assert disposition.removed == 1
    assert disposition.fields_classified == ("value",)


def test_generalizations_follow_safe_harbor():
    schema = {
        "zip": FieldSpec.zip_code(),
        "restricted_zip": FieldSpec.zip_code(),
        "admitted": FieldSpec.date(),
        "dob_young": FieldSpec.birth_date(),
        "dob_old": FieldSpec.birth_date(),
        "age_young": FieldSpec.age(),
        "age_old": FieldSpec.age(),
        "bad_date": FieldSpec.date(),
    }
    result = _run(_pipeline(schema), {
        "zip": "02139-4307",
        "restricted_zip": "03601",
        "admitted": date(2026, 3, 5),
        "dob_young": "1980-07-14",
        "dob_old": datetime(1934, 1, 2, 8, 30),
        "age_young": 47,
        "age_old": 92,
        "bad_date": "sometime in spring",
    })
    record = result.records[0]
    assert record == {
        "zip": "021",
        "restricted_zip": "000",
        "admitted": 2026,
        "dob_young": 1980,
        "dob_old": AGE_90_OR_OLDER,
        "age_young": 47,
        "age_old": AGE_90_OR_OLDER,
    }
    assert [t.action for t in _transformations(result, "bad_date")] == [
        TransformationAction.REMOVED
    ]
    # An age through 89 is permitted and kept, so it is not a transformation.
    assert _transformations(result, "age_young") == []
    assert _disposition(result, SafeHarborIdentifier.DATES).generalized == 4


def test_dates_ninety_years_old_are_aggregated_in_every_date_role():
    schema = {"admitted": FieldSpec.date(), "dob": FieldSpec.birth_date(), "note": FieldSpec.free_text()}
    result = _run(_pipeline(schema), {
        "admitted": "1934-06-01",
        "dob": "03/15/1931",  # not ISO: still parsed, not merely dropped
        "note": "Raised on a farm; in 1931 the family moved and in 1934 she enrolled.",
    })
    record = result.records[0]
    assert record["admitted"] == AGE_90_OR_OLDER
    assert record["dob"] == AGE_90_OR_OLDER
    assert "1931" not in record["note"] and "1934" not in record["note"]


def test_evidence_counts_each_birth_year_once():
    result = _run(_pipeline({"note": FieldSpec.free_text()}), {"note": "DOB 03/15/1950, MRN 12345"})
    [birth] = [t for t in _transformations(result, "note") if t.detector == "pattern:birth_date"]
    assert birth.occurrences == 1


def test_an_age_of_exactly_ninety_does_not_mangle_the_aggregate():
    schema = {"age": FieldSpec.age(), "note": FieldSpec.free_text()}
    result = _run(_pipeline(schema), {"age": 90, "note": "aged 90, a 93-year-old sibling"})
    # The record's own "90" is removed; the generated "90+" survives intact.
    assert result.records[0]["note"] == (
        f"aged {redaction_placeholder(SafeHarborIdentifier.DATES)}, "
        f"a {AGE_90_OR_OLDER}-year-old sibling"
    )


def test_float_age_is_scrubbed_from_text_in_its_integer_form():
    schema = {"age": FieldSpec.age(), "note": FieldSpec.free_text()}
    result = _run(_pipeline(schema), {"age": 95.0, "note": "Patient, 95, fell at home."})
    assert result.records[0]["age"] == AGE_90_OR_OLDER
    assert "95" not in result.records[0]["note"]


def test_restricted_zip_list_is_configuration_recorded_in_the_digest():
    default = _pipeline({"zip": FieldSpec.zip_code()})
    custom = _pipeline({"zip": FieldSpec.zip_code()}, restricted_zip3_prefixes={"021"})
    assert default.config_digest != custom.config_digest
    assert _run(custom, {"zip": "02139"}).records[0]["zip"] == "000"


def test_reference_date_decides_birth_year_aggregation():
    pipeline = _pipeline({"dob": FieldSpec.birth_date()})
    assert _run(pipeline, {"dob": "1940-06-01"}, reference_date=date(2029, 1, 1)).records[0] == {
        "dob": 1940
    }
    result = _run(pipeline, {"dob": "1940-06-01"}, reference_date=date(2030, 1, 1))
    assert result.records[0] == {"dob": AGE_90_OR_OLDER}
    assert result.evidence.reference_date == "2030-01-01"


# ── Free text ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, secret, category",
    [
        ("write to rowan@example.test today", "rowan@example.test", SafeHarborIdentifier.EMAIL_ADDRESSES),
        ("see https://example.test/p?id=7", "example.test", SafeHarborIdentifier.WEB_URLS),
        ("from 192.0.2.44 overnight", "192.0.2.44", SafeHarborIdentifier.IP_ADDRESSES),
        ("from 2001:db8:0:0:1:0:0:1 overnight", "2001:db8", SafeHarborIdentifier.IP_ADDRESSES),
        ("adapter 00:1A:2B:3C:4D:5E online", "00:1A:2B", SafeHarborIdentifier.DEVICE_IDENTIFIERS),
        ("ssn 078-05-1120 on file", "078-05-1120", SafeHarborIdentifier.SOCIAL_SECURITY_NUMBERS),
        ("SSN: 078 05 1120 on file", "078 05 1120", SafeHarborIdentifier.SOCIAL_SECURITY_NUMBERS),
        ("call (555) 010-0199 later", "010-0199", SafeHarborIdentifier.TELEPHONE_NUMBERS),
        ("call 555-0142 later", "555-0142", SafeHarborIdentifier.TELEPHONE_NUMBERS),
        ("fax 555-0142 later", "555-0142", SafeHarborIdentifier.FAX_NUMBERS),
        ("call +44 20 7946 0958 later", "7946", SafeHarborIdentifier.TELEPHONE_NUMBERS),
        ("her blog is quill-art.example.com/about", "quill-art", SafeHarborIdentifier.WEB_URLS),
        ("fax: 555-010-0142 records", "010-0142", SafeHarborIdentifier.FAX_NUMBERS),
        ("MRN: Q7781234 updated", "Q7781234", SafeHarborIdentifier.MEDICAL_RECORD_NUMBERS),
        ("MR# 482913 updated", "482913", SafeHarborIdentifier.MEDICAL_RECORD_NUMBERS),
        ("member id XJ-55821 active", "XJ-55821", SafeHarborIdentifier.HEALTH_PLAN_BENEFICIARY_NUMBERS),
        ("Medicare: 1EG4-TE5-MK73 active", "1EG4-TE5-MK73", SafeHarborIdentifier.HEALTH_PLAN_BENEFICIARY_NUMBERS),
        ("MRN#: 4471-22 updated", "4471-22", SafeHarborIdentifier.MEDICAL_RECORD_NUMBERS),
        ("acct #00-81723 billed", "00-81723", SafeHarborIdentifier.ACCOUNT_NUMBERS),
        ("license no. D1234567 verified", "D1234567", SafeHarborIdentifier.CERTIFICATE_LICENSE_NUMBERS),
        ("license plate 7ABC123 parked", "7ABC123", SafeHarborIdentifier.VEHICLE_IDENTIFIERS),
        ("VIN 1HGCM82633A004352 towed", "1HGCM82633A004352", SafeHarborIdentifier.VEHICLE_IDENTIFIERS),
        ("serial number SN-99812 replaced", "SN-99812", SafeHarborIdentifier.DEVICE_IDENTIFIERS),
        ("case id 4471-B closed", "4471-B", SafeHarborIdentifier.OTHER_UNIQUE_IDENTIFIERS),
        ("ref 123456789 attached", "123456789", SafeHarborIdentifier.OTHER_UNIQUE_IDENTIFIERS),
        ("ref 482913 attached", "482913", SafeHarborIdentifier.OTHER_UNIQUE_IDENTIFIERS),
        ("lives at 42 Quarry Lane now", "Quarry", SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS),
        ("mail to PO Box 4410 only", "4410", SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS),
        ("postal 02139 noted", "02139", SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS),
        ("met in Fairhaven", "Fairhaven", SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS),
    ],
)
def test_free_text_removes_each_detectable_category(text, secret, category):
    pipeline = _pipeline(
        {"note": FieldSpec.free_text()},
        entity_detector=ListedNames(
            "Fairhaven", category=SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS
        ),
    )
    result = _run(pipeline, {"note": text})
    note = result.records[0]["note"]
    assert secret not in note
    assert redaction_placeholder(category) in note
    assert _disposition(result, category).transformed >= 1


@pytest.mark.parametrize(
    "text, expected",
    [
        ("seen 03/05/2026 again", "seen 2026 again"),
        ("seen 2026-03-05T10:15:00Z again", "seen 2026 again"),
        ("seen March 5, 2026 again", "seen 2026 again"),
        ("seen 5th of March 2026 again", "seen 2026 again"),
        ("seen March 2026 again", "seen 2026 again"),
        ("seen 03/2026 again", "seen 2026 again"),
        ("seen 3/5/26 again", f"seen {redaction_placeholder(SafeHarborIdentifier.DATES)} again"),
        ("seen 3.15.20 again", f"seen {redaction_placeholder(SafeHarborIdentifier.DATES)} again"),
        ("seen on 3/15 again", f"seen on {redaction_placeholder(SafeHarborIdentifier.DATES)} again"),
        ("Admitted 15-Mar-2024 to ward", "Admitted 2024 to ward"),
        ("seen march 3, 2024", "seen 2024"),
        ("DOB: MARCH 15, 1931", f"DOB: {AGE_90_OR_OLDER}"),
        ("Admitted 2024.03.15", "Admitted 2024"),
        ("YOB: 1931", f"YOB: {AGE_90_OR_OLDER}"),
        ("95F with CHF", f"{AGE_90_OR_OLDER}F with CHF"),
        ("seen 2026-03 again", "seen 2026 again"),
        ("take 1/2 tablet", "take 1/2 tablet"),
        ("a 93-year-old", f"a {AGE_90_OR_OLDER}-year-old"),
        ("aged 97 at intake", f"aged {AGE_90_OR_OLDER} at intake"),
        ("a 95 y.o. woman", f"a {AGE_90_OR_OLDER} y.o. woman"),
        ("died at the age of 93", f"died at the age of {AGE_90_OR_OLDER}"),
        ("a 45-year-old", "a 45-year-old"),
        ("born 1931 abroad", f"born {AGE_90_OR_OLDER} abroad"),
        ("DOB: 04/02/1980 noted", "DOB: 1980 noted"),
        # A birth date a few words past its cue, and any date 90+ years old.
        ("Her date of birth was 03/15/1931.", f"Her date of birth was {AGE_90_OR_OLDER}."),
        ("Her date of birth was 1931.", f"Her date of birth was {AGE_90_OR_OLDER}."),
        ("Born on the 3rd of March 1931.", f"Born on the {AGE_90_OR_OLDER}."),
        ("admitted 06/01/1934 once", f"admitted {AGE_90_OR_OLDER} once"),
    ],
)
def test_free_text_dates_keep_only_the_year(text, expected):
    result = _run(_pipeline({"note": FieldSpec.free_text()}), {"note": text})
    assert result.records[0]["note"] == expected


def test_generalizing_a_date_cannot_leave_a_new_identifier_behind():
    """"Insurance: Jan 5, 2024" generalizes to a labelled number; it is removed
    too, rather than tripping the residual scan on genuine output."""
    result = _run(_pipeline({"note": FieldSpec.free_text()}), {"note": "Insurance: Jan 5, 2024"})
    assert result.records[0]["note"] == (
        f"Insurance: {redaction_placeholder(SafeHarborIdentifier.HEALTH_PLAN_BENEFICIARY_NUMBERS)}"
    )


def test_cascading_generalizations_settle_before_the_entity_pass():
    """Each round exposes the next match: the year/month date, then the
    labelled number its year leaves behind. One round per settlement is not
    enough."""
    result = _run(
        _pipeline({"note": FieldSpec.free_text()}), {"note": "Insurance: 12/Jan 5, 2024"}
    )
    assert result.records[0]["note"] == (
        f"Insurance: {redaction_placeholder(SafeHarborIdentifier.HEALTH_PLAN_BENEFICIARY_NUMBERS)}"
    )


def test_text_that_does_not_settle_is_refused(monkeypatch):
    monkeypatch.setattr(detectors_module, "_MAX_SCRUB_ROUNDS", 1)
    with pytest.raises(DeidentificationError, match="stable"):
        _run(_pipeline({"note": FieldSpec.free_text()}), {"note": "Insurance: Jan 5, 2024"})


def test_ner_adapter_keeps_names_the_redactors_regexes_would_shadow():
    text = "Seen 2 days ago with son Tom Lee and Dr Patel."

    class StubPII:
        has_ner = True

        def detect(self, _text):
            raise AssertionError("detect() lets an address regex shadow the name")

        def detect_entities(self, _text):
            start = text.index("Tom Lee")
            return [PIIMatch(PIIType.PERSON, "Tom Lee", start, start + 7)]

    result = _run(
        _pipeline({"note": FieldSpec.free_text()}, entity_detector=NerEntityDetector(StubPII())),
        {"note": text},
    )
    assert "Tom Lee" not in result.records[0]["note"]
    assert "entity:spacy-ner" in result.evidence.detectors


def test_free_text_names_come_from_the_entity_detector():
    result = _run(
        _pipeline({"note": FieldSpec.free_text()}),
        {"note": "Rowan Example reported dizziness."},
    )
    assert result.records[0]["note"] == (
        f"{redaction_placeholder(SafeHarborIdentifier.NAMES)} reported dizziness."
    )
    [transformation] = _transformations(result, "note")
    assert transformation.detector == "entity:synthetic-names"
    assert "entity:synthetic-names" in result.evidence.detectors


class SpanningDetector:
    """Reports spans that overlap an earlier placeholder and each other."""

    name = "entity:spanning"

    def detect(self, text):
        # The scrubber re-runs until the text is stable; report nothing once
        # the spans are gone.
        surname = text.find("Quillfeather")
        if surname != -1:
            # From the start of the text (an earlier placeholder) through the surname.
            yield EntitySpan(0, surname + len("Quillfeather"), SafeHarborIdentifier.NAMES)
        town = text.find("Fairhaven")
        if town != -1:
            # Two spans overlapping one another within the same pass.
            yield EntitySpan(town, town + len("Fairhaven"), SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS)
            yield EntitySpan(town + 4, town + len("Fairhaven Mills"), SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS)


def test_overlapping_entity_spans_are_trimmed_not_dropped():
    """A span that runs into an earlier replacement keeps the rest covered."""
    pipeline = _pipeline(
        {"name": FieldSpec.remove(SafeHarborIdentifier.NAMES), "note": FieldSpec.free_text()},
        entity_detector=SpanningDetector(),
    )
    result = _run(
        pipeline,
        {"name": "Avery Doe", "note": "Doe Quillfeather moved to Fairhaven Mills."},
    )
    names = redaction_placeholder(SafeHarborIdentifier.NAMES)
    places = redaction_placeholder(SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS)
    assert result.records[0]["note"] == f"{names} {names} moved to {places} {places}."


def test_free_text_scrubs_the_records_own_identifier_values():
    schema = {
        "name": FieldSpec.remove(SafeHarborIdentifier.NAMES),
        "chart": FieldSpec.remove(SafeHarborIdentifier.MEDICAL_RECORD_NUMBERS),
        "age": FieldSpec.age(),
        "note": FieldSpec.free_text(),
    }
    result = _run(_pipeline(schema), {
        "name": "Avery Jo-Lynn Quillfeather",
        "chart": "ZX88123",
        "age": 94,
        "note": "Ms. Quillfeather (chart ZX88123) is 94; Jo-Lynn prefers mornings.",
    })
    note = result.records[0]["note"]
    for value in ("Quillfeather", "ZX88123", "94", "Jo-Lynn"):
        assert value not in note
    detectors = {t.detector for t in _transformations(result, "note")}
    assert detectors == {"known_value"}


def test_email_is_removed_whole_even_when_it_embeds_the_name():
    """Regression: a name token must not split an e-mail and strand its domain."""
    schema = {"name": FieldSpec.remove(SafeHarborIdentifier.NAMES), "note": FieldSpec.free_text()}
    result = _run(_pipeline(schema), {"name": "Avery Quillfeather", "note": "mail avery@quill.example.test"})
    assert result.records[0]["note"] == (
        f"mail {redaction_placeholder(SafeHarborIdentifier.EMAIL_ADDRESSES)}"
    )


def test_birth_year_literal_does_not_strand_month_and_day():
    """Regression: the record's 90+ birth year must not split a date."""
    schema = {"dob": FieldSpec.birth_date(), "note": FieldSpec.free_text()}
    result = _run(_pipeline(schema), {
        "dob": "1931-04-02",
        "note": "records list 04/02/1931 and, separately, the year 1931.",
    })
    note = result.records[0]["note"]
    assert "04/02" not in note
    assert "1931" not in note


def test_known_value_inside_a_larger_structure_does_not_fragment_it():
    """The ZIP+4 is removed whole; matching the record's ZIP inside it first
    would strand the "-4307" extension where no pattern sees it."""
    schema = {"zip": FieldSpec.zip_code(), "note": FieldSpec.free_text()}
    result = _run(_pipeline(schema), {"zip": "02139", "note": "mail to 02139-4307 only"})
    assert result.records[0]["note"] == (
        f"mail to {redaction_placeholder(SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS)} only"
    )


def test_known_value_joined_to_an_unrecognized_structure_is_still_removed():
    """No pattern recognizes "QX-5512/2", so the closing known-value pass must."""
    schema = {
        "chart": FieldSpec.remove(SafeHarborIdentifier.MEDICAL_RECORD_NUMBERS),
        "note": FieldSpec.free_text(),
    }
    result = _run(_pipeline(schema), {"chart": "QX-5512", "note": "file QX-5512/2 attached"})
    assert "QX-5512" not in result.records[0]["note"]


def test_free_text_without_an_entity_detector_is_refused(monkeypatch):
    with pytest.raises(DeidentificationConfigError, match="category A"):
        DeidentificationPipeline(
            {"note": FieldSpec.free_text()}, source_digest_key=KEY, entity_detector=None
        )
    # The default resolves to the spaCy model; without one the schema is refused.
    monkeypatch.setattr(detectors_module, "default_entity_detector", lambda: None)
    import kestrel_sovereign.deidentification.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "default_entity_detector", lambda: None)
    with pytest.raises(DeidentificationConfigError):
        DeidentificationPipeline({"note": FieldSpec.free_text()}, source_digest_key=KEY)
    # A schema with no free text needs no entity detector.
    DeidentificationPipeline({"zip": FieldSpec.zip_code()}, source_digest_key=KEY)


# ── Refusals ─────────────────────────────────────────────────────────────────


def test_unclassified_field_is_refused():
    with pytest.raises(DeidentificationError, match="does not classify"):
        _run(_pipeline({"zip": FieldSpec.zip_code()}), {"zip": "02139", "nickname": "Ro"})


@pytest.mark.parametrize(
    "value",
    [
        "contact rowan@example.test",
        "visit 2026-03-05",
        "Avery Quillfeather",  # the record's own name
        "follow-up on 3/15",
        "see quill-art.example.com",
        date(2026, 3, 5),
    ],
)
def test_non_identifying_classification_is_refuted_by_its_value(value):
    schema = {
        "name": FieldSpec.remove(SafeHarborIdentifier.NAMES),
        "diagnosis": FieldSpec.non_identifying(),
    }
    with pytest.raises(DeidentificationError, match="non-identifying"):
        _run(_pipeline(schema), {"name": "Avery Quillfeather", "diagnosis": value})


def test_a_non_identifying_number_equal_to_an_identifier_is_refused():
    schema = {
        "chart": FieldSpec.remove(SafeHarborIdentifier.MEDICAL_RECORD_NUMBERS),
        "chart_ref": FieldSpec.non_identifying(),
    }
    with pytest.raises(DeidentificationError, match="non-identifying"):
        _run(_pipeline(schema), {"chart": "4471223", "chart_ref": 4471223})


def test_a_name_token_that_is_also_a_word_does_not_refute():
    """Patient "Mary White" with race "White": the surname is scrubbed from
    free text but is not evidence against a categorical field."""
    schema = {
        "name": FieldSpec.remove(SafeHarborIdentifier.NAMES),
        "race": FieldSpec.non_identifying(),
        "note": FieldSpec.free_text(),
    }
    result = _run(_pipeline(schema), {"name": "Mary White", "race": "White", "note": "Ms. White is well."})
    assert result.records[0]["race"] == "White"
    assert "White" not in result.records[0]["note"]


@pytest.mark.parametrize(
    "text",
    [
        "Metformin 500-1000 mg BID",
        "I/O +500 mL",
        "pain 8/10, now 10/10",
        "1 mm ST depression",
        "follow up in 2 weeks with Dr. Lee",
        "serial q4h neuro checks",
        "serial q12h troponins",
        "may 1 tab, dec 2 mg",
        "Temp 101 F",
    ],
)
def test_ordinary_clinical_text_is_not_treated_as_an_identifier(text):
    """Over-matching here would refuse every batch with such a value."""
    result = _run(_pipeline({"value": FieldSpec.non_identifying()}), {"value": text})
    assert result.records[0]["value"] == text


def test_version_strings_do_not_refute_a_non_identifying_value():
    result = _run(_pipeline({"software": FieldSpec.non_identifying()}), {"software": "3.10.12"})
    assert result.records[0] == {"software": "3.10.12"}


def test_the_records_own_age_does_not_refute_a_non_identifying_value():
    """A 92-year-old's blood pressure of 120/92 is not an age."""
    schema = {"age": FieldSpec.age(), "bp": FieldSpec.non_identifying()}
    result = _run(_pipeline(schema), {"age": 92, "bp": "120/92"})
    assert result.records[0] == {"age": AGE_90_OR_OLDER, "bp": "120/92"}


def test_safe_harbor_without_attestation_produces_no_artifact():
    pipeline = _pipeline({"zip": FieldSpec.zip_code()})
    with pytest.raises(DeidentificationRefused, match="attestation"):
        _run(pipeline, {"zip": "02139"}, attestation=None)
    with pytest.raises(DeidentificationRefused, match="explicitly true"):
        _run(
            pipeline,
            {"zip": "02139"},
            attestation=ActualKnowledgeAttestation("operator:synthetic", False),
        )


def test_refusal_happens_before_any_record_is_read():
    class Exploding:
        record_id = "rec"

        @property
        def fields(self):
            raise AssertionError("record was read before the precondition check")

    pipeline = _pipeline({"zip": FieldSpec.zip_code()})
    with pytest.raises(DeidentificationRefused):
        pipeline.run([Exploding()], operator=OPERATOR)


def test_attestation_must_be_made_for_this_release():
    pipeline = _pipeline({"zip": FieldSpec.zip_code()})

    def attested(at):
        return ActualKnowledgeAttestation("operator:synthetic", True, attested_at=at.isoformat())

    with pytest.raises(DeidentificationRefused, match="dated after the run"):
        _run(pipeline, {"zip": "02139"}, attestation=attested(FIXED_NOW + timedelta(hours=1)))
    with pytest.raises(DeidentificationRefused, match="older than"):
        _run(pipeline, {"zip": "02139"}, attestation=attested(FIXED_NOW - timedelta(days=2)))
    _run(pipeline, {"zip": "02139"}, attestation=attested(FIXED_NOW - timedelta(hours=1)))

    lenient = _pipeline({"zip": FieldSpec.zip_code()}, attestation_max_age=timedelta(days=7))
    assert lenient.config_digest != pipeline.config_digest
    _run(lenient, {"zip": "02139"}, attestation=attested(FIXED_NOW - timedelta(days=2)))


def test_expert_determination_is_never_claimed_without_a_report():
    pipeline = _pipeline({"zip": FieldSpec.zip_code()})
    with pytest.raises(DeidentificationRefused, match="expert"):
        _run(
            pipeline,
            {"zip": "02139"},
            method=DeidentificationMethod.EXPERT_DETERMINATION,
            attestation=None,
        )
    report = ExpertDeterminationReference(
        report_reference="doc:expert-report/synthetic-1",
        report_digest=hashlib.sha256(b"synthetic expert report").hexdigest(),
        expert_id="expert:synthetic",
    )
    with pytest.raises(DeidentificationRefused, match="cannot cite an expert report"):
        _run(pipeline, {"zip": "02139"}, expert_determination=report)

    result = _run(
        pipeline,
        {"zip": "02139"},
        method="expert_determination",
        attestation=None,
        expert_determination=report,
    )
    assert result.evidence.method is DeidentificationMethod.EXPERT_DETERMINATION
    assert result.evidence.assurance == "expert_determination"
    assert result.evidence.expert_determination == report
    validate_evidence(result.evidence, result.records, required_assurance="expert_determination")


def test_expert_report_digest_must_be_a_sha256():
    with pytest.raises(DeidentificationRefused, match="report_digest"):
        _run(
            _pipeline({"zip": FieldSpec.zip_code()}),
            {"zip": "02139"},
            method="expert_determination",
            attestation=None,
            expert_determination=ExpertDeterminationReference(
                "doc:report", "not-a-digest", "expert:synthetic"
            ),
        )


def test_short_source_digest_key_is_refused():
    with pytest.raises(DeidentificationConfigError, match="32 bytes"):
        DeidentificationPipeline({"zip": FieldSpec.zip_code()}, source_digest_key=b"short")


# ── Evidence artifact ────────────────────────────────────────────────────────


SAMPLE = {
    "name": "Avery Quillfeather",
    "ssn": "078-05-1120",
    "zip": "02139",
    "dob": "1950-02-03",
    "note": "Rowan Example drove Ms. Quillfeather in on 2026-03-05.",
    "diagnosis": "synthetic condition",
}
SAMPLE_SCHEMA = {
    "name": FieldSpec.remove(SafeHarborIdentifier.NAMES),
    "ssn": FieldSpec.remove(SafeHarborIdentifier.SOCIAL_SECURITY_NUMBERS),
    "zip": FieldSpec.zip_code(),
    "dob": FieldSpec.birth_date(),
    "note": FieldSpec.free_text(),
    "diagnosis": FieldSpec.non_identifying(),
}


def _sample_result(**kwargs):
    return _run(_pipeline(SAMPLE_SCHEMA), SourceRecord("chart-778812", SAMPLE), **kwargs)


def test_evidence_records_what_the_issue_requires():
    result = _sample_result()
    evidence = result.evidence
    validate_evidence(evidence, result.records, required_assurance="safe_harbor")
    assert evidence.method is DeidentificationMethod.SAFE_HARBOR
    assert evidence.assurance == "safe_harbor"
    assert evidence.policy_version == SAFE_HARBOR_POLICY_VERSION
    assert evidence.pipeline_version == PIPELINE_VERSION
    assert evidence.config_digest == _pipeline(SAMPLE_SCHEMA).config_digest
    assert evidence.created_at == FIXED_NOW.isoformat()
    assert evidence.operator == OPERATOR
    assert evidence.attestation.statement == ACTUAL_KNOWLEDGE_STATEMENT
    assert evidence.expert_determination is None
    assert [c.category for c in evidence.categories] == list(SafeHarborIdentifier)
    [record] = evidence.records
    assert record.output_digest == output_record_digest(result.records[0])
    actions = {(t.field, t.category, t.action) for t in record.transformations}
    assert ("name", SafeHarborIdentifier.NAMES, TransformationAction.REMOVED) in actions
    assert ("zip", SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS, TransformationAction.GENERALIZED) in actions
    assert ("note", SafeHarborIdentifier.DATES, TransformationAction.GENERALIZED) in actions
    assert ("note", SafeHarborIdentifier.NAMES, TransformationAction.TRANSFORMED) in actions


def test_evidence_never_contains_source_values():
    payload = _sample_result().evidence.to_json_bytes().decode()
    for value in ("Avery", "Quillfeather", "078-05-1120", "02139", "1950-02-03", "chart-778812", "Rowan"):
        assert value not in payload


def test_source_digests_are_keyed_and_deterministic():
    first = _sample_result().evidence.records[0]
    second = _sample_result().evidence.records[0]
    assert first.source_digest == second.source_digest
    assert first.source_id_digest == second.source_id_digest
    assert first.source_id_digest != hashlib.sha256(b"chart-778812").hexdigest()
    other = DeidentificationPipeline(
        SAMPLE_SCHEMA,
        source_digest_key=b"another-synthetic-digest-key-002",
        entity_detector=ListedNames("Rowan Example"),
        clock=lambda: FIXED_NOW,
    )
    rekeyed = _run(other, SourceRecord("chart-778812", SAMPLE)).evidence
    assert rekeyed.records[0].source_digest != first.source_digest
    assert rekeyed.source_digest_key_id != _sample_result().evidence.source_digest_key_id


def test_evidence_round_trips_through_json():
    result = _sample_result()
    restored = DeidentificationEvidence.from_json_bytes(result.evidence.to_json_bytes())
    assert restored == result.evidence
    validate_evidence(restored, result.records)


def _tamper(evidence, mutate):
    data = evidence.to_dict()
    mutate(data)
    return data


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(attestation=None), "attestation"),
        (lambda d: d["attestation"].update(no_actual_knowledge=False), "explicitly true"),
        (lambda d: d.update(identifier_categories=d["identifier_categories"][:-1]), "eighteen"),
        (lambda d: d["identifier_categories"][0].update(removed=7), "disagrees"),
        (lambda d: d["records"][0]["transformations"].pop(), "disagrees"),
        (lambda d: d["operator"].update(request_id="request-9999"), "artifact_digest"),
        (lambda d: d["pipeline"].update(version="9.9.9"), "pipeline.version"),
        (lambda d: d["pipeline"].update(name="some.other.pipeline"), "not the Kestrel"),
        (lambda d: d.update(evidence_id="MRN-A1029384"), "evidence_id"),
        (lambda d: d["attestation"].update(attested_at="2026-10-04T12:00:00+00:00"), "dated after the run"),
        (lambda d: d.update(policy_version="made-up/v0"), "policy_version"),
        (lambda d: d["operator"].update(operator_id=" "), "operator_id"),
        (lambda d: d.update(created_at="2026-10-03T12:00:00"), "UTC offset"),
    ],
)
def test_tampered_evidence_is_rejected(mutate, message):
    data = _tamper(_sample_result().evidence, mutate)
    with pytest.raises(EvidenceValidationError, match=message):
        validate_evidence(DeidentificationEvidence.from_dict(data))


def test_evidence_must_describe_the_records_it_is_presented_with():
    result = _sample_result()
    altered = dict(result.records[0], diagnosis="a different value")
    with pytest.raises(EvidenceValidationError, match="output digest"):
        DeidentificationResult((altered,), result.evidence)
    with pytest.raises(EvidenceValidationError, match="covers 1 record"):
        validate_evidence(result.evidence, result.records * 2)


def test_result_records_are_read_only():
    result = _sample_result()
    with pytest.raises(TypeError):
        result.records[0]["diagnosis"] = "changed"


def test_forged_result_with_residual_identifier_is_rejected():
    """Self-consistent but forged evidence cannot launder a raw identifier."""
    result = _sample_result()
    forged_record = dict(result.records[0], diagnosis="contact rowan@example.test")
    record_evidence = dataclasses.replace(
        result.evidence.records[0], output_digest=output_record_digest(forged_record)
    )
    draft = dataclasses.replace(result.evidence, records=(record_evidence,))
    forged = dataclasses.replace(draft, artifact_digest=draft.compute_digest())
    validate_evidence(forged, (forged_record,))  # the artifact itself is consistent
    with pytest.raises(EvidenceValidationError, match="still carries"):
        DeidentificationResult((forged_record,), forged)


def test_required_assurance_is_a_generic_check():
    evidence = _sample_result().evidence
    validate_evidence(evidence, required_assurance="safe_harbor")
    with pytest.raises(EvidenceValidationError, match="not the required"):
        validate_evidence(evidence, required_assurance="expert_determination")
    with pytest.raises(EvidenceValidationError, match="not an assurance"):
        validate_evidence(evidence, required_assurance="pii_redacted")


def test_export_bundle_always_embeds_its_evidence():
    result = _sample_result()
    bundle = json.loads(result.export_bundle())
    assert bundle["schema"] == EXPORT_SCHEMA
    assert bundle["records"] == result.records_as_dicts()
    restored = DeidentificationEvidence.from_dict(bundle["evidence"])
    validate_evidence(restored, bundle["records"], required_assurance="safe_harbor")


def test_batch_run_produces_one_artifact_covering_every_record():
    pipeline = _pipeline({"zip": FieldSpec.zip_code(), "ssn": FieldSpec.remove(SafeHarborIdentifier.SOCIAL_SECURITY_NUMBERS)})
    result = _run(pipeline, {"zip": "02139", "ssn": "078-05-1120"}, {"zip": "94110"})
    assert result.records_as_dicts() == [{"zip": "021"}, {"zip": "941"}]
    assert len(result.evidence.records) == 2
    assert _disposition(result, SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS).generalized == 2
    assert _disposition(result, SafeHarborIdentifier.SOCIAL_SECURITY_NUMBERS).removed == 1
    with pytest.raises(DeidentificationError, match="unique"):
        pipeline.run(
            [SourceRecord("same", {"zip": "02139"}), SourceRecord("same", {"zip": "02139"})],
            operator=OPERATOR,
            attestation=_attestation(),
        )
