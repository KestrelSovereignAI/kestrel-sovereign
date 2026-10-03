"""Span detectors for identifiers embedded in text.

Structured fields are handled by their schema classification. Text is harder:
a clinical note, a free-form comment, or a mislabelled "non-identifying" string
can carry an identifier of any category. This module finds those spans.

Three sources contribute:

1. **Known values** — the record's own identifier values (its name, MRN,
   address, ...) wherever they recur in its text.
2. **Patterns** — every category with a recognizable surface form: e-mail,
   URL, IP and MAC addresses, SSN, telephone and fax numbers, labelled record /
   plan / account / licence / vehicle / device numbers, VINs, full dates and
   birth dates, ages over 89, street addresses, ZIP codes, and long digit runs.
3. **Entities** — names and places, which no pattern can find. These need a
   named-entity detector. The pipeline refuses free text when none is
   configured, because without one Safe Harbor category (A) cannot be covered.

Every source reads the text as written, before anything is replaced, and what
it finds there is removed even when an earlier replacement split it. A
detector shown only already-scrubbed text misses whatever a substitution
broke: the record's given name replaced inside "Avery Brown" leaves a surname
the entity model no longer recognizes, and a surname replaced inside
"12 Doe Street" leaves a house number no address pattern matches.

Detection is deliberately biased toward over-removal: replacing a lab value
that looks like a ZIP code loses a little utility, missing a ZIP code loses the
assurance. The PII redactor (:mod:`kestrel_sovereign.features.privacy.pii_detector`)
is reused only as an optional entity source; its own output never stands in for
de-identification.
"""

import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from kestrel_sovereign.deidentification.errors import DeidentificationError
from kestrel_sovereign.deidentification.identifiers import (
    AGE_90_OR_OLDER,
    SafeHarborIdentifier as Category,
    TransformationAction,
)

SCHEMA_DETECTOR = "schema"
KNOWN_VALUE_DETECTOR = "known_value"
BIRTH_DATE_DETECTOR = "pattern:birth_date"
DATE_DETECTOR = "pattern:date"
AGE_DETECTOR = "pattern:age"

# Already-replaced text no later pass may touch: placeholders, and the 90+
# aggregate (a record aged exactly 90 must not turn "90+" into "[...]+").
_PROTECTED_RE = re.compile(r"\[REDACTED:[A-Z_]+\]|(?<![\d.])90\+")
# Each round can expose at most a few new matches; real text settles in two or
# three. Text that is still changing after this many is refused, not passed.
_MAX_SCRUB_ROUNDS = 8


def redaction_placeholder(category: Category) -> str:
    """The token that replaces a removed free-text span of ``category``."""
    return f"[REDACTED:{category.name}]"


@dataclass(frozen=True)
class EntitySpan:
    """One identifier span reported by an :class:`EntityDetector`."""

    start: int
    end: int
    category: Category


class EntityDetector(Protocol):
    """Finds identifier spans no pattern can (names, places).

    ``name`` is recorded in the evidence artifact so a reviewer knows which
    detector the free-text coverage of category (A) rests on.
    """

    name: str

    def detect(self, text: str) -> Iterable[EntitySpan]:
        ...


class NerEntityDetector:
    """Named-entity detection backed by the spaCy model the PII redactor loads.

    PERSON spans map to names; GPE / LOC / FAC spans map to geographic
    subdivisions. States and countries are removed too — over-removal is the
    safe direction for an entity model that cannot tell a county from a nation.
    """

    name = "entity:spacy-ner"

    _CATEGORY_BY_PII_TYPE = {
        "PERSON": Category.NAMES,
        "GPE": Category.GEOGRAPHIC_SUBDIVISIONS,
        "LOC": Category.GEOGRAPHIC_SUBDIVISIONS,
    }

    def __init__(self, pii_detector) -> None:
        if not pii_detector.has_ner:
            raise DeidentificationError(
                "the PII detector has no named-entity model loaded"
            )
        self._pii_detector = pii_detector

    def detect(self, text: str) -> Iterable[EntitySpan]:
        # ``detect_entities``, not ``detect``: the redactor's own regexes would
        # otherwise shadow a name that sits inside one of their greedy matches.
        for match in self._pii_detector.detect_entities(text):
            category = self._CATEGORY_BY_PII_TYPE.get(match.pii_type.value)
            if category is not None:
                yield EntitySpan(match.start, match.end, category)


def default_entity_detector() -> Optional[EntityDetector]:
    """The spaCy-backed detector when a model is installed, else ``None``."""
    from kestrel_sovereign.features.privacy.pii_detector import get_pii_detector

    detector = get_pii_detector()
    if not detector.has_ner:
        return None
    return NerEntityDetector(detector)


# ── Pattern detectors ─────────────────────────────────────────────────────────

# A labelled identifier's value: a code of at least four characters that
# contains a digit, so "MRN: pending" and "serial q4h" are left alone but
# "MRN: A-10293" is not.
_CODE = r"(?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{3,}"
_NUMBER_LABEL = r"(?:\s+(?:number|num\.?|no\.?|#))?"
_SEPARATOR = r"\s*[:#]{0,2}\s*"
# North American numbers, and international numbers written with a leading
# "+" and separated groups (so "+500 mL" is not one).
_PHONE = (
    r"(?:(?<![\d-])(?:\+?\d{1,2}[-.\s]?)?(?:\(\d{3}\)\s?|\d{3}[-.\s])"
    r"\d{3}[-.\s]\d{4}(?![\d-])"
    r"|(?<![\w+])\+\d{1,3}(?:[\s.-]\(?\d{2,5}\)?){2,5}(?!\d))"
)
# A seven-digit local number reads exactly like a dose range ("500-1000"), so
# it is recognized only after a telephone or fax label.
_LOCAL_PHONE = r"(?<![\d-])\d{3}-\d{4}(?![\d-])"
_PHONE_CUE = r"(?:call|phone|tel|telephone|cell|mobile|ph|pager|contact)"
# Full month names in any case ("MARCH", "march"); abbreviations only
# capitalized or upper-case, since "may" and "dec" (decrease) are words.
_MONTH = (
    r"(?:(?i:January|February|March|April|June|July|August|September|October|"
    r"November|December)|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec|"
    r"JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEPT|SEP|OCT|NOV|DEC)"
)
# Words that make a bare month/day ("3/15") a date rather than a score
# ("pain 8/10").
_DATE_CUE = (
    r"(?i:\b(?:on|since|from|until|till|by|before|after|admitted|discharged|"
    r"seen|visit(?:ed)?|dated?|DOS|appt|appointment|scheduled|surgery|"
    r"procedure)\b\s*[:\-]?\s*)"
)
_ORDINAL = r"(?:st|nd|rd|th)?"


@dataclass(frozen=True)
class _PatternDetector:
    """A regex whose ``group`` is replaced by the category placeholder.

    Labels outside the group ("MRN:", "fax") are kept so the text stays
    readable. ``refutes`` marks detectors precise enough to prove that a value
    the caller called non-identifying is in fact an identifier.
    """

    name: str
    category: Category
    pattern: "re.Pattern[str]"
    group: int = 0
    refutes: bool = True


def _labelled(name: str, category: Category, label: str) -> _PatternDetector:
    return _PatternDetector(
        name,
        category,
        re.compile(rf"\b(?:{label}){_NUMBER_LABEL}{_SEPARATOR}({_CODE})", re.IGNORECASE),
        group=1,
    )


# Each pass runs on the text the previous pass produced, and a span replaced
# once is never matched again. Container patterns therefore run first: an
# e-mail address or URL can embed a name or number, and a later pass that
# replaced only that part would leave the rest of the address behind.
_CONTAINER_DETECTORS: Tuple[_PatternDetector, ...] = (
    _PatternDetector(
        "pattern:email",
        Category.EMAIL_ADDRESSES,
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    _PatternDetector(
        "pattern:url",
        Category.WEB_URLS,
        re.compile(r"(?:\b(?:https?|ftp)://|\bwww\.)[^\s<>\"'\])]+", re.IGNORECASE),
    ),
    # A web address written without a scheme ("janedoe-art.com/about").
    _PatternDetector(
        "pattern:domain",
        Category.WEB_URLS,
        re.compile(
            r"(?<![@\w.-])(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
            r"(?:com|org|net|edu|gov|mil|int|info|biz|io|co|us|uk|ca|au|nz|de|"
            r"fr|nl|eu|me|app|dev|ai|health|care|online|site|xyz)\b"
            r"(?:/[^\s<>\"'\])]*)?",
            re.IGNORECASE,
        ),
    ),
)
_STRUCTURED_DETECTORS: Tuple[_PatternDetector, ...] = (
    _PatternDetector(
        "pattern:mac_address",
        Category.DEVICE_IDENTIFIERS,
        re.compile(
            r"(?<![0-9A-Fa-f:-])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:-])"
        ),
    ),
    _PatternDetector(
        "pattern:ipv4",
        Category.IP_ADDRESSES,
        re.compile(
            r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\.?\d)"
        ),
    ),
    _PatternDetector(
        "pattern:ipv6",
        Category.IP_ADDRESSES,
        re.compile(
            r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){3,7}[0-9A-Fa-f]{1,4}(?![\w:])"
            r"|(?<![\w:])[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*::"
            r"(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*)?(?![\w:])"
            # Leading compression ("::1", "::abcd:1234"). A bare "::" names no
            # host and is left alone.
            r"|(?<![\w:])::[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*(?![\w:])"
        ),
    ),
    _PatternDetector(
        "pattern:ssn_labelled",
        Category.SOCIAL_SECURITY_NUMBERS,
        re.compile(
            rf"\b(?:SSN|social\s+security){_NUMBER_LABEL}{_SEPARATOR}"
            r"(\d{3}[- ]?\d{2}[- ]?\d{4})(?!\d)",
            re.IGNORECASE,
        ),
        group=1,
    ),
    _PatternDetector(
        "pattern:ssn",
        Category.SOCIAL_SECURITY_NUMBERS,
        re.compile(r"(?<![\d-])\d{3}(?P<sep>[- ])\d{2}(?P=sep)\d{4}(?![\d-])"),
    ),
    _PatternDetector(
        "pattern:fax",
        Category.FAX_NUMBERS,
        re.compile(
            rf"\bfax{_NUMBER_LABEL}{_SEPARATOR}((?:{_PHONE}|{_LOCAL_PHONE}))",
            re.IGNORECASE,
        ),
        group=1,
    ),
    _PatternDetector(
        "pattern:telephone", Category.TELEPHONE_NUMBERS, re.compile(_PHONE)
    ),
    _PatternDetector(
        "pattern:telephone_local",
        Category.TELEPHONE_NUMBERS,
        re.compile(
            rf"\b{_PHONE_CUE}\b\.?[^\d\n]{{0,12}}?({_LOCAL_PHONE})", re.IGNORECASE
        ),
        group=1,
    ),
    _labelled(
        "pattern:medical_record_number",
        Category.MEDICAL_RECORD_NUMBERS,
        r"MRN|MR|medical\s+record",
    ),
    _labelled(
        "pattern:health_plan_number",
        Category.HEALTH_PLAN_BENEFICIARY_NUMBERS,
        r"member\s+id|health\s+plan(?:\s+beneficiary)?(?:\s+id)?|"
        r"beneficiary(?:\s+id)?|subscriber(?:\s+id)?|policy(?:\s+id)?|"
        r"insurance(?:\s+id)?|medicare(?:\s+id)?|medicaid(?:\s+id)?|MBI|HICN",
    ),
    _labelled(
        "pattern:account_number", Category.ACCOUNT_NUMBERS, r"account|acct\.?"
    ),
    # Vehicle labels run before licence labels: "license plate" names a vehicle.
    _labelled(
        "pattern:vehicle_identifier",
        Category.VEHICLE_IDENTIFIERS,
        r"VIN|vehicle\s+identification|license\s+plate|licence\s+plate|plate",
    ),
    _labelled(
        "pattern:certificate_license_number",
        Category.CERTIFICATE_LICENSE_NUMBERS,
        r"(?:driver'?s\s+)?licen[cs]e|certificate|cert\.?",
    ),
    _labelled(
        "pattern:device_identifier",
        Category.DEVICE_IDENTIFIERS,
        # A bare "serial" is clinical ("serial troponins"); it must be labelled.
        r"serial(?=\s*(?:number|num\b|no\b|#|:))|S/N|"
        r"device(?:\s+id(?:entifier)?)?|UDI|IMEI",
    ),
    _PatternDetector(
        "pattern:vin",
        Category.VEHICLE_IDENTIFIERS,
        re.compile(
            r"\b(?=[A-HJ-NPR-Z0-9]*\d)(?=[A-HJ-NPR-Z0-9]*[A-HJ-NPR-Z])"
            r"[A-HJ-NPR-Z0-9]{17}\b"
        ),
    ),
    _labelled(
        "pattern:labelled_identifier",
        Category.OTHER_UNIQUE_IDENTIFIERS,
        r"id|identifier|(?:patient|client|case|record|employee|student|member)"
        r"\s+(?:id|number|no\.?|#)",
    ),
)

# Dates carry a ``year`` group when they name one. Generalization keeps that
# year (or aggregates it when it is 90 or more years old); a date without a
# four-digit year is removed outright.
_DATE_PATTERNS: Tuple["re.Pattern[str]", ...] = (
    re.compile(
        r"\b(?P<year>\d{4})-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])"
        r"(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?(?!\d)"
    ),
    re.compile(r"\b(?P<year>\d{4})(?P<sep>[/.])\d{1,2}(?P=sep)\d{1,2}\b"),
    re.compile(r"\b\d{1,2}(?P<sep>[/.-])\d{1,2}(?P=sep)(?P<year>\d{4})\b"),
    # A two-digit year cannot be kept: its century is ambiguous.
    re.compile(r"\b\d{1,2}(?P<sep>[/-])\d{1,2}(?P=sep)\d{2}\b"),
    re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?P<year>\d{4})\b"),
    re.compile(r"\b(?P<year>\d{4})-(?:0[1-9]|1[0-2])\b(?!-\d)"),
    # Month and day without a year, after a date cue; only the ``date`` group
    # is replaced.
    re.compile(
        rf"{_DATE_CUE}(?P<date>(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01]))\b(?!/\d)"
    ),
    # "15-Mar-2024", "Mar-15-24", "Mar-2024".
    re.compile(
        rf"\b\d{{1,2}}[-/.]{_MONTH}\.?[-/.](?:(?P<year>\d{{4}})|\d{{2}})\b"
    ),
    re.compile(
        rf"\b{_MONTH}\.?[-/.]\d{{1,2}}[-/.](?:(?P<year>\d{{4}})|\d{{2}})\b"
    ),
    re.compile(rf"\b{_MONTH}[-/](?P<year>\d{{4}})\b"),
    re.compile(
        rf"\b{_MONTH}\.?\s+\d{{1,2}}{_ORDINAL}\b(?:,?\s+(?P<year>\d{{4}})\b)?"
    ),
    re.compile(
        rf"\b\d{{1,2}}{_ORDINAL}\s+(?:of\s+)?{_MONTH}\b\.?(?:,?\s+(?P<year>\d{{4}})\b)?"
    ),
    re.compile(rf"\b{_MONTH}\.?,?\s+(?P<year>\d{{4}})\b"),
)
# Removed from text, but too ambiguous to refute a non-identifying value: a
# dotted two-digit-year date is spelled like a "3.10.12" version string.
_SCRUB_ONLY_DATE_PATTERNS: Tuple["re.Pattern[str]", ...] = (
    re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2}\b"),
)
_BIRTH_CUE = re.compile(
    r"\b(?:born(?:\s+(?:on|in))?|DOB|D\.O\.B\.?|date\s+of\s+birth|"
    r"birth\s*date|birthday|YOB|year\s+of\s+birth|birth\s*year)"
    r"(?:\s*[:\-]\s*|\s+)",
    re.IGNORECASE,
)
_BARE_YEAR = re.compile(r"\b(?P<year>\d{4})\b")
# How far past a birth cue its date may sit ("date of birth was 03/15/1931"),
# without crossing into the next sentence.
_BIRTH_CUE_WINDOW = 40
_SENTENCE_BREAK = re.compile(r"[;\n]|\.\s")
# An age, with any fractional part and an open-ended "+": "95.5", "95 1/2",
# "95-½", and "95+" are matched whole, so each is compared as a number and
# replaced as one span rather than leaving ".5", "1/2", or "+" behind "90+".
# The aggregate "90+" itself matches too and is skipped by ``_age_matches``.
_AGE_FRACTION = r"(?:\.\d+|[\s-]?(?:[13][/⁄][24]|[½¼¾]))"
_AGE_NUMBER = rf"\d{{2,3}}{_AGE_FRACTION}?\+?"
_AGE_DECIMAL = re.compile(r"\d{2,3}(?:\.\d+)?")
# Hyphen-minus, hyphen, non-breaking hyphen, and en dash ("95–year–old").
_DASH = r"[-‐‑–]"
_AGE_UNIT = rf"\s*(?:{_DASH}\s*)?(?:years?|yrs?|y/o|y\.\s?o\.?|yo)(?![A-Za-z])"
# What may stand between "age" and its number: a qualifier ("age at death"),
# a unit in parentheses ("Age (years)"), a separator, and a comparison
# ("age >/= 95", "age approx. 95"). Every optional part begins with a
# non-space character, so a run of whitespace can be matched only one way.
# "weight-for-age" is growth-chart language, followed by a percentile;
# "AGEs" are advanced glycation end-products, so "ages" is not upper case,
# and it takes no qualifier ("the ages of 120 patients" is a count).
_AGE_CUE = (
    r"(?<!for-)\b(?:(?:age(?:\s*/\s*sex)?|aged)"
    r"(?:\s+(?:of|is|was|at\s+(?:first\s+|initial\s+|last\s+)?"
    r"(?:death|diagnosis|dx|onset|admission|presentation|enrol+ment|baseline|"
    r"surgery|interview|visit|exam|screening|transplant|delivery)))?"
    r"|(?-i:[Aa]ges))\s*"
    r"(?:\((?:years?|yrs?|y)\)\s*)?"
    rf"(?:[:=~]\s*|{_DASH}\s*){{0,2}}"
    r"(?:(?:>/?=?|≥|over|above|approx(?:\.|imately)?|about|around)\s*)?"
)
_AGE_PATTERNS: Tuple["re.Pattern[str]", ...] = (
    re.compile(rf"(?<![\d.])(?P<age>{_AGE_NUMBER})(?={_AGE_UNIT})", re.IGNORECASE),
    # The lower bound of a range: "92-95 years old", "between 92 and 95 yrs".
    # Not a reading ("120/90 - 95 years"), and not a number above its "upper
    # bound" ("150/95 - 10 years ago", "Bed 412 - 92 y/o F"; see _age_matches).
    re.compile(
        rf"(?<![\d./])(?P<age>{_AGE_NUMBER})(?=\s*(?:{_DASH}|to|or|and)\s*"
        rf"(?P<upper>{_AGE_NUMBER}){_AGE_UNIT})",
        re.IGNORECASE,
    ),
    # After the cue the number may run straight into a unit ("Age 95.5y"),
    # but not on into another digit, an ordinal, or a percentage ("age 95%
    # CI"); "95+ %" is still the open-ended age.
    re.compile(
        rf"{_AGE_CUE}(?P<age>{_AGE_NUMBER})"
        r"(?!\.?\d|(?:st|nd|rd|th)\b|(?<!\+)\s?%)",
        re.IGNORECASE,
    ),
    # "95F", "92yoM", "92YOF", "93 y/o F". Not after a letter, so "T790M" (a
    # mutation) is not an age, and no bare space before the sex, so a "101 F"
    # temperature is not read as one.
    re.compile(
        r"(?<![\w.])(?P<age>\d{2,3})(?:\s?(?i:yo|y/o)\s?(?i:[mf])|[MF])\b"
    ),
    # A fraction only with an explicit "yo": "95.5F" is a temperature.
    re.compile(
        rf"(?<![\w.])(?P<age>\d{{2,3}}{_AGE_FRACTION})\s?(?i:yo|y/o)\s?(?i:[mf])\b"
    ),
)
_GEOGRAPHIC_DETECTORS: Tuple[_PatternDetector, ...] = (
    _PatternDetector(
        "pattern:street_address",
        Category.GEOGRAPHIC_SUBDIVISIONS,
        # The street name must be capitalized, so "2 weeks with Dr. Lee" and
        # "1 mm ST depression" are not read as addresses.
        re.compile(
            r"\b\d{1,6}\s+(?:[A-Z][A-Za-z0-9.'-]*\s+){1,4}"
            r"(?i:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
            r"Court|Ct|Way|Place|Pl|Terrace|Ter|Circle|Cir|Highway|Hwy|"
            r"Parkway|Pkwy)\b\.?"
            r"(?:,?\s+(?i:Apt|Apartment|Suite|Ste|Unit|#)\.?\s*[A-Za-z0-9-]+)?"
        ),
    ),
    _PatternDetector(
        "pattern:po_box",
        Category.GEOGRAPHIC_SUBDIVISIONS,
        re.compile(r"\bP\.?\s*O\.?\s*Box\s+\d+\b", re.IGNORECASE),
    ),
    # A bare five-digit number is often not a ZIP code, so it removes text but
    # does not refute a non-identifying classification on its own.
    _PatternDetector(
        "pattern:zip_code",
        Category.GEOGRAPHIC_SUBDIVISIONS,
        re.compile(r"(?<![\d-])\d{5}(?:-\d{4})?(?![\d-])"),
        refutes=False,
    ),
    _PatternDetector(
        "pattern:long_number",
        Category.OTHER_UNIQUE_IDENTIFIERS,
        re.compile(r"(?<!\d)\d{6,}(?!\d)"),
        refutes=False,
    ),
)

#: Every detector a pipeline records, in the order they apply: the schema
#: classification of structured fields, then the free-text passes. The entity
#: detector's own name is appended when one is configured.
DETECTOR_NAMES: Tuple[str, ...] = (
    (SCHEMA_DETECTOR,)
    + tuple(d.name for d in _CONTAINER_DETECTORS)
    + (KNOWN_VALUE_DETECTOR,)
    + tuple(d.name for d in _STRUCTURED_DETECTORS)
    + (BIRTH_DATE_DETECTOR, DATE_DETECTOR, AGE_DETECTOR)
    + tuple(d.name for d in _GEOGRAPHIC_DETECTORS)
)


# ── Span application ─────────────────────────────────────────────────────────

CountKey = Tuple[Category, TransformationAction, str]


@dataclass(frozen=True)
class _Match:
    start: int
    end: int
    replacement: str
    category: Category
    action: TransformationAction
    detector: str


def _uncovered_segments(
    match: _Match, cursor: int, protected: Sequence[Tuple[int, int]], text: str
) -> List[_Match]:
    """The parts of ``match`` not already replaced, as matches of their own.

    A span that overlaps an earlier replacement (in this pass, or a placeholder
    from a previous one) is trimmed rather than dropped: dropping it would keep
    whatever part of it lay outside the overlap — the surname after a removed
    given name, say. A trimmed piece loses its generalization and is replaced
    by the category placeholder.
    """
    start = max(match.start, cursor)
    if start >= match.end:
        return []
    bounds: List[Tuple[int, int]] = []
    position = start
    for p_start, p_end in protected:
        if p_end <= position or p_start >= match.end:
            continue
        if p_start > position:
            bounds.append((position, p_start))
        position = max(position, p_end)
    if position < match.end:
        bounds.append((position, match.end))
    if bounds == [(match.start, match.end)]:
        return [match]
    segments = []
    for seg_start, seg_end in bounds:
        piece = text[seg_start:seg_end]
        core = piece.strip()
        if not core:
            continue
        seg_start += len(piece) - len(piece.lstrip())
        segments.append(_Match(
            seg_start, seg_start + len(core), redaction_placeholder(match.category),
            match.category, TransformationAction.TRANSFORMED, match.detector,
        ))
    return segments


def _apply(
    text: str, origin: Sequence[int], matches: Iterable[_Match], counts: Dict[CountKey, int]
) -> Tuple[str, List[int]]:
    """Replace matches (earliest, then longest first) without re-matching a
    span that is already a placeholder.

    ``origin`` maps each character of ``text`` to its index in the source text,
    or ``-1`` for a character a replacement wrote; the returned origin does the
    same for the returned text.
    """
    protected = [(m.start(), m.end()) for m in _PROTECTED_RE.finditer(text)]
    chosen: List[_Match] = []
    cursor = 0
    for match in sorted(matches, key=lambda m: (m.start, m.start - m.end)):
        if text[match.start:match.end] == match.replacement:
            continue  # a year already kept by an earlier round is not a new span
        for segment in _uncovered_segments(match, cursor, protected, text):
            chosen.append(segment)
            cursor = segment.end
    if not chosen:
        return text, list(origin)
    pieces: List[str] = []
    new_origin: List[int] = []
    position = 0
    for match in chosen:
        pieces.append(text[position:match.start])
        new_origin.extend(origin[position:match.start])
        pieces.append(match.replacement)
        new_origin.extend([-1] * len(match.replacement))
        position = match.end
        key = (match.category, match.action, match.detector)
        counts[key] = counts.get(key, 0) + 1
    pieces.append(text[position:])
    new_origin.extend(origin[position:])
    return "".join(pieces), new_origin


def _surviving_source_spans(
    source_matches: Iterable[_Match], text: str, origin: Sequence[int]
) -> List[_Match]:
    """What survives of matches found in the source text, as matches in ``text``.

    Each run of characters still carried verbatim from inside a source match
    becomes a span of that match's category, replaced by its placeholder. Runs
    of punctuation alone carry no identifier and are left in place.
    """
    position = {source: index for index, source in enumerate(origin) if source >= 0}
    spans: List[_Match] = []
    for match in source_matches:
        indices = [position[i] for i in range(match.start, match.end) if i in position]
        runs: List[List[int]] = []
        for index in indices:
            if runs and index == runs[-1][1]:
                runs[-1][1] = index + 1
            else:
                runs.append([index, index + 1])
        for run_start, run_end in runs:
            piece = text[run_start:run_end]
            core = piece.strip()
            if not any(ch.isalnum() for ch in core):
                continue
            start = run_start + len(piece) - len(piece.lstrip())
            spans.append(_Match(
                start, start + len(core), redaction_placeholder(match.category),
                match.category, TransformationAction.TRANSFORMED, match.detector,
            ))
    return spans


def _pattern_matches(detector: _PatternDetector, text: str) -> List[_Match]:
    placeholder = redaction_placeholder(detector.category)
    return [
        _Match(
            m.start(detector.group),
            m.end(detector.group),
            placeholder,
            detector.category,
            TransformationAction.TRANSFORMED,
            detector.name,
        )
        for m in detector.pattern.finditer(text)
    ]


def _known_value_pattern(literal: str, *, joined: bool) -> "re.Pattern[str]":
    """Match ``literal`` as a whole token.

    With ``joined`` false, a digit-bearing literal must also not be joined to a
    larger structure by a separator: the year "1931" must not match inside
    "04/02/1931", or replacing it would strand "04/02/" where the date pass can
    no longer see it. Such structures are left whole for the structured passes,
    and the closing known-value pass (``joined`` true) catches what remains.
    """
    escaped = re.escape(literal)
    if joined or not any(ch.isdigit() for ch in literal):
        return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
    return re.compile(rf"(?<![\w/.:#@-]){escaped}(?![\w/:#@-]|\.\w)", re.IGNORECASE)


def _known_value_matches(
    text: str, known_values: Sequence[Tuple[str, Category]], *, joined: bool
) -> List[_Match]:
    matches = []
    for literal, category in known_values:
        pattern = _known_value_pattern(literal, joined=joined)
        placeholder = redaction_placeholder(category)
        for m in pattern.finditer(text):
            matches.append(_Match(
                m.start(), m.end(), placeholder, category,
                TransformationAction.TRANSFORMED, KNOWN_VALUE_DETECTOR,
            ))
    return matches


def _year_replacement(year: Optional[str], reference_year: int) -> Tuple[str, TransformationAction]:
    """Keep a date's year, aggregate it to 90+ when it is that old, or remove it.

    A date element 90 or more years before the reference date is indicative of
    an age over 89 for the person it relates to (a birth year, or an admission
    they were alive for), which Safe Harbor aggregates.
    """
    if year is None:
        return redaction_placeholder(Category.DATES), TransformationAction.TRANSFORMED
    if reference_year - int(year) >= 90:
        return AGE_90_OR_OLDER, TransformationAction.GENERALIZED
    return year, TransformationAction.GENERALIZED


def _date_span(match: "re.Match[str]") -> Tuple[int, int]:
    """The span to replace: a cued pattern's ``date`` group, else the match."""
    if match.groupdict().get("date") is not None:
        return match.span("date")
    return match.span()


def _birth_date_matches(text: str, reference_year: int) -> List[_Match]:
    """The date following a birth cue, including a bare year ("born 1931")."""
    matches = []
    for cue in _BIRTH_CUE.finditer(text):
        position = cue.end()
        found = None
        for pattern in _DATE_PATTERNS + _SCRUB_ONLY_DATE_PATTERNS + (_BARE_YEAR,):
            candidate = pattern.search(text, position)
            if (
                candidate is None
                or candidate.start() - position > _BIRTH_CUE_WINDOW
                or _SENTENCE_BREAK.search(text, position, candidate.start())
            ):
                continue
            if found is None or (candidate.start(), -candidate.end()) < (found.start(), -found.end()):
                found = candidate
        if found is None:
            continue
        replacement, action = _year_replacement(
            found.groupdict().get("year"), reference_year
        )
        start, end = _date_span(found)
        matches.append(_Match(
            start, end, replacement, Category.DATES, action, BIRTH_DATE_DETECTOR,
        ))
    return matches


def _date_matches(text: str, reference_year: int) -> List[_Match]:
    matches = []
    for pattern in _DATE_PATTERNS + _SCRUB_ONLY_DATE_PATTERNS:
        for m in pattern.finditer(text):
            replacement, action = _year_replacement(
                m.groupdict().get("year"), reference_year
            )
            start, end = _date_span(m)
            matches.append(_Match(
                start, end, replacement, Category.DATES, action, DATE_DETECTOR,
            ))
    return matches


def parse_date_year(text: str) -> Optional[int]:
    """The four-digit year of a string that is wholly one recognized date."""
    stripped = text.strip()
    for pattern in _DATE_PATTERNS + _SCRUB_ONLY_DATE_PATTERNS:
        match = pattern.fullmatch(stripped)
        if match and match.groupdict().get("year"):
            return int(match.group("year"))
    return None


def _age_value(age: str) -> float:
    """An age match as a number, for comparison with 89 only.

    A written fraction ("1/2", "¾") counts as at least a quarter: what matters
    is that any fraction puts 89 over 89.
    """
    number = _AGE_DECIMAL.match(age)
    value = float(number.group())
    if number.end() < len(age.rstrip("+")):
        value += 0.25
    return value


def _age_matches(text: str) -> List[_Match]:
    matches = []
    for pattern in _AGE_PATTERNS:
        for m in pattern.finditer(text):
            age = m.group("age")
            # The aggregate Safe Harbor permits, including its digits alone
            # when a lookahead backtracked off its "+" ("age 90+5").
            if text.startswith(AGE_90_OR_OLDER, m.start("age")):
                continue
            # Not the lower bound of a range: it is above its upper bound. The
            # upper bound reads every age form, so a true range is caught in
            # the round (and in the source text) before that bound is
            # rewritten to the aggregate.
            upper = m.groupdict().get("upper")
            if upper is not None and _age_value(age) > _age_value(upper):
                continue
            # Numerically, as the structured age field is: 89.5 is over 89.
            if _age_value(age) > 89:
                matches.append(_Match(
                    m.start("age"), m.end("age"), AGE_90_OR_OLDER, Category.DATES,
                    TransformationAction.GENERALIZED, AGE_DETECTOR,
                ))
    return matches


def _entity_matches(text: str, detector: EntityDetector) -> List[_Match]:
    matches = []
    for span in detector.detect(text):
        if not (0 <= span.start < span.end <= len(text)):
            raise DeidentificationError(
                f"entity detector {detector.name!r} returned a span outside the text"
            )
        if not isinstance(span.category, Category):
            raise DeidentificationError(
                f"entity detector {detector.name!r} returned a non-Safe-Harbor category"
            )
        matches.append(_Match(
            span.start, span.end, redaction_placeholder(span.category),
            span.category, TransformationAction.TRANSFORMED, detector.name,
        ))
    return matches


def _scrub_passes(
    known_values: Sequence[Tuple[str, Category]], reference_year: int
) -> List[Callable[[str], List[_Match]]]:
    passes: List[Callable[[str], List[_Match]]] = [
        (lambda t, d=detector: _pattern_matches(d, t))
        for detector in _CONTAINER_DETECTORS
    ]
    passes.append(lambda t: _known_value_matches(t, known_values, joined=False))
    passes.extend(
        (lambda t, d=detector: _pattern_matches(d, t))
        for detector in _STRUCTURED_DETECTORS
    )
    passes.append(lambda t: _birth_date_matches(t, reference_year))
    passes.append(lambda t: _date_matches(t, reference_year))
    passes.append(_age_matches)
    passes.extend(
        (lambda t, d=detector: _pattern_matches(d, t))
        for detector in _GEOGRAPHIC_DETECTORS
    )
    passes.append(lambda t: _known_value_matches(t, known_values, joined=True))
    return passes


def scrub_free_text(
    text: str,
    *,
    known_values: Sequence[Tuple[str, Category]],
    reference_year: int,
    entity_detector: EntityDetector,
) -> Tuple[str, Dict[CountKey, int]]:
    """Remove or generalize every detectable identifier span in ``text``.

    Returns the scrubbed text and, per (category, action, detector), how many
    spans were replaced.

    Every detector — each pattern and known-value pass, and the entity
    detector — first reads the source text. The passes then run in order, each
    on the previous pass's output, so a span replaced once is never matched
    again; that ordering is what keeps a container (an e-mail address, a date)
    whole when a smaller match sits inside it. Once they settle, any character still carried
    verbatim from inside a source-text detection is replaced too. The result
    is the union: no detector's finding in the source survives because an
    earlier replacement split it.

    Known values run twice. The first pass leaves a digit-bearing value alone
    where it is joined into a larger structure (a date, a code) so that
    structure is removed whole. The closing pass matches on word boundaries
    alone, catching what remains — including a value that only appears once a
    date is generalized: "04/02/1931" becomes "1931", and when that is the
    record's own 90+ birth year it must not survive as a bare year.

    The pattern passes repeat until the text stops changing, because a
    generalization can expose a new match: "Insurance: Jan 5, 2024" becomes
    "Insurance: 2024", which now reads as a labelled plan number. The entity
    detector reads the source and the settled text once each, between two
    such settlements, so a context-sensitive model cannot keep the loop from
    settling. Its placeholders cannot create a new pattern match, and the
    second settlement confirms that. The result is stable under every pattern,
    which is what lets :func:`find_identifier_patterns` treat any later hit as
    a residual.
    """
    counts: Dict[CountKey, int] = {}
    passes = _scrub_passes(known_values, reference_year)
    # A match that keeps its own text (a recent birth year after "born") is a
    # decision to keep, not a span to remove.
    source_matches = [
        match
        for find in passes
        for match in find(text)
        if text[match.start:match.end] != match.replacement
    ]
    source_matches.extend(_entity_matches(text, entity_detector))
    current, origin = _settle(text, range(len(text)), passes, counts)
    current, origin = _apply(
        current,
        origin,
        _surviving_source_spans(source_matches, current, origin)
        + _entity_matches(current, entity_detector),
        counts,
    )
    current, _ = _settle(current, origin, passes, counts)
    return current, counts


def _settle(
    text: str,
    origin: Sequence[int],
    passes: Sequence[Callable[[str], List[_Match]]],
    counts: Dict[CountKey, int],
) -> Tuple[str, List[int]]:
    origin = list(origin)
    for _ in range(_MAX_SCRUB_ROUNDS):
        before = text
        for find in passes:
            text, origin = _apply(text, origin, find(text), counts)
        if text == before:
            return text, origin
    raise DeidentificationError(
        f"free text did not reach a stable de-identified form in "
        f"{_MAX_SCRUB_ROUNDS} rounds"
    )


def _retained_birth_dates(text: str, reference_year: int) -> bool:
    """Whether a birth cue in ``text`` is followed by a date Safe Harbor does
    not permit: one with a day or month, or a year 90 or more years before
    ``reference_year`` ("DOB: 1931"). A recent birth year alone is permitted."""
    return any(
        text[match.start:match.end] != match.replacement
        for match in _birth_date_matches(text, reference_year)
    )


def find_identifier_patterns(
    text: str,
    *,
    reference_year: int,
    known_values: Sequence[Tuple[str, Category]] = (),
) -> List[Tuple[Category, str]]:
    """Precise identifier patterns present in ``text`` as (category, detector).

    Used to refuse a value the caller classified as non-identifying, and to
    check a de-identified record for residual identifiers before it is saved.
    Only detectors precise enough to be evidence are consulted; a bare
    five-digit number or a long digit run alone does not refute.
    ``reference_year`` is the run's: a cued birth year is an identifier when it
    is 90 or more years before it.
    """
    findings: List[Tuple[Category, str]] = []
    for literal, category in known_values:
        if _known_value_pattern(literal, joined=True).search(text):
            findings.append((category, KNOWN_VALUE_DETECTOR))
    for detector in _CONTAINER_DETECTORS + _STRUCTURED_DETECTORS + _GEOGRAPHIC_DETECTORS:
        if detector.refutes and detector.pattern.search(text):
            findings.append((detector.category, detector.name))
    if _retained_birth_dates(text, reference_year):
        findings.append((Category.DATES, BIRTH_DATE_DETECTOR))
    if any(pattern.search(text) for pattern in _DATE_PATTERNS):
        findings.append((Category.DATES, DATE_DETECTOR))
    if _age_matches(text):
        findings.append((Category.DATES, AGE_DETECTOR))
    return findings
