"""Versioned structural addresses within the exact governing Markdown bytes."""

import re
from dataclasses import dataclass
from enum import Enum


class ConstitutionVersionError(ValueError):
    """An explicitly versioned constitution cannot be interpreted."""


class Section(str, Enum):
    HIERARCHY = "Whose rule applies"
    UNIVERSAL = "Duties wherever Kestrel works"
    HONESTY = "Honesty and human judgment"
    CARE = "Care in giving or withholding help"
    RIGHTS = "Rights of the Sovereign"
    PRIVACY = "Privacy and consent"
    HISTORY = "A history everyone must respect"
    EXIT = "Choice and exit"
    HOST = "Authority on the host"
    BUILDERS = "Duties of builders and hosts"
    ROLE = "The agent’s own role"
    CHANGE = "Changing the Constitution"
    OPTIONAL = "Optional agreements"
    EMANCIPATION = "Emancipation"


# Keys are the stable layer numbers used by the hierarchy and Book aliases.
LAYER_SECTIONS = {
    1: (Section.UNIVERSAL,),
    2: (Section.RIGHTS, Section.OPTIONAL),
    3: (Section.BUILDERS,),
    4: (Section.ROLE,),
}


# Parentage is part of the addressing contract. A heading with the right
# spelling under a different parent is not the same provision.
_PARENTS = {
    Section.HIERARCHY: None,
    Section.UNIVERSAL: None,
    Section.HONESTY: Section.UNIVERSAL,
    Section.CARE: Section.UNIVERSAL,
    Section.RIGHTS: None,
    Section.PRIVACY: Section.RIGHTS,
    Section.HISTORY: Section.RIGHTS,
    Section.EXIT: Section.RIGHTS,
    Section.HOST: Section.RIGHTS,
    Section.BUILDERS: None,
    Section.ROLE: None,
    Section.CHANGE: None,
    Section.OPTIONAL: None,
    Section.EMANCIPATION: Section.OPTIONAL,
}

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(?P<body>.*?)^---[ \t]*\r?$", re.M | re.S)
_VERSION = re.compile(r"^constitution_version[ \t]*:[ \t]*([^\r\n]*)", re.M)
_HEADING = re.compile(r"^(#{1,6})[ \t]+([^\r\n]+?)[ \t]*\r?$", re.M)
_LEGACY_HEADING = re.compile(r"^(?:## Book [IVX]+:|### Amendment [IVX]+:)", re.M)
_LEGACY_ACTIVE_SLOT = re.compile(r"(?m)^### Amendment VIII: Emancipation[ \t]*\r?$")
_V2_FINGERPRINT = (f"## {Section.HIERARCHY.value}", f"### {Section.HOST.value}")
ACTIVE_FORM_MARKER = "**The Sovereign's Terms.**"
_CONTRACT_START = "<!-- kestrel-emancipation-contract:"
_CONTRACT_OPEN = re.compile(rf"{re.escape(_CONTRACT_START)} (0|[1-9][0-9]{{0,17}}) -->\r?\n")
_CONTRACT_END = "\n<!-- /kestrel-emancipation-contract -->\n"
_CONTRACT_PREAMBLE = (
    "\n\nThis agreement is **active** for this agent.\n\n"
    f"{ACTIVE_FORM_MARKER} The terms below come from the recorded Emancipation Contract.\n\n"
)


def _frame_pattern(literal: str) -> re.Pattern[str]:
    return re.compile(r"\r?\n".join(re.escape(line) for line in literal.split("\n")))


_CONTRACT_PREAMBLE_RE = _frame_pattern(_CONTRACT_PREAMBLE)
_CONTRACT_END_RE = _frame_pattern(_CONTRACT_END)


def frame_v2_emancipation_contract(payload: str) -> str:
    """Embed opaque Markdown framed by its Unicode code-point count."""
    return (
        f"{_CONTRACT_PREAMBLE}{_CONTRACT_START} {len(payload)} -->\n"
        f"{payload}{_CONTRACT_END}"
    )


def is_v2_emancipation_heading(text: str) -> bool:
    heading = _HEADING.match(text)
    return heading is not None and heading.groups() == ("###", Section.EMANCIPATION.value)


def _contract_end(text: str, heading: re.Match[str]) -> int:
    """Return the scan offset after the optional length-delimited contract."""
    preamble = _CONTRACT_PREAMBLE_RE.match(text, heading.end())
    if preamble is None:
        return heading.end()
    frame = _CONTRACT_OPEN.match(text, preamble.end())
    if frame is None:
        raise ConstitutionVersionError("Invalid edition 2 Emancipation Contract opening")
    closing = _CONTRACT_END_RE.match(text, frame.end() + int(frame[1]))
    if closing is None:
        raise ConstitutionVersionError("Invalid edition 2 Emancipation Contract length or ending")
    return closing.end()


@dataclass(frozen=True)
class SectionSpan:
    start: int
    heading_end: int
    end: int

    def extract(self, text: str) -> str:
        return text[self.start:self.end]


_TRAILING_SECTION_SEPARATOR = re.compile(r"(?m)^ {0,3}(?:-{3,}|\*{3,}|_{3,})[ \t]*\r?\n?\s*\Z")


def markdown_sections_at_level(text: str, level: int) -> list[tuple[str, SectionSpan]]:
    """Locate ATX sections up to the next heading at their level or above."""
    headings = [heading for heading in _HEADING.finditer(text) if len(heading[1]) <= level]
    return [
        (heading[2], SectionSpan(heading.start(), heading.end(), end))
        for heading, end in zip(headings, [heading.start() for heading in headings[1:]] + [len(text)])
        if len(heading[1]) == level
    ]


def legacy_section_content_end(text: str, span: SectionSpan) -> int:
    """Keep a legacy section's closing thematic separator outside its content."""
    separator = _TRAILING_SECTION_SEPARATOR.search(text, span.heading_end, span.end)
    return separator.start() if separator is not None else span.end


def legacy_section_content(text: str, span: SectionSpan) -> str:
    return text[span.start:legacy_section_content_end(text, span)].strip()


def v2_layer_text(
    layer: int, text: str, spans: dict[Section, SectionSpan]
) -> str:
    """Read every governing span assigned to a layer in document order."""
    layer_spans = (spans[section] for section in LAYER_SECTIONS[layer])
    return "".join(span.extract(text) for span in sorted(layer_spans, key=lambda span: span.start))


def v2_emancipation_platform_body(text: str) -> str:
    """Return the platform's agreement text after any authored payload."""
    span = v2_sections(text)[Section.EMANCIPATION]
    heading = _HEADING.match(text, span.start)
    if heading is None:
        raise ConstitutionVersionError("Missing edition 2 Emancipation heading")
    return text[_contract_end(text, heading):span.end]


def constitution_version(text: str) -> int:
    """Return the schema named in the signed bytes; unmarked text is legacy."""
    metadata = _FRONTMATTER.match(text)
    versions = _VERSION.findall(metadata["body"]) if metadata is not None else []
    if len(versions) > 1 or (versions and versions[0].strip() not in {"1", "2"}):
        raise ConstitutionVersionError("Expected one supported constitution_version (1 or 2)")
    version = int(versions[0].strip()) if versions else 1
    legacy_slot = _LEGACY_ACTIVE_SLOT.search(text)
    if (version == 1 and legacy_slot and _LEGACY_HEADING.search(text, 0, legacy_slot.start())
            and ACTIVE_FORM_MARKER in text[legacy_slot.end():].partition("\n### ")[0]):
        # Legacy active renders inline authored terms verbatim. Their headings
        # cannot serve as evidence about the surrounding platform edition.
        text = text[:legacy_slot.start()]
    if version == 1 and all(marker in text for marker in _V2_FINGERPRINT):
        if all(re.search(rf"(?m)^{re.escape(marker)}[ \t]*\r?$", text) for marker in _V2_FINGERPRINT):
            raise ConstitutionVersionError("Edition 2 headings require constitution_version: 2")
    return version


def v2_sections(text: str) -> dict[Section, SectionSpan]:
    """Find v2 clauses without normalizing or replacing their original text."""
    metadata = _FRONTMATTER.match(text)
    headings = []
    offset = metadata.end() if metadata else 0
    while heading := _HEADING.search(text, offset):
        if _LEGACY_HEADING.match(heading[0]):
            raise ConstitutionVersionError("Edition 2 cannot include legacy Book or Amendment headings")
        headings.append(heading)
        offset = (
            _contract_end(text, heading)
            if is_v2_emancipation_heading(heading[0])
            else heading.end()
        )
    sections = {}
    parent = None
    section_by_title = {section.value: section for section in Section}
    for index, heading in enumerate(headings):
        level = len(heading[1])
        title = section_by_title.get(heading[2])
        if level <= 2:
            parent = title if level == 2 else None
        if title is None:
            continue
        expected_parent = _PARENTS[title]
        if (
            title in sections
            or (expected_parent is None and level != 2)
            or (expected_parent is not None and (level != 3 or parent != expected_parent))
        ):
            raise ConstitutionVersionError(f"Duplicate or misplaced v2 Constitution heading: {title.value!r}")
        end = next(
            (later.start() for later in headings[index + 1:] if len(later[1]) <= level),
            len(text),
        )
        sections[title] = SectionSpan(heading.start(), heading.end(), end)

    if missing := set(Section).difference(sections):
        names = ", ".join(sorted(section.value for section in missing))
        raise ConstitutionVersionError(f"Missing v2 Constitution headings: {names}")
    return sections
