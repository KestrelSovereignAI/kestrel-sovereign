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
    1: Section.UNIVERSAL,
    2: Section.RIGHTS,
    3: Section.BUILDERS,
    4: Section.ROLE,
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
_V2_FINGERPRINT = (f"## {Section.HIERARCHY.value}", f"### {Section.HOST.value}")


@dataclass(frozen=True)
class SectionSpan:
    start: int
    heading_end: int
    end: int

    def extract(self, text: str) -> str:
        return text[self.start:self.end]


def constitution_version(text: str) -> int:
    """Return the schema named in the signed bytes; unmarked text is legacy."""
    metadata = _FRONTMATTER.match(text)
    versions = _VERSION.findall(metadata["body"]) if metadata is not None else []
    if len(versions) > 1 or (versions and versions[0].strip() not in {"1", "2"}):
        raise ConstitutionVersionError("Expected one supported constitution_version (1 or 2)")
    version = int(versions[0].strip()) if versions else 1
    if version == 1 and all(marker in text for marker in _V2_FINGERPRINT):
        if all(re.search(rf"(?m)^{re.escape(marker)}[ \t]*\r?$", text) for marker in _V2_FINGERPRINT):
            raise ConstitutionVersionError("Edition 2 headings require constitution_version: 2")
    return version


def v2_sections(text: str) -> dict[Section, SectionSpan]:
    """Find v2 clauses without normalizing or replacing their original text."""
    metadata = _FRONTMATTER.match(text)
    body_start = metadata.end() if metadata else 0
    if _LEGACY_HEADING.search(text, body_start):
        raise ConstitutionVersionError("Edition 2 cannot include legacy Book or Amendment headings")
    headings = list(_HEADING.finditer(text, body_start))
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
