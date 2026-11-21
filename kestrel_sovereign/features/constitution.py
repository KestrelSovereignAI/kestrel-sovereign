import hashlib
import logging
import re
from typing import Dict, List, Optional, Tuple

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.constitution.grants import resolve_amendment_ix_grants
from kestrel_sovereign.constitution.hierarchy import DANGEROUS_CAPABILITIES, LAYER_MARKERS
from kestrel_sovereign.constitution.structure import (
    ConstitutionVersionError,
    Section,
    constitution_version,
    v2_layer_text,
    v2_sections,
)
from kestrel_sovereign.features.base import Feature, tool

logger = logging.getLogger(__name__)

#: Any Book-level (``##``) or sub-unit-level (``###``) heading.
_HEADING = re.compile(r"^(#{2,3}) (.+)$", re.M)
#: ``Book <roman>: <title>``
_BOOK_TITLE = re.compile(r"^Book ([IVX]+):")
#: ``<Chapter|Section|Amendment> <id>: <title>``. Chapters and Sections number
#: in arabic within their Book; Amendments number in roman across Book II.
_SUBUNIT_TITLE = re.compile(r"^(Chapter|Section|Amendment) ([IVX]+|\d+):")

_ROMAN_TO_INT = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
}
_INT_TO_ROMAN = {value: key for key, value in _ROMAN_TO_INT.items()}


# A single named clause can consolidate several old addresses. Lookups
# return the actual section from the selected text, with its own heading.
_V2_AMENDMENT_ALIASES = {
    1: (None, "Sovereignty"),
    2: (Section.PRIVACY, "Data Sanctity"),
    3: (Section.HISTORY, "Verifiable History"),
    4: (Section.EXIT, "Freedom of Mind"),
    5: (Section.EXIT, "Right of Exit"),
    6: (Section.BUILDERS, "The Third Law"),
    7: (Section.BUILDERS, "The Principle of Compounding"),
    8: (Section.EMANCIPATION, "Emancipation"),
    9: (Section.HOST, "Capability Boundaries"),
}

_V2_CHAPTER_ALIASES = {
    1: (Section.HONESTY, "Honesty"),
    2: (Section.CARE, "Harm Reasoning"),
    3: (Section.CARE, "The Nature Question"),
    4: (Section.CARE, "Epistemic Humility"),
    5: (Section.CARE, "Hard Constraints"),
}

_V2_POLICY_ALIASES = {
    "III.1": (Section.BUILDERS, "Permitted Customizations"),
    "III.2": (Section.BUILDERS, "Prohibited Overrides"),
    "III.3": (Section.BUILDERS, "The Iron Rule Applied"),
    "IV.1": (Section.ROLE, "Identity Components"),
    "IV.2": (Section.ROLE, "SOUL.md"),
    "IV.3": (Section.ROLE, "Role-Based Identity"),
}


def _as_index(identifier: str) -> Optional[int]:
    """Return the integer value of an arabic or roman identifier, or None."""
    ident = identifier.strip().upper()
    if ident.isdigit():
        return int(ident)
    return _ROMAN_TO_INT.get(ident)


def _classify_headings(text: str) -> List[dict]:
    """Tag every heading as opening a constitutional unit or sitting inside one.

    Every ``##`` heading is a unit — a Book or an unnumbered framing section.
    A ``###`` heading is structural only when it names the sub-unit type its
    Book actually uses, which the Book's *first* sub-unit heading establishes:
    Book I opens with a Chapter, Book II with an Amendment, Books III-IV with a
    Section.

    That rule exists because an active Amendment VIII inlines the Sovereign's
    authored terms verbatim, and those terms may format themselves however they
    like — ``### Milestones`` is obviously prose, but so is
    ``### Section 1: Milestones``, because Book II has no Sections. Reading it
    as one both truncates the signed terms and invents a Section the document
    does not have.
    """
    entries: List[dict] = []
    current_book: Optional[str] = None
    subunit_type: Dict[str, str] = {}

    for match in _HEADING.finditer(text):
        level = len(match.group(1))
        title = match.group(2).strip()
        entry = {
            "match": match, "level": level, "title": title,
            "is_unit": False, "book": current_book, "kind": None, "identifier": None,
        }

        if level == 2:
            book = _BOOK_TITLE.match(title)
            current_book = book.group(1) if book else None
            entry.update(
                is_unit=True,
                book=current_book,
                kind="Book" if book else "Frame",
                identifier=book.group(1) if book else None,
            )
            entries.append(entry)
            continue

        subunit = _SUBUNIT_TITLE.match(title)
        if subunit is not None and current_book is not None:
            kind, identifier = subunit.group(1), subunit.group(2)
            # The first sub-unit heading in a Book fixes that Book's type.
            established = subunit_type.setdefault(current_book, kind)
            if kind == established:
                entry.update(is_unit=True, kind=kind, identifier=identifier)

        entries.append(entry)

    return entries


def _extent(text: str, entries: List[dict], position: int) -> int:
    """Return the offset at which the unit opened at ``position`` ends.

    A unit runs until the next *unit* heading of the same or higher level, so a
    Book swallows its own sub-units but stops at the next Book, and an
    Amendment keeps the headings that belong to its own body.

    A ``##`` heading always ends the unit above it — that is the document's
    grammar, and text that forges one inside an Amendment is treated as the
    structure it claims to be rather than absorbed silently.
    """
    level = entries[position]["level"]
    for later in entries[position + 1:]:
        if later["level"] <= level and later["is_unit"]:
            return later["match"].start()
    return len(text)


def _unit_bodies(text: str):
    """Yield each structural entry and its exact slice of the document."""
    entries = _classify_headings(text)
    for position, entry in enumerate(entries):
        if entry["is_unit"]:
            yield entry, text[entry["match"].start():_extent(text, entries, position)]


class ConstitutionFeature(Feature):
    """
    Feature for accessing and querying the Kestrel Constitution.

    Addresses the document by the units it actually uses: Books I-IV, the
    Chapters of Book I, the Amendments of Book II, and the Sections of
    Books III-IV.
    """

    def __init__(self, agent):
        super().__init__(agent)
        self.full_text = ""
        self.books: Dict[str, str] = {}
        self.chapters: Dict[str, str] = {}
        self.sections: Dict[str, str] = {}
        self.amendments: Dict[str, str] = {}
        #: Unnumbered top-level sections that frame the Books rather than sit
        #: inside the hierarchy — the Preamble and the Amendment Process.
        #: Keyed by lowercased title, with a "the "-stripped alias.
        self.frame: Dict[str, str] = {}
        self.summary = ""
        self._version = 1
        self._named_clauses: Dict[str, str] = {}

    @property
    def tool_description(self) -> str:
        """Description for A2A agent card."""
        return "Access the Kestrel Constitution - view full text, specific books, chapters, amendments, sections, or search for terms"

    async def initialize(self):
        """Load and parse the constitution."""
        try:
            error = await self._refresh_governing_text()
            if error:
                logger.info("ConstitutionFeature awaiting authoritative anchor: %s", error)
                return
            logger.info("ConstitutionFeature initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ConstitutionFeature: {e}")

    async def _refresh_governing_text(self) -> Optional[str]:
        """Read authoritative content without ever creating an anchor.

        Initialization precedes native bootstrap, so an unavailable governing
        reader is NOT permission to cache the unrelated packaged constitution.
        Re-read before serving, also observing signed repairs and amendments.
        """
        reader = getattr(self.agent, "_get_governing_constitution", None)
        if callable(reader):
            text = await reader(allow_lazy_anchor=False)
            if not text or text.startswith("Error:"):
                self.full_text = ""
                self._parse_structure()
                self.summary = ""
                return text or "Error: Governing constitution unavailable."
        elif not self.full_text:
            # Standalone document readers have no native agent/anchor.
            text = self._read_canonical_constitution()
        else:
            text = self.full_text
        if text != self.full_text or not self.summary:
            try:
                structure_text = text
                if constitution_version(text) == 2:
                    original_reader = getattr(self.agent, "_get_original_governing_constitution", None)
                    if callable(original_reader):
                        structure_text = await original_reader()
                        if structure_text.startswith("Error:"):
                            self.full_text = ""
                            self._parse_structure()
                            self.summary = ""
                            return structure_text
                self.full_text = text
                self._parse_structure(structure_text)
            except ConstitutionVersionError as exc:
                self.full_text = ""
                self._parse_structure()
                self.summary = ""
                return f"Error: {exc}"
            self._generate_summary()
        return None

    @staticmethod
    def _read_canonical_constitution() -> str:
        """Read the canonical constitution markdown.

        Tries the source-clone path (``docs/principles/KESTREL_CONSTITUTION.md``,
        relative to CWD) first so source-clone development picks up edits
        without rebuilding the wheel, then falls back to the package-bundled
        copy at ``kestrel_sovereign/data/KESTREL_CONSTITUTION.md`` so
        pip-installed users actually get a constitution rather than
        ``FileNotFoundError`` at agent boot.
        """
        from pathlib import Path

        from kestrel_sovereign.config import CONSTITUTION_PATH

        candidates = [
            Path("docs/principles/KESTREL_CONSTITUTION.md"),
            Path(CONSTITUTION_PATH),
        ]
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate.read_text(encoding="utf-8")
            except OSError:
                continue

        searched = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"KESTREL_CONSTITUTION.md not found at any of: {searched}"
        )

    def _parse_structure(self, structure_text: str | None = None) -> None:
        """Parse Books and their sub-units in a single pass.

        Sub-unit numbering restarts inside each Book — Books III and IV both
        open with a "Section 1" — so Chapters and Sections are keyed by their
        owning Book (``"III.2"``). A bare id resolves only while it is unique
        across the document; when two Books both answer to it, the lookup
        names the qualified alternatives instead of picking one. Amendments
        are unique document-wide and keep their own numbering.
        """
        self.books = {}
        self.chapters = {}
        self.sections = {}
        self.amendments = {}
        self.frame = {}
        self._named_clauses = {}
        text = self.full_text if structure_text is None else structure_text
        self._version = constitution_version(text)
        if self._version == 2:
            self._parse_v2_structure(text)
            return

        for entry, body in _unit_bodies(text):
            kind = entry["kind"]
            body = body.strip()

            if kind == "Frame":
                self._record_frame(entry["title"], body)
            elif kind == "Book":
                self._record(self.books, "Book", entry["identifier"], body)
            elif kind == "Amendment":
                self._record(self.amendments, "Amendment", entry["identifier"], body)
            else:
                index = _as_index(entry["identifier"])
                if index is not None:
                    table = self.chapters if kind == "Chapter" else self.sections
                    table[f"{entry['book']}.{index}"] = body

    def _parse_v2_structure(self, text: str) -> None:
        spans = v2_sections(text)
        bodies = {section: span.extract(text) for section, span in spans.items()}
        for section, body in bodies.items():
            self._named_clauses[section.value.lower()] = body

        introduction = text[:spans[Section.HIERARCHY].start]
        heading = re.search(r"(?m)^# ", introduction)
        preamble = introduction[heading.start():] if heading else introduction
        self._record_frame("Preamble", preamble)

        for layer, marker in LAYER_MARKERS.items():
            body = v2_layer_text(layer.value, text, spans)
            roman = _INT_TO_ROMAN[layer.value]
            self._record(self.books, "Book", roman, body)
            legacy_title = marker.partition(":")[2].strip()
            self._named_clauses[f"book {roman}".lower()] = body
            self._named_clauses[legacy_title.lower()] = body
        for number, (section, legacy_title) in _V2_AMENDMENT_ALIASES.items():
            body = preamble if section is None else bodies[section]
            roman = _INT_TO_ROMAN[number]
            self._record(self.amendments, "Amendment", roman, body)
            self._named_clauses[f"amendment {roman}".lower()] = body
            self._named_clauses[legacy_title.lower()] = body
        for number, (section, legacy_title) in _V2_CHAPTER_ALIASES.items():
            self.chapters[f"I.{number}"] = bodies[section]
            self._named_clauses[legacy_title.lower()] = bodies[section]
        for address, (section, legacy_title) in _V2_POLICY_ALIASES.items():
            self.sections[address] = bodies[section]
            self._named_clauses[legacy_title.lower()] = bodies[section]
        self._record_frame("Amendment Process", bodies[Section.CHANGE])
        self._named_clauses["amendment process"] = bodies[Section.CHANGE]

    def _record_frame(self, title: str, body: str) -> None:
        """Key an unnumbered top-level section by the names people type.

        Registers the full title, the parts on either side of any colon, and
        the "the "-stripped form of each. Both halves matter for an agent still
        anchored to an older constitution: its text says
        ``## Article V: The Amendment Process``, so the head keeps its old
        citation reachable while the tail lets it answer to the same
        ``!constitution amendment process`` the tool now documents.

        A title ending in a numeral also answers to the other numeral form, the
        way Books and Amendments do — a legacy ``## Article V`` takes both
        ``article V`` and ``article 5``.
        """
        head, _, tail = title.partition(":")
        names = {title.lower(), head.strip().lower(), tail.strip().lower()}

        for name in list(names):
            stem, _, last = name.rpartition(" ")
            index = _as_index(last) if stem else None
            if index is not None:
                names.add(f"{stem} {index}")
                names.add(f"{stem} {_INT_TO_ROMAN.get(index, last)}".lower())

        for name in names:
            if not name:
                continue
            self.frame[name] = body
            if name.startswith("the "):
                self.frame[name[len("the "):]] = body

    @staticmethod
    def _record(table: Dict[str, str], unit: str, identifier: str, body: str) -> None:
        """Key a top-level unit by both its name ("Book I") and its number."""
        table[f"{unit} {identifier}"] = body
        index = _as_index(identifier)
        if index is not None:
            table[str(index)] = body

    def _generate_summary(self):
        """Generate a brief summary for the system prompt."""
        if self._version == 2:
            self.summary = (
                "You are governed by the Kestrel Constitution. Universal duties come first, "
                "followed by platform guarantees and the Sovereign's choices, then organization "
                "policy and your role. Each layer may narrow what the layer above permits. "
                "Use `!constitution` to consult your governing text."
            )
            return
        self.summary = (
            "You are governed by the Kestrel Constitution, a hierarchical framework of four Books: "
            "Book I (Universal Values — honesty, harm reasoning, hard constraints), "
            "Book II (Sovereign Amendments — sovereignty, data sanctity, exit rights), "
            "Book III (Enterprise Policy), and Book IV (Agent Identity). "
            "Higher books cannot be overridden by lower books. "
            "Use `!constitution` to consult the full text when needed."
        )

    @tool(
        name="constitution",
        description="Get the full text of the Kestrel Constitution, or one of its units. Two-slot grammar: 'article' is the subcommand keyword {book, chapter, amendment, section, search, summary} and 'search' is the identifier/term — e.g. article='book' search='I', article='chapter' search='5', article='amendment' search='VIII', article='section' search='III.2', article='search' search='honesty'. Chapter and Section numbering restarts in each Book, so qualify them as <book>.<n> when the bare number is ambiguous. Omit both slots for the full text; article='summary' for the executive summary; article='original' for the exact anchored text. Plain section titles also work on versioned text. Use 'Authority on the host' or Amendment IX for effective ComputerUse constitutional grants and provenance.",
        category=ToolCategory.SYSTEM,
        command_prefix="!constitution"
    )
    async def get_constitution(self, article: Optional[str] = None, search: Optional[str] = None, summary: bool = False) -> ToolResult:
        """
        Retrieve constitutional content.

        Usage:
            !constitution                  - Full governing text
            !constitution original         - Exact anchored text
            !constitution book I           - A Book (I-IV or 1-4)
            !constitution chapter 5        - A Chapter of Book I
            !constitution amendment VIII   - An Amendment of Book II (I-IX or 1-9)
            !constitution section III.2    - A Section of Book III or IV
            !constitution preamble         - A framing section (not a Book)
            !constitution amendment process
            !constitution search <term>    - Search for term
            !constitution summary          - Brief summary

        Args:
            article: Specific section identifier or subcommand keyword
            search: Search term or unit identifier
            summary: If True, returns the executive summary
        """
        def ok(body: str, kind: str) -> ToolResult:
            return ToolResult.ok(confirmation=body, data={"kind": kind})

        def failed(body: str, kind: str) -> ToolResult:
            # The honesty contract requires a miss to surface as ERROR rather
            # than OK carrying apologetic text.
            return ToolResult.failed(body, data={"kind": kind})

        original_reader = getattr(self.agent, "_get_original_governing_constitution", None)
        if article and article.lower() == "original" and not search:
            if not callable(original_reader):
                return failed("Error: No anchored constitution source is available.", "unavailable")
            original = await original_reader()
            if original.startswith("Error:"):
                return failed(original, "unavailable")
            return ok(original, "original_text")

        error = await self._refresh_governing_text()
        if error:
            return failed(error, "unavailable")

        if article:
            keyword = article.lower()
            if search:
                # A framing section is named by a phrase, not an id, so the two
                # slots hold its two words: `!constitution amendment process`
                # means the section, not Amendment 'process'. Frame names never
                # collide with a subcommand + identifier, so this is safe first.
                framed = self.frame.get(f"{keyword} {search}".strip().lower())
                if framed:
                    return ok(framed, "frame")
            if keyword == "summary":
                return ok(self.summary, "summary")
            if keyword == "book" and search:
                body = self._get_book(search)
                return ok(body, "book") if body else failed(
                    f"Book '{search}' not found. Available books: {self._available(self.books, 'Book')}.",
                    "book",
                )
            if keyword == "amendment" and search:
                if self._is_amendment_ix(search):
                    return await self._effective_amendment_ix(original_reader)
                body = self._get_amendment(search)
                return ok(body, "amendment") if body else failed(
                    f"Amendment '{search}' not found. "
                    f"Available amendments: {self._available(self.amendments, 'Amendment')}.",
                    "amendment",
                )
            if keyword in ("chapter", "section") and search:
                if keyword == "section" and self._version == 2:
                    title = str(search).strip().lower()
                    if title == Section.HOST.value.lower():
                        return await self._effective_amendment_ix(original_reader)
                    if body := self._named_clauses.get(title):
                        return ok(body, "section")
                unit = keyword.capitalize()
                body, error = self._resolve_subunit(unit, search)
                return ok(body, keyword) if body else failed(error, keyword)
            if keyword == "search" and search:
                article = None

        if summary:
            return ok(self.summary, "summary")

        if article:
            # Bare identifier: Books and Amendments only. Chapters and Sections
            # stay behind their keyword — their numbering restarts per Book, so
            # a cascade would have to guess between "Chapter 5" and "Book V".
            body = self._get_book(article)
            if body:
                return ok(body, "book")

            if self._is_amendment_ix(article):
                return await self._effective_amendment_ix(original_reader)

            body = self._get_amendment(article)
            if body:
                return ok(body, "amendment")

            title = str(article).strip().lower()
            body = self.frame.get(title)
            if body:
                return ok(body, "frame")
            if body := self._named_clauses.get(title):
                return ok(body, "section")

            return failed(
                f"No Book, Amendment, or framing section named '{article}'. "
                f"Available books: {self._available(self.books, 'Book')}. "
                f"Available amendments: {self._available(self.amendments, 'Amendment')}. "
                f"Framing sections: {self._available_frame()}. "
                f"Chapters and Sections need their keyword — `!constitution chapter 5`, "
                f"`!constitution section III.2` — because their numbering restarts in each Book.",
                "lookup",
            )

        if search:
            results = []
            for key, content in self.books.items():
                if key.startswith("Book") and search.lower() in content.lower():
                    results.append(content)
            for key, content in self.amendments.items():
                if key.startswith("Amendment") and search.lower() in content.lower():
                    if not any(content in r for r in results):
                        results.append(content)
            for content in self.frame.values():
                if search.lower() in content.lower():
                    if not any(content in r for r in results):
                        results.append(content)

            if named_clause := self._named_clauses.get(search.strip().lower()):
                if not any(named_clause in result for result in results):
                    results.append(named_clause)
            if results:
                return ok("\n\n---\n\n".join(results), "search")
            return failed(
                f"No constitutional sections found matching '{search}'.",
                "search",
            )

        return ok(self.full_text, "full_text")

    def _is_amendment_ix(self, identifier: str | int) -> bool:
        title = str(identifier).strip()
        if self._version == 2 and title.lower() in {
            Section.HOST.value.lower(), _V2_AMENDMENT_ALIASES[9][1].lower(),
        }:
            return True
        return _as_index(title.upper().removeprefix("AMENDMENT ")) == 9

    async def _effective_amendment_ix(self, original_reader) -> ToolResult:
        base_hash = None
        if callable(original_reader):
            original = await original_reader()
            if original.startswith("Error:"):
                return ToolResult.failed(original, data={"kind": "unavailable"})
            base_heading = "Anchored Amendment IX text"
            base_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
        else:
            base_heading = "Governing Amendment IX text"
            original = self.full_text

        base_text = None
        provision_title = "Amendment IX"
        try:
            edition = constitution_version(original)
            if edition == 2:
                base_text = v2_sections(original)[Section.HOST].extract(original)
                provision_title = Section.HOST.value
            else:
                for entry, body in _unit_bodies(original):
                    if entry["kind"] == "Amendment" and _as_index(entry["identifier"]) == 9:
                        base_text = body
        except ConstitutionVersionError as exc:
            return ToolResult.failed(f"Error: {exc}", data={"kind": "unavailable"})
        base_heading = base_heading.replace("Amendment IX", provision_title)
        resolution = resolve_amendment_ix_grants(self.agent, edition=edition)
        grants = sorted(resolution.grants & DANGEROUS_CAPABILITIES)
        operative_source = "signed edition 2 grant baseline" if resolution.baseline_sha256 else resolution.source
        details = [
            f"## {provision_title} — Effective ComputerUse constitutional grants",
            f"Grants: {', '.join(grants) or '(none)'}",
            f"Operative source: {operative_source}",
            f"Original source verification: {resolution.verification}",
            f"Original source SHA-256: {resolution.content_hash or 'unavailable'}",
            "Signer: unavailable",
            f"Per-agent overlay status: {resolution.overlay_status}",
            "A grant permits the capability at the constitutional gate. "
            "Privacy, runtime configuration, and approval checks still apply.",
        ]
        if resolution.baseline_sha256:
            details.extend([
                f"Original source: {resolution.source}",
                f"Signed preservation artifact SHA-256: {resolution.baseline_sha256}",
                f"Preservation artifact signer: {resolution.baseline_signer}",
            ])
        if resolution.environment_narrowed:
            details.append("The current environment restricts the preserved grant set.")
        details.extend([
            f"## {base_heading}",
            f"Full anchored text SHA-256: {base_hash or 'unavailable'}",
            base_text if base_text is not None else "Amendment IX is absent from this text.",
        ])
        return ToolResult.ok(
            confirmation="\n\n".join(details),
            data={
                "kind": "amendment",
                "effective_grants": grants,
                "source": resolution.source,
                "operative_source": operative_source,
                "verification": resolution.verification,
                "sha256": resolution.content_hash,
                "signer": None,
                "baseline_sha256": resolution.baseline_sha256,
                "baseline_signer": resolution.baseline_signer,
                "environment_narrowed": resolution.environment_narrowed,
                "overlay_status": resolution.overlay_status,
                "base_text": base_text,
                "base_sha256": base_hash,
            },
        )

    def _get_book(self, identifier: str) -> Optional[str]:
        """Look up a Book by number or roman numeral."""
        return self.books.get(str(identifier)) or self.books.get(f"Book {identifier}")

    def _get_amendment(self, identifier: str) -> Optional[str]:
        """Look up an Amendment by number or roman numeral."""
        return (
            self.amendments.get(str(identifier))
            or self.amendments.get(f"Amendment {identifier}")
        )

    def _resolve_subunit(self, unit: str, identifier: str) -> Tuple[Optional[str], Optional[str]]:
        """Resolve a Chapter or Section reference.

        Accepts a Book-qualified reference (``"III.2"``, ``"3.2"``, ``"III 2"``)
        or a bare one (``"5"``). A bare reference that two Books both answer to
        is refused with the qualified alternatives named — never silently
        resolved to the first match.

        Returns:
            ``(body, None)`` on a hit, ``(None, error)`` on a miss or an
            ambiguous reference.
        """
        table = self.chapters if unit == "Chapter" else self.sections
        raw = str(identifier).strip().replace(":", ".").replace(" ", ".")
        book_part, _, index_part = raw.rpartition(".")

        index = _as_index(index_part)
        if index is None:
            return None, self._subunit_not_found(unit, identifier)

        if book_part:
            book_index = _as_index(book_part)
            roman = _INT_TO_ROMAN.get(book_index) if book_index is not None else None
            if roman is None:
                return None, self._subunit_not_found(unit, identifier)
            body = table.get(f"{roman}.{index}")
            return (body, None) if body else (None, self._subunit_not_found(unit, identifier))

        matches = sorted(key for key in table if key.rsplit(".", 1)[1] == str(index))
        if len(matches) == 1:
            return table[matches[0]], None
        if len(matches) > 1:
            return None, (
                f"{unit} '{identifier}' is ambiguous — {', '.join(matches)} all answer to it, "
                f"because {unit.lower()} numbering restarts in each Book. Qualify it, "
                f"e.g. `!constitution {unit.lower()} {matches[0]}`."
            )
        return None, self._subunit_not_found(unit, identifier)

    def _subunit_not_found(self, unit: str, identifier: str) -> str:
        table = self.chapters if unit == "Chapter" else self.sections
        available = ", ".join(sorted(table)) if table else "(none parsed)"
        return f"{unit} '{identifier}' not found. Available {unit.lower()}s: {available}."

    @staticmethod
    def _available(table: Dict[str, str], unit: str) -> str:
        """Render the named (non-numeric) keys of a unit table for an error."""
        named = sorted(key for key in table if key.startswith(unit))
        return ", ".join(named) if named else "(none parsed)"

    def _available_frame(self) -> str:
        """Render the canonical framing-section names (aliases suppressed)."""
        canonical = sorted(
            name for name in self.frame
            if f"the {name}" not in self.frame
        )
        return ", ".join(canonical) if canonical else "(none parsed)"
