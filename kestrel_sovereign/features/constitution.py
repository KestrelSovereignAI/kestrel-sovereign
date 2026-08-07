import logging
import re
from typing import Dict, List, Optional, Tuple

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
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


def _as_index(identifier: str) -> Optional[int]:
    """Return the integer value of an arabic or roman identifier, or None."""
    ident = identifier.strip().upper()
    if ident.isdigit():
        return int(ident)
    return _ROMAN_TO_INT.get(ident)


def _opens_a_unit(match: "re.Match") -> bool:
    """Whether a heading opens a constitutional unit or merely sits inside one.

    Every ``##`` heading is a unit — a Book or a framing section. A ``###``
    heading is one only when it names a Chapter, Section, or Amendment: an
    active Amendment VIII inlines the Sovereign's authored terms verbatim, and
    those terms may carry their own sub-headings (``### Milestones``), which
    are prose inside the Amendment rather than the start of a new one.
    """
    if len(match.group(1)) == 2:
        return True
    return _SUBUNIT_TITLE.match(match.group(2).strip()) is not None


def _extent(text: str, headings: List["re.Match"], position: int, level: int) -> int:
    """Return the offset at which the unit opened at ``position`` ends.

    A unit runs until the next *unit* heading of the same or higher level, so a
    Book swallows its own Chapters/Sections/Amendments but stops at the next
    Book, and an Amendment keeps sub-headings that belong to its own body.

    A ``##`` heading always ends the unit above it — that is the document's
    grammar, and text that forges one inside an Amendment is treated as the
    structure it claims to be rather than absorbed silently.
    """
    for later in headings[position + 1:]:
        if len(later.group(1)) <= level and _opens_a_unit(later):
            return later.start()
    return len(text)


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

    @property
    def tool_description(self) -> str:
        """Description for A2A agent card."""
        return "Access the Kestrel Constitution - view full text, specific books, chapters, amendments, sections, or search for terms"

    async def initialize(self):
        """Load and parse the constitution."""
        try:
            # Try to get from agent first to include amendments/anchored version
            if hasattr(self.agent, '_get_governing_constitution'):
                text = await self.agent._get_governing_constitution()
                if text and not text.startswith("Error:"):
                    self.full_text = text
                else:
                    self.full_text = self._read_canonical_constitution()
            else:
                self.full_text = self._read_canonical_constitution()

            self._parse_structure()
            self._generate_summary()
            logger.info("ConstitutionFeature initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ConstitutionFeature: {e}")

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

    def _parse_structure(self) -> None:
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

        headings = list(_HEADING.finditer(self.full_text))
        current_book: Optional[str] = None

        for position, match in enumerate(headings):
            level = len(match.group(1))
            title = match.group(2).strip()
            end = _extent(self.full_text, headings, position, level)
            body = self.full_text[match.start():end].strip()

            if level == 2:
                book = _BOOK_TITLE.match(title)
                current_book = book.group(1) if book else None
                if book:
                    self._record(self.books, "Book", book.group(1), body)
                else:
                    self._record_frame(title, body)
                continue

            subunit = _SUBUNIT_TITLE.match(title)
            if subunit is None:
                continue

            unit, identifier = subunit.group(1), subunit.group(2)
            if unit == "Amendment":
                self._record(self.amendments, "Amendment", identifier, body)
            elif current_book is not None:
                index = _as_index(identifier)
                if index is not None:
                    table = self.chapters if unit == "Chapter" else self.sections
                    table[f"{current_book}.{index}"] = body

    def _record_frame(self, title: str, body: str) -> None:
        """Key an unnumbered top-level section by the names people type.

        Registers the full title, the parts on either side of any colon, and
        the "the "-stripped form of each. Both halves matter for an agent still
        anchored to an older constitution: its text says
        ``## Article V: The Amendment Process``, so the head keeps its old
        citation reachable while the tail lets it answer to the same
        ``!constitution amendment process`` the tool now documents.
        """
        head, _, tail = title.partition(":")
        for name in {title.lower(), head.strip().lower(), tail.strip().lower()}:
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
        description="Get the full text of the Kestrel Constitution, or one of its units. Two-slot grammar: 'article' is the subcommand keyword {book, chapter, amendment, section, search, summary} and 'search' is the identifier/term — e.g. article='book' search='I', article='chapter' search='5', article='amendment' search='VIII', article='section' search='III.2', article='search' search='honesty'. Chapter and Section numbering restarts in each Book, so qualify them as <book>.<n> when the bare number is ambiguous. Omit both slots for the full text; article='summary' for the executive summary.",
        category=ToolCategory.SYSTEM,
        command_prefix="!constitution"
    )
    async def get_constitution(self, article: Optional[str] = None, search: Optional[str] = None, summary: bool = False) -> ToolResult:
        """
        Retrieve constitutional content.

        Usage:
            !constitution                  - Full text
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
                body = self._get_amendment(search)
                return ok(body, "amendment") if body else failed(
                    f"Amendment '{search}' not found. "
                    f"Available amendments: {self._available(self.amendments, 'Amendment')}.",
                    "amendment",
                )
            if keyword in ("chapter", "section") and search:
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

            body = self._get_amendment(article)
            if body:
                return ok(body, "amendment")

            body = self.frame.get(str(article).strip().lower())
            if body:
                return ok(body, "frame")

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

            if results:
                return ok("\n\n---\n\n".join(results), "search")
            return failed(
                f"No constitutional sections found matching '{search}'.",
                "search",
            )

        return ok(self.full_text, "full_text")

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
