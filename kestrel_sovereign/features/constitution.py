import logging
import re
from typing import Dict, List, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool

logger = logging.getLogger(__name__)

class ConstitutionFeature(Feature):
    """
    Feature for accessing and querying the Kestrel Constitution.

    Supports the hierarchical constitution (Books I-IV, Amendments I-VIII)
    as well as legacy Article-based queries for backward compatibility.
    """

    def __init__(self, agent):
        super().__init__(agent)
        self.full_text = ""
        self.articles: Dict[str, str] = {}
        self.books: Dict[str, str] = {}
        self.amendments: Dict[str, str] = {}
        self.summary = ""

    @property
    def tool_description(self) -> str:
        """Description for A2A agent card."""
        return "Access the Kestrel Constitution - view full text, specific books, amendments, or search for terms"

    async def initialize(self):
        """Load and parse the constitution."""
        try:
            # Try to get from agent first to include amendments/anchored version
            if hasattr(self.agent, '_get_governing_constitution'):
                text = await self.agent._get_governing_constitution()
                if text and not text.startswith("Error:"):
                    self.full_text = text
                else:
                    # Fallback to file
                    with open("docs/principles/KESTREL_CONSTITUTION.md", "r", encoding="utf-8") as f:
                        self.full_text = f.read()
            else:
                with open("docs/principles/KESTREL_CONSTITUTION.md", "r", encoding="utf-8") as f:
                    self.full_text = f.read()

            self._parse_articles()
            self._parse_books()
            self._parse_amendments()
            self._generate_summary()
            logger.info("ConstitutionFeature initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ConstitutionFeature: {e}")

    def _parse_articles(self):
        """Parse the constitution into articles (legacy + Article V which remains)."""
        # Split by "## Article"
        parts = re.split(r'(?=## Article)', self.full_text)

        for part in parts:
            if part.strip().startswith("## Article"):
                # Extract article number/title
                match = re.match(r'## (Article [IVX]+):', part)
                if match:
                    key = match.group(1) # e.g., "Article I"
                    # Normalize key for easier lookup (e.g., "1", "I")
                    self.articles[key] = part.strip()

                    # Also map numeric index if possible
                    roman_to_int = {
                        "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
                        "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10"
                    }
                    roman = key.split(" ")[1]
                    if roman in roman_to_int:
                        self.articles[roman_to_int[roman]] = part.strip()

    def _parse_books(self):
        """Parse the constitution into Books (I-IV)."""
        book_pattern = re.compile(r'(?=## Book [IVX]+:)')
        parts = book_pattern.split(self.full_text)

        for part in parts:
            if part.strip().startswith("## Book"):
                match = re.match(r'## (Book [IVX]+):', part)
                if match:
                    key = match.group(1)  # e.g., "Book I"
                    self.books[key] = part.strip()

                    # Also map by number
                    roman_to_int = {
                        "I": "1", "II": "2", "III": "3", "IV": "4",
                    }
                    roman = key.split(" ")[1]
                    if roman in roman_to_int:
                        self.books[roman_to_int[roman]] = part.strip()

    def _parse_amendments(self):
        """Parse Book II into individual Amendments."""
        amendment_pattern = re.compile(r'(?=### Amendment [IVX]+:)')
        parts = amendment_pattern.split(self.full_text)

        for part in parts:
            if part.strip().startswith("### Amendment"):
                match = re.match(r'### (Amendment [IVX]+):', part)
                if match:
                    key = match.group(1)  # e.g., "Amendment I"
                    self.amendments[key] = part.strip()

                    # Also map by number
                    roman_to_int = {
                        "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
                        "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10"
                    }
                    roman = key.split(" ")[1]
                    if roman in roman_to_int:
                        self.amendments[roman_to_int[roman]] = part.strip()

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
        description="Get the full text of the Kestrel Constitution, specific books, amendments, or articles.",
        category=ToolCategory.SYSTEM,
        command_prefix="!constitution"
    )
    async def get_constitution(self, article: Optional[str] = None, search: Optional[str] = None, summary: bool = False) -> ToolResult:
        """
        Retrieve constitutional content.

        Usage:
            !constitution                  - Full text
            !constitution book I           - Specific book (I, II, III, IV or 1, 2, 3, 4)
            !constitution amendment I      - Specific amendment (I-VIII or 1-8)
            !constitution article V        - Article V (amendment process)
            !constitution search <term>    - Search for term
            !constitution summary          - Brief summary

        Args:
            article: Specific section identifier or subcommand keyword
            search: Search term or section identifier
            summary: If True, returns the executive summary
        """
        def _wrap(body: str, *, kind: str = "section") -> ToolResult:
            """Detect not-found bodies via prefix match and route to ERROR.
            Constitution lookups return strings whose first token signals
            the outcome ("Book 'X' not found.", "Amendment 'X' not
            found.", "Section 'X' not found...", "No constitutional
            sections found matching..."). The honesty contract requires
            those to surface as ERROR rather than OK with apologetic
            text.

            Two-track classification: bodies that quote a specific id
            (Book/Amendment/Section) AND say "not found" are ERRORs;
            "No constitutional sections found matching ..." (no
            specific id, just a search miss) is also ERROR — the
            search failed to find anything, the agent must speak that
            rather than narrate a happy summary.
            """
            specific_not_found_prefixes = (
                "Book '", "No book",
                "Amendment '", "No amendment",
                "Section '",
            )
            empty_search_prefix = "No constitutional sections found"
            if any(body.startswith(p) for p in specific_not_found_prefixes) and "not found" in body:
                return ToolResult.failed(body, data={"kind": kind})
            if body.startswith(empty_search_prefix):
                return ToolResult.failed(body, data={"kind": kind})
            return ToolResult.ok(confirmation=body, data={"kind": kind})

        if article:
            article_lower = article.lower()
            if article_lower == "summary":
                return _wrap(self.summary, kind="summary")
            elif article_lower == "book" and search:
                return _wrap(self._get_book(search), kind="book")
            elif article_lower == "amendment" and search:
                return _wrap(self._get_amendment(search), kind="amendment")
            elif article_lower == "article" and search:
                article = search
                search = None
            elif article_lower == "search" and search:
                article = None

        if summary:
            return _wrap(self.summary, kind="summary")

        if article:
            # Try books first, then amendments, then articles
            result = self._get_book(article)
            if not result.startswith("Book '") and not result.startswith("No book"):
                return _wrap(result, kind="book")

            result = self._get_amendment(article)
            if not result.startswith("Amendment '") and not result.startswith("No amendment"):
                return _wrap(result, kind="amendment")

            content = self.articles.get(str(article))
            if not content:
                content = self.articles.get(f"Article {article}")

            if content:
                return _wrap(content, kind="article")
            return _wrap(
                f"Section '{article}' not found. "
                f"Available books: {', '.join(sorted([k for k in self.books.keys() if k.startswith('Book')]))}. "
                f"Available amendments: {', '.join(sorted([k for k in self.amendments.keys() if k.startswith('Amendment')]))}.",
                kind="section",
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
            for key, content in self.articles.items():
                if key.startswith("Article") and search.lower() in content.lower():
                    if not any(content in r for r in results):
                        results.append(content)

            if results:
                return _wrap("\n\n---\n\n".join(results), kind="search")
            return _wrap(
                f"No constitutional sections found matching '{search}'.",
                kind="search",
            )

        return _wrap(self.full_text, kind="full_text")

    def _get_book(self, identifier: str) -> str:
        """Look up a book by number or roman numeral."""
        content = self.books.get(str(identifier))
        if not content:
            content = self.books.get(f"Book {identifier}")
        if content:
            return content
        return f"Book '{identifier}' not found."

    def _get_amendment(self, identifier: str) -> str:
        """Look up an amendment by number or roman numeral."""
        content = self.amendments.get(str(identifier))
        if not content:
            content = self.amendments.get(f"Amendment {identifier}")
        if content:
            return content
        return f"Amendment '{identifier}' not found."
