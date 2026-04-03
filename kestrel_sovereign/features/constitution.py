import logging
import re
from typing import Dict, List, Optional
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

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
    async def get_constitution(self, article: Optional[str] = None, search: Optional[str] = None, summary: bool = False) -> str:
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
        # Handle subcommand-style parsing: "article I" or "search term"
        # The parser sends first arg as 'article', so check if it's a keyword
        if article:
            article_lower = article.lower()
            if article_lower == "summary":
                return self.summary
            elif article_lower == "book" and search:
                # "!constitution book I" → look up book
                return self._get_book(search)
            elif article_lower == "amendment" and search:
                # "!constitution amendment I" → look up amendment
                return self._get_amendment(search)
            elif article_lower == "article" and search:
                # "!constitution article V" → article="article", search="V"
                article = search
                search = None
            elif article_lower == "search" and search:
                # "!constitution search sovereignty" → search
                article = None

        if summary:
            return self.summary

        if article:
            # Try books first, then amendments, then articles
            result = self._get_book(article)
            if not result.startswith("Book '") and not result.startswith("No book"):
                return result

            result = self._get_amendment(article)
            if not result.startswith("Amendment '") and not result.startswith("No amendment"):
                return result

            # Try articles (legacy compatibility)
            content = self.articles.get(str(article))
            if not content:
                content = self.articles.get(f"Article {article}")

            if content:
                return content
            return (
                f"Section '{article}' not found. "
                f"Available books: {', '.join(sorted([k for k in self.books.keys() if k.startswith('Book')]))}. "
                f"Available amendments: {', '.join(sorted([k for k in self.amendments.keys() if k.startswith('Amendment')]))}."
            )

        if search:
            results = []
            # Search books
            for key, content in self.books.items():
                if key.startswith("Book") and search.lower() in content.lower():
                    results.append(content)
            # Search amendments
            for key, content in self.amendments.items():
                if key.startswith("Amendment") and search.lower() in content.lower():
                    # Only add if not already part of a book result
                    if not any(content in r for r in results):
                        results.append(content)
            # Search articles (legacy)
            for key, content in self.articles.items():
                if key.startswith("Article") and search.lower() in content.lower():
                    if not any(content in r for r in results):
                        results.append(content)

            if results:
                return "\n\n---\n\n".join(results)
            return f"No constitutional sections found matching '{search}'."

        return self.full_text

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
