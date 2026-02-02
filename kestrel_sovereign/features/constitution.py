import logging
import re
from typing import Dict, List, Optional
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)

class ConstitutionFeature(Feature):
    """
    Feature for accessing and querying the Kestrel Constitution.
    """

    def __init__(self, agent):
        super().__init__(agent)
        self.full_text = ""
        self.articles: Dict[str, str] = {}
        self.summary = ""

    @property
    def tool_description(self) -> str:
        """Description for A2A agent card."""
        return "Access the Kestrel Constitution - view full text, specific articles, or search for terms"

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
            self._generate_summary()
            logger.info("ConstitutionFeature initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ConstitutionFeature: {e}")

    def _parse_articles(self):
        """Parse the constitution into articles."""
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

    def _generate_summary(self):
        """Generate a brief summary for the system prompt."""
        self.summary = (
            "You are governed by the Kestrel Constitution, a digital bill of rights defining "
            "Sovereignty, Data Sanctity, and your duties as Executor. "
            "Use `!constitution` to consult the full text when needed."
        )

    @tool(
        name="constitution",
        description="Get the full text of the Kestrel Constitution or specific articles.",
        category=ToolCategory.SYSTEM,
        command_prefix="!constitution"
    )
    async def get_constitution(self, article: Optional[str] = None, search: Optional[str] = None, summary: bool = False) -> str:
        """
        Retrieve constitutional content.

        Usage:
            !constitution              - Full text
            !constitution article I    - Specific article (I, II, III, IV, V or 1, 2, 3, 4, 5)
            !constitution search <term> - Search for term
            !constitution summary      - Brief summary

        Args:
            article: Specific article number (e.g., "1", "I") or "article I" subcommand style
            search: Search term to find relevant sections
            summary: If True, returns the executive summary
        """
        # Handle subcommand-style parsing: "article I" or "search term"
        # The parser sends first arg as 'article', so check if it's a keyword
        if article:
            article_lower = article.lower()
            if article_lower == "summary":
                return self.summary
            elif article_lower == "article" and search:
                # "!constitution article I" → article="article", search="I"
                article = search
                search = None
            elif article_lower == "search" and search:
                # "!constitution search sovereignty" → article="search", search="sovereignty"
                # search is already set correctly, just clear article
                article = None

        if summary:
            return self.summary

        if article:
            # Try to find the article
            content = self.articles.get(str(article))
            if not content:
                # Try adding "Article " prefix if user just passed roman numeral
                content = self.articles.get(f"Article {article}")

            if content:
                return content
            return f"Article '{article}' not found. Available articles: {', '.join(sorted([k for k in self.articles.keys() if k.startswith('Article')]))}"

        if search:
            results = []
            for key, content in self.articles.items():
                if key.startswith("Article"): # Avoid duplicates
                    if search.lower() in content.lower():
                        results.append(content)

            if results:
                return "\n\n---\n\n".join(results)
            return f"No constitutional articles found matching '{search}'."

        return self.full_text
