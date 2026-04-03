"""
Search Provider Abstract Base Class.

Defines the interface that all search providers must implement.
External packages can register search providers via entry_points::

    [project.entry-points."kestrel_sovereign.search_providers"]
    BraveSearch = "kestrel_search_brave:BraveSearchProvider"
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SearchProvider(ABC):
    """Abstract base for web search providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider identifier (e.g. 'tavily', 'brave', 'serper')."""
        ...

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether the provider is configured and available."""
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform a web search.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            **kwargs: Provider-specific options.

        Returns:
            Dict with at least: success (bool), results (list), and
            optionally error (str), answer (str).
        """
        ...

    def format_results_for_llm(self, search_result: Dict[str, Any]) -> str:
        """Format search results into a readable string for LLM consumption.

        Default implementation — providers may override.

        Args:
            search_result: Result dictionary from search().

        Returns:
            Formatted string with search results.
        """
        if not search_result.get("success"):
            return f"Search failed: {search_result.get('error', 'Unknown error')}"

        output = []

        if search_result.get("answer"):
            output.append(f"Summary: {search_result['answer']}\n")

        results = search_result.get("results", [])
        if results:
            output.append(f"Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                output.append(f"{i}. {result.get('title', 'Untitled')}")
                output.append(f"   URL: {result.get('url', 'N/A')}")
                if result.get("content"):
                    content = result["content"][:300]
                    if len(result["content"]) > 300:
                        content += "..."
                    output.append(f"   {content}")
                output.append("")

        return "\n".join(output)

    async def search_and_format(
        self,
        query: str,
        max_results: int = 5,
        **kwargs,
    ) -> str:
        """Search and return formatted results.

        Args:
            query: Search query.
            max_results: Maximum results to return.
            **kwargs: Provider-specific options.

        Returns:
            Formatted string ready for LLM.
        """
        result = await self.search(query=query, max_results=max_results, **kwargs)
        return self.format_results_for_llm(result)
