"""
Web Search Tool for Kestrel Agents.

Provides web search capabilities via a pluggable provider registry.
Built-in provider: Tavily. External providers discoverable via entry_points.

External packages register search providers via entry_points::

    [project.entry-points."kestrel_sovereign.search_providers"]
    BraveSearch = "kestrel_search_brave:BraveSearchProvider"
"""

import logging
from typing import Dict, Any, List, Optional, Type

from kestrel_sovereign.entrypoints import discover_entry_point_classes
from .base import SearchProvider
from .tavily_provider import TavilySearchProvider

logger = logging.getLogger(__name__)

SEARCH_PROVIDER_ENTRY_POINT_GROUP = "kestrel_sovereign.search_providers"


class WebSearchTool:
    """
    Web search tool with pluggable provider registry.

    Discovers search providers from:
    1. Built-in providers (Tavily)
    2. Entry_points (external packages)

    The first enabled provider is used as the default.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize web search tool.

        Args:
            api_key: Optional Tavily API key (for backward compatibility).
        """
        self._providers: Dict[str, SearchProvider] = {}
        self._default_provider: Optional[SearchProvider] = None

        # Phase 1: Register built-in providers
        tavily = TavilySearchProvider(api_key=api_key)
        if tavily.enabled:
            self._providers[tavily.name] = tavily
            self._default_provider = tavily

        # Phase 2: Discover entry_point providers
        self._discover_entrypoint_providers()

        # Set default to first enabled if Tavily wasn't available
        if self._default_provider is None:
            for provider in self._providers.values():
                if provider.enabled:
                    self._default_provider = provider
                    break

        self.enabled = self._default_provider is not None

    def _discover_entrypoint_providers(self) -> None:
        """Discover external search providers via entry_points."""
        classes = discover_entry_point_classes(
            SEARCH_PROVIDER_ENTRY_POINT_GROUP, SearchProvider,
        )
        for ep_name, cls in classes.items():
            try:
                provider = cls()
                if provider.name in self._providers:
                    logger.debug(
                        "Skipping entry_point search provider '%s': "
                        "built-in '%s' already registered",
                        ep_name, provider.name,
                    )
                    continue
                self._providers[provider.name] = provider
                logger.info("Registered entry_point search provider: %s", ep_name)
            except Exception as e:
                logger.warning(
                    "Failed to load entry_point search provider '%s': %s",
                    ep_name, e,
                )

    def get_provider(self, name: Optional[str] = None) -> Optional[SearchProvider]:
        """Get a search provider by name, or the default.

        Args:
            name: Provider name. If None, returns the default.

        Returns:
            SearchProvider or None.
        """
        if name is None:
            return self._default_provider
        return self._providers.get(name)

    def list_providers(self) -> List[str]:
        """List registered search provider names."""
        return list(self._providers.keys())

    async def search(
        self,
        query: str,
        max_results: int = 5,
        provider: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform a web search using the specified or default provider.

        Args:
            query: Search query string.
            max_results: Maximum results to return.
            provider: Optional provider name override.
            **kwargs: Provider-specific options.

        Returns:
            Dict with search results.
        """
        search_provider = self.get_provider(provider)
        if search_provider is None or not search_provider.enabled:
            return {
                "success": False,
                "error": "Web search is not enabled (no provider available)",
                "results": [],
            }
        return await search_provider.search(query, max_results=max_results, **kwargs)

    def format_results_for_llm(self, search_result: Dict[str, Any]) -> str:
        """Format search results for LLM consumption.

        Delegates to the default provider's formatter.

        Args:
            search_result: Result dictionary from search().

        Returns:
            Formatted string.
        """
        provider = self._default_provider
        if provider is not None:
            return provider.format_results_for_llm(search_result)

        # Fallback if no provider
        if not search_result.get("success"):
            return f"Search failed: {search_result.get('error', 'Unknown error')}"
        results = search_result.get("results", [])
        return f"Found {len(results)} results."

    async def search_and_format(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> str:
        """Search and return formatted results.

        Backward-compatible convenience method.

        Args:
            query: Search query.
            max_results: Maximum results to return.
            search_depth: Search depth (Tavily-specific, passed as kwarg).

        Returns:
            Formatted string ready for LLM.
        """
        result = await self.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,
        )
        return self.format_results_for_llm(result)


# Singleton instance
_web_search_tool: Optional[WebSearchTool] = None


def get_web_search_tool() -> WebSearchTool:
    """Get or create the singleton web search tool instance."""
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool()
    return _web_search_tool
