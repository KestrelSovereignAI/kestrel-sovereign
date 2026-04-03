"""
Tavily Search Provider.

First-party implementation of SearchProvider using the Tavily API.
"""

import os
import logging
from typing import Dict, Any
from datetime import datetime, timezone

import httpx

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_DEFAULT
from .base import SearchProvider

logger = logging.getLogger(__name__)


class TavilySearchProvider(SearchProvider):
    """Search provider using the Tavily API."""

    def __init__(self, api_key: str | None = None):
        """Initialize Tavily search provider.

        Args:
            api_key: Tavily API key (falls back to TAVILY_API_KEY env var).
        """
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._enabled = bool(self._api_key)
        self.base_url = "https://api.tavily.com"

        if not self._enabled:
            logger.warning("No Tavily API key found. Tavily search will not be available.")

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def search(
        self,
        query: str,
        max_results: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform web search using Tavily API.

        Args:
            query: Search query string.
            max_results: Maximum number of results (1-10).
            **kwargs: Optional: search_depth, include_answer, include_raw_content, include_images.

        Returns:
            Dict with search results.
        """
        if not self._enabled:
            return {
                "success": False,
                "error": "Web search is not enabled (missing API key)",
                "results": [],
            }

        search_depth = kwargs.get("search_depth", "basic")
        include_answer = kwargs.get("include_answer", True)
        include_raw_content = kwargs.get("include_raw_content", False)
        include_images = kwargs.get("include_images", False)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "api_key": self._api_key,
                        "query": query,
                        "max_results": min(max_results, 10),
                        "search_depth": search_depth,
                        "include_answer": include_answer,
                        "include_raw_content": include_raw_content,
                        "include_images": include_images,
                    },
                    timeout=HTTP_TIMEOUT_DEFAULT,
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "query": query,
                        "answer": data.get("answer"),
                        "results": data.get("results", []),
                        "images": data.get("images", []) if include_images else [],
                        "search_depth": search_depth,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                else:
                    logger.error("Tavily API error: %s - %s", response.status_code, response.text)
                    return {
                        "success": False,
                        "error": f"API error: {response.status_code}",
                        "results": [],
                    }

        except httpx.TimeoutException:
            logger.error("Tavily API timeout for query: %s", query)
            return {
                "success": False,
                "error": "Search request timed out",
                "results": [],
            }
        except Exception as e:
            logger.error("Web search error: %s", e)
            return {
                "success": False,
                "error": str(e),
                "results": [],
            }
