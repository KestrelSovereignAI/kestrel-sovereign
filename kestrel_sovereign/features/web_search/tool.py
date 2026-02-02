"""
Web Search Tool for Kestrel Agents
Provides web search capabilities using Tavily API
"""

import os
import logging
from typing import Optional, List, Dict, Any
import httpx
from datetime import datetime, timezone

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_DEFAULT

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Web search tool that integrates with Tavily API
    Provides real-time web search capabilities for agents
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search tool

        Args:
            api_key: Tavily API key (if not provided, uses TAVILY_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("No Tavily API key found. Web search will not be available.")
            self.enabled = False
        else:
            self.enabled = True
            self.base_url = "https://api.tavily.com"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_images: bool = False
    ) -> Dict[str, Any]:
        """
        Perform web search using Tavily API

        Args:
            query: Search query string
            max_results: Maximum number of results to return (1-10)
            search_depth: 'basic' or 'advanced' (advanced uses more credits)
            include_answer: Include AI-generated answer summary
            include_raw_content: Include full page content
            include_images: Include image results

        Returns:
            Dictionary with search results and metadata
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "Web search is not enabled (missing API key)",
                "results": []
            }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": min(max_results, 10),
                        "search_depth": search_depth,
                        "include_answer": include_answer,
                        "include_raw_content": include_raw_content,
                        "include_images": include_images
                    },
                    timeout=HTTP_TIMEOUT_DEFAULT
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
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                else:
                    logger.error(f"Tavily API error: {response.status_code} - {response.text}")
                    return {
                        "success": False,
                        "error": f"API error: {response.status_code}",
                        "results": []
                    }

        except httpx.TimeoutException:
            logger.error(f"Tavily API timeout for query: {query}")
            return {
                "success": False,
                "error": "Search request timed out",
                "results": []
            }
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    def format_results_for_llm(self, search_result: Dict[str, Any]) -> str:
        """
        Format search results into a readable string for LLM consumption

        Args:
            search_result: Result dictionary from search()

        Returns:
            Formatted string with search results
        """
        if not search_result.get("success"):
            return f"Search failed: {search_result.get('error', 'Unknown error')}"

        output = []

        # Add AI-generated answer if available
        if search_result.get("answer"):
            output.append(f"Summary: {search_result['answer']}\n")

        # Add search results
        results = search_result.get("results", [])
        if results:
            output.append(f"Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                output.append(f"{i}. {result.get('title', 'Untitled')}")
                output.append(f"   URL: {result.get('url', 'N/A')}")
                if result.get("content"):
                    # Truncate long content
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
        search_depth: str = "basic"
    ) -> str:
        """
        Convenience method to search and return formatted results

        Args:
            query: Search query
            max_results: Maximum results to return
            search_depth: 'basic' or 'advanced'

        Returns:
            Formatted string ready for LLM
        """
        result = await self.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True
        )
        return self.format_results_for_llm(result)


# Singleton instance
_web_search_tool: Optional[WebSearchTool] = None


def get_web_search_tool() -> WebSearchTool:
    """Get or create the singleton web search tool instance"""
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool()
    return _web_search_tool
