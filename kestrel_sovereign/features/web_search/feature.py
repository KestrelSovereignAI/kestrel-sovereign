import logging
from typing import Dict, Any, List

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool

from .tool import WebSearchTool

logger = logging.getLogger(__name__)


class WebSearchFeature(Feature):
    """
    Feature for Web Search capabilities.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Search the web for current information - find news, facts, documentation, "
            "and real-time data from the internet"
        )

    async def initialize(self):
        self.tool = WebSearchTool()
        if not self.tool.enabled:
            logger.warning("WebSearchFeature initialized but disabled (missing API key).")

    @tool(
        name="web_search",
        description="Search the web for information.",
        category=ToolCategory.WEB_SEARCH,
        command_prefix="!web-search"
    )
    async def search(self, query: str, max_results: int = 5) -> ToolResult:
        """
        Perform a web search.
        """
        if not self.tool.enabled:
            return ToolResult.failed(
                "Web search is disabled.",
                data={"success": False, "error": "Web search is disabled."},
            )

        result = await self.tool.search(query, max_results=max_results)

        # Underlying tool returns {"success": True/False, "results": [...], ...}.
        # Map to ToolResult based on the success flag.
        if isinstance(result, dict) and result.get("success") is False:
            return ToolResult.failed(
                result.get("error") or "web search failed",
                data=result,
            )

        # Success path. Pull a meaningful confirmation from the data
        # so the agent has something concrete to narrate.
        results_list = result.get("results", []) if isinstance(result, dict) else []
        return ToolResult.ok(
            confirmation=f"Found {len(results_list)} result(s) for {query!r}",
            data=result if isinstance(result, dict) else {"raw": result},
        )
