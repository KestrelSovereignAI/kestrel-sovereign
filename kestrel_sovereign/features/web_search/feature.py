import logging
from typing import Dict, Any, List
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory
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
    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform a web search.
        """
        if not self.tool.enabled:
            return {"success": False, "error": "Web search is disabled."}
            
        return await self.tool.search(query, max_results=max_results)
