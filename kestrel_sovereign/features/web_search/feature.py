import logging
import os
from typing import Dict, Any, List, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool

from .tool import WebSearchTool

logger = logging.getLogger(__name__)


class WebSearchFeature(Feature):
    """
    Feature for Web Search capabilities.

    Dogfoods the config-schema UI hints (#2045): a sectioned settings form with a
    write-only masked secret (the provider API key) and a "Test connection" action
    button — all from ``config_schema`` + the feature's own router, no custom
    frontend code. See ``docs/architecture/features/CONFIG_SCHEMA_UI_HINTS.md``.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Search the web for current information - find news, facts, documentation, "
            "and real-time data from the internet"
        )

    async def initialize(self):
        persisted = await self.load_persisted_config() or {}
        self._config: Dict[str, Any] = persisted if isinstance(persisted, dict) else {}
        self.tool = WebSearchTool(api_key=self._resolved_api_key())
        if not self.tool.enabled:
            logger.warning("WebSearchFeature initialized but disabled (missing API key).")

    def _resolved_api_key(self) -> Optional[str]:
        """Stored config key wins; fall back to the TAVILY_API_KEY env var."""
        stored = (getattr(self, "_config", {}) or {}).get("api_key")
        return stored or os.getenv("TAVILY_API_KEY")

    @property
    def config_schema(self) -> Optional[Dict[str, Any]]:
        return {
            "type": "object",
            "properties": {
                "api_key": {
                    "type": "string",
                    "title": "Tavily API Key",
                    "description": (
                        "API key for the Tavily search provider. Stored securely "
                        "and never displayed; leave blank to keep the current value."
                    ),
                    "writeOnly": True,
                    "format": "password",
                },
                "status": {
                    "type": "string",
                    "title": "Provider Status",
                    "description": "Current web-search provider connectivity.",
                    "readOnly": True,
                },
            },
            "x-kestrel-ui": {
                "sections": [
                    {
                        "title": "Credentials",
                        "description": "Authentication for the search provider.",
                        "fields": ["api_key"],
                    },
                    {"title": "Status", "fields": ["status"]},
                ],
                "actions": [
                    {
                        "label": "Test connection",
                        "method": "GET",
                        "path": "/api/features/web_search/test",
                    },
                ],
            },
        }

    async def get_config(self) -> Dict[str, Any]:
        config = dict(getattr(self, "_config", {}) or {})
        enabled = bool(getattr(getattr(self, "tool", None), "enabled", False))
        config["status"] = "Connected" if enabled else "Not configured"
        return config

    async def set_config(self, config: Dict[str, Any]) -> None:
        incoming = {k: v for k, v in (config or {}).items() if k != "status"}
        merged = {**(getattr(self, "_config", {}) or {}), **incoming}
        self._config = merged
        await self.persist_config(merged)
        self.tool = WebSearchTool(api_key=self._resolved_api_key())

    def get_router(self):
        from fastapi import APIRouter, Request

        from kestrel_sovereign.endpoints.agent_helpers import get_agent

        router = APIRouter()

        @router.get("/api/features/web_search/test")
        async def test_connection(request: Request) -> Dict[str, Any]:
            """Live connectivity check used by the config "Test connection" button.

            Request-scoped, not closure-scoped: in multi-agent mode every loaded
            agent's router is mounted at the same global path, so FastAPI matches
            the first-registered handler regardless of which agent the request
            targets. Resolving the feature from ``request.state.agent`` (set by the
            ``/api/agents/{name}/...`` routing middleware) ensures the test runs
            against the selected agent's ``WebSearchFeature``, not whichever one
            happened to mount first.
            """
            agent = get_agent(request)
            feature = (getattr(agent, "features", {}) or {}).get(type(self).__name__)
            tool = getattr(feature, "tool", None) if feature is not None else None
            if tool is None or not getattr(tool, "enabled", False):
                return {
                    "ok": False,
                    "message": "No search provider configured (set a Tavily API key).",
                }
            result = await tool.search("kestrel sovereign connectivity check", max_results=1)
            if isinstance(result, dict) and result.get("success"):
                provider = tool.get_provider()
                provider_name = provider.name if provider is not None else "provider"
                return {"ok": True, "message": f"Connection OK — '{provider_name}' responded."}
            error = result.get("error") if isinstance(result, dict) else "unknown error"
            return {"ok": False, "message": f"Connection failed: {error}"}

        return router

    @tool(
        name="web_search",
        description="Search the web for information. max_results is typically 1-10 (default 5). A 'disabled' error means no search provider is configured — set a provider API key (e.g. TAVILY_API_KEY).",
        category=ToolCategory.WEB_SEARCH,
        command_prefix="!web-search"
    )
    async def search(self, query: str, max_results: int = 5) -> ToolResult:
        """
        Perform a web search.

        Args:
            query: The search query.
            max_results: Maximum results to return (typically 1-10, default 5).

        Returns ToolResult.failed when web search is disabled (no provider
        configured — e.g. set TAVILY_API_KEY) or the provider call fails.
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
