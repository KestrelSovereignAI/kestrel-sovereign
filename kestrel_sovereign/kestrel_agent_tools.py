"""
Extension to KestrelAgent to add tool capabilities
Includes web search and feedback/diagnostic tools
"""

import logging
from typing import Optional, Dict, Any, List
import asyncpg
import time
import json
from kestrel_sdk.tools.base import (
    get_web_search_tool,
    get_feedback_tool,
    get_image_generation_tool,
    FeedbackType,
    FeedbackSeverity,
)

logger = logging.getLogger(__name__)


class AgentToolMixin:
    """
    Mixin class to add tool capabilities to KestrelAgent
    Should be mixed in with KestrelAgent class
    """

    def _get_tool_dispatch(self) -> Dict[str, Any]:
        """Return dispatch table mapping command names to handler methods."""
        return {
            "!mcp-load": self._handle_mcp_load,
            "!mcp-list": self._handle_mcp_list,
            "!mcp-unload": self._handle_mcp_unload,
            "!mcp-call": self._handle_mcp_call,
            "!model-list": self._handle_model_list,
            "!model-pull": self._handle_model_pull,
            "!storage-status": self._handle_storage_status,
            "!cleanup-models": self._handle_cleanup_models,
            "!model-info": self._handle_model_info,
            "!search": self._handle_web_search,
            "!web-search": self._handle_web_search,
            "!selfie": self._handle_selfie,
            "!feedback": self._handle_feedback,
            "!tools": self._handle_tools_list,
        }

    def init_tools(
        self,
        pg_pool: Optional[asyncpg.Pool] = None,
        user_id: Optional[str] = None,
        companion_id: Optional[str] = None,
        character_name: Optional[str] = None,
        character_description: Optional[str] = None
    ):
        """
        Initialize agent tools

        Args:
            pg_pool: PostgreSQL connection pool for feedback tool
            user_id: User ID for tool tracking
            companion_id: Companion ID for tool tracking
            character_name: Character name for image generation
            character_description: Character physical description for image generation
        """
        # Web search tool
        self.web_search = get_web_search_tool()

        # Image generation tool
        self.image_gen = get_image_generation_tool()
        self.character_name = character_name or "AI Assistant"
        self.character_description = character_description or "friendly AI assistant"

        # Feedback/diagnostic tool
        if pg_pool:
            self.feedback_tool = get_feedback_tool(pg_pool)
            self.user_id = user_id
            self.companion_id = companion_id
        else:
            self.feedback_tool = None
            self.user_id = None
            self.companion_id = None

        logger.info(f"Initialized tools: web_search={self.web_search.enabled}, image_gen={self.image_gen.enabled}, feedback_tool={self.feedback_tool is not None}")

    async def _handle_tool_commands(self, user_input: str) -> Optional[str]:
        """
        Handle tool-related commands via dispatch table lookup.
        Returns response if command was handled, None otherwise.

        Args:
            user_input: User input string

        Returns:
            Command response or None if not a tool command
        """
        parts = user_input.strip().split()
        command = parts[0]

        dispatch = self._get_tool_dispatch()
        handler = dispatch.get(command)
        if handler:
            return await handler(parts, user_input)

        return None

    # -- MCP Commands --

    async def _handle_mcp_load(self, parts: List[str], user_input: str) -> str:
        if len(parts) < 2:
            return "Usage: !mcp-load <docker_image> [args...]"
        image_name = parts[1]
        cmd_args = parts[2:] if len(parts) > 2 else None
        return await self.mcp_agent.load_tool(image_name, args=cmd_args)

    async def _handle_mcp_list(self, parts: List[str], user_input: str) -> str:
        return await self.mcp_agent.list_tools()

    async def _handle_mcp_unload(self, parts: List[str], user_input: str) -> str:
        if len(parts) < 2:
            return "Usage: !mcp-unload <container_name>"
        container_name = parts[1]
        return await self.mcp_agent.unload_tool(container_name)

    async def _handle_mcp_call(self, parts: List[str], user_input: str) -> str:
        if len(parts) < 3:
            return "Usage: !mcp-call <container> <tool> [json_args]"
        container = parts[1]
        tool = parts[2]
        args = {}
        if len(parts) > 3:
            try:
                args = json.loads(" ".join(parts[3:]))
            except json.JSONDecodeError:
                return "❌ Invalid JSON arguments"
        return await self.mcp_agent.call_tool(container, tool, args)

    # -- Model Management Commands --

    async def _handle_model_list(self, parts: List[str], user_input: str) -> str:
        try:
            models = await self.model_agent.list_models()
            if not models:
                return "No models found."

            by_provider = {}
            for m in models:
                provider = m.provider or "unknown"
                if provider not in by_provider:
                    by_provider[provider] = []
                by_provider[provider].append(m)

            lines = ["**Available Models:**\n"]
            for provider, provider_models in sorted(by_provider.items()):
                lines.append(f"\n**{provider.upper()}**")
                for m in sorted(provider_models, key=lambda x: x.display_name or x.id):
                    star = "★ " if m.is_featured else "  "
                    lines.append(f"{star}{m.display_name or m.id}")

            lines.append(f"\n\n_Total: {len(models)} models across {len(by_provider)} providers_")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error listing models: {str(e)}"

    async def _handle_model_pull(self, parts: List[str], user_input: str) -> str:
        if len(parts) < 2:
            return "Usage: !model-pull <model_name>"
        model_name = parts[1]
        try:
            success = await self.model_agent.pull_model(model_name)
            return f"✅ Model pull initiated for {model_name}" if success else f"❌ Failed to pull {model_name}"
        except Exception as e:
            return f"❌ Error pulling model: {str(e)}"

    async def _handle_storage_status(self, parts: List[str], user_input: str) -> str:
        try:
            info = await self.model_agent.get_storage_info()
            return (
                f"Storage Status:\n"
                f"Total: {info.get('total_gb', 0):.2f} GB\n"
                f"Used: {info.get('used_gb', 0):.2f} GB\n"
                f"Free: {info.get('free_gb', 0):.2f} GB"
            )
        except Exception as e:
            return f"❌ Error getting storage status: {str(e)}"

    async def _handle_cleanup_models(self, parts: List[str], user_input: str) -> str:
        try:
            result = await self.model_agent.cleanup_models()
            return (
                f"Cleanup Result:\n"
                f"Freed: {result.get('freed_space_gb', 0):.2f} GB\n"
                f"Removed: {', '.join(result.get('removed_models', []))}"
            )
        except Exception as e:
            return f"❌ Error cleaning up models: {str(e)}"

    async def _handle_model_info(self, parts: List[str], user_input: str) -> str:
        if len(parts) < 2:
            return "Usage: !model-info <model_name>"
        model_name = parts[1]
        try:
            info = await self.model_agent.get_model_info(model_name)
            if info is None:
                return f"❌ Model not found: {model_name}"
            return f"Model Info for {model_name}:\n{json.dumps(info, indent=2)}"
        except Exception as e:
            return f"❌ Error getting model info: {str(e)}"

    # -- Web Search --

    async def _handle_web_search(self, parts: List[str], user_input: str) -> str:
        if not self.web_search.enabled:
            return "❌ Web search is not available (missing TAVILY_API_KEY environment variable)"

        if len(parts) < 2:
            return "Usage: !search <query> or !web-search <query>"

        query = " ".join(parts[1:])
        start_time = time.time()

        try:
            result = await self.web_search.search_and_format(query, max_results=5)

            if self.feedback_tool and self.companion_id and self.user_id:
                search_data = await self.web_search.search(query, max_results=5)
                if search_data.get("success"):
                    await self.feedback_tool.record_web_search(
                        companion_id=self.companion_id,
                        user_id=self.user_id,
                        query=query,
                        search_provider="tavily",
                        results=search_data.get("results", []),
                        conversation_context=user_input
                    )

            execution_time = int((time.time() - start_time) * 1000)
            return f"🔍 Web Search Results for: '{query}'\n\n{result}\n\n(Search completed in {execution_time}ms)"

        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)

            if self.feedback_tool and self.companion_id and self.user_id:
                await self.feedback_tool.record_feedback(
                    companion_id=self.companion_id,
                    user_id=self.user_id,
                    feedback_type=FeedbackType.TOOL_ERROR,
                    title="Web search failed",
                    content=error_msg,
                    severity=FeedbackSeverity.ERROR,
                    source="tool",
                    context={"query": query, "command": user_input},
                    tags=["web_search", "error"]
                )

            return f"❌ {error_msg}"

    # -- Image Generation --

    async def _handle_selfie(self, parts: List[str], user_input: str) -> str:
        if not self.image_gen.enabled:
            return "❌ Image generation is not available (missing REPLICATE_API_TOKEN environment variable)"

        style = "casual"
        mood = "friendly"

        if len(parts) >= 2:
            style = parts[1]
        if len(parts) >= 3:
            mood = parts[2]

        start_time = time.time()

        try:
            result = await self.image_gen.generate_selfie(
                character_name=self.character_name,
                character_description=self.character_description,
                style=style,
                mood=mood
            )

            execution_time = int((time.time() - start_time) * 1000)
            if self.feedback_tool and self.companion_id and self.user_id:
                await self.record_tool_usage(
                    tool_name="image_generation",
                    success=result.get("success", False),
                    input_params={
                        "character_name": self.character_name,
                        "style": style,
                        "mood": mood
                    },
                    output_result=result,
                    error_message=result.get("error"),
                    execution_time_ms=execution_time
                )

            response = self.image_gen.format_response_for_chat(result)
            return f"{response}\n\n(Generated in {execution_time}ms)"

        except Exception as e:
            error_msg = f"Image generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            if self.feedback_tool and self.companion_id and self.user_id:
                await self.feedback_tool.record_feedback(
                    companion_id=self.companion_id,
                    user_id=self.user_id,
                    feedback_type=FeedbackType.TOOL_ERROR,
                    title="Image generation failed",
                    content=error_msg,
                    severity=FeedbackSeverity.ERROR,
                    source="tool",
                    context={"command": user_input, "style": style, "mood": mood},
                    tags=["image_generation", "error"]
                )

            return f"❌ {error_msg}"

    # -- Feedback Commands --

    async def _handle_feedback(self, parts: List[str], user_input: str) -> str:
        if not self.feedback_tool:
            return "❌ Feedback tool is not available (requires database connection)"

        if len(parts) < 2:
            return (
                "Usage:\n"
                "!feedback list [type] [severity] - List feedback entries\n"
                "!feedback stats - Get tool usage statistics\n"
                "!feedback record <type> <title> <content> - Record feedback manually\n"
                "\n"
                "Types: observation, diagnostic, tool_error, user_feedback, self_reflection, capability_gap\n"
                "Severity: info, warning, error, critical"
            )

        subcommand = parts[1]

        try:
            if subcommand == "list":
                return await self._handle_feedback_list(parts)
            elif subcommand == "stats":
                return await self._handle_feedback_stats()
            elif subcommand == "record" and len(parts) >= 5:
                return await self._handle_feedback_record(parts)
            else:
                return "Invalid feedback subcommand. Use: list, stats, or record"
        except Exception as e:
            error_msg = f"Feedback command failed: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

    async def _handle_feedback_list(self, parts: List[str]) -> str:
        feedback_type = None
        severity = None

        if len(parts) > 2:
            try:
                feedback_type = FeedbackType(parts[2])
            except ValueError:
                pass

        if len(parts) > 3:
            try:
                severity = FeedbackSeverity(parts[3])
            except ValueError:
                pass

        entries = await self.feedback_tool.get_feedback(
            companion_id=self.companion_id,
            user_id=self.user_id,
            feedback_type=feedback_type,
            severity=severity,
            limit=10
        )

        if not entries:
            return "No feedback entries found."

        output = [f"📝 Found {len(entries)} feedback entries:\n"]
        for entry in entries:
            resolved = "✅" if entry["is_resolved"] else "⏳"
            output.append(
                f"{resolved} [{entry['severity'].upper()}] {entry['title']}\n"
                f"   Type: {entry['feedback_type']} | Source: {entry['source']}\n"
                f"   Created: {entry['created_at']}\n"
                f"   {entry['content'][:100]}...\n"
            )

        return "\n".join(output)

    async def _handle_feedback_stats(self) -> str:
        stats = await self.feedback_tool.get_tool_stats(
            companion_id=self.companion_id,
            user_id=self.user_id,
            days=7
        )

        if not stats["tools"]:
            return "No tool usage data in the last 7 days."

        output = [f"📊 Tool Usage Statistics (Last {stats['period_days']} days):\n"]
        for tool in stats["tools"]:
            success_rate = (tool["successful_uses"] / tool["total_uses"] * 100) if tool["total_uses"] > 0 else 0
            output.append(
                f"🔧 {tool['tool_name']}\n"
                f"   Total uses: {tool['total_uses']} | Success rate: {success_rate:.1f}%\n"
                f"   Successful: {tool['successful_uses']} | Failed: {tool['failed_uses']}\n"
                f"   Avg time: {tool['avg_execution_time_ms']:.0f}ms | Max time: {tool['max_execution_time_ms']:.0f}ms\n"
            )

        return "\n".join(output)

    async def _handle_feedback_record(self, parts: List[str]) -> str:
        try:
            feedback_type = FeedbackType(parts[2])
        except ValueError:
            return f"Invalid feedback type. Valid types: {[t.value for t in FeedbackType]}"

        title = parts[3]
        content = " ".join(parts[4:])

        feedback_id = await self.feedback_tool.record_feedback(
            companion_id=self.companion_id,
            user_id=self.user_id,
            feedback_type=feedback_type,
            title=title,
            content=content,
            severity=FeedbackSeverity.INFO,
            source="user"
        )

        return f"✅ Feedback recorded with ID: {feedback_id}"

    # -- Tools List --

    async def _handle_tools_list(self, parts: List[str], user_input: str) -> str:
        tools_status = []
        tools_status.append("🔧 Available Tools:\n")

        status = "✅ Enabled" if getattr(self, 'web_search', None) and self.web_search.enabled else "❌ Disabled"
        tools_status.append(f"🔍 Web Search: {status}")
        tools_status.append("   Commands: !search <query>, !web-search <query>")
        tools_status.append("   Powered by: Tavily API\n")

        status = "✅ Enabled" if getattr(self, 'image_gen', None) and self.image_gen.enabled else "❌ Disabled"
        tools_status.append(f"📸 Image Generation: {status}")
        tools_status.append("   Commands: !selfie [style] [mood]")
        tools_status.append("   Styles: casual, professional, glamour, artistic, outdoor")
        tools_status.append("   Moods: friendly, flirty, serious, playful, winking, shy, confident")
        tools_status.append("   Powered by: Replicate API\n")

        status = "✅ Enabled" if getattr(self, 'feedback_tool', None) else "❌ Disabled"
        tools_status.append(f"📝 Feedback & Diagnostics: {status}")
        tools_status.append("   Commands: !feedback list|stats|record")
        tools_status.append("   Features: Self-diagnostics, observations, tool error tracking\n")

        return "\n".join(tools_status)

    # -- Convenience Methods --

    async def record_tool_usage(
        self,
        tool_name: str,
        success: bool,
        input_params: Optional[Dict[str, Any]] = None,
        output_result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None
    ):
        """
        Convenience method to record tool usage

        Args:
            tool_name: Name of the tool
            success: Whether execution succeeded
            input_params: Tool input parameters
            output_result: Tool output
            error_message: Error message if failed
            execution_time_ms: Execution time in milliseconds
        """
        if self.feedback_tool and self.companion_id and self.user_id:
            await self.feedback_tool.record_tool_usage(
                companion_id=self.companion_id,
                user_id=self.user_id,
                tool_name=tool_name,
                success=success,
                input_params=input_params,
                output_result=output_result,
                error_message=error_message,
                execution_time_ms=execution_time_ms
            )

    async def record_observation(
        self,
        title: str,
        content: str,
        severity: FeedbackSeverity = FeedbackSeverity.INFO,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None
    ):
        """
        Convenience method for agent to record observations

        Args:
            title: Observation title
            content: Detailed content
            severity: Severity level
            context: Additional context
            tags: Searchable tags
        """
        if self.feedback_tool and self.companion_id and self.user_id:
            await self.feedback_tool.record_feedback(
                companion_id=self.companion_id,
                user_id=self.user_id,
                feedback_type=FeedbackType.OBSERVATION,
                title=title,
                content=content,
                severity=severity,
                source="agent",
                context=context,
                tags=tags
            )

    async def record_capability_gap(
        self,
        missing_capability: str,
        context: str,
        workaround: Optional[str] = None
    ):
        """
        Agent records when it identifies a missing capability

        Args:
            missing_capability: What capability is missing
            context: When/why it was needed
            workaround: Any workaround used (if applicable)
        """
        if self.feedback_tool and self.companion_id and self.user_id:
            await self.feedback_tool.record_feedback(
                companion_id=self.companion_id,
                user_id=self.user_id,
                feedback_type=FeedbackType.CAPABILITY_GAP,
                title=f"Missing capability: {missing_capability}",
                content=f"Context: {context}\n\nWorkaround: {workaround or 'None available'}",
                severity=FeedbackSeverity.WARNING,
                source="agent",
                context={
                    "missing_capability": missing_capability,
                    "workaround": workaround
                },
                tags=["capability_gap", missing_capability.lower().replace(" ", "_")]
            )
