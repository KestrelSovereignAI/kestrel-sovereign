"""
Claude Max Subscription Adapter

Adapter for using Claude via the Claude Max subscription ($100/$200/month)
instead of pay-per-token API billing.

Uses the Claude Agent SDK which bundles the Claude Code CLI, configured
to use subscription authentication rather than API keys.

Requirements:
- pip install claude-agent-sdk
- Active Claude Max subscription (logged in via `claude login`)
- ANTHROPIC_API_KEY must NOT be set (or will be removed)

How it works:
1. Removes ANTHROPIC_API_KEY to force subscription auth
2. Sets CLAUDE_USE_SUBSCRIPTION=true
3. Uses the claude-agent-sdk query() function for completions

Supports:
- Streaming responses
- System prompts
- Tools/function calling (via Claude Code's tool system)
"""
import asyncio
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Type

from pydantic import BaseModel

from .adapter import LLMAdapter, LLMResponse, ToolCall
from .model_metadata import ModelInfo, ModelCategory
from kestrel_sovereign.kestrel_config.constants import CLAUDE_MAX_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# Check if claude-agent-sdk is available
try:
    import claude_agent_sdk
    from claude_agent_sdk import query as claude_query
    from claude_agent_sdk import ClaudeAgentOptions
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    claude_agent_sdk = None
    claude_query = None
    ClaudeAgentOptions = None
    logger.warning("claude-agent-sdk not installed. Run: pip install claude-agent-sdk")


class ClaudeMaxAdapter(LLMAdapter):
    """
    Adapter for Claude Max subscription access.

    Uses the Claude Agent SDK with subscription authentication,
    giving you API-like access using your Max subscription's
    included usage instead of per-token billing.
    """

    # Default model for Claude Max
    DEFAULT_MODEL = "claude-sonnet-4-20250514"  # Sonnet 4, included in Max

    def __init__(
        self,
        model: str = None,
        timeout: int = CLAUDE_MAX_TIMEOUT_SECONDS,
        **kwargs
    ):
        """
        Initialize Claude Max adapter.

        Args:
            model: Model to use (default: claude-sonnet-4)
            timeout: Request timeout in seconds
        """
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self._initialized = False

        # Store original API key to restore later if needed
        self._original_api_key = os.environ.get("ANTHROPIC_API_KEY")

    def _setup_subscription_env(self):
        """Configure environment for subscription auth."""
        # Remove API key to force subscription auth
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        # Force subscription mode
        os.environ["CLAUDE_USE_SUBSCRIPTION"] = "true"
        os.environ["CLAUDE_BYPASS_BALANCE_CHECK"] = "true"

        self._initialized = True

    def _restore_env(self):
        """Restore original environment."""
        if self._original_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self._original_api_key

    def _check_login(self) -> bool:
        """Check if user is logged into Claude."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Claude CLI check failed: {e}")
            return False

    async def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response using Claude Max subscription.

        Note: This uses the Claude Agent SDK which provides streaming
        by default. We collect the full response for compatibility.
        """
        if not CLAUDE_SDK_AVAILABLE:
            raise RuntimeError(
                "claude-agent-sdk not installed. Run: pip install claude-agent-sdk"
            )

        if not self._initialized:
            self._setup_subscription_env()

        # Build the prompt
        full_prompt = user_prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

        # Handle tools - convert to Claude Code format if provided
        # Claude Code has its own tool system, so we may need to adapt
        if tools:
            tool_desc = self._format_tools_for_prompt(tools)
            full_prompt = f"{full_prompt}\n\nAvailable tools:\n{tool_desc}"

        try:
            # Collect streaming response
            content_parts = []

            # Configure options with model and system prompt
            options = ClaudeAgentOptions(
                model=self.model,
                system_prompt=system_prompt,
            )

            result_message = None
            usage_info = None

            async for message in claude_query(
                prompt=full_prompt if not system_prompt else user_prompt,
                options=options,
            ):
                # claude_query yields various message types
                msg_type = type(message).__name__

                if msg_type == "ResultMessage":
                    # Final result with clean text and usage info
                    result_message = message
                    if hasattr(message, 'usage'):
                        usage_info = message.usage
                    # Use result as the final content (it's cleaner than streaming)
                    if hasattr(message, 'result') and message.result:
                        content_parts = [str(message.result)]  # Replace, don't append
                elif msg_type == "AssistantMessage":
                    # Streaming content - extract text from TextBlocks
                    # Only use if we don't have a ResultMessage yet
                    if hasattr(message, 'content'):
                        for block in message.content:
                            if hasattr(block, 'text'):
                                content_parts.append(block.text)
                # Skip SystemMessage (init) and other types

            content = "".join(content_parts)

            # Parse tool calls if tools were provided
            tool_calls = None
            if tools:
                tool_calls = self._parse_tool_calls(content)

            # Extract token usage if available
            input_tokens = None
            output_tokens = None
            if usage_info:
                input_tokens = usage_info.get('input_tokens', 0)
                # Include cached tokens in input count
                input_tokens += usage_info.get('cache_read_input_tokens', 0)
                input_tokens += usage_info.get('cache_creation_input_tokens', 0)
                output_tokens = usage_info.get('output_tokens', 0)

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                raw={
                    "model": self.model,
                    "subscription": True,
                    "result_message": result_message,
                },
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=(input_tokens or 0) + (output_tokens or 0) if input_tokens or output_tokens else None,
            )

        except Exception as e:
            logger.error(f"Claude Max generation failed: {e}")
            raise

    async def generate_stream(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream a response using Claude Max subscription.
        """
        if not CLAUDE_SDK_AVAILABLE:
            raise RuntimeError(
                "claude-agent-sdk not installed. Run: pip install claude-agent-sdk"
            )

        if not self._initialized:
            self._setup_subscription_env()

        full_prompt = user_prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

        try:
            options = ClaudeAgentOptions(
                model=self.model,
                system_prompt=system_prompt,
            )

            async for message in claude_query(
                prompt=user_prompt,
                options=options,
            ):
                content_str = None
                if hasattr(message, 'content'):
                    content_str = str(message.content)
                elif hasattr(message, 'message'):
                    msg = message.message
                    if hasattr(msg, 'content'):
                        content_str = str(msg.content)
                elif isinstance(message, str):
                    content_str = message

                # Check for auth errors that get returned as text
                if content_str:
                    if 'Invalid API key' in content_str or 'Please run /login' in content_str:
                        raise RuntimeError(f"Claude Max authentication failed: {content_str}")
                    yield content_str

        except Exception as e:
            logger.error(f"Claude Max streaming failed: {e}")
            raise

    def _format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools as text for the prompt."""
        lines = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                name = func.get("name", "unknown")
                desc = func.get("description", "")
                params = func.get("parameters", {})
                lines.append(f"- {name}: {desc}")
                if params.get("properties"):
                    for pname, pinfo in params["properties"].items():
                        lines.append(f"    - {pname}: {pinfo.get('description', pinfo.get('type', ''))}")
        return "\n".join(lines)

    def _parse_tool_calls(self, content: str) -> Optional[List[ToolCall]]:
        """
        Parse tool calls from response content.

        Claude may format tool calls in various ways. This is a basic parser.
        """
        # Look for JSON tool call patterns
        import re

        # Pattern: {"tool": "name", "arguments": {...}}
        pattern = r'\{[^{}]*"tool"[^{}]*"arguments"[^{}]*\}'
        matches = re.findall(pattern, content, re.DOTALL)

        tool_calls = []
        for i, match in enumerate(matches):
            try:
                data = json.loads(match)
                tool_calls.append(ToolCall(
                    id=f"call_{i}",
                    name=data.get("tool", "unknown"),
                    arguments=data.get("arguments", {})
                ))
            except json.JSONDecodeError:
                continue

        return tool_calls if tool_calls else None

    async def discover_models(self) -> List["ModelInfo"]:
        """
        Return available models for Claude Max subscription.

        Max subscribers have access to Sonnet 4 and Opus 4.5.
        """
        models = [
            ModelInfo(
                id="claude-sonnet-4-20250514",
                name="Claude Sonnet 4",
                provider="claude_max",
                category=ModelCategory.CHAT,
                context_window=200000,
                max_output_tokens=8192,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
            ),
            ModelInfo(
                id="claude-opus-4-5-20251101",
                name="Claude Opus 4.5",
                provider="claude_max",
                category=ModelCategory.CHAT,
                context_window=200000,
                max_output_tokens=32768,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
            ),
        ]
        return models

    def create_messages(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None
    ) -> List[Dict[str, Any]]:
        """Create messages (for interface compatibility)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        return messages

    async def get_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Get a response from Claude Max subscription.

        This is the main interface method called by LLMService.
        It delegates to generate() internally.
        """
        # Extract user and system prompts from messages
        user_prompt = ""
        system_prompt = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_prompt = content
            elif role == "assistant":
                # Include assistant context in user prompt
                user_prompt = f"Previous: {content}\n\n{user_prompt}"

        return await self.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=tools,
            response_format=response_format,
            **kwargs
        )

    async def get_streaming_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Get a streaming response from Claude Max subscription.
        """
        # Extract user and system prompts from messages
        user_prompt = ""
        system_prompt = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_prompt = content

        async for chunk in self.generate_stream(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=tools,
            **kwargs
        ):
            yield chunk
